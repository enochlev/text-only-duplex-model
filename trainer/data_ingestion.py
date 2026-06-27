"""data_ingestion.py — unified data sources and episode simulation.

All data sources (WAV file, TTS script, GPT voice) are normalized to
EpisodeData before the episode loop runs. PlaybackSimulator handles all
pre-built sources identically; GPTVoiceSimulator handles live bidirectional
sessions. Use ScriptTTSSource for text-only training (produces silence audio
with WPM-estimated timestamps — no real TTS required).

Source classes expose make_simulator() → PlaybackSimulator | GPTVoiceSimulator.
DataPool samples from a weighted mix of sources; use from_lists() to build
one from raw lists of scripts, WAV paths, and GPT prompts.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

from full_duplex import ASR_SAMPLE_RATE, TTS_MODEL, preload_kokoro_voice, kokoro_synthesize, _resample


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int) -> Tuple[np.ndarray, float]:
    """Load WAV or MP3, mono-mix, resample to target_sr. Returns (float32, duration_s)."""
    try:
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        try:
            import librosa  # type: ignore
            audio, sr = librosa.load(path, sr=None, mono=True, dtype=np.float32)
        except ImportError:
            raise RuntimeError(
                f"Cannot load audio from {path!r}. "
                "Install 'soundfile' (WAV) or 'librosa' (MP3)."
            )
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import torch
        import torchaudio.functional as AF  # type: ignore
        t = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        audio = AF.resample(t, sr, target_sr).squeeze(0).numpy()
    return audio.astype(np.float32), len(audio) / target_sr


def _wpm_duration_s(text: str, wpm: int) -> float:
    """Estimate speech duration of text at given words-per-minute."""
    words = len(text.split())
    return max(0.3, words / max(1, wpm) * 60.0)


def _estimate_word_timestamps(
    text: str, t_start: float, t_end: float, wpm: int
) -> List[Tuple[float, float, str]]:
    """Distribute words uniformly across [t_start, t_end)."""
    words = text.split()
    if not words:
        return []
    duration = max(0.0, t_end - t_start)
    per_word = duration / len(words)
    return [
        (t_start + i * per_word, t_start + (i + 1) * per_word, w)
        for i, w in enumerate(words)
    ]


# ---------------------------------------------------------------------------
# Shared TTS->ASR augmentation cache
# ---------------------------------------------------------------------------
# Real Kokoro TTS + Parakeet ASR is expensive, so each (text, voice) round-trip
# is computed ONCE and cached as JSON under ~/.cache/full_duplex_trainer/asr_aug/.
# Training sources read the cached ASR transcript + word timestamps so the
# training loop never triggers a live TTS/ASR call. Pre-fill with
# scripts/warm_asr_cache.py. Motivation: training previously fed the model clean
# ground-truth text (estimated WPM timestamps); this injects the realistic ASR
# noise (wrong words, timing) the model actually sees at inference.

_ASR_CACHE_VERSION = 1
_ASR_MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
# Kokoro-82M voices for augmentation diversity. Verify against the installed
# Kokoro build; an unknown id falls back to Kokoro's default at synthesis time.
_ASR_AUG_VOICES = ["af_heart", "af_bella", "am_michael", "am_adam", "bf_emma", "bm_george"]


def _asr_aug_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, "full_duplex_trainer", "asr_aug")
    os.makedirs(path, exist_ok=True)
    return path


def _asr_cache_key(
    text: str,
    voice_id: str,
    asr_model_id: str = _ASR_MODEL_ID,
    sample_rate: int = ASR_SAMPLE_RATE,
    version: int = _ASR_CACHE_VERSION,
) -> str:
    payload = f"{version}|{asr_model_id}|{sample_rate}|{voice_id}|{text}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def voices_for_text(
    text: str, n_variants: int = 3, voices: Optional[List[str]] = None
) -> List[str]:
    """Deterministic subset of `voices` for a text (hash-seeded shuffle).

    The same text always maps to the same N voices across runs/processes, so the
    (text, voice) cache entries are stable and never thrash. random.Random(str)
    seeds from a sha512 of the string (not the per-process builtin hash), so this
    is reproducible.
    """
    pool = list(voices or _ASR_AUG_VOICES)
    n = max(1, min(n_variants, len(pool)))
    rng = random.Random(_asr_cache_key(text, "__voicepick__"))
    rng.shuffle(pool)
    return pool[:n]


def asr_cache_entry(text: str, voice_id: str) -> Optional[dict]:
    """Cached ASR record for (text, voice_id), or None if not warmed. Read-only."""
    path = os.path.join(_asr_aug_cache_dir(), _asr_cache_key(text, voice_id) + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def synthesize_and_asr(
    text: str, voice_id: str, device: Optional[str] = None, store_audio: bool = False
) -> dict:
    """text -> Kokoro TTS -> Parakeet ASR, cached as JSON. EXPENSIVE on a miss.

    Returns a record::

        {schema, text, voice_id, asr_model_id, sample_rate, asr_text,
         duration_s, words}

    where ``words`` are (start_s, end_s, word) absolute within the synthesized
    clip (re-based onto the episode timeline by the caller). On a cache hit no
    TTS/ASR runs. ``store_audio=True`` also persists the 16 kHz waveform as
    ``<key>.npy`` (hook for a future live-ASR mode; the text-only training path
    needs only ``duration_s`` + ``words``).

    Called only by the warmer (and, defensively, by sources on a miss); the
    training loop reads via ``asr_cache_entry`` and never calls this.
    """
    cached = asr_cache_entry(text, voice_id)
    if cached is not None:
        return cached

    import tempfile

    import soundfile as sf  # type: ignore

    from full_duplex import preload_asr_model, _asr_lock  # heavy — import on demand

    voice = preload_kokoro_voice(tts_model=voice_id, device=device)
    sr, audio_int16 = kokoro_synthesize(voice, text)
    audio_f32 = audio_int16.astype(np.float32) / 32767.0
    if sr != ASR_SAMPLE_RATE:
        audio_f32 = _resample(audio_f32, sr, ASR_SAMPLE_RATE)
    duration_s = len(audio_f32) / ASR_SAMPLE_RATE

    model = preload_asr_model()
    tmp_path = None
    words: List[Tuple[float, float, str]] = []
    asr_text = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(f.name, audio_f32, ASR_SAMPLE_RATE)
        with _asr_lock:
            out = model.transcribe([tmp_path], timestamps=True, verbose=False)
        if out:
            asr_text = (out[0].text or "").strip()
            for seg in out[0].timestamp.get("word", []):
                w = (seg.get("word") or "").strip()
                if w:
                    words.append((float(seg.get("start", 0.0)), float(seg.get("end", 0.0)), w))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    record = {
        "schema": _ASR_CACHE_VERSION,
        "text": text,
        "voice_id": voice_id,
        "asr_model_id": _ASR_MODEL_ID,
        "sample_rate": ASR_SAMPLE_RATE,
        "asr_text": asr_text,
        "duration_s": duration_s,
        "words": words,
    }
    cache_dir = _asr_aug_cache_dir()
    key = _asr_cache_key(text, voice_id)
    json_path = os.path.join(cache_dir, key + ".json")
    tmp_json = json_path + ".tmp"
    with open(tmp_json, "w") as f:
        json.dump(record, f)
    os.replace(tmp_json, json_path)  # atomic
    if store_audio:
        np.save(os.path.join(cache_dir, key + ".npy"), audio_f32)
    return record


# ---------------------------------------------------------------------------
# EpisodeData — unified pre-processed episode representation
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    """Pre-processed episode ready for simulation.

    All sources reduce to this before the episode loop runs so PlaybackSimulator
    can handle all of them with identical logic.
    """

    audio: np.ndarray
    """float32 at ASR_SAMPLE_RATE."""

    word_timestamps: List[Tuple[float, float, str]]
    """(t_start_s, t_end_s, word) — ground-truth transcript timing."""

    silence_after_s: float = 8.0
    max_episode_s: float = 72.0
    block_s: float = 2.0
    wpm: int = 150
    source_id: str = ""


# ---------------------------------------------------------------------------
# Source classes — each has load() -> EpisodeData and make_simulator()
# ---------------------------------------------------------------------------

class StaticWavSource:
    """Load a WAV or MP3 file as episode audio.

    If script_lines is provided, ground-truth word timestamps are estimated
    from the text at WPM rate. Otherwise the transcript is empty and the agent
    relies on real ASR (mocked to empty strings in simulation).
    """

    def __init__(
        self,
        path: str,
        script_lines: Optional[List[str]] = None,
        silence_after_s: float = 8.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        source_id: Optional[str] = None,
    ) -> None:
        self.path = path
        self.script_lines = script_lines
        self.silence_after_s = silence_after_s
        self.max_episode_s = max_episode_s
        self.block_s = block_s
        self.wpm = wpm
        self.source_id = source_id

    def load(self) -> EpisodeData:
        audio, duration_s = _load_audio(self.path, ASR_SAMPLE_RATE)
        word_timestamps: List[Tuple[float, float, str]] = []
        if self.script_lines:
            joined = " ".join(self.script_lines)
            word_timestamps = _estimate_word_timestamps(joined, 0.0, duration_s, self.wpm)
        return EpisodeData(
            audio=audio,
            word_timestamps=word_timestamps,
            silence_after_s=self.silence_after_s,
            max_episode_s=self.max_episode_s,
            block_s=self.block_s,
            wpm=self.wpm,
            source_id=self.source_id or "",
        )

    def make_simulator(self) -> "PlaybackSimulator":
        return PlaybackSimulator(self.load())


class ScriptTTSSource:
    """Build episode audio from a list of script lines.

    Each line becomes a silence segment of WPM-estimated duration (no real TTS
    required — mock ASR reads ground-truth text directly from word_timestamps).
    An inter-turn pause follows each line so the agent has time to respond
    before the next user turn begins.

    This is functionally identical to StaticWavSource; the difference is just
    the preprocessing step (text → silence + timestamps vs. loading a file).
    """

    def __init__(
        self,
        script_lines: List[str],
        inter_turn_pause_s: float = 12.0,
        silence_after_s: float = 8.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        source_id: Optional[str] = None,
        tts_model: str = "",
        device: Optional[str] = None,
    ) -> None:
        self.script_lines = script_lines
        self.inter_turn_pause_s = inter_turn_pause_s
        self.silence_after_s = silence_after_s
        self.max_episode_s = max_episode_s
        self.block_s = block_s
        self.wpm = wpm
        self.source_id = source_id
        self.tts_model = tts_model or TTS_MODEL
        self.device = device

    def load(self) -> EpisodeData:
        voice = preload_kokoro_voice(tts_model=self.tts_model, device=self.device)
        segments: List[np.ndarray] = []
        word_timestamps: List[Tuple[float, float, str]] = []
        # 0.5 s leading silence ensures the first word's midpoint clears the
        # first block window (~0 – block_s), which has no matching agent block yet.
        _LEAD_S = 0.5
        segments.append(np.zeros(int(_LEAD_S * ASR_SAMPLE_RATE), dtype=np.float32))
        t = _LEAD_S
        for line in self.script_lines:
            sr, audio_int16 = kokoro_synthesize(voice, line)
            audio_f32 = audio_int16.astype(np.float32) / 32767.0
            if sr != ASR_SAMPLE_RATE:
                audio_f32 = _resample(audio_f32, sr, ASR_SAMPLE_RATE)
            dur = len(audio_f32) / ASR_SAMPLE_RATE
            # ±15% WPM jitter for word timestamp estimation.
            jittered_wpm = max(60, int(self.wpm * random.uniform(0.85, 1.15)))
            word_timestamps.extend(_estimate_word_timestamps(line, t, t + dur, jittered_wpm))
            segments.append(audio_f32)
            t += dur
            # 4–8 block gap (8–16 s at block_s=2.0) between user turns.
            pause_s = random.uniform(8.0, 16.0)
            segments.append(np.zeros(int(pause_s * ASR_SAMPLE_RATE), dtype=np.float32))
            t += pause_s
        audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
        return EpisodeData(
            audio=audio,
            word_timestamps=word_timestamps,
            silence_after_s=self.silence_after_s,
            max_episode_s=self.max_episode_s,
            block_s=self.block_s,
            wpm=self.wpm,
            source_id=self.source_id or "",
        )

    def make_simulator(self) -> "PlaybackSimulator":
        return PlaybackSimulator(self.load())


class AsrAugmentedScriptSource:
    """Like ScriptTTSSource, but each line's transcript + word timestamps come
    from a cached TTS->ASR round-trip (realistic ASR noise) instead of clean
    estimated text. Audio is silence of the cached duration — the text-only
    training path uses only timing + the transcript override (see
    rl_trainer.sim_seal_mic_block), never the waveform.

    Read-only on the cache: a line/voice that isn't warmed falls back to clean
    estimated timestamps so training NEVER blocks on a live TTS/ASR call. Warm
    with scripts/warm_asr_cache.py.
    """

    def __init__(
        self,
        script_lines: List[str],
        inter_turn_pause_s: float = 12.0,  # kept for signature parity (unused: pause is jittered)
        silence_after_s: float = 8.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        source_id: Optional[str] = None,
        voices: Optional[List[str]] = None,
        n_variants: int = 3,
        device: Optional[str] = None,
    ) -> None:
        self.script_lines = script_lines
        self.inter_turn_pause_s = inter_turn_pause_s
        self.silence_after_s = silence_after_s
        self.max_episode_s = max_episode_s
        self.block_s = block_s
        self.wpm = wpm
        self.source_id = source_id
        self.voices = list(voices or _ASR_AUG_VOICES)
        self.n_variants = n_variants
        self.device = device

    def load(self) -> EpisodeData:
        _LEAD_S = 0.5
        segments: List[np.ndarray] = [np.zeros(int(_LEAD_S * ASR_SAMPLE_RATE), dtype=np.float32)]
        word_timestamps: List[Tuple[float, float, str]] = []
        t = _LEAD_S
        for line in self.script_lines:
            voice = random.choice(voices_for_text(line, self.n_variants, self.voices))
            rec = asr_cache_entry(line, voice)
            if rec is not None and rec.get("words"):
                dur = float(rec["duration_s"])
                word_timestamps.extend((t + ws, t + we, w) for ws, we, w in rec["words"])
            else:
                # Not warmed → clean fallback (same timing model as ScriptTTSSource).
                dur = _wpm_duration_s(line, self.wpm)
                jittered_wpm = max(60, int(self.wpm * random.uniform(0.85, 1.15)))
                word_timestamps.extend(_estimate_word_timestamps(line, t, t + dur, jittered_wpm))
            segments.append(np.zeros(int(dur * ASR_SAMPLE_RATE), dtype=np.float32))
            t += dur
            pause_s = random.uniform(8.0, 16.0)  # 4–8 block gap between user turns
            segments.append(np.zeros(int(pause_s * ASR_SAMPLE_RATE), dtype=np.float32))
            t += pause_s
        audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
        return EpisodeData(
            audio=audio,
            word_timestamps=word_timestamps,
            silence_after_s=self.silence_after_s,
            max_episode_s=self.max_episode_s,
            block_s=self.block_s,
            wpm=self.wpm,
            source_id=self.source_id or "asr_script",
        )

    def make_simulator(self) -> "PlaybackSimulator":
        return PlaybackSimulator(self.load())


# ---------------------------------------------------------------------------
# UltraChatTTSSource — single-question episodes from UltraChat 200k
# ---------------------------------------------------------------------------

_ULTRACHAT_PROMPTS: Optional[List[str]] = None
_ULTRACHAT_EMBEDDINGS: Optional["np.ndarray"] = None  # shape (N, D), L2-normalised
_ULTRACHAT_MAX_CACHE = 100_000
_ULTRACHAT_SIM_TOP_K = 100
_ULTRACHAT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Imperative generation-task verbs — these produce prompts like "Write a blog post..."
# which are not conversational and cause the model to go fully silent (no RM1 gradient path).
_GENERATION_PREFIXES: frozenset = frozenset([
    "write", "create", "design", "build", "generate", "compose",
    "code", "program", "develop", "implement", "list", "draft",
    "make", "produce", "construct", "formulate", "outline",
    "summarize", "rewrite", "translate", "convert", "calculate",
    "analyze", "analyse", "describe", "compare", "evaluate",
])

_ULTRACHAT_CONV_INDICES: Optional[List[int]] = None  # cached indices of conversational prompts
_ULTRACHAT_CONV_SET: Optional[set] = None            # same as a set for O(1) lookup in _sample_similar_idx


def _is_generation_task(text: str) -> bool:
    """Return True for imperative generation tasks unsuitable for voice conversation."""
    first = text.lower().split()[0].rstrip(".,;:!?") if text.split() else ""
    return first in _GENERATION_PREFIXES


def _get_conversational_indices() -> List[int]:
    """Return (and cache) the indices of UltraChat prompts that are conversational questions."""
    global _ULTRACHAT_CONV_INDICES, _ULTRACHAT_CONV_SET
    if _ULTRACHAT_CONV_INDICES is not None:
        return _ULTRACHAT_CONV_INDICES
    prompts = _load_ultrachat_prompts()
    _ULTRACHAT_CONV_INDICES = [i for i, p in enumerate(prompts) if not _is_generation_task(p)]
    _ULTRACHAT_CONV_SET = set(_ULTRACHAT_CONV_INDICES)
    n = len(_ULTRACHAT_CONV_INDICES)
    print(f"[UltraChatTTSSource] {n}/{len(prompts)} prompts are conversational ({n / len(prompts) * 100:.0f}%)")
    return _ULTRACHAT_CONV_INDICES


# Use sentence-transformers<3.0 to avoid pulling in a newer torch:
#   pip install "sentence-transformers<3.0"


def _ultrachat_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    path = os.path.join(base, "full_duplex_trainer")
    os.makedirs(path, exist_ok=True)
    return path


def _load_ultrachat_prompts(max_prompts: int = _ULTRACHAT_MAX_CACHE) -> List[str]:
    """Load short first-turn user prompts from UltraChat 200k, with disk cache.

    First run streams from HuggingFace and writes to
    ~/.cache/full_duplex_trainer/ultrachat_prompts_{max_prompts}.json.
    Subsequent runs load from disk in ~0.5 s.
    """
    global _ULTRACHAT_PROMPTS
    if _ULTRACHAT_PROMPTS is not None:
        return _ULTRACHAT_PROMPTS

    cache_path = os.path.join(_ultrachat_cache_dir(), f"ultrachat_prompts_{max_prompts}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _ULTRACHAT_PROMPTS = json.load(f)
        print(f"[UltraChatTTSSource] loaded {len(_ULTRACHAT_PROMPTS)} prompts from disk cache")
        return _ULTRACHAT_PROMPTS

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError(
            "UltraChatTTSSource requires 'datasets'. pip install datasets"
        )
    print("[UltraChatTTSSource] streaming ultrachat_200k (first run only — will cache to disk)…")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    prompts: List[str] = []
    for example in ds:
        messages = example.get("messages") or []
        if not messages:
            continue
        first = messages[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        if text and len(text.split()) < 20:
            prompts.append(text)
            if len(prompts) >= max_prompts:
                break
    _ULTRACHAT_PROMPTS = prompts
    with open(cache_path, "w") as f:
        json.dump(prompts, f)
    print(f"[UltraChatTTSSource] cached {len(prompts)} prompts → {cache_path}")
    return _ULTRACHAT_PROMPTS


_ULTRACHAT_EMBED_DEVICE: str = "cpu"
"""Device for the one-shot MiniLM embedding pass.  Default cpu avoids polluting
the primary training GPU with the ~1 GB residual that persists after del+empty_cache.
Set to e.g. 'cuda:1' to use a secondary GPU."""


def set_embed_device(device: str) -> None:
    """Call before make_default_data_pool() to redirect MiniLM to a specific device."""
    global _ULTRACHAT_EMBED_DEVICE
    _ULTRACHAT_EMBED_DEVICE = device


def _get_ultrachat_embeddings() -> "np.ndarray":
    """Lazily encode all cached prompts, with disk cache for the embeddings.

    Embeddings are saved to
    ~/.cache/full_duplex_trainer/ultrachat_embeddings_{N}.npy (~70 MB).
    Loading from disk takes ~0.3 s vs ~5 s to re-encode.
    """
    global _ULTRACHAT_EMBEDDINGS
    if _ULTRACHAT_EMBEDDINGS is not None:
        return _ULTRACHAT_EMBEDDINGS
    import numpy as _np
    prompts = _load_ultrachat_prompts()
    model_tag = _ULTRACHAT_EMBED_MODEL.replace("/", "_")
    cache_path = os.path.join(
        _ultrachat_cache_dir(),
        f"ultrachat_embeddings_{len(prompts)}_{model_tag}.npy",
    )
    if os.path.exists(cache_path):
        _ULTRACHAT_EMBEDDINGS = _np.load(cache_path)
        print(f"[UltraChatTTSSource] embeddings loaded from disk  shape={_ULTRACHAT_EMBEDDINGS.shape}")
        return _ULTRACHAT_EMBEDDINGS
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Similarity sampling requires sentence-transformers. "
            "pip install 'sentence-transformers<3.0'"
        )
    print(f"[UltraChatTTSSource] encoding {len(prompts)} sentences with {_ULTRACHAT_EMBED_MODEL} "
          f"on {_ULTRACHAT_EMBED_DEVICE} …")
    model = SentenceTransformer(_ULTRACHAT_EMBED_MODEL, device=_ULTRACHAT_EMBED_DEVICE)
    emb = model.encode(
        prompts,
        batch_size=512,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    del model
    import torch as _torch
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    _ULTRACHAT_EMBEDDINGS = emb.astype(_np.float32)
    _np.save(cache_path, _ULTRACHAT_EMBEDDINGS)
    print(f"[UltraChatTTSSource] embeddings cached → {cache_path}  shape={_ULTRACHAT_EMBEDDINGS.shape}")
    return _ULTRACHAT_EMBEDDINGS


def _sample_similar_idx(
    anchor_idx: int,
    k: int = _ULTRACHAT_SIM_TOP_K,
    temperature: float = 0.1,
    valid_set: Optional[set] = None,
) -> int:
    """Return an index sampled from the top-k most similar prompts to anchor_idx.

    Cosine similarities (embeddings are pre-normalised, so this is a dot
    product) are converted to a probability distribution via temperature-scaled
    softmax.  Lower temperature → samples cluster near the top-1 neighbour.

    If valid_set is provided, only indices in that set are eligible candidates.
    """
    import numpy as _np
    emb = _get_ultrachat_embeddings()
    scores = emb @ emb[anchor_idx]          # (N,) cosine similarities
    scores[anchor_idx] = -2.0               # exclude self
    if valid_set is not None:
        mask = _np.zeros(len(scores), dtype=bool)
        mask[list(valid_set)] = True
        mask[anchor_idx] = False
        scores[~mask] = -2.0
    top_k = min(k, int((scores > -1.5).sum()))
    top_k = max(top_k, 1)
    top_idx = _np.argpartition(scores, -top_k)[-top_k:]
    top_scores = scores[top_idx] / temperature
    top_scores -= top_scores.max()          # numerical stability
    probs = _np.exp(top_scores)
    probs /= probs.sum()
    return int(_np.random.choice(top_idx, p=probs))


class UltraChatTTSSource:
    """Multi-turn episode source drawn from UltraChat 200k.

    Episode structure:
      line 1 — random prompt from the corpus
      line 2 — sampled from the top-100 most similar prompts to line 1
                (probability ∝ softmax of cosine similarity)
      line 3 — 50 % chance: same similarity sampling seeded from line 2

    silence_after_s defaults to 3× the ScriptTTSSource value (24 s) so
    the model has an extended window to answer each question.
    """

    def __init__(
        self,
        silence_after_s: float = 24.0,
        inter_turn_pause_s: float = 20.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        source_id: Optional[str] = None,
        tts_model: str = "",
        device: Optional[str] = None,
        similarity_temperature: float = 0.1,
    ) -> None:
        self.silence_after_s = silence_after_s
        self.inter_turn_pause_s = inter_turn_pause_s
        self.max_episode_s = max_episode_s
        self.block_s = block_s
        self.wpm = wpm
        self.source_id = source_id
        self.tts_model = tts_model or TTS_MODEL
        self.device = device
        self.similarity_temperature = similarity_temperature

    def load(self) -> EpisodeData:
        prompts = _load_ultrachat_prompts()
        conv = _get_conversational_indices()  # also populates _ULTRACHAT_CONV_SET
        idx1 = random.choice(conv)
        idx2 = _sample_similar_idx(idx1, temperature=self.similarity_temperature, valid_set=_ULTRACHAT_CONV_SET)
        lines = [prompts[idx1], prompts[idx2]]
        if random.random() < 0.5:
            idx3 = _sample_similar_idx(idx2, temperature=self.similarity_temperature, valid_set=_ULTRACHAT_CONV_SET)
            lines.append(prompts[idx3])
        return ScriptTTSSource(
            script_lines=lines,
            inter_turn_pause_s=self.inter_turn_pause_s,
            silence_after_s=self.silence_after_s,
            max_episode_s=self.max_episode_s,
            block_s=self.block_s,
            wpm=self.wpm,
            source_id=self.source_id or "ultrachat",
            tts_model=self.tts_model,
            device=self.device,
        ).load()

    def make_simulator(self) -> "PlaybackSimulator":
        return PlaybackSimulator(self.load())


class AsrAugmentedUltraChatSource:
    """Single-prompt UltraChat episode with an ASR-noised transcript (cached).

    Draws one conversational prompt from the warmed prefix (first ``prompt_cap``
    indices, which scripts/warm_asr_cache.py fills) and runs it through
    AsrAugmentedScriptSource. Read-only on the cache (clean fallback if cold).
    """

    def __init__(
        self,
        silence_after_s: float = 24.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        source_id: Optional[str] = None,
        voices: Optional[List[str]] = None,
        n_variants: int = 3,
        device: Optional[str] = None,
        prompt_cap: int = 2000,
    ) -> None:
        self.silence_after_s = silence_after_s
        self.max_episode_s = max_episode_s
        self.block_s = block_s
        self.wpm = wpm
        self.source_id = source_id
        self.voices = list(voices or _ASR_AUG_VOICES)
        self.n_variants = n_variants
        self.device = device
        self.prompt_cap = prompt_cap

    def load(self) -> EpisodeData:
        prompts = _load_ultrachat_prompts()
        conv = _get_conversational_indices()
        capped = [i for i in conv if i < self.prompt_cap] or conv[: self.prompt_cap]
        idx = random.choice(capped)
        return AsrAugmentedScriptSource(
            script_lines=[prompts[idx]],
            silence_after_s=self.silence_after_s,
            max_episode_s=self.max_episode_s,
            block_s=self.block_s,
            wpm=self.wpm,
            source_id=self.source_id or "asr_ultrachat",
            voices=self.voices,
            n_variants=self.n_variants,
            device=self.device,
        ).load()

    def make_simulator(self) -> "PlaybackSimulator":
        return PlaybackSimulator(self.load())


# ---------------------------------------------------------------------------
# LongMonologueSource — one long user turn (teach "don't interrupt")
# ---------------------------------------------------------------------------

_ULTRACHAT_LONG: Optional[List[str]] = None


def _load_ultrachat_long_messages(
    min_words: int = 40, max_words: int = 90, max_msgs: int = 20000
) -> List[str]:
    """Long first-user messages from UltraChat 200k, with disk cache.

    Unlike _load_ultrachat_prompts (which keeps <20-word prompts), this keeps
    messages in [min_words, max_words] — long enough to span ~8–15 blocks when
    synthesized. Generation tasks ("Write a blog post…") are dropped. Cached to
    ~/.cache/full_duplex_trainer/ultrachat_long_{min}_{max}_{max_msgs}.json.
    """
    global _ULTRACHAT_LONG
    if _ULTRACHAT_LONG is not None:
        return _ULTRACHAT_LONG
    cache_path = os.path.join(
        _ultrachat_cache_dir(), f"ultrachat_long_{min_words}_{max_words}_{max_msgs}.json"
    )
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            _ULTRACHAT_LONG = json.load(f)
        print(f"[LongMonologueSource] loaded {len(_ULTRACHAT_LONG)} long messages from disk cache")
        return _ULTRACHAT_LONG
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        raise RuntimeError("LongMonologueSource requires 'datasets'. pip install datasets")
    print("[LongMonologueSource] streaming ultrachat_200k for long messages (first run only — will cache)…")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)
    msgs: List[str] = []
    for example in ds:
        messages = example.get("messages") or []
        if not messages:
            continue
        first = messages[0]
        if first.get("role") != "user":
            continue
        text = (first.get("content") or "").strip()
        n = len(text.split())
        if min_words <= n <= max_words and not _is_generation_task(text):
            msgs.append(text)
            if len(msgs) >= max_msgs:
                break
    _ULTRACHAT_LONG = msgs
    with open(cache_path, "w") as f:
        json.dump(msgs, f)
    print(f"[LongMonologueSource] cached {len(msgs)} long messages → {cache_path}")
    return _ULTRACHAT_LONG


class LongMonologueSource:
    """One continuous long user turn (no multi-turn, no similarity) to teach the
    model NOT to interrupt while the user keeps talking.

    Picks a long UltraChat message, uses its cached TTS->ASR transcript +
    timestamps (realistic ASR noise), and keeps it only if the synthesized
    duration lands in the target block span (8–15 blocks by default). The
    message is one unbroken turn; PlaybackSimulator's trailing ``silence_after_s``
    is the window where the bot should finally respond (kept >= a few blocks so
    the episode has a post-monologue speech step to carry the RM3 idle-reward
    gradient back — see CLAUDE.md §10).

    Read-only on the cache: only messages in the warmed prefix
    (first ``warm_cap``) are considered; if none are cached/in-range after
    ``max_tries`` it falls back to clean estimated timestamps so training never
    blocks on live TTS/ASR. Warm with scripts/warm_asr_cache.py.
    """

    def __init__(
        self,
        min_words: int = 40,
        max_words: int = 90,
        target_blocks: Tuple[int, int] = (8, 15),
        silence_after_s: float = 12.0,
        block_s: float = 2.0,
        wpm: int = 150,
        source_id: str = "long_monologue",
        voices: Optional[List[str]] = None,
        n_variants: int = 3,
        device: Optional[str] = None,
        warm_cap: int = 500,
        max_tries: int = 16,
    ) -> None:
        self.min_words = min_words
        self.max_words = max_words
        self.target_blocks = target_blocks
        self.silence_after_s = silence_after_s
        self.block_s = block_s
        self.wpm = wpm
        self.source_id = source_id
        self.voices = list(voices or _ASR_AUG_VOICES)
        self.n_variants = n_variants
        self.device = device
        self.warm_cap = warm_cap
        self.max_tries = max_tries

    def _episode(self, words_rebased, dur: float) -> EpisodeData:
        _LEAD_S = 0.5
        audio = np.zeros(int((_LEAD_S + dur) * ASR_SAMPLE_RATE), dtype=np.float32)
        # +block_s margin so the last word's block fully closes before silence.
        max_episode_s = _LEAD_S + dur + self.silence_after_s + self.block_s + 5.0
        return EpisodeData(
            audio=audio,
            word_timestamps=words_rebased,
            silence_after_s=self.silence_after_s,
            max_episode_s=max_episode_s,
            block_s=self.block_s,
            wpm=self.wpm,
            source_id=self.source_id,
        )

    def load(self) -> EpisodeData:
        _LEAD_S = 0.5
        msgs = _load_ultrachat_long_messages(self.min_words, self.max_words)
        candidates = msgs[: self.warm_cap] if self.warm_cap else msgs
        lo = self.target_blocks[0] * self.block_s
        hi = self.target_blocks[1] * self.block_s
        for _ in range(self.max_tries):
            msg = random.choice(candidates)
            voice = random.choice(voices_for_text(msg, self.n_variants, self.voices))
            rec = asr_cache_entry(msg, voice)
            if rec is not None and rec.get("words") and lo <= float(rec["duration_s"]) <= hi:
                dur = float(rec["duration_s"])
                words = [(_LEAD_S + ws, _LEAD_S + we, w) for ws, we, w in rec["words"]]
                return self._episode(words, dur)
        # Cold/out-of-range fallback: clean estimated timestamps for one message.
        msg = random.choice(candidates)
        dur = _wpm_duration_s(msg, self.wpm)
        words = _estimate_word_timestamps(msg, _LEAD_S, _LEAD_S + dur, self.wpm)
        print("[LongMonologueSource] WARNING: no warmed in-range message found; "
              "using clean fallback. Run scripts/warm_asr_cache.py to enable ASR noise.")
        return self._episode(words, dur)

    def make_simulator(self) -> "PlaybackSimulator":
        return PlaybackSimulator(self.load())


# ---------------------------------------------------------------------------
# PlaybackSimulator — single simulator for all pre-built sources
# ---------------------------------------------------------------------------

class PlaybackSimulator:
    """Plays pre-built EpisodeData through the episode loop.

    Both StaticWavSource and ScriptTTSSource reduce to EpisodeData; this class
    handles the actual per-chunk playback identically for both.
    """

    def __init__(self, data: EpisodeData) -> None:
        self._data = data
        self._offset: int = 0
        self._silence_remaining: int = 0
        self._done: bool = False

    # Episode params exposed for VirtualSimulationConnection
    @property
    def block_s(self) -> float:
        return self._data.block_s

    @property
    def wpm(self) -> int:
        return self._data.wpm

    @property
    def max_episode_s(self) -> float:
        return self._data.max_episode_s

    def reset(self) -> None:
        self._offset = 0
        self._silence_remaining = int(self._data.silence_after_s * ASR_SAMPLE_RATE)
        self._done = False

    def get_audio_chunk(self, chunk_samples: int, sample_rate: int) -> Optional[np.ndarray]:
        if self._done:
            return None
        audio = self._data.audio
        if self._offset < len(audio):
            chunk = audio[self._offset : self._offset + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            self._offset += chunk_samples
            return chunk.astype(np.float32)
        if self._silence_remaining > 0:
            n = min(chunk_samples, self._silence_remaining)
            self._silence_remaining -= n
            return np.zeros(chunk_samples, dtype=np.float32)
        self._done = True
        return None

    def get_transcript_at_time(self, t_start: float, t_end: float) -> str:
        # Include words whose MIDPOINT falls inside the window so each word
        # is assigned to exactly one block (no bleed-across-boundary doubling).
        return " ".join(
            w for ws, we, w in self._data.word_timestamps
            if t_start <= (ws + we) / 2 < t_end
        )

    def on_agent_tts(self, sample_rate: int, audio: np.ndarray) -> None:
        pass  # pre-built sources don't react to agent TTS

    def make_simulator(self) -> "PlaybackSimulator":
        return PlaybackSimulator(self._data)


# ---------------------------------------------------------------------------
# GPTVoiceSimulator — bidirectional live simulator (unchanged from original)
# ---------------------------------------------------------------------------

class GPTVoiceSimulator:
    """Bidirectional simulator using the OpenAI Realtime API.

    A background asyncio thread holds the WebSocket.
    - The agent's TTS audio is piped into GPT's microphone.
    - GPT's spoken response becomes mic audio for the agent.
    - Requires OPENAI_API_KEY and the `websockets` package.

    If script_lines is provided, the first line is sent as a text message
    to GPT at session start (primes the conversation).

    Note: this simulator makes real API calls and incurs cost. Use
    StaticWavSource or ScriptTTSSource for bulk training runs.
    """

    has_audio: bool = True  # live audio, always scored by reward model

    def __init__(
        self,
        gpt_model: str = "gpt-4o-realtime-preview",
        gpt_system_prompt: Optional[str] = None,
        script_lines: Optional[List[str]] = None,
        block_s: float = 2.0,
        wpm: int = 120,
        max_episode_s: float = 72.0,
        source_id: Optional[str] = None,
    ) -> None:
        self.gpt_model = gpt_model
        self.gpt_system_prompt = gpt_system_prompt
        self.script_lines = script_lines
        self.block_s = block_s
        self.wpm = wpm
        self.max_episode_s = max_episode_s
        self.source_id = source_id

        self._incoming: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._outgoing: queue.Queue[Optional[Tuple]] = queue.Queue()
        self._transcripts: List[Tuple[float, float, str]] = []
        self._done_event = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        self._episode_start_real: float = 0.0
        self._lock = threading.Lock()

    def make_simulator(self) -> "GPTVoiceSimulator":
        return GPTVoiceSimulator(
            gpt_model=self.gpt_model,
            gpt_system_prompt=self.gpt_system_prompt,
            script_lines=self.script_lines,
            block_s=self.block_s,
            wpm=self.wpm,
            max_episode_s=self.max_episode_s,
            source_id=self.source_id,
        )

    def reset(self) -> None:
        self._incoming = queue.Queue()
        self._outgoing = queue.Queue()
        with self._lock:
            self._transcripts = []
        self._done_event = threading.Event()
        self._episode_start_real = time.time()
        self._ws_thread = threading.Thread(
            target=self._run_ws, daemon=True, name="gpt-voice-ws"
        )
        self._ws_thread.start()
        if self.script_lines:
            self._outgoing.put(("text", self.script_lines[0]))

    def on_agent_tts(self, sample_rate: int, audio: np.ndarray) -> None:
        self._outgoing.put(("audio", (sample_rate, audio)))

    def get_audio_chunk(self, chunk_samples: int, sample_rate: int) -> Optional[np.ndarray]:
        if self._done_event.is_set():
            try:
                return self._incoming.get_nowait()
            except queue.Empty:
                return None
        try:
            return self._incoming.get(timeout=0.5)
        except queue.Empty:
            return np.zeros(chunk_samples, dtype=np.float32)

    def get_transcript_at_time(self, t_start: float, t_end: float) -> str:
        with self._lock:
            return " ".join(
                w for ws, we, w in self._transcripts if ws < t_end and we > t_start
            )

    def _run_ws(self) -> None:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._ws_async())
        except Exception as exc:
            print(f"[GPTVoiceSimulator] error: {exc!r}")
        finally:
            loop.close()
            self._done_event.set()

    async def _ws_async(self) -> None:
        try:
            import websockets  # type: ignore
        except ImportError:
            raise RuntimeError("GPTVoiceSimulator requires 'websockets'. pip install websockets")

        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — required for GPTVoiceSimulator")

        url = f"wss://api.openai.com/v1/realtime?model={self.gpt_model}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        async with websockets.connect(url, additional_headers=headers) as ws:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "modalities": ["text", "audio"],
                    "instructions": (
                        self.gpt_system_prompt
                        or "You are a conversational partner. Keep replies brief."
                    ),
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                    },
                },
            }))
            send_task = asyncio.create_task(self._send_loop(ws))
            recv_task = asyncio.create_task(self._recv_loop(ws))
            done, pending = await asyncio.wait(
                [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _send_loop(self, ws: Any) -> None:
        loop = asyncio.get_event_loop()
        while not self._done_event.is_set():
            try:
                item = await loop.run_in_executor(
                    None, lambda: self._outgoing.get(timeout=0.1)
                )
            except queue.Empty:
                continue
            if item is None:
                break
            kind, data = item[0], item[1]
            if kind == "text":
                await ws.send(json.dumps({
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": data}],
                    },
                }))
                await ws.send(json.dumps({"type": "response.create"}))
            elif kind == "audio":
                sr, arr = data
                if arr.dtype == np.float32:
                    arr = (np.clip(arr, -1.0, 1.0) * 32767).astype(np.int16)
                else:
                    arr = arr.astype(np.int16)
                encoded = base64.b64encode(arr.tobytes()).decode()
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": encoded,
                }))

    async def _recv_loop(self, ws: Any) -> None:
        async for raw in ws:
            msg = json.loads(raw)
            mtype = msg.get("type", "")

            if mtype == "response.audio.delta":
                pcm = base64.b64decode(msg.get("delta", ""))
                arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                self._incoming.put(arr)

            elif mtype == "response.audio_transcript.done":
                transcript = msg.get("transcript", "").strip()
                if transcript:
                    dur = _wpm_duration_s(transcript, self.wpm)
                    t_real = time.time() - self._episode_start_real
                    wts = _estimate_word_timestamps(
                        transcript, max(0.0, t_real - dur), t_real, self.wpm
                    )
                    with self._lock:
                        self._transcripts.extend(wts)

            elif mtype in ("response.done", "error"):
                if mtype == "error":
                    print(f"[GPTVoiceSimulator] API error: {msg}")
                self._done_event.set()
                break


# ---------------------------------------------------------------------------
# DataPool — weighted sampler across multiple sources
# ---------------------------------------------------------------------------

class DataPool:
    """Weighted pool of episode sources.

    Each source must implement ``make_simulator()``.  Sources are sampled with
    replacement by default; use ``iter_batches()`` for epoch-based training
    where each source appears exactly once per epoch.

    Build from raw lists with ``DataPool.from_lists()``::

        pool = DataPool.from_lists(
            scripts=[["Hello!", "How are you?"]],
            wav_paths=["calls/customer.wav"],
            gpt_prompts=["You are a curious student."],
            script_weight=2.0, gpt_weight=1.0,
        )
    """

    def __init__(
        self,
        sources: List[Any],
        weights: Optional[List[float]] = None,
    ) -> None:
        if not sources:
            raise ValueError("DataPool requires at least one source")
        self._sources = sources
        if weights is not None:
            if len(weights) != len(sources):
                raise ValueError("weights length must match sources length")
            total = sum(weights)
            self._weights = [w / total for w in weights]
        else:
            self._weights = [1.0 / len(sources)] * len(sources)

    def __len__(self) -> int:
        return len(self._sources)

    def sample(self, k: int) -> List[Any]:
        """Return k simulators sampled with replacement (weighted)."""
        chosen = random.choices(self._sources, weights=self._weights, k=k)
        return [s.make_simulator() for s in chosen]

    def iter_batches(
        self, batch_size: int, shuffle: bool = True
    ) -> "Iterator[List[Any]]":
        """Yield batches covering every source exactly once (one epoch).

        Sources are expanded by weight so higher-weighted sources appear more
        often. The expanded list is optionally shuffled before batching.
        """
        import math
        from itertools import islice

        # Scale weights so the minimum non-zero weight → 1 repeat; others scale up.
        min_w = min(w for w in self._weights if w > 0)
        expanded: List[Any] = []
        for src, w in zip(self._sources, self._weights):
            repeats = max(1, round(w / min_w))
            expanded.extend([src] * repeats)

        if shuffle:
            random.shuffle(expanded)

        it = iter(expanded)
        while True:
            batch_sources = list(islice(it, batch_size))
            if not batch_sources:
                break
            yield [s.make_simulator() for s in batch_sources]

    @classmethod
    def from_lists(
        cls,
        scripts: Optional[List[List[str]]] = None,
        wav_paths: Optional[List[str]] = None,
        gpt_prompts: Optional[List[str]] = None,
        *,
        script_weight: float = 1.0,
        wav_weight: float = 1.0,
        gpt_weight: float = 1.0,
        # episode defaults applied to all constructed sources
        inter_turn_pause_s: float = 7.0,
        silence_after_s: float = 8.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        gpt_model: str = "gpt-4o-realtime-preview",
        tts_model: str = "",
        device: Optional[str] = None,
    ) -> "DataPool":
        """Build a DataPool from raw lists of scripts, WAV paths, and GPT prompts.

        Per-type weights are divided equally across all entries of that type, so
        ``script_weight=2.0`` with 3 scripts and ``gpt_weight=1.0`` with 1 GPT
        gives each script a relative weight of 2/3 and the GPT a weight of 1.

        Args:
            scripts:      List of script_lines lists; each becomes a ScriptTTSSource.
            wav_paths:    List of file paths; each becomes a StaticWavSource.
            gpt_prompts:  List of system-prompt strings; each seeds a GPTVoiceSimulator.
            script_weight / wav_weight / gpt_weight: Relative weight per source type.
        """
        sources: List[Any] = []
        weights: List[float] = []

        for i, lines in enumerate(scripts or []):
            sources.append(ScriptTTSSource(
                script_lines=lines,
                inter_turn_pause_s=inter_turn_pause_s,
                silence_after_s=silence_after_s,
                max_episode_s=max_episode_s,
                block_s=block_s,
                wpm=wpm,
                source_id=f"script_{i:02d}",
                tts_model=tts_model,
                device=device,
            ))
            weights.append(script_weight / max(1, len(scripts or [])))

        for i, path in enumerate(wav_paths or []):
            sources.append(StaticWavSource(
                path=path,
                silence_after_s=silence_after_s,
                max_episode_s=max_episode_s,
                block_s=block_s,
                wpm=wpm,
                source_id=f"wav_{i:02d}",
            ))
            weights.append(wav_weight / max(1, len(wav_paths or [])))

        for i, prompt in enumerate(gpt_prompts or []):
            sources.append(GPTVoiceSimulator(
                gpt_model=gpt_model,
                gpt_system_prompt=prompt,
                block_s=block_s,
                wpm=wpm,
                max_episode_s=max_episode_s,
                source_id=f"gpt_{i:02d}",
            ))
            weights.append(gpt_weight / max(1, len(gpt_prompts or [])))

        if not sources:
            raise ValueError("DataPool.from_lists requires at least one non-empty list")

        return cls(sources, weights=weights)


# ---------------------------------------------------------------------------
# Built-in training scripts
# ---------------------------------------------------------------------------

def _load_training_scripts() -> List[List[str]]:
    json_path = os.path.join(os.path.dirname(__file__), "training_scripts.json")
    with open(json_path) as f:
        return json.load(f)


def _load_boosted_training_scripts() -> List[List[str]]:
    json_path = os.path.join(os.path.dirname(__file__), "training_scripts_boosted.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        return json.load(f)


TRAINING_SCRIPTS: List[List[str]] = _load_training_scripts()
BOOSTED_TRAINING_SCRIPTS: List[List[str]] = _load_boosted_training_scripts()


def make_default_data_pool(
    silence_after_s: float = 8.0,
    inter_turn_pause_s: float = 7.0,
    max_episode_s: float = 72.0,
    block_s: float = 2.0,
    wpm: int = 150,
    tts_model: str = "",
    device: Optional[str] = None,
    *,
    asr_noisy_fraction: float = 0.5,
    asr_voices: Optional[List[str]] = None,
    asr_n_variants: int = 3,
    ultrachat_asr_cap: int = 2000,
    monologue_weight_frac: float = 0.1,
    long_word_range: Tuple[int, int] = (40, 90),
) -> DataPool:
    """Build a DataPool from the built-in scripts + UltraChat, with optional
    TTS->ASR noise augmentation and a long-monologue dataset.

    BOOSTED_TRAINING_SCRIPTS (training_scripts_boosted.json) are sampled at 2x
    the weight of standard scripts — pure conversational Q&A that produce richer
    epsilon/RM3 gradient signal.

    ASR noise (asr_noisy_fraction in (0,1)): for each script, AND for UltraChat,
    a clean source and a cache-backed ASR-noised source are registered so the
    noisy share ≈ asr_noisy_fraction. Noisy script variants are only added if the
    script is ASR-cache-warmed (else skipped with a warning) so training never
    blocks on a live TTS/ASR call — warm with `python -m scripts.warm_asr_cache`.

    monologue_weight_frac: minority share given to LongMonologueSource (one long
    user turn, to teach "don't interrupt"). 0 disables it.
    """
    sources: List[Any] = []
    weights: List[float] = []
    f = asr_noisy_fraction
    noisy_mult = (f / (1.0 - f)) if 0.0 < f < 1.0 else 0.0
    skipped_cold = [0]

    def _warmed(lines: List[str]) -> bool:
        return all(
            any(asr_cache_entry(line, v) is not None
                for v in voices_for_text(line, asr_n_variants, asr_voices))
            for line in lines
        )

    def _add_script(lines: List[str], sid_prefix: str, i: int, clean_w: float) -> None:
        sources.append(ScriptTTSSource(
            script_lines=lines, inter_turn_pause_s=inter_turn_pause_s,
            silence_after_s=silence_after_s, max_episode_s=max_episode_s,
            block_s=block_s, wpm=wpm, source_id=f"{sid_prefix}_{i:02d}",
            tts_model=tts_model, device=device,
        ))
        weights.append(clean_w)
        if noisy_mult > 0:
            if _warmed(lines):
                sources.append(AsrAugmentedScriptSource(
                    script_lines=lines, inter_turn_pause_s=inter_turn_pause_s,
                    silence_after_s=silence_after_s, max_episode_s=max_episode_s,
                    block_s=block_s, wpm=wpm, source_id=f"asr_{sid_prefix}_{i:02d}",
                    voices=asr_voices, n_variants=asr_n_variants, device=device,
                ))
                weights.append(clean_w * noisy_mult)
            else:
                skipped_cold[0] += 1

    clean_weight_sum = 0.0
    for i, lines in enumerate(TRAINING_SCRIPTS):
        w = 0.2 if max(len(l.split()) for l in lines) <= 3 else 1.0
        clean_weight_sum += w
        _add_script(lines, "script", i, w)

    for i, lines in enumerate(BOOSTED_TRAINING_SCRIPTS):
        clean_weight_sum += 2.0
        _add_script(lines, "boosted", i, 2.0)

    if skipped_cold[0]:
        print(f"[make_default_data_pool] {skipped_cold[0]} script(s) not ASR-warmed → "
              f"noisy variant skipped. Run: python -m scripts.warm_asr_cache")

    # UltraChat: total weight ≈ all clean scripts combined × 1.5 (~50% sampling),
    # split clean/noisy by asr_noisy_fraction.
    uc_total = clean_weight_sum * 1.5
    sources.append(UltraChatTTSSource(
        silence_after_s=silence_after_s, inter_turn_pause_s=inter_turn_pause_s,
        max_episode_s=max_episode_s, block_s=block_s, wpm=wpm,
        source_id="ultrachat", tts_model=tts_model, device=device,
    ))
    weights.append(uc_total * (1.0 - f) if noisy_mult > 0 else uc_total)
    if noisy_mult > 0:
        sources.append(AsrAugmentedUltraChatSource(
            silence_after_s=silence_after_s, max_episode_s=max_episode_s,
            block_s=block_s, wpm=wpm, source_id="asr_ultrachat",
            voices=asr_voices, n_variants=asr_n_variants, device=device,
            prompt_cap=ultrachat_asr_cap,
        ))
        weights.append(uc_total * f)

    # Long-monologue dataset (minority weight) — teach no-interrupt on long turns.
    if monologue_weight_frac and monologue_weight_frac > 0:
        sources.append(LongMonologueSource(
            min_words=long_word_range[0], max_words=long_word_range[1],
            block_s=block_s, wpm=wpm, source_id="long_monologue",
            voices=asr_voices, n_variants=asr_n_variants, device=device,
        ))
        denom = max(1e-9, 1.0 - monologue_weight_frac)
        weights.append(sum(weights) * monologue_weight_frac / denom)

    return DataPool(sources, weights=weights)

"""
full_duplex.py — Real-time duplex audio agent.

Exports:
- DuplexAudioAgent : audio-in / audio-out duplex agent (Piper TTS + Parakeet ASR)
- DuplexAudioBlock : finalized conversation block dataclass
- llm_generate     : thin OpenAI wrapper used by DuplexAudioAgent
"""

import contextlib
import io
import math
import os
import queue
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import numpy as np
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from groq import Groq


load_dotenv()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

_llm_client: Optional[OpenAI] = None



def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        _llm_client = OpenAI(api_key=api_key)
    return _llm_client

def _get_groq_client() -> Groq:
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        _llm_client = Groq(api_key=api_key)
    return _llm_client


def llm_generate_openai(system_prompt: str, user_message: str) -> str:
    client = _get_llm_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-nano").strip() or "gpt-5.4-nano"
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=[{"role": "user", "content": user_message}],
        reasoning={"effort": "none"},
        max_output_tokens=12,
    )
    return response.output[0].content[0].text.strip()

GROQ_MODEL_CONFIGS = [
    {"model": "qwen/qwen3-32b", "params": {"reasoning_effort": "none"}},
    {"model": "llama-3.1-8b-instant", "params": {}},
    #{"model": "openai/gpt-oss-20b", "params": {"reasoning_effort": "low"}},
    {"model": "llama-3.3-70b-versatile", "params": {}},
    {"model": "meta-llama/llama-4-scout-17b-16e-instruct", "params": {}},
]
_next_model_index = 0
_last_used_model: str = ""

def llm_generate_groq(system_prompt: str, user_message: str) -> str:
    global _next_model_index, _last_used_model
    client = _get_groq_client()
    config = GROQ_MODEL_CONFIGS[_next_model_index]
    model = config["model"]
    _last_used_model = model
    _next_model_index = (_next_model_index + 1) % len(GROQ_MODEL_CONFIGS)
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 12,
        "temperature": 0.0,
    }
    request_params.update(config["params"])
    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content.strip()


# call local vllm. Not with caht but with completion
def local_llm_generate(system_prompt: str, user_message: str) -> str:
    import requests
    url = f"http://localhost:{os.getenv('VLLM_PORT', '8000')}/v1/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "fd",
        "prompt": user_message,
        "max_tokens": 12,
        "temperature": 0.0,
        # logprobs
        "logprobs": 5
    }
    response = requests.post(url, json=payload, headers=headers)
    out = response.json().get("choices", [{}])[0].get("text", "").strip()
    return out    
# #test it
# response  = local_llm_generate("system prompt", "Hi how are you?")
# print(response)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WPM       = 150
DEFAULT_BLOCK_S   = 2.0
TTS_SAMPLE_RATE   = 16000    # Piper PCM output rate (fallback for silence blocks)
ASR_SAMPLE_RATE   = 16000    # Parakeet expects 16 kHz; incoming mic is resampled to this
MAX_MIC_BLOCKS    = 8       # rolling mic audio window (last N agent blocks)
MAX_AUDIO_QUEUE_S = MAX_MIC_BLOCKS * DEFAULT_BLOCK_S   # ≈ 20s safety cap
MAX_HISTORY_S     = 600.0    # prune blocks older than 10 minutes

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
_template_env = Environment(loader=FileSystemLoader(_MODULE_DIR))
_prompt_template = _template_env.get_template("full-duplex.jinja2")

#TTS_MODEL = os.path.join(_MODULE_DIR, "en_US-lessac-medium.onnx") # 22kHz model
TTS_MODEL = os.path.join(_MODULE_DIR, "voices","en_US-danny-low.onnx") # 16kHz model with faster inference but lower quality

# ---------------------------------------------------------------------------
# ASR model (lazy-loaded on first use)
# ---------------------------------------------------------------------------

import logging as _logging
import os as _os
import threading as _threading
for _noisy in ("nemo", "nemo_logging", "lightning", "pytorch_lightning", "omegaconf"):
    _logging.getLogger(_noisy).setLevel(_logging.ERROR)
_os.environ.setdefault("TQDM_DISABLE", "1")  # suppress NeMo progress bars

_asr_model = None
_piper_voice_cache = {}
# NeMo's transcribe() is not thread-safe (freeze/unfreeze race).
# All background ASR threads must hold this lock before calling transcribe().
_asr_lock = _threading.Lock()
_piper_lock = _threading.Lock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AsrAlignedWord:
    text: str
    end_time: float


@dataclass
class AsrTimestampWindow:
    window_id: str
    start_ts: float
    end_ts: float
    words: List[AsrAlignedWord]
    revision: int = 0
    frozen: bool = False


@dataclass
class DuplexAudioBlock:
    """Used by DuplexAudioAgent."""
    block_id: str
    start_ts: float
    end_ts: float
    user_text: str = ""
    assistant_text: str = ""
    assistant_text_stale: bool = False
    mic_audio: Optional[np.ndarray] = None   # mic PCM resampled to ASR_SAMPLE_RATE (float32)
    tts_audio: Optional[np.ndarray] = None   # TTS PCM (int16) at tts_sr
    tts_sr: int = TTS_SAMPLE_RATE
    tts_latency_s: Optional[float] = None    # wall-clock seconds to synthesize TTS
    asr_latency_s: Optional[float] = None    # wall-clock seconds for the ASR pass that covered this block
    llm_latency_s: Optional[float] = None    # wall-clock seconds for accepted LLM generation
    total_latency_s: Optional[float] = None  # ASR-start → audio-ready latency for first reply chunk
    asr_started_perf_s: Optional[float] = None
    response_source_block_id: Optional[str] = None
    timeline_start_ts: Optional[float] = None
    timeline_end_ts: Optional[float] = None
    lead_silence_s: float = 0.0


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    import torch
    import torchaudio
    tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, from_sr, to_sr)
    return resampled.squeeze(0).numpy()


def resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def preload_piper_voice(
    tts_model: str = TTS_MODEL,
    device: Optional[str] = None,
):
    resolved_device = resolve_device(device)
    cache_key = (tts_model, resolved_device)
    with _piper_lock:
        cached = _piper_voice_cache.get(cache_key)
        if cached is not None:
            return cached

        from pathlib import Path
        from piper.download_voices import download_voice

        download_voice("en_US-danny-low", Path(tts_model).parent, force_redownload=False)

        from piper.voice import PiperVoice

        voice = PiperVoice.load(tts_model, use_cuda=(resolved_device == "cuda"))
        _piper_voice_cache[cache_key] = voice
        return voice


def preload_asr_model():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr

        device = resolve_device()
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2",
            map_location=device,
        )
        _asr_model.to(device)
        try:
            from nemo.utils import logging as _nemo_logging
            import logging as _py_logging

            _nemo_logging.setLevel(_py_logging.ERROR)
        except Exception:
            pass
    return _asr_model


def preload_duplex_models(
    tts_model: str = TTS_MODEL,
    device: Optional[str] = None,
) -> None:
    resolved_device = resolve_device(device)
    preload_piper_voice(tts_model=tts_model, device=resolved_device)
    preload_asr_model()


def piper_synthesize(voice: Any, text: str) -> tuple:
    """Synthesize text with a loaded PiperVoice. Returns (sample_rate, int16_array).

    Handles all known Piper Python API variants (synthesize / synthesize_stream_raw
    / synthesize_wav) so callers outside DuplexAudioAgent can reuse this logic.
    """
    if hasattr(voice, "synthesize"):
        sample_rate = getattr(getattr(voice, "config", None), "sample_rate", TTS_SAMPLE_RATE)
        audio_chunks: List[bytes] = []
        for chunk in voice.synthesize(text):
            chunk_rate = getattr(chunk, "sample_rate", None)
            if chunk_rate:
                sample_rate = chunk_rate
            audio_bytes = getattr(chunk, "audio_int16_bytes", b"")
            if audio_bytes:
                audio_chunks.append(audio_bytes)
        return sample_rate, np.frombuffer(b"".join(audio_chunks), dtype=np.int16)

    if hasattr(voice, "synthesize_stream_raw"):
        audio_bytes = b"".join(voice.synthesize_stream_raw(text))
        sample_rate = getattr(getattr(voice, "config", None), "sample_rate", TTS_SAMPLE_RATE)
        return sample_rate, np.frombuffer(audio_bytes, dtype=np.int16)

    if hasattr(voice, "synthesize_wav"):
        import io
        import wave as _wave
        wav_buffer = io.BytesIO()
        with _wave.open(wav_buffer, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        wav_buffer.seek(0)
        with _wave.open(wav_buffer, "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            audio_bytes = wav_file.readframes(wav_file.getnframes())
        return sample_rate, np.frombuffer(audio_bytes, dtype=np.int16)

    raise RuntimeError("Unsupported PiperVoice API: no synthesize method available")


# ---------------------------------------------------------------------------
# DuplexAudioAgent  (audio-in / audio-out — Piper TTS + Parakeet ASR)
# ---------------------------------------------------------------------------

class DuplexAudioAgent:
    def __init__(
        self,
        wpm: int = DEFAULT_WPM,
        default_block_s: float = DEFAULT_BLOCK_S,
        tts_model: str = TTS_MODEL,
        device: Optional[str] = None,
        llm_generate_fn: Callable[[str, str], str] = llm_generate_groq,
        max_prompt_blocks: int = 20,
        # Injected for testing (None → use real implementations)
        tts_fn: Optional[Callable[[str], tuple]] = None,
        asr_fn: Optional[Callable] = None,
    ):
        device = resolve_device(device)

        self._n = math.ceil(wpm * default_block_s / 60)
        self._default_block_s = default_block_s
        self._device = device
        self._llm_generate_fn = llm_generate_fn
        self._max_prompt_blocks = max_prompt_blocks
        self._tts_model = tts_model

        # Conversation history
        self.blocks: List[DuplexAudioBlock] = []
        self._current_block: Optional[DuplexAudioBlock] = None

        # LLM / word queue
        self.context_version: int = 0
        self._llm_in_flight: bool = False
        self._pending_words: List[str] = []
        self._committed_words: List[str] = []
        self._pending_llm_latency_s: Optional[float] = None
        self._pending_response_asr_latency_s: Optional[float] = None
        self._pending_response_source_block_id: Optional[str] = None
        self._latest_user_source_block_id: Optional[str] = None
        self._last_accepted_response_context_version: Optional[int] = None
        self._last_ctx_flush_user_fingerprint: str = ""
        self.last_llm_error: Optional[str] = None
        self.last_llm_error_seq: int = 0

        # Block timing
        self._next_block_ts: float = 0.0

        # Audio
        self._tts_fn = tts_fn
        self._piper_voice = None
        self._audio_queue: queue.Queue[tuple] = queue.Queue()

        # Logging
        self.quiet: bool = False

        # Mic ASR
        self._asr_fn = asr_fn
        self._mic_rolling: List[tuple] = []
        self._mic_current: np.ndarray = np.zeros(0, dtype=np.float32)
        self._executor = ThreadPoolExecutor(max_workers=2)

        # ASR windows
        self.asr_windows: List[AsrTimestampWindow] = []
        self.max_asr_windows: int = 20
        self.mutable_asr_windows: int = 5

        # Block timestamp tracking — carries previous block's end_ts as next start_ts
        self._block_start_ts: float = 0.0

        # Eagerly load Piper TTS + Parakeet ASR so first-use latency is zero.
        # Only runs when both implementations are real (neither tts_fn nor asr_fn
        # is injected). Test doubles bypass this block entirely.
        if self._tts_fn is None and self._asr_fn is None:
            print("[init] loading Piper TTS voice…")
            self._get_piper_voice()
            print("[init] Piper TTS ready")
            print("[init] loading Parakeet ASR model…")
            self._get_asr_model()
            print("[init] Parakeet ASR ready")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _now(self) -> float:
        return time.time()

    def _get_block_by_id(self, block_id: Optional[str]) -> Optional[DuplexAudioBlock]:
        if not block_id:
            return None
        if self._current_block is not None and self._current_block.block_id == block_id:
            return self._current_block
        for block in reversed(self.blocks):
            if block.block_id == block_id:
                return block
        return None

    def _clear_pending_response_timing(self) -> None:
        self._pending_llm_latency_s = None
        self._pending_response_asr_latency_s = None
        self._pending_response_source_block_id = None

    def _clear_timeline_metadata(self, block: DuplexAudioBlock) -> None:
        block.timeline_start_ts = None
        block.timeline_end_ts = None
        block.lead_silence_s = 0.0

    def _invalidate_future_assistant_continuation(self) -> None:
        self._pending_words = []
        self._committed_words = []
        self._clear_pending_response_timing()

    def _mark_assistant_history_stale_from(self, start_index: int) -> None:
        for block in self.blocks[start_index:]:
            if block.assistant_text:
                block.assistant_text_stale = True

    def _assistant_text_for_prompt(self, block: DuplexAudioBlock) -> str:
        if block.assistant_text and not block.assistant_text_stale:
            return block.assistant_text
        return ""

    @staticmethod
    def _short_text(text: str, limit: int = 48) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _change_tokens(self, text: str) -> List[str]:
        normalized = self._normalize(text).lower().strip()
        if not normalized:
            return []
        collapsed = re.sub(r"[^a-z0-9'\s]+", " ", normalized)
        return [token for token in collapsed.split() if token]

    def _classify_text_change(self, old: str, new: str) -> str:
        if not old:
            return "new"

        old_tokens = self._change_tokens(old)
        new_tokens = self._change_tokens(new)

        if old_tokens == new_tokens:
            return "punctuation"

        if old_tokens and new_tokens:
            shared_prefix = 0
            for old_token, new_token in zip(old_tokens, new_tokens):
                if old_token != new_token:
                    break
                shared_prefix += 1

            if shared_prefix == min(len(old_tokens), len(new_tokens)):
                if len(new_tokens) > len(old_tokens):
                    return "extension"
                return "trim"

            overlap = len(set(old_tokens) & set(new_tokens))
            smaller = min(len(set(old_tokens)), len(set(new_tokens)))
            if smaller > 0 and overlap >= max(1, smaller - 1):
                return "rephrase"

        return "topic-shift"

    def _history_summary(self, blocks: List[DuplexAudioBlock]) -> str:
        if not blocks:
            return "(empty)"

        parts = []
        for block in blocks:
            user_text = self._short_text(block.user_text) if block.user_text else "-"
            visible_assistant_text = self._assistant_text_for_prompt(block)
            assistant_text = self._short_text(visible_assistant_text) if visible_assistant_text else "-"
            if block.assistant_text and block.assistant_text_stale:
                hidden_text = self._short_text(block.assistant_text)
                parts.append(
                    f"[u={user_text!r} ai={assistant_text!r} stale_ai={hidden_text!r}]"
                )
            else:
                parts.append(f"[u={user_text!r} ai={assistant_text!r}]")
        return " | ".join(parts)

    def _get_piper_voice(self):
        if self._piper_voice is None:
            self._piper_voice = preload_piper_voice(
                tts_model=self._tts_model,
                device=self._device,
            )
        return self._piper_voice

    def _get_asr_model(self):
        return preload_asr_model()

    def _ensure_current_block(self, now: float) -> None:
        if self._current_block is None:
            self._current_block = DuplexAudioBlock(
                block_id=self._new_id(),
                start_ts=now,
                end_ts=now + self._default_block_s,
            )

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "--")
        )

    def _norm(self, text: str) -> str:
        return self._normalize(text).strip()

    def _queue_word_key(self, word: str) -> str:
        normalized = self._normalize(word).lower().strip()
        stripped = re.sub(r"^[^a-z0-9']+", "", normalized)
        stripped = re.sub(r"[^a-z0-9']+$", "", stripped)
        return stripped or normalized

    def _shared_suffix_prefix_len(self, left: List[str], right: List[str]) -> int:
        max_overlap = min(len(left), len(right))
        if max_overlap == 0:
            return 0

        left_keys = [self._queue_word_key(word) for word in left]
        right_keys = [self._queue_word_key(word) for word in right]
        for size in range(max_overlap, 0, -1):
            if left_keys[-size:] == right_keys[:size]:
                return size
        return 0

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def _vlog(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def _synthesize_piper_audio(self, voice, text: str) -> tuple[int, np.ndarray]:
        return piper_synthesize(voice, text)

    def _generate_tts(self, text: str) -> tuple:
        """Returns (sample_rate, audio_int16, latency_s)."""
        if self._tts_fn is not None:
            sr, arr = self._tts_fn(text)
            return sr, arr, 0.0
        t0 = time.perf_counter()
        voice = self._get_piper_voice()
        sr, arr = self._synthesize_piper_audio(voice, text)
        elapsed = time.perf_counter() - t0
        self._vlog(f"[tts] {repr(text)} → {len(arr)/sr:.2f}s audio  (synthesized in {elapsed:.3f}s)")
        return sr, arr, elapsed

    # ------------------------------------------------------------------
    # Mic ASR
    # ------------------------------------------------------------------

    def receive_mic_chunk(self, sample_rate: int, audio_array: np.ndarray) -> Optional[tuple]:
        """Accumulate mic audio. Returns next TTS chunk from output queue if ready."""
        arr = np.array(audio_array, dtype=np.float32)
        if sample_rate != ASR_SAMPLE_RATE:
            arr = _resample(arr, sample_rate, ASR_SAMPLE_RATE)
        self._mic_current = np.concatenate([self._mic_current, arr])
        chunk = self._drain_audio_queue()
        return chunk

    def _seal_mic_block(self, start_ts: float, end_ts: float) -> None:
        # Timestamps use server receive time, not client "spoken at" time.
        # Audio streams over a single ordered TCP connection so chunks always
        # arrive in order; for local deployment latency is ~1ms and consistent,
        # so server clock accurately reflects the real-time audio flow.
        sealed = self._mic_current.copy()
        self._mic_current = np.zeros(0, dtype=np.float32)
        self._mic_rolling.append((start_ts, end_ts, sealed))
        if len(self._mic_rolling) > MAX_MIC_BLOCKS:
            self._mic_rolling.pop(0)

        # Store mic PCM into the matching historical block
        for block in reversed(self.blocks):
            if abs(block.start_ts - start_ts) < 0.5:
                block.mic_audio = sealed
                break

        rolling_copy = list(self._mic_rolling)
        if self._asr_fn is not None:
            self._executor.submit(self._asr_fn, rolling_copy, self)
        else:
            self._executor.submit(self._run_parakeet, rolling_copy)

    def _user_content_fingerprint(self, rolling: list) -> str:
        """Normalized token sequence from all rolling blocks' user text.
        Identical fingerprint across ASR passes means the same words shifted
        block slots (timestamp drift) but no new content arrived."""
        tokens = []
        for start_ts, _, _ in rolling:
            for block in self.blocks:
                if abs(block.start_ts - start_ts) < 0.5:
                    if block.user_text:
                        tokens.extend(self._change_tokens(block.user_text))
                    break
        return " ".join(tokens)

    def _strip_boundary_duplicates(self, n_rolling: int) -> None:
        """Remove leading words from a block that duplicate the trailing word of the prior block.

        Operates on self.blocks (historical text), not the windows dict, so it
        catches cases where a frozen block kept a stale word that later drifted
        into the mutable block that follows it.  Only scans the recent window to
        keep this O(n_rolling).
        """
        start = max(0, len(self.blocks) - n_rolling - 1)
        for i in range(start, len(self.blocks) - 1):
            prev = self.blocks[i]
            curr = self.blocks[i + 1]
            if not prev.user_text or not curr.user_text:
                continue
            prev_words = prev.user_text.split()
            curr_words = curr.user_text.split()
            changed = False
            while prev_words and curr_words and self._norm(prev_words[-1]) == self._norm(curr_words[0]):
                curr_words.pop(0)
                changed = True
            if changed:
                new_text = " ".join(curr_words)
                print(
                    f"[asr→dedup] block@{curr.start_ts:.1f} stripped leading duplicate "
                    f"from {curr.user_text!r} → {new_text!r}"
                )
                curr.user_text = new_text

    def _deduplicate_block_boundary(self, windows: dict, n_rolling: int) -> None:
        """Drop first word of first-mutable block when it duplicates last word of frozen-boundary block.

        NeMo's word timestamps can shift slightly when given more audio context, causing the
        same word to appear at the tail of the last frozen block and the head of the first
        mutable block in successive ASR passes. Since the frozen block can't be corrected,
        we drop the duplicate from the mutable side.
        """
        frozen_idx = n_rolling - self.mutable_asr_windows - 1
        mutable_idx = n_rolling - self.mutable_asr_windows
        if frozen_idx < 0 or mutable_idx >= n_rolling:
            return
        frozen_words = windows.get(frozen_idx)
        mutable_words = windows.get(mutable_idx)
        if not frozen_words or not mutable_words:
            return
        if self._norm(frozen_words[-1][0]) == self._norm(mutable_words[0][0]):
            print(
                f"[asr→dedup] boundary idx={frozen_idx}/{mutable_idx} "
                f"dropped duplicate {mutable_words[0][0]!r} from mutable block"
            )
            mutable_words.pop(0)
            if not mutable_words:
                del windows[mutable_idx]

    def _run_parakeet(self, rolling: List[tuple]) -> None:
        import tempfile
        import soundfile as sf

        if not rolling:
            return
        full_audio = np.concatenate([audio for _, _, audio in rolling])
        if len(full_audio) == 0:
            return

        buf_start_ts = rolling[0][0]
        model = self._get_asr_model()
        asr_started_perf = time.perf_counter()

        tmp_path = None
        asr_t0 = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(f.name, full_audio, ASR_SAMPLE_RATE)
            with _asr_lock:
                with contextlib.redirect_stderr(io.StringIO()):
                    output = model.transcribe([tmp_path], timestamps=True)
        except Exception as exc:
            print(f"[asr] transcribe error: {exc!r}")
            import traceback; traceback.print_exc()
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        asr_latency = time.perf_counter() - asr_t0

        text = repr(output[0].text) if output else "(empty)"
        print(f"[asr] transcribed {len(full_audio)/ASR_SAMPLE_RATE:.1f}s -> {text}  ({asr_latency:.3f}s)")
        word_segments = output[0].timestamp.get("word", []) if output else []

        # Map each word to its rolling block by absolute end timestamp
        windows: dict = {}
        for seg in word_segments:
            word = seg.get("word", "").strip()
            if not word:
                continue
            abs_end = buf_start_ts + seg["end"]
            for idx, (start, end, _) in enumerate(rolling):
                if start <= abs_end < end:
                    windows.setdefault(idx, []).append((word, abs_end))
                    break

        # Directly write user text into the matching historical block.
        # Last mutable_asr_windows blocks can be corrected; older ones are frozen.
        # NOTE: ASR corrections do NOT increment context_version — that would race
        # with in-flight LLM calls and cause every result to be discarded as stale.
        # The periodic poll() reruns the LLM each block, so corrections are picked up
        # automatically on the next cycle.
        n_rolling = len(rolling)
        self._deduplicate_block_boundary(windows, n_rolling)

        # Fix A: clear mutable blocks whose words shifted to a later block this pass.
        # Without this, stale text persists in those slots and the fingerprint check
        # below sees spurious duplicate tokens.
        _mutable_start_idx = max(0, n_rolling - self.mutable_asr_windows)
        for _idx in range(_mutable_start_idx, n_rolling):
            if _idx in windows:
                continue
            _block_start_ts = rolling[_idx][0]
            for _blk in self.blocks:
                if abs(_blk.start_ts - _block_start_ts) < 0.5:
                    if _blk.user_text:
                        print(f"[asr→block] block@{_block_start_ts:.1f} shift-clear old={_blk.user_text!r}")
                        _blk.user_text = ""
                    break

        # Fix C: freeze ASR updates after the user goes silent.
        # Silence is measured from HISTORICAL block text, not the current ASR windows.
        # Measuring from windows would fail when a timestamp-drifted trailing word
        # appears in the latest empty block — that word defeats a windows-based freeze
        # but the historical blocks correctly show the user has been silent.
        _recent_blocks = self.blocks[-n_rolling:] if len(self.blocks) >= n_rolling else self.blocks
        _trailing_silence = 0
        for _blk in reversed(_recent_blocks):
            if _blk.user_text:
                break
            _trailing_silence += 1
        _speech_frozen = _trailing_silence >= 2

        any_changed = False
        earliest_changed_index = None
        latest_changed_target = None
        for idx, words in windows.items():
            block_start_ts, _, _ = rolling[idx]
            word_text = " ".join(w for w, _ in words)
            is_mutable = idx >= (n_rolling - self.mutable_asr_windows)

            # Find the historical block whose start_ts matches
            target = None
            target_index = None
            for block_index in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[block_index]
                if abs(block.start_ts - block_start_ts) < 0.5:
                    target = block
                    target_index = block_index
                    break

            if target is not None:
                target.asr_latency_s = asr_latency
                target.asr_started_perf_s = asr_started_perf
                if is_mutable or not target.user_text:
                    if _speech_frozen:
                        if target.user_text:
                            continue  # already-transcribed block: always freeze
                        else:
                            # Empty block during freeze: allow only if the word contains
                            # tokens NOT seen in the most recent non-empty block.
                            # A word whose tokens are all already there is a timestamp-
                            # drifted trailing word, not genuine new speech.
                            _prev_text = ''
                            for _pi in range(target_index - 1, -1, -1):
                                if self.blocks[_pi].user_text:
                                    _prev_text = self.blocks[_pi].user_text
                                    break
                            if _prev_text:
                                _new_tok = set(self._change_tokens(word_text))
                                _prev_tok = set(self._change_tokens(_prev_text))
                                if _new_tok and _new_tok.issubset(_prev_tok):
                                    continue  # drift word, skip
                    old = target.user_text
                    target.user_text = word_text
                    if old != word_text:
                        change_kind = self._classify_text_change(old, word_text)
                        print(
                            f"[asr→block] block@{block_start_ts:.1f} {change_kind} "
                            f"old={old!r} new={word_text!r}"
                        )
                        if change_kind != "punctuation":
                            any_changed = True
                            if target_index is not None and (
                                earliest_changed_index is None or target_index < earliest_changed_index
                            ):
                                earliest_changed_index = target_index
                            latest_changed_target = target
            else:
                print(f"[asr] no matching block for start_ts={block_start_ts:.1f}")

        # Fix D: strip duplicate boundary words from historical block text.
        # When Parakeet's timestamp for a word drifts across a block boundary between
        # ASR passes, the word ends up in BOTH the tail of the previous block and the
        # head of the next block — the previous block keeps its stale text (because it
        # may be frozen or guarded by _speech_frozen) while the new pass assigns the
        # same word to the next block.  _deduplicate_block_boundary only covers the
        # one frozen/mutable boundary in the windows dict; this pass covers all
        # consecutive pairs in the actual historical block list.
        self._strip_boundary_duplicates(n_rolling)

        # Fix B: suppress flush when user content is unchanged across rolling blocks.
        # Same words in different block slots (NeMo timestamp drift) must not restart the LLM.
        if any_changed:
            _current_fp = self._user_content_fingerprint(rolling)
            if _current_fp == self._last_ctx_flush_user_fingerprint:
                any_changed = False   # same words, different slots — skip flush
            else:
                self._last_ctx_flush_user_fingerprint = _current_fp

        # Only bump context_version when text actually changed so the next LLM
        # poll picks up the new transcription — but in-flight calls are NOT
        # invalidated (staleness check uses the version captured at call start).
        if any_changed:
            if latest_changed_target is not None:
                self._latest_user_source_block_id = latest_changed_target.block_id
            self._invalidate_future_assistant_continuation()
            if earliest_changed_index is not None:
                self._mark_assistant_history_stale_from(earliest_changed_index)
            print(
                "[asr→ctx] action=flush_future_continuation+mark_stale_history "
                f"earliest_block_index={earliest_changed_index} "
                f"latest_source_block_id={self._latest_user_source_block_id}"
            )
            self._last_accepted_response_context_version = None
            self.context_version += 1

    # ------------------------------------------------------------------
    # Word queue management
    # ------------------------------------------------------------------

    def _commit_block_words(self) -> None:
        total = len(self._pending_words)
        if total == 0:
            return
        # Distribute pending words evenly across however many blocks they span.
        # e.g. 11 words → 3 blocks of [4, 4, 3]; 6 → [3, 3]; 1-5 → single block.
        # TTS pads any block under 2s, so short blocks are fine.
        n_blocks = max(1, math.ceil(total / self._n))
        base = total // n_blocks
        remainder = total % n_blocks
        n_to_commit = base + (1 if remainder > 0 else 0)
        to_commit = self._pending_words[:n_to_commit]
        self._pending_words = self._pending_words[n_to_commit:]
        if to_commit:
            text = " ".join(to_commit)
            self._current_block.assistant_text = text
            self._current_block.assistant_text_stale = False
            self._current_block.llm_latency_s = self._pending_llm_latency_s
            self._current_block.response_source_block_id = self._pending_response_source_block_id
            if self._pending_response_asr_latency_s is not None:
                self._current_block.asr_latency_s = self._pending_response_asr_latency_s
            self._committed_words.extend(to_commit)
            self._clear_pending_response_timing()
            pending_preview = " ".join(self._pending_words[:6]) + ("…" if len(self._pending_words) > 6 else "")
            self._vlog(f"[commit ✓] {text!r}  pending={len(self._pending_words)} words  [{pending_preview}]")
            # Words were consumed — allow LLM to continue generating the next chunk
            # regardless of whether context_version changed. Without this, the AI
            # goes silent after the first response until the user speaks again.
            if not self._pending_words:
                self._last_accepted_response_context_version = None

    def _update_pending_queue(self, proposal_words: List[str]) -> None:
        committed = self._committed_words

        committed_overlap = self._shared_suffix_prefix_len(committed, proposal_words)
        proposal_tail = proposal_words[committed_overlap:]

        proposal_tail_norm = [self._norm(w) for w in proposal_tail]
        pending_keys = [self._queue_word_key(word) for word in self._pending_words]
        proposal_tail_keys = [self._queue_word_key(word) for word in proposal_tail_norm]

        mismatch_idx = min(len(self._pending_words), len(proposal_tail_norm))
        for i, (qw, pw) in enumerate(zip(pending_keys, proposal_tail_keys)):
            if qw != pw:
                mismatch_idx = i
                break

        retained = self._pending_words[:mismatch_idx]
        new_tail = proposal_tail_norm[mismatch_idx:]
        self._pending_words = retained + new_tail
        all_words = " ".join(self._pending_words[:10]) + ("…" if len(self._pending_words) > 10 else "")
        self._vlog(f"[queue] kept={len(retained)} +new={len(new_tail)} → {len(self._pending_words)} pending  [{all_words}]")

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _format_timeblocks(self) -> str:
        parts = []
        for block in self.blocks[-self._max_prompt_blocks:]:
            user_seg = block.user_text if block.user_text else "<idle>"
            ai_seg = self._assistant_text_for_prompt(block)
            parts.append(f"<user>{user_seg}<AI>{ai_seg}</s>")
        current_user = self._current_block.user_text if self._current_block else ""
        user_seg = current_user if current_user else "<idle>"
        parts.append(f"<user>{user_seg}<AI>")
        return "".join(parts)

    def _build_prompt(self) -> tuple:
        system_prompt = _prompt_template.render()
        user_message = self._format_timeblocks()
        return system_prompt, user_message

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _maybe_run_llm(self) -> None:
        if self._llm_in_flight:
            return
        if self._last_accepted_response_context_version == self.context_version:
            return
        has_user_input = any(b.user_text for b in self.blocks)
        if not has_user_input and (
            self._current_block is None or not self._current_block.user_text
        ):
            return

        self._llm_in_flight = True
        generation_context_version = self.context_version
        generation_source_block_id = self._latest_user_source_block_id
        try:
            system_prompt, user_message = self._build_prompt()
            _log_blocks = self.blocks[-self._max_prompt_blocks:]
            _W = 55
            header = f"┌─ LLM REQUEST  ctx={generation_context_version} {'─' * max(0, _W - 20 - len(str(generation_context_version)))}"
            lines = [header]
            for i, blk in enumerate(_log_blocks):
                idx_label = f"B[{i - len(_log_blocks)}]"
                u = (blk.user_text[:35] + "…") if len(blk.user_text or "") > 35 else (blk.user_text or "-")
                visible_ai = self._assistant_text_for_prompt(blk)
                if blk.assistant_text and blk.assistant_text_stale:
                    ai_part = f'ai="-"  stale={blk.assistant_text[:30]!r}'
                elif visible_ai:
                    ai_part = f"ai={visible_ai[:35]!r}"
                else:
                    ai_part = 'ai="-"'
                lines.append(f"│  {idx_label}  u={u!r:<38}  {ai_part}")
            lines.append(f"│  ({len(self.blocks)} total blocks, showing last {len(_log_blocks)})")
            tail = user_message[-60:].replace("\n", "↵")
            lines.append(f"│  tail → …{tail}")
            self._vlog("\n".join(lines))
            llm_t0 = time.perf_counter()
            raw_response = self._llm_generate_fn(system_prompt, user_message)
            llm_latency = time.perf_counter() - llm_t0
            raw = raw_response.strip() if isinstance(raw_response, str) else str(raw_response or "").strip()

            if generation_context_version != self.context_version:
                self._clear_pending_response_timing()
                self._vlog(f"└─ LLM ← STALE (ctx {generation_context_version} → {self.context_version})  discarded {raw!r}")
                return

            cleaned = self._normalize(raw).strip()
            for _tok in ("</s>", "<AI>", "<user>", "<s>", "<idle>"):
                cleaned = cleaned.replace(_tok, " ")
            cleaned = cleaned.strip()
            model_tag = _last_used_model.split("/")[-1] if _last_used_model else "?"
            self._vlog(f"└─ LLM ← [{model_tag}]  {cleaned!r}  ({len(cleaned.split())} words, {llm_latency:.2f}s)")
            self.last_llm_error = None

            if not cleaned:
                self._clear_pending_response_timing()
                self._pending_words = []
                self._last_accepted_response_context_version = generation_context_version
                return

            proposal_words = cleaned.split()
            if not proposal_words:
                self._clear_pending_response_timing()
                self._pending_words = []
                self._last_accepted_response_context_version = generation_context_version
                return

            source_block = self._get_block_by_id(generation_source_block_id)
            self._pending_llm_latency_s = llm_latency
            self._pending_response_source_block_id = generation_source_block_id
            self._pending_response_asr_latency_s = (
                source_block.asr_latency_s if source_block is not None else None
            )
            self._update_pending_queue(proposal_words)
            self._last_accepted_response_context_version = generation_context_version
        except Exception as exc:
            self.last_llm_error = f"{type(exc).__name__}: {exc}"
            self.last_llm_error_seq += 1
            self._clear_pending_response_timing()
            self._pending_words = []
            print(f"[llm×] {self.last_llm_error}")
        finally:
            self._llm_in_flight = False

    # ------------------------------------------------------------------
    # Audio queue
    # ------------------------------------------------------------------

    def _enqueue_audio(self, sr: int, audio: np.ndarray) -> None:
        total_queued_s = sum(len(a) / s for s, a in list(self._audio_queue.queue))
        if total_queued_s < MAX_AUDIO_QUEUE_S:
            self._audio_queue.put((sr, audio))

    def _drain_audio_queue(self) -> Optional[tuple]:
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # History pruning
    # ------------------------------------------------------------------

    def _prune_history(self, now: float) -> None:
        cutoff = now - MAX_HISTORY_S
        self.blocks = [b for b in self.blocks if b.end_ts >= cutoff]

    # ------------------------------------------------------------------
    # User text input
    # ------------------------------------------------------------------

    def receive_text_message(self, text: str, ts: Optional[float] = None) -> None:
        text = text.strip()
        if not text:
            return
        self.context_version += 1
        self._last_accepted_response_context_version = None
        self._invalidate_future_assistant_continuation()
        now = ts if ts is not None else self._now()
        self._ensure_current_block(now)
        print(f"[user] ctx={self.context_version} {repr(text)}")
        if self._current_block.user_text:
            self._current_block.user_text += " " + text
        else:
            self._current_block.user_text = text
        self._latest_user_source_block_id = self._current_block.block_id

    # ------------------------------------------------------------------
    # Main poll loop
    # ------------------------------------------------------------------

    def poll(self) -> Optional[tuple]:
        """
        Advance the block schedule. Returns (sample_rate, audio_array) when
        audio is ready to play, or None if nothing new is available yet.
        """
        now = self._now()
        if now < self._next_block_ts:
            chunk = self._drain_audio_queue()
            if chunk is not None:
                return chunk
            self._maybe_run_llm()
            return None

        # Use previous block's end time as this block's start (0.0 → use now on first poll)
        block_start = self._block_start_ts if self._block_start_ts else now
        self._ensure_current_block(block_start)
        self._commit_block_words()

        finalized = self._current_block
        finalized.end_ts = now
        self._block_start_ts = now   # carry forward for next block's start_ts
        self.blocks.append(finalized)
        self._current_block = None

        self._vlog(f"[poll] block#{len(self.blocks)} user={repr(finalized.user_text)} ai={repr(finalized.assistant_text)}")

        if finalized.assistant_text:
            sr, playback_audio, tts_latency = self._generate_tts(finalized.assistant_text)
            finalized.tts_sr = sr
            finalized.tts_latency_s = tts_latency
            source_block = self._get_block_by_id(finalized.response_source_block_id)
            if source_block is not None and source_block.asr_started_perf_s is not None:
                finalized.total_latency_s = time.perf_counter() - source_block.asr_started_perf_s
            audio = playback_audio
            duration = len(playback_audio) / sr
            if duration > 4.0:
                self._vlog(f"[tts] WARNING: block {duration:.2f}s > 4s for {repr(finalized.assistant_text)}")
            elif duration < 1.0:
                self._vlog(f"[tts] WARNING: block {duration:.2f}s < 1s for {repr(finalized.assistant_text)}")
            if finalized.assistant_text[-1] in ".!?,;:" and duration < self._default_block_s:
                padding = np.zeros(int((self._default_block_s - duration) * sr), dtype=np.int16)
                audio = np.concatenate([audio, padding])
                duration = self._default_block_s
                self._vlog(f"[tts] padded to {duration:.2f}s (sentence boundary)")
            finalized.lead_silence_s = finalized.total_latency_s or 0.0
            if finalized.lead_silence_s > 0.0:
                lead_silence = np.zeros(int(finalized.lead_silence_s * sr), dtype=np.int16)
                finalized.tts_audio = np.concatenate([lead_silence, audio])
                reply_ready_ts = now + tts_latency
                finalized.timeline_start_ts = reply_ready_ts - finalized.lead_silence_s
                finalized.timeline_end_ts = reply_ready_ts + (len(audio) / sr)
            else:
                finalized.tts_audio = audio
                self._clear_timeline_metadata(finalized)
        else:
            sr = TTS_SAMPLE_RATE
            audio = np.zeros(int(self._default_block_s * sr), dtype=np.int16)
            duration = self._default_block_s
            finalized.tts_audio = audio
            finalized.tts_sr = sr
            self._clear_timeline_metadata(finalized)

        self._next_block_ts = now + duration
        self._vlog(f"[poll] next_block_ts in {duration:.2f}s  queue_depth={self._audio_queue.qsize()}")
        self._enqueue_audio(sr, audio)

        self._seal_mic_block(finalized.start_ts, now)
        self._maybe_run_llm()
        self._prune_history(now)

        return self._drain_audio_queue()

    # ------------------------------------------------------------------
    # ASR window management
    # ------------------------------------------------------------------

    def _commit_frozen_window(self, window: AsrTimestampWindow) -> None:
        if window.frozen:
            return
        committed_words = [w.text for w in window.words if w.text]
        if committed_words:
            self.receive_text_message(" ".join(committed_words), ts=window.end_ts)
        window.frozen = True

    def _apply_asr_window_policy(self) -> None:
        self.asr_windows.sort(key=lambda w: w.end_ts)
        mutable_start = max(0, len(self.asr_windows) - self.mutable_asr_windows)
        for index, window in enumerate(self.asr_windows):
            if index < mutable_start:
                self._commit_frozen_window(window)
        if len(self.asr_windows) > self.max_asr_windows:
            overflow = len(self.asr_windows) - self.max_asr_windows
            for window in self.asr_windows[:overflow]:
                self._commit_frozen_window(window)
            self.asr_windows = self.asr_windows[overflow:]

    def _is_window_mutable(self, window_id: str) -> bool:
        self.asr_windows.sort(key=lambda w: w.end_ts)
        mutable_ids = {w.window_id for w in self.asr_windows[-self.mutable_asr_windows:]}
        return window_id in mutable_ids

    def ingest_parakeet_window(
        self,
        start_ts: float,
        end_ts: float,
        words: List[tuple],
        window_id: Optional[str] = None,
    ) -> bool:
        resolved_id = window_id or f"{start_ts:.3f}-{end_ts:.3f}"
        normalized_words: List[AsrAlignedWord] = []
        for text, word_end_time in words:
            n = self._norm(text)
            if n:
                normalized_words.append(AsrAlignedWord(text=n, end_time=float(word_end_time)))
        normalized_words.sort(key=lambda w: w.end_time)

        existing = None
        for idx, window in enumerate(self.asr_windows):
            if window.window_id == resolved_id:
                existing = (idx, window)
                break

        if existing is None:
            self.asr_windows.append(AsrTimestampWindow(
                window_id=resolved_id,
                start_ts=float(start_ts),
                end_ts=float(end_ts),
                words=normalized_words,
            ))
            self._apply_asr_window_policy()
            return True

        _, existing_window = existing
        if not self._is_window_mutable(existing_window.window_id):
            return False
        existing_window.start_ts = float(start_ts)
        existing_window.end_ts = float(end_ts)
        existing_window.words = normalized_words
        existing_window.revision += 1
        self._apply_asr_window_policy()
        return True

    def get_asr_window_state(self) -> List[dict]:
        self.asr_windows.sort(key=lambda w: w.end_ts)
        return [
            {
                "window_id": w.window_id,
                "start_ts": w.start_ts,
                "end_ts": w.end_ts,
                "words": [word.text for word in w.words],
                "word_end_times": [word.end_time for word in w.words],
                "revision": w.revision,
                "frozen": w.frozen,
            }
            for w in self.asr_windows
        ]

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def get_chat_history(self) -> list:
        history: list = []
        all_blocks = list(self.blocks)
        if self._current_block is not None:
            all_blocks.append(self._current_block)
        for block in all_blocks:
            if block.user_text:
                if history and history[-1]["role"] == "user":
                    history[-1]["content"] += " " + block.user_text
                else:
                    history.append({"role": "user", "content": block.user_text})
            if block.assistant_text:
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] += " " + block.assistant_text
                else:
                    history.append({"role": "assistant", "content": block.assistant_text})
        return history

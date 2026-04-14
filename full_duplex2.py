"""
full_duplex2.py — Integrated audio duplex agent.

Differences from full_duplex.py:
- No per-word TTS timestamps (PurposedWord).  Word scheduling is block-level:
  N = ceil(WPM * block_s / 60) words committed per block; block duration = TTS
  audio length (or default_block_s when silent).
- Integrated OpenAI TTS (PCM) and Parakeet-TDT ASR.
- Prompt format: <user>/<AI>/<idle>/</s> token stream (full-duplex2.jinja2).
- poll() returns Optional[(sample_rate, audio_array)] for Gradio audio output.

full_duplex.py is unchanged; the Text tab still uses it.
"""

import math
import os
import queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

from full_duplex import AsrAlignedWord, AsrTimestampWindow, llm_generate

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_WPM       = 150
DEFAULT_BLOCK_S   = 2.0      # silence block duration
TTS_SAMPLE_RATE   = 24000    # OpenAI PCM output rate
MIC_SAMPLE_RATE   = 16000    # Parakeet expects 16 kHz
MAX_MIC_BLOCKS    = 10       # rolling mic audio window (last N agent blocks)
MAX_AUDIO_QUEUE_S = MAX_MIC_BLOCKS * DEFAULT_BLOCK_S   # ≈ 20s safety cap
MAX_HISTORY_S     = 600.0    # prune blocks older than 10 minutes

_template_dir = os.path.dirname(os.path.abspath(__file__)) or "."
_template_env = Environment(loader=FileSystemLoader(_template_dir))
_prompt_template = _template_env.get_template("full-duplex2.jinja2")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DuplexAudioBlock:
    block_id: str
    start_ts: float
    end_ts: float
    user_text: str = ""
    assistant_text: str = ""


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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DuplexAudioAgent:
    def __init__(
        self,
        wpm: int = DEFAULT_WPM,
        default_block_s: float = DEFAULT_BLOCK_S,
        tts_voice: str = "alloy",
        device: Optional[str] = None,
        llm_generate_fn: Callable[[str, str], str] = llm_generate,
        max_prompt_blocks: int = 20,
        # Injected for testing (None → use real implementations)
        tts_fn: Optional[Callable[[str], tuple]] = None,
        asr_fn: Optional[Callable] = None,
    ):
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self._n = math.ceil(wpm * default_block_s / 60)
        self._default_block_s = default_block_s
        self._device = device
        self._llm_generate_fn = llm_generate_fn
        self._max_prompt_blocks = max_prompt_blocks
        self._tts_voice = tts_voice

        # Conversation history
        self.blocks: List[DuplexAudioBlock] = []
        self._current_block: Optional[DuplexAudioBlock] = None

        # LLM / word queue
        self.context_version: int = 0
        self._llm_in_flight: bool = False
        self._pending_words: List[str] = []    # proposed, not yet spoken
        self._committed_words: List[str] = []  # spoken this utterance

        # Block timing
        self._next_block_ts: float = 0.0

        # Audio
        self._tts_fn = tts_fn
        self._tts_client: Optional[OpenAI] = None
        self._audio_queue: queue.Queue[tuple] = queue.Queue()

        # Mic ASR
        self._asr_fn = asr_fn
        self._asr_model = None   # lazy-loaded on first real ASR call
        self._mic_rolling: List[tuple] = []  # (start_ts, end_ts, audio_arr)
        self._mic_current: np.ndarray = np.zeros(0, dtype=np.float32)
        self._executor = ThreadPoolExecutor(max_workers=2)

        # ASR windows (same semantics as full_duplex.py)
        self.asr_windows: List[AsrTimestampWindow] = []
        self.max_asr_windows: int = 20
        self.mutable_asr_windows: int = 10

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _now(self) -> float:
        return time.time()

    def _get_tts_client(self) -> OpenAI:
        if self._tts_client is None:
            self._tts_client = OpenAI()
        return self._tts_client

    def _get_asr_model(self):
        if self._asr_model is None:
            import nemo.collections.asr as nemo_asr
            self._asr_model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-tdt-0.6b-v2",
            )
            self._asr_model.to(self._device)
        return self._asr_model

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

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def _generate_tts(self, text: str) -> tuple:
        if self._tts_fn is not None:
            return self._tts_fn(text)
        client = self._get_tts_client()
        response = client.audio.speech.create(
            model="tts-1",
            voice=self._tts_voice,
            input=text,
            response_format="pcm",
        )
        arr = np.frombuffer(response.content, dtype=np.int16)
        return TTS_SAMPLE_RATE, arr

    # ------------------------------------------------------------------
    # Mic ASR
    # ------------------------------------------------------------------

    def receive_mic_chunk(
        self, sample_rate: int, audio_array: np.ndarray
    ) -> Optional[tuple]:
        """Accumulate mic audio. Returns next TTS chunk from output queue if ready."""
        arr = np.array(audio_array, dtype=np.float32)
        if sample_rate != MIC_SAMPLE_RATE:
            arr = _resample(arr, sample_rate, MIC_SAMPLE_RATE)
        self._mic_current = np.concatenate([self._mic_current, arr])
        return self._drain_audio_queue()

    def _seal_mic_block(self, start_ts: float, end_ts: float) -> None:
        """Seal current mic accumulation into the rolling buffer and submit ASR."""
        sealed = self._mic_current.copy()
        self._mic_current = np.zeros(0, dtype=np.float32)
        self._mic_rolling.append((start_ts, end_ts, sealed))
        if len(self._mic_rolling) > MAX_MIC_BLOCKS:
            self._mic_rolling.pop(0)
        rolling_copy = list(self._mic_rolling)
        if self._asr_fn is not None:
            self._executor.submit(self._asr_fn, rolling_copy, self)
        else:
            self._executor.submit(self._run_parakeet, rolling_copy)

    def _run_parakeet(self, rolling: List[tuple]) -> None:
        """Run Parakeet on the full rolling buffer; distribute words by timestamp."""
        import tempfile
        import soundfile as sf

        if not rolling:
            return
        full_audio = np.concatenate([audio for _, _, audio in rolling])
        if len(full_audio) == 0:
            return

        buf_start_ts = rolling[0][0]
        model = self._get_asr_model()

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(f.name, full_audio, MIC_SAMPLE_RATE)
            output = model.transcribe([tmp_path], timestamps=True)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        word_segments = (
            output[0].timestamp.get("word", []) if output else []
        )

        # Group words into windows by matching abs end_time to a block
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

        for idx, words in windows.items():
            start_ts, end_ts, _ = rolling[idx]
            self.ingest_parakeet_window(start_ts, end_ts, words, f"mic-{idx}")

    # ------------------------------------------------------------------
    # Word queue management
    # ------------------------------------------------------------------

    def _commit_block_words(self) -> None:
        """Commit next N pending words to the current block."""
        to_commit = self._pending_words[:self._n]
        self._pending_words = self._pending_words[self._n:]
        if to_commit:
            text = " ".join(to_commit)
            self._current_block.assistant_text = text
            self._committed_words.extend(to_commit)
            print(f"[commit] {to_commit} | committed={self._committed_words}")

    def _update_pending_queue(self, proposal_words: List[str]) -> None:
        """
        Reconcile _pending_words against a new LLM proposal.

        Queue non-empty: LLM may echo committed words verbatim before continuing.
          Strip that echoed prefix, then diff proposal tail against unspoken queue.

        Queue empty: LLM returns a complete utterance that may include already-
          committed words. Walk proposal sequentially, consuming committed words
          until divergence, then treat the remainder as new.
        """
        committed = self._committed_words

        if self._pending_words:
            # Strip echoed committed prefix
            committed_overlap = 0
            for cw, pw in zip(committed, proposal_words):
                if self._norm(pw) == cw:
                    committed_overlap += 1
                else:
                    break
            proposal_tail = proposal_words[committed_overlap:]
        else:
            # Strip already-spoken words from proposal head
            matched = 0
            proposal_tail = []
            for i, pw in enumerate(proposal_words):
                if matched < len(committed) and self._norm(pw) == committed[matched]:
                    matched += 1
                else:
                    proposal_tail = proposal_words[i:]
                    break

        proposal_tail_norm = [self._norm(w) for w in proposal_tail]

        # Mismatch detection against unspoken queue
        mismatch_idx = min(len(self._pending_words), len(proposal_tail_norm))
        for i, (qw, pw) in enumerate(zip(self._pending_words, proposal_tail_norm)):
            if qw != pw:
                mismatch_idx = i
                break

        retained = self._pending_words[:mismatch_idx]
        new_tail = proposal_tail_norm[mismatch_idx:]
        self._pending_words = retained + new_tail
        print(
            f"[queue] retained={retained} new={new_tail} "
            f"→ pending={self._pending_words}"
        )

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _format_timeblocks(self) -> str:
        """Build <user>/<AI>/<idle>/</s> token stream for LLM prompt."""
        forced = self._pending_words[:self._n]
        parts = []
        for block in self.blocks[-self._max_prompt_blocks:]:
            user_seg = block.user_text if block.user_text else "<idle>"
            ai_seg = block.assistant_text if block.assistant_text else ""
            parts.append(f"<user>{user_seg}<AI>{ai_seg}</s>")
        current_user = self._current_block.user_text if self._current_block else ""
        user_seg = current_user if current_user else "<idle>"
        parts.append(f"<user>{user_seg}<AI>")
        if forced:
            parts.append(" ".join(forced))
        return "".join(parts)

    def _build_prompt(self) -> tuple:
        forced = self._pending_words[:self._n]
        system_prompt = _prompt_template.render(has_forced_words=bool(forced))
        user_message = self._format_timeblocks()
        return system_prompt, user_message

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _maybe_run_llm(self) -> None:
        if self._llm_in_flight:
            return
        has_user_input = any(b.user_text for b in self.blocks)
        if not has_user_input and (
            self._current_block is None or not self._current_block.user_text
        ):
            return

        self._llm_in_flight = True
        generation_context_version = self.context_version
        try:
            system_prompt, user_message = self._build_prompt()
            history_summary = " | ".join(
                f"[{b.user_text!r}/{b.assistant_text!r}]"
                for b in self.blocks[-3:]
            )
            print(
                f"[llm→] ctx={generation_context_version} "
                f"history(last3)={history_summary} "
                f"prompt_tail={repr(user_message[-80:])}"
            )
            raw = self._llm_generate_fn(system_prompt, user_message).strip()

            if generation_context_version != self.context_version:
                print(
                    f"[llm←] stale (ctx {generation_context_version} < "
                    f"{self.context_version}), discarding {repr(raw)}"
                )
                return

            cleaned = self._normalize(raw).strip()
            if cleaned.endswith("</s>"):
                cleaned = cleaned[:-4].strip()
            print(f"[llm←] {repr(cleaned)}")

            if not cleaned:
                self._pending_words = []
                return

            self._update_pending_queue(cleaned.split())
        finally:
            self._llm_in_flight = False

    # ------------------------------------------------------------------
    # Audio queue
    # ------------------------------------------------------------------

    def _enqueue_audio(self, sr: int, audio: np.ndarray) -> None:
        total_queued_s = sum(
            len(a) / s for s, a in list(self._audio_queue.queue)
        )
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
        """Accept a user text message (used by Text tab or frozen ASR windows)."""
        text = text.strip()
        if not text:
            return
        self.context_version += 1
        self._committed_words = []
        now = ts if ts is not None else self._now()
        self._ensure_current_block(now)
        print(f"[user] ctx={self.context_version} {repr(text)}")
        if self._current_block.user_text:
            self._current_block.user_text += " " + text
        else:
            self._current_block.user_text = text

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
            return self._drain_audio_queue()

        # Advance block
        self._ensure_current_block(now)
        self._commit_block_words()

        finalized = self._current_block
        finalized.end_ts = now
        self.blocks.append(finalized)
        self._current_block = None

        # TTS for committed words, or silence
        if finalized.assistant_text:
            sr, audio = self._generate_tts(finalized.assistant_text)
        else:
            sr = TTS_SAMPLE_RATE
            audio = np.zeros(int(self._default_block_s * sr), dtype=np.int16)

        duration = len(audio) / sr
        self._next_block_ts = now + duration
        self._enqueue_audio(sr, audio)

        # Seal mic block and submit Parakeet asynchronously
        self._seal_mic_block(finalized.start_ts, now)

        # LLM call (synchronous)
        self._maybe_run_llm()

        # Prune old history
        self._prune_history(now)

        return self._drain_audio_queue()

    # ------------------------------------------------------------------
    # ASR window management (reused from full_duplex.py semantics)
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
        mutable_ids = {
            w.window_id for w in self.asr_windows[-self.mutable_asr_windows:]
        }
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
                normalized_words.append(
                    AsrAlignedWord(text=n, end_time=float(word_end_time))
                )
        normalized_words.sort(key=lambda w: w.end_time)

        existing = None
        for idx, window in enumerate(self.asr_windows):
            if window.window_id == resolved_id:
                existing = (idx, window)
                break

        if existing is None:
            self.asr_windows.append(
                AsrTimestampWindow(
                    window_id=resolved_id,
                    start_ts=float(start_ts),
                    end_ts=float(end_ts),
                    words=normalized_words,
                )
            )
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

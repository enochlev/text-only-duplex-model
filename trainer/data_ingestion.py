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
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

from full_duplex import ASR_SAMPLE_RATE


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
        inter_turn_pause_s: float = 8.0,
        silence_after_s: float = 8.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        source_id: Optional[str] = None,
    ) -> None:
        self.script_lines = script_lines
        self.inter_turn_pause_s = inter_turn_pause_s
        self.silence_after_s = silence_after_s
        self.max_episode_s = max_episode_s
        self.block_s = block_s
        self.wpm = wpm
        self.source_id = source_id

    def load(self) -> EpisodeData:
        segments: List[np.ndarray] = []
        word_timestamps: List[Tuple[float, float, str]] = []
        t = 0.0
        for line in self.script_lines:
            dur = _wpm_duration_s(line, self.wpm)
            seg = np.zeros(int(dur * ASR_SAMPLE_RATE), dtype=np.float32)
            word_timestamps.extend(_estimate_word_timestamps(line, t, t + dur, self.wpm))
            segments.append(seg)
            t += dur
            pause_s = self.inter_turn_pause_s
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
        # Overlapping interval: include any word that touches the window,
        # not just words entirely contained within it.
        return " ".join(
            w for ws, we, w in self._data.word_timestamps
            if ws < t_end and we > t_start
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
        inter_turn_pause_s: float = 6.0,
        silence_after_s: float = 8.0,
        max_episode_s: float = 72.0,
        block_s: float = 2.0,
        wpm: int = 150,
        gpt_model: str = "gpt-4o-realtime-preview",
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

TRAINING_SCRIPTS: List[List[str]] = [
    # Casual small talk
    ["Hey, how's it going?",
     "Not bad. What have you been up to lately?",
     "That sounds fun. Do you have any weekend plans?"],

    # Question-and-answer factual
    ["What's the capital of France?",
     "How far is it from London?",
     "Is there a fast train between them?"],

    # Short follow-ups (tests barge-in handling)
    ["Tell me a joke.",
     "Another one.",
     "Okay that's enough."],

    # Task-oriented
    ["Can you help me write a grocery list?",
     "I need milk, eggs, and bread.",
     "Also some coffee and maybe some fruit.",
     "That's all, thanks."],

    # Opinion / open-ended
    ["What do you think about remote work?",
     "Do you think it's better for productivity?",
     "What about collaboration?"],

    # Tech support
    ["My laptop keeps freezing.",
     "It happens when I open too many browser tabs.",
     "I have eight gigabytes of RAM.",
     "Should I upgrade?"],

    # Number / calculation (short answers expected)
    ["What's fifteen percent of two hundred?",
     "How about twenty percent?",
     "And a thirty dollar tip on a hundred fifty dollar bill?"],

    # Storytelling / longer context
    ["Tell me a short story about a cat.",
     "Make it funnier.",
     "Give it a surprise ending."],

    # Interruption simulation — very short prompts back-to-back
    ["Wait.",
     "Sorry, go on.",
     "Actually never mind.",
     "Okay I'm ready now.",
     "What were you saying?"],

    # Single long user turn (tests truncation handling)
    ["I went to the store this morning and they were completely out of the "
     "brand of coffee I normally buy, so I had to try a different one, and "
     "honestly I don't think it's as good. What would you recommend instead?"],

    # Multi-language mix (names / places)
    ["I'm visiting Tokyo next month.",
     "What neighborhoods should I explore?",
     "Any food recommendations?"],

    # Emotional / supportive
    ["I've been really stressed lately.",
     "Work has been overwhelming.",
     "Do you have any tips for managing stress?"],

    # Silence-heavy: very short turns force idle decisions
    ["Hi.",
     "Yeah.",
     "Okay.",
     "Bye."],

    # Debate-style back-and-forth
    ["Is coffee better than tea?",
     "But tea has so many varieties.",
     "Coffee has espresso though.",
     "Fair point."],

    # Follow-up clarification
    ["Explain quantum entanglement simply.",
     "Can it be used for communication?",
     "Why not?"],
]


def make_default_data_pool(
    silence_after_s: float = 8.0,
    inter_turn_pause_s: float = 6.0,
    max_episode_s: float = 72.0,
    block_s: float = 2.0,
    wpm: int = 150,
) -> DataPool:
    """Build a DataPool from the built-in TRAINING_SCRIPTS."""
    sources = [
        ScriptTTSSource(
            script_lines=lines,
            inter_turn_pause_s=inter_turn_pause_s,
            silence_after_s=silence_after_s,
            max_episode_s=max_episode_s,
            block_s=block_s,
            wpm=wpm,
            source_id=f"script_{i:02d}",
        )
        for i, lines in enumerate(TRAINING_SCRIPTS)
    ]
    return DataPool(sources)

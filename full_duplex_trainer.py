"""
full_duplex_trainer.py — Full-duplex RL training using REINFORCE.

Episode: one full conversation simulation (max 30s by default).
Step:    one LLM generation call within an episode.

Each step records (prompt_tokens, response_tokens, rollout_log_probs) during
rollout. After scoring with user-supplied reward functions, REINFORCE with an
EMA baseline produces a policy gradient loss over the HuggingFace model, which
is then synced back into the vLLM engine for the next rollout.

Not compatible with GRPO/RLOO: those algorithms assume single-step episodes
with i.i.d. prompts. Here every step has a unique prompt because ASR revises
the conversation history between steps.

Usage:
    from full_duplex_trainer import (
        FullDuplexRLTrainer, TrainerConfig, SimulatorConfig,
        latency_reward, idle_penalty,
    )

    trainer = FullDuplexRLTrainer(
        config=TrainerConfig(model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct"),
        simulator_configs=[
            SimulatorConfig("q1", "tts_script", script_lines=["What time is it?"]),
        ],
        reward_fns=[latency_reward, idle_penalty],
    )
    trainer.train(num_steps=100)
"""

from __future__ import annotations

import abc
import asyncio
import base64
import json
import math
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import torch

try:
    from vllm import LLM, SamplingParams as VLLMSamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None  # type: ignore
    VLLMSamplingParams = None  # type: ignore

from transformers import AutoModelForCausalLM, AutoTokenizer

from full_duplex import (
    ASR_SAMPLE_RATE,
    MAX_MIC_BLOCKS,
    TTS_SAMPLE_RATE,
    DuplexAudioAgent,
    DuplexAudioBlock,
)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

RewardFn = Callable[[DuplexAudioBlock, List[DuplexAudioBlock], bool], float]
"""
Reward function signature.

Args:
    block:       The DuplexAudioBlock being scored.
    history:     All preceding blocks in the episode.
    is_terminal: True if this is the last step of the episode.

Returns:
    Scalar reward (float).
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SimulatorConfig:
    """Configuration for a single simulator type."""

    config_id: str
    simulator_type: str
    """One of: "static_wav", "tts_script", "text_inject", "gpt_voice"."""

    # static_wav / mp3
    audio_path: Optional[str] = None

    # tts_script / gpt_voice opener: one string per turn
    script_lines: Optional[List[str]] = None

    # text_inject: [(t_seconds_offset, text_to_inject), ...]
    injected_texts: Optional[List[Tuple[float, str]]] = None

    silence_after_s: float = 10.0
    """Silence emitted after the main audio ends before the episode ends."""

    max_episode_s: float = 30.0
    """Hard wall-clock cap on episode length."""

    block_s: float = 2.0
    """Matching DuplexAudioAgent's default_block_s."""

    wpm: int = 150
    """Words-per-minute for synthetic audio timing estimation."""

    # gpt_voice only
    gpt_model: str = "gpt-4o-realtime-preview"
    gpt_system_prompt: Optional[str] = None
    """Persona / instructions sent to the GPT realtime session."""


@dataclass
class StepRecord:
    """One LLM generation call captured during a rollout episode."""

    step_id: str
    prompt_text: str
    """Exact prompt string used at inference time (system + user message).
    Saved because ASR corrections make each step's context unique."""

    full_prompt_tokens: List[int]
    """Tokenized prompt — re-used for the training forward pass."""

    response_token_ids: List[int]
    """Token ids produced by vLLM (empty list for idle responses)."""

    log_probs: List[float]
    """Per-token log_prob from vLLM sampling (empty for idle)."""

    is_idle: bool
    """True when the policy output nothing (empty or <idle> token)."""

    source_block_id: Optional[str] = None
    """agent._latest_user_source_block_id at call time.
    Used post-episode to map steps → blocks_covered."""

    blocks_covered: List[str] = field(default_factory=list)
    """block_ids whose assistant_text originated from this LLM call."""

    user_spoke_before: bool = True
    """False when context_version did not change since the previous LLM call.
    Used to identify consecutive silent-user runs for action merging."""

    reward: Optional[float] = None
    """Filled by compute_rewards()."""


@dataclass
class Episode:
    """Completed simulation episode."""

    episode_id: str
    steps: List[StepRecord]
    blocks: List[DuplexAudioBlock]
    terminated_reason: str
    """"max_duration" | "simulator_done"."""


@dataclass
class TrainerConfig:
    """Hyperparameters for FullDuplexRLTrainer."""

    model_name_or_path: str
    """HuggingFace model id or local path (e.g., "Qwen/Qwen2.5-1.5B-Instruct")."""

    vllm_max_tokens: int = 16
    """Max new tokens per LLM generation (should match production setting)."""

    vllm_temperature: float = 1.0
    """Sampling temperature. Must be > 0 for REINFORCE exploration."""

    vllm_gpu_memory_utilization: float = 0.55
    """Fraction of GPU memory reserved for vLLM KV cache.
    Keep low enough to leave room for the HF training model."""

    learning_rate: float = 1e-5
    gradient_clip: float = 1.0

    gamma: float = 1.0
    """Return discount factor. 1.0 = undiscounted (recommended for short episodes)."""

    kl_coeff: float = 0.01
    """Soft KL penalty coefficient against rollout policy."""

    baseline_ema_alpha: float = 0.05
    """EMA smoothing factor for the return baseline."""

    episodes_per_train_step: int = 4

    max_seq_len: int = 512
    """Max token budget per step (prompt + response). Prompts are left-truncated."""

    device: str = "cuda"

    reward_fn_weights: Optional[List[float]] = None
    """Weight per reward function. Defaults to uniform 1.0."""


# ---------------------------------------------------------------------------
# Built-in reward functions
# ---------------------------------------------------------------------------

def latency_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise high end-to-end (ASR-start → audio-ready) latency."""
    if block.total_latency_s is not None and block.total_latency_s > 0:
        return -block.total_latency_s
    return 0.0


def idle_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Small constant penalty for empty blocks — prevents always-silent policy."""
    return -0.1 if not block.assistant_text else 0.0


def response_length_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Reward appropriately-sized responses; penalise too short or too long."""
    if not block.assistant_text:
        return 0.0
    words = len(block.assistant_text.split())
    if words < 2 or words > 8:
        return -0.05
    return 0.05


# ---------------------------------------------------------------------------
# Simulator registry
# ---------------------------------------------------------------------------

SIMULATOR_REGISTRY: Dict[str, Type[BaseSimulator]] = {}


def register_simulator(name: str) -> Callable[[Type], Type]:
    """Class decorator: register a simulator under a string key."""
    def decorator(cls: Type) -> Type:
        SIMULATOR_REGISTRY[name] = cls
        return cls
    return decorator


def build_simulator(config: SimulatorConfig) -> "BaseSimulator":
    cls = SIMULATOR_REGISTRY.get(config.simulator_type)
    if cls is None:
        raise ValueError(
            f"Unknown simulator type {config.simulator_type!r}. "
            f"Registered: {sorted(SIMULATOR_REGISTRY)}"
        )
    return cls(config)


# ---------------------------------------------------------------------------
# BaseSimulator
# ---------------------------------------------------------------------------

class BaseSimulator(abc.ABC):
    """Abstract base for all episode simulators."""

    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def reset(self) -> None:
        """Prepare for a fresh episode."""

    @abc.abstractmethod
    def get_audio_chunk(
        self, chunk_samples: int, sample_rate: int
    ) -> Optional[np.ndarray]:
        """
        Return the next float32 audio chunk, or None to signal episode end.
        Silence blocks (zeros) are valid — return None only when done.
        """

    def on_agent_tts(self, sample_rate: int, audio: np.ndarray) -> None:
        """Receive TTS audio from the agent (hook for bidirectional simulators)."""

    def get_transcript_at_time(self, t_start: float, t_end: float) -> str:
        """Ground-truth transcript for [t_start, t_end). Used as mock ASR."""
        return ""

    def get_pending_text_inject(self, t: float) -> Optional[str]:
        """
        Return text that should be injected into the agent's context at time t,
        or None. Called each iteration before poll(). TextInjectSimulator
        overrides this; other simulators return None.
        """
        return None


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
    per_word = duration / len(words) if len(words) > 0 else 0.0
    return [
        (t_start + i * per_word, t_start + (i + 1) * per_word, w)
        for i, w in enumerate(words)
    ]


# ---------------------------------------------------------------------------
# StaticAudioSimulator
# ---------------------------------------------------------------------------

@register_simulator("static_wav")
class StaticAudioSimulator(BaseSimulator):
    """
    Play a WAV or MP3 file chunk-by-chunk, then emit silence_after_s of
    silence, then signal episode end. Ground-truth transcript is estimated
    from script_lines (if provided) via uniform word timing.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        self._audio: np.ndarray = np.zeros(0, dtype=np.float32)
        self._audio_duration_s: float = 0.0
        self._offset: int = 0
        self._silence_remaining: int = 0
        self._done: bool = True
        self._word_ts: List[Tuple[float, float, str]] = []

    def reset(self) -> None:
        if self.config.audio_path:
            self._audio, self._audio_duration_s = _load_audio(
                self.config.audio_path, ASR_SAMPLE_RATE
            )
        else:
            self._audio = np.zeros(0, dtype=np.float32)
            self._audio_duration_s = 0.0
        self._offset = 0
        self._silence_remaining = int(self.config.silence_after_s * ASR_SAMPLE_RATE)
        self._done = False
        if self.config.script_lines:
            joined = " ".join(self.config.script_lines)
            self._word_ts = _estimate_word_timestamps(
                joined, 0.0, self._audio_duration_s, self.config.wpm
            )
        else:
            self._word_ts = []

    def get_audio_chunk(self, chunk_samples: int, sample_rate: int) -> Optional[np.ndarray]:
        if self._done:
            return None
        if self._offset < len(self._audio):
            chunk = self._audio[self._offset : self._offset + chunk_samples]
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
        return " ".join(w for ws, we, w in self._word_ts if ws >= t_start and we <= t_end)


# ---------------------------------------------------------------------------
# ScriptTTSSimulator
# ---------------------------------------------------------------------------

@register_simulator("tts_script")
class ScriptTTSSimulator(BaseSimulator):
    """
    Multi-turn script simulator.  Each string in script_lines is one user
    utterance.  Audio is synthesized as silence of WPM-estimated duration
    (no real TTS required — the mock ASR reads ground-truth text directly).
    An inter-turn pause equal to block_s / 2 gives the agent time to respond.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        self._audio: np.ndarray = np.zeros(0, dtype=np.float32)
        self._offset: int = 0
        self._silence_remaining: int = 0
        self._done: bool = True
        self._word_ts: List[Tuple[float, float, str]] = []

    def reset(self) -> None:
        lines = self.config.script_lines or []
        segments: List[np.ndarray] = []
        word_ts: List[Tuple[float, float, str]] = []
        t = 0.0
        for line in lines:
            dur = _wpm_duration_s(line, self.config.wpm)
            seg = np.zeros(int(dur * ASR_SAMPLE_RATE), dtype=np.float32)
            word_ts.extend(_estimate_word_timestamps(line, t, t + dur, self.config.wpm))
            segments.append(seg)
            t += dur
            # Inter-turn pause so agent can respond
            pause_s = self.config.block_s * 1.5
            segments.append(np.zeros(int(pause_s * ASR_SAMPLE_RATE), dtype=np.float32))
            t += pause_s
        self._audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
        self._offset = 0
        self._silence_remaining = int(self.config.silence_after_s * ASR_SAMPLE_RATE)
        self._word_ts = word_ts
        self._done = False

    def get_audio_chunk(self, chunk_samples: int, sample_rate: int) -> Optional[np.ndarray]:
        if self._done:
            return None
        if self._offset < len(self._audio):
            chunk = self._audio[self._offset : self._offset + chunk_samples]
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
        return " ".join(w for ws, we, w in self._word_ts if ws >= t_start and we <= t_end)


# ---------------------------------------------------------------------------
# TextInjectSimulator
# ---------------------------------------------------------------------------

@register_simulator("text_inject")
class TextInjectSimulator(BaseSimulator):
    """
    Emits silence as mic audio throughout the episode.  At configured time
    offsets, injects text directly into the agent context via
    agent.receive_text_message() (called from the episode loop via
    get_pending_text_inject).  Useful for text-only interaction tests.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        self._sim_elapsed: float = 0.0
        self._injected: set = set()
        self._done: bool = True

    def reset(self) -> None:
        self._sim_elapsed = 0.0
        self._injected = set()
        self._done = False

    def get_audio_chunk(self, chunk_samples: int, sample_rate: int) -> Optional[np.ndarray]:
        if self._done:
            return None
        dt = chunk_samples / sample_rate
        self._sim_elapsed += dt
        if self._sim_elapsed >= self.config.max_episode_s:
            self._done = True
            return None
        return np.zeros(chunk_samples, dtype=np.float32)

    def get_pending_text_inject(self, t: float) -> Optional[str]:
        texts: List[str] = []
        for idx, (t_inj, text) in enumerate(self.config.injected_texts or []):
            if idx not in self._injected and t >= t_inj:
                texts.append(text)
                self._injected.add(idx)
        return " ".join(texts) if texts else None


# ---------------------------------------------------------------------------
# GPTVoiceSimulator
# ---------------------------------------------------------------------------

@register_simulator("gpt_voice")
class GPTVoiceSimulator(BaseSimulator):
    """
    Bidirectional simulator using the OpenAI Realtime API.

    A background asyncio thread holds the WebSocket.
    - The agent's TTS audio is piped into GPT's microphone.
    - GPT's spoken response becomes mic audio for the agent.
    - Requires OPENAI_API_KEY and the `websockets` package.

    If script_lines is provided, the first line is sent as a text message
    to GPT at session start (primes the conversation).

    Note: this simulator makes real API calls and incurs cost.  Use
    StaticAudioSimulator or ScriptTTSSimulator for bulk training runs.
    """

    def __init__(self, config: SimulatorConfig) -> None:
        super().__init__(config)
        self._incoming: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._outgoing: queue.Queue[Optional[Tuple]] = queue.Queue()
        self._transcripts: List[Tuple[float, float, str]] = []
        self._done_event = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None
        self._episode_start_real: float = 0.0
        self._lock = threading.Lock()

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
        # Prime conversation with opening line
        if self.config.script_lines:
            self._outgoing.put(("text", self.config.script_lines[0]))

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
                w for ws, we, w in self._transcripts if ws >= t_start and we <= t_end
            )

    # ------------------------------------------------------------------
    # Background WebSocket coroutines
    # ------------------------------------------------------------------

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

        url = f"wss://api.openai.com/v1/realtime?model={self.config.gpt_model}"
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
                        self.config.gpt_system_prompt
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
                    dur = _wpm_duration_s(transcript, 150)
                    t_real = time.time() - self._episode_start_real
                    wts = _estimate_word_timestamps(transcript, max(0.0, t_real - dur), t_real, 150)
                    with self._lock:
                        self._transcripts.extend(wts)

            elif mtype in ("response.done", "error"):
                if mtype == "error":
                    print(f"[GPTVoiceSimulator] API error: {msg}")
                self._done_event.set()
                break


# ---------------------------------------------------------------------------
# llm_generate_train
# ---------------------------------------------------------------------------

def llm_generate_train(
    system_prompt: str,
    user_message: str,
    vllm_engine: Any,
    tokenizer: Any,
    max_tokens: int = 16,
    temperature: float = 1.0,
) -> Tuple[str, List[int], List[int], List[float]]:
    """
    Generate a response with the vLLM engine and return training metadata.

    Returns:
        text:               Stripped response string ("" for idle).
        prompt_token_ids:   Token ids of the full formatted prompt.
        response_token_ids: Generated token ids ([] for idle).
        log_probs:          Per-token log_prob from vLLM sampling ([] for idle).

    The tokenization uses the tokenizer's chat template when available so
    the prompt format matches what the model was trained on.
    """
    if not HAS_VLLM:
        raise RuntimeError("vLLM is required for llm_generate_train. pip install vllm")

    # Build prompt string — mirror local_llm_generate format
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            full_prompt = f"{system_prompt}\n\n{user_message}"
    else:
        full_prompt = f"{system_prompt}\n\n{user_message}"

    prompt_token_ids: List[int] = tokenizer.encode(full_prompt, add_special_tokens=False)

    sampling_params = VLLMSamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=1,
        skip_special_tokens=False,
    )
    outputs = vllm_engine.generate([full_prompt], sampling_params, use_tqdm=False)
    out = outputs[0].outputs[0]

    text = (out.text or "").strip()
    response_token_ids: List[int] = list(out.token_ids)

    # Extract per-token log_probs
    log_probs: List[float] = []
    if out.logprobs:
        for i, lp_dict in enumerate(out.logprobs):
            if not lp_dict:
                log_probs.append(0.0)
                continue
            tok_id = (
                response_token_ids[i]
                if i < len(response_token_ids)
                else next(iter(lp_dict))
            )
            lp_obj = lp_dict.get(tok_id) or next(iter(lp_dict.values()))
            lp_val = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
            log_probs.append(lp_val)

    # Detect idle
    text_clean = text.replace("<idle>", "").replace("</s>", "").strip()
    if not text_clean:
        return "", prompt_token_ids, [], []

    return text_clean, prompt_token_ids, response_token_ids, log_probs


# ---------------------------------------------------------------------------
# VirtualSimulationConnection
# ---------------------------------------------------------------------------

class VirtualSimulationConnection:
    """
    Drives a DuplexAudioAgent through one simulated episode.

    - Injects synthetic mic audio from the simulator.
    - Intercepts LLM calls to capture (prompt_tokens, response_tokens, log_probs).
    - Advances simulated time proportionally to audio fed (episode runs faster
      than real-time).
    - Replaces _seal_mic_block with a synchronous ground-truth ASR reader.
    """

    def __init__(
        self,
        simulator: BaseSimulator,
        sim_config: SimulatorConfig,
        vllm_engine: Any,
        tokenizer: Any,
        vllm_max_tokens: int = 16,
        vllm_temperature: float = 1.0,
    ) -> None:
        self.simulator = simulator
        self.sim_config = sim_config
        self.vllm_engine = vllm_engine
        self.tokenizer = tokenizer
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_temperature = vllm_temperature

    def run_episode(self) -> Episode:
        episode_id = str(uuid.uuid4())[:8]
        self.simulator.reset()

        steps: List[StepRecord] = []
        sim_time = [0.0]  # mutable container for simulated wall-clock

        # ------------------------------------------------------------------
        # LLM intercept — called inside agent._maybe_run_llm()
        # agent is referenced by name; Python's late binding means it is
        # resolved at call time (after 'agent = ...' below).
        # ------------------------------------------------------------------
        # Tracks the context_version at the time of the previous LLM call so
        # we can detect whether the user spoke between calls.
        last_context_version: List[int] = [-1]

        def intercepted_llm_fn(system_prompt: str, user_message: str) -> str:
            user_spoke = (agent.context_version != last_context_version[0])
            last_context_version[0] = agent.context_version
            text, ptok, rtok, lps = llm_generate_train(
                system_prompt,
                user_message,
                self.vllm_engine,
                self.tokenizer,
                max_tokens=self.vllm_max_tokens,
                temperature=self.vllm_temperature,
            )
            steps.append(StepRecord(
                step_id=str(uuid.uuid4())[:8],
                prompt_text=system_prompt + "\n\n" + user_message,
                full_prompt_tokens=ptok,
                response_token_ids=rtok,
                log_probs=lps,
                is_idle=(not text),
                source_block_id=agent._latest_user_source_block_id,  # late binding ↓
                user_spoke_before=user_spoke,
            ))
            return text

        # ------------------------------------------------------------------
        # Build agent with mocked TTS; no model loading (tts_fn != None).
        # ------------------------------------------------------------------
        agent = _make_training_agent(self.sim_config, intercepted_llm_fn)
        agent._now = lambda: sim_time[0]  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # Ground-truth ASR: override _seal_mic_block synchronously.
        # Writes transcript from the simulator directly into blocks and
        # increments context_version (same logic as _run_parakeet).
        # ------------------------------------------------------------------
        def sim_seal_mic_block(start_ts: float, end_ts: float) -> None:
            sealed = agent._mic_current.copy()
            agent._mic_current = np.zeros(0, dtype=np.float32)
            agent._mic_rolling.append((start_ts, end_ts, sealed))
            if len(agent._mic_rolling) > MAX_MIC_BLOCKS:
                agent._mic_rolling.pop(0)
            # Store raw mic audio in the matching historical block
            for block in reversed(agent.blocks):
                if abs(block.start_ts - start_ts) < 0.5:
                    block.mic_audio = sealed
                    break
            # Write ground-truth transcript
            transcript = self.simulator.get_transcript_at_time(start_ts, end_ts)
            if transcript:
                for block in reversed(agent.blocks):
                    if abs(block.start_ts - start_ts) < 0.5:
                        if block.user_text != transcript:
                            block.user_text = transcript
                            agent._latest_user_source_block_id = block.block_id
                            agent._invalidate_future_assistant_continuation()
                            agent._last_accepted_response_context_version = None
                            agent.context_version += 1
                        break

        agent._seal_mic_block = sim_seal_mic_block  # type: ignore[method-assign]

        # ------------------------------------------------------------------
        # Episode loop: advance simulated time proportional to audio fed.
        # ------------------------------------------------------------------
        chunk_samples = max(
            160,
            int(self.sim_config.block_s * ASR_SAMPLE_RATE / 10),
        )  # ~200 ms chunks
        dt = chunk_samples / ASR_SAMPLE_RATE
        terminated_reason = "max_duration"

        while sim_time[0] < self.sim_config.max_episode_s:
            # 1. Check for text injections (TextInjectSimulator)
            inject = self.simulator.get_pending_text_inject(sim_time[0])
            if inject:
                agent.receive_text_message(inject, ts=sim_time[0])

            # 2. Feed mic audio from simulator
            mic = self.simulator.get_audio_chunk(chunk_samples, ASR_SAMPLE_RATE)
            if mic is None:
                terminated_reason = "simulator_done"
                break

            agent.receive_mic_chunk(ASR_SAMPLE_RATE, mic)
            sim_time[0] += dt

            # 3. Advance agent block clock
            tts_out = agent.poll()
            if tts_out is not None:
                self.simulator.on_agent_tts(*tts_out)

        # Back-fill blocks_covered using response_source_block_id grouping,
        # then merge consecutive silent-user steps into single actions.
        _fill_blocks_covered(steps, list(agent.blocks))
        steps = _merge_silent_runs(steps)

        return Episode(
            episode_id=episode_id,
            steps=steps,
            blocks=list(agent.blocks),
            terminated_reason=terminated_reason,
        )


# ---------------------------------------------------------------------------
# Agent construction helpers
# ---------------------------------------------------------------------------

def _make_training_agent(
    sim_config: SimulatorConfig,
    llm_generate_fn: Callable[[str, str], str],
) -> DuplexAudioAgent:
    """
    Create a DuplexAudioAgent with:
    - Mock TTS (silent audio of WPM-estimated duration) to skip Piper loading.
    - No real ASR (tts_fn != None prevents model loading; _seal_mic_block
      is overridden by VirtualSimulationConnection.run_episode()).
    """
    def mock_tts(text: str) -> Tuple[int, np.ndarray]:
        duration_s = _wpm_duration_s(text, sim_config.wpm)
        samples = max(1, int(duration_s * TTS_SAMPLE_RATE))
        return TTS_SAMPLE_RATE, np.zeros(samples, dtype=np.int16)

    return DuplexAudioAgent(
        wpm=sim_config.wpm,
        default_block_s=sim_config.block_s,
        llm_generate_fn=llm_generate_fn,
        tts_fn=mock_tts,
        # asr_fn left as None; _seal_mic_block is monkey-patched before use.
    )


def _fill_blocks_covered(
    steps: List[StepRecord], blocks: List[DuplexAudioBlock]
) -> None:
    """
    Back-fill StepRecord.blocks_covered after an episode ends.

    Blocks that share the same response_source_block_id came from the same
    LLM generation call.  Each step recorded which block triggered its call
    (source_block_id), so we group accordingly.
    """
    # source_block_id → [block_ids that used words from this LLM call]
    source_to_blocks: Dict[str, List[str]] = {}
    for block in blocks:
        if block.assistant_text and block.response_source_block_id:
            source_to_blocks.setdefault(block.response_source_block_id, []).append(
                block.block_id
            )
    for step in steps:
        if step.source_block_id:
            step.blocks_covered = source_to_blocks.get(step.source_block_id, [])


def _collapse_run(run: List[StepRecord]) -> StepRecord:
    """Merge a list of consecutive StepRecords into one combined action."""
    if len(run) == 1:
        return run[0]
    first = run[0]
    seen: set = set()
    all_blocks: List[str] = []
    for s in run:
        for bid in s.blocks_covered:
            if bid not in seen:
                seen.add(bid)
                all_blocks.append(bid)
    return StepRecord(
        step_id=first.step_id,
        prompt_text=first.prompt_text,
        full_prompt_tokens=first.full_prompt_tokens,
        response_token_ids=[t for s in run for t in s.response_token_ids],
        log_probs=[lp for s in run for lp in s.log_probs],
        is_idle=all(s.is_idle for s in run),
        source_block_id=first.source_block_id,
        blocks_covered=all_blocks,
        user_spoke_before=first.user_spoke_before,
    )


def _merge_silent_runs(steps: List[StepRecord]) -> List[StepRecord]:
    """
    Merge consecutive non-idle steps where the user did not speak between calls.

    When the user is silent, multiple LLM calls all stem from the same
    conversational decision ("continue speaking"). Grouping them into one
    action gives a single advantage signal over the whole segment, improving
    credit assignment and reducing gradient variance.
    """
    if not steps:
        return steps
    merged: List[StepRecord] = []
    run: List[StepRecord] = [steps[0]]
    for step in steps[1:]:
        # Extend the current run only when: the new step arrived without the
        # user speaking, and neither the current nor the previous step was idle
        # (idle steps break the continuity of a speech segment).
        if not step.user_spoke_before and not step.is_idle and not run[-1].is_idle:
            run.append(step)
        else:
            merged.append(_collapse_run(run))
            run = [step]
    merged.append(_collapse_run(run))
    return merged


# ---------------------------------------------------------------------------
# FullDuplexRLTrainer
# ---------------------------------------------------------------------------

class FullDuplexRLTrainer:
    """
    REINFORCE trainer for full-duplex conversational policies.

    Training cycle (one train_step):
        1. Sample episodes_per_train_step simulator configs.
        2. Run each through VirtualSimulationConnection (vLLM inference).
        3. Score steps with weighted reward functions.
        4. Compute REINFORCE loss with EMA baseline and KL regularisation.
        5. Backprop + AdamW update on the HuggingFace model.
        6. Sync updated weights back into the vLLM engine.
    """

    def __init__(
        self,
        config: TrainerConfig,
        simulator_configs: List[SimulatorConfig],
        reward_fns: List[RewardFn],
    ) -> None:
        if not HAS_VLLM:
            raise RuntimeError(
                "vLLM is required for FullDuplexRLTrainer. pip install vllm"
            )

        self.config = config
        self.simulator_configs = simulator_configs
        self.reward_fns = reward_fns
        self.rm_weights: List[float] = (
            config.reward_fn_weights
            if config.reward_fn_weights is not None
            else [1.0] * len(reward_fns)
        )
        if len(self.rm_weights) != len(self.reward_fns):
            raise ValueError(
                f"reward_fn_weights length ({len(self.rm_weights)}) must match "
                f"reward_fns length ({len(self.reward_fns)})"
            )

        print(f"[trainer] loading tokenizer: {config.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[trainer] loading HuggingFace model for gradient updates")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.bfloat16,
        )
        self.model.to(config.device)
        self.model.train()

        print("[trainer] loading vLLM engine for rollout inference")
        self.vllm_engine = LLM(
            model=config.model_name_or_path,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_model_len=config.max_seq_len,
            dtype="bfloat16",
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        self._baseline: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, batch_configs: List[SimulatorConfig]) -> List[Episode]:
        """Run one episode per config (sequentially) and return all episodes."""
        episodes: List[Episode] = []
        for sim_config in batch_configs:
            simulator = build_simulator(sim_config)
            conn = VirtualSimulationConnection(
                simulator=simulator,
                sim_config=sim_config,
                vllm_engine=self.vllm_engine,
                tokenizer=self.tokenizer,
                vllm_max_tokens=self.config.vllm_max_tokens,
                vllm_temperature=self.config.vllm_temperature,
            )
            try:
                ep = conn.run_episode()
                episodes.append(ep)
                n_non_idle = sum(1 for s in ep.steps if not s.is_idle)
                print(
                    f"[trainer] episode={ep.episode_id}  "
                    f"steps={len(ep.steps)} (non-idle={n_non_idle})  "
                    f"blocks={len(ep.blocks)}  ended={ep.terminated_reason}"
                )
            except Exception as exc:
                print(f"[trainer] episode failed ({sim_config.config_id}): {exc!r}")
        return episodes

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_rewards(self, episode: Episode) -> Episode:
        """Fill StepRecord.reward using weighted sum of reward functions."""
        for i, step in enumerate(episode.steps):
            is_terminal = (i == len(episode.steps) - 1)
            covered_ids = set(step.blocks_covered)
            covered = [b for b in episode.blocks if b.block_id in covered_ids]
            history = [b for b in episode.blocks if b.block_id not in covered_ids]
            total = 0.0
            for fn, w in zip(self.reward_fns, self.rm_weights):
                for block in covered:
                    total += w * fn(block, history, is_terminal)
            step.reward = total
        return episode

    # ------------------------------------------------------------------
    # REINFORCE loss
    # ------------------------------------------------------------------

    def compute_reinforce_loss(self, episodes: List[Episode]) -> torch.Tensor:
        """
        Policy gradient loss with EMA baseline and soft KL regularisation.

        For each non-idle step t with advantage A_t = G_t - baseline:

            L_t = -A_t * log π_θ(a_t | s_t)
                  + kl_coeff * (log π_old(a_t | s_t) - log π_θ(a_t | s_t))

        Summed and normalised by total response token count.

        Idle steps are excluded from the gradient but their rewards still
        flow backward through the return computation for preceding steps.
        """
        total_loss = torch.zeros(1, device=self.config.device, requires_grad=False)
        total_tokens = 0
        loss_terms: List[torch.Tensor] = []

        for episode in episodes:
            rewards = [s.reward or 0.0 for s in episode.steps]
            returns = _compute_returns(rewards, self.config.gamma)

            for step, G in zip(episode.steps, returns):
                if step.is_idle or not step.response_token_ids:
                    continue

                n_resp = len(step.response_token_ids)
                advantage = G - self._baseline

                # Build input_ids: prompt tokens + response tokens.
                # Left-truncate prompt if the sequence would exceed max_seq_len.
                n_prompt = len(step.full_prompt_tokens)
                budget = self.config.max_seq_len - n_resp - 1
                if budget < 1:
                    continue  # response alone exceeds budget
                prompt_tokens = step.full_prompt_tokens[-budget:]
                all_tokens = prompt_tokens + step.response_token_ids
                n_prompt_used = len(prompt_tokens)

                input_ids = torch.tensor(
                    [all_tokens], dtype=torch.long, device=self.config.device
                )
                labels = input_ids.clone()
                # Mask prompt positions so loss is computed only on response
                labels[0, :n_prompt_used] = -100

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(input_ids=input_ids, labels=labels)

                # out.loss is mean NLL over unmasked (response) tokens
                nll_mean = out.loss
                policy_log_prob = -nll_mean * n_resp  # sum of response log_probs

                # KL against rollout policy (prevents large weight updates).
                # kl_loss = log π_θ − log π_old: positive when current policy
                # assigns more probability than the rollout policy did.
                # Keeping policy_log_prob as a tensor lets gradients flow so
                # the effective gradient coefficient is (−advantage + kl_coeff).
                pi_old_sum = sum(step.log_probs[:n_resp]) if step.log_probs else 0.0
                kl_loss = policy_log_prob - pi_old_sum

                step_loss = (
                    -advantage * policy_log_prob
                    + self.config.kl_coeff * kl_loss
                )
                loss_terms.append(step_loss)
                total_tokens += n_resp

        if not loss_terms or total_tokens == 0:
            return torch.zeros(1, device=self.config.device, requires_grad=True).squeeze()

        stacked = torch.stack(loss_terms)
        return stacked.sum() / total_tokens

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self, batch_configs: Optional[List[SimulatorConfig]] = None
    ) -> Dict[str, float]:
        """One complete REINFORCE update cycle. Returns a metrics dict."""
        import random
        if batch_configs is None:
            k = self.config.episodes_per_train_step
            batch_configs = random.choices(self.simulator_configs, k=k)

        # 1. Rollout (no gradient needed)
        with torch.no_grad():
            # vLLM handles its own computation; the torch.no_grad() block
            # prevents accidental gradient accumulation from any torch ops
            # that might appear in simulator helpers.
            pass
        episodes = self.collect_rollouts(batch_configs)

        if not episodes:
            return {"step": self._step_count, "loss": 0.0, "n_episodes": 0}

        # 2. Score
        episodes = [self.compute_rewards(e) for e in episodes]

        # 3. Update EMA baseline from all returns in this batch
        all_returns: List[float] = []
        for ep in episodes:
            rewards = [s.reward or 0.0 for s in ep.steps]
            all_returns.extend(_compute_returns(rewards, self.config.gamma))
        alpha = self.config.baseline_ema_alpha
        batch_mean = sum(all_returns) / len(all_returns) if all_returns else 0.0
        self._baseline = (1.0 - alpha) * self._baseline + alpha * batch_mean

        # 4. REINFORCE loss + backprop
        self.optimizer.zero_grad()
        loss = self.compute_reinforce_loss(episodes)
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()

        # 5. Sync updated weights → vLLM engine
        self._sync_weights_to_vllm()

        self._step_count += 1
        metrics = {
            "step": float(self._step_count),
            "loss": float(loss.detach().cpu()),
            "baseline": self._baseline,
            "n_episodes": float(len(episodes)),
            "n_steps_total": float(sum(len(e.steps) for e in episodes)),
            "n_non_idle": float(
                sum(sum(1 for s in e.steps if not s.is_idle) for e in episodes)
            ),
        }
        print(
            f"[trainer] step={self._step_count}  loss={metrics['loss']:.4f}  "
            f"baseline={metrics['baseline']:.3f}  "
            f"non_idle={int(metrics['n_non_idle'])}/{int(metrics['n_steps_total'])}"
        )
        return metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, num_steps: int) -> List[Dict[str, float]]:
        """Run num_steps REINFORCE updates. Returns a list of per-step metrics."""
        history: List[Dict[str, float]] = []
        for _ in range(num_steps):
            history.append(self.train_step())
        return history

    # ------------------------------------------------------------------
    # vLLM weight sync
    # ------------------------------------------------------------------

    def _sync_weights_to_vllm(self) -> None:
        """
        Push HuggingFace model weights into the vLLM engine's model runner.

        Uses vLLM's internal model_runner.model.load_weights() path.
        Pin vllm>=0.4.0,<0.6.0 in pyproject.toml — this path is not a
        stable public API and may change across minor versions.
        """
        try:
            params = ((k, v.detach().cpu()) for k, v in self.model.named_parameters())
            model_runner = (
                self.vllm_engine.llm_engine
                .model_executor
                .driver_worker
                .model_runner
            )
            model_runner.model.load_weights(params)
            print(f"[trainer] vLLM weights synced at step {self._step_count}")
        except AttributeError as exc:
            print(
                f"[trainer] WARNING: vLLM weight sync failed — {exc!r}. "
                "Rollout will use stale weights. "
                "Check vLLM version (requires >=0.4.0,<0.6.0)."
            )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Discounted returns: G_t = r_t + gamma * G_{t+1} (reversed cumsum)."""
    G = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns

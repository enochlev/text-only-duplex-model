"""rewards.py — built-in reward functions for full-duplex RL training."""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from typing import Callable, List, Optional

import httpx
import numpy as np

from full_duplex import DuplexAudioBlock

import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

COHERENCE_SERVER_URL = f"http://localhost:{_os.getenv('COHERENCE_PORT', '10001')}"

_IDLE_TOKENS: frozenset = frozenset({
    "<idle>", "<|im_end|>", "<|endoftext|>", "</s>", "<eos>",
})


def _is_effectively_idle(text: str) -> bool:
    stripped = text.strip()
    return not stripped or stripped in _IDLE_TOKENS


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




def respond_after_user_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise silence when the user finished speaking more than 1 block ago.

    Lag=1 (respond in the block immediately after user finishes) is acceptable.
    Only fires when the user spoke 2+ blocks back, signalling a missed
    turn-taking opportunity that the model should correct.
    """
    if block.assistant_text or len(history) < 2:
        return 0.0
    user_spoke_2ago = _user_finished_in(history[-2])
    return -0.3 if user_spoke_2ago else 0.0




def coherence_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
    gamma: float = 0.9,
    timeout: float = 5.0,
    server_url: Optional[str] = None,
) -> float:
    """Score the block against the teacher LLM via the coherence reward server.

    Only fires when the current block has text AND the immediately preceding
    block also had text (i.e. the model is mid-monologue). Idle blocks are
    handled by idle_penalty; this function returns 0.0 for them.

    The last non-empty bot block in history becomes last_bot_message (the
    prefix the teacher conditions on). The last user turn in history becomes
    last_user_message.
    """
    proposed = (block.assistant_text or "").strip()
    if _is_effectively_idle(proposed):
        return 0.0

    # Accumulate all consecutive bot blocks that preceded the most recent user
    # turn. This gives the teacher full context on how long the model has been
    # speaking, enabling it to score whether continuing is appropriate.
    last_user = ""
    bot_prefix_parts: list = []
    for b in reversed(history):
        if b.user_text:
            last_user = b.user_text
            break
        if b.assistant_text:
            bot_prefix_parts.insert(0, b.assistant_text)

    prev_bot = " ".join(bot_prefix_parts).strip()
    if not prev_bot and not last_user:
        return 0.0

    payload = {
        "history": [
            {"user": b.user_text or "", "bot": b.assistant_text or ""}
            for b in history
        ],
        "last_user_message": last_user,
        "last_bot_message": prev_bot,
        "proposed_next": proposed,
        "gamma": gamma,
    }

    try:
        url = (server_url or COHERENCE_SERVER_URL) + "/reward"
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return float(resp.json()["reward"])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# VAD clients — call vad_server.py endpoints.
# Both fall back to ASR word-count heuristics when the server is unreachable.
# ---------------------------------------------------------------------------

_VAD_BASE_URL     = f"http://localhost:{_os.getenv('VAD_PORT', '10002')}"
_VAD_COMPLETE_URL = f"{_VAD_BASE_URL}/vad/complete"
_VAD_OVERLAP_URL  = f"{_VAD_BASE_URL}/vad/overlap"
_VAD_TIMEOUT_S    = 1.0   # generous — rewards are computed offline, not real-time
_VAD_RETRY_AFTER  = 20    # skip N calls after a failure, then retry

_complete_fail_count = 0
_overlap_fail_count  = 0


def _post(url: str, payload: bytes, timeout: float) -> Optional[dict]:
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _vad_turn_complete(
    text: str, mic_audio: Optional[np.ndarray] = None
) -> Optional[bool]:
    """True if user utterance is a complete turn, False if not, None on failure.

    Passes both text and audio_b64 so the server can route to whichever backend
    is configured (Namo uses text; smart-turn uses audio_b64).
    """
    global _complete_fail_count
    if not text or not text.strip():
        return None
    if _complete_fail_count >= _VAD_RETRY_AFTER:
        return None
    payload: dict = {"text": text}
    if mic_audio is not None and len(mic_audio) > 0:
        payload["audio_b64"] = base64.b64encode(
            np.clip(mic_audio, -1.0, 1.0).astype(np.float32).tobytes()
        ).decode()
    try:
        data = _post(
            _VAD_COMPLETE_URL,
            json.dumps(payload).encode(),
            _VAD_TIMEOUT_S,
        )
        _complete_fail_count = 0
        return bool(data["complete"])
    except Exception:
        _complete_fail_count += 1
        return None


def _vad_overlap(mic_audio: np.ndarray, tts_audio: np.ndarray) -> Optional[bool]:
    """pyannote OSD: True if user and bot were speaking simultaneously, None on failure."""
    score = _vad_overlap_score(mic_audio, tts_audio)
    if score is None:
        return None
    return score > 0.08


def _vad_overlap_score(mic_audio: np.ndarray, tts_audio: np.ndarray) -> Optional[float]:
    """pyannote OSD: returns overlap_ratio (0.0–1.0) as soft product of top-2 speaker scores.

    High value means both speakers were clearly active simultaneously.
    Near-zero for backchannels or single-speaker audio.
    Returns None on server failure.
    """
    global _overlap_fail_count
    if _overlap_fail_count >= _VAD_RETRY_AFTER:
        return None
    mic_b64 = base64.b64encode(
        np.clip(mic_audio, -1.0, 1.0).astype(np.float32).tobytes()
    ).decode()
    tts_b64 = base64.b64encode(
        np.clip(tts_audio, -1.0, 1.0).astype(np.float32).tobytes()
    ).decode()
    try:
        data = _post(
            _VAD_OVERLAP_URL,
            json.dumps({"mic_b64": mic_b64, "tts_b64": tts_b64, "sample_rate": 16_000}).encode(),
            _VAD_TIMEOUT_S,
        )
        _overlap_fail_count = 0
        return float(data.get("overlap_ratio", 0.0))
    except Exception:
        _overlap_fail_count += 1
        return None


_RMS_SILENCE = 1e-4  # below this RMS on both channels → silence, skip pyannote


def _user_finished_in(block: DuplexAudioBlock) -> bool:
    """True if user's utterance in this block is a complete conversational turn.

    Uses VAD server — passes both text and mic_audio so the server can route to
    whichever backend is configured (Namo uses text; smart-turn uses audio_b64).
    Falls back to ASR word-count heuristic when server is unreachable.
    """
    mic = block.mic_audio if (block.mic_audio is not None and len(block.mic_audio) > 0) else None
    result = _vad_turn_complete(block.user_text or "", mic_audio=mic)
    if result is not None:
        return result
    return len((block.user_text or "").split()) >= 1


# ---------------------------------------------------------------------------
# New audio-VAD reward functions
# ---------------------------------------------------------------------------


def interruption_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
    next_block: Optional[DuplexAudioBlock] = None,
) -> float:
    """Block-level crossover penalty — escalates when both parties speak simultaneously.

    Only fires when BOTH user and bot have text in the current block.
    Does NOT penalise the bot for staying silent.

    Consecutive simultaneous-speech run length:
      1 block  → -0.05  (single overlap, may be unavoidable in full-duplex)
      2 blocks → -0.15
      3 blocks → -0.35
      4+ blocks → -0.65

    Exception: isolated 1–2 word bot outputs (fillers like "right", "uh-huh")
    are exempt when both the preceding and following blocks are completely
    silent (no user or bot text), indicating a genuine backchannel rather than
    a disruptive interruption.
    """
    if not block.assistant_text or not block.user_text:
        return 0.0

    run = 1
    for prev in reversed(history):
        if prev.user_text and prev.assistant_text:
            run += 1
        else:
            break

    if run == 1 and len(block.assistant_text.split()) <= 2:
        prev_block = history[-1] if history else None
        prev_silent = prev_block is None or (not prev_block.user_text and not prev_block.assistant_text)
        next_silent = next_block is None or (not next_block.user_text and not next_block.assistant_text)
        if prev_silent and next_silent:
            return 0.0

    if run == 1:
        return -0.05
    if run == 2:
        return -0.15
    if run == 3:
        return -0.35
    return -0.65


def interruption_penalty_overlap(
    block: DuplexAudioBlock,
    _history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Progressive VAD penalty proportional to the pyannote overlap ratio.

    Trains the model to speak only when the VAD signal is low (user not
    actively holding the floor). Uses the soft product of pyannote's top-2
    per-frame speaker scores, so backchannels ("ya", "ok") produce near-zero
    penalty while genuine floor-takes produce high penalty.

    Only fires when BOTH user AND bot have text AND audio is non-silent.
    Falls back to 0.0 when audio is zeroed (simulation) or server is down.

    Penalty = -overlap_ratio  (range: 0.0 to -1.0)
    """
    if not block.assistant_text or not block.user_text:
        return 0.0

    mic = block.mic_audio
    tts = block.tts_audio
    if mic is None or len(mic) == 0 or tts is None or len(tts) == 0:
        return 0.0

    mic_f32 = mic.astype(np.float32)
    tts_f32 = tts.astype(np.float32)
    if tts.dtype == np.int16:
        tts_f32 = tts_f32 / 32768.0

    if (np.sqrt(np.mean(mic_f32 ** 2)) <= _RMS_SILENCE
            and np.sqrt(np.mean(tts_f32 ** 2)) <= _RMS_SILENCE):
        return 0.0

    ratio = _vad_overlap_score(mic_f32, tts_f32)
    if ratio is None:
        return 0.0

    return -ratio


def silence_too_long_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise sustained silence after the user has finished their turn.

    Walks back through consecutive bot-silent blocks to find the most recent
    user finish, then measures lag from there. Lag=1 (first silent block after
    user finishes) is acceptable; penalty starts at lag=2.

    Escalates:
      lag=2 → -0.10
      lag=3 → -0.25
      lag=4+ → -0.50

    Register twice with different weights for two independently-tunable tiers.
    """
    if block.assistant_text or not history:
        return 0.0

    blocks_back = 0
    user_finish_at: Optional[int] = None

    for prev in reversed(history):
        if prev.assistant_text:
            break
        blocks_back += 1
        if user_finish_at is None and _user_finished_in(prev):
            user_finish_at = blocks_back

    if user_finish_at is None:
        return 0.0

    lag = user_finish_at  # 1 = user finished last block, this is first silent block
    if lag <= 1:
        return 0.0
    if lag == 2:
        return -0.10
    if lag == 3:
        return -0.25
    return -0.50
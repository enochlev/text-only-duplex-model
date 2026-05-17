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

COHERENCE_SERVER_URL = "http://localhost:8001"

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
    """Smooth penalty for deviation from the target response length.

    Uses a quadratic bowl centred on TARGET_WORDS so the gradient always
    points toward the sweet spot rather than a hard cliff.  Max penalty
    is capped at -0.3 so it doesn't dominate turn-taking signals.
    """
    if not block.assistant_text:
        return 0.0
    TARGET_WORDS = 5
    words = len(block.assistant_text.split())
    penalty = -0.005 * (words - TARGET_WORDS) ** 2
    return max(penalty, -0.3)


def respond_after_user_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise silence only when the user just finished speaking.

    Intentionally stacks with idle_penalty: idle_penalty is a constant floor
    that discourages silence everywhere; this adds an extra −0.3 only when
    silence is a missed turn-taking opportunity (user spoke in the last 2 blocks).
    The combined −0.4 signal is stronger than either alone, which is the intent.
    """
    if block.assistant_text:
        return 0.0
    recent = history[-2:]
    user_just_spoke = any(
        len((b.user_text or "").split()) >= 1 for b in recent
    )
    return -0.3 if user_just_spoke else 0.0


def overlap_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise talking over the user in consecutive blocks.

    A 'user block' is one where the user spoke more than 2 words.
    If the bot has assistant_text while the user is actively speaking,
    that is overlap. The penalty escalates with consecutive overlap length:
      1 block  → -0.05  (hard to avoid in full-duplex, mild signal)
      2 blocks → -0.20  (clearly talking over)
      3+ blocks → -0.50  (sustained interruption, strong signal)
    """
    def _user_active(b: DuplexAudioBlock) -> bool:
        return len((b.user_text or "").split()) > 2

    if not block.assistant_text or not _user_active(block):
        return 0.0

    prior_run = 0
    for prev in reversed(history):
        if prev.assistant_text and _user_active(prev):
            prior_run += 1
        else:
            break

    run_length = prior_run + 1
    if run_length == 1:
        return -0.05
    elif run_length == 2:
        return -0.20
    else:
        return -0.50


def first_sentence_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise responses that continue past the first sentence boundary.

    Trains the model to emit one crisp sentence per block and then yield
    the floor. No penalty when the first punctuation mark is at (or near)
    the end of the response.
    """
    text = (block.assistant_text or "").strip()
    if not text:
        return 0.0
    m = re.search(r'[.!?]', text)
    if m and m.end() < len(text) - 1:
        return -0.2
    return 0.0


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
    if not proposed:
        return 0.0

    # only applies while the user is silent (model is continuing its own turn)
    prev_bot = next(
        (b.assistant_text for b in reversed(history) if b.assistant_text),
        None,
    )
    if not prev_bot:
        return 0.0

    last_user = next(
        (b.user_text for b in reversed(history) if b.user_text),
        "",
    )

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
# VAD client — calls the smart-turn server (vad_server.py) on localhost.
# Falls back to ASR word-count heuristic when the server is unreachable.
# ---------------------------------------------------------------------------

_VAD_URL = "http://localhost:8765/vad"
_VAD_TIMEOUT_S = 0.5  # 500 ms — generous for CPU inference of 32 MB ONNX
_VAD_SILENCE_RMS = 1e-4  # below this → treat audio as silence, skip VAD
_vad_fail_count = 0
_VAD_RETRY_AFTER = 20   # re-probe server after this many consecutive failures


def _vad_classify(mic_audio: np.ndarray) -> Optional[bool]:
    """Return True if user's turn is complete, False if still talking, None on failure.

    Returns None (→ ASR fallback) when:
      - audio RMS is below silence threshold (ScriptTTSSource zero-fills mic_audio)
      - VAD server is unreachable (retried every _VAD_RETRY_AFTER failures)
    """
    global _vad_fail_count

    # Skip VAD on near-silence — avoids false "complete" on zero-filled simulation audio
    if float(np.sqrt(np.mean(mic_audio.astype(np.float32) ** 2))) < _VAD_SILENCE_RMS:
        return None

    if _vad_fail_count >= _VAD_RETRY_AFTER:
        return None

    audio_b64 = base64.b64encode(
        np.clip(mic_audio, -1.0, 1.0).astype(np.float32).tobytes()
    ).decode()
    payload = json.dumps({"audio_b64": audio_b64, "sample_rate": 16_000}).encode()

    try:
        req = urllib.request.Request(
            _VAD_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_VAD_TIMEOUT_S) as resp:
            data = json.loads(resp.read())
        _vad_fail_count = 0
        return bool(data["complete"])
    except Exception:
        _vad_fail_count += 1
        return None


def _user_speaking(block: DuplexAudioBlock) -> bool:
    """True if user was mid-turn during this block (VAD preferred, ASR fallback)."""
    if block.mic_audio is not None and len(block.mic_audio) > 0:
        result = _vad_classify(block.mic_audio)
        if result is not None:
            return not result  # complete=True means done, so not-complete = still speaking
    return len((block.user_text or "").split()) > 2


def _user_finished_in(block: DuplexAudioBlock) -> bool:
    """True if user completed their turn during this block (VAD preferred, ASR fallback)."""
    if block.mic_audio is not None and len(block.mic_audio) > 0:
        result = _vad_classify(block.mic_audio)
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
) -> float:
    """Penalise speaking while the user's turn is still in progress.

    Uses smart-turn audio VAD when available; falls back to ASR word count.
    Escalates with consecutive interruption run length:
      1 block  → -0.15
      2 blocks → -0.35
      3+ blocks → -0.65
    """
    if not block.assistant_text or not _user_speaking(block):
        return 0.0

    prior_run = 0
    for prev in reversed(history):
        if prev.assistant_text and len((prev.user_text or "").split()) > 2:
            prior_run += 1
        else:
            break

    run_length = prior_run + 1
    if run_length == 1:
        return -0.15
    if run_length == 2:
        return -0.35
    return -0.65


def silence_too_long_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise sustained silence after the user has finished their turn.

    Checks whether the user's turn completed in the most recent history block.
    Uses smart-turn audio VAD when available; falls back to ASR heuristic.
    Escalates with consecutive silent blocks:
      1 block  → -0.10
      2 blocks → -0.25
      3+ blocks → -0.50

    Add this function twice to reward_fns with different weights to produce
    two independently-tunable penalty tiers.
    """
    if block.assistant_text or not history:
        return 0.0

    if not _user_finished_in(history[-1]):
        return 0.0

    prior_run = 0
    for prev in reversed(history):
        if prev.assistant_text:
            break
        prior_run += 1

    run_length = prior_run + 1
    if run_length == 1:
        return -0.10
    if run_length == 2:
        return -0.25
    return -0.50
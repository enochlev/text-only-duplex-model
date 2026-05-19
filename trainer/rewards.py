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
    """Penalise silence when the user finished speaking more than 1 block ago.

    Lag=1 (respond in the block immediately after user finishes) is acceptable.
    Only fires when the user spoke 2+ blocks back, signalling a missed
    turn-taking opportunity that the model should correct.
    """
    if block.assistant_text or len(history) < 2:
        return 0.0
    user_spoke_2ago = len((history[-2].user_text or "").split()) >= 1
    return -0.3 if user_spoke_2ago else 0.0


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
    if _is_effectively_idle(proposed):
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
# VAD clients — call vad_server.py endpoints.
# Both fall back to ASR word-count heuristics when the server is unreachable.
# ---------------------------------------------------------------------------

_VAD_COMPLETE_URL = "http://localhost:10002/vad/complete"
_VAD_OVERLAP_URL  = "http://localhost:10002/vad/overlap"
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
        return bool(data["overlap"])
    except Exception:
        _overlap_fail_count += 1
        return None


_RMS_SILENCE = 1e-4  # below this RMS on both channels → silence, skip pyannote


def _user_speaking(block: DuplexAudioBlock) -> bool:
    """True if user and bot were speaking simultaneously (overlap detected).

    Uses pyannote OSD when both channels are non-silent (production audio).
    Falls back to ASR word-count heuristic in simulation (zero mic/tts audio)
    and when the VAD server is unreachable.
    """
    mic = block.mic_audio
    tts = block.tts_audio
    if mic is not None and len(mic) > 0 and tts is not None and len(tts) > 0:
        tts_f32 = tts.astype(np.float32)
        if tts.dtype == np.int16:
            tts_f32 = tts_f32 / 32768.0
        mic_f32 = mic.astype(np.float32)
        # Silence check — simulation fills both arrays with zeros; skip pyannote
        # so the ASR word-count fallback below still fires during training.
        if (np.sqrt(np.mean(mic_f32 ** 2)) > _RMS_SILENCE
                or np.sqrt(np.mean(tts_f32 ** 2)) > _RMS_SILENCE):
            result = _vad_overlap(mic_f32, tts_f32)
            if result is not None:
                return result
    # ASR fallback: user is interrupting only if they have words AND haven't
    # finished their turn.  Namo is text-based so this works in simulation.
    # Without the turn-complete check, complete short sentences (e.g. "My
    # laptop keeps freezing.") would falsely trigger interruption_penalty.
    words = (block.user_text or "").split()
    if len(words) <= 2:
        return False
    return not _user_finished_in(block)


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
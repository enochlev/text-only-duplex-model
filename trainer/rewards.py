"""rewards.py — built-in reward functions for full-duplex RL training."""

from __future__ import annotations

import base64
import json
import traceback
import urllib.request
from typing import Callable, List, Optional

import numpy as np

from full_duplex import DuplexAudioBlock

import os as _os
from dotenv import load_dotenv as _load_dotenv
_load_dotenv()


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


def _blocks_since_user_finished(history: List[DuplexAudioBlock]) -> Optional[int]:
    """Blocks of consecutive bot-silence since the most recent user turn-complete.

    Returns None if the user hasn't finished or the bot already responded.
    Returns 1 if the user finished in the immediately preceding block, etc.
    """
    for lag, b in enumerate(reversed(history), start=1):
        if b.assistant_text:
            return None  # bot already spoke — silence is intentional
        if _user_finished_in(b):
            return lag
    return None


def respond_after_user_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise bot silence after the user finishes a turn.

    lag=1 (first silent block): -1.0
    lag=2: -2.0
    lag=3+: -3.0

    No penalty while the user is still speaking or the bot already responded.
    """
    if block.user_text or block.assistant_text:
        return 0.0
    lag = _blocks_since_user_finished(history)
    if lag is None:
        return 0.0
    if lag == 1:
        return -1.0
    if lag == 2:
        return -2.0
    return -3.0


def interruption_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise consecutive simultaneous-speech blocks.

    First overlap is free (no causal visibility of the user's speech).
    Escalates with run length:
      run=2: -0.5   run=3: -1.0   run=4+: -2.0
    """
    if not block.assistant_text or not block.user_text:
        return 0.0

    run = 1
    for prev in reversed(history):
        if prev.user_text and prev.assistant_text:
            run += 1
        else:
            break

    if run == 1:
        return 0.0
    if run == 2:
        return -0.5
    if run == 3:
        return -1.0
    return -2.0


def interruption_penalty_overlap(
    block: DuplexAudioBlock,
    _history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Progressive VAD penalty proportional to the pyannote overlap ratio.

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


# ---------------------------------------------------------------------------
# Backchannel loop penalty
# ---------------------------------------------------------------------------

_BACKCHANNELS: frozenset = frozenset({
    "ya", "yeah", "yep", "yup",
    "okay", "ok",
    "right", "alright",
    "sure",
    "mm", "hmm", "uh-huh", "mhm", "uhh",
    "i know", "i see", "i understand",
    "got it", "that's right", "exactly", "of course",
})


def _normalize_bot_text(text: str) -> str:
    import re as _re
    text = _re.sub(r"<[^>]+>", "", text)  # strip angle-bracket tags
    return text.lower().strip().rstrip(".,!?").strip()


def backchannel_loop_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    _is_terminal: bool,
) -> float:
    """Penalise consecutive backchannel-only bot blocks.

    A single backchannel is natural and free. Penalty escalates at -0.5
    per additional consecutive backchannel block:
      run=1: 0.0   run=2: -0.5   run=3: -1.0   run=4: -1.5 ...
    """
    if not block.assistant_text:
        return 0.0

    norm = _normalize_bot_text(block.assistant_text)
    if norm not in _BACKCHANNELS:
        return 0.0

    run = 1
    for b in reversed(history):
        if not b.assistant_text:
            break
        if _normalize_bot_text(b.assistant_text) in _BACKCHANNELS:
            run += 1
        else:
            break

    if run <= 1:
        return 0.0
    return -1.0 * (run - 1)


# ---------------------------------------------------------------------------
# VAD clients
# ---------------------------------------------------------------------------

_VAD_BASE_URL     = f"http://localhost:{_os.getenv('VAD_PORT', '10002')}"
_VAD_COMPLETE_URL = f"{_VAD_BASE_URL}/vad/complete"
_VAD_OVERLAP_URL  = f"{_VAD_BASE_URL}/vad/overlap"
_VAD_TIMEOUT_S    = 20.0
_VAD_RETRY_AFTER  = 20

_complete_fail_count = 0
_overlap_fail_count  = 0
_RMS_SILENCE         = 1e-4


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
    """True if user utterance is a complete turn, False if not, None on failure."""
    global _complete_fail_count
    if not text or not text.strip():
        return None
    if _complete_fail_count >= _VAD_RETRY_AFTER:
        _complete_fail_count = 0
    payload: dict = {"text": text}
    if mic_audio is not None and len(mic_audio) > 0:
        payload["audio_b64"] = base64.b64encode(
            np.clip(mic_audio, -1.0, 1.0).astype(np.float32).tobytes()
        ).decode()
    try:
        data = _post(_VAD_COMPLETE_URL, json.dumps(payload).encode(), _VAD_TIMEOUT_S)
        _complete_fail_count = 0
        return bool(data["complete"])
    except Exception:
        if _complete_fail_count == 0:
            print("[VAD /complete] server error (further failures silenced):")
            traceback.print_exc()
        _complete_fail_count += 1
        return None


def _vad_overlap_score(mic_audio: np.ndarray, tts_audio: np.ndarray) -> Optional[float]:
    """pyannote OSD: returns overlap_ratio (0.0–1.0). None on server failure."""
    global _overlap_fail_count
    if _overlap_fail_count >= _VAD_RETRY_AFTER:
        _overlap_fail_count = 0
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
        if _overlap_fail_count == 0:
            print("[VAD /overlap] server error (further failures silenced):")
            traceback.print_exc()
        _overlap_fail_count += 1
        return None


def _user_finished_in(block: DuplexAudioBlock) -> bool:
    """True if the user's utterance in this block is a complete conversational turn."""
    mic = block.mic_audio if (block.mic_audio is not None and len(block.mic_audio) > 0) else None
    result = _vad_turn_complete(block.user_text or "", mic_audio=mic)
    if result is not None:
        return result
    return len((block.user_text or "").split()) >= 1


# ---------------------------------------------------------------------------
# Startup server health check
# ---------------------------------------------------------------------------

def check_rm_servers(vad_url: Optional[str] = None) -> None:
    """Assert the VAD server is reachable before training starts."""
    v_url = (vad_url or _VAD_BASE_URL).rstrip("/")
    errors: list = []

    try:
        data = _post(
            f"{v_url}/vad/complete",
            json.dumps({"text": "hello there"}).encode(),
            5.0,
        )
        if "complete" not in data:
            errors.append(f"VAD /vad/complete missing 'complete' key: {data}")
    except Exception as exc:
        errors.append(f"VAD server /vad/complete unreachable at {v_url}  ({exc})")

    try:
        silent = np.zeros(160, dtype=np.float32)
        b64 = base64.b64encode(silent.tobytes()).decode()
        data = _post(
            f"{v_url}/vad/overlap",
            json.dumps({"mic_b64": b64, "tts_b64": b64, "sample_rate": 16_000}).encode(),
            5.0,
        )
        if "overlap_ratio" not in data:
            errors.append(f"VAD /vad/overlap missing 'overlap_ratio' key: {data}")
    except Exception as exc:
        errors.append(f"VAD server /vad/overlap unreachable at {v_url}  ({exc})")

    if errors:
        bullets = "\n".join(f"  • {e}" for e in errors)
        raise RuntimeError(
            f"\n[check_rm] {len(errors)} server(s) failed pre-flight check:\n{bullets}\n\n"
            f"Start vad_server.py (port {_VAD_BASE_URL.rsplit(':', 1)[-1]}) before training."
        )

    print(f"[check_rm] VAD server OK  (vad={v_url})")

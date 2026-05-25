"""rewards.py — built-in reward functions for full-duplex RL training."""

from __future__ import annotations

import base64
import json
import traceback
import urllib.request
from typing import Callable, List, Optional

import numpy as np

from full_duplex import DuplexAudioBlock

import re as _re
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


def _silent_blocks_since_user_spoke(history: List[DuplexAudioBlock]) -> Optional[int]:
    """Count trailing consecutive silent blocks (no user_text) since user last spoke.

    Pure block-level: no VAD calls.
    Returns None if bot already responded (silence is intentional) or user never spoke.
    Returns 0 if user spoke in the immediately preceding block (history[-1].user_text set).
    Returns 1 if one silent block elapsed since user last spoke, etc.
    """
    silent = 0
    for b in reversed(history):
        if b.assistant_text and not b.user_text:
            return None  # bot gave a clean response — subsequent silence is intentional
        if b.user_text:
            return silent  # found where user last spoke (overlap blocks count as user speech)
        silent += 1
    return None  # no user speech found in history


# ---------------------------------------------------------------------------
# Block-level turn-taking rewards (no VAD calls)
# ---------------------------------------------------------------------------

def block_silence_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise bot silence after user stops speaking. Pure block-level, no VAD.

    Looks back through history for how many consecutive silent blocks have
    elapsed since the user last spoke.

    silent=0 (user spoke last block, this is first silence): -1.0
    silent=1: -2.0   silent>=2: -3.0
    """
    if block.user_text or block.assistant_text:
        return 0.0
    silent = _silent_blocks_since_user_spoke(history)
    if silent is None:
        return 0.0
    if silent == 0:
        return -1.0
    if silent == 1:
        return -2.0
    return -3.0


def block_interruption_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Penalise bot speaking while user is also speaking. Pure block-level, no VAD.

    First overlap is free ONLY when the user was NOT already speaking in the
    source block (history[-1]).  If the user was already speaking there, the bot
    had causal visibility and chose to interrupt; no free pass.  Escalates with
    run length:
      true-interrupt (run=1, source had user): -0.5
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

    source_had_user = bool(history) and bool(history[-1].user_text)
    if run == 1 and not source_had_user:
        return 0.0
    if run <= 2:
        return -0.5
    if run == 3:
        return -1.0
    return -2.0


def block_idle_reward(
    block: DuplexAudioBlock,
    _history: List[DuplexAudioBlock],
    _is_terminal: bool,
) -> float:
    """Reward bot silence while user is speaking. Pure block-level, no VAD.

    Mid-sentence determination (lookahead) is handled by the caller (_idle_rm1_reward)
    which has access to the full episode. This function is the raw +0.5 signal;
    the caller gates when it fires.
    """
    if block.assistant_text:
        return 0.0
    if not block.user_text:
        return 0.0
    return +0.5


def timely_response_reward(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    is_terminal: bool,
) -> float:
    """Reward bot for responding promptly after the user finishes speaking.

    lag=0: source block had user speech AND user finished there → +1.0 (ideal)
    lag=1: one silent block since user stopped                   → +0.75
    lag=2: two silent blocks since user stopped                  → +0.5

    Guard: if source block had user speech but user was mid-sentence (no
    terminal punctuation, VAD says not complete), this is a true interruption
    and RM2 already penalises it — return 0 to avoid double-counting.
    Also skip covered blocks that are themselves overlap blocks (user still
    speaking there) since RM2 covers those too.
    """
    if not block.assistant_text:
        return 0.0
    if not history:
        return 0.0
    # Covered block itself is an overlap (user still speaking) — let RM2 handle.
    if block.user_text:
        return 0.0
    src = history[-1]
    if src.user_text:
        # Source had user speech. Only reward if the user actually finished there;
        # mid-sentence source = true interruption = RM2's territory.
        if not _user_finished_in(src):
            return 0.0
        # User finished at source block → ideal timing (lag=0).
        return +1.0
    # Source block was silent — count how many blocks since user last spoke.
    lag = _silent_blocks_since_user_spoke(history)
    if lag is None or lag > 2:
        return 0.0
    if lag == 1:
        return +0.75
    return +0.5


# ---------------------------------------------------------------------------
# VAD-based rewards (require running VAD server)
# ---------------------------------------------------------------------------

def vad_overlap_penalty(
    block: DuplexAudioBlock,
    _history: List[DuplexAudioBlock],
    _is_terminal: bool,
) -> float:
    """Progressive penalty proportional to pyannote audio overlap ratio.

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
    # affirmations
    "ya", "yeah", "yep", "yup", "yes", "yes please", "yes sure", "yes of course",
    "okay", "ok",
    "right", "alright",
    "sure", "sure let's", "sure thing",
    "absolutely", "certainly", "definitely", "totally",
    "exactly", "that's right", "thats right",
    "of course", "of course next", "of course sure",
    # fillers / minimal acknowledgements
    "mm", "hmm", "uh-huh", "mhm", "uhh", "uhm",
    "wow", "really", "oh", "ah", "oh i see",
    # comprehension signals
    "i know", "i see", "i understand", "i hear you", "i hear that",
    "got it", "makes sense", "fair enough", "i agree",
    "noted", "understood", "copy that",
    # hollow topic acknowledgements
    "interesting", "interesting topic", "interesting point", "interesting question",
    "great", "great question", "great point",
    "good", "good question", "good point",
    "nice", "fair point", "that's fair", "thats fair",
    "sounds good", "sounds great", "sounds right", "sounds about right",
    "that makes sense", "makes total sense",
    # short clarifying fragments (non-variable forms)
    "which ones", "which one", "like what", "how so", "such as",
    # conversation steering (hollow)
    "let's continue", "let's continue there", "let's go",
    "let's start", "let's", "let's see",
    "go on", "go ahead", "continue", "proceed",
    "please continue", "please go on", "tell me more",
})

# Prefix-matched backchannels — covers variable-length clarifying questions
# like "What kind of style?" / "What kind of engaging tone?" / "What type of X?"
_BACKCHANNEL_PREFIXES: frozenset = frozenset({
    "what kind of",
    "what type of",
    "which kind of",
    "which type of",
})


def _normalize_bot_text(text: str) -> str:
    text = _re.sub(r"<[^>]+>", "", text)   # strip angle-bracket tags
    text = _re.sub(r"[,;:!?]", "", text)    # strip internal punctuation (keep hyphens/apostrophes)
    return text.lower().strip().rstrip(".,!?").strip()


def _is_backchannel(norm: str) -> bool:
    """True if norm is a backchannel — exact match or prefix match.

    Also strips a single leading backchannel word (e.g. "sure what kind of X"
    after comma-removal) before testing prefixes, so the model can't escape
    detection by prepending a filler like "sure," or "right,".
    """
    if norm in _BACKCHANNELS:
        return True
    if any(norm.startswith(p) for p in _BACKCHANNEL_PREFIXES):
        return True
    # strip one leading single-word backchannel and re-check prefixes
    first, _, rest = norm.partition(" ")
    if first in _BACKCHANNELS and rest:
        if any(rest.startswith(p) for p in _BACKCHANNEL_PREFIXES):
            return True
    return False


def backchannel_loop_penalty(
    block: DuplexAudioBlock,
    history: List[DuplexAudioBlock],
    _is_terminal: bool,
) -> float:
    """Penalise backchannel-only bot responses.

    A single backchannel is free ONLY when the user was mid-sentence at the
    source block (history[-1].user_text set with no terminal punctuation).
    If the user already finished their turn (source block silent, or text ends
    a sentence), even run=1 costs -0.5 — a backchannel is not a real answer.

    Escalation:
      mid-sentence, run=1: 0.0   run=2: -0.5   run=3: -1.0  ...
      post-turn,    run=1: -0.5  run=2: -1.0   run=3: -1.5  ...
    """
    if not block.assistant_text:
        return 0.0

    norm = _normalize_bot_text(block.assistant_text)
    if not _is_backchannel(norm):
        return 0.0

    run = 1
    for b in reversed(history):
        if not b.assistant_text:
            break
        if _is_backchannel(_normalize_bot_text(b.assistant_text)):
            run += 1
        else:
            break

    # Determine if the user was mid-sentence at the source block.
    src = history[-1] if history else None
    _TERM = frozenset(".!?…")
    user_mid_sentence = bool(
        src and src.user_text
        and src.user_text.strip()
        and src.user_text.strip()[-1] not in _TERM
    )

    if run <= 1 and user_mid_sentence:
        return 0.0  # single backchannel during user speech is a natural filler
    # run=1 post-turn OR any run>1: escalate starting at -0.5
    return -0.5 * run


# ---------------------------------------------------------------------------
# Junk output penalty
# ---------------------------------------------------------------------------

# HTML tags, lone angle-bracket fragments, and Qwen function-call tokens that
# the model generates when it "wants to be idle" but outputs text instead of EOS.
# These are not TTS-speakable and should be penalised harder than a real
# interruption so the model has a clear incentive to prefer EOS over junk.
_JUNK_RE = _re.compile(
    r'<[^>]{0,40}>'          # any HTML/XML tag   e.g. <i>idle</i>, <span>, <img>
    r'|^[\s<>{}\[\]|]+$'     # line that is ONLY punctuation / brackets
)


def junk_output_penalty(
    block: DuplexAudioBlock,
    _history: List[DuplexAudioBlock],
    _is_terminal: bool,
) -> float:
    """Extra penalty when the model outputs non-TTS-valid junk instead of speech.

    The model occasionally generates HTML-like tokens (<i>idle</i>, <iidle>,
    <ul>, <img>, <span class="idle">, <Funcion>) when it wants to stay silent
    but doesn't produce EOS.  These receive the same RM2 interruption penalty
    as real speech, giving the model no reason to prefer genuine responses over
    garbage.  This RM adds an additional penalty to break that tie.
    """
    text = block.assistant_text
    if not text:
        return 0.0
    if _JUNK_RE.search(text):
        return -1.0
    return 0.0


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


_TERMINAL_PUNCT = frozenset(".!?…")


def _text_turn_complete(text: str) -> bool:
    """Heuristic: text ends with terminal punctuation and has at least 2 words.

    Minimum is 2 (not 1) so single-word fillers like "Yeah?" don't trigger it,
    but short complete phrases like "they preferred?" or "Is it?" do.
    """
    stripped = text.strip()
    return bool(stripped) and stripped[-1] in _TERMINAL_PUNCT and len(stripped.split()) >= 2


def _user_finished_in(block: DuplexAudioBlock) -> bool:
    """True if the user's utterance in this block is a complete conversational turn.

    Terminal-punctuation heuristic overrides VAD when the text clearly ends a sentence.
    VAD is consulted for ambiguous cases (no terminal punctuation).
    """
    text = block.user_text or ""
    if not text.strip():
        return False
    # Clear sentence-ending punctuation: trust the text signal — VAD is unreliable here.
    if _text_turn_complete(text):
        return True
    mic = block.mic_audio if (block.mic_audio is not None and len(block.mic_audio) > 0) else None
    result = _vad_turn_complete(text, mic_audio=mic)
    if result is not None:
        return result
    # Fallback: any text without punctuation is treated as mid-sentence.
    return False


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

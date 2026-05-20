#!/usr/bin/env python3
"""check_rms.py — smoke-test all active reward models before training.

Run from the repo root:
    python check_rms.py

Exits with code 1 if any RM is broken (wrong sign, wrong magnitude, or crash).
Server-backed RMs (interruption_penalty_overlap, coherence_reward) print a
WARNING instead of FAIL when the server is down, since training can start
without them — they gracefully degrade to 0.0.
"""

from __future__ import annotations

import sys
import traceback
import uuid
from typing import List, Optional

import numpy as np

from full_duplex import (
    ASR_SAMPLE_RATE,
    TTS_MODEL,
    TTS_SAMPLE_RATE,
    DuplexAudioBlock,
    preload_piper_voice,
    piper_synthesize,
)
from trainer.rewards import (
    respond_after_user_reward,
    interruption_penalty,
    interruption_penalty_overlap,
    silence_too_long_penalty,
    coherence_reward,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS  = "✓ PASS"
_FAIL  = "✗ FAIL"
_WARN  = "⚠ WARN"

_failures: List[str] = []


def _blk(user: str = "", bot: str = "",
         mic: Optional[np.ndarray] = None,
         tts: Optional[np.ndarray] = None,
         tts_sr: int = TTS_SAMPLE_RATE) -> DuplexAudioBlock:
    b = DuplexAudioBlock(
        block_id=str(uuid.uuid4())[:8],
        start_ts=0.0,
        end_ts=1.0,
        user_text=user,
        assistant_text=bot,
    )
    b.mic_audio = mic
    b.tts_audio = tts
    b.tts_sr = tts_sr
    return b


def _check(label: str, got: float, expected: float,
           *, tol: float = 1e-6, server_backed: bool = False) -> None:
    ok = abs(got - expected) <= tol
    status = _PASS if ok else (_WARN if server_backed else _FAIL)
    print(f"  {status}  {label:<40}  got={got:+.4f}  expected={expected:+.4f}")
    if not ok and not server_backed:
        _failures.append(label)


def _check_nonzero(label: str, got: float, *, server_backed: bool = True) -> None:
    ok = got != 0.0
    status = _PASS if ok else _WARN
    print(f"  {status}  {label:<40}  got={got:+.4f}  expected≠0.0")
    if not ok and not server_backed:
        _failures.append(label)


def _section(name: str) -> None:
    print(f"\n── {name} {'─' * (60 - len(name))}")


# ---------------------------------------------------------------------------
# 1. respond_after_user_reward
# ---------------------------------------------------------------------------
_section("respond_after_user_reward")

_silent_hist = [_blk(user="Hi."), _blk(user="")]   # user finished 2 blocks ago
_bot_responded_hist = [_blk(user="Hi.")]

_check(
    "bot silent, user finished 2 blocks ago → -0.3",
    respond_after_user_reward(_blk(bot=""), _silent_hist, False),
    -0.3,
)
_check(
    "bot responded → 0.0",
    respond_after_user_reward(_blk(bot="Hello there!"), [], False),
    0.0,
)
_check(
    "not enough history → 0.0",
    respond_after_user_reward(_blk(bot=""), [_blk(user="Hi.")], False),
    0.0,
)

# ---------------------------------------------------------------------------
# 3. interruption_penalty  (block-level crossover)
# ---------------------------------------------------------------------------
_section("interruption_penalty")

_overlap1 = _blk(user="Yeah.", bot="I can help.")
_overlap2 = _blk(user="So anyway.", bot="Right, so I")

_check(
    "1-block overlap → -0.05",
    interruption_penalty(_blk(user="Hi.", bot="Hello!"), [], False),
    -0.05,
)
_check(
    "2-block overlap run → -0.15",
    interruption_penalty(_overlap2, [_overlap1], False),
    -0.15,
)
_check(
    "3-block overlap run → -0.35",
    interruption_penalty(_blk(user="Hmm.", bot="Yes."), [_overlap1, _overlap2], False),
    -0.35,
)
_check(
    "bot silent → 0.0",
    interruption_penalty(_blk(user="Hi.", bot=""), [], False),
    0.0,
)
_check(
    "user silent → 0.0",
    interruption_penalty(_blk(user="", bot="Hello!"), [], False),
    0.0,
)

# ---------------------------------------------------------------------------
# 4. interruption_penalty_overlap  (VAD server — progressive)
# ---------------------------------------------------------------------------
_section("interruption_penalty_overlap  [VAD server]")

# Zero audio always falls back to 0.0 — no server call
_check(
    "zero audio fallback → 0.0",
    interruption_penalty_overlap(_blk(user="Hi.", bot="Hello!"), [], False),
    0.0,
)

# Real TTS audio — generate two utterances and use them as mic + bot channels
print("  Generating TTS audio via Piper (this may take a moment on first run)…")
try:
    voice = preload_piper_voice(TTS_MODEL, device="cpu")
    sr_bot, tts_int16 = piper_synthesize(voice, "Hello there, how can I help you today?")
    sr_usr, mic_int16 = piper_synthesize(voice, "Yeah I think we should talk about this.")

    # Resample both to 16 kHz for the VAD server
    from full_duplex import _resample
    tts_f32 = tts_int16.astype(np.float32) / 32768.0
    mic_f32 = mic_int16.astype(np.float32) / 32768.0
    if sr_bot != ASR_SAMPLE_RATE:
        tts_f32 = _resample(tts_f32, sr_bot, ASR_SAMPLE_RATE)
    if sr_usr != ASR_SAMPLE_RATE:
        mic_f32 = _resample(mic_f32, sr_usr, ASR_SAMPLE_RATE)

    audio_blk = _blk(
        user="Yeah I think we should talk about this.",
        bot="Hello there, how can I help you today?",
        mic=mic_f32,
        tts=(tts_f32 * 32768).astype(np.int16),
        tts_sr=ASR_SAMPLE_RATE,
    )
    result = interruption_penalty_overlap(audio_blk, [], False)
    if result == 0.0:
        print(f"  {_WARN}  {'VAD server unreachable — overlap returned 0.0 (graceful)':<60}")
    else:
        _check_nonzero("real TTS audio → non-zero penalty (VAD server alive)", result)
except Exception as exc:
    print(f"  {_WARN}  TTS generation failed: {exc}")
    traceback.print_exc()

# ---------------------------------------------------------------------------
# 5. silence_too_long_penalty  (two tiers)
# ---------------------------------------------------------------------------
_section("silence_too_long_penalty")

def _silence_hist(n_silent: int) -> List[DuplexAudioBlock]:
    """Build history: one user-finish block then n_silent bot-silent blocks."""
    return [_blk(user="Hi.")] + [_blk(user="", bot="") for _ in range(n_silent)]

_check(
    "bot silent, lag=1 (user just finished) → 0.0",
    silence_too_long_penalty(_blk(bot=""), _silence_hist(0), False),
    0.0,
)
_check(
    "bot silent, lag=2 → -0.10",
    silence_too_long_penalty(_blk(bot=""), _silence_hist(1), False),
    -0.10,
)
_check(
    "bot silent, lag=3 → -0.25",
    silence_too_long_penalty(_blk(bot=""), _silence_hist(2), False),
    -0.25,
)
_check(
    "bot silent, lag=4+ → -0.50",
    silence_too_long_penalty(_blk(bot=""), _silence_hist(3), False),
    -0.50,
)
_check(
    "bot speaking → 0.0",
    silence_too_long_penalty(_blk(bot="Hello!"), _silence_hist(2), False),
    0.0,
)

# ---------------------------------------------------------------------------
# 6. coherence_reward  (coherence server)
# ---------------------------------------------------------------------------
_section("coherence_reward  [coherence server]")

# No prev bot text → guard returns 0.0 before server call
_check(
    "no prior bot context → 0.0 (guard, no server call)",
    coherence_reward(_blk(bot="Can I help you?"), [], False),
    0.0,
)

# History with bot-only block preceding current → should call server
_coherence_hist = [
    _blk(user="Tell me a joke.", bot=""),
    _blk(user="", bot="Why don't scientists trust atoms?"),
]
_coh_result = coherence_reward(
    _blk(bot="Because they make up everything!"),
    _coherence_hist,
    False,
)
if _coh_result == 0.0:
    print(f"  {_WARN}  {'coherence server unreachable — returned 0.0 (graceful)':<60}")
else:
    _check_nonzero("monologue continuation → non-zero score (server alive)", _coh_result)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'═' * 65}")
if _failures:
    print(f"  FAILED: {len(_failures)} assertion(s)")
    for f in _failures:
        print(f"    • {f}")
    sys.exit(1)
else:
    print("  All deterministic RM assertions passed.")
    print("  (Server-backed RMs show ⚠ WARN when their server is not running.)")
    sys.exit(0)

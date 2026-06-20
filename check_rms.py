#!/usr/bin/env python3
"""check_rms.py — smoke-test all active reward models before training.

Run from the repo root:
    python check_rms.py

Exits with code 1 if any RM is broken (wrong sign, wrong magnitude, or crash).
Server-backed RMs (interruption_penalty_overlap) print a
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
    preload_kokoro_voice,
    kokoro_synthesize,
)
from trainer.rewards import (
    backchannel_loop_penalty,
    respond_after_user_reward,
    interruption_penalty,
    interruption_penalty_overlap,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS  = "✓ PASS"
_FAIL  = "✗ FAIL"
_WARN  = "⚠ WARN"
_ZERO_TOL = 1e-6

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
    status = _PASS if ok else _FAIL
    if not ok and server_backed:
        status = _WARN
    print(f"  {status}  {label:<40}  got={got:+.4f}  expected={expected:+.4f}")
    if not ok and not server_backed:
        _failures.append(label)


def _check_nonzero(label: str, got: float, *, server_backed: bool = True,
                   tol: float = _ZERO_TOL) -> None:
    ok = abs(got) > tol
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

_silent_hist = [_blk(user="Hi."), _blk(user="", bot="")]   # user finished 2 blocks ago

_check(
    "bot silent, user finished 1 block ago → -1.0",
    respond_after_user_reward(_blk(bot=""), [_blk(user="Hi.")], False),
    -1.0,
)
_check(
    "bot silent, user finished 2 blocks ago → -2.0",
    respond_after_user_reward(_blk(bot=""), _silent_hist, False),
    -2.0,
)
_check(
    "bot responded → 0.0",
    respond_after_user_reward(_blk(bot="Hello there!"), [_blk(user="Hi.")], False),
    0.0,
)
_check(
    "no prior user turn → 0.0",
    respond_after_user_reward(_blk(bot=""), [], False),
    0.0,
)

# ---------------------------------------------------------------------------
# 3. interruption_penalty  (block-level crossover)
# ---------------------------------------------------------------------------
_section("interruption_penalty")

_overlap1 = _blk(user="Yeah.", bot="I can help.")
_overlap2 = _blk(user="So anyway.", bot="Right, so I")

_check(
    "1-block overlap → 0.0",
    interruption_penalty(_blk(user="Hi.", bot="Hello!"), [], False),
    0.0,
)
_check(
    "2-block overlap run → -0.5",
    interruption_penalty(_overlap2, [_overlap1], False),
    -0.5,
)
_check(
    "3-block overlap run → -1.0",
    interruption_penalty(_blk(user="Hmm.", bot="Yes."), [_overlap1, _overlap2], False),
    -1.0,
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
# 4. backchannel_loop_penalty
# ---------------------------------------------------------------------------
_section("backchannel_loop_penalty")

_check(
    "first backchannel → 0.0",
    backchannel_loop_penalty(_blk(bot="Yeah"), [], False),
    0.0,
)
_check(
    "second backchannel run → -1.0",
    backchannel_loop_penalty(_blk(bot="OK"), [_blk(bot="yeah")], False),
    -1.0,
)
_check(
    "tag noise still matched → -1.0",
    backchannel_loop_penalty(_blk(bot="OK<AI>"), [_blk(bot="yeah")], False),
    -1.0,
)
_check(
    "leading filler prefix still matched → -1.0",
    backchannel_loop_penalty(_blk(bot="Sure, what kind of style?"), [_blk(bot="yeah")], False),
    -1.0,
)

# ---------------------------------------------------------------------------
# 5. interruption_penalty_overlap  (VAD server — progressive)
# ---------------------------------------------------------------------------
_section("interruption_penalty_overlap  [VAD server]")

# Zero audio always falls back to 0.0 — no server call
_check(
    "zero audio fallback → 0.0",
    interruption_penalty_overlap(_blk(user="Hi.", bot="Hello!"), [], False),
    0.0,
)

# Real TTS audio — generate two utterances and use them as mic + bot channels
print("  Generating TTS audio via Kokoro (this may take a moment on first run)…")
try:
    voice = preload_kokoro_voice(TTS_MODEL, device="cpu")
    sr_bot, tts_int16 = kokoro_synthesize(voice, "Hello there, how can I help you today?")
    sr_usr, mic_int16 = kokoro_synthesize(voice, "Yeah I think we should talk about this.")

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
    if abs(result) <= _ZERO_TOL:
        print(f"  {_WARN}  {'VAD server unreachable — overlap returned 0.0 (graceful)':<60}")
    else:
        _check_nonzero("real TTS audio → non-zero penalty (VAD server alive)", result)
except Exception as exc:
    print(f"  {_WARN}  TTS generation failed: {exc}")
    traceback.print_exc()

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

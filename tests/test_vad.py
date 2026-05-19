#!/usr/bin/env python3
"""test_vad.py — smoke-test both VAD server endpoints.

/vad/complete  (Namo):
  For 10 phrases, send the full text (expect complete=True) and the
  first half of words (expect complete=False).

/vad/overlap  (pyannote OSD):
  Synthesise TTS audio for 5 phrase pairs. Mix user+bot audio to check
  overlap detection.  Also verify silence+silence → no overlap.

Usage:
    # terminal 1
    python vad_server.py

    # terminal 2
    python test_vad.py
"""

import base64
import json
import sys
import urllib.request
import os
import dotenv

dotenv.load_dotenv()  # for COHERENCE_PORT, VLLM_PORT

import numpy as np

PHRASES = [
    "The quick brown fox jumped over the lazy dog.",
    "I was just thinking about what you said earlier today.",
    "Can you help me understand how this whole system works?",
    "It's a really nice day outside, perfect for a walk.",
    "I completely forgot to send that email before the meeting.",
    "What time does the next train leave from the central station?",
    "She finished the report just before the deadline last night.",
    "Do you think we should reschedule the call to tomorrow morning?",
    "Actually, I changed my mind about the whole thing.",
    "Could you explain that one more time, I didn't quite follow?",
]

SERVER = f"http://localhost:{os.getenv('VAD_PORT', '10002')}"
SR = 16_000


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _post(path: str, body: dict, timeout: float = 5.0) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        SERVER + path,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def check_server() -> None:
    try:
        _post("/vad/complete", {"text": "hello"}, timeout=2.0)
    except ConnectionRefusedError:
        print("ERROR: VAD server not reachable. Start it first:\n  python vad_server.py")
        sys.exit(1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# TTS helper (for overlap tests)
# ---------------------------------------------------------------------------

def synthesize(text: str) -> np.ndarray:
    """Return float32 PCM at 16 kHz using the project's Piper voice."""
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from full_duplex import preload_piper_voice, TTS_MODEL

    voice = preload_piper_voice(tts_model=TTS_MODEL, device="cpu")

    if hasattr(voice, "synthesize"):
        chunks = [getattr(c, "audio_int16_bytes", b"") for c in voice.synthesize(text)]
        int16 = np.frombuffer(b"".join(chunks), dtype=np.int16)
    elif hasattr(voice, "synthesize_stream_raw"):
        int16 = np.frombuffer(b"".join(voice.synthesize_stream_raw(text)), dtype=np.int16)
    else:
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            int16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    return int16.astype(np.float32) / 32768.0


def to_b64(audio: np.ndarray) -> str:
    return base64.b64encode(
        np.clip(audio, -1.0, 1.0).astype(np.float32).tobytes()
    ).decode()


# ---------------------------------------------------------------------------
# Test 1: /vad/complete
# ---------------------------------------------------------------------------

def test_complete() -> None:
    print("\n" + "=" * 80)
    print("TEST 1: /vad/complete  (Namo Turn Detector — text input)")
    print("=" * 80)
    print(f"{'#':<3}  {'full→complete':>14}  {'half→incomplete':>16}  phrase")
    print("-" * 80)

    full_ok = half_ok = 0
    for i, phrase in enumerate(PHRASES, 1):
        words = phrase.split()
        half_text = " ".join(words[: len(words) // 2])

        r_full = _post("/vad/complete", {"text": phrase})
        r_half = _post("/vad/complete", {"text": half_text})

        fok = r_full["complete"]          # expect True
        hok = not r_half["complete"]      # expect False
        full_ok += fok
        half_ok += hok

        print(
            f"{i:<3}  "
            f"{str(r_full['complete']):>7} {('✓' if fok else '✗')} {r_full['confidence']:.3f}   "
            f"{str(r_half['complete']):>7} {('✓' if hok else '✗')} {r_half['confidence']:.3f}   "
            f"{phrase[:55]}"
        )

    print("-" * 80)
    print(f"Full phrase → complete : {full_ok}/{len(PHRASES)}")
    print(f"Half phrase → incomplete: {half_ok}/{len(PHRASES)}")


# ---------------------------------------------------------------------------
# Test 2: /vad/overlap
# ---------------------------------------------------------------------------

OVERLAP_PAIRS = [
    ("What time is it?",                  "It's three o'clock in the afternoon."),
    ("Can you help me with something?",   "Of course, what do you need?"),
    ("I was thinking we could go out.",   "That sounds like a great idea."),
    ("Do you remember what we talked?",   "Yes, I remember everything clearly."),
    ("Tell me more about your day.",      "Well, it started pretty early."),
]


def test_overlap() -> None:
    print("\n" + "=" * 80)
    print("TEST 2: /vad/overlap  (pyannote OSD — mic+tts audio)")
    print("=" * 80)

    # Silence baseline
    sil = to_b64(np.zeros(SR * 2, np.float32))   # 2 s silence
    r = _post("/vad/overlap", {"mic_b64": sil, "tts_b64": sil}, timeout=15)
    ok = not r["overlap"]
    print(f"  silence + silence → overlap={r['overlap']}  ratio={r['overlap_ratio']}  {'✓' if ok else '✗'}")

    print(f"\n  Overlap detection on mixed user+bot TTS audio:")
    overlap_detected = 0

    for i, (user_text, bot_text) in enumerate(OVERLAP_PAIRS, 1):
        user_audio = synthesize(user_text)
        bot_audio  = synthesize(bot_text)
        r = _post(
            "/vad/overlap",
            {"mic_b64": to_b64(user_audio), "tts_b64": to_b64(bot_audio)},
            timeout=15,
        )
        overlap_detected += r["overlap"]
        print(
            f"  {i}  overlap={r['overlap']}  ratio={r['overlap_ratio']:.3f}"
            f"  user={user_text[:35]!r}"
        )

    print(f"\n  Pairs with overlap detected: {overlap_detected}/{len(OVERLAP_PAIRS)}")
    print("  (expect most True — both real voices mixed = simultaneous speech)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    check_server()
    test_complete()
    test_overlap()
    print()


if __name__ == "__main__":
    main()

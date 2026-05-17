#!/usr/bin/env python3
"""test_vad.py — smoke-test the VAD server against Piper TTS audio.

For 10 phrases, synthesizes full audio via Piper, then queries the VAD server
twice per phrase:
  - mid  : first half of the audio (user still talking)
  - full : complete audio          (user finished)

Expected: mid → complete=False, full → complete=True

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

import numpy as np

# ---------------------------------------------------------------------------
# Phrases — 5 short statements + 5 questions, varied rhythm
# ---------------------------------------------------------------------------
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

VAD_URL = "http://localhost:8765/vad"
SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Piper TTS
# ---------------------------------------------------------------------------

def synthesize(text: str) -> np.ndarray:
    """Return float32 PCM at 16 kHz for *text* using the project's Piper voice."""
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from full_duplex import preload_piper_voice, TTS_MODEL

    voice = preload_piper_voice(tts_model=TTS_MODEL, device="cpu")

    # Try each Piper API variant (mirrors full_duplex._synthesize_piper_audio)
    if hasattr(voice, "synthesize"):
        chunks = []
        for chunk in voice.synthesize(text):
            raw = getattr(chunk, "audio_int16_bytes", b"")
            if raw:
                chunks.append(raw)
        int16 = np.frombuffer(b"".join(chunks), dtype=np.int16)

    elif hasattr(voice, "synthesize_stream_raw"):
        raw = b"".join(voice.synthesize_stream_raw(text))
        int16 = np.frombuffer(raw, dtype=np.int16)

    elif hasattr(voice, "synthesize_wav"):
        import io, wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            voice.synthesize_wav(text, wf)
        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        int16 = np.frombuffer(raw, dtype=np.int16)

    else:
        raise RuntimeError("Unknown Piper voice API — no synthesize* method found")

    return int16.astype(np.float32) / 32768.0  # normalise to [-1, 1]


# ---------------------------------------------------------------------------
# VAD client
# ---------------------------------------------------------------------------

def query_vad(audio: np.ndarray) -> dict:
    audio_b64 = base64.b64encode(
        np.clip(audio, -1.0, 1.0).astype(np.float32).tobytes()
    ).decode()
    payload = json.dumps({"audio_b64": audio_b64, "sample_rate": SAMPLE_RATE}).encode()
    req = urllib.request.Request(
        VAD_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_server() -> None:
    try:
        urllib.request.urlopen(VAD_URL.replace("/vad", "/docs"), timeout=1.0)
    except Exception:
        # /docs might 404 but a connection error means server is down
        try:
            query_vad(np.zeros(1600, dtype=np.float32))
        except ConnectionRefusedError:
            print("ERROR: VAD server not reachable. Start it first:\n  python vad_server.py")
            sys.exit(1)
        except Exception:
            pass  # any other error (bad model etc.) is fine at this stage


def main() -> None:
    check_server()

    print(f"{'#':<3}  {'complete@mid':>13}  {'prob@mid':>9}  {'complete@full':>14}  {'prob@full':>10}  phrase")
    print("-" * 100)

    correct_mid  = 0
    correct_full = 0

    for i, phrase in enumerate(PHRASES, 1):
        audio = synthesize(phrase)
        mid   = audio[: len(audio) // 2]
        full  = audio

        r_mid  = query_vad(mid)
        r_full = query_vad(full)

        mid_ok  = not r_mid["complete"]   # expect False (still talking)
        full_ok = r_full["complete"]       # expect True  (done)
        correct_mid  += mid_ok
        correct_full += full_ok

        mid_flag  = "✓" if mid_ok  else "✗"
        full_flag = "✓" if full_ok else "✗"

        print(
            f"{i:<3}  "
            f"{str(r_mid['complete']):>12} {mid_flag}  "
            f"{r_mid['probability']:>9.3f}  "
            f"{str(r_full['complete']):>13} {full_flag}  "
            f"{r_full['probability']:>10.3f}  "
            f"{phrase[:60]}"
        )

    print("-" * 100)
    print(
        f"Mid-sentence accuracy : {correct_mid}/{len(PHRASES)}  "
        f"({'expect all False — incomplete turn'})\n"
        f"Full-audio accuracy   : {correct_full}/{len(PHRASES)}  "
        f"({'expect all True  — complete turn'})"
    )


if __name__ == "__main__":
    main()

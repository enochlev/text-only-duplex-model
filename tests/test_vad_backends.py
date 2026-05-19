"""tests/test_vad_backends.py — pytest suite for both VAD /vad/complete backends.

Requires the VAD server running:
    python vad_server.py

Skips automatically when the server is not reachable.

Smart-turn requires audio_b64; we synthesize it via Piper TTS if available,
otherwise we fall back to white noise (tests plumbing, not model accuracy).
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Optional

import numpy as np
import pytest

SERVER = "http://localhost:8765"
SR = 16_000

# Full conversational phrases — expect complete=True
PHRASES_COMPLETE = [
    "The quick brown fox jumped over the lazy dog.",
    "I was just thinking about what you said earlier today.",
    "It's a really nice day outside, perfect for a walk.",
    "I completely forgot to send that email before the meeting.",
    "Do you think we should reschedule the call to tomorrow morning?",
]

# First half of the same phrases — expect complete=False (clearly mid-sentence)
PHRASES_INCOMPLETE = [
    "The quick brown fox jumped",
    "I was just thinking about what",
    "It's a really nice day",
    "I completely forgot to send that",
    "Do you think we should reschedule",
]


# ---------------------------------------------------------------------------
# Fixtures & helpers
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


@pytest.fixture(scope="session", autouse=True)
def require_server():
    """Skip the entire module if the VAD server isn't running."""
    try:
        _post("/vad/complete/namo", {"text": "hello"}, timeout=2.0)
    except (ConnectionRefusedError, urllib.error.URLError):
        pytest.skip("VAD server not reachable — start it with: python vad_server.py")


def _try_synthesize(text: str) -> Optional[np.ndarray]:
    """Return float32 PCM via Piper TTS, or None if Piper not available."""
    try:
        import io
        import os
        import sys
        import wave

        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from full_duplex import TTS_MODEL, preload_piper_voice

        voice = preload_piper_voice(tts_model=TTS_MODEL, device="cpu")
        if hasattr(voice, "synthesize_stream_raw"):
            raw = b"".join(voice.synthesize_stream_raw(text))
            int16 = np.frombuffer(raw, dtype=np.int16)
        else:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                voice.synthesize_wav(text, wf)
            buf.seek(0)
            with wave.open(buf, "rb") as wf:
                int16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        return int16.astype(np.float32) / 32768.0
    except Exception:
        return None


def _noise_audio(seconds: float = 2.0) -> np.ndarray:
    """White noise as fallback audio for plumbing tests."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(SR * seconds)) * 0.1).astype(np.float32)


def _to_b64(audio: np.ndarray) -> str:
    return base64.b64encode(
        np.clip(audio, -1.0, 1.0).astype(np.float32).tobytes()
    ).decode()


# ---------------------------------------------------------------------------
# Namo backend tests  (/vad/complete/namo)
# ---------------------------------------------------------------------------


class TestNamo:
    @pytest.mark.parametrize("phrase", PHRASES_COMPLETE)
    def test_complete_phrase(self, phrase: str):
        r = _post("/vad/complete/namo", {"text": phrase})
        assert r["backend"] == "namo"
        assert r["complete"] is True, f"Expected complete=True for {phrase!r}, got {r}"

    @pytest.mark.parametrize("fragment", PHRASES_INCOMPLETE)
    def test_incomplete_fragment(self, fragment: str):
        r = _post("/vad/complete/namo", {"text": fragment})
        assert r["backend"] == "namo"
        assert r["complete"] is False, f"Expected complete=False for {fragment!r}, got {r}"

    def test_response_schema(self):
        r = _post("/vad/complete/namo", {"text": "Hello, how are you today?"})
        assert "complete" in r
        assert "confidence" in r
        assert "backend" in r
        assert isinstance(r["complete"], bool)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_overall_accuracy(self):
        """At least 8/10 classifications correct across all phrases."""
        correct = 0
        total = len(PHRASES_COMPLETE) + len(PHRASES_INCOMPLETE)
        for phrase in PHRASES_COMPLETE:
            r = _post("/vad/complete/namo", {"text": phrase})
            correct += int(r["complete"] is True)
        for fragment in PHRASES_INCOMPLETE:
            r = _post("/vad/complete/namo", {"text": fragment})
            correct += int(r["complete"] is False)
        assert correct >= 8, f"Namo accuracy {correct}/{total} — below threshold"


# ---------------------------------------------------------------------------
# Smart-turn backend tests  (/vad/complete/smart-turn)
# ---------------------------------------------------------------------------


class TestSmartTurn:
    @pytest.fixture(scope="class")
    def audio_complete(self):
        """Synthesized audio for a complete phrase (Piper or white noise)."""
        audio = _try_synthesize("The quick brown fox jumped over the lazy dog.")
        if audio is None:
            audio = _noise_audio(2.0)
        return _to_b64(audio)

    @pytest.fixture(scope="class")
    def audio_incomplete(self):
        """Shorter audio — less than 1 second of noise (simulates a cut-off utterance)."""
        audio = _try_synthesize("The quick")
        if audio is None:
            audio = _noise_audio(0.5)
        return _to_b64(audio)

    def test_response_schema(self, audio_complete: str):
        r = _post("/vad/complete/smart-turn", {"text": "", "audio_b64": audio_complete})
        assert "complete" in r
        assert "confidence" in r
        assert "backend" in r
        assert r["backend"] == "smart-turn"
        assert isinstance(r["complete"], bool)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_requires_audio(self):
        """Server returns 400 when audio_b64 is missing."""
        import urllib.error
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post("/vad/complete/smart-turn", {"text": "hello"})
        assert exc_info.value.code == 400

    def test_complete_phrase_with_piper(self):
        """If Piper TTS is available, test accuracy on a complete phrase."""
        audio = _try_synthesize("I completely forgot to send that email before the meeting.")
        if audio is None:
            pytest.skip("Piper TTS not available — skipping model accuracy test")
        r = _post("/vad/complete/smart-turn", {"text": "", "audio_b64": _to_b64(audio)})
        assert r["backend"] == "smart-turn"
        assert r["complete"] is True, f"Expected complete=True, got {r}"

    def test_incomplete_phrase_with_piper(self):
        """If Piper TTS is available, test accuracy on a clearly mid-sentence fragment."""
        audio = _try_synthesize("I completely forgot to send")
        if audio is None:
            pytest.skip("Piper TTS not available — skipping model accuracy test")
        r = _post("/vad/complete/smart-turn", {"text": "", "audio_b64": _to_b64(audio)})
        assert r["backend"] == "smart-turn"
        assert r["complete"] is False, f"Expected complete=False, got {r}"


# ---------------------------------------------------------------------------
# /vad/complete routing — env-var backend selection
# ---------------------------------------------------------------------------


class TestRoutingEndpoint:
    def test_returns_valid_response(self):
        """The /vad/complete endpoint should respond regardless of backend."""
        r = _post("/vad/complete", {"text": "What time is it?"})
        assert "complete" in r
        assert "backend" in r

    def test_backend_field_present(self):
        r = _post("/vad/complete", {"text": "Hello there."})
        assert r["backend"] in ("namo", "smart-turn")


# ---------------------------------------------------------------------------
# Side-by-side comparison (printed, not asserted)
# ---------------------------------------------------------------------------


def test_compare_backends_on_same_phrases(capsys):
    """Print Namo vs smart-turn classifications for the same text phrases.

    Smart-turn is audio-based and will use white noise here, so its text-derived
    accuracy is not meaningful.  This test always passes — it's a diagnostic table.
    """
    print("\n\nNamo vs smart-turn (text phrases, smart-turn gets noise audio):")
    print(f"{'phrase':<45}  {'namo':>8}  {'smart-turn':>10}")
    print("-" * 70)

    for phrase in PHRASES_COMPLETE + PHRASES_INCOMPLETE:
        r_namo = _post("/vad/complete/namo", {"text": phrase})
        audio = _try_synthesize(phrase) if len(phrase) > 10 else _noise_audio(0.5)
        if audio is None:
            audio = _noise_audio(2.0)
        r_st = _post(
            "/vad/complete/smart-turn",
            {"text": phrase, "audio_b64": _to_b64(audio)},
        )
        namo_lbl = "DONE" if r_namo["complete"] else "more"
        st_lbl   = "DONE" if r_st["complete"] else "more"
        print(f"{phrase[:44]:<45}  {namo_lbl:>8}  {st_lbl:>10}")

    print()

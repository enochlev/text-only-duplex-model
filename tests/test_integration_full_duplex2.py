"""
Integration tests for DuplexAudioAgent using the real Parakeet ASR model.

Requires NeMo and the parakeet-tdt-0.6b-v2 checkpoint to be available.
Skipped automatically when NeMo is not installed.

Run with:
    pytest tests/test_integration_full_duplex2.py -v
"""

import math

import numpy as np
import pytest

nemo_asr = pytest.importorskip("nemo.collections.asr", reason="NeMo not installed")

from full_duplex2 import MIC_SAMPLE_RATE, TTS_SAMPLE_RATE, DuplexAudioAgent, _asr_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_real_agent() -> DuplexAudioAgent:
    return DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda t: (TTS_SAMPLE_RATE, np.zeros(int(2.0 * TTS_SAMPLE_RATE), dtype=np.int16)),
    )


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * MIC_SAMPLE_RATE), dtype=np.float32)


def _tone(seconds: float, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * MIC_SAMPLE_RATE), endpoint=False)
    return (0.5 * np.sin(2 * math.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

def test_model_loaded_at_import():
    """_asr_model is not None — loaded when the module was imported."""
    assert _asr_model is not None


def test_get_asr_model_returns_singleton():
    """Two agents share the same model object."""
    a = _make_real_agent()
    b = _make_real_agent()
    assert a._get_asr_model() is b._get_asr_model()


# ---------------------------------------------------------------------------
# _run_parakeet with real audio
# ---------------------------------------------------------------------------

def test_run_parakeet_silent_no_crash():
    """Silent audio should not raise — no words expected."""
    agent = _make_real_agent()
    agent._run_parakeet([(1000.0, 1002.0, _silence(2.0))])


def test_run_parakeet_tone_no_crash():
    """Pure-tone audio runs through ASR without error."""
    agent = _make_real_agent()
    agent._run_parakeet([(1000.0, 1002.0, _tone(2.0))])


def test_run_parakeet_empty_rolling():
    """Empty rolling list returns immediately without error."""
    agent = _make_real_agent()
    agent._run_parakeet([])


def test_run_parakeet_multiple_blocks():
    """Multi-block rolling buffer processed without error."""
    agent = _make_real_agent()
    rolling = [
        (1000.0, 1002.0, _silence(2.0)),
        (1002.0, 1004.0, _tone(2.0)),
        (1004.0, 1006.0, _silence(2.0)),
    ]
    agent._run_parakeet(rolling)


# ---------------------------------------------------------------------------
# receive_mic_chunk → _seal_mic_block → _run_parakeet (async)
# ---------------------------------------------------------------------------

def test_receive_mic_chunk_and_seal():
    """Streaming mic audio accumulates then triggers real ASR when a block is sealed."""
    agent = _make_real_agent()
    agent._frozen_time = 1000.0
    agent._now = lambda: agent._frozen_time

    chunk = _silence(0.25)
    for _ in range(4):
        agent.receive_mic_chunk(MIC_SAMPLE_RATE, chunk)

    agent._next_block_ts = 0.0
    agent.poll()
    agent._executor.shutdown(wait=True)

    assert len(agent._mic_rolling) == 1


def test_asr_window_state_after_poll():
    """After a block + ASR completes, get_asr_window_state() returns a list."""
    agent = _make_real_agent()
    agent._frozen_time = 2000.0
    agent._now = lambda: agent._frozen_time

    agent.receive_mic_chunk(MIC_SAMPLE_RATE, _silence(2.0))
    agent._next_block_ts = 0.0
    agent.poll()
    agent._executor.shutdown(wait=True)

    assert isinstance(agent.get_asr_window_state(), list)

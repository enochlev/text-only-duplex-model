"""Unit tests for the monotonic-commit ASR path (_run_parakeet_monotonic).

No GPU / real Parakeet: a scripted fake model returns canned word timestamps so we can
drive drift/re-segmentation deterministically. Proves the two invariants that make the
monotonic commit worth the refactor:
  1. A block whose audio has settled past the commit cursor is FROZEN — later ASR passes
     that rewrite that region (timestamp drift, re-segmentation) cannot change it, so they
     cannot trigger a spurious context flush ("cut off mid-talking").
  2. A genuinely new trailing word (real continued speech) still updates the provisional
     tail and still flushes the in-flight response, exactly like the legacy path.
"""
import numpy as np
import pytest

import full_duplex
from full_duplex import TTS_SAMPLE_RATE, ASR_SAMPLE_RATE, DuplexAudioAgent, DuplexAudioBlock


class _FakeHyp:
    def __init__(self, text, words):
        self.text = text
        self.timestamp = {"word": [{"word": w, "start": s, "end": e} for (w, s, e) in words]}


class _FakeAsrModel:
    """Returns a scripted (text, words) per call, ignoring the audio it is handed."""

    def __init__(self, scripts):
        self._scripts = list(scripts)
        self._i = 0

    def transcribe(self, paths, timestamps=False, verbose=False):
        script = self._scripts[min(self._i, len(self._scripts) - 1)]
        self._i += 1
        text, words = script
        return [_FakeHyp(text, words)]


def _agent():
    agent = DuplexAudioAgent(
        wpm=150,
        default_block_s=2.0,
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda t: (TTS_SAMPLE_RATE, np.zeros(int(2.0 * TTS_SAMPLE_RATE), dtype=np.int16)),
    )
    agent.blocks = [
        DuplexAudioBlock("b0", 0.0, 2.0),
        DuplexAudioBlock("b1", 2.0, 4.0),
        DuplexAudioBlock("b2", 4.0, 6.0),
        DuplexAudioBlock("b3", 6.0, 8.0),
    ]
    agent._asr_commit_cursor_ts = 0.0
    return agent


def _rolling():
    a = np.ones(int(2.0 * ASR_SAMPLE_RATE), dtype=np.float32)
    return [(0.0, 2.0, a), (2.0, 4.0, a), (4.0, 6.0, a), (6.0, 8.0, a)]


def test_monotonic_freezes_settled_blocks_and_flushes_only_on_new_speech(monkeypatch):
    agent = _agent()
    # window_end = 8s, _ASR_RIGHT_CONTEXT_S = 2 -> settle_boundary = 6s.
    # Blocks b0/b1/b2 (end <= 6) freeze after pass 1; b3 (end = 8) stays provisional.
    pass1 = ("hello world how are",
             [("hello", 0.1, 1.0), ("world", 2.1, 3.0), ("how", 4.1, 5.0), ("are", 6.1, 7.0)])
    # pass 2: settled region drifts (must be IGNORED) + a genuine new trailing word "you"
    pass2 = ("hallo word hao are you",
             [("hallo", 0.1, 1.0), ("word", 2.1, 3.0), ("hao", 4.1, 5.0),
              ("are", 6.1, 7.0), ("you", 7.1, 7.8)])
    # pass 3: settled region drifts again, NO new trailing word -> must not flush
    pass3 = ("bonjour wrd huh are you",
             [("bonjour", 0.1, 1.0), ("wrd", 2.1, 3.0), ("huh", 4.1, 5.0),
              ("are", 6.1, 7.0), ("you", 7.1, 7.8)])
    fake = _FakeAsrModel([pass1, pass2, pass3])  # one instance so its call cursor advances
    monkeypatch.setattr(agent, "_get_asr_model", lambda: fake)

    agent._run_parakeet_monotonic(_rolling())
    assert [b.user_text for b in agent.blocks] == ["hello", "world", "how", "are"]
    assert agent._asr_commit_cursor_ts == pytest.approx(6.0)
    v1 = agent.context_version
    assert v1 >= 1  # first speech flushes

    agent._run_parakeet_monotonic(_rolling())
    # settled blocks frozen despite the drifted transcription; provisional tail gains "you"
    assert [b.user_text for b in agent.blocks[:3]] == ["hello", "world", "how"]
    assert agent.blocks[3].user_text == "are you"
    v2 = agent.context_version
    assert v2 == v1 + 1  # genuine new trailing word flushes

    agent._run_parakeet_monotonic(_rolling())
    # pure re-segmentation of settled audio, no new content -> frozen + NO spurious flush
    assert [b.user_text for b in agent.blocks[:3]] == ["hello", "world", "how"]
    assert agent.blocks[3].user_text == "are you"
    assert agent.context_version == v2


def test_monotonic_cursor_only_advances(monkeypatch):
    """The commit cursor must never rewind, even if a later window somehow reports earlier."""
    agent = _agent()
    agent._asr_commit_cursor_ts = 5.0
    p = ("a b", [("a", 0.1, 1.0), ("b", 6.1, 7.0)])
    monkeypatch.setattr(agent, "_get_asr_model", lambda: _FakeAsrModel([p]))
    agent._run_parakeet_monotonic(_rolling())
    assert agent._asr_commit_cursor_ts >= 5.0

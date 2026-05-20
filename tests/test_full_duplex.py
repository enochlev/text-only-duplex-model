"""
Tests for DuplexAudioAgent (full_duplex.py).

All tests use mock TTS/ASR — no API keys, GPU, or real audio required.
Timing-sensitive tests use frozen _now() for determinism.
"""

import math
import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import full_duplex

from full_duplex import (
    DEFAULT_BLOCK_S,
    MAX_AUDIO_QUEUE_S,
    MAX_HISTORY_S,
    TTS_SAMPLE_RATE,
    DuplexAudioAgent,
    DuplexAudioBlock,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_agent(
    llm_fn=None,
    start_time: float = 1000.0,
    wpm: int = 150,
    block_s: float = 2.0,
    tts_duration: float = 2.0,
) -> DuplexAudioAgent:
    """Agent with mock TTS, frozen clock, no real ASR."""
    if llm_fn is None:
        llm_fn = lambda *_: ""

    def mock_tts(text):
        samples = int(tts_duration * TTS_SAMPLE_RATE)
        return TTS_SAMPLE_RATE, np.zeros(samples, dtype=np.int16)

    agent = DuplexAudioAgent(
        wpm=wpm,
        default_block_s=block_s,
        llm_generate_fn=llm_fn,
        tts_fn=mock_tts,
        # asr_fn=None means _seal_mic_block won't call Parakeet in most tests
    )
    agent._seal_mic_block = lambda start, end: None  # disable ASR side-effects
    agent._frozen_time = start_time
    agent._now = lambda: agent._frozen_time
    return agent


def advance(agent: DuplexAudioAgent, seconds: float) -> None:
    agent._frozen_time += seconds


def force_block(agent: DuplexAudioAgent):
    """Force a block advance regardless of timing."""
    agent._next_block_ts = 0.0
    return agent.poll()


# ---------------------------------------------------------------------------
# N words per block
# ---------------------------------------------------------------------------

def test_n_words_per_block_default():
    # N = ceil(150 * 2.0 / 60) = ceil(5.0) = 5
    agent = make_agent(wpm=150, block_s=2.0)
    assert agent._n == 5


def test_n_words_per_block_custom():
    # N = ceil(120 * 3.0 / 60) = ceil(6.0) = 6
    agent = make_agent(wpm=120, block_s=3.0)
    assert agent._n == 6


def test_n_words_per_block_rounds_up():
    # N = ceil(100 * 1.0 / 60) = ceil(1.67) = 2
    agent = make_agent(wpm=100, block_s=1.0)
    assert agent._n == 2


# ---------------------------------------------------------------------------
# _commit_block_words
# ---------------------------------------------------------------------------

def test_commit_block_words_takes_n():
    agent = make_agent()  # N=5
    agent._pending_words = ["a", "b", "c", "d", "e", "f", "g"]
    agent._ensure_current_block(1000.0)
    agent._commit_block_words()
    assert agent._current_block.assistant_text == "a b c d e"
    assert agent._pending_words == ["f", "g"]


def test_commit_block_words_fewer_than_n():
    agent = make_agent()
    agent._pending_words = ["hello", "world."]
    agent._ensure_current_block(1000.0)
    agent._commit_block_words()
    assert agent._current_block.assistant_text == "hello world."
    assert agent._pending_words == []


def test_commit_block_words_empty_pending():
    agent = make_agent()
    agent._pending_words = []
    agent._ensure_current_block(1000.0)
    agent._commit_block_words()
    assert agent._current_block.assistant_text == ""
    assert agent._committed_words == []


def test_commit_block_words_updates_committed():
    agent = make_agent()
    agent._pending_words = ["hello", "world."]
    agent._ensure_current_block(1000.0)
    agent._commit_block_words()
    assert agent._committed_words == ["hello", "world."]


def test_commit_block_words_accumulates_committed_across_calls():
    agent = make_agent()
    agent._pending_words = ["a", "b", "c", "d", "e", "f."]
    agent._ensure_current_block(1000.0)
    agent._commit_block_words()  # commits a b c d e
    # Simulate second block
    agent._current_block = DuplexAudioBlock("b1", 1002.0, 1004.0)
    agent._commit_block_words()  # commits f
    assert agent._committed_words == ["a", "b", "c", "d", "e", "f."]


def test_commit_block_words_drops_short_nonterminal_tail_and_reopens_llm():
    agent = make_agent()
    agent.context_version = 3
    agent._last_accepted_response_context_version = 3
    agent._pending_words = ["tail"]
    agent._pending_llm_latency_s = 0.123
    agent._pending_response_asr_latency_s = 0.456
    agent._pending_response_source_block_id = "src-1"
    agent._ensure_current_block(1000.0)

    agent._commit_block_words()

    assert agent._current_block.assistant_text == ""
    assert agent._pending_words == []
    assert agent._pending_llm_latency_s is None
    assert agent._pending_response_asr_latency_s is None
    assert agent._pending_response_source_block_id is None
    assert agent._last_accepted_response_context_version is None


def test_commit_block_words_attaches_pending_response_timings():
    agent = make_agent()
    agent._pending_words = ["hello", "world."]
    agent._pending_llm_latency_s = 0.123
    agent._pending_response_asr_latency_s = 0.456
    agent._pending_response_source_block_id = "src-1"
    agent._ensure_current_block(1000.0)

    agent._commit_block_words()

    assert agent._current_block.assistant_text == "hello world."
    assert agent._current_block.llm_latency_s == 0.123
    assert agent._current_block.asr_latency_s == 0.456
    assert agent._current_block.response_source_block_id == "src-1"
    assert agent._pending_llm_latency_s is None
    assert agent._pending_response_asr_latency_s is None
    assert agent._pending_response_source_block_id is None


# ---------------------------------------------------------------------------
# _update_pending_queue — two-branch reconciliation
# ---------------------------------------------------------------------------

def test_update_pending_queue_empty_queue_no_committed():
    """Queue empty + nothing committed → full proposal becomes pending."""
    agent = make_agent()
    agent._committed_words = []
    agent._pending_words = []
    agent._update_pending_queue(["hello", "world"])
    assert agent._pending_words == ["hello", "world"]


def test_update_pending_queue_empty_queue_strips_committed():
    """Queue empty → strip already-spoken committed words from proposal head."""
    agent = make_agent()
    agent._committed_words = ["a", "b", "c"]
    agent._pending_words = []
    agent._update_pending_queue(["a", "b", "c", "d", "e"])
    assert agent._pending_words == ["d", "e"]


def test_update_pending_queue_empty_queue_all_spoken():
    """Proposal only contains already-committed words → empty queue."""
    agent = make_agent()
    agent._committed_words = ["a", "b"]
    agent._pending_words = []
    agent._update_pending_queue(["a", "b"])
    assert agent._pending_words == []


def test_update_pending_queue_nonempty_strips_echo():
    """Queue non-empty: LLM echoes committed prefix → strip echo, retain match."""
    agent = make_agent()
    agent._committed_words = ["yeah", "im", "here"]
    agent._pending_words = ["what", "is", "up"]
    # LLM returns full utterance including committed prefix
    agent._update_pending_queue(["yeah", "im", "here", "what", "is", "up"])
    assert agent._pending_words == ["what", "is", "up"]


def test_update_pending_queue_strips_normalized_suffix_overlap():
    """Queue empty: strip replayed committed suffix despite punctuation/case drift."""
    agent = make_agent()
    agent._committed_words = ["Yeah", "here."]
    agent._pending_words = []

    agent._update_pending_queue(["yeah", "here,", "what", "is", "up?"])

    assert agent._pending_words == ["what", "is", "up?"]


def test_update_pending_queue_nonempty_mismatch():
    """Queue non-empty: proposal diverges → replace from mismatch point."""
    agent = make_agent()
    agent._committed_words = ["yeah"]
    agent._pending_words = ["old", "words"]
    agent._update_pending_queue(["yeah", "new", "words"])
    assert agent._pending_words == ["new", "words"]


def test_update_pending_queue_mismatch_mid_queue():
    """Mismatch at position 1 in unspoken queue → retain up to mismatch."""
    agent = make_agent()
    agent._committed_words = []
    agent._pending_words = ["a", "b", "c"]
    agent._update_pending_queue(["a", "x", "y"])
    assert agent._pending_words == ["a", "x", "y"]


def test_update_pending_queue_extends_matching_tail():
    """Proposal adds new words beyond matching prefix → append new words."""
    agent = make_agent()
    agent._committed_words = []
    agent._pending_words = ["a", "b"]
    agent._update_pending_queue(["a", "b", "c", "d"])
    assert agent._pending_words == ["a", "b", "c", "d"]


def test_update_pending_queue_shorter_proposal():
    """Proposal shorter than queue → truncate to proposal tail."""
    agent = make_agent()
    agent._committed_words = []
    agent._pending_words = ["a", "b", "c"]
    agent._update_pending_queue(["a"])
    assert agent._pending_words == ["a"]


def test_llm_generate_groq_omits_unsupported_reasoning_effort(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(full_duplex, "_llm_client", fake_client)
    monkeypatch.setattr(full_duplex, "_next_model_index", 1)

    result = full_duplex.llm_generate_groq("sys", "user")

    assert result == "ok"
    assert captured["model"] == "llama-3.1-8b-instant"
    assert "reasoning_effort" not in captured


def test_llm_generate_groq_keeps_supported_reasoning_effort(monkeypatch):
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    monkeypatch.setattr(full_duplex, "_llm_client", fake_client)
    monkeypatch.setattr(full_duplex, "_next_model_index", 2)

    result = full_duplex.llm_generate_groq("sys", "user")

    assert result == "ok"
    assert captured["model"] == "openai/gpt-oss-20b"
    assert captured["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# _format_timeblocks — new token format
# ---------------------------------------------------------------------------

def test_format_timeblocks_empty():
    """No history, no current block, no forced words."""
    agent = make_agent()
    result = agent._format_timeblocks()
    assert result == "<user><idle><AI>"


def test_format_timeblocks_idle_block():
    """Block with no user text → <idle>."""
    agent = make_agent()
    agent.blocks = [DuplexAudioBlock("b0", 999.0, 1001.0, assistant_text="hi")]
    result = agent._format_timeblocks()
    assert "<user><idle><AI>hi</s>" in result
    assert result.endswith("<user><idle><AI>")


def test_format_timeblocks_with_user_text():
    """Block with user text → <user>TEXT<AI>TEXT</s>."""
    agent = make_agent()
    agent.blocks = [
        DuplexAudioBlock("b0", 999.0, 1001.0, user_text="hello", assistant_text="hi")
    ]
    result = agent._format_timeblocks()
    assert "<user>hello<AI>hi</s>" in result

def test_format_timeblocks_omits_stale_assistant_text():
    agent = make_agent()
    agent.blocks = [
        DuplexAudioBlock(
            "b0",
            999.0,
            1001.0,
            user_text="corrected user text",
            assistant_text="stale assistant text",
            assistant_text_stale=True,
        )
    ]

    result = agent._format_timeblocks()

    assert "<user>corrected user text<AI></s>" in result
    assert "stale assistant text" not in result


def test_history_summary_marks_stale_assistant_text():
    agent = make_agent()
    agent.blocks = [
        DuplexAudioBlock("b0", 999.0, 1001.0, user_text="hi", assistant_text="hello there"),
        DuplexAudioBlock(
            "b1",
            1001.0,
            1003.0,
            user_text="corrected topic",
            assistant_text="old answer",
            assistant_text_stale=True,
        ),
    ]

    summary = agent._history_summary(agent.blocks)

    assert "u='hi' ai='hello there'" in summary
    assert "u='corrected topic' ai='-' stale_ai='old answer'" in summary

def test_format_timeblocks_no_forced_words():
    """_format_timeblocks does not include pending words — ends with bare <AI>."""
    agent = make_agent()
    agent._pending_words = ["hello", "world", "how", "are", "you", "extra"]
    result = agent._format_timeblocks()
    assert result.endswith("<user><idle><AI>")


def test_format_timeblocks_current_block_user_text():
    """Current block user text appears in prompt."""
    agent = make_agent()
    agent._ensure_current_block(1000.0)
    agent._current_block.user_text = "what time is it"
    result = agent._format_timeblocks()
    assert result.endswith("<user>what time is it<AI>")


def test_format_timeblocks_multiple_blocks():
    """Multiple finalized blocks appear sequentially."""
    agent = make_agent()
    agent.blocks = [
        DuplexAudioBlock("b0", 998.0, 1000.0, user_text="hi", assistant_text="hello"),
        DuplexAudioBlock("b1", 1000.0, 1002.0, assistant_text="there"),
    ]
    result = agent._format_timeblocks()
    assert "<user>hi<AI>hello</s><user><idle><AI>there</s>" in result


# ---------------------------------------------------------------------------
# poll() — block advance and scheduling
# ---------------------------------------------------------------------------

def test_poll_returns_none_before_scheduled():
    """poll() returns None when next_block_ts is in the future."""
    agent = make_agent()
    force_block(agent)   # sets next_block_ts = now + tts_duration
    result = agent.poll()
    assert result is None


def test_poll_advances_block_when_due():
    """poll() creates a new finalized block when called after next_block_ts."""
    agent = make_agent()
    n_before = len(agent.blocks)
    force_block(agent)
    assert len(agent.blocks) == n_before + 1


def test_tts_duration_sets_next_block_ts():
    """next_block_ts = now + actual TTS audio duration."""
    tts_dur = 3.5

    def mock_tts(text):
        return TTS_SAMPLE_RATE, np.zeros(int(tts_dur * TTS_SAMPLE_RATE), dtype=np.int16)

    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=mock_tts,
    )
    agent._seal_mic_block = lambda *_: None
    agent._frozen_time = 1000.0
    agent._now = lambda: agent._frozen_time

    agent._pending_words = ["hello", "world", "from", "tts", "test"]
    force_block(agent)
    assert abs(agent._next_block_ts - (1000.0 + tts_dur)) < 0.01


def test_silence_block_uses_default_block_s():
    """When no words pending, silence duration = default_block_s."""
    agent = make_agent()
    agent._pending_words = []
    force_block(agent)
    assert abs(agent._next_block_ts - (1000.0 + DEFAULT_BLOCK_S)) < 0.01


def test_silence_block_has_no_assistant_text():
    """Silent block has empty assistant_text."""
    agent = make_agent()
    agent._pending_words = []
    force_block(agent)
    assert agent.blocks[-1].assistant_text == ""


def test_poll_returns_audio_chunk():
    """poll() returns (sr, arr) when TTS audio is queued."""
    agent = make_agent()
    agent._pending_words = ["hello", "world", "how", "are", "you"]
    result = force_block(agent)
    assert result is not None
    sr, arr = result
    assert sr == TTS_SAMPLE_RATE
    assert isinstance(arr, np.ndarray)


def test_poll_no_audio_when_queue_empty():
    """Subsequent poll() before next_block_ts returns None when queue drained."""
    agent = make_agent()
    force_block(agent)     # drains the TTS chunk
    result = agent.poll()  # still before next_block_ts → None
    assert result is None


# ---------------------------------------------------------------------------
# Silence audio
# ---------------------------------------------------------------------------

def test_silence_audio_is_zeros():
    """Silent block emits zero-filled audio at TTS_SAMPLE_RATE."""
    agent = make_agent()
    agent._pending_words = []
    result = force_block(agent)
    # poll() drains and returns the audio chunk directly
    assert result is not None
    sr, audio = result
    assert sr == TTS_SAMPLE_RATE
    assert np.all(audio == 0)
    expected_samples = int(DEFAULT_BLOCK_S * TTS_SAMPLE_RATE)
    assert len(audio) == expected_samples


# ---------------------------------------------------------------------------
# receive_mic_chunk
# ---------------------------------------------------------------------------

def test_receive_mic_chunk_is_fast():
    """receive_mic_chunk returns immediately (ASR happens in background)."""
    agent = make_agent()
    arr = np.zeros(16000, dtype=np.float32)
    start = time.time()
    agent.receive_mic_chunk(16000, arr)
    elapsed = time.time() - start
    assert elapsed < 0.1


def test_receive_mic_chunk_accumulates():
    """Multiple mic chunks accumulate in _mic_current."""
    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda t: (TTS_SAMPLE_RATE, np.zeros(48000, dtype=np.int16)),
    )
    agent._seal_mic_block = lambda *_: None
    arr = np.zeros(800, dtype=np.float32)
    agent.receive_mic_chunk(16000, arr)
    agent.receive_mic_chunk(16000, arr)
    assert len(agent._mic_current) == 1600


# ---------------------------------------------------------------------------
# Mic block sealing
# ---------------------------------------------------------------------------

def test_mic_sealed_per_block():
    """_seal_mic_block called on each block advance."""
    sealed_calls = []

    def mock_tts(text):
        return TTS_SAMPLE_RATE, np.zeros(int(2.0 * TTS_SAMPLE_RATE), dtype=np.int16)

    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=mock_tts,
    )
    agent._frozen_time = 1000.0
    agent._now = lambda: agent._frozen_time
    agent._seal_mic_block = lambda s, e: sealed_calls.append((s, e))

    force_block(agent)
    assert len(sealed_calls) == 1


def test_mic_rolling_buffer_capped():
    """_mic_rolling never exceeds MAX_MIC_BLOCKS entries."""
    from full_duplex import MAX_MIC_BLOCKS

    asr_rolling_sizes = []

    def mock_asr(rolling, agent_ref):
        asr_rolling_sizes.append(len(rolling))

    def mock_tts(text):
        return TTS_SAMPLE_RATE, np.zeros(int(2.0 * TTS_SAMPLE_RATE), dtype=np.int16)

    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=mock_tts,
        asr_fn=mock_asr,
    )
    agent._frozen_time = 1000.0
    agent._now = lambda: agent._frozen_time

    for _ in range(MAX_MIC_BLOCKS + 3):
        force_block(agent)

    agent._executor.shutdown(wait=True)

    assert max(asr_rolling_sizes) == MAX_MIC_BLOCKS


# ---------------------------------------------------------------------------
# Audio output queue cap
# ---------------------------------------------------------------------------

def test_audio_queue_cap():
    """Chunks dropped when total queued audio exceeds MAX_AUDIO_QUEUE_S."""
    agent = make_agent()
    chunk_s = 2.0
    chunk = np.zeros(int(chunk_s * TTS_SAMPLE_RATE), dtype=np.int16)
    # Enqueue far more than the cap allows
    n_to_enqueue = int(MAX_AUDIO_QUEUE_S / chunk_s) + 10
    for _ in range(n_to_enqueue):
        agent._enqueue_audio(TTS_SAMPLE_RATE, chunk)

    total = sum(len(a) / s for s, a in list(agent._audio_queue.queue))
    # Should be at most cap + one extra chunk (the last one that fit)
    assert total <= MAX_AUDIO_QUEUE_S + chunk_s


# ---------------------------------------------------------------------------
# History pruning
# ---------------------------------------------------------------------------

def test_history_pruned_after_10min():
    """Blocks older than MAX_HISTORY_S are removed."""
    agent = make_agent(start_time=1000.0)
    old = DuplexAudioBlock("old", 0.0, 100.0)
    recent = DuplexAudioBlock("recent", 990.0, 1000.0)
    agent.blocks = [old, recent]
    agent._prune_history(1000.0)
    ids = [b.block_id for b in agent.blocks]
    assert "old" not in ids
    assert "recent" in ids


def test_history_not_pruned_within_window():
    """Blocks within MAX_HISTORY_S window are kept."""
    agent = make_agent(start_time=1000.0)
    agent.blocks = [
        DuplexAudioBlock("b0", 400.0, 401.0),  # within 600s window
        DuplexAudioBlock("b1", 999.0, 1000.0),
    ]
    agent._prune_history(1000.0)
    assert len(agent.blocks) == 2


# ---------------------------------------------------------------------------
# receive_text_message
# ---------------------------------------------------------------------------

def test_receive_text_message_resets_committed():
    """New user message clears committed word tracking."""
    agent = make_agent()
    agent._committed_words = ["hello", "world"]
    agent.receive_text_message("new input")
    assert agent._committed_words == []

def test_receive_text_message_clears_pending_future_continuation():
    agent = make_agent()
    agent._pending_words = ["old", "reply"]
    agent._committed_words = ["already", "spoken"]
    agent._pending_llm_latency_s = 0.123
    agent._pending_response_asr_latency_s = 0.456
    agent._pending_response_source_block_id = "src"

    agent.receive_text_message("new input")

    assert agent._pending_words == []
    assert agent._committed_words == []
    assert agent._pending_llm_latency_s is None
    assert agent._pending_response_asr_latency_s is None
    assert agent._pending_response_source_block_id is None

def test_mark_assistant_history_stale_from_hides_corrected_tail_from_prompt():
    agent = make_agent()
    agent.blocks = [
        DuplexAudioBlock("b0", 0.0, 2.0, user_text="hi", assistant_text="hello there"),
        DuplexAudioBlock("b1", 2.0, 4.0, user_text="old topic", assistant_text="old answer"),
        DuplexAudioBlock("b2", 4.0, 6.0, assistant_text="continued stale answer"),
    ]

    agent._mark_assistant_history_stale_from(1)

    assert agent.blocks[0].assistant_text_stale is False
    assert agent.blocks[1].assistant_text_stale is True
    assert agent.blocks[2].assistant_text_stale is True

    prompt = agent._format_timeblocks()
    assert "<user>hi<AI>hello there</s>" in prompt
    assert "<user>old topic<AI></s>" in prompt
    assert "old answer" not in prompt
    assert "continued stale answer" not in prompt

def test_receive_text_message_bumps_context_version():
    agent = make_agent()
    v_before = agent.context_version
    agent.receive_text_message("hi there")
    assert agent.context_version == v_before + 1


def test_receive_text_message_sets_user_text():
    agent = make_agent()
    agent.receive_text_message("hello there")
    assert agent._current_block.user_text == "hello there"


def test_classify_text_change_punctuation_only():
    agent = make_agent()
    assert agent._classify_text_change("relativity", "relativity?") == "punctuation"


def test_classify_text_change_topic_shift():
    agent = make_agent()
    assert agent._classify_text_change("two plus two", "theory of relativity") == "topic-shift"


# ---------------------------------------------------------------------------
# Context version / stale detection
# ---------------------------------------------------------------------------

def test_stale_llm_response_discarded():
    """LLM response for a previous context version is discarded."""
    def intercepting_llm(sys, usr):
        # Bump context mid-call to simulate new user message
        return "hello world"

    agent = make_agent(llm_fn=intercepting_llm)
    agent.receive_text_message("first message")

    orig_fn = agent._llm_generate_fn
    def bumping_fn(sys, usr):
        result = orig_fn(sys, usr)
        agent.context_version += 1  # simulate interrupt during LLM call
        return result

    agent._llm_generate_fn = bumping_fn
    force_block(agent)
    # Stale response should be discarded → pending queue unchanged (empty)
    assert agent._pending_words == []


def test_llm_exception_captured_without_breaking_poll():
    """LLM failures are recorded and do not leave the agent stuck in flight."""
    def raising_llm(_, __):
        raise RuntimeError("model exploded")

    agent = make_agent(llm_fn=raising_llm)
    agent.receive_text_message("hi")

    result = force_block(agent)

    assert result is not None
    assert agent.last_llm_error == "RuntimeError: model exploded"
    assert agent.last_llm_error_seq == 1
    assert agent._llm_in_flight is False
    assert agent._pending_words == []


def test_llm_success_clears_previous_error_message():
    """Successful generations clear the current error message after a prior failure."""
    calls = {"count": 0}

    def flaky_llm(_, __):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary failure")
        return "hello world"

    agent = make_agent(llm_fn=flaky_llm)
    agent.receive_text_message("hi")

    force_block(agent)
    assert agent.last_llm_error == "RuntimeError: temporary failure"

    advance(agent, 2.1)
    force_block(agent)

    assert agent.last_llm_error is None
    assert agent._pending_words == ["hello", "world"]


def test_same_context_does_not_rerun_after_nonempty_response_accepted():
    calls = {"count": 0}

    def llm_fn(_, __):
        calls["count"] += 1
        return "hello world again"

    agent = make_agent(llm_fn=llm_fn)
    agent.receive_text_message("hi")

    force_block(agent)
    assert calls["count"] == 1

    advance(agent, 0.5)
    agent._next_block_ts = agent._frozen_time + 1.0
    agent.poll()
    assert calls["count"] == 1


def test_waiting_poll_can_start_llm_before_next_block_ts():
    calls = {"count": 0}

    def llm_fn(_, __):
        calls["count"] += 1
        return "hello world"

    agent = make_agent(llm_fn=llm_fn)
    agent.receive_text_message("hi")
    force_block(agent)
    assert calls["count"] == 1

    agent.context_version += 1
    agent._last_accepted_response_context_version = None
    agent._latest_user_source_block_id = agent.blocks[-1].block_id
    agent._next_block_ts = agent._frozen_time + 5.0
    advance(agent, 1.0)

    agent.poll()

    assert calls["count"] == 2


def test_dropped_short_tail_allows_same_context_llm_continuation():
    responses = iter([
        "one two three four five six seven eight nine ten eleven",
        "twelve thirteen fourteen fifteen sixteen",
    ])
    calls = {"count": 0}

    def llm_fn(_, __):
        calls["count"] += 1
        return next(responses)

    agent = make_agent(llm_fn=llm_fn)
    agent.receive_text_message("hi")

    force_block(agent)
    assert calls["count"] == 1
    assert agent._pending_words == [
        "one", "two", "three", "four", "five",
        "six", "seven", "eight", "nine", "ten", "eleven",
    ]

    advance(agent, 2.1)
    force_block(agent)
    assert agent.blocks[-1].assistant_text == "one two three four five"

    advance(agent, 2.1)
    force_block(agent)
    assert agent.blocks[-1].assistant_text == "six seven eight nine ten"
    assert agent._pending_words == ["eleven"]
    assert calls["count"] == 1

    advance(agent, 2.1)
    force_block(agent)

    assert calls["count"] == 2
    assert agent._pending_words == [
        "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    ]


# ---------------------------------------------------------------------------
# End-to-end poll flow
# ---------------------------------------------------------------------------

def test_words_committed_to_block_on_advance():
    """Words from pending queue appear in finalized block after poll."""
    def llm_fn(_, __):
        return "alpha beta gamma delta epsilon"

    agent = make_agent(llm_fn=llm_fn)
    agent.receive_text_message("hi")
    force_block(agent)  # LLM called, pending fills
    # pending_words should have words now
    assert len(agent._pending_words) > 0

    # Second block advance → commits first N words to blocks[-1]
    advance(agent, 2.1)
    force_block(agent)
    assert agent.blocks[-1].assistant_text != ""


def test_accepted_llm_output_truncated_to_n_words():
    agent = make_agent(llm_fn=lambda *_: "one two three four five six seven", wpm=90, block_s=2.0)
    agent.receive_text_message("hi")

    force_block(agent)

    assert agent._n == 3
    assert agent._pending_words == ["one", "two", "three"]


def test_no_duplicate_word_replay():
    """Each word appears exactly once in history despite repeated LLM responses."""
    responses = [
        "Yeah here.",
        "Yeah here. What's up?",
        "Yeah here. What's up?",
    ]
    idx = {"v": 0}

    def llm_fn(_, __):
        r = responses[min(idx["v"], len(responses) - 1)]
        idx["v"] += 1
        return r

    agent = make_agent(llm_fn=llm_fn, wpm=90, block_s=2.0)  # N=3
    agent.receive_text_message("hi")

    for _ in range(6):
        force_block(agent)
        advance(agent, 2.1)

    all_words = []
    for block in agent.blocks:
        if block.assistant_text:
            all_words.extend(block.assistant_text.split())

    from collections import Counter
    counts = Counter(all_words)
    for word in ["Yeah", "here.", "What's", "up?"]:
        assert counts[word] <= 1, f"{word!r} appeared {counts[word]} times"


def test_first_reply_block_carries_llm_and_total_timing(monkeypatch):
    perf_values = iter([10.0, 10.25, 10.8])
    monkeypatch.setattr(time, "perf_counter", lambda: next(perf_values))

    agent = make_agent(llm_fn=lambda *_: "hello world")
    user_block = DuplexAudioBlock(
        "user-1",
        998.0,
        1000.0,
        user_text="hi",
        asr_latency_s=0.456,
        asr_started_perf_s=5.0,
    )
    agent.blocks = [user_block]
    agent._latest_user_source_block_id = user_block.block_id

    force_block(agent)
    advance(agent, 2.1)
    force_block(agent)

    reply_block = agent.blocks[-1]
    assert reply_block.assistant_text == "hello world"
    assert reply_block.llm_latency_s == pytest.approx(0.25)
    assert reply_block.asr_latency_s == pytest.approx(0.456)
    assert reply_block.total_latency_s == pytest.approx(5.8)
    assert reply_block.response_source_block_id == "user-1"


def test_reply_block_uses_elastic_timeline_and_debug_audio(monkeypatch):
    perf_values = iter([10.0, 10.25, 10.8])
    monkeypatch.setattr(time, "perf_counter", lambda: next(perf_values))

    agent = make_agent(llm_fn=lambda *_: "hello world", tts_duration=2.0)
    user_block = DuplexAudioBlock(
        "user-1",
        998.0,
        1000.0,
        user_text="hi",
        asr_latency_s=0.456,
        asr_started_perf_s=5.0,
    )
    agent.blocks = [user_block]
    agent._latest_user_source_block_id = user_block.block_id

    force_block(agent)
    advance(agent, 2.1)
    result = force_block(agent)

    reply_block = agent.blocks[-1]
    assert result is not None
    sr, playback_audio = result
    assert sr == TTS_SAMPLE_RATE
    assert len(playback_audio) == int(2.0 * TTS_SAMPLE_RATE)
    assert reply_block.lead_silence_s == pytest.approx(5.8)
    assert reply_block.timeline_start_ts == pytest.approx(996.3)
    assert reply_block.timeline_end_ts == pytest.approx(1004.1)
    assert len(reply_block.tts_audio) == int((5.8 + 2.0) * TTS_SAMPLE_RATE)


# ---------------------------------------------------------------------------
# get_chat_history
# ---------------------------------------------------------------------------

def test_get_chat_history_empty():
    agent = make_agent()
    assert agent.get_chat_history() == []


def test_get_chat_history_merges_consecutive_roles():
    agent = make_agent()
    agent.blocks = [
        DuplexAudioBlock("b0", 0.0, 2.0, user_text="hi"),
        DuplexAudioBlock("b1", 2.0, 4.0, user_text="there"),
        DuplexAudioBlock("b2", 4.0, 6.0, assistant_text="hello world"),
    ]
    history = agent.get_chat_history()
    assert history[0] == {"role": "user", "content": "hi there"}
    assert history[1] == {"role": "assistant", "content": "hello world"}


# ---------------------------------------------------------------------------
# ASR window management (reused from full_duplex.py)
# ---------------------------------------------------------------------------

def test_asr_window_ingest_basic():
    agent = make_agent()
    result = agent.ingest_parakeet_window(
        start_ts=1000.0,
        end_ts=1002.0,
        words=[("hello", 1001.0)],
        window_id="win-0",
    )
    assert result is True
    state = agent.get_asr_window_state()
    assert len(state) == 1
    assert state[0]["words"] == ["hello"]


def test_asr_window_mutable_correction():
    """Recent windows accept corrections."""
    agent = make_agent()
    agent.ingest_parakeet_window(1000.0, 1002.0, [("orig", 1001.0)], "win-0")
    accepted = agent.ingest_parakeet_window(1000.0, 1002.0, [("fixed", 1001.0)], "win-0")
    assert accepted is True
    state = agent.get_asr_window_state()
    assert state[0]["words"] == ["fixed"]
    assert state[0]["revision"] == 1


def test_asr_window_old_not_mutable():
    """Windows outside the mutable tail reject corrections."""
    agent = make_agent()
    # Fill 20 windows (max_asr_windows)
    for i in range(20):
        agent.ingest_parakeet_window(float(i), float(i + 1), [(f"w{i}", float(i + 1))], f"win-{i}")

    # win-0 is now outside mutable range → should be rejected
    accepted = agent.ingest_parakeet_window(0.0, 1.0, [("patched", 0.5)], "win-0")
    assert accepted is False

"""
Tests for TextOnlyDuplexAgent.

Timing-sensitive tests use frozen-time helpers so they never call
``time.sleep`` and are fully deterministic.
"""

import time

from full_duplex import (
    DuplexAgentTimeBlock,
    PurposedWord,
    TextOnlyDuplexAgent,
    _default_word_timing_fn,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_timed_agent(
    llm_fn,
    start_time: float = 1000.0,
    poll_ms: int = 800,
    word_dur: float = 0.25,
) -> TextOnlyDuplexAgent:
    """Agent with frozen _now() for deterministic timing tests."""
    def timing_fn(text: str):
        words = text.split()
        result, cursor = [], 0.0
        for w in words:
            result.append((w, cursor, cursor + word_dur))
            cursor += word_dur
        return result

    agent = TextOnlyDuplexAgent(
        llm_generate_fn=llm_fn,
        agent_poll_ms=poll_ms,
        word_timing_fn=timing_fn,
    )
    agent._frozen_time = start_time
    agent._now = lambda: agent._frozen_time
    return agent


def advance(agent: TextOnlyDuplexAgent, seconds: float) -> None:
    agent._frozen_time += seconds


def force_tick(agent: TextOnlyDuplexAgent) -> DuplexAgentTimeBlock:
    agent.next_scheduler_ts = 0
    return agent.poll()


# ---------------------------------------------------------------------------
# _default_word_timing_fn
# ---------------------------------------------------------------------------

def test_default_timing_fn_uniform_spacing():
    result = _default_word_timing_fn("hello there friend")
    assert len(result) == 3
    words = [r[0] for r in result]
    assert words == ["hello", "there", "friend"]
    # Each slot is 0.25 s wide, back-to-back.
    assert result[0] == ("hello",  0.0,  0.25)
    assert result[1] == ("there",  0.25, 0.50)
    assert result[2] == ("friend", 0.50, 0.75)


# ---------------------------------------------------------------------------
# PurposedWord creation
# ---------------------------------------------------------------------------

def test_purposed_word_creation_timing():
    agent = make_timed_agent(lambda *_: "", start_time=100.0, word_dur=0.25)
    words = agent._create_purposed_words("hello there", tts_start_ts=100.0)
    assert len(words) == 2
    assert words[0].text == "hello"
    assert words[0].start_time == 100.0
    assert words[0].end_time  == 100.25
    assert words[1].text == "there"
    assert words[1].start_time == 100.25
    assert words[1].end_time  == 100.50


def test_purposed_word_creation_normalizes_unicode():
    agent = make_timed_agent(lambda *_: "", start_time=0.0)
    # curly apostrophe should become straight
    words = agent._create_purposed_words("\u201chello\u201d there", tts_start_ts=0.0)
    assert words[0].text == '"hello"'


# ---------------------------------------------------------------------------
# _update_purposed_queue — core mismatch logic
# ---------------------------------------------------------------------------

def test_update_queue_initial_fill():
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    agent._update_purposed_queue(["hello", "there", "friend"])
    assert [w.text for w in agent.purposed_audio] == ["hello", "there", "friend"]
    # All future timestamps (start time is 1000.0).
    assert agent.purposed_audio[0].end_time == 1000.25
    assert agent.purposed_audio[2].end_time == 1000.75


def test_update_queue_extends_matching_tail():
    """Matching prefix retains same PurposedWord instances (same timestamps)."""
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    agent._update_purposed_queue(["A", "B"])
    original_a = agent.purposed_audio[0]
    original_b = agent.purposed_audio[1]

    # Proposal extends with a third word — A and B must be retained.
    agent._update_purposed_queue(["A", "B", "C"])
    assert agent.purposed_audio[0] is original_a
    assert agent.purposed_audio[1] is original_b
    assert agent.purposed_audio[2].text == "C"
    # C starts where B ends.
    assert agent.purposed_audio[2].start_time == original_b.end_time


def test_update_queue_divergence_discards_from_mismatch():
    """Mismatch at index N → words N+ deleted, replaced with new tail."""
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    agent._update_purposed_queue(["A", "B", "C"])
    original_a = agent.purposed_audio[0]

    # New proposal: A matches, B diverges → B and C discarded.
    agent._update_purposed_queue(["A", "X", "Y"])
    assert len(agent.purposed_audio) == 3
    assert agent.purposed_audio[0] is original_a     # retained
    assert agent.purposed_audio[1].text == "X"        # new
    assert agent.purposed_audio[2].text == "Y"        # new
    # X starts where A ended.
    assert agent.purposed_audio[1].start_time == original_a.end_time


def test_update_queue_preserves_spoken_prefix():
    """Words committed to history are skipped in mismatch comparison."""
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    block = DuplexAgentTimeBlock("b0", start_ts=999.0, end_ts=1005.0)
    agent.blocks = [block]

    # Fill initial queue.
    agent._update_purposed_queue(["A", "B", "C"])

    # Advance time past A and B and commit them.
    advance(agent, 0.6)   # now=1000.6; A.end=1000.25, B.end=1000.50 both past
    agent._commit_spoken_words(agent._now())
    assert agent._committed_words_current_utterance == ["A", "B"]
    assert [w.text for w in agent.purposed_audio] == ["C"]

    # New proposal: LLM echoes A and B, then diverges from C to X.
    # committed_count=2, proposal_tail=["C"→wait, it says "X"] so X != C → mismatch at 0.
    agent._update_purposed_queue(["A", "B", "X"])
    # purposed_audio should have just X now.
    assert [w.text for w in agent.purposed_audio] == ["X"]


def test_update_queue_shorter_proposal_no_new_words():
    """If proposal has no words beyond committed, queue ends up empty (no new tail)."""
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    block = DuplexAgentTimeBlock("b0", start_ts=999.0, end_ts=1005.0)
    agent.blocks = [block]

    agent._update_purposed_queue(["A", "B", "C"])
    advance(agent, 0.8)   # all three past end_time
    agent._commit_spoken_words(agent._now())
    # committed_count=3; proposal only has 1 word → proposal_tail = []
    agent._update_purposed_queue(["A"])
    assert agent.purposed_audio == []


# ---------------------------------------------------------------------------
# _commit_spoken_words — block assignment by end_time
# ---------------------------------------------------------------------------

def test_commit_spoken_words_assigns_correct_block():
    """Word goes to the block whose [start_ts, end_ts) contains its end_time."""
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)

    # Manually create two finalized blocks.
    block0 = DuplexAgentTimeBlock("b0", start_ts=1000.0, end_ts=1000.8)
    block1 = DuplexAgentTimeBlock("b1", start_ts=1000.8, end_ts=1001.6)
    agent.blocks = [block0, block1]

    # Word A ends at 1000.25 → block0; word B ends at 1000.9 → block1.
    agent.purposed_audio = [
        PurposedWord("A", 1000.0,  1000.25),
        PurposedWord("B", 1000.8,  1000.9),
    ]
    agent._commit_spoken_words(now=1001.0)

    assert block0.assistant_text == "A"
    assert block1.assistant_text == "B"
    assert agent.purposed_audio == []


def test_commit_spoken_words_leaves_future_words():
    """Words with end_time > now remain in purposed_audio."""
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    block = DuplexAgentTimeBlock("b0", start_ts=999.0, end_ts=1002.0)
    agent.blocks = [block]

    agent.purposed_audio = [
        PurposedWord("past",   999.0, 999.9),
        PurposedWord("future", 1000.0, 1001.0),
    ]
    agent._commit_spoken_words(now=1000.0)   # end_time <= 1000.0 → "past" only

    assert block.assistant_text == "past"
    assert len(agent.purposed_audio) == 1
    assert agent.purposed_audio[0].text == "future"


def test_commit_spoken_words_fallback_to_latest_block():
    """If no block window matches, word goes to the last finalized block."""
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    block = DuplexAgentTimeBlock("b0", start_ts=500.0, end_ts=501.0)
    agent.blocks = [block]

    # Word's end_time is far in the past — no block window matches.
    agent.purposed_audio = [PurposedWord("orphan", 0.0, 0.1)]
    agent._commit_spoken_words(now=1000.0)

    # Should land in the only available block (fallback).
    assert block.assistant_text == "orphan"


# ---------------------------------------------------------------------------
# _compute_forced_purposed_words
# ---------------------------------------------------------------------------

def test_compute_forced_words_correct_window():
    agent = make_timed_agent(lambda *_: "", start_time=1000.0, word_dur=0.25)
    # now=1000.0, next_block_end=1000.8
    agent.purposed_audio = [
        PurposedWord("past",    999.0, 999.5),    # end_time < now → not forced
        PurposedWord("forced1", 1000.0, 1000.3),  # in [now, 1000.8) → forced
        PurposedWord("forced2", 1000.0, 1000.79), # in [now, 1000.8) → forced
        PurposedWord("future",  1000.0, 1001.0),  # end_time >= next_block_end → not forced
    ]
    forced = agent._compute_forced_purposed_words(next_block_end_ts=1000.8)
    assert forced == ["forced1", "forced2"]


def test_compute_forced_words_empty_when_no_upcoming():
    agent = make_timed_agent(lambda *_: "", start_time=1000.0)
    agent.purposed_audio = [PurposedWord("far", 1000.0, 1005.0)]
    forced = agent._compute_forced_purposed_words(next_block_end_ts=1000.8)
    assert forced == []


# ---------------------------------------------------------------------------
# _format_timeblocks with forced words
# ---------------------------------------------------------------------------

def test_format_timeblocks_appends_forced_words():
    agent = make_timed_agent(lambda *_: "")
    result = agent._format_timeblocks(forced_words=["hello", "world"])
    assert result.endswith("<new_assistant_message>hello world")


def test_format_timeblocks_no_forced_words():
    agent = make_timed_agent(lambda *_: "")
    result = agent._format_timeblocks(forced_words=None)
    assert result.endswith("<new_assistant_message>")
    # No trailing space or extra content.
    assert result == "<new_assistant_message>"


def test_format_timeblocks_includes_history():
    agent = make_timed_agent(lambda *_: "")
    agent.blocks = [
        DuplexAgentTimeBlock("b0", 0.0, 0.8, user_text="hi", assistant_text="hello"),
    ]
    result = agent._format_timeblocks()
    assert '<user>"hi"' in result
    assert '<assistant>"hello"' in result
    assert result.endswith("<new_assistant_message>")


# ---------------------------------------------------------------------------
# End-to-end poll flow
# ---------------------------------------------------------------------------

def test_speaks_words_after_tts_duration():
    """After TTS duration elapses words appear in history blocks."""
    def llm_fn(_, __):
        return "hello there friend"

    agent = make_timed_agent(llm_fn, start_time=1000.0, poll_ms=800, word_dur=0.25)
    agent.receive_user_message("hi")

    # Tick 1: LLM called, purposed_audio filled, no words committed yet (now=1000.0).
    force_tick(agent)
    assert [w.text for w in agent.purposed_audio] == ["hello", "there", "friend"]
    assert not any(b.assistant_text for b in agent.blocks)

    # Advance 0.3 s → "hello" (end=1000.25) is spoken; "there" (1000.50) is not.
    advance(agent, 0.3)
    force_tick(agent)
    all_text = " ".join(b.assistant_text for b in agent.blocks if b.assistant_text)
    assert "hello" in all_text
    assert "there" not in all_text

    # Advance another 0.3 s → "there" now spoken.
    advance(agent, 0.3)
    force_tick(agent)
    all_text = " ".join(b.assistant_text for b in agent.blocks if b.assistant_text)
    assert "there" in all_text


def test_no_duplicate_word_replay_timed():
    """Repeated identical LLM responses don't replay already-spoken words."""
    responses = [
        "Yeah I'm here.",
        "Yeah I'm here. What's up?",
        "Yeah I'm here. What's up?",
    ]
    index = {"value": 0}

    def llm_fn(_, __):
        if index["value"] < len(responses):
            r = responses[index["value"]]
            index["value"] += 1
            return r
        return responses[-1]

    agent = make_timed_agent(llm_fn, start_time=1000.0, poll_ms=800, word_dur=0.25)
    agent.receive_user_message("hi")

    # Tick 1: fills queue with "Yeah I'm here."
    force_tick(agent)

    # Advance past all 3 words (0.75 s), then several more ticks.
    advance(agent, 0.8)
    for _ in range(5):
        force_tick(agent)
        advance(agent, 0.8)

    all_assistant_words = []
    for block in agent.blocks:
        if block.assistant_text:
            all_assistant_words.extend(block.assistant_text.split())

    # "Yeah", "I'm", "here.", "What's", "up?" each appear exactly once.
    from collections import Counter
    counts = Counter(all_assistant_words)
    for word in ["Yeah", "I'm", "here.", "What's", "up?"]:
        assert counts[word] == 1, f"word {word!r} appeared {counts[word]} times"


def test_queue_overwrite_keeps_matched_prefix():
    """LLM extension retains matching unspoken words with original timestamps."""
    calls = {"count": 0}

    def llm_fn(_, __):
        calls["count"] += 1
        if calls["count"] == 1:
            return "Yeah I'm here."
        return "Yeah I'm here. What's up?"

    agent = make_timed_agent(llm_fn, start_time=1000.0, poll_ms=800, word_dur=0.25)
    agent.receive_user_message("hello")

    force_tick(agent)
    assert [w.text for w in agent.purposed_audio] == ["Yeah", "I'm", "here."]
    original_yeah = agent.purposed_audio[0]

    force_tick(agent)
    # "Yeah" still unspoken (0.25 s not elapsed), so it is retained.
    assert agent.purposed_audio[0] is original_yeah
    # Extended with "What's" and "up?".
    assert [w.text for w in agent.purposed_audio][-2:] == ["What's", "up?"]


def test_spoken_words_replaced_on_new_context():
    """After new user message, old spoken words are gone and new proposal queued fresh."""
    responses = ["Hello there", "Hello there", "Hello world"]
    index = {"value": 0}

    def llm_fn(_, __):
        r = responses[index["value"]]
        index["value"] += 1
        return r

    agent = make_timed_agent(llm_fn, start_time=1000.0, poll_ms=800, word_dur=0.25)
    agent.receive_user_message("hi")

    # Tick 1: fills queue ["Hello", "there"]
    force_tick(agent)
    assert [w.text for w in agent.purposed_audio] == ["Hello", "there"]

    # Advance 0.3 s → "Hello" (0.25) is spoken.
    advance(agent, 0.3)
    # Tick 2: "Hello" committed; LLM returns "Hello there" again → "Hello" retained
    # (spoken), "there" matched as unspoken retained.
    force_tick(agent)

    # New user message → context_version bumps. "Hello world" next from LLM.
    agent.receive_user_message("new message")

    advance(agent, 0.5)
    force_tick(agent)
    # New proposal is "Hello world"; spoken "Hello" stays in queue until committed.
    unspoken_texts = [
        w.text for w in agent.purposed_audio if w.end_time >= agent._now()
    ]
    assert "world" in unspoken_texts


def test_only_spoken_tokens_in_history():
    """Only words whose TTS time has elapsed appear in block history."""
    def llm_fn(_, __):
        return "a b c d"

    agent = make_timed_agent(llm_fn, start_time=1000.0, poll_ms=800, word_dur=0.25)
    agent.receive_user_message("user")

    force_tick(agent)            # fills queue [a, b, c, d] at 1000.0
    advance(agent, 0.55)         # a.end=1000.25 ✓, b.end=1000.50 ✓, c.end=1000.75 ✗
    force_tick(agent)            # commits a and b; LLM called again

    spoken = " ".join(b.assistant_text for b in agent.blocks if b.assistant_text)
    assert "a" in spoken
    assert "b" in spoken
    assert "c" not in spoken


def test_empty_block_created_on_idle_tick():
    agent = TextOnlyDuplexAgent(llm_generate_fn=lambda *_: "", agent_poll_ms=500)

    block = force_tick(agent)

    assert block is not None
    assert block.user_text == ""
    assert block.assistant_text == ""


def test_scheduler_cadence_independent_of_fast_checks():
    agent = TextOnlyDuplexAgent(llm_generate_fn=lambda *_: "x y", agent_poll_ms=500)
    agent.receive_user_message("u")

    first = agent.poll()
    assert first is not None

    before = len(agent.blocks)
    for _ in range(100):
        agent.poll()
    after = len(agent.blocks)

    assert after == before

    agent.next_scheduler_ts = time.time() - 1
    second = agent.poll()
    assert second is not None


# ---------------------------------------------------------------------------
# ASR window tests — unchanged logic
# ---------------------------------------------------------------------------

def test_parakeet_keeps_only_20_recent_windows():
    agent = TextOnlyDuplexAgent(llm_generate_fn=lambda *_: "", agent_poll_ms=500)

    for idx in range(25):
        start = float(idx)
        end = start + 0.2
        agent.ingest_parakeet_window(
            start_ts=start,
            end_ts=end,
            words=[(f"w{idx}", end)],
            window_id=f"win-{idx}",
        )

    state = agent.get_asr_window_state()
    assert len(state) == 20
    assert [row["window_id"] for row in state] == [f"win-{idx}" for idx in range(5, 25)]


def test_parakeet_correction_allowed_only_latest_10_windows():
    agent = TextOnlyDuplexAgent(llm_generate_fn=lambda *_: "", agent_poll_ms=500)

    for idx in range(20):
        start = float(idx)
        end = start + 0.2
        agent.ingest_parakeet_window(
            start_ts=start,
            end_ts=end,
            words=[(f"orig{idx}", end)],
            window_id=f"win-{idx}",
        )

    accepted_old = agent.ingest_parakeet_window(
        start_ts=2.0,
        end_ts=2.2,
        words=[("patched_old", 2.2)],
        window_id="win-2",
    )
    assert accepted_old is False

    accepted_new = agent.ingest_parakeet_window(
        start_ts=15.0,
        end_ts=15.2,
        words=[("patched_new", 15.2)],
        window_id="win-15",
    )
    assert accepted_new is True

    state = {row["window_id"]: row for row in agent.get_asr_window_state()}
    assert state["win-2"]["words"] == ["orig2"]
    assert state["win-15"]["words"] == ["patched_new"]
    assert state["win-15"]["revision"] == 1


def test_frozen_windows_commit_words_once_to_user_stream():
    agent = TextOnlyDuplexAgent(llm_generate_fn=lambda *_: "", agent_poll_ms=500)

    for idx in range(11):
        start = float(idx)
        end = start + 0.2
        agent.ingest_parakeet_window(
            start_ts=start,
            end_ts=end,
            words=[(f"word{idx}", end)],
            window_id=f"win-{idx}",
        )

    history = agent.get_chat_history()
    assert history
    assert history[0]["role"] == "user"
    assert "word0" in history[0]["content"]

    accepted = agent.ingest_parakeet_window(
        start_ts=0.0,
        end_ts=0.2,
        words=[("updated", 0.2)],
        window_id="win-0",
    )
    assert accepted is False
    assert agent.get_chat_history()[0]["content"].count("word0") == 1

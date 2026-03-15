import time

from full_duplex import TextOnlyDuplexAgent


def force_tick(agent: TextOnlyDuplexAgent):
    agent.next_scheduler_ts = 0
    return agent.poll()


def test_speaks_one_word_per_tick():
    calls = {"count": 0}

    def llm_fn(_, __):
        calls["count"] += 1
        return "hello there friend"

    agent = TextOnlyDuplexAgent(llm_generate_fn=llm_fn, agent_poll_ms=500)
    agent.receive_user_message("hi there")

    force_tick(agent)
    assert agent.pending_assistant_words == ["hello", "there", "friend"]

    force_tick(agent)
    assert agent.blocks[-1].assistant_text == "hello"

    force_tick(agent)
    assert agent.blocks[-1].assistant_text == "there"


def test_queue_overwrite_keeps_unspoken_suffix_only():
    calls = {"count": 0}

    def llm_fn(_, __):
        calls["count"] += 1
        if calls["count"] == 1:
            return "Yeah, I'm here."
        return "Yeah, I'm here. What's up?"

    agent = TextOnlyDuplexAgent(llm_generate_fn=llm_fn, agent_poll_ms=500)
    agent.receive_user_message("hello")

    force_tick(agent)
    assert agent.pending_assistant_words == ["Yeah,", "I'm", "here."]

    force_tick(agent)
    assert agent.blocks[-1].assistant_text == "Yeah,"

    force_tick(agent)
    assert agent.blocks[-1].assistant_text == "I'm"
    assert agent.pending_assistant_words == ["here.", "What's", "up?"]


def test_only_spoken_tokens_persisted_to_blocks():
    def llm_fn(_, __):
        return "a b c d"

    agent = TextOnlyDuplexAgent(llm_generate_fn=llm_fn, agent_poll_ms=500)
    agent.receive_user_message("user")

    force_tick(agent)
    force_tick(agent)
    force_tick(agent)

    spoken = [block.assistant_text for block in agent.blocks if block.assistant_text]
    assert spoken == ["a", "b"]
    assert agent.pending_assistant_words == ["c", "d"]


def test_empty_block_created_on_idle_tick():
    agent = TextOnlyDuplexAgent(llm_generate_fn=lambda *_: "", agent_poll_ms=500)

    block = force_tick(agent)

    assert block is not None
    assert block.user_text == ""
    assert block.assistant_text == ""


def test_no_duplicate_word_replay_regression():
    responses = [
        "Yeah, I'm here.",
        "Yeah, I'm here. What’s up?",
        "Yeah, I'm here. What’s up?",
    ]
    index = {"value": 0}

    def llm_fn(_, __):
        if index["value"] < len(responses):
            value = responses[index["value"]]
            index["value"] += 1
            return value
        return responses[-1]

    agent = TextOnlyDuplexAgent(llm_generate_fn=llm_fn, agent_poll_ms=500)
    agent.receive_user_message("hi")

    force_tick(agent)
    force_tick(agent)
    force_tick(agent)
    force_tick(agent)
    force_tick(agent)
    force_tick(agent)

    assistant_tokens = []
    for block in agent.blocks:
        if block.assistant_text:
            assistant_tokens.extend(block.assistant_text.split())

    assert assistant_tokens == ["Yeah,", "I'm", "here.", "What’s", "up?"]


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

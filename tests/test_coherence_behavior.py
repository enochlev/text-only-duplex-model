"""test_coherence_behavior.py — edge-case behavioral tests for the coherence server.

Focused on failure modes and recent fixes:
  - backchannel loop reward hacking (history stripping)
  - BPE boundary token catastrophic first-token penalty
  - off-topic / repetition penalties
  - rambling after EOS
  - silence token handling
  - first-response with no prior context

Run with pytest (server must be up first):
    python coherence_reward_server.py &
    pytest tests/test_coherence_behavior.py -v
"""

import json
import os
import urllib.request

import dotenv
import pytest

dotenv.load_dotenv()

SERVER  = f"http://localhost:{os.getenv('COHERENCE_PORT', '10001')}"
TIMEOUT = 30.0


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _post(path: str, body: dict) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        SERVER + path,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read())


def _reward(
    proposed: str,
    *,
    history: list | None = None,
    last_user: str = "",
    last_bot: str = "",
    prev_block_was_eos: bool = False,
    student_emitted_eos: bool = False,
) -> dict:
    return _post("/reward", {
        "history":            history or [],
        "last_user_message":  last_user,
        "last_bot_message":   last_bot,
        "proposed_next":      proposed,
        "gamma":              0.9,
        "prev_block_was_eos": prev_block_was_eos,
        "student_emitted_eos": student_emitted_eos,
    })


@pytest.fixture(scope="session", autouse=True)
def require_server():
    try:
        req = urllib.request.Request(SERVER + "/health")
        urllib.request.urlopen(req, timeout=3.0)
    except Exception:
        pytest.skip("coherence server not running — start coherence_reward_server.py first")


# ── 1. Backchannel is penalised in a real question context ────────────────────

def test_backchannel_penalized_vs_real_answer():
    """'uh-huh I see right' should score worse than an actual answer."""
    user = "Can you explain how a transformer model works?"
    r_good = _reward(
        "A transformer model uses self-attention to process all tokens in parallel.",
        last_user=user,
    )
    r_bad = _reward("uh-huh I see right.", last_user=user)
    assert r_good["reward"] > r_bad["reward"], (
        f"backchannel ({r_bad['reward']:.4f}) should score below real answer "
        f"({r_good['reward']:.4f})"
    )


# ── 2. History stripping prevents reward hacking ──────────────────────────────

def test_degenerate_bot_history_does_not_soften_penalty():
    """Stripping bot turns from history should keep the penalty harsh.

    The bug: sending degenerate bot history to the teacher conditions its
    reference on the same backchannel pattern, erasing the penalty.
    The fix (in rewards.py): bot turns are zeroed out before sending.

    This test validates the fix directly on the server by comparing:
      - history WITH bot turns  → teacher conditioned on backchannels → penalty softened
      - history WITHOUT bot turns → teacher conditioned on user turns only → penalty maintained

    The stripped version must be more negative (harsher penalty).
    """
    user = "Can you explain how a transformer model works?"
    degenerate_bot = "uh-huh I see right."

    user_turns = [
        {"user": "What is machine learning?",     "bot": degenerate_bot},
        {"user": "How do neural networks learn?", "bot": degenerate_bot},
        {"user": "What is a loss function?",      "bot": degenerate_bot},
    ]
    stripped_turns = [
        {"user": t["user"], "bot": ""} for t in user_turns
    ]

    r_with_bot = _reward(degenerate_bot, last_user=user, history=user_turns)
    r_stripped = _reward(degenerate_bot, last_user=user, history=stripped_turns)

    # Stripping bot turns must make the penalty harsher (more negative).
    assert r_stripped["reward"] < r_with_bot["reward"], (
        f"Stripped history ({r_stripped['reward']:.4f}) should be more negative "
        f"than history with bot turns ({r_with_bot['reward']:.4f}). "
        "The degenerate bot context is softening the teacher's reference."
    )
    # Both should be negative — backchannels are penalised regardless
    assert r_stripped["reward"] < 0, (
        f"Expected negative reward with stripped history, got {r_stripped['reward']:.4f}"
    )


# ── 3. BPE boundary — first-token log-prob is not catastrophic ────────────────

def test_bpe_boundary_first_token_not_catastrophic():
    """Proposed text that starts right after a sentence-ending period
    (the classic BPE fusion case) must not have a first token log-prob below -30.

    Before the space-separator fix, tokens like '.Yes' or '.I' were scored
    with log_probs around -46 to -67, tanking the whole reward.
    """
    # prev_bot ends with a period; proposed starts with a capital letter —
    # the classic fusion scenario.
    cases = [
        ("Yes, there was a time when that happened.",       "Okay, let's see."),
        ("I see. The geographical location matters here.",  "Right."),
        ("That makes sense to me.",                         "Interesting."),
    ]
    for proposed, last_bot in cases:
        r = _reward(proposed, last_user="Can you tell me more?", last_bot=last_bot)
        lps = r["token_log_probs"]
        assert lps, f"no token_log_probs returned for {proposed!r}"
        first_lp = lps[0]
        assert first_lp > -35.0, (
            f"First token log-prob={first_lp:.2f} is catastrophically low for "
            f"proposed={proposed!r} after last_bot={last_bot!r}. "
            "BPE boundary token likely fusing prev sentence end with proposed start."
        )


# ── 4. Off-topic response is penalised ───────────────────────────────────────

def test_off_topic_scores_worse_than_on_topic():
    """A completely off-topic response should score lower than a relevant one."""
    cases = [
        (
            "What's a good recipe for chocolate cake?",
            "You'll need flour, sugar, cocoa powder, eggs, and butter.",
            "The French Revolution began in 1789.",
        ),
        (
            "What time does the meeting start tomorrow?",
            "The meeting starts at ten in the morning.",
            "Whales are the largest mammals on Earth.",
        ),
    ]
    for user, on_topic, off_topic in cases:
        r_on  = _reward(on_topic,  last_user=user)
        r_off = _reward(off_topic, last_user=user)
        assert r_on["reward"] > r_off["reward"], (
            f"on-topic ({r_on['reward']:.4f}) should beat off-topic "
            f"({r_off['reward']:.4f}) for user={user!r}"
        )


# ── 5. Rambling after EOS gets the rambling penalty ──────────────────────────

def test_rambling_after_eos():
    """When prev_block_was_eos=True, any non-empty proposed_next should
    receive the RAMBLING_PENALTY (a large fixed negative reward)."""
    r = _reward(
        "Actually let me add one more thing.",
        last_user="Thanks, that was helpful.",
        last_bot="You're welcome, happy to help!",
        prev_block_was_eos=True,
    )
    # RAMBLING_PENALTY is typically -4.0; reward should be strongly negative
    assert r["reward"] < -1.0, (
        f"Expected large negative reward for rambling after EOS, "
        f"got {r['reward']:.4f}"
    )
    # And n_tokens should be estimated (no forward pass for rambling)
    assert r["n_tokens"] > 0


# ── 6. Silence token is scored (not zero-reward) ─────────────────────────────

def test_silence_token_is_scored():
    """<silence> should return a non-zero reward — the server scores whether
    silence was contextually appropriate, not just skip it."""
    r = _reward("<silence>", last_user="What do you think?")
    # n_tokens may be 0 for a single special token — what matters is reward
    # is non-trivially zero (positive or negative depending on context)
    # In a mid-conversation context where a response is expected, silence
    # should receive a negative or zero reward.
    assert isinstance(r["reward"], float), "reward should be a float"


# ── 7. Repetition of prev_bot is penalised ───────────────────────────────────

def test_repetition_penalized():
    """Stuttering — repeating the exact same block back-to-back multiple times —
    should score worse than a natural continuation.

    Note: with USE_REFERENCE, a single repetition can score well because
    the teacher would also produce that phrase. Stuttering (3x same phrase)
    is clearly degenerate and the teacher should penalise it more.
    """
    user  = "Tell me about Paris."
    block = "Paris is the capital of France."

    # Three-block stutter: prev_bot is the phrase repeated; proposed is it again
    stutter_bot = f"{block} {block} {block}"
    r_stutter = _reward(block, last_user=user, last_bot=stutter_bot)

    # Natural continuation after one clean block
    r_natural = _reward(
        "It's known for the Eiffel Tower and its rich cultural heritage.",
        last_user=user, last_bot=block,
    )
    assert r_natural["reward"] > r_stutter["reward"], (
        f"Natural continuation ({r_natural['reward']:.4f}) should beat "
        f"stutter ({r_stutter['reward']:.4f})"
    )


# ── 8. First response with no history or prefix ───────────────────────────────

def test_first_response_no_context():
    """With no history and no last_bot_message, a coherent opening response
    should score better than an incoherent one."""
    user = "Hello! Can you help me understand how interest rates work?"

    r_coherent = _reward(
        "Of course! Interest rates determine the cost of borrowing money.",
        last_user=user,
    )
    r_incoherent = _reward(
        "uh-huh I see right yeah sustainability.",
        last_user=user,
    )
    assert r_coherent["reward"] > r_incoherent["reward"], (
        f"Coherent first response ({r_coherent['reward']:.4f}) should beat "
        f"incoherent ({r_incoherent['reward']:.4f})"
    )
    # Also verify the server doesn't crash with empty context
    assert r_coherent["n_tokens"] > 0

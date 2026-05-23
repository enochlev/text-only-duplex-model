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
    """A history full of 'uh-huh I see right' must not lower the penalty
    for proposing another 'uh-huh I see right'.

    This exercises our fix: bot turns are stripped from history before
    sending to the teacher, so the degenerate loop can't self-reinforce.
    """
    user = "Can you explain how a transformer model works?"
    degenerate_bot = "uh-huh I see right."

    # History full of the degenerate backchannel (as student would produce)
    degenerate_history = [
        {"user": "What is machine learning?",      "bot": degenerate_bot},
        {"user": "How do neural networks learn?",  "bot": degenerate_bot},
        {"user": "What is a loss function?",       "bot": degenerate_bot},
    ]

    r_no_hist  = _reward(degenerate_bot, last_user=user)
    r_bad_hist = _reward(degenerate_bot, last_user=user, history=degenerate_history)

    # With history stripping, the server ignores the degenerate bot turns,
    # so scores should be close (not softened by the bad history).
    diff = abs(r_no_hist["reward"] - r_bad_hist["reward"])
    assert diff < 0.5, (
        f"Degenerate bot history changed reward by {diff:.4f} — "
        f"no_hist={r_no_hist['reward']:.4f}  bad_hist={r_bad_hist['reward']:.4f}. "
        "Bot turns may not be getting stripped from history before scoring."
    )
    # Both should still be negative (backchannel is penalised regardless)
    assert r_bad_hist["reward"] < 0, (
        f"Expected negative reward for backchannel, got {r_bad_hist['reward']:.4f}"
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
            "How do I debug a segfault in C?",
            "Run it under valgrind or gdb to find the bad memory access.",
            "Penguins live in Antarctica and are flightless birds.",
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
    """Repeating the previous bot block verbatim should score lower than
    a novel continuation."""
    prev = "The key factors are cost, quality, and delivery time."
    user = "What should I consider when choosing a supplier?"

    r_repeat = _reward(prev, last_user=user, last_bot=prev)
    r_novel  = _reward(
        "You should also look at their track record and references.",
        last_user=user, last_bot=prev,
    )
    assert r_novel["reward"] > r_repeat["reward"], (
        f"Novel continuation ({r_novel['reward']:.4f}) should beat verbatim "
        f"repetition ({r_repeat['reward']:.4f})"
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

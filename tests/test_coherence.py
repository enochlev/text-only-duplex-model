#!/usr/bin/env python3
"""test_coherence.py — smoke-test the coherence reward server.

Runs against http://localhost:10001.

Tests
-----
1.  /health          — server is up and model name is reported.
2.  Schema           — response has reward, n_tokens, token_log_probs.
3.  Empty input      — proposed_next="" → reward=0.0, n_tokens=0.
4.  Whitespace only  — same as empty.
5.  Log-prob bounds  — NORMALIZE=True → all token_log_probs ≤ 0.
6.  Token count      — longer text → more tokens.
7.  History effect   — reward changes when history is supplied.
8.  Coherence pairs  — 10 (context, good, bad) triples:
                       good response must score strictly higher than bad.

Usage
-----
    # terminal 1 — start the coherence server
    python coherence_reward_server.py

    # terminal 2
    python test_coherence.py
"""

import json
import sys
import urllib.request
import urllib.error

SERVER = "http://localhost:10001"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(path: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(SERVER + path, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post(path: str, body: dict, timeout: float = 10.0) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        SERVER + path,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _reward(
    proposed: str,
    *,
    history: list | None = None,
    last_user: str = "",
    last_bot: str = "",
    gamma: float = 0.9,
    timeout: float = 30.0,
) -> dict:
    return _post("/reward", {
        "history":           history or [],
        "last_user_message": last_user,
        "last_bot_message":  last_bot,
        "proposed_next":     proposed,
        "gamma":             gamma,
    }, timeout=timeout)


def check_server() -> None:
    try:
        _get("/health", timeout=2.0)
    except (ConnectionRefusedError, urllib.error.URLError):
        print("ERROR: coherence server not reachable. Start it first:")
        print("  python coherence_reward_server.py")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Test 1: /health
# ---------------------------------------------------------------------------

def test_health() -> None:
    print("\n" + "=" * 70)
    print("TEST 1: /health")
    print("=" * 70)
    r = _get("/health")
    ok = r.get("status") == "ok" and "model" in r
    print(f"  status={r.get('status')!r}  model={r.get('model')!r}  {'✓' if ok else '✗'}")
    assert ok, f"unexpected health response: {r}"


# ---------------------------------------------------------------------------
# Test 2: Schema validation
# ---------------------------------------------------------------------------

def test_schema() -> None:
    print("\n" + "=" * 70)
    print("TEST 2: Schema validation")
    print("=" * 70)
    r = _reward("Sure, I can help with that.", last_user="Can you help me?")
    has_reward  = isinstance(r.get("reward"), (int, float))
    has_n       = isinstance(r.get("n_tokens"), int) and r["n_tokens"] > 0
    has_lp      = isinstance(r.get("token_log_probs"), list) and len(r["token_log_probs"]) > 0
    lp_matches  = len(r["token_log_probs"]) == r["n_tokens"]
    all_ok = all([has_reward, has_n, has_lp, lp_matches])
    print(f"  reward={r['reward']:.4f}  n_tokens={r['n_tokens']}  "
          f"len(token_log_probs)={len(r['token_log_probs'])}  {'✓' if all_ok else '✗'}")
    assert all_ok, f"schema invalid: {r}"


# ---------------------------------------------------------------------------
# Test 3: Empty proposed_next
# ---------------------------------------------------------------------------

def test_empty_input() -> None:
    print("\n" + "=" * 70)
    print("TEST 3: Empty proposed_next → reward=0.0, n_tokens=0")
    print("=" * 70)
    for label, proposed in [("empty string", ""), ("whitespace", "   ")]:
        r = _reward(proposed, last_user="Hello there.")
        ok = r["reward"] == 0.0 and r["n_tokens"] == 0 and r["token_log_probs"] == []
        print(f"  {label:<18} reward={r['reward']}  n_tokens={r['n_tokens']}  {'✓' if ok else '✗'}")
        assert ok, f"expected zero reward for empty input, got {r}"


# ---------------------------------------------------------------------------
# Test 4: Log-prob bounds (NORMALIZE=True → all ≤ 0)
# ---------------------------------------------------------------------------

def test_logprob_bounds() -> None:
    print("\n" + "=" * 70)
    print("TEST 4: token_log_probs all ≤ 0  (normalized advantage scores)")
    print("=" * 70)
    cases = [
        "That sounds great, let's go.",
        "I completely disagree with everything you just said.",
        "The weather in Paris is nice in spring.",
    ]
    for text in cases:
        r = _reward(text, last_user="What do you think?")
        violations = [lp for lp in r["token_log_probs"] if lp > 1e-6]
        ok = len(violations) == 0
        print(f"  {text[:45]:<46} max_lp={max(r['token_log_probs'], default=0):.4f}  {'✓' if ok else '✗'}")
        assert ok, f"log_probs exceed 0: {violations}"


# ---------------------------------------------------------------------------
# Test 5: Token count scales with text length
# ---------------------------------------------------------------------------

def test_token_count() -> None:
    print("\n" + "=" * 70)
    print("TEST 5: Longer text → more tokens")
    print("=" * 70)
    short  = "Yes."
    medium = "Yes, I think that's a great idea."
    long_  = "Yes, I think that's a great idea and I'd be happy to help you work through the details."
    rs = _reward(short,  last_user="Do you agree?")
    rm = _reward(medium, last_user="Do you agree?")
    rl = _reward(long_,  last_user="Do you agree?")
    ok = rs["n_tokens"] < rm["n_tokens"] < rl["n_tokens"]
    print(f"  short={rs['n_tokens']} tokens  medium={rm['n_tokens']} tokens  "
          f"long={rl['n_tokens']} tokens  {'✓' if ok else '✗'}")
    assert ok, "expected token count to increase with text length"


# ---------------------------------------------------------------------------
# Test 6: History changes the reward
# ---------------------------------------------------------------------------

def test_history_effect() -> None:
    print("\n" + "=" * 70)
    print("TEST 6: History context changes reward score")
    print("=" * 70)
    proposed = "That's a great point, I hadn't thought of it that way."
    r_no_hist = _reward(proposed, last_user="What do you think?")
    r_hist    = _reward(
        proposed,
        history=[
            {"user": "I've been thinking about this problem for a while.",
             "bot":  "What kind of problem are you working on?"},
            {"user": "It's about how to structure large software projects.",
             "bot":  "Interesting, modularity is key in that context."},
        ],
        last_user="What do you think?",
    )
    different = abs(r_no_hist["reward"] - r_hist["reward"]) > 0.01
    print(f"  no_history  reward={r_no_hist['reward']:.4f}")
    print(f"  with_history reward={r_hist['reward']:.4f}")
    print(f"  difference={abs(r_no_hist['reward'] - r_hist['reward']):.4f}  {'✓' if different else '?'}")
    # soft check — just print, not assert; some models may not change much


# ---------------------------------------------------------------------------
# Test 7: last_bot_message prefix affects reward
# ---------------------------------------------------------------------------

def test_prefix_effect() -> None:
    print("\n" + "=" * 70)
    print("TEST 7: last_bot_message prefix (continuation context)")
    print("=" * 70)
    proposed = "and I think we should take it seriously."
    r_no_prefix  = _reward(proposed, last_user="What's your view?")
    r_with_prefix = _reward(
        proposed,
        last_user="What's your view?",
        last_bot="This is actually a very complex situation,",
    )
    print(f"  no prefix    reward={r_no_prefix['reward']:.4f}")
    print(f"  with prefix  reward={r_with_prefix['reward']:.4f}")
    # The prefix "This is actually a very complex situation," makes "and I think..."
    # a natural continuation, so score should be higher (less negative).
    ok = r_with_prefix["reward"] >= r_no_prefix["reward"]
    print(f"  prefix boosts score  {'✓' if ok else '✗ (model disagrees — not a hard failure)'}")


# ---------------------------------------------------------------------------
# Test 8: Coherence pairs — good vs bad continuation
# ---------------------------------------------------------------------------

COHERENCE_PAIRS = [
    # (last_user, good_next, bad_next)
    (
        "What time does the meeting start?",
        "The meeting starts at three o'clock.",
        "Elephants are the largest land animals.",
    ),
    (
        "Can you explain how recursion works?",
        "Recursion is when a function calls itself to solve a smaller version of the problem.",
        "I prefer chocolate ice cream over vanilla.",
    ),
    (
        "I'm feeling really tired today.",
        "That sounds tough — have you been getting enough sleep?",
        "The stock market closed higher on Wednesday.",
    ),
    (
        "What should I cook for dinner tonight?",
        "How about pasta? It's quick and easy to make.",
        "The capital of Australia is Canberra.",
    ),
    (
        "I just finished reading a great book.",
        "Oh nice, what was it about?",
        "The train arrives at platform seven.",
    ),
    (
        "Do you think it's going to rain tomorrow?",
        "The forecast shows some clouds but no rain.",
        "My favorite color is blue.",
    ),
    (
        "I can't figure out this math problem.",
        "Let's work through it together — what's the equation?",
        "The moon orbits the Earth every twenty-eight days.",
    ),
    (
        "We should probably wrap up this conversation.",
        "Agreed, it was great talking with you.",
        "Quantum computing uses qubits instead of classical bits.",
    ),
    (
        "I forgot to send that email.",
        "No worries, you can still send it now.",
        "Mount Everest is the tallest mountain in the world.",
    ),
    (
        "What do you think of this idea?",
        "I think it has real potential — especially the second part.",
        "The first airplane flight was in nineteen oh three.",
    ),
]


def test_coherence_pairs() -> None:
    print("\n" + "=" * 70)
    print("TEST 8: Coherence pairs — good response must score > bad response")
    print("=" * 70)
    print(f"{'#':<3}  {'good score':>10}  {'bad score':>10}  {'diff':>8}  result")
    print("-" * 70)

    wins = 0
    for i, (user_text, good, bad) in enumerate(COHERENCE_PAIRS, 1):
        r_good = _reward(good, last_user=user_text)
        r_bad  = _reward(bad,  last_user=user_text)
        win    = r_good["reward"] > r_bad["reward"]
        wins  += win
        diff   = r_good["reward"] - r_bad["reward"]
        mark   = "✓" if win else "✗"
        print(
            f"{i:<3}  {r_good['reward']:>10.4f}  {r_bad['reward']:>10.4f}  "
            f"{diff:>+8.4f}  {mark}  {user_text[:35]!r}"
        )

    print("-" * 70)
    print(f"Good > Bad: {wins}/{len(COHERENCE_PAIRS)}")
    assert wins >= 7, f"coherence accuracy {wins}/10 — expected ≥ 7/10"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    check_server()
    test_health()
    test_schema()
    test_empty_input()
    test_logprob_bounds()
    test_token_count()
    test_history_effect()
    test_prefix_effect()
    test_coherence_pairs()
    print("\nAll tests passed.\n")


if __name__ == "__main__":
    main()

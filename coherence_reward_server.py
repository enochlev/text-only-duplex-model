"""coherence_reward_server.py — scores a proposed next block against a teacher LLM.

The teacher (e.g. Qwen2.5-Instruct) sees the full conversation history and the
last bot block as a prefix, then the server measures how likely the proposed next
block is under the teacher's conditional distribution.

Reward = discounted sum of per-token log-probs:
    R = Σ_i  γ^i · log P_teacher(token_i | prefix + tokens_0..i-1)

Earlier tokens dominate (γ < 1), so:
  - boundary noise from later tokens is dampened
  - cascading tokenization errors don't blow up the signal
  - a single <eos> continuation (model decides to stop) scores fine

Only called when the previous block was non-idle. Hard idle penalty lives
in trainer/rewards.py (idle_penalty).
"""

from __future__ import annotations

import argparse
import os
import contextlib
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

# ── config ────────────────────────────────────────────────────────────────────

MODEL_NAME = os.getenv("COHERENCE_MODEL", "Qwen/Qwen3-0.6B")
GAMMA      = float(os.getenv("COHERENCE_GAMMA", "0.9"))
PORT       = int(os.getenv("COHERENCE_PORT", "10001"))
# Normalize reward by subtracting the greedy log-prob at each position.
# reward_i = log P(proposed_i) - log P(greedy_i), always in (-inf, 0].
# 0 = matched teacher's best choice; more negative = teacher preferred something else.
NORMALIZE  = True

# Tokens the chat template appends after the assistant turn (stripped when
# locating the proposed block inside the full token sequence).
_END_STRINGS = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "</s>"]

# ── server ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Coherence Reward Server")

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None     = None
_end_ids: set[int]                   = set()
_debug: bool                         = False


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _tokenizer, _end_ids

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    device = os.getenv("COHERENCE_DEVICE", "auto")
    dtype  = torch.float32 if device == "cpu" else torch.float16
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    _model.eval()

    # collect all plausible end-of-turn token ids for this tokenizer
    if _tokenizer.eos_token_id is not None:
        _end_ids.add(_tokenizer.eos_token_id)
    for s in _END_STRINGS:
        tid = _tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != _tokenizer.unk_token_id:
            _end_ids.add(tid)

    print(f"[coherence_reward_server] loaded {MODEL_NAME}  end_ids={_end_ids}")

    # Warn if the chat template injects <think> blocks into assistant turns — Qwen3
    # does this even with enable_thinking=False, which biases reward scores because
    # the model evaluates proposed tokens in a post-thinking context.
    _test = _tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    if "<think>" in _test:
        print(
            "[WARNING] This model's chat template inserts <think> blocks into assistant turns "
            "even with enable_thinking=False. Reward scores will be slightly biased — the model "
            "evaluates proposed_next in a post-thinking context. Not fixable without patching "
            "the chat template or switching to a non-thinking model variant."
        )

    yield


app.router.lifespan_context = lifespan

# ── prompt template ───────────────────────────────────────────────────────────

_SYSTEM_TMPL = """\
You are a full-duplex conversational agent. You respond to the user appropriately \
and continue naturally through interruptions. Responses are thoughtful yet concise.

Conversation so far:
{history}

Continue the response naturally. End your turn when the thought is complete.\
"""


def _fmt_history(blocks: list[dict]) -> str:
    lines: list[str] = []
    for i, b in enumerate(blocks, 1):
        parts = [f"<block{i}>"]
        if b.get("user"):
            parts.append(f"<user> {b['user']}")
        if b.get("bot"):
            parts.append(f"<bot> {b['bot']}")
        lines.append(" ".join(parts))
    return "\n".join(lines) if lines else "(none)"


# ── schema ────────────────────────────────────────────────────────────────────

class Block(BaseModel):
    user: str = ""
    bot:  str = ""


class RewardRequest(BaseModel):
    history:           list[Block]  # all blocks before the current one
    last_user_message: str          # most recent user turn
    last_bot_message:  str          # text the model already emitted this turn (prefix)
    proposed_next:     str          # new block text to score
    gamma:             float = GAMMA


class RewardResponse(BaseModel):
    reward:          float
    n_tokens:        int
    token_log_probs: list[float]


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/reward", response_model=RewardResponse)
async def compute_reward(req: RewardRequest) -> RewardResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "model not loaded")

    if not req.proposed_next.strip():
        return RewardResponse(reward=0.0, n_tokens=0, token_log_probs=[])

    history_str    = _fmt_history([b.model_dump() for b in req.history])
    system_content = _SYSTEM_TMPL.format(history=history_str)

    messages_full = [
        {"role": "system",    "content": system_content},
        {"role": "user",      "content": req.last_user_message},
        {"role": "assistant", "content": req.last_bot_message + req.proposed_next},
    ]

    full_ids: torch.Tensor = _tokenizer.apply_chat_template(
        messages_full,
        add_generation_prompt=False,
        return_tensors="pt",
        enable_thinking=False,
    ).to(_model.device)  # [1, seq_len]
    if _debug:
        rendered = _tokenizer.apply_chat_template(
            messages_full,
            add_generation_prompt=False,
            tokenize=False,
            enable_thinking=False,
        )
        sep = "─" * 60
        print(f"\n{sep}  PROMPT  {sep}")
        print(rendered)
        print(f"{sep}  TOKEN COUNT: {full_ids.shape[1]}  {sep}\n")

    # ── locate proposed_next tokens inside full_ids ───────────────────────────
    #
    # Encode proposed_next alone (no special tokens) to count its tokens.
    # The chat template appends end-of-turn tokens after the assistant message;
    # strip those to find where the proposed block ends.

    proposed_only_ids = _tokenizer.encode(req.proposed_next, add_special_tokens=False)
    n_proposed = len(proposed_only_ids)

    if n_proposed == 0:
        return RewardResponse(reward=0.0, n_tokens=0, token_log_probs=[])

    seq_len = full_ids.shape[1]

    # TODO: verify <idle> ↔ <eos> alignment.
    #
    # When the trained model emits an <idle> block (no text, silence decision),
    # the coherence server is NOT called (caller enforces this). But the mirror
    # question is: when Qwen's teacher distribution would naturally emit <eos>
    # here (it has nothing more to say), does that correctly map to a high-reward
    # signal for the model choosing silence?
    #
    # Need to:
    #   1. Find the exact token id(s) Qwen uses for "stop now" — could be
    #      <|im_end|>, <|endoftext|>, or a model-specific stop token.
    #   2. Confirm _end_ids covers all of them (log _end_ids at startup and
    #      check against tokenizer.special_tokens_map).
    #   3. Run a test case where proposed_next == "" (idle) and verify the
    #      server returns 0.0 without erroring, AND that the caller-side hard
    #      penalty for idle is applied instead.
    #   4. Run a test case where proposed_next contains only whitespace/newline
    #      (degenerate output) — check n_tokens and reward are sensible.
    #   5. Confirm that when Qwen would emit <eos> mid-proposed block (i.e.
    #      the block is longer than what Qwen thinks is natural), the log-prob
    #      of those trailing tokens drops sharply — that's the reward signal
    #      telling the model "you should have stopped earlier."
    #
    # Strip trailing end-of-turn tokens and newlines the template appended.
    # Qwen3 emits <|im_end|>\n (newline after the end token), so we must also
    # strip \n or the end token never gets reached.
    _nl_ids = set(_tokenizer.encode("\n", add_special_tokens=False))
    _strippable = _end_ids | _nl_ids
    end_pos = seq_len
    while end_pos > 0 and full_ids[0, end_pos - 1].item() in _strippable:
        end_pos -= 1

    proposed_start = end_pos - n_proposed
    if proposed_start <= 0:
        raise HTTPException(400, "proposed_next longer than full sequence — check inputs")

    # ── forward pass ──────────────────────────────────────────────────────────

    with torch.no_grad():
        logits = _model(full_ids).logits[0]          # [seq_len, vocab]
    log_probs = F.log_softmax(logits, dim=-1)        # [seq_len, vocab]

    # logits[p-1] predicts the token at position p
    token_log_probs: list[float] = []
    for p in range(proposed_start, end_pos):
        tok_id  = full_ids[0, p].item()
        lp      = log_probs[p - 1, tok_id].item()
        if NORMALIZE:
            lp -= log_probs[p - 1].max().item()  # subtract greedy log-prob → advantage in (-inf, 0]
        token_log_probs.append(lp)

    reward = sum(req.gamma ** i * lp for i, lp in enumerate(token_log_probs))

    if _debug:
        sep = "─" * 60
        print(f"{sep}  SCORES  {sep}")
        for i, (lp, tok_pos) in enumerate(zip(token_log_probs, range(proposed_start, end_pos))):
            tok_str = _tokenizer.decode([full_ids[0, tok_pos].item()])
            print(f"  [{i:2d}]  γ^i={req.gamma**i:.3f}  logp={lp:7.3f}  weighted={req.gamma**i * lp:7.3f}  token={tok_str!r}")
        print(f"  reward = {reward:.4f}  (n_tokens={len(token_log_probs)})")
        print(f"{sep}\n")

    return RewardResponse(
        reward=reward,
        n_tokens=len(token_log_probs),
        token_log_probs=token_log_probs,
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="print full prompt and per-token scores on each request")
    parser.add_argument("--port",  type=int, default=PORT)
    args = parser.parse_args()

    _debug = args.debug

    uvicorn.run(
        app,  # pass object directly — string form re-imports the module, losing _debug
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


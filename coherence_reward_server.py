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

MODEL_NAME = os.getenv("COHERENCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GAMMA      = float(os.getenv("COHERENCE_GAMMA", "0.9"))
PORT       = int(os.getenv("COHERENCE_PORT", "8001"))

# Tokens the chat template appends after the assistant turn (stripped when
# locating the proposed block inside the full token sequence).
_END_STRINGS = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "</s>"]

# ── server ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Coherence Reward Server")

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None     = None
_end_ids: set[int]                   = set()


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _tokenizer, _end_ids

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
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

    # ── build the full token sequence ─────────────────────────────────────────
    #
    # Chat layout:
    #   system  → full history + instruction
    #   user    → last_user_message
    #   assistant → last_bot_message + proposed_next   ← we score this suffix
    #
    # We pass the assistant content as a single string so the tokenizer sees
    # last_bot_message + proposed_next as one unit (no BPE boundary artifact).

    messages_full = [
        {"role": "system",    "content": system_content},
        {"role": "user",      "content": req.last_user_message},
        {"role": "assistant", "content": req.last_bot_message + req.proposed_next},
    ]

    full_ids: torch.Tensor = _tokenizer.apply_chat_template(
        messages_full,
        add_generation_prompt=False,
        return_tensors="pt",
    ).to(_model.device)  # [1, seq_len]

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
    # walk backward past any end-of-turn tokens the template appended
    end_pos = seq_len
    while end_pos > 0 and full_ids[0, end_pos - 1].item() in _end_ids:
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
        tok_id = full_ids[0, p].item()
        token_log_probs.append(log_probs[p - 1, tok_id].item())

    reward = sum(req.gamma ** i * lp for i, lp in enumerate(token_log_probs))

    return RewardResponse(
        reward=reward,
        n_tokens=len(token_log_probs),
        token_log_probs=token_log_probs,
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "coherence_reward_server:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )

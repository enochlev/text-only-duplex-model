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
import math
import os
import contextlib
from typing import Optional

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

MODEL_NAME      = os.getenv("COHERENCE_MODEL", "Qwen/Qwen3.5-4B")  # or "Qwen/Qwen2.5-Instruct"
MODEL_NAME      = os.getenv("COHERENCE_MODEL", "Qwen/Qwen3-14B-FP8")  # --- IGNORE ---

_IS_QUANTIZED = any(tag in MODEL_NAME.lower() for tag in ("awq", "fp8", "gptq", "gguf"))
if _IS_QUANTIZED:
    # Quantized models need CUDA; FP8 also needs compressed-tensors: pip install compressed-tensors
    assert torch.cuda.is_available(), f"Quantized model '{MODEL_NAME}' requires CUDA, but no GPU is available"

GAMMA           = float(os.getenv("COHERENCE_GAMMA", "0.9"))
PORT            = int(os.getenv("COHERENCE_PORT", "10001"))
# Normalize reward by subtracting the greedy log-prob at each position.
# reward_i = log P(proposed_i) - log P(greedy_i), always in (-inf, 0].
# 0 = matched teacher's best choice; more negative = teacher preferred something else.
NORMALIZE       = True
# Scale factor applied after per-token averaging. Brings the mean per-token
# advantage (typically -4..0) into the same range as the other reward signals
# (capped at ~-0.5). Set via COHERENCE_SCALE env var to tune without code changes.
REWARD_SCALE    = float(os.getenv("COHERENCE_SCALE", "0.2"))

# Nonlinear shaping: map the "good zone" [SHAPE_THRESHOLD, 0] → [SHAPE_OUT_LO, SHAPE_OUT_HI]
# using a convex (t^2) curve so near-perfect responses receive the steepest positive gradient.
# Scores below SHAPE_THRESHOLD pass through unchanged (remain negative).
_SHAPE_THRESHOLD = float(os.getenv("COHERENCE_SHAPE_THRESHOLD", "-0.25"))
_SHAPE_OUT_LO    = float(os.getenv("COHERENCE_SHAPE_OUT_LO",    "0.25"))
_SHAPE_OUT_HI    = float(os.getenv("COHERENCE_SHAPE_OUT_HI",    "1.0"))

# Tokens the chat template appends after the assistant turn (stripped when
# locating the proposed block inside the full token sequence).
_END_STRINGS = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "</s>"]


def _shape_reward(r: float) -> float:
    if r >= _SHAPE_THRESHOLD:
        t = (r - _SHAPE_THRESHOLD) / (0.0 - _SHAPE_THRESHOLD)  # 0 at threshold, 1 at 0
        return _SHAPE_OUT_LO + (_SHAPE_OUT_HI - _SHAPE_OUT_LO) * t ** 2
    return r


# ── server ────────────────────────────────────────────────────────────────────

app = FastAPI(title="Coherence Reward Server")

_model: AutoModelForCausalLM | None = None
_tokenizer: AutoTokenizer | None     = None
_end_ids: set[int]                   = set()


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"



@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    global _model, _tokenizer, _end_ids

    device = _detect_device()
    print(f"[coherence] device: {device}  quantized={_IS_QUANTIZED}")

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if _IS_QUANTIZED:
        # quantization config in the model drives dtype; sdpa works on CUDA/Ada FP8 cores
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    else:
        dtype      = torch.float32 if device == "cpu" else torch.bfloat16
        device_map = "auto" if device == "cuda" else device
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="eager",  # MPS can't handle GQA in fused kernels
        )
    _model.eval()

    # collect all plausible end-of-turn token ids for this tokenizer
    if _tokenizer.eos_token_id is not None:
        _end_ids.add(_tokenizer.eos_token_id)
    for s in _END_STRINGS:
        tid = _tokenizer.convert_tokens_to_ids(s)
        if tid is not None and tid != _tokenizer.unk_token_id:
            _end_ids.add(tid)

    yield


app.router.lifespan_context = lifespan

# ── prompt template ───────────────────────────────────────────────────────────

_SYSTEM_TMPL = """\
You are a full-duplex conversational agent. You respond to the user appropriately \
and continue naturally through interruptions. Responses are thoughtful yet concise.
Avoid repeats and converse naturally. If conversation history is suffering from repeats, do the best you can to break the cycle and move the conversation forward consince, to the point, natural and helpful.

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
    n_proposed_tokens: Optional[int] = None  # pre-computed token count; skips server scan when set


class RewardResponse(BaseModel):
    reward:          float
    n_tokens:        int
    token_log_probs: list[float]


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "eos_token_ids": sorted(_end_ids)}


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

    _tmpl_out = _tokenizer.apply_chat_template(
        messages_full,
        add_generation_prompt=False,
        return_tensors="pt",
        enable_thinking=False,
    )
    # newer transformers returns BatchEncoding; older returns a plain tensor
    full_ids: torch.Tensor = (
        _tmpl_out.input_ids if hasattr(_tmpl_out, "input_ids") else _tmpl_out
    ).to(_model.device)  # [1, seq_len]

    # ── locate proposed_next tokens inside full_ids ───────────────────────────
    #
    # We cannot encode proposed_next in isolation OR build a prefix template —
    # BPE merges tokens across BOTH the last_bot_message/proposed_next boundary
    # AND the proposed_next/end-token boundary. The only reliable method is to
    # scan backwards through full_ids itself: find the shortest suffix of the
    # pre-end-token region that decodes to proposed_next.

    seq_len = full_ids.shape[1]

    # Strip trailing end-of-turn tokens and newlines the template appended.
    # Qwen3 emits <|im_end|>\n (newline after the end token), so we must also
    # strip \n or the end token never gets reached.
    _nl_ids = set(_tokenizer.encode("\n", add_special_tokens=False))
    _strippable = _end_ids | _nl_ids
    end_pos = seq_len
    while end_pos > 0 and full_ids[0, end_pos - 1].item() in _strippable:
        end_pos -= 1

    if req.n_proposed_tokens is not None and req.n_proposed_tokens > 0:
        n_proposed    = req.n_proposed_tokens
        proposed_start = end_pos - n_proposed
    else:
        # Fallback scan for callers that don't pre-compute the token count.
        # BPE tokenizers are context-sensitive at boundaries, so isolated
        # encoding of proposed_next can differ from in-context tokenization.
        proposed_text  = req.proposed_next.strip()
        proposed_start = None
        n_proposed     = 0
        max_scan       = min(end_pos, max(len(req.proposed_next) // 2 + 20, 40))
        for n in range(1, max_scan + 1):
            start = end_pos - n
            if start < 0:
                break
            if _tokenizer.decode(full_ids[0, start:end_pos].tolist()).strip() == proposed_text:
                proposed_start = start
                n_proposed     = n
                break
        if proposed_start is None:
            ids = _tokenizer.encode(req.proposed_next, add_special_tokens=False)
            n_proposed     = len(ids)
            proposed_start = end_pos - n_proposed
        decoded = _tokenizer.decode(full_ids[0, proposed_start:end_pos].tolist())
        if decoded.strip() != req.proposed_next.strip():
            print(
                f"[coherence SERVER] token-alignment mismatch!  "
                f"proposed_next={req.proposed_next!r}  "
                f"decoded_from_ids={decoded!r}  "
                f"seq_len={seq_len}  n_proposed={n_proposed}  "
                f"proposed_start={proposed_start}  end_pos={end_pos}"
            )

    if n_proposed <= 0:
        return RewardResponse(reward=0.0, n_tokens=0, token_log_probs=[])
    if proposed_start <= 0:
        raise HTTPException(400, "proposed_next longer than full sequence — check inputs")

    # ── forward pass ──────────────────────────────────────────────────────────

    with torch.inference_mode():
        full_logits = _model(full_ids).logits[0]          # [seq_len, vocab]
        # Slice only the positions that predict proposed tokens before log_softmax.
        # logits[p-1] predicts token at position p, so we need [proposed_start-1 : end_pos-1].
        needed_logits = full_logits[proposed_start - 1 : end_pos - 1].clone()  # [n_proposed, vocab]
        del full_logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_probs = F.log_softmax(needed_logits, dim=-1)  # [n_proposed, vocab]
    del needed_logits

    token_log_probs: list[float] = []
    for i, p in enumerate(range(proposed_start, end_pos)):
        tok_id  = full_ids[0, p].item()
        lp      = log_probs[i, tok_id].item()
        if NORMALIZE:
            lp -= log_probs[i].max().item()  # subtract greedy log-prob → advantage in (-inf, 0]
        token_log_probs.append(lp)

    n_tokens = len(token_log_probs)
    raw = sum(req.gamma ** i * lp for i, lp in enumerate(token_log_probs))
    # Divide by sqrt(n_tokens) so longer consistently-good responses score higher than
    # short fillers, then apply nonlinear shaping to reward near-perfect scores positively.
    reward = _shape_reward((raw / math.sqrt(n_tokens)) * REWARD_SCALE) if n_tokens > 0 else 0.0

    return RewardResponse(
        reward=reward,
        n_tokens=n_tokens,
        token_log_probs=token_log_probs,
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


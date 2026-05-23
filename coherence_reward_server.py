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
MODEL_NAME      = os.getenv("COHERENCE_MODEL", "Qwen/Qwen3-8B-FP8")  # --- IGNORE ---

_IS_QUANTIZED = any(tag in MODEL_NAME.lower() for tag in ("awq", "fp8", "gptq", "gguf"))
if _IS_QUANTIZED:
    # Quantized models need CUDA; FP8 also needs compressed-tensors: pip install compressed-tensors
    assert torch.cuda.is_available(), f"Quantized model '{MODEL_NAME}' requires CUDA, but no GPU is available"

GAMMA           = float(os.getenv("COHERENCE_GAMMA", "0.9"))
PORT            = int(os.getenv("COHERENCE_PORT", "10001"))
# Normalize reward by subtracting the greedy log-prob at each position.
# reward_i = log P(proposed_i) - log P(greedy_i), always in (-inf, 0].
# 0 = matched teacher's best choice; more negative = teacher preferred something else.
# Superseded by USE_REFERENCE when that mode is active.
NORMALIZE       = True
# Scale factor applied after per-token averaging. Brings the mean per-token
# advantage (typically -4..0) into the same range as the other reward signals
# (capped at ~-0.5). Set via COHERENCE_SCALE env var to tune without code changes.
REWARD_SCALE    = float(os.getenv("COHERENCE_SCALE", "0.2"))
# Penalty applied when both teacher and student ended the previous block (mutual EOS)
# but the student talks again in the next block (rambling after turn completion).
# Bypasses the teacher forward pass — the answer is structurally wrong regardless of content.
RAMBLING_PENALTY = float(os.getenv("COHERENCE_RAMBLING_PENALTY", "-4.0"))
# Reference-based scoring: greedily generate a teacher reference response, then
# reward = normalized_score(proposed) - normalized_score(reference).
# Supersedes per-token NORMALIZE when enabled.
USE_REFERENCE        = os.getenv("COHERENCE_USE_REFERENCE", "1") == "1"
REFERENCE_MAX_TOKENS = int(os.getenv("COHERENCE_REFERENCE_MAX_TOKENS", "16"))
# per-token normalization is superseded by reference-based normalization
_NORMALIZE_PER_TOKEN = NORMALIZE and not USE_REFERENCE

# Nonlinear shaping: map the "good zone" [SHAPE_THRESHOLD, 0] → [SHAPE_OUT_LO, SHAPE_OUT_HI]
# using a convex (t^2) curve so near-perfect responses receive the steepest positive gradient.
# Scores below SHAPE_THRESHOLD pass through unchanged (remain negative).
_SHAPE_THRESHOLD = float(os.getenv("COHERENCE_SHAPE_THRESHOLD", "-0.25"))
_SHAPE_OUT_LO    = float(os.getenv("COHERENCE_SHAPE_OUT_LO",    "0.25"))
_SHAPE_OUT_HI    = float(os.getenv("COHERENCE_SHAPE_OUT_HI",    "1.0"))

# Tokens the chat template appends after the assistant turn (stripped when
# locating the proposed block inside the full token sequence).
_END_STRINGS = ["<|im_end|>", "<|endoftext|>", "<|end_of_text|>", "</s>"]

# Punctuation that marks a sentence as complete. Anything else gets " ..."
# appended to signal the turn was cut mid-stream.
_SENTENCE_TERMINALS = frozenset('.?!:;')


def _frame_user_msg(text: str) -> str:
    """Add syntactic distance between the user's live speech and the assistant turn.

    Without framing, the teacher may treat the assistant's opening tokens as a
    direct sentence-completion of the user's trailing words (e.g. user ends with
    "machine learning", assistant opens with "and..." → teacher assigns p≈1).

    Two defenses:
    - '[User]:' prefix — role label inside the content breaks the open-sentence
      surface so the teacher reads this as a discrete turn to respond TO.
    - ' ...' suffix   — appended when the message has no terminal punctuation,
      signalling the user is still speaking (duplex: user and bot overlap in time).
    """
    stripped = text.rstrip()
    if stripped and stripped[-1] not in _SENTENCE_TERMINALS:
        stripped += " ..."
    return f"[User]: {stripped}"


def _shape_reward(r: float) -> float:
    if r >= _SHAPE_THRESHOLD:
        t = (r - _SHAPE_THRESHOLD) / (0.0 - _SHAPE_THRESHOLD)  # 0 at threshold, 1 at 0
        t = min(t, 1.0)  # clamp: r > 0 (proposed beats greedy) still caps at SHAPE_OUT_HI
        return _SHAPE_OUT_LO + (_SHAPE_OUT_HI - _SHAPE_OUT_LO) * t ** 2
    return r


def _score_tokens(
    logits_slice: torch.Tensor,
    token_ids: list[int],
    gamma: float,
    normalize_per_token: bool,
) -> tuple[float, list[float]]:
    """Score a token sequence given pre-sliced logits [n, vocab]."""
    log_probs = F.log_softmax(logits_slice, dim=-1)
    token_lps: list[float] = []
    for i, tok_id in enumerate(token_ids):
        lp = log_probs[i, tok_id].item()
        if normalize_per_token:
            lp -= log_probs[i].max().item()
        token_lps.append(lp)
    n = len(token_lps)
    raw = sum(gamma ** i * lp for i, lp in enumerate(token_lps))
    score = (raw / math.sqrt(n)) * REWARD_SCALE if n > 0 else 0.0
    return score, token_lps


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
    # True when the previous block ended with mutual EOS (teacher proposed EOS, student agreed).
    # Any non-empty proposed_next in this state is rambling and receives RAMBLING_PENALTY directly.
    prev_block_was_eos: bool = False
    # True when the student emitted EOS before the block boundary (ended early).
    # The EOS token is included in scoring so the teacher's low P(EOS) at that position
    # contributes a natural penalty — symmetric counterpart to RAMBLING_PENALTY.
    student_emitted_eos: bool = False


class RewardResponse(BaseModel):
    reward:          float
    n_tokens:        int
    token_log_probs: list[float]
    greedy_text:     str = ""


# ── endpoint ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL_NAME, "eos_token_ids": sorted(_end_ids)}


def _find_text_start_by_hint(
    full_ids: torch.Tensor, text_end: int, target: str, hint: int
) -> Optional[int]:
    """Try hint, hint+1, hint-1 offsets back from text_end. Returns start pos or None.

    BPE can merge the last char of the prefix with the first char of proposed_next,
    shifting the in-context token count by ±1 vs. the standalone-encoded hint.
    """
    for k in (hint, hint + 1, hint - 1):
        if k <= 0:
            continue
        start = text_end - k
        if start >= 0 and _tokenizer.decode(full_ids[0, start:text_end].tolist()).strip() == target:
            return start
    return None


def _find_text_start_by_scan(
    full_ids: torch.Tensor, text_end: int, target: str
) -> Optional[int]:
    """Scan backwards from text_end up to max_scan tokens. Returns start pos or None."""
    max_scan = min(text_end, max(len(target) // 2 + 20, 40))
    for k in range(1, max_scan + 1):
        start = text_end - k
        if start < 0:
            break
        if _tokenizer.decode(full_ids[0, start:text_end].tolist()).strip() == target:
            return start
    return None


def _locate_tokens(
    full_ids: torch.Tensor,
    text: str,
    n_tokens_hint: Optional[int],
    nl_ids: set[int],
    include_terminal_eos: bool = False,
) -> tuple[int, int, int]:
    """Return (text_start, end_pos, n_tokens) for `text` within full_ids.

    When include_terminal_eos=True the scoring window is extended by one EOS
    token past the text, so the teacher's P(EOS) at that position is scored.
    This penalises premature endings when the teacher would have continued.
    """
    seq_len    = full_ids.shape[1]
    strippable = _end_ids | nl_ids
    target     = text.strip()

    # Strip trailing EOS/NL to find where text tokens actually end.
    text_end = seq_len
    while text_end > 0 and full_ids[0, text_end - 1].item() in strippable:
        text_end -= 1

    # Optionally extend by one EOS for premature-EOS scoring.
    eos_appended = (
        include_terminal_eos
        and text_end < seq_len
        and full_ids[0, text_end].item() in _end_ids
    )
    end_pos = text_end + (1 if eos_appended else 0)

    # Locate where the proposed text starts, fastest path first.
    text_start: Optional[int] = None
    if n_tokens_hint is not None and n_tokens_hint > 0:
        text_start = _find_text_start_by_hint(full_ids, text_end, target, n_tokens_hint)
    if text_start is None:
        text_start = _find_text_start_by_scan(full_ids, text_end, target)
    if text_start is None:
        # Last resort: trust standalone encoding length.
        n_standalone = len(_tokenizer.encode(text, add_special_tokens=False))
        text_start   = end_pos - n_standalone - (1 if eos_appended else 0)

    n_tokens = end_pos - text_start
    decoded  = _tokenizer.decode(full_ids[0, text_start:text_end].tolist())
    if decoded.strip() != target:
        print(
            f"[coherence SERVER] token-alignment mismatch!  "
            f"text={text!r}  decoded={decoded!r}  "
            f"seq_len={seq_len}  n={n_tokens}  start={text_start}  end={end_pos}"
        )
    return text_start, end_pos, n_tokens


def _build_assistant_ids(
    system_content: str,
    last_user: str,
    assistant_text: str,
) -> torch.Tensor:
    """Tokenize a complete chat turn and return input_ids on the model device."""
    messages = [
        {"role": "system",    "content": system_content},
        {"role": "user",      "content": _frame_user_msg(last_user)},
        {"role": "assistant", "content": assistant_text},
    ]
    out = _tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, return_tensors="pt", enable_thinking=False
    )
    ids: torch.Tensor = out.input_ids if hasattr(out, "input_ids") else out
    return ids.to(_model.device)


def _forward_score(
    system_content: str,
    last_user: str,
    assistant_text: str,
    text_to_score: str,
    n_hint: Optional[int],
    gamma: float,
    nl_ids: set[int],
    normalize: bool,
    include_terminal_eos: bool = False,
) -> tuple[float, list[float], int]:
    """Score `text_to_score` inside a full chat sequence. Returns (score, lps, n_tokens)."""
    full_ids = _build_assistant_ids(system_content, last_user, assistant_text)
    t_start, end_pos, n_tokens = _locate_tokens(full_ids, text_to_score, n_hint, nl_ids, include_terminal_eos)
    if n_tokens <= 0 or t_start <= 0:
        return 0.0, [], 0
    with torch.inference_mode():
        logits = _model(full_ids).logits[0][t_start - 1 : end_pos - 1].clone()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    token_ids = full_ids[0, t_start:end_pos].tolist()
    score, lps = _score_tokens(logits, token_ids, gamma, normalize)
    return score, lps, n_tokens


def _generate_greedy_reference(
    system_content: str, last_user: str
) -> tuple[str, list[int]]:
    """Greedily generate a teacher reference response. Returns (text, token_ids)."""
    messages_prefix = [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": _frame_user_msg(last_user)},
    ]
    pfx_out = _tokenizer.apply_chat_template(
        messages_prefix, add_generation_prompt=True, return_tensors="pt", enable_thinking=False
    )
    prefix_ids: torch.Tensor = (
        pfx_out.input_ids if hasattr(pfx_out, "input_ids") else pfx_out
    ).to(_model.device)
    with torch.inference_mode():
        gen_out = _model.generate(
            prefix_ids,
            max_new_tokens=REFERENCE_MAX_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_ids = gen_out[0, prefix_ids.shape[1]:].tolist()
    while new_ids and new_ids[-1] in _end_ids:
        new_ids.pop()
    text = _tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, new_ids


@app.post("/reward", responses={503: {"description": "Model not loaded"}})
async def compute_reward(req: RewardRequest) -> RewardResponse:
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "model not loaded")

    if not req.proposed_next.strip():
        return RewardResponse(reward=0.0, n_tokens=0, token_log_probs=[])

    # Rambling: previous block was a mutual EOS (turn complete) but model talks again.
    # The ideal teacher output here is a single EOS — no forward pass needed.
    if req.prev_block_was_eos:
        n_est = req.n_proposed_tokens or len(_tokenizer.encode(req.proposed_next, add_special_tokens=False))
        print(
            f"[coherence RAMBLING] penalty={RAMBLING_PENALTY}  n_tokens={n_est}"
            f"  proposed={req.proposed_next!r:.40}"
        )
        return RewardResponse(reward=RAMBLING_PENALTY, n_tokens=n_est, token_log_probs=[])

    history_str    = _fmt_history([b.model_dump() for b in req.history])
    system_content = _SYSTEM_TMPL.format(history=history_str)
    nl_ids         = set(_tokenizer.encode("\n", add_special_tokens=False))

    greedy_text    = ""
    greedy_new_ids: list[int] = []
    if USE_REFERENCE:
        greedy_text, greedy_new_ids = _generate_greedy_reference(
            system_content, req.last_user_message
        )

    # Space separator prevents BPE from fusing the last char of last_bot_message
    # with the first char of proposed_next into a single boundary-spanning token.
    sep = " " if req.last_bot_message else ""
    assistant_full = req.last_bot_message + sep + req.proposed_next
    proposed_score, token_log_probs, n_proposed = _forward_score(
        system_content, req.last_user_message, assistant_full,
        req.proposed_next, req.n_proposed_tokens, req.gamma, nl_ids, _NORMALIZE_PER_TOKEN,
        include_terminal_eos=req.student_emitted_eos,
    )

    if n_proposed <= 0:
        return RewardResponse(reward=0.0, n_tokens=0, token_log_probs=[])

    greedy_score = 0.0
    if USE_REFERENCE and greedy_new_ids:
        greedy_assistant = req.last_bot_message + sep + greedy_text
        greedy_score, _, _ = _forward_score(
            system_content, req.last_user_message, greedy_assistant,
            greedy_text, None, req.gamma, nl_ids, normalize=False,
        )

    raw = proposed_score - greedy_score if (USE_REFERENCE and greedy_text) else proposed_score
    return RewardResponse(
        reward=_shape_reward(raw),
        n_tokens=n_proposed,
        token_log_probs=token_log_probs,
        greedy_text=greedy_text,
    )


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")


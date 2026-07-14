"""
full_duplex.py — Real-time duplex audio agent.

Exports:
- DuplexAudioAgent : audio-in / audio-out duplex agent (Kokoro TTS + Parakeet ASR)
- DuplexAudioBlock : finalized conversation block dataclass
- llm_generate     : thin OpenAI wrapper used by DuplexAudioAgent
"""

import contextlib
import inspect
import io
import math
import os
import queue
import re
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Optional

import numpy as np
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from groq import Groq


load_dotenv()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

_llm_client: Optional[OpenAI] = None



def _get_llm_client() -> OpenAI:
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required")
        _llm_client = OpenAI(api_key=api_key)
    return _llm_client

def _get_groq_client() -> Groq:
    global _llm_client
    if _llm_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required")
        _llm_client = Groq(api_key=api_key)
    return _llm_client


def llm_generate_openai(system_prompt: str, user_message: str) -> str:
    client = _get_llm_client()
    model = os.getenv("OPENAI_MODEL", "gpt-5.4-nano").strip() or "gpt-5.4-nano"
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=[{"role": "user", "content": user_message}],
        reasoning={"effort": "none"},
        max_output_tokens=60,
    )
    return response.output[0].content[0].text.strip()

GROQ_MODEL_CONFIGS = [
    {"model": "qwen/qwen3-32b", "params": {"reasoning_effort": "none"}},
    {"model": "llama-3.1-8b-instant", "params": {}},
    #{"model": "openai/gpt-oss-20b", "params": {"reasoning_effort": "low"}},
    {"model": "llama-3.3-70b-versatile", "params": {}},
    {"model": "meta-llama/llama-4-scout-17b-16e-instruct", "params": {}},
]
_next_model_index = 0
_last_used_model: str = ""

# Serving generation cap (single-shot). Raised 80→200 on 2026-06-23: with a large
# enough window the model emits its ENTIRE turn (ending in eos </s>) in one call,
# so there is no truncation seam.
# At ~300 tok/s a 200-token cap is ~0.7s of gen — masked by block quantization.
# This is a CEILING, not a target: short answers should still end early on eos.
# If you see "[llm] WARNING: hit max_tokens" a lot, eos isn't firing (the model is
# rambling to the cap) — dial this back and investigate the eos/turn-end behavior.
_SERVE_MAX_TOKENS = 200

# --- ADHOC LATENCY HACK (2026-06-22) — early-emit / pull-tick-forward ---------
# Normally a response that finishes mid-block isn't committed/played until the
# next fixed block tick, so first-audio latency is up to ~2 blocks (≤3.4s @1.7s).
# When enabled, poll() fires the tick the instant a fresh response is fully
# buffered (during a silence/user gap, never mid-response), cutting ~1 block of
# latency. NOTE: this is the "speak sooner" lever (P2-A). Known trade-off: with an
# untrained base it can amplify false-starts (the base re-answers on every partial
# ASR revision), and it can wedge the reply block into a still-settling utterance
# (ASR re-timestamps trailing words across blocks), occasionally splitting the user's
# sentence in the transcript.
# ENABLED 2026-06-24: user prefers the ~1-block-lower first-audio latency and accepts
# that occasional block-wedge artifact. Flip to False for strict tick-paced emission.
_ENABLE_EARLY_EMIT = False

# --- EXPERIMENT (2026-07-11) — strip punctuation from user turns in the prompt ---
# Hypothesis: the model overfits to punctuation as an end-of-turn signal — ASR only
# emits terminal punctuation ("."/"?"/"!") once it has decided the user finished
# speaking, so "punctuation present → safe to talk" is an easy shortcut that does NOT
# transfer (real ASR punctuation is late/unreliable). We want the model to infer
# turn completion from the WORDS/context, not from a punctuation token.
#
# We strip punctuation from the user text ONLY at prompt-render time (_format_timeblocks),
# not from the stored block.user_text. This means:
#   • the model never sees punctuation in user turns — in BOTH training and prod, since
#     both build the prompt through _format_timeblocks (symmetric by construction);
#   • the training harness keeps its punctuated ground truth intact, so _user_finished_in
#     (epsilon-greedy gate) and _classify_text_change ("punctuation"-only churn suppression)
#     still work — those are legitimate harness supervision, not model-visible leakage.
# Apostrophes inside contractions (don't, it's) are kept — they are part of the word,
# not a turn/clause boundary. Flip to False to restore punctuated prompts.
# 2026-07-12: set False. Live testing showed the base MiniCPM-duplex relies on terminal
# punctuation as its turn-end ("user finished → speak") cue — with the strip ON the base
# model (zero RL) went mute on 3/4 casual turns. The crutch is load-bearing AND Parakeet
# supplies punctuation timely enough in prod, so keep it. See CLAUDE.md §11 (2026-07-12).
_STRIP_USER_PUNCTUATION = False

# Drops every punctuation mark (. , ? ! ; : " ( ) - — … etc.) but preserves apostrophes
# that sit between word characters so contractions survive (don't → don't, not do nt).
_USER_PUNCT_RE = re.compile(r"[^\w\s']")
_EDGE_APOSTROPHE_RE = re.compile(r"(?<!\w)'|'(?!\w)")

# --- ASR commit strategy -------------------------------------------------------
# 2026-07-12: the legacy _run_parakeet re-transcribes the whole rolling mic window
# every block tick, so word timestamps drift and already-emitted words get rewritten
# as more right-context arrives. Offline measurement on real recordings: ~28 retroactive
# word rewrites per 10s (asr_compare.py). Those retro-edits are what the Fix A/B/C/D
# heuristics fight, and the ones they miss trip a context flush that cuts the bot off
# mid-response. _run_parakeet_monotonic replaces the block-granular mutable-window with a
# single TIME cursor: any block whose audio ends before (window_end - _ASR_RIGHT_CONTEXT_S)
# is FROZEN and never rewritten (structural, not heuristic), so re-segmentation of settled
# audio can no longer flush. Words within the right-context tail stay provisional (live).
# Word timestamps are unchanged — we still use model.transcribe(timestamps=True) — so block
# assignment and the source→covered reward attribution are untouched. Override at launch with
# ASR_MONOTONIC_COMMIT=1 (and optionally ASR_RIGHT_CONTEXT_S=<sec>) to A/B live vs the legacy path.
_ASR_MONOTONIC_COMMIT = os.environ.get("ASR_MONOTONIC_COMMIT", "0") == "1"
_ASR_RIGHT_CONTEXT_S = float(os.environ.get("ASR_RIGHT_CONTEXT_S", "2.0"))  # revisable tail (s) before freeze


def strip_user_punctuation(text: str) -> str:
    """Remove punctuation from a user-turn string for prompt rendering.

    Keeps letters/digits/whitespace and word-internal apostrophes; collapses the
    whitespace left behind. Returns text unchanged when the experiment toggle is off.
    """
    if not _STRIP_USER_PUNCTUATION or not text:
        return text
    text = _USER_PUNCT_RE.sub(" ", text)
    text = _EDGE_APOSTROPHE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def llm_generate_groq(system_prompt: str, user_message: str) -> str:
    global _next_model_index, _last_used_model
    client = _get_groq_client()
    config = GROQ_MODEL_CONFIGS[_next_model_index]
    model = config["model"]
    _last_used_model = model
    _next_model_index = (_next_model_index + 1) % len(GROQ_MODEL_CONFIGS)
    request_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 80,
        "temperature": 0.0,
    }
    request_params.update(config["params"])
    response = client.chat.completions.create(**request_params)
    return response.choices[0].message.content.strip()


def llm_stream_groq(system_prompt: str, user_message: str) -> Generator[str, None, None]:
    """Streaming Groq call — yields token strings one at a time."""
    global _next_model_index, _last_used_model
    client = _get_groq_client()
    config = GROQ_MODEL_CONFIGS[_next_model_index]
    model = config["model"]
    _last_used_model = model
    _next_model_index = (_next_model_index + 1) % len(GROQ_MODEL_CONFIGS)
    request_params: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "max_tokens": 60,
        "temperature": 0.0,
        "stream": True,
    }
    request_params.update(config["params"])
    for chunk in client.chat.completions.create(**request_params):
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def local_generate(system_prompt: str, user_message: str) -> str:
    """Call the locally-hosted trained model via OpenAI-compatible /v1/chat/completions.

    Matches the training prompt format exactly:
      - system role  → rendered full-duplex.jinja2 content
      - user role    → <user>TEXT<AI>TEXT</s>...<user>CURRENT<AI> block format
    Both assembled as chat messages so the model's chat template is applied
    server-side, identical to how llm_generate_train formats them during RL.

    This talks to the OpenAI-compatible model backend on VLLM_PORT, not to the
    duplex websocket server on SERVER_PORT.
    """
    import requests
    url = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
    payload = {
        "model": "text-duplex",
        # Match cpm_generate's larger window so a full turn lands in one call (the
        # chat template's eos terminates it; <用户> isn't used in this format, so no
        # ban needed here).
        "max_tokens": _SERVE_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "temperature": 0.0,
    }
    response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
    return response.json()["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# MiniCPM-duplex generate function
# ---------------------------------------------------------------------------

# Matches completed blocks: <user>U<AI>A</s>  and the trailing open block: <user>U<AI>
_CPM_BLOCK_RE = re.compile(r"<user>(.*?)<AI>(.*?)(?:</s>|$)", re.DOTALL)

# MiniCPM-duplex has no system-prompt slot — the format is a raw <用户>/<AI> turn
# sequence (see _build_cpm_prompt). To still give the model a persona we inject one
# leading turn pair (a user instruction + a short assistant ack) at the front of
# every prompt. Both deployment (cpm_generate) and RL training (llm_generate_train,
# which imports _build_cpm_prompt) go through this single builder, so the persona is
# applied identically — tuned and non-tuned, train and inference stay in sync.
_CPM_SYSTEM_PROMPT = (
    "You are a warm, natural voice assistant in a live spoken conversation. "
    "Everything you say is spoken aloud, so reply in one or two short sentences — "
    "no lists, bullet points, step-by-step breakdowns, or markdown. Lead with the "
    "answer (don't restate the question), skip greetings and self-description unless "
    "asked, and don't over-explain — give the direct answer and stop. Vary your "
    "wording so you never sound scripted."
)
# The ack establishes the <用户>/<AI> turn format AND primes the assistant's voice,
# so keep it a neutral instruction-acknowledgement, NOT a user-facing greeting —
# any phrase here (e.g. the old "happy to help!") gets echoed back into replies.
_CPM_SYSTEM_PREFIX = f"<用户>{_CPM_SYSTEM_PROMPT}<AI>Got it."


def _build_cpm_prompt(user_message: str) -> str:
    """Convert block-format history to MiniCPM <用户>/<AI> turn format.

    Input (block format):  <user>U<AI>A</s>...<user>CURRENT<AI>
    Output (CPM format):   <用户>USER_TURN<AI>AI_TURN<用户>USER_TURN<AI>

    Consecutive user blocks are aggregated into one user turn; consecutive AI
    blocks into one AI turn. <idle> blocks are skipped. The output always ends
    with <AI> so the model continues the assistant's response.

    Every prompt is prefixed with _CPM_SYSTEM_PREFIX — a faked "system prompt"
    (MiniCPM has no system slot) injected as a leading instruction turn — so the
    persona conditions both deployment and training identically.
    """
    result = ""
    u_buf: list = []
    a_buf: list = []

    for raw_u, raw_a in _CPM_BLOCK_RE.findall(user_message):
        u = raw_u.strip()
        a = raw_a.strip()
        if u == "<idle>":
            u = ""

        if u:
            if a_buf:
                result += " ".join(a_buf)
                a_buf = []
            u_buf.append(u)

        if a:
            if u_buf:
                result += "<用户>" + " ".join(u_buf) + "<AI>"
                u_buf = []
            a_buf.append(a)

    # Flush final user buffer — the trailing open <user>CURRENT<AI> block
    if u_buf:
        result += "<用户>" + " ".join(u_buf) + "<AI>"
    elif a_buf:
        # AI was speaking last with no new user turn (shouldn't happen normally)
        result += " ".join(a_buf) + "<用户><AI>"

    return _CPM_SYSTEM_PREFIX + (result or "<用户><AI>")


def cpm_generate(system_prompt: str, user_message: str) -> str:
    """Call MiniCPM-duplex via vLLM /v1/completions on VLLM_PORT.

    system_prompt is ignored — MiniCPM uses raw <用户>/<AI> format with no system turn.
    user_message is in the standard block format; reformatted internally to CPM format.

    The model's own turn ends on eos (</s>), which fires reliably. stop=["<用户>"]
    catches the rare case where the base role-flips into a hallucinated user turn
    WITHOUT first emitting eos — a string match on the decoded output, so it's robust
    to however <用户> tokenizes. (logit_bias bans on < / > were tried and dropped: a
    char's token id is context-dependent, so banning the standalone ids did not
    suppress the in-context symbols. Removing symbols for TTS is a post-processing job.)

    This talks to the OpenAI-compatible MiniCPM backend on VLLM_PORT, not to the
    duplex websocket server on SERVER_PORT.
    """
    import requests
    prompt = _build_cpm_prompt(user_message)
    url = f"http://localhost:{VLLM_PORT}/v1/completions"
    payload = {
        "model": "cpm-text-duplex",
        "prompt": prompt,
        "max_tokens": _SERVE_MAX_TOKENS,
        "temperature": 0.0,
        "stop": ["<用户>"],          # robust string-match guard against role-flips
    }
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=10)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    # finish_reason: "length" → cut off by the token cap; "stop" → turn complete.
    finish_reason = choice.get("finish_reason")
    # stop_reason disambiguates a "stop": None → eos (</s>, model's turn-complete
    # signal); a value → it hit the "<用户>" stop string (role-flip caught).
    stop_reason = choice.get("stop_reason")
    text = choice["text"].strip()
    if finish_reason == "stop":
        ended_on = "eos" if stop_reason is None else f"stop:{stop_reason!r}"
    elif finish_reason == "length":
        ended_on = "length(hit max_tokens, no eos)"
        print(f"[llm] WARNING: hit max_tokens={_SERVE_MAX_TOKENS} without eos — turn not self-terminating")
    else:
        ended_on = str(finish_reason)
    print(f"[llm] ended_on={ended_on} words={len(text.split())}")
    return text


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Two ports configure the whole duplex stack:
#   SERVER_PORT — the websocket server that the UI clients connect to.
#   VLLM_PORT   — the OpenAI-compatible model backend that local_generate() /
#                 cpm_generate() call.
# Both have env-var defaults and can be overridden by server.py's CLI args
# (which assign back to full_duplex.VLLM_PORT before the agent is created).
SERVER_PORT = int(os.getenv("SERVER_PORT", "8998"))
VLLM_PORT = int(os.getenv("VLLM_PORT", "8555"))

DEFAULT_WPM       = 150
DEFAULT_BLOCK_S   = 2.0
TTS_SAMPLE_RATE   = 24000    # Kokoro PCM output rate (also fallback for silence blocks)
ASR_SAMPLE_RATE   = 16000    # Parakeet expects 16 kHz; incoming mic is resampled to this
MAX_MIC_BLOCKS    = 8       # rolling mic audio window (last N agent blocks)
MAX_AUDIO_QUEUE_S = MAX_MIC_BLOCKS * DEFAULT_BLOCK_S   # ≈ 20s safety cap
MAX_HISTORY_S     = 600.0    # prune blocks older than 10 minutes

# LLM response trimming: find first sentence-end at or after this char position.
# ~160 chars ≈ 40 tokens for English text (60-token max ÷ 1.5 chars/token avg).
_SENT_END_RE            = re.compile(r'[.!?]')
_LLM_TRIM_SEARCH_START  = 160

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) or "."
_template_env = Environment(loader=FileSystemLoader(_MODULE_DIR))
_prompt_template = _template_env.get_template("full-duplex.jinja2")

# ---------------------------------------------------------------------------
# Kokoro TTS config (replaces Piper)
# ---------------------------------------------------------------------------
# Kokoro-82M runs on the existing torch install (GPU via device='cuda'); there is
# no per-file model path like Piper's .onnx. The default voice id is kept under the
# historical name TTS_MODEL so trainer/data-ingestion code that threads
# `tts_model=...` keeps working unchanged — the value is now a Kokoro voice id
# (e.g. "af_heart"), not a filesystem path.
KOKORO_REPO = os.getenv("KOKORO_REPO", "hexgrad/Kokoro-82M")
KOKORO_LANG = os.getenv("KOKORO_LANG", "a")          # 'a' = American English, 'b' = British
TTS_MODEL   = os.getenv("KOKORO_VOICE", "af_heart")  # default Kokoro voice id

# ---------------------------------------------------------------------------
# ASR model (lazy-loaded on first use)
# ---------------------------------------------------------------------------

import logging as _logging
import os as _os
import threading as _threading
# NeMo's own logger is named "nemo_logger" (not "nemo"/"nemo_logging"), so the
# original two names never matched it — "nemo_logger" is the one that silences the
# recurring Lhotse-dataloader warnings emitted on every transcribe().
for _noisy in ("nemo", "nemo_logger", "nemo_logging", "lightning", "pytorch_lightning", "omegaconf"):
    _logging.getLogger(_noisy).setLevel(_logging.ERROR)
_os.environ.setdefault("TQDM_DISABLE", "1")  # suppress NeMo progress bars


# setLevel(ERROR) alone does NOT silence these — NeMo's transcribe() re-raises its
# logging verbosity mid-call, so the recurring Lhotse-dataloader warnings print on
# every transcribe. A record filter runs on the handler regardless of the active
# level, so it survives the re-raise. Attached in _silence_nemo_logging() once the
# nemo logger exists (after the ASR model loads).
_NEMO_SPAM_NEEDLES = (
    "ignored by Lhotse dataloader",
    "non-tarred dataset",
    "pretokenize=True",
    "setup_training_data",
    "setup_validation_data",
)


class _NemoSpamFilter(_logging.Filter):
    def filter(self, record: "_logging.LogRecord") -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return not any(n in msg for n in _NEMO_SPAM_NEEDLES)


def _silence_nemo_logging() -> None:
    """Drop NeMo's recurring Lhotse-dataloader warnings (see _NEMO_SPAM_NEEDLES).

    Attaches the filter to every candidate logger AND its handlers so the spam is
    dropped even when transcribe() bumps the level back up mid-call. Best-effort:
    NeMo's logging-object layout varies by version, so each attach is guarded.
    """
    _filt = _NemoSpamFilter()
    candidates = []
    try:
        from nemo.utils import logging as _nl
        _nl.setLevel(_logging.ERROR)
        candidates.append(getattr(_nl, "_logger", None))
    except Exception:
        pass
    candidates += [_logging.getLogger("nemo_logger"), _logging.getLogger("nemo")]
    for _lg in candidates:
        if _lg is None:
            continue
        try:
            _lg.addFilter(_filt)
            for _h in list(getattr(_lg, "handlers", [])):
                _h.addFilter(_filt)
        except Exception:
            pass

_asr_model = None
_kokoro_cache = {}   # (lang_code, device) -> KPipeline (heavy model, loaded once)
# NeMo's transcribe() is not thread-safe (freeze/unfreeze race).
# All background ASR threads must hold this lock before calling transcribe().
_asr_lock = _threading.Lock()
_kokoro_lock = _threading.Lock()   # guards one-time KPipeline cache load
# All sessions share ONE KPipeline instance (per lang/device), so concurrent
# sessions (e.g. parallel inference_FDB workers) must serialize the synthesis
# forward pass — a torch module is not safe under concurrent forward calls.
_tts_lock = _threading.Lock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AsrAlignedWord:
    text: str
    end_time: float


@dataclass
class AsrTimestampWindow:
    window_id: str
    start_ts: float
    end_ts: float
    words: List[AsrAlignedWord]
    revision: int = 0
    frozen: bool = False


@dataclass
class DuplexAudioBlock:
    """Used by DuplexAudioAgent."""
    block_id: str
    start_ts: float
    end_ts: float
    user_text: str = ""
    assistant_text: str = ""
    assistant_text_stale: bool = False
    mic_audio: Optional[np.ndarray] = None   # mic PCM resampled to ASR_SAMPLE_RATE (float32)
    tts_audio: Optional[np.ndarray] = None   # TTS PCM (int16) at tts_sr
    tts_sr: int = TTS_SAMPLE_RATE
    tts_latency_s: Optional[float] = None    # wall-clock seconds to synthesize TTS
    asr_latency_s: Optional[float] = None    # wall-clock seconds for the ASR pass that covered this block
    llm_latency_s: Optional[float] = None    # wall-clock seconds for accepted LLM generation
    total_latency_s: Optional[float] = None  # ASR-start → audio-ready latency for first reply chunk
    asr_started_perf_s: Optional[float] = None
    response_source_block_id: Optional[str] = None
    timeline_start_ts: Optional[float] = None
    timeline_end_ts: Optional[float] = None
    lead_silence_s: float = 0.0


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    # numpy/scipy only — NO torch. torch.from_numpy here would intermittently crash the live
    # server with "aten::lift_fresh has no schema" once the torch dispatcher degraded after many
    # per-session model reloads (2026-07-14). scipy.resample_poly is anti-aliased (good for the
    # 48k/44.1k→16k mic downsample into Parakeet); np.interp is a dependency-free fallback.
    if from_sr == to_sr:
        return audio
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) == 0:
        return audio
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(from_sr), int(to_sr))
        return resample_poly(audio, to_sr // g, from_sr // g).astype(np.float32)
    except Exception:
        n_out = max(1, int(round(len(audio) * to_sr / from_sr)))
        x_out = np.linspace(0, len(audio), n_out, endpoint=False)
        return np.interp(x_out, np.arange(len(audio)), audio).astype(np.float32)


def resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class _KokoroVoice:
    """Opaque voice handle: a loaded Kokoro KPipeline plus the voice id to use.

    Mirrors the role the old Piper `voice` object played, so callers
    (kokoro_synthesize, DuplexAudioAgent) treat it identically.
    """
    __slots__ = ("pipeline", "voice", "lang_code")

    def __init__(self, pipeline: Any, voice: str, lang_code: str) -> None:
        self.pipeline = pipeline
        self.voice = voice
        self.lang_code = lang_code


def preload_kokoro_voice(
    tts_model: str = TTS_MODEL,
    device: Optional[str] = None,
    lang_code: str = KOKORO_LANG,
) -> "_KokoroVoice":
    """Load (and cache) a Kokoro pipeline and return a voice handle.

    `tts_model` is the Kokoro voice id (e.g. "af_heart"); the parameter name is
    retained for drop-in compatibility with the old Piper signature. The heavy
    KPipeline is cached per (lang_code, device); the lightweight voice id is just
    attached to the returned handle.
    """
    resolved_device = resolve_device(device)
    voice = tts_model or TTS_MODEL
    cache_key = (lang_code, resolved_device)
    with _kokoro_lock:
        pipeline = _kokoro_cache.get(cache_key)
        if pipeline is None:
            from kokoro import KPipeline

            try:
                pipeline = KPipeline(
                    lang_code=lang_code, repo_id=KOKORO_REPO, device=resolved_device
                )
            except TypeError:
                # Older kokoro builds lack the repo_id and/or device kwarg.
                try:
                    pipeline = KPipeline(lang_code=lang_code, device=resolved_device)
                except TypeError:
                    pipeline = KPipeline(lang_code=lang_code)
            _kokoro_cache[cache_key] = pipeline
    return _KokoroVoice(pipeline, voice, lang_code)


def preload_asr_model():
    global _asr_model
    if _asr_model is None:
        import nemo.collections.asr as nemo_asr

        device = resolve_device()
        _asr_model = nemo_asr.models.ASRModel.from_pretrained(
            "nvidia/parakeet-tdt-0.6b-v2",
            map_location=device,
        )
        _asr_model.to(device)
        _silence_nemo_logging()
    return _asr_model


def preload_duplex_models(
    tts_model: str = TTS_MODEL,
    device: Optional[str] = None,
) -> None:
    resolved_device = resolve_device(device)
    preload_kokoro_voice(tts_model=tts_model, device=resolved_device)
    preload_asr_model()


def warmup_duplex_models(
    tts_model: str = TTS_MODEL,
    device: Optional[str] = None,
    min_seconds: float = 6.0,
) -> None:
    """Run dummy TTS + ASR inferences so the first real turn doesn't pay one-time
    kernel-compilation cost. Observed cold starts: Kokoro first synth ~3.2s,
    Parakeet first transcribe ~1.4s; warm calls are ~0.1s. Loops for ~min_seconds
    so every kernel path (varied audio length) is compiled before serving.
    """
    import tempfile
    import soundfile as sf

    resolved_device = resolve_device(device)
    voice = preload_kokoro_voice(tts_model=tts_model, device=resolved_device)
    model = preload_asr_model()

    phrase = "How are you doing today?"
    t0 = time.time()
    iters = 0
    while True:
        sr, pcm16 = kokoro_synthesize(voice, phrase)            # warms Kokoro kernels
        if pcm16.size:
            audio = pcm16.astype(np.float32) / 32767.0
            if sr != ASR_SAMPLE_RATE:
                audio = _resample(audio, sr, ASR_SAMPLE_RATE)
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    tmp_path = f.name
                    sf.write(f.name, audio, ASR_SAMPLE_RATE)
                with _asr_lock:
                    with contextlib.redirect_stderr(io.StringIO()):
                        model.transcribe([tmp_path], timestamps=True, verbose=False)  # warms Parakeet
            except Exception as exc:
                print(f"[boot] warmup transcribe skipped: {exc!r}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
        iters += 1
        if time.time() - t0 >= min_seconds:
            break
    print(f"[boot] warmup complete ({iters} iters, {time.time() - t0:.1f}s)")


def kokoro_synthesize(voice: Any, text: str) -> tuple:
    """Synthesize text with a loaded Kokoro voice handle.

    Returns (sample_rate, int16_array) — the same contract the old piper_synthesize
    exposed, so DuplexAudioAgent's whole-response cache + per-block slicing is
    unchanged. Kokoro yields one float32 waveform per sentence at 24 kHz; we
    concatenate them and convert to int16.
    """
    pipeline = voice.pipeline
    voice_id = voice.voice
    chunks: List[np.ndarray] = []
    # Serialize the GPU forward pass across sessions (shared pipeline). Numpy
    # post-processing below is on local data, so it stays outside the lock.
    with _tts_lock:
        for result in pipeline(text, voice=voice_id, speed=1.0):
            # KPipeline yields a Result with .audio (recent) or a (gs, ps, audio) tuple.
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)):
                audio = result[-1]
            if audio is None:
                continue
            if hasattr(audio, "detach"):          # torch.Tensor → numpy
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            if audio.size:
                chunks.append(audio)
    if not chunks:
        return TTS_SAMPLE_RATE, np.zeros(0, dtype=np.int16)
    audio = np.concatenate(chunks)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    return TTS_SAMPLE_RATE, pcm16


# ---------------------------------------------------------------------------
# DuplexAudioAgent  (audio-in / audio-out — Kokoro TTS + Parakeet ASR)
# ---------------------------------------------------------------------------

class DuplexAudioAgent:
    def __init__(
        self,
        wpm: int = DEFAULT_WPM,
        default_block_s: float = DEFAULT_BLOCK_S,
        tts_model: str = TTS_MODEL,
        device: Optional[str] = None,
        llm_generate_fn: Callable = local_generate,
        max_prompt_blocks: int = 20,
        # Injected for testing (None → use real implementations)
        tts_fn: Optional[Callable[[str], tuple]] = None,
        asr_fn: Optional[Callable] = None,
    ):
        device = resolve_device(device)

        self._n = math.ceil(wpm * default_block_s / 60)
        self._default_block_s = default_block_s
        self._device = device
        self._llm_generate_fn = llm_generate_fn
        self._max_prompt_blocks = max_prompt_blocks
        self._tts_model = tts_model

        # Conversation history
        self.blocks: List[DuplexAudioBlock] = []
        self._current_block: Optional[DuplexAudioBlock] = None

        # LLM / word queue
        self.context_version: int = 0
        self._llm_in_flight: bool = False
        self._pending_words: List[str] = []
        self._committed_words: List[str] = []
        self._pending_llm_latency_s: Optional[float] = None
        self._pending_response_asr_latency_s: Optional[float] = None
        self._pending_response_source_block_id: Optional[str] = None
        self._latest_user_source_block_id: Optional[str] = None
        self._last_accepted_response_context_version: Optional[int] = None
        # Number of committed blocks at the time of the last idle (empty) response.
        # Prevents duplicate LLM calls within the same block period when the model
        # chose silence but _last_accepted_response_context_version is not set.
        self._last_idle_call_n_blocks: int = -1
        self._last_ctx_flush_user_fingerprint: str = ""
        self.last_llm_error: Optional[str] = None
        self.last_llm_error_seq: int = 0

        # Streaming LLM state — stream thread writes to _stream_word_queue;
        # poll() drains it under DuplexSession.lock so _pending_words is single-threaded.
        self._stream_word_queue: queue.Queue = queue.Queue()
        self._stream_pending_accumulator: List[str] = []
        self._stream_generation_context_version: Optional[int] = None

        # Block timing
        self._next_block_ts: float = 0.0

        # Audio
        self._tts_fn = tts_fn
        self._kokoro_voice = None
        self._audio_queue: queue.Queue[tuple] = queue.Queue()

        # Whole-response TTS cache: the entire LLM response is synthesized in ONE
        # Kokoro call, then sliced into block-sized pieces (see _slice_response_audio)
        # so playback is one continuous waveform instead of stuttering per-fragment
        # utterances. Cleared on barge-in / ASR flush and after the response is fully
        # spoken (_reset_full_tts_cache).
        self._full_tts_sr: Optional[int] = None
        self._full_tts_audio: Optional[np.ndarray] = None
        self._full_tts_words: List[str] = []
        self._full_tts_cursor_words: int = 0
        self._full_tts_cursor_samples: int = 0

        # Logging
        self.quiet: bool = False

        # Mic ASR
        self._asr_fn = asr_fn
        self._mic_rolling: List[tuple] = []
        self._mic_current: np.ndarray = np.zeros(0, dtype=np.float32)
        self._executor = ThreadPoolExecutor(max_workers=2)

        # ASR windows
        self.asr_windows: List[AsrTimestampWindow] = []
        self.max_asr_windows: int = 20
        self.mutable_asr_windows: int = 2
        # Monotonic-commit cursor (used only when _ASR_MONOTONIC_COMMIT): absolute audio
        # time up to which user words are frozen. Advances forward only; never rewinds.
        self._asr_commit_cursor_ts: float = 0.0

        # Block timestamp tracking — carries previous block's end_ts as next start_ts
        self._block_start_ts: float = 0.0

        # Eagerly load Kokoro TTS + Parakeet ASR so first-use latency is zero.
        # Only runs when both implementations are real (neither tts_fn nor asr_fn
        # is injected). Test doubles bypass this block entirely.
        if self._tts_fn is None and self._asr_fn is None:
            print("[init] loading Kokoro TTS voice…")
            self._get_kokoro_voice()
            print("[init] Kokoro TTS ready")
            print("[init] loading Parakeet ASR model…")
            self._get_asr_model()
            print("[init] Parakeet ASR ready")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

    def _now(self) -> float:
        return time.time()

    def _get_block_by_id(self, block_id: Optional[str]) -> Optional[DuplexAudioBlock]:
        if not block_id:
            return None
        if self._current_block is not None and self._current_block.block_id == block_id:
            return self._current_block
        for block in reversed(self.blocks):
            if block.block_id == block_id:
                return block
        return None

    def _clear_pending_response_timing(self) -> None:
        self._pending_llm_latency_s = None
        self._pending_response_asr_latency_s = None
        self._pending_response_source_block_id = None

    def _clear_timeline_metadata(self, block: DuplexAudioBlock) -> None:
        block.timeline_start_ts = None
        block.timeline_end_ts = None
        block.lead_silence_s = 0.0

    def _invalidate_future_assistant_continuation(self) -> None:
        self._pending_words = []
        self._committed_words = []
        self._clear_pending_response_timing()
        # Flushed words → flushed audio: drop the whole-response synthesis and any
        # already-queued bot audio so a barge-in stops playback immediately instead
        # of finishing the now-stale response.
        self._reset_full_tts_cache()
        while True:
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def _mark_assistant_history_stale_from(self, start_index: int) -> None:
        for block in self.blocks[start_index:]:
            if block.assistant_text:
                block.assistant_text_stale = True

    def _assistant_text_for_prompt(self, block: DuplexAudioBlock) -> str:
        if block.assistant_text and not block.assistant_text_stale:
            return block.assistant_text
        return ""

    @staticmethod
    def _short_text(text: str, limit: int = 48) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def _change_tokens(self, text: str) -> List[str]:
        normalized = self._normalize(text).lower().strip()
        if not normalized:
            return []
        collapsed = re.sub(r"[^a-z0-9'\s]+", " ", normalized)
        return [token for token in collapsed.split() if token]

    def _classify_text_change(self, old: str, new: str) -> str:
        if not old:
            return "new"

        old_tokens = self._change_tokens(old)
        new_tokens = self._change_tokens(new)

        if old_tokens == new_tokens:
            return "punctuation"

        if old_tokens and new_tokens:
            shared_prefix = 0
            for old_token, new_token in zip(old_tokens, new_tokens):
                if old_token != new_token:
                    break
                shared_prefix += 1

            if shared_prefix == min(len(old_tokens), len(new_tokens)):
                if len(new_tokens) > len(old_tokens):
                    return "extension"
                return "trim"

            overlap = len(set(old_tokens) & set(new_tokens))
            smaller = min(len(set(old_tokens)), len(set(new_tokens)))
            if smaller > 0 and overlap >= max(1, smaller - 1):
                return "rephrase"

        return "topic-shift"

    def _history_summary(self, blocks: List[DuplexAudioBlock]) -> str:
        if not blocks:
            return "(empty)"

        parts = []
        for block in blocks:
            user_text = self._short_text(block.user_text) if block.user_text else "-"
            visible_assistant_text = self._assistant_text_for_prompt(block)
            assistant_text = self._short_text(visible_assistant_text) if visible_assistant_text else "-"
            if block.assistant_text and block.assistant_text_stale:
                hidden_text = self._short_text(block.assistant_text)
                parts.append(
                    f"[u={user_text!r} ai={assistant_text!r} stale_ai={hidden_text!r}]"
                )
            else:
                parts.append(f"[u={user_text!r} ai={assistant_text!r}]")
        return " | ".join(parts)

    def _get_kokoro_voice(self):
        if self._kokoro_voice is None:
            self._kokoro_voice = preload_kokoro_voice(
                tts_model=self._tts_model,
                device=self._device,
            )
        return self._kokoro_voice

    def _get_asr_model(self):
        return preload_asr_model()

    def _ensure_current_block(self, now: float) -> None:
        if self._current_block is None:
            self._current_block = DuplexAudioBlock(
                block_id=self._new_id(),
                start_ts=now,
                end_ts=now + self._default_block_s,
            )

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "--")
        )

    def _norm(self, text: str) -> str:
        return self._normalize(text).strip()

    def _queue_word_key(self, word: str) -> str:
        normalized = self._normalize(word).lower().strip()
        stripped = re.sub(r"^[^a-z0-9']+", "", normalized)
        stripped = re.sub(r"[^a-z0-9']+$", "", stripped)
        return stripped or normalized

    def _shared_suffix_prefix_len(self, left: List[str], right: List[str]) -> int:
        max_overlap = min(len(left), len(right))
        if max_overlap == 0:
            return 0

        left_keys = [self._queue_word_key(word) for word in left]
        right_keys = [self._queue_word_key(word) for word in right]
        for size in range(max_overlap, 0, -1):
            if left_keys[-size:] == right_keys[:size]:
                return size
        return 0

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def _vlog(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def _synthesize_kokoro_audio(self, voice, text: str) -> tuple[int, np.ndarray]:
        return kokoro_synthesize(voice, text)

    def _generate_tts(self, text: str) -> tuple:
        """Returns (sample_rate, audio_int16, latency_s)."""
        if self._tts_fn is not None:
            sr, arr = self._tts_fn(text)
            return sr, arr, 0.0
        t0 = time.perf_counter()
        voice = self._get_kokoro_voice()
        sr, arr = self._synthesize_kokoro_audio(voice, text)
        elapsed = time.perf_counter() - t0
        self._vlog(f"[tts] {repr(text)} → {len(arr)/sr:.2f}s audio  (synthesized in {elapsed:.3f}s)")
        return sr, arr, elapsed

    # ------------------------------------------------------------------
    # Whole-response TTS cache + per-block slicing
    # ------------------------------------------------------------------

    def _reset_full_tts_cache(self) -> None:
        self._full_tts_sr = None
        self._full_tts_audio = None
        self._full_tts_words = []
        self._full_tts_cursor_words = 0
        self._full_tts_cursor_samples = 0

    def _ensure_full_tts_cache(self) -> None:
        """Synthesize the entire pending response in ONE Kokoro call, once.

        Building a single continuous waveform for the whole response is what lets
        the per-block slices taken from it (see _slice_response_audio) play back
        seamlessly instead of as independent, stuttering fragments. Re-synthesis
        only happens after a flush (_reset_full_tts_cache) clears the cache.

        Must run before _commit_block_words pops any words, so _pending_words still
        holds the complete response. While the streaming LLM path is still
        assembling the response (_llm_in_flight), we skip — the per-fragment
        fallback in poll() covers those blocks, and the cache engages once the
        stream completes. The deployed sync path always has the full response ready.
        """
        if self._full_tts_audio is not None:
            return  # already synthesized for this response — valid until flush
        if not self._pending_words or self._llm_in_flight:
            return
        sr, audio, _ = self._generate_tts(" ".join(self._pending_words))
        self._full_tts_sr = sr
        self._full_tts_audio = audio
        self._full_tts_words = list(self._pending_words)
        self._full_tts_cursor_words = 0
        self._full_tts_cursor_samples = 0

    def _snap_to_silence(self, ideal: int) -> int:
        """Nudge a slice boundary to the nearest low-energy trough within ±150ms.

        Inter-slice playback stays seamless (the shared boundary just moves to a
        quieter sample), while a barge-in cut lands on a word gap rather than
        mid-phoneme. Falls back to `ideal` (clamped) if the search window is empty.
        """
        audio = self._full_tts_audio
        sr = self._full_tts_sr or TTS_SAMPLE_RATE
        n = len(audio)
        w = int(0.15 * sr)
        lo = max(self._full_tts_cursor_samples + 1, ideal - w)
        hi = min(n - 1, ideal + w)
        if hi <= lo:
            return min(max(ideal, self._full_tts_cursor_samples + 1), n)
        window = np.abs(audio[lo:hi].astype(np.float32))
        frame = max(1, int(0.005 * sr))
        if frame > 1 and len(window) >= frame:
            window = np.convolve(window, np.ones(frame) / frame, mode="same")
        return lo + int(np.argmin(window))

    def _slice_response_audio(self, text: str) -> tuple:
        """Return (sr, int16 slice) for the words committed this block, taken from
        the single whole-response synthesis.

        Returns (None, None) on a cache miss / edge so the caller falls back to
        per-fragment synthesis (no regression). Boundaries are computed against the
        absolute (total_words, total_samples) and each slice starts exactly where
        the previous one ended, so consecutive slices are sample-contiguous and
        reassemble the original waveform when played in order.
        """
        if self._full_tts_audio is None:
            return None, None
        n = len(text.split())
        total_words = len(self._full_tts_words)
        total_samples = len(self._full_tts_audio)
        if n == 0 or total_words == 0:
            return None, None
        end_word = self._full_tts_cursor_words + n
        if end_word > total_words:
            return None, None  # committed text doesn't match the cache — fall back
        if end_word >= total_words:
            end_sample = total_samples  # final slice: true utterance end, no snap
        else:
            ideal = round(end_word / total_words * total_samples)
            end_sample = self._snap_to_silence(ideal)
        start_sample = self._full_tts_cursor_samples
        chunk = self._full_tts_audio[start_sample:end_sample]
        sr = self._full_tts_sr
        self._full_tts_cursor_words = end_word
        self._full_tts_cursor_samples = end_sample
        if not self._pending_words and self._full_tts_cursor_words >= total_words:
            self._reset_full_tts_cache()  # response fully spoken — next response re-synthesizes
        return sr, chunk

    # ------------------------------------------------------------------
    # Mic ASR
    # ------------------------------------------------------------------

    def receive_mic_chunk(self, sample_rate: int, audio_array: np.ndarray) -> Optional[tuple]:
        """Accumulate mic audio. Returns next TTS chunk from output queue if ready."""
        arr = np.array(audio_array, dtype=np.float32)
        if sample_rate != ASR_SAMPLE_RATE:
            arr = _resample(arr, sample_rate, ASR_SAMPLE_RATE)
        self._mic_current = np.concatenate([self._mic_current, arr])
        chunk = self._drain_audio_queue()
        return chunk

    def _seal_mic_block(self, start_ts: float, end_ts: float) -> None:
        # Timestamps use server receive time, not client "spoken at" time.
        # Audio streams over a single ordered TCP connection so chunks always
        # arrive in order; for local deployment latency is ~1ms and consistent,
        # so server clock accurately reflects the real-time audio flow.
        sealed = self._mic_current.copy()
        self._mic_current = np.zeros(0, dtype=np.float32)
        self._mic_rolling.append((start_ts, end_ts, sealed))
        if len(self._mic_rolling) > MAX_MIC_BLOCKS:
            self._mic_rolling.pop(0)

        # Store mic PCM into the matching historical block
        for block in reversed(self.blocks):
            if abs(block.start_ts - start_ts) < 0.5:
                block.mic_audio = sealed
                break

        rolling_copy = list(self._mic_rolling)
        if self._asr_fn is not None:
            self._executor.submit(self._asr_fn, rolling_copy, self)
        elif _ASR_MONOTONIC_COMMIT:
            self._executor.submit(self._run_parakeet_monotonic, rolling_copy)
        else:
            self._executor.submit(self._run_parakeet, rolling_copy)

    def _user_content_fingerprint(self, rolling: list) -> str:
        """Normalized token sequence from all rolling blocks' user text.
        Identical fingerprint across ASR passes means the same words shifted
        block slots (timestamp drift) but no new content arrived."""
        tokens = []
        for start_ts, _, _ in rolling:
            for block in self.blocks:
                if abs(block.start_ts - start_ts) < 0.5:
                    if block.user_text:
                        tokens.extend(self._change_tokens(block.user_text))
                    break
        # Collapse consecutive duplicate tokens. ASR boundary overlap puts the same
        # word at the tail of one block and the head of the next ("…with a" |
        # "a squared…" → "a a"), which the multiset diff in Fix B would otherwise read
        # as a new word and flush on. Non-adjacent repeats ("squared … squared") survive.
        collapsed: List[str] = []
        for _t in tokens:
            if not collapsed or collapsed[-1] != _t:
                collapsed.append(_t)
        return " ".join(collapsed)

    def _strip_boundary_duplicates(self, n_rolling: int) -> None:
        """Remove leading words from a block that duplicate the trailing word of the prior block.

        Operates on self.blocks (historical text), not the windows dict, so it
        catches cases where a frozen block kept a stale word that later drifted
        into the mutable block that follows it.  Only scans the recent window to
        keep this O(n_rolling).
        """
        start = max(0, len(self.blocks) - n_rolling - 1)
        for i in range(start, len(self.blocks) - 1):
            prev = self.blocks[i]
            curr = self.blocks[i + 1]
            if not prev.user_text or not curr.user_text:
                continue
            prev_words = prev.user_text.split()
            curr_words = curr.user_text.split()
            changed = False
            while prev_words and curr_words and self._norm(prev_words[-1]) == self._norm(curr_words[0]):
                curr_words.pop(0)
                changed = True
            if changed:
                new_text = " ".join(curr_words)
                print(
                    f"[asr→dedup] block@{curr.start_ts:.1f} stripped leading duplicate "
                    f"from {curr.user_text!r} → {new_text!r}"
                )
                curr.user_text = new_text

    def _deduplicate_block_boundary(self, windows: dict, n_rolling: int) -> None:
        """Drop first word of first-mutable block when it duplicates last word of frozen-boundary block.

        NeMo's word timestamps can shift slightly when given more audio context, causing the
        same word to appear at the tail of the last frozen block and the head of the first
        mutable block in successive ASR passes. Since the frozen block can't be corrected,
        we drop the duplicate from the mutable side.
        """
        frozen_idx = n_rolling - self.mutable_asr_windows - 1
        mutable_idx = n_rolling - self.mutable_asr_windows
        if frozen_idx < 0 or mutable_idx >= n_rolling:
            return
        frozen_words = windows.get(frozen_idx)
        mutable_words = windows.get(mutable_idx)
        if not frozen_words or not mutable_words:
            return
        if self._norm(frozen_words[-1][0]) == self._norm(mutable_words[0][0]):
            print(
                f"[asr→dedup] boundary idx={frozen_idx}/{mutable_idx} "
                f"dropped duplicate {mutable_words[0][0]!r} from mutable block"
            )
            mutable_words.pop(0)
            if not mutable_words:
                del windows[mutable_idx]

    def _run_parakeet(self, rolling: List[tuple]) -> None:
        import tempfile
        import soundfile as sf

        if not rolling:
            return
        full_audio = np.concatenate([audio for _, _, audio in rolling])
        if len(full_audio) == 0:
            return

        buf_start_ts = rolling[0][0]
        model = self._get_asr_model()
        asr_started_perf = time.perf_counter()

        tmp_path = None
        asr_t0 = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(f.name, full_audio, ASR_SAMPLE_RATE)
            with _asr_lock:
                with contextlib.redirect_stderr(io.StringIO()):
                    output = model.transcribe([tmp_path], timestamps=True, verbose=False)
        except Exception as exc:
            print(f"[asr] transcribe error: {exc!r}")
            import traceback; traceback.print_exc()
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        asr_latency = time.perf_counter() - asr_t0

        text = repr(output[0].text) if output else "(empty)"
        print(f"[asr] transcribed {len(full_audio)/ASR_SAMPLE_RATE:.1f}s -> {text}  ({asr_latency:.3f}s)")
        word_segments = output[0].timestamp.get("word", []) if output else []

        # Map each word to its rolling block by absolute end timestamp
        windows: dict = {}
        for seg in word_segments:
            word = seg.get("word", "").strip()
            if not word:
                continue
            abs_end = buf_start_ts + seg["end"]
            for idx, (start, end, _) in enumerate(rolling):
                if start <= abs_end < end:
                    windows.setdefault(idx, []).append((word, abs_end))
                    break

        # Directly write user text into the matching historical block.
        # Last mutable_asr_windows blocks can be corrected; older ones are frozen.
        # NOTE: ASR corrections do NOT increment context_version — that would race
        # with in-flight LLM calls and cause every result to be discarded as stale.
        # The periodic poll() reruns the LLM each block, so corrections are picked up
        # automatically on the next cycle.
        n_rolling = len(rolling)
        self._deduplicate_block_boundary(windows, n_rolling)

        # Fix A: clear mutable blocks whose words shifted to a later block this pass.
        # Without this, stale text persists in those slots and the fingerprint check
        # below sees spurious duplicate tokens.
        _mutable_start_idx = max(0, n_rolling - self.mutable_asr_windows)
        for _idx in range(_mutable_start_idx, n_rolling):
            if _idx in windows:
                continue
            _block_start_ts = rolling[_idx][0]
            for _blk in self.blocks:
                if abs(_blk.start_ts - _block_start_ts) < 0.5:
                    if _blk.user_text:
                        print(f"[asr→block] block@{_block_start_ts:.1f} shift-clear old={_blk.user_text!r}")
                        _blk.user_text = ""
                    break

        # Fix C: freeze ASR updates after the user goes silent.
        # Silence is measured from HISTORICAL block text, not the current ASR windows.
        # Measuring from windows would fail when a timestamp-drifted trailing word
        # appears in the latest empty block — that word defeats a windows-based freeze
        # but the historical blocks correctly show the user has been silent.
        _recent_blocks = self.blocks[-n_rolling:] if len(self.blocks) >= n_rolling else self.blocks
        _trailing_silence = 0
        for _blk in reversed(_recent_blocks):
            if _blk.user_text:
                break
            _trailing_silence += 1
        _speech_frozen = _trailing_silence >= 2

        any_changed = False
        earliest_changed_index = None
        latest_changed_target = None
        for idx, words in windows.items():
            block_start_ts, _, _ = rolling[idx]
            word_text = " ".join(w for w, _ in words)
            is_mutable = idx >= (n_rolling - self.mutable_asr_windows)

            # Find the historical block whose start_ts matches
            target = None
            target_index = None
            for block_index in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[block_index]
                if abs(block.start_ts - block_start_ts) < 0.5:
                    target = block
                    target_index = block_index
                    break

            if target is not None:
                target.asr_latency_s = asr_latency
                target.asr_started_perf_s = asr_started_perf
                if is_mutable or not target.user_text:
                    if _speech_frozen:
                        if target.user_text:
                            continue  # already-transcribed block: always freeze
                        else:
                            # Empty block during freeze: allow only if the word contains
                            # tokens NOT seen in the most recent non-empty block.
                            # A word whose tokens are all already there is a timestamp-
                            # drifted trailing word, not genuine new speech.
                            _prev_text = ''
                            for _pi in range(target_index - 1, -1, -1):
                                if self.blocks[_pi].user_text:
                                    _prev_text = self.blocks[_pi].user_text
                                    break
                            if _prev_text:
                                _new_tok = set(self._change_tokens(word_text))
                                _prev_tok = set(self._change_tokens(_prev_text))
                                if _new_tok and _new_tok.issubset(_prev_tok):
                                    continue  # drift word, skip
                    old = target.user_text
                    target.user_text = word_text
                    if old != word_text:
                        change_kind = self._classify_text_change(old, word_text)
                        print(
                            f"[asr→block] block@{block_start_ts:.1f} {change_kind} "
                            f"old={old!r} new={word_text!r}"
                        )
                        # Only trigger a context flush for mutable-window changes.
                        # Frozen blocks (outside mutable_asr_windows) may silently
                        # receive new text for history display, but must NOT restart
                        # the LLM — Parakeet sometimes fills old empty frozen blocks
                        # with hallucinated short words ("Okay.") which would cancel
                        # an in-flight response unnecessarily.
                        if change_kind != "punctuation" and is_mutable:
                            any_changed = True
                            if target_index is not None and (
                                earliest_changed_index is None or target_index < earliest_changed_index
                            ):
                                earliest_changed_index = target_index
                            latest_changed_target = target
            else:
                print(f"[asr] no matching block for start_ts={block_start_ts:.1f}")

        # Fix D: strip duplicate boundary words from historical block text.
        # When Parakeet's timestamp for a word drifts across a block boundary between
        # ASR passes, the word ends up in BOTH the tail of the previous block and the
        # head of the next block — the previous block keeps its stale text (because it
        # may be frozen or guarded by _speech_frozen) while the new pass assigns the
        # same word to the next block.  _deduplicate_block_boundary only covers the
        # one frozen/mutable boundary in the windows dict; this pass covers all
        # consecutive pairs in the actual historical block list.
        self._strip_boundary_duplicates(n_rolling)

        # Fix B: suppress flush unless GENUINELY NEW user words arrived since the last
        # flush. Full-sequence equality was too strict — a rolling-window slide (old
        # words dropping off the front), pure re-segmentation, or re-capitalization
        # changes the fingerprint with NO new speech, which spuriously restarted the
        # bot's in-progress answer (logs2.txt ctx=5: "A squared.." → "a squared.." after
        # the window slid). Compare token MULTISETS instead: slide/re-seg/case never add
        # words, so (current − previous) is empty and we skip; only a real new word makes
        # the difference non-empty. Reference is kept (not updated) on a skip so a slide
        # alone never advances it.
        # Limitation: a word flushed earlier, dropped off the window, then re-spoken can
        # be missed — acceptable stopgap; cache-aware streaming ASR will retire this.
        if any_changed:
            _current_fp = self._user_content_fingerprint(rolling)
            _new_words = Counter(_current_fp.split()) - Counter(
                self._last_ctx_flush_user_fingerprint.split()
            )
            if not _new_words:
                any_changed = False   # only slide / re-seg / case — skip flush
            else:
                self._last_ctx_flush_user_fingerprint = _current_fp

        # Only bump context_version when text actually changed so the next LLM
        # poll picks up the new transcription — but in-flight calls are NOT
        # invalidated (staleness check uses the version captured at call start).
        if any_changed:
            if latest_changed_target is not None:
                self._latest_user_source_block_id = latest_changed_target.block_id
            self._invalidate_future_assistant_continuation()
            if earliest_changed_index is not None:
                self._mark_assistant_history_stale_from(earliest_changed_index)
            print(
                "[asr→ctx] action=flush_future_continuation+mark_stale_history "
                f"earliest_block_index={earliest_changed_index} "
                f"latest_source_block_id={self._latest_user_source_block_id}"
            )
            self._last_accepted_response_context_version = None
            self.context_version += 1

    def _run_parakeet_monotonic(self, rolling: List[tuple]) -> None:
        """Monotonic-commit ASR (see _ASR_MONOTONIC_COMMIT). Re-transcribes the rolling mic
        window each tick to obtain word timestamps (so block assignment / reward attribution
        are unchanged), but FREEZES any block whose audio ends at/before the commit cursor
        (= window_end - _ASR_RIGHT_CONTEXT_S). Frozen blocks are never rewritten or cleared,
        so drift / re-segmentation of settled audio can no longer trip a context flush — the
        structural replacement for the legacy Fix A/B/C/D churn heuristics. Only genuinely
        new trailing words (Fix-B fingerprint gate) flush an in-flight response."""
        import tempfile
        import soundfile as sf

        if not rolling:
            return
        full_audio = np.concatenate([audio for _, _, audio in rolling])
        if len(full_audio) == 0:
            return

        buf_start_ts = rolling[0][0]
        window_end_ts = rolling[-1][1]
        model = self._get_asr_model()
        asr_started_perf = time.perf_counter()

        tmp_path = None
        asr_t0 = time.perf_counter()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
                sf.write(f.name, full_audio, ASR_SAMPLE_RATE)
            with _asr_lock:
                with contextlib.redirect_stderr(io.StringIO()):
                    output = model.transcribe([tmp_path], timestamps=True, verbose=False)
        except Exception as exc:
            print(f"[asr] transcribe error: {exc!r}")
            import traceback; traceback.print_exc()
            return
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        asr_latency = time.perf_counter() - asr_t0

        text = repr(output[0].text) if output else "(empty)"
        print(f"[asr•mono] transcribed {len(full_audio)/ASR_SAMPLE_RATE:.1f}s -> {text}  ({asr_latency:.3f}s)")
        word_segments = output[0].timestamp.get("word", []) if output else []

        # Group words by rolling-block index via absolute end timestamp (same as _run_parakeet).
        windows: dict = {}
        for seg in word_segments:
            word = seg.get("word", "").strip()
            if not word:
                continue
            abs_end = buf_start_ts + seg["end"]
            for idx, (start, end, _) in enumerate(rolling):
                if start <= abs_end < end:
                    windows.setdefault(idx, []).append((word, abs_end))
                    break

        # The commit cursor trails the window end by _ASR_RIGHT_CONTEXT_S. A block whose audio
        # ends at/before the cursor is frozen; the fresher tail stays provisional (revisable).
        cursor = self._asr_commit_cursor_ts
        settle_boundary = window_end_ts - _ASR_RIGHT_CONTEXT_S

        # Clear PROVISIONAL blocks that lost all their words this pass (a word drifted to a
        # neighbouring block). Frozen blocks are never cleared. Mirrors the legacy Fix A but
        # scoped to the revisable tail, so it cannot churn settled history.
        for idx, (b_start, b_end, _) in enumerate(rolling):
            if b_end <= cursor or idx in windows:
                continue
            for _blk in self.blocks:
                if abs(_blk.start_ts - b_start) < 0.5:
                    if _blk.user_text:
                        print(f"[asr•mono→block] block@{b_start:.1f} provisional-clear old={_blk.user_text!r}")
                        _blk.user_text = ""
                    break

        any_changed = False
        earliest_changed_index = None
        latest_changed_target = None
        for idx in sorted(windows):
            block_start_ts, block_end_ts, _ = rolling[idx]
            if block_end_ts <= cursor:
                continue  # fully frozen — never rewrite settled audio (this kills the churn)
            word_text = " ".join(w for w, _ in windows[idx])

            target = None
            target_index = None
            for block_index in range(len(self.blocks) - 1, -1, -1):
                block = self.blocks[block_index]
                if abs(block.start_ts - block_start_ts) < 0.5:
                    target = block
                    target_index = block_index
                    break
            if target is None:
                print(f"[asr•mono] no matching block for start_ts={block_start_ts:.1f}")
                continue

            target.asr_latency_s = asr_latency
            target.asr_started_perf_s = asr_started_perf
            old = target.user_text
            if old != word_text:
                target.user_text = word_text
                change_kind = self._classify_text_change(old, word_text)
                print(
                    f"[asr•mono→block] block@{block_start_ts:.1f} {change_kind} "
                    f"old={old!r} new={word_text!r}"
                )
                if change_kind != "punctuation":
                    any_changed = True
                    if earliest_changed_index is None or target_index < earliest_changed_index:
                        earliest_changed_index = target_index
                    latest_changed_target = target

        # Strip a word that drifted across the frozen/provisional boundary into a duplicate.
        self._strip_boundary_duplicates(len(rolling))

        # Advance the freeze cursor. Monotonic — only moves forward.
        if settle_boundary > self._asr_commit_cursor_ts:
            self._asr_commit_cursor_ts = settle_boundary

        # Fix-B flush gate (identical policy to _run_parakeet): flush only when GENUINELY NEW
        # user words arrived since the last flush. Frozen-block immutability above has already
        # removed the re-segmentation/drift edits that used to reach here spuriously.
        if any_changed:
            _current_fp = self._user_content_fingerprint(rolling)
            _new_words = Counter(_current_fp.split()) - Counter(
                self._last_ctx_flush_user_fingerprint.split()
            )
            if not _new_words:
                any_changed = False
            else:
                self._last_ctx_flush_user_fingerprint = _current_fp

        if any_changed:
            if latest_changed_target is not None:
                self._latest_user_source_block_id = latest_changed_target.block_id
            self._invalidate_future_assistant_continuation()
            if earliest_changed_index is not None:
                self._mark_assistant_history_stale_from(earliest_changed_index)
            print(
                "[asr•mono→ctx] action=flush_future_continuation+mark_stale_history "
                f"earliest_block_index={earliest_changed_index} "
                f"latest_source_block_id={self._latest_user_source_block_id}"
            )
            self._last_accepted_response_context_version = None
            self.context_version += 1

    # ------------------------------------------------------------------
    # Word queue management
    # ------------------------------------------------------------------

    def _commit_block_words(self) -> None:
        total = len(self._pending_words)
        if total == 0:
            return
        # Distribute pending words evenly across however many blocks they span.
        # e.g. 11 words → 3 blocks of [4, 4, 3]; 6 → [3, 3]; 1-5 → single block.
        # TTS pads any block under 2s, so short blocks are fine.
        n_blocks = max(1, math.ceil(total / self._n))
        base = total // n_blocks
        remainder = total % n_blocks
        n_to_commit = base + (1 if remainder > 0 else 0)
        to_commit = self._pending_words[:n_to_commit]
        self._pending_words = self._pending_words[n_to_commit:]
        if to_commit:
            text = " ".join(to_commit)
            self._current_block.assistant_text = text
            self._current_block.assistant_text_stale = False
            self._current_block.llm_latency_s = self._pending_llm_latency_s
            self._current_block.response_source_block_id = self._pending_response_source_block_id
            if self._pending_response_asr_latency_s is not None:
                self._current_block.asr_latency_s = self._pending_response_asr_latency_s
            self._committed_words.extend(to_commit)
            self._clear_pending_response_timing()
            pending_preview = " ".join(self._pending_words[:6]) + ("…" if len(self._pending_words) > 6 else "")
            self._vlog(f"[commit ✓] {text!r}  pending={len(self._pending_words)} words  [{pending_preview}]")

    def _update_pending_queue(self, proposal_words: List[str]) -> None:
        committed = self._committed_words

        committed_overlap = self._shared_suffix_prefix_len(committed, proposal_words)
        proposal_tail = proposal_words[committed_overlap:]

        proposal_tail_norm = [self._norm(w) for w in proposal_tail]
        pending_keys = [self._queue_word_key(word) for word in self._pending_words]
        proposal_tail_keys = [self._queue_word_key(word) for word in proposal_tail_norm]

        mismatch_idx = min(len(self._pending_words), len(proposal_tail_norm))
        for i, (qw, pw) in enumerate(zip(pending_keys, proposal_tail_keys)):
            if qw != pw:
                mismatch_idx = i
                break

        retained = self._pending_words[:mismatch_idx]
        new_tail = proposal_tail_norm[mismatch_idx:]
        self._pending_words = retained + new_tail
        all_words = " ".join(self._pending_words[:10]) + ("…" if len(self._pending_words) > 10 else "")
        self._vlog(f"[queue] kept={len(retained)} +new={len(new_tail)} → {len(self._pending_words)} pending  [{all_words}]")

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _format_timeblocks(self) -> str:
        parts = []
        for block in self.blocks[-self._max_prompt_blocks:]:
            user_seg = strip_user_punctuation(block.user_text) or "<idle>"
            ai_seg = self._assistant_text_for_prompt(block)
            parts.append(f"<user>{user_seg}<AI>{ai_seg}</s>")
        current_user = self._current_block.user_text if self._current_block else ""
        user_seg = strip_user_punctuation(current_user) or "<idle>"
        parts.append(f"<user>{user_seg}<AI>")
        return "".join(parts)

    def _build_prompt(self) -> tuple:
        system_prompt = _prompt_template.render()
        user_message = self._format_timeblocks()
        return system_prompt, user_message

    # ------------------------------------------------------------------
    # Generation window
    # ------------------------------------------------------------------

    def _within_generation_window(self) -> bool:
        """True if the model is allowed to generate in the current block.

        Allowed windows:
          0 – user is currently speaking (current block has user_text)
          1 – one block after user stopped (previous block had user_text)
          2 – two blocks after user stopped, but ONLY if the bot has not
              yet spoken in block[-1]
          3 – three blocks after user stopped, but ONLY if bot has not
              spoken in block[-1] or block[-2] (ASR churn buffer: gives one
              extra block when word-assignment reshuffling delays the response)
        """
        current_user = self._current_block.user_text if self._current_block else ""
        if current_user:
            return True

        if not self.blocks:
            return False

        last = self.blocks[-1]
        if last.user_text:
            return True  # one block after user stopped

        if len(self.blocks) >= 2 and self.blocks[-2].user_text:
            # Two blocks after user stopped — only if bot hasn't spoken in block[-1]
            return not bool(last.assistant_text)

        if len(self.blocks) >= 3 and self.blocks[-3].user_text:
            # Three blocks after user stopped — only if bot hasn't spoken at all yet
            return not bool(last.assistant_text) and not bool(self.blocks[-2].assistant_text)

        return False

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    # Matches the _ROLE_RE in llm_generate_train — strips Qwen chat-template
    # special tokens that may leak into the output if skip_special_tokens is off
    # or if the model echoes the prompt format (e.g. <|im_end|>, <|assistant|>).
    _ROLE_RE = re.compile(
        r"<\|(?:im_end|im_start|system|user|assistant)\|>"
        r"|<\|?(?:im_end|im_start|user|assistant|system)[|\s>][^>]*>?",
        re.I,
    )

    def _clean_llm_buffer(self, text: str, partial: bool) -> str:
        """Apply the standard LLM output cleaning pipeline.

        When partial=True, strips incomplete (unclosed) <think> blocks so that
        mid-stream content inside an open tag is not surfaced as speech.
        """
        cleaned = self._normalize(text).strip()
        cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL)
        if partial:
            cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        # Strip Qwen/chat-template role markers before the protocol tokens so
        # e.g. "<|im_end|></s>" doesn't leave a dangling "</s>" behind.
        cleaned = self._ROLE_RE.sub(" ", cleaned)
        for _tok in ("</s>", "<AI>", "<user>", "<s>", "<idle>"):
            cleaned = cleaned.replace(_tok, " ")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        # Bare "idle" / "Idle" leading word means the model chose silence —
        # same check as llm_generate_train's post-clean guard.
        if re.match(r"^[Ii]dle\b", cleaned):
            return ""
        return cleaned

    def _drain_stream_word_queue(self) -> None:
        """Drain words queued by the background streaming LLM thread.

        Called from poll() which holds DuplexSession.lock, so _pending_words
        and _committed_words access remains single-threaded.
        """
        while True:
            try:
                item = self._stream_word_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                # Stream complete — reset accumulator for next stream
                self._stream_pending_accumulator = []
            elif isinstance(item, dict):
                # Timing/context metadata from the background thread
                self._pending_llm_latency_s = item.get("llm_latency_s")
                self._pending_response_source_block_id = item.get("source_block_id")
                source_block = self._get_block_by_id(item.get("source_block_id"))
                self._pending_response_asr_latency_s = (
                    source_block.asr_latency_s if source_block is not None else None
                )
                self._last_accepted_response_context_version = item.get("context_version")
            else:
                # Word batch from the stream
                batch: List[str] = item
                self._stream_pending_accumulator.extend(batch)
                self._update_pending_queue(self._stream_pending_accumulator)

    def _run_llm_background(
        self,
        system_prompt: str,
        user_message: str,
        gen_ctx_ver: int,
        gen_src_id: Optional[str],
        llm_t0: float,
    ) -> None:
        """Background thread: stream tokens from LLM, batch into word groups, enqueue."""
        BATCH_SIZE = 5
        raw_buffer = ""
        flushed_count = 0
        meta_sent = False
        trim_applied = False

        try:
            for token in self._llm_generate_fn(system_prompt, user_message):
                if self.context_version != gen_ctx_ver:
                    break  # stale — discard remaining stream

                if not token:
                    continue

                if not meta_sent:
                    # Send metadata with TTFT on first token arrival
                    ttft = time.perf_counter() - llm_t0
                    self._stream_word_queue.put({
                        "llm_latency_s": ttft,
                        "source_block_id": gen_src_id,
                        "context_version": gen_ctx_ver,
                    })
                    meta_sent = True

                raw_buffer += token
                cleaned = self._clean_llm_buffer(raw_buffer, partial=not trim_applied)

                if not trim_applied and len(cleaned) >= _LLM_TRIM_SEARCH_START:
                    m = _SENT_END_RE.search(cleaned, _LLM_TRIM_SEARCH_START)
                    if m:
                        cleaned = cleaned[: m.end()].strip()
                        trim_applied = True

                words = cleaned.split()
                # If raw_buffer doesn't end with whitespace, last word may be incomplete
                flushable = words if (trim_applied or (raw_buffer and raw_buffer[-1].isspace())) else words[:-1]

                new_words = flushable[flushed_count:]
                while len(new_words) >= BATCH_SIZE:
                    self._stream_word_queue.put(new_words[:BATCH_SIZE])
                    flushed_count += BATCH_SIZE
                    new_words = new_words[BATCH_SIZE:]

                if trim_applied:
                    if new_words:
                        self._stream_word_queue.put(new_words)
                        flushed_count += len(new_words)
                    break

            # Final flush — only if stream was not cancelled due to staleness
            if self.context_version == gen_ctx_ver:
                cleaned = self._clean_llm_buffer(raw_buffer, partial=False)
                if not trim_applied and cleaned:
                    m = _SENT_END_RE.search(cleaned, _LLM_TRIM_SEARCH_START)
                    if m:
                        cleaned = cleaned[: m.end()].strip()
                final_words = cleaned.split()
                remaining = final_words[flushed_count:]
                if remaining:
                    self._stream_word_queue.put(remaining)

                if not meta_sent:
                    # Zero-token response — still send metadata so context_version is set
                    self._stream_word_queue.put({
                        "llm_latency_s": time.perf_counter() - llm_t0,
                        "source_block_id": gen_src_id,
                        "context_version": gen_ctx_ver,
                    })

                model_tag = _last_used_model.split("/")[-1] if _last_used_model else "?"
                total_words = len(cleaned.split())
                total_s = time.perf_counter() - llm_t0
                self._vlog(
                    f"└─ LLM ← [{model_tag}] stream  {cleaned!r}"
                    f"  ({total_words} words, {total_s:.2f}s)"
                )

        except Exception as exc:
            self.last_llm_error = f"{type(exc).__name__}: {exc}"
            self.last_llm_error_seq += 1
            print(f"[llm stream×] {self.last_llm_error}")

        finally:
            self._stream_word_queue.put(None)  # always signal completion
            self._llm_in_flight = False

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    def _has_unanswered_user_turn(self) -> bool:
        """True if the user has spoken since the bot last committed words.

        Scans backwards through historical blocks: if we find user text before
        we find any non-stale bot text, the user's turn is unanswered. This
        catches cases where ASR assigns words to historical blocks without
        triggering a context_version bump (e.g. fingerprint dedup suppressed
        the flush), which would otherwise leave context_version stale and cause
        _maybe_run_llm to exit on the version guard.

        Returns False if _pending_words is non-empty — a response is already
        in the queue waiting to be committed. Without this check the version
        guard is bypassed on every poll tick until words drain, causing the
        LLM to fire in an infinite tight loop.
        """
        if self._pending_words:
            return False
        for blk in reversed(self.blocks):
            if blk.assistant_text and not blk.assistant_text_stale:
                return False  # bot spoke more recently than user
            if blk.user_text:
                return True
        return False

    def _maybe_run_llm(self) -> None:
        if self._llm_in_flight:
            return
        # Bypass the version guard if there's unanswered user speech: ASR may
        # have assigned words to a historical block without bumping context_version
        # (fingerprint dedup suppressed the flush). Without this bypass the bot
        # stays silent until the user speaks again.
        unanswered = self._has_unanswered_user_turn()
        if not unanswered and self._last_accepted_response_context_version == self.context_version:
            return
        # If the last response was idle (empty), block re-calls within the same
        # block so we don't duplicate step records intra-block.
        if self._last_idle_call_n_blocks == len(self.blocks):
            return
        has_user_input = any(b.user_text for b in self.blocks)
        if not has_user_input and (
            self._current_block is None or not self._current_block.user_text
        ):
            return
        # Also bypass the window guard when there's unanswered user speech. The
        # window check prevents chatty bot behavior after long silence, but when the
        # user asked something the bot never answered (words assigned to a block >3
        # deep by ASR churn) the answer spans past the 3-block window — we must still
        # generate.
        if not unanswered and not self._within_generation_window():
            return

        generation_context_version = self.context_version
        generation_source_block_id = self._latest_user_source_block_id
        system_prompt, user_message = self._build_prompt()
        self._log_llm_request(generation_context_version, user_message)
        llm_t0 = time.perf_counter()

        if inspect.isgeneratorfunction(self._llm_generate_fn):
            # Streaming path — background thread manages _llm_in_flight and queue.
            self._llm_in_flight = True
            self._stream_pending_accumulator = []
            self._stream_generation_context_version = generation_context_version
            try:
                self._executor.submit(
                    self._run_llm_background,
                    system_prompt, user_message,
                    generation_context_version, generation_source_block_id, llm_t0,
                )
            except Exception:
                self._llm_in_flight = False
                raise
            return  # background thread owns the rest

        # Synchronous path (string-returning functions — used by all existing tests)
        self._llm_in_flight = True
        try:
            raw_response = self._llm_generate_fn(system_prompt, user_message)
            llm_latency = time.perf_counter() - llm_t0
            raw = raw_response.strip() if isinstance(raw_response, str) else str(raw_response or "").strip()

            if generation_context_version != self.context_version:
                self._clear_pending_response_timing()
                self._vlog(f"└─ LLM ← STALE (ctx {generation_context_version} → {self.context_version})  discarded {raw!r}")
                return

            cleaned = self._clean_llm_buffer(raw, partial=False)
            if cleaned:
                m = _SENT_END_RE.search(cleaned, _LLM_TRIM_SEARCH_START)
                if m:
                    cleaned = cleaned[: m.end()].strip()

            model_tag = _last_used_model.split("/")[-1] if _last_used_model else "?"
            self._vlog(f"└─ LLM ← [{model_tag}]  {cleaned!r}  ({len(cleaned.split())} words, {llm_latency:.2f}s)")
            self.last_llm_error = None

            proposal_words = cleaned.split()
            if not proposal_words:
                self._clear_pending_response_timing()
                self._pending_words = []
                # Do NOT set _last_accepted_response_context_version here.
                # For idle (empty) responses the model chose silence; subsequent
                # blocks within _within_generation_window() should still be
                # allowed to call the LLM so that escalating RM1 penalties can
                # fire if the model keeps failing to respond.
                # Use block-count guard to prevent duplicate calls within the
                # same block period (intra-block poll ticks).
                self._last_idle_call_n_blocks = len(self.blocks)
                return

            source_block = self._get_block_by_id(generation_source_block_id)
            self._pending_llm_latency_s = llm_latency
            self._pending_response_source_block_id = generation_source_block_id
            self._pending_response_asr_latency_s = (
                source_block.asr_latency_s if source_block is not None else None
            )
            self._update_pending_queue(proposal_words)
            self._last_accepted_response_context_version = generation_context_version
        except Exception as exc:
            self.last_llm_error = f"{type(exc).__name__}: {exc}"
            self.last_llm_error_seq += 1
            self._clear_pending_response_timing()
            self._pending_words = []
            print(f"[llm×] {self.last_llm_error}")
        finally:
            self._llm_in_flight = False

    def _log_llm_request(self, ctx_ver: int, user_message: str) -> None:
        _log_blocks = self.blocks[-self._max_prompt_blocks:]
        _W = 55
        header = f"┌─ LLM REQUEST  ctx={ctx_ver}  strip={_STRIP_USER_PUNCTUATION} {'─' * max(0, _W - 34 - len(str(ctx_ver)))}"
        lines = [header]
        for i, blk in enumerate(_log_blocks):
            idx_label = f"B[{i - len(_log_blocks)}]"
            # Show the MODEL-VISIBLE user text (post-strip) so the log matches exactly what
            # the LLM receives — not the punctuated text we keep stored for harness logic.
            u_model = strip_user_punctuation(blk.user_text or "")
            u = (u_model[:35] + "…") if len(u_model) > 35 else (u_model or "-")
            visible_ai = self._assistant_text_for_prompt(blk)
            if blk.assistant_text and blk.assistant_text_stale:
                ai_part = f'ai="-"  stale={blk.assistant_text[:30]!r}'
            elif visible_ai:
                ai_part = f"ai={visible_ai[:35]!r}"
            else:
                ai_part = 'ai="-"'
            lines.append(f"│  {idx_label}  u={u!r:<38}  {ai_part}")
        lines.append(f"│  ({len(self.blocks)} total blocks, showing last {len(_log_blocks)})")
        tail = user_message[-60:].replace("\n", "↵")
        lines.append(f"│  tail → …{tail}")
        self._vlog("\n".join(lines))

    # ------------------------------------------------------------------
    # Audio queue
    # ------------------------------------------------------------------

    def _enqueue_audio(self, sr: int, audio: np.ndarray) -> None:
        total_queued_s = sum(len(a) / s for s, a in list(self._audio_queue.queue))
        if total_queued_s < MAX_AUDIO_QUEUE_S:
            self._audio_queue.put((sr, audio))

    def _drain_audio_queue(self) -> Optional[tuple]:
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None

    # ------------------------------------------------------------------
    # History pruning
    # ------------------------------------------------------------------

    def _prune_history(self, now: float) -> None:
        cutoff = now - MAX_HISTORY_S
        self.blocks = [b for b in self.blocks if b.end_ts >= cutoff]

    # ------------------------------------------------------------------
    # User text input
    # ------------------------------------------------------------------

    def receive_text_message(self, text: str, ts: Optional[float] = None) -> None:
        text = text.strip()
        if not text:
            return
        self.context_version += 1
        self._last_accepted_response_context_version = None
        self._invalidate_future_assistant_continuation()
        now = ts if ts is not None else self._now()
        self._ensure_current_block(now)
        print(f"[user] ctx={self.context_version} {repr(text)}")
        if self._current_block.user_text:
            self._current_block.user_text += " " + text
        else:
            self._current_block.user_text = text
        self._latest_user_source_block_id = self._current_block.block_id

    # ------------------------------------------------------------------
    # Main poll loop
    # ------------------------------------------------------------------

    def poll(self) -> Optional[tuple]:
        """
        Advance the block schedule. Returns (sample_rate, audio_array) when
        audio is ready to play, or None if nothing new is available yet.
        """
        now = self._now()
        self._drain_stream_word_queue()
        if now < self._next_block_ts:
            chunk = self._drain_audio_queue()
            if chunk is not None:
                return chunk
            self._maybe_run_llm()
            # ADHOC early-emit (P2-A): a fresh response is now fully buffered but
            # the fixed block boundary hasn't arrived. Pull the tick forward to
            # now so the first speech chunk plays immediately instead of waiting
            # out the rest of the block (~1.7s). Guards:
            #   _pending_words      → there is a response to speak
            #   not _llm_in_flight  → stream/gen complete (full_tts_cache needs this)
            #   last block silent   → we're in a gap starting a NEW response, not
            #                          mid-utterance; continuous multi-block speech
            #                          must keep using _next_block_ts as its clock.
            # When all hold we fall through to the tick path below to emit now.
            early_emit = (
                _ENABLE_EARLY_EMIT
                and self._pending_words
                and not self._llm_in_flight
                and not (self.blocks and self.blocks[-1].assistant_text)
            )
            if not early_emit:
                return None
            self._vlog(f"[early-emit] pulling tick forward ({self._next_block_ts - now:.2f}s early)")
            self._next_block_ts = now

        # Use previous block's end time as this block's start (0.0 → use now on first poll)
        block_start = self._block_start_ts if self._block_start_ts else now
        self._ensure_current_block(block_start)
        self._ensure_full_tts_cache()   # synthesize whole response once, before committing
        self._commit_block_words()

        finalized = self._current_block
        finalized.end_ts = now
        self._block_start_ts = now   # carry forward for next block's start_ts
        self.blocks.append(finalized)
        self._current_block = None

        self._vlog(f"[poll] block#{len(self.blocks)} user={repr(finalized.user_text)} ai={repr(finalized.assistant_text)}")

        if finalized.assistant_text:
            sr, playback_audio = self._slice_response_audio(finalized.assistant_text)
            if playback_audio is None:
                # Cache miss / edge (e.g. streaming response not yet complete) —
                # fall back to per-fragment synthesis so behaviour never regresses.
                sr, playback_audio, finalized.tts_latency_s = self._generate_tts(finalized.assistant_text)
            else:
                finalized.tts_latency_s = 0.0  # synthesis cost was paid once at cache build
                self._vlog(
                    f"[slice] block#{len(self.blocks)} {finalized.assistant_text!r} "
                    f"→ {len(playback_audio)/sr:.2f}s audio"
                )
            finalized.tts_sr = sr
            source_block = self._get_block_by_id(finalized.response_source_block_id)
            if source_block is not None and source_block.asr_started_perf_s is not None:
                finalized.total_latency_s = time.perf_counter() - source_block.asr_started_perf_s
            audio = playback_audio
            duration = len(playback_audio) / sr
            if duration > 4.0:
                self._vlog(f"[tts] WARNING: block {duration:.2f}s > 4s for {repr(finalized.assistant_text)}")
            finalized.lead_silence_s = finalized.total_latency_s or 0.0
            if finalized.lead_silence_s > 0.0:
                lead_silence = np.zeros(int(finalized.lead_silence_s * sr), dtype=np.int16)
                finalized.tts_audio = np.concatenate([lead_silence, audio])
                reply_ready_ts = now + (finalized.tts_latency_s or 0.0)
                finalized.timeline_start_ts = reply_ready_ts - finalized.lead_silence_s
                finalized.timeline_end_ts = reply_ready_ts + (len(audio) / sr)
            else:
                finalized.tts_audio = audio
                self._clear_timeline_metadata(finalized)
        else:
            sr = TTS_SAMPLE_RATE
            audio = np.zeros(int(self._default_block_s * sr), dtype=np.int16)
            duration = self._default_block_s
            finalized.tts_audio = audio
            finalized.tts_sr = sr
            self._clear_timeline_metadata(finalized)

        self._next_block_ts = now + duration
        self._vlog(f"[poll] next_block_ts in {duration:.2f}s  queue_depth={self._audio_queue.qsize()}")
        self._enqueue_audio(sr, audio)

        self._seal_mic_block(finalized.start_ts, now)
        self._maybe_run_llm()
        self._prune_history(now)

        return self._drain_audio_queue()

    # ------------------------------------------------------------------
    # ASR window management
    # ------------------------------------------------------------------

    def _commit_frozen_window(self, window: AsrTimestampWindow) -> None:
        if window.frozen:
            return
        committed_words = [w.text for w in window.words if w.text]
        if committed_words:
            self.receive_text_message(" ".join(committed_words), ts=window.end_ts)
        window.frozen = True

    def _apply_asr_window_policy(self) -> None:
        self.asr_windows.sort(key=lambda w: w.end_ts)
        mutable_start = max(0, len(self.asr_windows) - self.mutable_asr_windows)
        for index, window in enumerate(self.asr_windows):
            if index < mutable_start:
                self._commit_frozen_window(window)
        if len(self.asr_windows) > self.max_asr_windows:
            overflow = len(self.asr_windows) - self.max_asr_windows
            for window in self.asr_windows[:overflow]:
                self._commit_frozen_window(window)
            self.asr_windows = self.asr_windows[overflow:]

    def _is_window_mutable(self, window_id: str) -> bool:
        self.asr_windows.sort(key=lambda w: w.end_ts)
        mutable_ids = {w.window_id for w in self.asr_windows[-self.mutable_asr_windows:]}
        return window_id in mutable_ids

    def ingest_parakeet_window(
        self,
        start_ts: float,
        end_ts: float,
        words: List[tuple],
        window_id: Optional[str] = None,
    ) -> bool:
        resolved_id = window_id or f"{start_ts:.3f}-{end_ts:.3f}"
        normalized_words: List[AsrAlignedWord] = []
        for text, word_end_time in words:
            n = self._norm(text)
            if n:
                normalized_words.append(AsrAlignedWord(text=n, end_time=float(word_end_time)))
        normalized_words.sort(key=lambda w: w.end_time)

        existing = None
        for idx, window in enumerate(self.asr_windows):
            if window.window_id == resolved_id:
                existing = (idx, window)
                break

        if existing is None:
            self.asr_windows.append(AsrTimestampWindow(
                window_id=resolved_id,
                start_ts=float(start_ts),
                end_ts=float(end_ts),
                words=normalized_words,
            ))
            self._apply_asr_window_policy()
            return True

        _, existing_window = existing
        if not self._is_window_mutable(existing_window.window_id):
            return False
        existing_window.start_ts = float(start_ts)
        existing_window.end_ts = float(end_ts)
        existing_window.words = normalized_words
        existing_window.revision += 1
        self._apply_asr_window_policy()
        return True

    def get_asr_window_state(self) -> List[dict]:
        self.asr_windows.sort(key=lambda w: w.end_ts)
        return [
            {
                "window_id": w.window_id,
                "start_ts": w.start_ts,
                "end_ts": w.end_ts,
                "words": [word.text for word in w.words],
                "word_end_times": [word.end_time for word in w.words],
                "revision": w.revision,
                "frozen": w.frozen,
            }
            for w in self.asr_windows
        ]

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def get_chat_history(self) -> list:
        history: list = []
        all_blocks = list(self.blocks)
        if self._current_block is not None:
            all_blocks.append(self._current_block)
        for block in all_blocks:
            if block.user_text:
                if history and history[-1]["role"] == "user":
                    history[-1]["content"] += " " + block.user_text
                else:
                    history.append({"role": "user", "content": block.user_text})
            if block.assistant_text:
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] += " " + block.assistant_text
                else:
                    history.append({"role": "assistant", "content": block.assistant_text})
        return history

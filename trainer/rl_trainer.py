"""rl_trainer.py — REINFORCE training core for full-duplex conversational policies.

Training cycle (one train_step):
    1. Sample episodes_per_train_step simulators from the DataPool.
    2. Run each through VirtualSimulationConnection (vLLM inference).
    3. Score steps with weighted reward functions.
    4. Compute REINFORCE loss with EMA baseline and KL regularisation.
    5. Backprop + AdamW update on the HuggingFace model.
    6. Sync updated weights back into the vLLM engine.

Not compatible with GRPO/RLOO: those algorithms assume single-step episodes
with i.i.d. prompts. Here every step has a unique prompt because ASR revises
the conversation history between steps.
"""

from __future__ import annotations

import concurrent.futures
import csv
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

try:
    import wandb as _wandb
    HAS_WANDB = True
except ImportError:
    _wandb = None  # type: ignore
    HAS_WANDB = False

import numpy as np
import torch

try:
    from vllm import LLM, SamplingParams as VLLMSamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False
    LLM = None  # type: ignore
    VLLMSamplingParams = None  # type: ignore

from transformers import AutoModelForCausalLM, AutoTokenizer

from full_duplex import (
    ASR_SAMPLE_RATE,
    MAX_MIC_BLOCKS,
    TTS_SAMPLE_RATE,
    TTS_MODEL,
    DuplexAudioAgent,
    DuplexAudioBlock,
    preload_piper_voice,
    _resample,
)

from .data_ingestion import DataPool, GPTVoiceSimulator, _wpm_duration_s
from .rewards import RewardFn, _user_finished_in

# OpenAI Realtime API uses 24 kHz PCM16 for both input and output audio.
_GPT_SAMPLE_RATE = 24_000

_SENT_END_RE = re.compile(r'[.!?]')


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One LLM generation call captured during a rollout episode."""

    step_id: str
    prompt_text: str
    """Exact prompt string used at inference time (system + user message).
    Saved because ASR corrections make each step's context unique."""

    full_prompt_tokens: List[int]
    """Tokenized prompt — re-used for the training forward pass."""

    response_token_ids: List[int]
    """Token ids produced by vLLM (empty list for idle responses)."""

    log_probs: List[float]
    """Per-token log_prob from vLLM sampling (empty for idle)."""

    is_idle: bool
    """True when the policy output nothing (empty or <idle> token)."""

    source_block_id: Optional[str] = None
    """agent._latest_user_source_block_id at call time.
    Used post-episode to map steps → blocks_covered."""

    blocks_covered: List[str] = field(default_factory=list)
    """block_ids whose assistant_text originated from this LLM call."""

    user_spoke_before: bool = True
    """False when context_version did not change since the previous LLM call.
    Used to identify consecutive silent-user runs for action merging."""

    reward: Optional[float] = None
    """Filled by compute_rewards()."""

    reward_breakdown: Dict[str, float] = field(default_factory=dict)
    """Weighted per-reward-function totals across blocks_covered. Filled by compute_rewards()."""


@dataclass
class Episode:
    """Completed simulation episode."""

    episode_id: str
    steps: List[StepRecord]
    blocks: List[DuplexAudioBlock]
    terminated_reason: str
    """"max_duration" | "simulator_done"."""
    eps_greedy_eligible: int = 0
    """LLM calls where user had text (epsilon-greedy was eligible)."""
    eps_greedy_vad_suppressed: int = 0
    """Eligible calls where VAD said user finished → epsilon suppressed."""
    eps_greedy_fired: int = 0
    """Calls where epsilon-greedy actually forced idle."""



@dataclass
class TrainerConfig:
    """Hyperparameters for FullDuplexRLTrainer."""

    model_name_or_path: str
    """HuggingFace model id or local path (e.g., "Qwen/Qwen2.5-1.5B-Instruct")."""

    vllm_max_tokens: int = 60
    """Max new tokens per LLM generation. Trimmed at the first .?! after the 40th token."""

    vllm_temperature: float = 1.0
    """Sampling temperature. Must be > 0 for REINFORCE exploration."""

    vllm_gpu_memory_utilization: float = 0.2
    """Fraction of GPU memory reserved for vLLM KV cache.
    Keep low enough to leave room for the HF training model + optimizer states."""

    learning_rate: float = 1e-5
    gradient_clip: float = 1.0

    gamma: float = 1.0
    """Return discount factor. 1.0 = undiscounted (recommended for short episodes)."""

    kl_coeff: float = 0.01
    """Soft KL penalty coefficient against rollout policy."""

    baseline_ema_alpha: float = 0.05
    """EMA smoothing factor for the return baseline."""

    episodes_per_train_step: int = 4

    max_seq_len: int = 712
    """Max token budget per step (prompt + response). Prompts are left-truncated."""

    device: str = "cuda"

    reward_fn_weights: Optional[List[float]] = None
    """Weight per reward function. Defaults to uniform 1.0."""

    output_dir: str = "./checkpoints"
    """Directory to save model checkpoints."""

    save_every_n_steps: int = 10
    """Save a checkpoint every N training steps. 0 = only save at the end."""

    debug: bool = False
    """Print per-reward-function scores and export block audio to debug_dir."""

    debug_dir: str = "./debug"
    """Root directory for debug audio exports (created automatically)."""

    reward_workers: int = 4
    """Thread-pool size for parallel reward evaluation.
    0 = auto (min(32, cpu_count)). Set to 1 to disable parallelism."""

    tts_model: str = ""
    """Piper TTS model path used for GPTVoiceSimulator real-time episodes.
    Defaults to the full_duplex module default (en_US-danny-low) when empty."""

    vllm_device: Optional[str] = None
    """Pin the vLLM inference engine to a specific GPU, e.g. 'cuda:3'.
    When set, vLLM and the training model run on separate GPUs so each can
    maximise its memory budget independently.  Defaults to None (vLLM shares
    the same device as config.device)."""

    ref_model_device: Optional[str] = None
    """Device for the frozen reference model. Defaults to None (uses config.device,
    same GPU as the training model). Set to e.g. 'cuda:1' to isolate KL computation."""

    ref_model_name_or_path: Optional[str] = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    """HF model id or local path for the frozen reference model.
    When set, enables the KL-against-reference reward (kl_coherence)."""

    kl_ref_coeff: float = 0.05
    """Scale factor for the KL-against-reference reward penalty."""

    kl_ref_clip: float = 5.0
    """Per-token KL clip value. Prevents BPE boundary outliers from dominating the mean."""

    max_blocks_after_user_speech: int = 2
    """Hard cap on consecutive LLM calls when the user is silent.
    0 = speak only while user is talking, 1 = one extra block, 2 = two extra blocks (default).
    At window==2, generation is further gated: only allowed when block[-1] has no bot speech
    (i.e. the bot has not yet started responding in the previous block).
    LLM calls beyond this window are forced idle without vLLM invocation."""


# ---------------------------------------------------------------------------
# vLLM generation with training metadata
# ---------------------------------------------------------------------------

def llm_generate_train(
    system_prompt: str,
    user_message: str,
    vllm_engine: Any,
    tokenizer: Any,
    max_tokens: int = 16,
    temperature: float = 1.0,
) -> Tuple[str, List[int], List[int], List[float]]:
    """
    Generate a response with the vLLM engine and return training metadata.

    Returns:
        text:               Stripped response string ("" for idle).
        prompt_token_ids:   Token ids of the full formatted prompt.
        response_token_ids: Generated token ids ([] for idle).
        log_probs:          Per-token log_prob from vLLM sampling ([] for idle).
    """
    if not HAS_VLLM:
        raise RuntimeError("vLLM is required for llm_generate_train. pip install vllm")

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except Exception:
            full_prompt = f"{system_prompt}\n\n{user_message}"
    else:
        full_prompt = f"{system_prompt}\n\n{user_message}"

    prompt_token_ids: List[int] = tokenizer.encode(full_prompt, add_special_tokens=False)

    sampling_params = VLLMSamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=1,
        skip_special_tokens=True,
    )
    outputs = vllm_engine.generate([full_prompt], sampling_params, use_tqdm=False)
    out = outputs[0].outputs[0]

    _ROLE_RE = re.compile(
        r'<\|?(?:im_end|im_start|user|assistant|system)[|\s>][^>]*>?'
        r'|</?(?:AI|s|user|idle)>',
        re.I,
    )
    text = _ROLE_RE.sub("", out.text or "")
    text = text.replace("</s>", "").replace("<s>", "").strip()
    # Strip Qwen3 thinking blocks: <think>...</think> and any orphaned tags.
    # vLLM's skip_special_tokens removes <think> (special token) but not </think>.
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'</?think>', '', text).strip()

    response_token_ids: List[int] = list(out.token_ids)

    log_probs: List[float] = []
    if out.logprobs:
        for i, lp_dict in enumerate(out.logprobs):
            if not lp_dict:
                log_probs.append(0.0)
                continue
            tok_id = (
                response_token_ids[i]
                if i < len(response_token_ids)
                else next(iter(lp_dict))
            )
            lp_obj = lp_dict.get(tok_id) or next(iter(lp_dict.values()))
            lp_val = lp_obj.logprob if hasattr(lp_obj, "logprob") else float(lp_obj)
            log_probs.append(lp_val)

    text_clean = text.replace("<idle>", "").strip()
    # If the model leads with "idle"/"Idle", it decided to be silent — discard
    # everything that follows too (the continuation is post-idle garbage).
    if re.match(r'^[Ii]dle', text_clean):
        text_clean = ""
    if not text_clean:
        # Keep tokens so REINFORCE can push against idle/EOS generation when penalised.
        return "", prompt_token_ids, response_token_ids, log_probs

    return text_clean, prompt_token_ids, response_token_ids, log_probs


# ---------------------------------------------------------------------------
# VirtualSimulationConnection
# ---------------------------------------------------------------------------

class VirtualSimulationConnection:
    """
    Drives a DuplexAudioAgent through one simulated episode.

    - Injects synthetic mic audio from the simulator.
    - Intercepts LLM calls to capture (prompt_tokens, response_tokens, log_probs).
    - Advances simulated time proportionally to audio fed (faster than real-time).
    - Replaces _seal_mic_block with a synchronous ground-truth ASR reader.
    """

    def __init__(
        self,
        simulator: Any,  # PlaybackSimulator | GPTVoiceSimulator
        vllm_engine: Any,
        tokenizer: Any,
        vllm_max_tokens: int = 60,
        vllm_temperature: float = 1.0,
        tts_model: str = "",
        device: Optional[str] = None,
        max_blocks_after_user_speech: int = 2,
    ) -> None:
        self.simulator = simulator
        self.vllm_engine = vllm_engine
        self.tokenizer = tokenizer
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_temperature = vllm_temperature
        self.tts_model = tts_model
        self.device = device
        self.max_blocks_after_user_speech = max_blocks_after_user_speech

    def run_episode(self) -> Episode:
        episode_id = str(uuid.uuid4())[:8]
        self.simulator.reset()

        steps: List[StepRecord] = []
        sim_time = [0.0]  # mutable container for simulated wall-clock

        # Tracks the context_version at the time of the previous LLM call so
        # we can detect whether the user spoke between calls.
        last_context_version: List[int] = [-1]
        # Counts consecutive LLM calls without user speech. Starts high so
        # the model doesn't generate before the first utterance.
        blocks_since_user_spoke: List[int] = [999]
        eps_eligible = [0]
        eps_vad_suppressed = [0]
        eps_fired = [0]

        def intercepted_llm_fn(system_prompt: str, user_message: str) -> str:
            user_spoke = (agent.context_version != last_context_version[0])
            last_context_version[0] = agent.context_version

            if user_spoke:
                blocks_since_user_spoke[0] = 0
            else:
                blocks_since_user_spoke[0] += 1

            if blocks_since_user_spoke[0] > self.max_blocks_after_user_speech:
                # Outside the speaking window — force idle without vLLM call.
                steps.append(StepRecord(
                    step_id=str(uuid.uuid4())[:8],
                    prompt_text=system_prompt + "\n\n" + user_message,
                    full_prompt_tokens=[],
                    response_token_ids=[],
                    log_probs=[],
                    is_idle=True,
                    source_block_id=agent._latest_user_source_block_id,
                    user_spoke_before=user_spoke,
                ))
                return ""

            # Window 2 (2 blocks after user stopped) is only valid if the bot
            # has not yet spoken in block[-1]. If it already has, force idle.
            if (blocks_since_user_spoke[0] == 2
                    and agent.blocks
                    and agent.blocks[-1].assistant_text):
                steps.append(StepRecord(
                    step_id=str(uuid.uuid4())[:8],
                    prompt_text=system_prompt + "\n\n" + user_message,
                    full_prompt_tokens=[],
                    response_token_ids=[],
                    log_probs=[],
                    is_idle=True,
                    source_block_id=agent._latest_user_source_block_id,
                    user_spoke_before=user_spoke,
                ))
                return ""

            # Epsilon-greedy: 20% chance to force idle when user is mid-sentence.
            # Generate max_tokens=1 to capture a real log_prob for the REINFORCE gradient
            # so the positive RM5 reward can propagate correctly via backprop.
            _last_blk = agent.blocks[-1] if agent.blocks else None
            _force_idle = False
            if _last_blk is not None and _last_blk.user_text:
                eps_eligible[0] += 1
                if _user_finished_in(_last_blk):
                    eps_vad_suppressed[0] += 1
                else:
                    # Crossover: bot was also speaking → model should stop and listen.
                    # Higher epsilon at overlap moments; lower at clean new-question starts.
                    _eps_rate = 0.50 if _last_blk.assistant_text else 0.20
                    if np.random.random() < _eps_rate:
                        eps_fired[0] += 1
                        _force_idle = True
            if _force_idle:
                _, ptok, rtok, lps = llm_generate_train(
                    system_prompt, user_message,
                    self.vllm_engine, self.tokenizer,
                    max_tokens=1, temperature=self.vllm_temperature,
                )
                text = ""
            else:
                text, ptok, rtok, lps = llm_generate_train(
                    system_prompt,
                    user_message,
                    self.vllm_engine,
                    self.tokenizer,
                    max_tokens=self.vllm_max_tokens,
                    temperature=self.vllm_temperature,
                )

            # Trim at first .?! after the 40th token; fall back to full text if none.
            if text and len(rtok) > 40:
                prefix = self.tokenizer.decode(rtok[:40], skip_special_tokens=True)
                m = _SENT_END_RE.search(text, len(prefix))
                if m:
                    text = text[:m.end()]
                    trunc_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    rtok = trunc_ids
                    lps = lps[:len(trunc_ids)]

            steps.append(StepRecord(
                step_id=str(uuid.uuid4())[:8],
                prompt_text=system_prompt + "\n\n" + user_message,
                full_prompt_tokens=ptok,
                response_token_ids=rtok,
                log_probs=lps,
                is_idle=(not text),
                source_block_id=agent._latest_user_source_block_id,
                user_spoke_before=user_spoke,
            ))
            return text

        block_s = self.simulator.block_s
        wpm = self.simulator.wpm
        max_episode_s = self.simulator.max_episode_s

        agent = _make_training_agent(wpm, block_s, intercepted_llm_fn, self.tts_model, self.device, quiet=True)
        agent._now = lambda: sim_time[0]  # type: ignore[assignment]

        # Ground-truth ASR: override _seal_mic_block synchronously.
        # Writes transcript from the simulator directly into blocks and
        # increments context_version (same logic as _run_parakeet).
        def sim_seal_mic_block(start_ts: float, end_ts: float) -> None:
            sealed = agent._mic_current.copy()
            agent._mic_current = np.zeros(0, dtype=np.float32)
            agent._mic_rolling.append((start_ts, end_ts, sealed))
            if len(agent._mic_rolling) > MAX_MIC_BLOCKS:
                agent._mic_rolling.pop(0)
            for block in reversed(agent.blocks):
                if abs(block.start_ts - start_ts) < 0.5:
                    block.mic_audio = sealed
                    break
            transcript = self.simulator.get_transcript_at_time(start_ts, end_ts)
            if transcript:
                # Verify this text comes from the simulator's known script.
                # If the simulator has word_timestamps, check that the returned
                # words actually exist there; mismatches indicate a timing bug.
                wts = getattr(getattr(self.simulator, "_data", None), "word_timestamps", None)
                if wts is not None:
                    known_words = {w for _, _, w in wts}
                    transcript_words = transcript.split()
                    unknown = [w for w in transcript_words if w not in known_words]
                    if unknown:
                        pass  # timing sanity check — unknown words expected during overlaps
                for block in reversed(agent.blocks):
                    if abs(block.start_ts - start_ts) < 0.5:
                        if block.user_text != transcript:
                            block.user_text = transcript
                            if block.assistant_text:
                                block.assistant_text_stale = True
                            agent._latest_user_source_block_id = block.block_id
                            agent._invalidate_future_assistant_continuation()
                            agent._last_accepted_response_context_version = None
                            agent.context_version += 1
                        break

        agent._seal_mic_block = sim_seal_mic_block  # type: ignore[method-assign]

        chunk_samples = max(
            160,
            int(block_s * ASR_SAMPLE_RATE / 10),
        )  # ~200 ms chunks
        dt = chunk_samples / ASR_SAMPLE_RATE
        terminated_reason = "max_duration"

        while sim_time[0] < max_episode_s:
            mic = self.simulator.get_audio_chunk(chunk_samples, ASR_SAMPLE_RATE)
            if mic is None:
                terminated_reason = "simulator_done"
                break

            agent.receive_mic_chunk(ASR_SAMPLE_RATE, mic)
            sim_time[0] += dt

            tts_out = agent.poll()
            if tts_out is not None:
                self.simulator.on_agent_tts(*tts_out)

        _fill_blocks_covered(steps, list(agent.blocks))
        steps = _merge_silent_runs(steps)

        return Episode(
            episode_id=episode_id,
            steps=steps,
            blocks=list(agent.blocks),
            terminated_reason=terminated_reason,
            eps_greedy_eligible=eps_eligible[0],
            eps_greedy_vad_suppressed=eps_vad_suppressed[0],
            eps_greedy_fired=eps_fired[0],
        )


# ---------------------------------------------------------------------------
# RealTimeGPTEpisodeRunner — wall-clock episode runner for GPTVoiceSimulator
# ---------------------------------------------------------------------------

class RealTimeGPTEpisodeRunner:
    """Drives a GPTVoiceSimulator episode at wall-clock speed with real Piper TTS.

    Key differences from VirtualSimulationConnection:
    - agent._now() uses real time.time() — block timestamps are absolute.
    - Real Piper TTS so GPT hears actual speech (not silence).
    - Both send and receive paths resample between Piper 16 kHz and GPT 24 kHz.
    - get_transcript_at_time receives episode-relative timestamps (GPT's frame)
      derived by subtracting simulator._episode_start_real from block timestamps.
    """

    def __init__(
        self,
        simulator: GPTVoiceSimulator,
        vllm_engine: Any,
        tokenizer: Any,
        vllm_max_tokens: int = 60,
        vllm_temperature: float = 1.0,
        tts_model: str = "",
        device: Optional[str] = None,
        max_blocks_after_user_speech: int = 2,
    ) -> None:
        self.simulator = simulator
        self.vllm_engine = vllm_engine
        self.tokenizer = tokenizer
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_temperature = vllm_temperature
        self.tts_model = tts_model
        self.device = device
        self.max_blocks_after_user_speech = max_blocks_after_user_speech

    def run_episode(self) -> Episode:
        episode_id = str(uuid.uuid4())[:8]
        self.simulator.reset()
        # Capture the exact wall-clock reference GPT uses for transcript timestamps.
        episode_start_real: float = self.simulator._episode_start_real

        steps: List[StepRecord] = []
        last_context_version: List[int] = [-1]
        blocks_since_user_spoke: List[int] = [999]
        eps_eligible = [0]
        eps_vad_suppressed = [0]
        eps_fired = [0]

        def intercepted_llm_fn(system_prompt: str, user_message: str) -> str:
            user_spoke = (agent.context_version != last_context_version[0])
            last_context_version[0] = agent.context_version

            if user_spoke:
                blocks_since_user_spoke[0] = 0
            else:
                blocks_since_user_spoke[0] += 1

            if blocks_since_user_spoke[0] > self.max_blocks_after_user_speech:
                steps.append(StepRecord(
                    step_id=str(uuid.uuid4())[:8],
                    prompt_text=system_prompt + "\n\n" + user_message,
                    full_prompt_tokens=[],
                    response_token_ids=[],
                    log_probs=[],
                    is_idle=True,
                    source_block_id=agent._latest_user_source_block_id,
                    user_spoke_before=user_spoke,
                ))
                return ""

            # Window 2 (2 blocks after user stopped) is only valid if the bot
            # has not yet spoken in block[-1]. If it already has, force idle.
            if (blocks_since_user_spoke[0] == 2
                    and agent.blocks
                    and agent.blocks[-1].assistant_text):
                steps.append(StepRecord(
                    step_id=str(uuid.uuid4())[:8],
                    prompt_text=system_prompt + "\n\n" + user_message,
                    full_prompt_tokens=[],
                    response_token_ids=[],
                    log_probs=[],
                    is_idle=True,
                    source_block_id=agent._latest_user_source_block_id,
                    user_spoke_before=user_spoke,
                ))
                return ""

            # Epsilon-greedy: 20% chance to force idle when user is mid-sentence.
            # Generate max_tokens=1 to capture a real log_prob for the REINFORCE gradient
            # so the positive RM5 reward can propagate correctly via backprop.
            _last_blk = agent.blocks[-1] if agent.blocks else None
            _force_idle = False
            if _last_blk is not None and _last_blk.user_text:
                eps_eligible[0] += 1
                if _user_finished_in(_last_blk):
                    eps_vad_suppressed[0] += 1
                else:
                    # Crossover: bot was also speaking → model should stop and listen.
                    # Higher epsilon at overlap moments; lower at clean new-question starts.
                    _eps_rate = 0.50 if _last_blk.assistant_text else 0.20
                    if np.random.random() < _eps_rate:
                        eps_fired[0] += 1
                        _force_idle = True
            if _force_idle:
                _, ptok, rtok, lps = llm_generate_train(
                    system_prompt, user_message,
                    self.vllm_engine, self.tokenizer,
                    max_tokens=1, temperature=self.vllm_temperature,
                )
                text = ""
            else:
                text, ptok, rtok, lps = llm_generate_train(
                    system_prompt,
                    user_message,
                    self.vllm_engine,
                    self.tokenizer,
                    max_tokens=self.vllm_max_tokens,
                    temperature=self.vllm_temperature,
                )

            # Trim at first .?! after the 40th token; fall back to full text if none.
            if text and len(rtok) > 40:
                prefix = self.tokenizer.decode(rtok[:40], skip_special_tokens=True)
                m = _SENT_END_RE.search(text, len(prefix))
                if m:
                    text = text[:m.end()]
                    trunc_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    rtok = trunc_ids
                    lps = lps[:len(trunc_ids)]

            steps.append(StepRecord(
                step_id=str(uuid.uuid4())[:8],
                prompt_text=system_prompt + "\n\n" + user_message,
                full_prompt_tokens=ptok,
                response_token_ids=rtok,
                log_probs=lps,
                is_idle=(not text),
                source_block_id=agent._latest_user_source_block_id,
                user_spoke_before=user_spoke,
            ))
            return text

        wpm = self.simulator.wpm
        block_s = self.simulator.block_s
        max_episode_s = self.simulator.max_episode_s

        # Real Piper TTS; Parakeet ASR skipped (overridden below).
        agent = _make_realtime_training_agent(
            wpm, block_s, intercepted_llm_fn,
            tts_model=self.tts_model,
            device=self.device,
            quiet=True,
        )
        # Do NOT override agent._now — it must use real time.time() so block
        # timestamps are wall-clock aligned with GPT transcript timestamps.

        def realtime_seal_mic_block(start_ts: float, end_ts: float) -> None:
            sealed = agent._mic_current.copy()
            agent._mic_current = np.zeros(0, dtype=np.float32)
            agent._mic_rolling.append((start_ts, end_ts, sealed))
            if len(agent._mic_rolling) > MAX_MIC_BLOCKS:
                agent._mic_rolling.pop(0)
            for block in reversed(agent.blocks):
                if abs(block.start_ts - start_ts) < 0.5:
                    block.mic_audio = sealed
                    break
            # Block timestamps are absolute; GPT timestamps are relative to episode start.
            rel_start = start_ts - episode_start_real
            rel_end = end_ts - episode_start_real
            transcript = self.simulator.get_transcript_at_time(rel_start, rel_end)
            if transcript:
                for block in reversed(agent.blocks):
                    if abs(block.start_ts - start_ts) < 0.5:
                        if block.user_text != transcript:
                            block.user_text = transcript
                            if block.assistant_text:
                                block.assistant_text_stale = True
                            agent._latest_user_source_block_id = block.block_id
                            agent._invalidate_future_assistant_continuation()
                            agent._last_accepted_response_context_version = None
                            agent.context_version += 1
                        break

        agent._seal_mic_block = realtime_seal_mic_block  # type: ignore[method-assign]

        chunk_samples = max(160, int(block_s * ASR_SAMPLE_RATE / 10))
        terminated_reason = "max_duration"
        episode_start = time.time()

        while time.time() - episode_start < max_episode_s:
            # GPT sends 24 kHz audio; tell the agent so it resamples to 16 kHz.
            mic = self.simulator.get_audio_chunk(chunk_samples, ASR_SAMPLE_RATE)
            if mic is None:
                terminated_reason = "simulator_done"
                break

            agent.receive_mic_chunk(_GPT_SAMPLE_RATE, mic)

            tts_out = agent.poll()
            if tts_out is not None:
                sr, tts_audio = tts_out
                # Resample Piper 16 kHz → GPT 24 kHz so GPT hears correct-speed speech.
                if sr != _GPT_SAMPLE_RATE:
                    audio_f32 = tts_audio.astype(np.float32) / 32767.0
                    audio_f32 = _resample(audio_f32, sr, _GPT_SAMPLE_RATE)
                    tts_audio = (np.clip(audio_f32, -1.0, 1.0) * 32767).astype(np.int16)
                    sr = _GPT_SAMPLE_RATE
                self.simulator.on_agent_tts(sr, tts_audio)

        _fill_blocks_covered(steps, list(agent.blocks))
        steps = _merge_silent_runs(steps)

        return Episode(
            episode_id=episode_id,
            steps=steps,
            blocks=list(agent.blocks),
            terminated_reason=terminated_reason,
            eps_greedy_eligible=eps_eligible[0],
            eps_greedy_vad_suppressed=eps_vad_suppressed[0],
            eps_greedy_fired=eps_fired[0],
        )


# ---------------------------------------------------------------------------
# Agent construction helper
# ---------------------------------------------------------------------------

def _make_training_agent(
    wpm: int,
    block_s: float,
    llm_generate_fn: Callable[[str, str], str],
    tts_model: str = "",
    device: Optional[str] = None,
    quiet: bool = False,
) -> DuplexAudioAgent:
    """Create a training agent with real Piper TTS and no Parakeet ASR.

    Real TTS is required so block.tts_audio is non-silent, enabling pyannote
    overlap detection and smart-turn VAD in the reward models.
    _seal_mic_block is overridden by the caller (VirtualSimulationConnection).
    """
    return _make_realtime_training_agent(wpm, block_s, llm_generate_fn, tts_model, device, quiet=quiet)


def _make_realtime_training_agent(
    wpm: int,
    block_s: float,
    llm_generate_fn: Callable[[str, str], str],
    tts_model: str = "",
    device: Optional[str] = None,
    quiet: bool = False,
) -> DuplexAudioAgent:
    """Create a DuplexAudioAgent with real Piper TTS for GPT Voice episodes.

    Uses real Piper so GPT actually hears speech audio instead of silence.
    Parakeet ASR is skipped — _seal_mic_block is overridden by the caller
    to read transcripts directly from GPTVoiceSimulator.
    """
    resolved_tts = tts_model or TTS_MODEL
    preload_piper_voice(tts_model=resolved_tts, device=device)

    def _dummy_asr(rolling: list, agent: DuplexAudioAgent) -> None:
        pass  # _seal_mic_block overridden; this path never executes

    agent = DuplexAudioAgent(
        wpm=wpm,
        default_block_s=block_s,
        llm_generate_fn=llm_generate_fn,
        tts_fn=None,        # real Piper
        asr_fn=_dummy_asr,  # suppresses Parakeet preload
        tts_model=resolved_tts,
        device=device,
    )
    agent.quiet = quiet
    agent._get_piper_voice()  # warm from cache now rather than first block
    return agent


# ---------------------------------------------------------------------------
# Episode post-processing helpers
# ---------------------------------------------------------------------------

def _print_episode_summary(ep: "Episode") -> None:
    """Print a compact block-by-block table for a just-completed episode."""
    UW, BW = 38, 32
    header = f"  {'#':>3}  {'user':<{UW}}  {'bot':<{BW}}"
    print(header)
    print(f"  {'─'*3}  {'─'*UW}  {'─'*BW}")
    for i, blk in enumerate(ep.blocks, 1):
        u = (blk.user_text or "-")
        b = (blk.assistant_text or "-")
        if len(u) > UW:
            u = u[:UW - 1] + "…"
        if len(b) > BW:
            b = b[:BW - 1] + "…"
        print(f"  {i:>3}  {u:<{UW}}  {b:<{BW}}")


def _fill_blocks_covered(
    steps: List[StepRecord], blocks: List[DuplexAudioBlock]
) -> None:
    """
    Back-fill StepRecord.blocks_covered after an episode ends.

    Blocks that share the same response_source_block_id came from the same
    LLM generation call. Each step recorded which block triggered its call
    (source_block_id), so we group accordingly.
    """
    source_to_blocks: Dict[str, List[str]] = {}
    for block in blocks:
        if block.assistant_text and block.response_source_block_id:
            source_to_blocks.setdefault(block.response_source_block_id, []).append(
                block.block_id
            )
    for step in steps:
        # Idle steps produced no speech — don't assign them blocks that belong
        # to a speech step sharing the same source_block_id.
        if step.source_block_id and not step.is_idle:
            step.blocks_covered = source_to_blocks.get(step.source_block_id, [])


def _collapse_run(run: List[StepRecord]) -> StepRecord:
    """Merge a list of consecutive StepRecords into one combined action."""
    if len(run) == 1:
        return run[0]
    first = run[0]
    seen: set = set()
    all_blocks: List[str] = []
    for s in run:
        for bid in s.blocks_covered:
            if bid not in seen:
                seen.add(bid)
                all_blocks.append(bid)
    return StepRecord(
        step_id=first.step_id,
        prompt_text=first.prompt_text,
        full_prompt_tokens=first.full_prompt_tokens,
        response_token_ids=[t for s in run for t in s.response_token_ids],
        log_probs=[lp for s in run for lp in s.log_probs],
        is_idle=all(s.is_idle for s in run),
        source_block_id=first.source_block_id,
        blocks_covered=all_blocks,
        user_spoke_before=first.user_spoke_before,
    )


def _merge_silent_runs(steps: List[StepRecord]) -> List[StepRecord]:
    """
    Merge consecutive non-idle steps where the user did not speak between calls.

    When the user is silent, multiple LLM calls all stem from the same
    conversational decision ("continue speaking"). Grouping them into one
    action gives a single advantage signal over the whole segment, improving
    credit assignment and reducing gradient variance.
    """
    if not steps:
        return steps
    merged: List[StepRecord] = []
    run: List[StepRecord] = [steps[0]]
    for step in steps[1:]:
        if not step.user_spoke_before and not step.is_idle and not run[-1].is_idle:
            run.append(step)
        else:
            merged.append(_collapse_run(run))
            run = [step]
    merged.append(_collapse_run(run))
    return merged


def _compute_returns(rewards: List[float], gamma: float) -> List[float]:
    """Discounted returns: G_t = r_t + gamma * G_{t+1} (reversed cumsum)."""
    G = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


# ---------------------------------------------------------------------------
# FullDuplexRLTrainer
# ---------------------------------------------------------------------------

def _create_vllm_engine(
    model_path: str,
    gpu_memory_utilization: float,
    max_model_len: int,
    dtype: str,
    local_rank: int = 0,
) -> Any:
    """Instantiate a vLLM LLM engine, optionally pinned to a specific GPU.

    vLLM V0 (GPUExecutor) always passes local_rank=0 to its single worker,
    which hard-wires the model onto cuda:0.  When local_rank != 0 we
    monkey-patch _create_worker for the duration of the LLM constructor so
    the engine lands on the requested device instead.
    """
    if not HAS_VLLM:
        raise RuntimeError("vLLM is required for FullDuplexRLTrainer. pip install vllm")

    if local_rank == 0:
        return LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=True

        )

    # vLLM <= 0.7.x used GPUExecutor; 0.8.x replaced it with UniProcExecutor.
    # Try both paths so the patch works across versions.
    import importlib
    _executor_cls = None
    _orig_create = None
    for _mod_path, _cls_name in [
        ("vllm.executor.gpu_executor", "GPUExecutor"),
        ("vllm.executor.uniproc_executor", "UniProcExecutor"),
    ]:
        try:
            _mod = importlib.import_module(_mod_path)
            _executor_cls = getattr(_mod, _cls_name)
            _orig_create = _executor_cls._create_worker
            break
        except (ImportError, AttributeError):
            continue

    if _executor_cls is not None:
        def _pinned_create(self, local_rank=0, rank=0, **kw):  # type: ignore[override]
            return _orig_create(self, local_rank=local_rank, rank=rank, **kw)

        _pinned_create.__defaults__ = (local_rank, 0)
        _executor_cls._create_worker = _pinned_create
        print(f"[trainer] vLLM pinned to cuda:{local_rank} via {_executor_cls.__name__} patch")
    else:
        print(f"[trainer] WARNING: could not pin vLLM to cuda:{local_rank} — no compatible executor found")

    try:
        engine = LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype=dtype,
            enforce_eager=True,
        )
    finally:
        if _orig_create is not None:
            _executor_cls._create_worker = _orig_create  # always restore

    return engine


class FullDuplexRLTrainer:
    """
    REINFORCE trainer for full-duplex conversational policies.

    Training cycle (one train_step):
        1. Sample episodes_per_train_step simulators from data_pool.
        2. Run each through VirtualSimulationConnection (vLLM inference).
        3. Score steps with weighted reward functions.
        4. Compute REINFORCE loss with EMA baseline and KL regularisation.
        5. Backprop + AdamW update on the HuggingFace model.
        6. Sync updated weights back into the vLLM engine.
    """

    def __init__(
        self,
        config: TrainerConfig,
        data_pool: DataPool,
        reward_fns: List[RewardFn],
    ) -> None:
        if not HAS_VLLM:
            raise RuntimeError(
                "vLLM is required for FullDuplexRLTrainer. pip install vllm"
            )

        self.config = config
        self.data_pool = data_pool
        self.reward_fns = reward_fns
        self.rm_weights: List[float] = (
            config.reward_fn_weights
            if config.reward_fn_weights is not None
            else [1.0] * len(reward_fns)
        )
        if len(self.rm_weights) != len(self.reward_fns):
            raise ValueError(
                f"reward_fn_weights length ({len(self.rm_weights)}) must match "
                f"reward_fns length ({len(self.reward_fns)})"
            )

        print(f"[trainer] loading tokenizer: {config.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[trainer] loading HuggingFace model for gradient updates")
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            dtype=torch.bfloat16,
        )
        self.model.to(config.device)
        self.model.train()
        # Recompute activations during backward instead of caching them all.
        # Trades ~30% extra compute for a large reduction in activation memory.
        self.model.gradient_checkpointing_enable()

        self.ref_model: Optional[Any] = None
        if config.ref_model_name_or_path:
            ref_device = config.ref_model_device or config.device
            print(f"[trainer] loading frozen reference model ({ref_device}): {config.ref_model_name_or_path}")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.ref_model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map=ref_device,
            )
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)

        vllm_local_rank = 0
        if config.vllm_device:
            try:
                vllm_local_rank = int(config.vllm_device.split(":")[-1])
            except (ValueError, IndexError):
                print(f"[trainer] WARNING: could not parse vllm_device={config.vllm_device!r}, "
                      "falling back to default GPU")

        _vllm_loc = f" (cuda:{vllm_local_rank})" if vllm_local_rank else ""
        print(f"[trainer] loading vLLM engine for rollout inference{_vllm_loc}")
        # Force legacy V0 engine — V1 (default in 0.8+) runs the model in a
        # subprocess, making direct weight access for sync impossible.
        os.environ["VLLM_USE_V1"] = "0"
        self.vllm_engine = _create_vllm_engine(
            model_path=config.model_name_or_path,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_model_len=config.max_seq_len,
            dtype="bfloat16",
            local_rank=vllm_local_rank,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=0.0
        )
        print("[trainer] using fp32 AdamW")
        self._baseline: float = 0.0
        self._step_count: int = 0
        self._total_episodes: int = 0
        self._run_summary_path: str = os.path.join(config.output_dir, "run_summary.txt")
        self._init_run_summary()

        if HAS_WANDB and os.getenv("WANDB_API_KEY"):
            _wandb.init(
                project=os.getenv("WANDB_PROJECT", "full-duplex-text"),
                name=os.getenv("WANDB_RUN_NAME"),
                config={
                    "model": config.model_name_or_path,
                    "learning_rate": config.learning_rate,
                    "kl_coeff": config.kl_coeff,
                    "episodes_per_train_step": config.episodes_per_train_step,
                    "vllm_max_tokens": config.vllm_max_tokens,
                    "vllm_temperature": config.vllm_temperature,
                    "gamma": config.gamma,
                    "baseline_ema_alpha": config.baseline_ema_alpha,
                    "gradient_clip": config.gradient_clip,
                },
            )
            print(f"[trainer] wandb run initialised: {_wandb.run.name}")
        elif not HAS_WANDB:
            print("[trainer] wandb not installed — metrics will not be logged remotely")
        else:
            print("[trainer] WANDB_API_KEY not set — skipping wandb init")

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, simulators: List[Any]) -> List[Episode]:
        """Run one episode per simulator (sequentially) and return all episodes."""
        episodes: List[Episode] = []
        for simulator in simulators:
            if isinstance(simulator, GPTVoiceSimulator):
                runner: Any = RealTimeGPTEpisodeRunner(
                    simulator=simulator,
                    vllm_engine=self.vllm_engine,
                    tokenizer=self.tokenizer,
                    vllm_max_tokens=self.config.vllm_max_tokens,
                    vllm_temperature=self.config.vllm_temperature,
                    tts_model=self.config.tts_model,
                    device=self.config.device,
                    max_blocks_after_user_speech=self.config.max_blocks_after_user_speech,
                )
            else:
                runner = VirtualSimulationConnection(
                    simulator=simulator,
                    vllm_engine=self.vllm_engine,
                    tokenizer=self.tokenizer,
                    vllm_max_tokens=self.config.vllm_max_tokens,
                    vllm_temperature=self.config.vllm_temperature,
                    tts_model=self.config.tts_model,
                    device=self.config.device,
                    max_blocks_after_user_speech=self.config.max_blocks_after_user_speech,
                )
            try:
                ep = runner.run_episode()
                self._total_episodes += 1
                episodes.append(ep)
                n_non_idle = sum(1 for s in ep.steps if not s.is_idle)
                src = getattr(simulator, "_data", None)
                src_id = getattr(src, "source_id", "") or getattr(simulator, "source_id", "")
                mode = "realtime" if isinstance(simulator, GPTVoiceSimulator) else "sim"
                last_prompt_tok = len(ep.steps[-1].full_prompt_tokens) if ep.steps else 0
                trunc_str = (
                    f"  [TRUNCATED {last_prompt_tok}>{self.config.max_seq_len}]"
                    if last_prompt_tok > self.config.max_seq_len else ""
                )
                eps_str = (
                    f"  eps=fired:{ep.eps_greedy_fired}/"
                    f"eligible:{ep.eps_greedy_eligible}"
                    f"(vad_blocked:{ep.eps_greedy_vad_suppressed})"
                )
                print(
                    f"[trainer] episode={ep.episode_id}  mode={mode}  "
                    f"steps={len(ep.steps)} (non-idle={n_non_idle})  "
                    f"blocks={len(ep.blocks)}  ended={ep.terminated_reason}"
                    + (f"  src={src_id}" if src_id else "")
                    + f"  last_prompt_tok={last_prompt_tok}"
                    + trunc_str
                    + eps_str
                )
                if not self.config.debug:
                    print()
                    _print_episode_summary(ep)
                    print()
            except Exception as exc:
                print(f"[trainer] episode failed: {exc!r}")
        return episodes

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    @staticmethod
    def _save_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
        """Write a float32 or int16 numpy array to a WAV file."""
        import wave as _wave
        arr = np.asarray(audio)
        if arr.dtype != np.int16:
            arr = (np.clip(arr.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with _wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(arr.tobytes())

    def _print_scored_episode_table(self, episode: Episode, ep_idx: int) -> None:
        """Print a conversation table with RM scores inline after each scored block."""
        UW, BW = 38, 32
        rm_names = [fn.__name__ for fn in self.reward_fns]

        # Map block_id → the step that covers it.
        # Walk blocks in episode order so the last assignment is the last covered block.
        blk_to_step: Dict[str, "StepRecord"] = {}
        for step in episode.steps:
            for bid in step.blocks_covered:
                blk_to_step[bid] = step

        # last_blk_of_step[id(step)] = block_id of that step's last covered block
        last_blk_of_step: Dict[int, str] = {}
        for blk in episode.blocks:
            if blk.block_id in blk_to_step:
                last_blk_of_step[id(blk_to_step[blk.block_id])] = blk.block_id

        ep_prompt_tok = sum(len(s.full_prompt_tokens) for s in episode.steps)
        ep_resp_tok = sum(len(s.response_token_ids) for s in episode.steps)
        trainable = sum(1 for s in episode.steps if not s.is_idle and s.response_token_ids)

        max_step_prompt = max(
            (len(s.full_prompt_tokens) for s in episode.steps if s.full_prompt_tokens),
            default=0,
        )
        trunc_str = (
            f"  ⚠ prompt_truncated(max_step={max_step_prompt}>{self.config.max_seq_len})"
            if max_step_prompt > self.config.max_seq_len else ""
        )
        print(f"\n  ep={ep_idx}  episode={episode.episode_id}  "
              f"[steps={len(episode.steps)} blocks={len(episode.blocks)} "
              f"ended={episode.terminated_reason}]")
        print(f"  prompt_tok={ep_prompt_tok:,}  response_tok={ep_resp_tok:,}  "
              f"trainable_steps={trainable}{trunc_str}")
        print(f"  eps: eligible={episode.eps_greedy_eligible}  "
              f"vad_blocked={episode.eps_greedy_vad_suppressed}  "
              f"fired={episode.eps_greedy_fired}  "
              f"effective_rate="
              f"{episode.eps_greedy_fired / max(episode.eps_greedy_eligible - episode.eps_greedy_vad_suppressed, 1):.0%}"
              )
        print(f"\n  {'#':>3}  {'user':<{UW}}  {'bot':<{BW}}")
        print(f"  {'─'*3}  {'─'*UW}  {'─'*BW}")

        for i, blk in enumerate(episode.blocks, 1):
            u = blk.user_text or "-"
            b = blk.assistant_text or "-"
            if len(u) > UW:
                u = u[:UW - 1] + "…"
            if len(b) > BW:
                b = b[:BW - 1] + "…"
            print(f"  {i:>3}  {u:<{UW}}  {b:<{BW}}")

            step = blk_to_step.get(blk.block_id)
            if step is None or step.is_idle:
                continue
            if last_blk_of_step.get(id(step)) != blk.block_id:
                continue  # print RM once, after the last covered block

            rm_vals = [step.reward_breakdown.get(nm, 0.0) for nm in rm_names]
            rm_sum = sum(rm_vals)
            rm_parts = " | ".join(f"RM{i+1}={v:+.2f}" for i, v in enumerate(rm_vals))
            kl = step.reward_breakdown.get("kl_coherence", None)
            if kl is not None and self.config.kl_ref_coeff:
                mean_kl = -(kl / self.config.kl_ref_coeff)
                kl_str = f" | KL={kl:+.3f} (mean_kl={mean_kl:+.3f})"
            else:
                kl_str = ""
            n_blks = len(step.blocks_covered)
            rm_label = "BLOCK_RM" if n_blks == 1 else f"STEP_RM(blks={n_blks})"
            print(f"  {rm_parts} | {rm_label}={rm_sum:+.3f}{kl_str}"
                  f" | TOTAL={step.reward or 0.0:+.3f} (post weighting)")

        print(f"  {'─'*3}  {'─'*UW}  {'─'*BW}")

    def compute_rewards(self, episode: Episode, ep_idx: int = 0) -> Episode:
        """Fill StepRecord.reward using weighted sum of reward functions.

        In non-debug mode, all steps are scored in parallel via a thread pool
        so blocking HTTP calls (coherence server, VAD server) overlap instead
        of stacking sequentially.
        """
        fn_names = [fn.__name__ for fn in self.reward_fns]
        n_steps = len(episode.steps)

        _block_idx = {b.block_id: i for i, b in enumerate(episode.blocks)}

        def _prior_history(covered_ids: set) -> List[DuplexAudioBlock]:
            first = min(
                (_block_idx[bid] for bid in covered_ids if bid in _block_idx),
                default=len(episode.blocks),
            )
            return episode.blocks[:first]

        def _call_fn(fn, block, hist, terminal):
            return fn(block, hist, terminal)

        def _idle_rm1_reward(step: "StepRecord") -> Tuple[float, Dict[str, float]]:
            """Apply respond_after_user_reward directly to an idle step.

            Silence has no token output so REINFORCE can't compute a gradient
            for this step, but setting its reward propagates back through
            _compute_returns and reduces the advantage of preceding speech steps.
            """
            pos = (
                _block_idx[step.source_block_id] + 1
                if step.source_block_id and step.source_block_id in _block_idx
                else len(episode.blocks)
            )
            hist = episode.blocks[:pos]
            if len(hist) < 2:
                return 0.0, {}
            rm1_w = self.rm_weights[0] if self.rm_weights else 1.0
            reward = 0.0
            breakdown: Dict[str, float] = {}
            last_finished = _user_finished_in(hist[-1])
            # lag=1: user just finished, first missed window → -1.0
            if last_finished:
                penalty = rm1_w * (-1.0)
                reward += penalty
                breakdown["respond_after_user_reward"] = penalty
            # lag=2: user finished two blocks ago, still idle → -2.0
            elif len(hist) >= 3 and _user_finished_in(hist[-2]):
                penalty = rm1_w * (-2.0)
                reward += penalty
                breakdown["respond_after_user_reward"] = penalty
            # RM5: reward idle while user is actively mid-sentence
            if hist[-1].user_text and not last_finished:
                rm5_w = self.rm_weights[4] if len(self.rm_weights) > 4 else 1.0
                bonus = rm5_w * 0.5
                reward += bonus
                breakdown["correct_idle_reward"] = bonus
            return reward, breakdown

        if self.config.debug:
            # Score all steps first (no printing), then render one combined table.
            for i, step in enumerate(episode.steps):
                is_terminal = (i == n_steps - 1)
                if step.is_idle:
                    step.reward, step.reward_breakdown = _idle_rm1_reward(step)
                    continue
                covered_ids = set(step.blocks_covered)
                covered = [b for b in episode.blocks if b.block_id in covered_ids]
                history = _prior_history(covered_ids)
                total = 0.0
                breakdown: Dict[str, float] = {}
                for fn, w in zip(self.reward_fns, self.rm_weights):
                    fn_total = 0.0
                    for blk_pos, block in enumerate(covered):
                        h = (history if fn.__name__ == "interruption_penalty"
                             else history + covered[:blk_pos])
                        fn_total += w * fn(block, h, is_terminal)
                    breakdown[fn.__name__] = fn_total
                    total += fn_total
                kl_penalty, _ = self._compute_kl_penalty(step)
                if kl_penalty:
                    total += kl_penalty
                    breakdown["kl_coherence"] = kl_penalty
                step.reward = total
                step.reward_breakdown = breakdown

            self._print_scored_episode_table(episode, ep_idx)
            return episode

        def _score_step(i: int) -> Tuple[float, Dict[str, float]]:
            step = episode.steps[i]
            if step.is_idle:
                return _idle_rm1_reward(step)
            is_terminal = (i == n_steps - 1)
            covered_ids = set(step.blocks_covered)
            covered = [b for b in episode.blocks if b.block_id in covered_ids]
            history = _prior_history(covered_ids)
            total = 0.0
            breakdown: Dict[str, float] = {}
            for fn, w in zip(self.reward_fns, self.rm_weights):
                fn_total = 0.0
                for blk_pos, block in enumerate(covered):
                    # interruption_penalty uses original history so T+2's overlap
                    # isn't penalised for T+1's (committed at the same decision point).
                    # All other RMs get augmented history so consecutive-backchannel
                    # run counts accumulate correctly across covered blocks.
                    h = history if fn.__name__ == "interruption_penalty" else history + covered[:blk_pos]
                    fn_total += w * _call_fn(fn, block, h, is_terminal)
                breakdown[fn.__name__] = fn_total
                total += fn_total
            return total, breakdown

        workers = self.config.reward_workers or min(32, os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_score_step, range(n_steps)))

        for step, (reward, breakdown) in zip(episode.steps, results):
            step.reward = reward
            step.reward_breakdown = breakdown

        return episode

    # ------------------------------------------------------------------
    # KL-against-reference reward
    # ------------------------------------------------------------------

    def _compute_kl_penalty(self, step: "StepRecord") -> Tuple[float, float]:
        """Compute KL-against-reference penalty for one non-idle step.

        Returns (penalty, mean_kl). Returns (0.0, 0.0) when ref_model is not
        loaded or the step has no response tokens.
        """
        if self.ref_model is None or step.is_idle or not step.response_token_ids or not step.log_probs:
            return 0.0, 0.0

        n_resp = len(step.response_token_ids)
        budget = self.config.max_seq_len - n_resp - 1
        if budget < 1:
            return 0.0, 0.0

        ref_device = next(self.ref_model.parameters()).device
        prompt_tokens = step.full_prompt_tokens[-budget:]
        all_tokens = prompt_tokens + step.response_token_ids
        n_prompt = len(prompt_tokens)

        input_ids = torch.tensor([all_tokens], dtype=torch.long, device=ref_device)
        with torch.no_grad():
            ref_logits = self.ref_model(input_ids=input_ids).logits  # [1, T, V]

        # Causal shift: logits[i] predicts token[i+1]
        shift_logits = ref_logits[0, n_prompt - 1: n_prompt - 1 + n_resp, :].float()
        shift_labels = torch.tensor(
            step.response_token_ids[:n_resp], dtype=torch.long, device=ref_device
        )
        per_token_ref_lp = -torch.nn.functional.cross_entropy(
            shift_logits, shift_labels, reduction="none"
        ).cpu()

        per_token_student_lp = torch.tensor(step.log_probs[:n_resp])
        per_token_kl = per_token_student_lp - per_token_ref_lp
        mean_kl = torch.clamp(per_token_kl, max=self.config.kl_ref_clip).mean().item()
        penalty = -self.config.kl_ref_coeff * mean_kl
        return penalty, mean_kl

    def compute_kl_ref_rewards(self, episodes: List[Episode]) -> List[Episode]:
        """Add KL-against-reference penalty to each non-idle step's reward.

        No-op when ref_model is not loaded. Steps already tagged with
        'kl_coherence' (debug path computed it inline) are skipped.
        """
        if self.ref_model is None:
            return episodes

        for episode in episodes:
            for step in episode.steps:
                if "kl_coherence" in step.reward_breakdown:
                    continue  # debug path already computed and applied
                penalty, _ = self._compute_kl_penalty(step)
                if penalty:
                    step.reward = (step.reward or 0.0) + penalty
                    step.reward_breakdown["kl_coherence"] = penalty

        return episodes

    # ------------------------------------------------------------------
    # REINFORCE loss
    # ------------------------------------------------------------------

    def _count_trainable_tokens(self, episodes: List[Episode]) -> int:
        total = 0
        for episode in episodes:
            for step in episode.steps:
                if not step.response_token_ids:
                    continue
                n_resp = len(step.response_token_ids)
                if self.config.max_seq_len - n_resp - 1 < 1:
                    continue
                total += n_resp
        return total

    def accumulate_reinforce_gradients(self, episodes: List[Episode]) -> float:
        """
        Accumulate REINFORCE gradients via per-step backward passes.

        For each non-idle step t with advantage A_t = G_t - baseline:

            L_t = (-A_t * log π_θ(a_t | s_t)
                   + kl_coeff * (log π_θ(a_t | s_t) - log π_old(a_t | s_t)))
                  / total_tokens

        Calling backward() immediately after each step means only one forward
        pass worth of activations lives in GPU memory at a time, cutting peak
        allocation from O(N_steps) to O(1).

        Returns the scalar total loss value for logging.
        """
        total_tokens = self._count_trainable_tokens(episodes)
        if total_tokens == 0:
            return 0.0

        total_loss_val = 0.0
        for episode in episodes:
            rewards = [s.reward or 0.0 for s in episode.steps]
            returns = _compute_returns(rewards, self.config.gamma)

            for step, G in zip(episode.steps, returns):
                if not step.response_token_ids:
                    continue

                n_resp = len(step.response_token_ids)
                advantage = G - self._baseline

                budget = self.config.max_seq_len - n_resp - 1
                if budget < 1:
                    continue
                prompt_tokens = step.full_prompt_tokens[-budget:]
                all_tokens = prompt_tokens + step.response_token_ids
                n_prompt_used = len(prompt_tokens)

                input_ids = torch.tensor(
                    [all_tokens], dtype=torch.long, device=self.config.device
                )
                labels = input_ids.clone()
                labels[0, :n_prompt_used] = -100

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = self.model(input_ids=input_ids, labels=labels)

                # out.loss is mean NLL over unmasked (response) tokens
                nll_mean = out.loss
                policy_log_prob = -nll_mean * n_resp  # sum of response log_probs

                # KL penalty: KL(π_old || π_θ) ≈ log π_old − log π_θ per sampled token.
                # Positive when the current policy has moved away from the rollout
                # distribution, adding to the loss and resisting large weight updates.
                pi_old_sum = sum(step.log_probs[:n_resp]) if step.log_probs else 0.0
                kl_loss = pi_old_sum - policy_log_prob

                step_loss = (
                    -advantage * policy_log_prob
                    + self.config.kl_coeff * kl_loss
                ) / total_tokens

                step_loss.backward()
                total_loss_val += step_loss.item()

        return total_loss_val

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, simulators: Optional[List[Any]] = None) -> Dict[str, float]:
        """One complete REINFORCE update cycle. Returns a metrics dict.

        Args:
            simulators: Pre-built simulator list. When None, ``data_pool.sample()``
                        is called. Pass explicitly only for epoch-based training
                        via ``train_epochs()``; prefer that method over direct
                        use of this parameter.
        """
        if simulators is None:
            simulators = self.data_pool.sample(self.config.episodes_per_train_step)

        episodes = self.collect_rollouts(simulators)

        if not episodes:
            return {"step": self._step_count, "loss": 0.0, "n_episodes": 0}

        # Score
        episodes = [self.compute_rewards(e, ep_idx=i) for i, e in enumerate(episodes)]

        # KL-against-reference penalty (no-op when ref_model is None)
        episodes = self.compute_kl_ref_rewards(episodes)

        # Update EMA baseline from all returns in this batch
        all_returns: List[float] = []
        for ep in episodes:
            rewards = [s.reward or 0.0 for s in ep.steps]
            all_returns.extend(_compute_returns(rewards, self.config.gamma))
        alpha = self.config.baseline_ema_alpha
        batch_mean = sum(all_returns) / len(all_returns) if all_returns else 0.0
        self._baseline = (1.0 - alpha) * self._baseline + alpha * batch_mean

        # REINFORCE loss + backprop (per-step backward to bound peak memory)
        self.optimizer.zero_grad()
        loss_val = self.accumulate_reinforce_gradients(episodes)
        has_grad = any(p.grad is not None for p in self.model.parameters())
        if has_grad:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()

        # Sync updated weights → vLLM engine, then release fragmented cache
        self._sync_weights_to_vllm()
        torch.cuda.empty_cache()

        self._step_count += 1
        all_steps = [s for e in episodes for s in e.steps if s.reward is not None]
        all_rewards = [s.reward for s in all_steps]
        n_scored = len(all_rewards)

        # Per-reward-function averages across all scored steps.
        # Pre-seeded with all RM names so every RM always appears even when it fires 0.
        fn_totals: Dict[str, float] = {fn.__name__: 0.0 for fn in self.reward_fns}
        for s in all_steps:
            for fn_name, val in s.reward_breakdown.items():
                fn_totals[fn_name] = fn_totals.get(fn_name, 0.0) + val
        fn_avgs: Dict[str, float] = {
            k: v / n_scored for k, v in fn_totals.items()
        } if n_scored else {fn.__name__: 0.0 for fn in self.reward_fns}

        metrics: Dict[str, float] = {
            "step": float(self._step_count),
            "loss": loss_val,
            "avg_reward": float(sum(all_rewards) / n_scored) if n_scored else 0.0,
            **{f"avg_{k}": v for k, v in fn_avgs.items()},
            "baseline": self._baseline,
            "n_episodes": float(len(episodes)),
            "n_steps_total": float(sum(len(e.steps) for e in episodes)),
            "n_non_idle": float(
                sum(sum(1 for s in e.steps if not s.is_idle) for e in episodes)
            ),
        }
        width = 70
        print(f"\n{'━'*width}")
        print(
            f"  Step {self._step_count:04d}  |  "
            f"loss={metrics['loss']:.4f}  "
            f"baseline={metrics['baseline']:.3f}  "
            f"non_idle={int(metrics['n_non_idle'])}/{int(metrics['n_steps_total'])}"
        )
        print(f"  {'─'*width}")
        print(f"  {'avg_total_reward':32s}  {metrics['avg_reward']:+.4f}")
        for fn_name, avg_val in fn_avgs.items():
            print(f"  {'avg_' + fn_name:32s}  {avg_val:+.4f}")
        print(f"{'━'*width}\n")

        self._write_step_to_summary(metrics, episodes)

        if HAS_WANDB and _wandb.run is not None:
            _wandb.log(metrics, step=self._step_count)

        return metrics

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        """Save model + tokenizer to output_dir/step-NNNN (or a custom tag)."""
        label = tag if tag else f"step-{self._step_count:04d}"
        save_path = os.path.join(self.config.output_dir, label)
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"[trainer] checkpoint saved → {save_path}")
        return save_path

    def _export_rewards_csv(self, history: List[Dict[str, float]], filename: str = "rewards.csv") -> str:
        """Write per-step metrics history to a CSV file in output_dir."""
        path = os.path.join(self.config.output_dir, filename)
        os.makedirs(self.config.output_dir, exist_ok=True)
        if not history:
            return path
        fieldnames = list(history[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"[trainer] rewards CSV saved → {path}")
        return path

    def train(self, num_steps: int) -> List[Dict[str, float]]:
        """Run num_steps REINFORCE updates (random sampling). Returns per-step metrics."""
        history: List[Dict[str, float]] = []
        for _ in range(num_steps):
            history.append(self.train_step())
            n = self.config.save_every_n_steps
            if n > 0 and self._step_count % n == 0:
                self.save_checkpoint()
        self.save_checkpoint(tag="final")
        self._export_rewards_csv(history)
        self._write_final_summary(history)
        if HAS_WANDB and _wandb.run is not None:
            _wandb.finish()
        return history

    def train_epochs(self, num_epochs: int, shuffle: bool = True) -> List[Dict[str, float]]:
        """Epoch-based training: each epoch covers every source in the pool exactly once.

        Within an epoch the pool is divided into batches of
        ``config.episodes_per_train_step`` simulators. Sources are optionally
        shuffled between epochs so the model sees different orderings.
        """
        history: List[Dict[str, float]] = []
        for epoch in range(num_epochs):
            print(f"[trainer] epoch {epoch + 1}/{num_epochs}")
            for batch in self.data_pool.iter_batches(
                self.config.episodes_per_train_step, shuffle=shuffle
            ):
                history.append(self.train_step(simulators=batch))
                n = self.config.save_every_n_steps
                if n > 0 and self._step_count % n == 0:
                    self.save_checkpoint()
        self.save_checkpoint(tag="final")
        self._export_rewards_csv(history)
        self._write_final_summary(history)
        if HAS_WANDB and _wandb.run is not None:
            _wandb.finish()
        return history

    # ------------------------------------------------------------------
    # Run summary file
    # ------------------------------------------------------------------

    def _init_run_summary(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        cfg = self.config
        rm_names = [fn.__name__ for fn in self.reward_fns]
        lines = [
            "=" * 80,
            "TRAINING RUN SUMMARY",
            f"Started : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Summary : {self._run_summary_path}",
            "=" * 80,
            "",
            "INITIALIZATION PARAMETERS",
            "-" * 40,
            f"  model              : {cfg.model_name_or_path}",
            f"  ref_model          : {cfg.ref_model_name_or_path}",
            f"  learning_rate      : {cfg.learning_rate}",
            f"  kl_coeff           : {cfg.kl_coeff}",
            f"  kl_ref_coeff       : {cfg.kl_ref_coeff}",
            f"  kl_ref_clip        : {cfg.kl_ref_clip}",
            f"  gamma              : {cfg.gamma}",
            f"  baseline_ema_alpha : {cfg.baseline_ema_alpha}",
            f"  gradient_clip      : {cfg.gradient_clip}",
            f"  episodes_per_step  : {cfg.episodes_per_train_step}",
            f"  vllm_max_tokens    : {cfg.vllm_max_tokens}",
            f"  vllm_temperature   : {cfg.vllm_temperature}",
            f"  max_seq_len        : {cfg.max_seq_len}",
            f"  max_blocks_after_u : {cfg.max_blocks_after_user_speech}",
            f"  device             : {cfg.device}",
            f"  vllm_device        : {cfg.vllm_device}",
            f"  reward_fns         : {rm_names}",
            f"  rm_weights         : {self.rm_weights}",
            "",
        ]
        with open(self._run_summary_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[trainer] run summary → {self._run_summary_path}")

    def _append_summary(self, text: str) -> None:
        with open(self._run_summary_path, "a") as f:
            f.write(text)

    def _write_step_to_summary(
        self, metrics: Dict[str, float], episodes: List[Episode]
    ) -> None:
        """Append one train-step section: metrics + 1–2 sampled episodes."""
        if not episodes:
            return
        step_n = int(metrics["step"])
        lines = [
            f"\n{'='*80}",
            f"STEP {step_n:04d}  |  "
            f"loss={metrics['loss']:.4f}  "
            f"avg_reward={metrics['avg_reward']:+.4f}  "
            f"baseline={metrics['baseline']:.3f}  "
            f"non_idle={int(metrics['n_non_idle'])}/{int(metrics['n_steps_total'])}",
        ]

        # Per-RM averages on one line
        rm_names = [fn.__name__ for fn in self.reward_fns]
        rm_parts = "  ".join(
            f"avg_RM{i+1}={metrics.get('avg_' + n, 0.0):+.3f}"
            for i, n in enumerate(rm_names)
        )
        kl_avg = metrics.get("avg_kl_coherence", None)
        kl_str = f"  avg_KL={kl_avg:+.3f}" if kl_avg is not None else ""
        lines.append(f"  {rm_parts}{kl_str}")
        lines.append("")

        # Sample up to 2 episodes
        n_sample = min(2, len(episodes))
        rng = np.random.default_rng(seed=step_n)
        sampled = list(rng.choice(len(episodes), size=n_sample, replace=False))
        for ep_i in sampled:
            ep = episodes[ep_i]
            lines.append(
                f"  ── Sampled episode {ep.episode_id}  "
                f"[steps={len(ep.steps)} blocks={len(ep.blocks)} ended={ep.terminated_reason}]"
            )

            # Build block → last-covered-step mapping for inline RM
            blk_to_step: Dict[str, "StepRecord"] = {}
            for st in ep.steps:
                for bid in st.blocks_covered:
                    blk_to_step[bid] = st
            last_blk: Dict[int, str] = {}
            for blk in ep.blocks:
                if blk.block_id in blk_to_step:
                    last_blk[id(blk_to_step[blk.block_id])] = blk.block_id

            UW, BW = 38, 30
            for bi, blk in enumerate(ep.blocks, 1):
                u = blk.user_text or "<silence>"
                b = blk.assistant_text or "<idle>"
                if len(u) > UW:
                    u = u[:UW - 1] + "…"
                if len(b) > BW:
                    b = b[:BW - 1] + "…"
                lines.append(f"    [{bi:02d}] U: {u:<{UW}}  B: {b}")

                st = blk_to_step.get(blk.block_id)
                if st is None or st.is_idle:
                    continue
                if last_blk.get(id(st)) != blk.block_id:
                    continue
                rm_str = "  ".join(
                    f"RM{i+1}={st.reward_breakdown.get(n, 0.0):+.2f}"
                    for i, n in enumerate(rm_names)
                )
                kl_v = st.reward_breakdown.get("kl_coherence", None)
                kl_part = f"  KL={kl_v:+.3f}" if kl_v is not None else ""
                n_b = len(st.blocks_covered)
                lines.append(
                    f"    [RM] blks={n_b}  {rm_str}{kl_part}"
                    f"  TOTAL={st.reward or 0.0:+.3f}"
                )
            lines.append("")

        self._append_summary("\n".join(lines) + "\n")

    def _write_final_summary(self, history: List[Dict[str, float]]) -> None:
        if not history:
            return
        n = len(history)
        split = max(1, n // 5)
        avg_r_all = sum(h["avg_reward"] for h in history) / n
        avg_r_early = sum(h["avg_reward"] for h in history[:split]) / split
        avg_r_late = sum(h["avg_reward"] for h in history[-split:]) / split
        avg_loss = sum(h["loss"] for h in history) / n

        rm_names = [fn.__name__ for fn in self.reward_fns]
        rm_summary = "  ".join(
            f"RM{i+1}={sum(h.get('avg_' + nm, 0.0) for h in history)/n:+.3f}"
            for i, nm in enumerate(rm_names)
        )
        kl_vals = [h.get("avg_kl_coherence", None) for h in history]
        kl_vals = [v for v in kl_vals if v is not None]
        kl_summary = f"  avg_KL={sum(kl_vals)/len(kl_vals):+.3f}" if kl_vals else ""

        lines = [
            f"\n{'='*80}",
            "FINAL TRAINING SUMMARY",
            f"Completed  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Steps      : {n}",
            f"Episodes   : {self._total_episodes}",
            "",
            f"  avg_loss               : {avg_loss:.4f}",
            f"  avg_reward (all steps) : {avg_r_all:+.4f}",
            f"  reward trend  early={avg_r_early:+.4f}  late={avg_r_late:+.4f}  "
            f"Δ={avg_r_late - avg_r_early:+.4f}",
            "",
            f"  Per-RM averages: {rm_summary}{kl_summary}",
            f"{'='*80}",
        ]
        self._append_summary("\n".join(lines) + "\n")
        print(f"[trainer] final summary written → {self._run_summary_path}")

    # ------------------------------------------------------------------
    # vLLM weight sync
    # ------------------------------------------------------------------

    def _sync_weights_to_vllm(self) -> None:
        """
        Push HuggingFace model weights into the vLLM engine's model runner.

        vLLM's internal layout changed across major versions:
          < 0.6 : llm_engine.model_executor.driver_worker.model_runner
          0.6-0.7: same path, executor may be GPUExecutor
          0.8+  : llm_engine stores executor as _executor; workers list replaces driver_worker
        We probe both attribute names so the same code works across versions.
        """
        try:
            named_params = [(k, v.detach().cpu()) for k, v in self.model.named_parameters()]
            engine = self.vllm_engine.llm_engine

            executor = (
                getattr(engine, "model_executor", None)
                or getattr(engine, "_executor", None)
            )
            if executor is None:
                raise AttributeError("no model_executor / _executor on LLMEngine")

            worker = getattr(executor, "driver_worker", None)
            if worker is None:
                workers = getattr(executor, "workers", [])
                worker = workers[0] if workers else None
            if worker is None:
                raise AttributeError("no driver_worker / workers on executor")

            worker.model_runner.model.load_weights(iter(named_params))
            print(f"[trainer] vLLM weights synced at step {self._step_count}")
        except Exception as exc:
            print(
                f"[trainer] WARNING: vLLM weight sync failed — {exc!r}. "
                "Rollout will use stale weights."
            )

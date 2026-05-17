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

import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    DuplexAudioAgent,
    DuplexAudioBlock,
)

from .data_ingestion import DataPool, _wpm_duration_s
from .rewards import RewardFn


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


@dataclass
class Episode:
    """Completed simulation episode."""

    episode_id: str
    steps: List[StepRecord]
    blocks: List[DuplexAudioBlock]
    terminated_reason: str
    """"max_duration" | "simulator_done"."""



@dataclass
class TrainerConfig:
    """Hyperparameters for FullDuplexRLTrainer."""

    model_name_or_path: str
    """HuggingFace model id or local path (e.g., "Qwen/Qwen2.5-1.5B-Instruct")."""

    vllm_max_tokens: int = 16
    """Max new tokens per LLM generation (should match production setting)."""

    vllm_temperature: float = 1.0
    """Sampling temperature. Must be > 0 for REINFORCE exploration."""

    vllm_gpu_memory_utilization: float = 0.35
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
                messages, tokenize=False, add_generation_prompt=True
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

    _ROLE_RE = re.compile(r'<\|?(?:im_end|im_start|user|assistant|system)[|\s>][^>]*>?', re.I)
    text = _ROLE_RE.sub("", out.text or "").strip()

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
    if not text_clean:
        return "", prompt_token_ids, [], []

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
        vllm_max_tokens: int = 16,
        vllm_temperature: float = 1.0,
    ) -> None:
        self.simulator = simulator
        self.vllm_engine = vllm_engine
        self.tokenizer = tokenizer
        self.vllm_max_tokens = vllm_max_tokens
        self.vllm_temperature = vllm_temperature

    def run_episode(self) -> Episode:
        episode_id = str(uuid.uuid4())[:8]
        self.simulator.reset()

        steps: List[StepRecord] = []
        sim_time = [0.0]  # mutable container for simulated wall-clock

        # Tracks the context_version at the time of the previous LLM call so
        # we can detect whether the user spoke between calls.
        last_context_version: List[int] = [-1]

        def intercepted_llm_fn(system_prompt: str, user_message: str) -> str:
            user_spoke = (agent.context_version != last_context_version[0])
            last_context_version[0] = agent.context_version
            text, ptok, rtok, lps = llm_generate_train(
                system_prompt,
                user_message,
                self.vllm_engine,
                self.tokenizer,
                max_tokens=self.vllm_max_tokens,
                temperature=self.vllm_temperature,
            )
            steps.append(StepRecord(
                step_id=str(uuid.uuid4())[:8],
                prompt_text=system_prompt + "\n\n" + user_message,
                full_prompt_tokens=ptok,
                response_token_ids=rtok,
                log_probs=lps,
                is_idle=(not text),
                source_block_id=agent._latest_user_source_block_id,  # late binding ↓
                user_spoke_before=user_spoke,
            ))
            return text

        block_s = self.simulator.block_s
        wpm = self.simulator.wpm
        max_episode_s = self.simulator.max_episode_s

        agent = _make_training_agent(wpm, block_s, intercepted_llm_fn)
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
                for block in reversed(agent.blocks):
                    if abs(block.start_ts - start_ts) < 0.5:
                        if block.user_text != transcript:
                            block.user_text = transcript
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
        )


# ---------------------------------------------------------------------------
# Agent construction helper
# ---------------------------------------------------------------------------

def _make_training_agent(
    wpm: int,
    block_s: float,
    llm_generate_fn: Callable[[str, str], str],
) -> DuplexAudioAgent:
    """
    Create a DuplexAudioAgent with:
    - Mock TTS (silent audio of WPM-estimated duration) to skip Piper loading.
    - No real ASR (_seal_mic_block is overridden by VirtualSimulationConnection).
    """
    def mock_tts(text: str) -> Tuple[int, np.ndarray]:
        duration_s = _wpm_duration_s(text, wpm)
        samples = max(1, int(duration_s * TTS_SAMPLE_RATE))
        return TTS_SAMPLE_RATE, np.zeros(samples, dtype=np.int16)

    return DuplexAudioAgent(
        wpm=wpm,
        default_block_s=block_s,
        llm_generate_fn=llm_generate_fn,
        tts_fn=mock_tts,
    )


# ---------------------------------------------------------------------------
# Episode post-processing helpers
# ---------------------------------------------------------------------------

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
        if step.source_block_id:
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
            torch_dtype=torch.bfloat16,
        )
        self.model.to(config.device)
        self.model.train()
        # Recompute activations during backward instead of caching them all.
        # Trades ~30% extra compute for a large reduction in activation memory.
        self.model.gradient_checkpointing_enable()

        print("[trainer] loading vLLM engine for rollout inference")
        # Force legacy V0 engine — V1 (default in 0.8+) runs the model in a
        # subprocess, making direct weight access for sync impossible.
        os.environ["VLLM_USE_V1"] = "0"
        self.vllm_engine = LLM(
            model=config.model_name_or_path,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_model_len=config.max_seq_len,
            dtype="bfloat16",
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )
        self._baseline: float = 0.0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollouts(self, simulators: List[Any]) -> List[Episode]:
        """Run one episode per simulator (sequentially) and return all episodes."""
        episodes: List[Episode] = []
        for simulator in simulators:
            conn = VirtualSimulationConnection(
                simulator=simulator,
                vllm_engine=self.vllm_engine,
                tokenizer=self.tokenizer,
                vllm_max_tokens=self.config.vllm_max_tokens,
                vllm_temperature=self.config.vllm_temperature,
            )
            try:
                ep = conn.run_episode()
                episodes.append(ep)
                n_non_idle = sum(1 for s in ep.steps if not s.is_idle)
                src = getattr(simulator, "_data", None)
                src_id = getattr(src, "source_id", "") or getattr(simulator, "source_id", "")
                print(
                    f"[trainer] episode={ep.episode_id}  "
                    f"steps={len(ep.steps)} (non-idle={n_non_idle})  "
                    f"blocks={len(ep.blocks)}  ended={ep.terminated_reason}"
                    + (f"  src={src_id}" if src_id else "")
                )
            except Exception as exc:
                print(f"[trainer] episode failed: {exc!r}")
        return episodes

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_rewards(self, episode: Episode) -> Episode:
        """Fill StepRecord.reward using weighted sum of reward functions."""
        for i, step in enumerate(episode.steps):
            is_terminal = (i == len(episode.steps) - 1)
            covered_ids = set(step.blocks_covered)
            covered = [b for b in episode.blocks if b.block_id in covered_ids]
            history = [b for b in episode.blocks if b.block_id not in covered_ids]
            total = 0.0
            for fn, w in zip(self.reward_fns, self.rm_weights):
                for block in covered:
                    total += w * fn(block, history, is_terminal)
            step.reward = total
        return episode

    # ------------------------------------------------------------------
    # REINFORCE loss
    # ------------------------------------------------------------------

    def _count_trainable_tokens(self, episodes: List[Episode]) -> int:
        total = 0
        for episode in episodes:
            for step in episode.steps:
                if step.is_idle or not step.response_token_ids:
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
                if step.is_idle or not step.response_token_ids:
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
        episodes = [self.compute_rewards(e) for e in episodes]

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
        metrics = {
            "step": float(self._step_count),
            "loss": loss_val,
            "baseline": self._baseline,
            "n_episodes": float(len(episodes)),
            "n_steps_total": float(sum(len(e.steps) for e in episodes)),
            "n_non_idle": float(
                sum(sum(1 for s in e.steps if not s.is_idle) for e in episodes)
            ),
        }
        print(
            f"[trainer] step={self._step_count}  loss={metrics['loss']:.4f}  "
            f"baseline={metrics['baseline']:.3f}  "
            f"non_idle={int(metrics['n_non_idle'])}/{int(metrics['n_steps_total'])}"
        )
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

    def train(self, num_steps: int) -> List[Dict[str, float]]:
        """Run num_steps REINFORCE updates (random sampling). Returns per-step metrics."""
        history: List[Dict[str, float]] = []
        for _ in range(num_steps):
            history.append(self.train_step())
            n = self.config.save_every_n_steps
            if n > 0 and self._step_count % n == 0:
                self.save_checkpoint()
        self.save_checkpoint(tag="final")
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
        return history

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

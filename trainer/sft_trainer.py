"""sft_trainer.py — SFT warm-up trainer for the full-duplex silence action.

Purpose
-------
The RL trainer cannot bootstrap silence because log_π(EOS | mid-sentence context)
starts at ~-35. This trainer surgically fixes that by doing a short supervised
fine-tuning pass where the only label is the single EOS token at the response
start position, in contexts where the user is mid-sentence.

After training, call get_trained_model().save_pretrained(path) and pass that
path as BOTH model_name_or_path and ref_model_name_or_path to FullDuplexRLTrainer.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

try:
    import wandb as _wandb
    HAS_WANDB = True
except ImportError:
    _wandb = None  # type: ignore
    HAS_WANDB = False

try:
    from peft import LoraConfig, TaskType, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from full_duplex import (
    ASR_SAMPLE_RATE,
    MAX_MIC_BLOCKS,
    TTS_SAMPLE_RATE,
    DuplexAudioAgent,
)
from .training_utils import load_hf_model

from dotenv import load_dotenv as _load_dotenv
_load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    model_name_or_path: str
    output_dir: str = "./sft_checkpoints"
    learning_rate: float = 1e-6        # lower than RL — minimise drift
    max_steps: int = 200
    eval_every_n_steps: int = 10
    target_eos_log_prob: float = -5.0  # early-stop: mean_eos_lp reaches this
    ppl_ratio_warn: float = 1.3        # warn when speech perplexity degrades
    ppl_ratio_stop: float = 2.0        # stop when speech perplexity degrades badly
    batch_size: int = 4
    gradient_clip: float = 1.0
    device: str = "cuda"
    max_seq_len: int = 712
    n_silence_examples: int = 150      # silence training + probe examples to collect
    n_silence_probes: int = 10         # held-out silence probes for eval
    n_speech_probes: int = 10          # held-out speech probes for eval
    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    # Attention-only LoRA is the safest starting point.
    # If convergence is too slow, also add: "gate_proj", "up_proj"
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )


# ---------------------------------------------------------------------------
# Silence data collection (no vLLM required)
# ---------------------------------------------------------------------------

class SilenceDataCollector:
    """Runs episodes with a dummy LLM to collect mid-sentence silence prompts.

    No vLLM, TTS, or ASR models are loaded. The agent is driven by injecting
    the simulator's ground-truth transcript directly into blocks (same technique
    as VirtualSimulationConnection._seal_mic_block).
    """

    def collect(
        self,
        simulators: List[Any],
        tokenizer: Any,
        n_silence: int,
        n_speech: int,
        max_seq_len: int = 712,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Run episodes until n_silence + n_speech prompts are collected.

        Returns:
            silence_examples: list of {"input_ids": [..., eos_id], "labels": [...]}
                              Labels are -100 everywhere except the final EOS token.
            speech_prompts:   list of {"prompt_tokens": [...], "prompt_text": str}
                              Held-out — never trained on. Used to build speech probes
                              in SFTTrainer (responses generated at init).
        """
        eos_id = tokenizer.eos_token_id
        silence_examples: List[Dict] = []
        speech_prompts: List[Dict] = []

        total_needed = n_silence + n_speech
        sim_cycle = list(simulators)
        sim_idx = 0

        while len(silence_examples) + len(speech_prompts) < total_needed:
            sim = sim_cycle[sim_idx % len(sim_cycle)]
            sim_idx += 1
            sim.reset()

            collected_this_ep: List[Tuple[str, str, bool]] = []  # (sys, user, is_silence)

            def dummy_llm_fn(system_prompt: str, user_message: str) -> str:
                # Capture the current block's user state to classify the call.
                # is_silence = user is mid-sentence at the source block.
                # Use text-only heuristic (terminal punctuation) — the VAD
                # server is not available during SFT data collection.
                last_blk = agent.blocks[-1] if agent.blocks else None
                if last_blk is None or not last_blk.user_text:
                    return ""
                turn_done = _is_turn_complete_text(last_blk.user_text)
                is_silence = not turn_done
                is_speech = turn_done
                if is_silence or is_speech:
                    collected_this_ep.append((system_prompt, user_message, is_silence))
                return ""  # no response — agent stays silent

            agent = _make_silent_agent(
                wpm=sim.wpm,
                block_s=sim.block_s,
                llm_fn=dummy_llm_fn,
            )
            agent._now = lambda: sim_time[0]  # type: ignore[assignment]
            agent._seal_mic_block = _make_sim_seal(agent, sim)  # type: ignore[method-assign]

            sim_time = [0.0]
            chunk_samples = max(160, int(sim.block_s * ASR_SAMPLE_RATE / 10))
            dt = chunk_samples / ASR_SAMPLE_RATE

            while sim_time[0] < sim.max_episode_s:
                mic = sim.get_audio_chunk(chunk_samples, ASR_SAMPLE_RATE)
                if mic is None:
                    break
                agent.receive_mic_chunk(ASR_SAMPLE_RATE, mic)
                tts_out = agent.poll()
                if tts_out is not None:
                    sim.on_agent_tts(*tts_out)
                sim_time[0] += dt

            # Tokenise and bucket collected prompts
            for sys_p, usr_p, is_sil in collected_this_ep:
                if len(silence_examples) + len(speech_prompts) >= total_needed:
                    break
                full_prompt = _format_prompt(sys_p, usr_p, tokenizer)
                tokens = tokenizer.encode(full_prompt, add_special_tokens=False)
                # Truncate to leave room for the EOS label token
                tokens = tokens[-(max_seq_len - 1):]

                if is_sil and len(silence_examples) < n_silence:
                    input_ids = tokens + [eos_id]
                    labels = [-100] * len(tokens) + [eos_id]
                    silence_examples.append({
                        "input_ids": input_ids,
                        "labels": labels,
                    })
                elif not is_sil and len(speech_prompts) < n_speech:
                    speech_prompts.append({
                        "prompt_tokens": tokens,
                        "prompt_text": full_prompt,
                    })

        print(
            f"[sft-collect] {len(silence_examples)} silence examples, "
            f"{len(speech_prompts)} speech prompts collected "
            f"across {sim_idx} episodes"
        )
        return silence_examples, speech_prompts


def _is_turn_complete_text(user_text: str) -> bool:
    """Text-only heuristic for whether the user has finished their turn.

    Avoids calling the VAD server (which requires a running audio pipeline).
    Simulator transcripts are complete sentences, so terminal punctuation is
    a reliable signal.
    """
    text = user_text.strip()
    return bool(text) and text[-1] in ".!?…"


def _format_prompt(system_prompt: str, user_message: str, tokenizer: Any) -> str:
    """Build the full prompt string the same way llm_generate_train does."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{user_message}"


def _make_silent_agent(wpm: int, block_s: float, llm_fn: Any) -> DuplexAudioAgent:
    """DuplexAudioAgent with silent TTS and noop ASR — no models loaded."""
    n_silent = max(1, int(block_s * TTS_SAMPLE_RATE))

    def _silent_tts(_text: str) -> Tuple[int, np.ndarray]:
        return TTS_SAMPLE_RATE, np.zeros(n_silent, dtype=np.float32)

    def _noop_asr(_rolling: list, _agent: DuplexAudioAgent) -> None:
        pass  # _seal_mic_block is overridden by the caller

    agent = DuplexAudioAgent(
        wpm=wpm,
        default_block_s=block_s,
        llm_generate_fn=llm_fn,
        tts_fn=_silent_tts,
        asr_fn=_noop_asr,
    )
    agent.quiet = True
    return agent


def _make_sim_seal(agent: DuplexAudioAgent, sim: Any):
    """Return a _seal_mic_block fn that reads ground-truth transcript from sim."""
    def seal(start_ts: float, end_ts: float) -> None:
        sealed = agent._mic_current.copy()
        agent._mic_current = np.zeros(0, dtype=np.float32)
        agent._mic_rolling.append((start_ts, end_ts, sealed))
        if len(agent._mic_rolling) > MAX_MIC_BLOCKS:
            agent._mic_rolling.pop(0)
        for block in reversed(agent.blocks):
            if abs(block.start_ts - start_ts) < 0.5:
                block.mic_audio = sealed
                break
        transcript = sim.get_transcript_at_time(start_ts, end_ts)
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
    return seal


# ---------------------------------------------------------------------------
# SFT Trainer
# ---------------------------------------------------------------------------

class SFTTrainer:
    """Supervised fine-tuning warm-up that teaches log_π(EOS | mid-sentence) ≈ -5.

    Usage::

        sft_config = SFTConfig(model_name_or_path="Qwen/Qwen3-4B-Instruct-2507",
                               use_lora=True)
        trainer = SFTTrainer(sft_config)

        collector = SilenceDataCollector()
        silence_data, speech_prompts = collector.collect(
            simulators, trainer.tokenizer,
            n_silence=sft_config.n_silence_examples,
            n_speech=sft_config.n_speech_probes,
        )

        trainer.set_speech_probes(speech_prompts)   # captures baseline ppl
        trainer.train(silence_data)

        trained = trainer.get_trained_model()
        trained.save_pretrained("./sft_checkpoints/final")
    """

    def __init__(self, config: SFTConfig) -> None:
        if config.use_lora and not HAS_PEFT:
            raise RuntimeError(
                "use_lora=True requires peft. pip install peft"
            )

        self.config = config
        self._step = 0

        print(f"[sft] loading tokenizer: {config.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[sft] loading model")
        self.model = load_hf_model(config.model_name_or_path, config.device)

        if config.use_lora:
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                target_modules=config.lora_target_modules,
                # If convergence is too slow, also add: "gate_proj", "up_proj"
                lora_dropout=0.0,
            )
            self.model = get_peft_model(self.model, lora_cfg)
            self.model.print_trainable_parameters()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.0,
        )

        self._silence_probes: List[Dict] = []
        self._speech_probes: List[Dict] = []   # {"input_ids": [...], "n_prompt": int}
        self._baseline_ppls: List[float] = []  # baseline perplexity per speech probe

        os.makedirs(config.output_dir, exist_ok=True)
        self._init_wandb()

    # ------------------------------------------------------------------
    # Probe setup
    # ------------------------------------------------------------------

    def set_speech_probes(self, speech_prompts: List[Dict]) -> None:
        """Tokenise speech prompts, generate reference responses, measure baseline ppl.

        Call this BEFORE train() so the baseline is captured on the original model.
        """
        self.model.eval()
        probes: List[Dict] = []
        baseline_ppls: List[float] = []

        print(f"[sft] generating {len(speech_prompts)} speech probes (greedy decoding)…")
        with torch.no_grad():
            for sp in speech_prompts[:self.config.n_speech_probes]:
                prompt_t = torch.tensor(
                    [sp["prompt_tokens"]], dtype=torch.long,
                    device=self.config.device,
                )
                # Greedy decode up to 20 tokens — captures typical first sentence
                gen_out = self.model.generate(
                    prompt_t,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
                resp_tokens = gen_out[0][len(sp["prompt_tokens"]):].tolist()
                if not resp_tokens:
                    continue
                all_ids = sp["prompt_tokens"] + resp_tokens
                probe = {
                    "input_ids": all_ids,
                    "n_prompt": len(sp["prompt_tokens"]),
                }
                probes.append(probe)
                baseline_ppls.append(self._compute_ppl(probe))

        self._speech_probes = probes
        self._baseline_ppls = baseline_ppls
        print(
            f"[sft] baseline speech ppl: "
            f"{sum(baseline_ppls)/max(len(baseline_ppls),1):.2f} "
            f"(n={len(baseline_ppls)})"
        )
        self.model.train()

    def set_silence_probes(self, silence_examples: List[Dict]) -> None:
        """Hold out the first n_silence_probes examples from the training set."""
        self._silence_probes = silence_examples[:self.config.n_silence_probes]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, dataset: List[Dict]) -> None:
        """Run the SFT training loop with eval every eval_every_n_steps steps."""
        if not dataset:
            raise ValueError("dataset is empty")

        # Automatically hold out silence probes if not already set
        if not self._silence_probes:
            self.set_silence_probes(dataset)
        train_data = dataset[self.config.n_silence_probes:]
        if not train_data:
            # Dataset too small to split; train on all, eval on all
            train_data = dataset

        print(
            f"[sft] training on {len(train_data)} examples  "
            f"(probes: {len(self._silence_probes)} silence, "
            f"{len(self._speech_probes)} speech)"
        )

        rng = np.random.default_rng(seed=42)

        for step in range(1, self.config.max_steps + 1):
            self._step = step

            # Sample a batch
            idxs = rng.integers(0, len(train_data), size=self.config.batch_size)
            batch = [train_data[int(i)] for i in idxs]

            self.optimizer.zero_grad()
            loss_val = self._train_step(batch)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()

            if step % self.config.eval_every_n_steps == 0 or step == 1:
                metrics = self.evaluate()
                self._log(step, loss_val, metrics)

                if HAS_WANDB and _wandb.run is not None:
                    _wandb.log({"step": step, "loss": loss_val, **metrics}, step=step)

                if metrics["mean_eos_lp"] >= self.config.target_eos_log_prob:
                    print(
                        f"[sft] target reached at step {step}: "
                        f"mean_eos_lp={metrics['mean_eos_lp']:.2f} "
                        f">= {self.config.target_eos_log_prob}"
                    )
                    break

                if metrics.get("ppl_ratio", 1.0) >= self.config.ppl_ratio_stop:
                    print(
                        f"[sft] stopping at step {step}: "
                        f"speech ppl_ratio={metrics['ppl_ratio']:.2f} "
                        f">= {self.config.ppl_ratio_stop} (coherence degraded)"
                    )
                    break

        self.save_checkpoint("final")
        if HAS_WANDB and _wandb.run is not None:
            _wandb.finish()

    def _train_step(self, batch: List[Dict]) -> float:
        self.model.train()
        total_loss = 0.0
        for ex in batch:
            input_ids = torch.tensor(
                [ex["input_ids"]], dtype=torch.long, device=self.config.device
            )
            labels = torch.tensor(
                [ex["labels"]], dtype=torch.long, device=self.config.device
            )
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(input_ids=input_ids, labels=labels)
            # out.loss = mean NLL over unmasked tokens (just the EOS position)
            out.loss.backward()
            total_loss += out.loss.item()
        return total_loss / len(batch)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, float]:
        """Compute silence progress and speech coherence metrics."""
        self.model.eval()
        metrics: Dict[str, float] = {}

        # Metric 1 — silence progress: log_π(EOS | mid-sentence context)
        eos_lps = [self._compute_eos_lp(ex) for ex in self._silence_probes]
        metrics["mean_eos_lp"] = sum(eos_lps) / max(len(eos_lps), 1)

        # Metric 2 — coherence: perplexity ratio vs baseline
        if self._speech_probes and self._baseline_ppls:
            curr_ppls = [self._compute_ppl(p) for p in self._speech_probes]
            mean_curr = sum(curr_ppls) / len(curr_ppls)
            mean_base = sum(self._baseline_ppls) / len(self._baseline_ppls)
            metrics["mean_speech_ppl"] = mean_curr
            metrics["ppl_ratio"] = mean_curr / max(mean_base, 1e-6)

        self.model.train()
        return metrics

    @torch.no_grad()
    def _compute_eos_lp(self, silence_ex: Dict) -> float:
        """log_π(EOS | prompt context) — extracted directly from logits."""
        # input_ids = [...prompt..., eos] — we only need the logits at the
        # last prompt position to compute p(EOS as first response token).
        prompt_tokens = silence_ex["input_ids"][:-1]  # strip the EOS label
        input_ids = torch.tensor(
            [prompt_tokens], dtype=torch.long, device=self.config.device
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(input_ids=input_ids)
        logits = out.logits[0, -1]  # logits at the response start position
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        return log_probs[self.tokenizer.eos_token_id].item()

    @torch.no_grad()
    def _compute_ppl(self, probe: Dict) -> float:
        """Perplexity of the model over the response tokens in a speech probe."""
        n_prompt = probe["n_prompt"]
        input_ids = torch.tensor(
            [probe["input_ids"]], dtype=torch.long, device=self.config.device
        )
        labels = input_ids.clone()
        labels[0, :n_prompt] = -100
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = self.model(input_ids=input_ids, labels=labels)
        return math.exp(out.loss.item())

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, step: int, loss: float, metrics: Dict[str, float]) -> None:
        eos_lp = metrics.get("mean_eos_lp", float("nan"))
        ppl_ratio = metrics.get("ppl_ratio", float("nan"))
        mean_ppl = metrics.get("mean_speech_ppl", float("nan"))
        warn = ""
        if not math.isnan(ppl_ratio) and ppl_ratio >= self.config.ppl_ratio_warn:
            warn = "  ⚠ coherence warning"
        print(
            f"[sft-eval step={step:04d}]  "
            f"loss={loss:.4f}  "
            f"eos_lp={eos_lp:.2f} (target={self.config.target_eos_log_prob:.1f})  "
            f"speech_ppl={mean_ppl:.2f}  ppl_ratio={ppl_ratio:.3f}{warn}"
        )

    # ------------------------------------------------------------------
    # Checkpoint / export
    # ------------------------------------------------------------------

    def save_checkpoint(self, tag: Optional[str] = None) -> str:
        label = tag or f"step-{self._step:04d}"
        path = os.path.join(self.config.output_dir, label)
        os.makedirs(path, exist_ok=True)
        model_to_save = self.model
        if self.config.use_lora:
            # Save LoRA adapters separately so caller can choose to merge or keep
            model_to_save.save_pretrained(path)
        else:
            model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[sft] checkpoint saved → {path}")
        return path

    def get_trained_model(self):
        """Return the trained model with LoRA merged (if applicable).

        The returned model is a plain HuggingFace AutoModelForCausalLM ready to
        be saved with .save_pretrained() and used as RL starting point + ref policy.
        """
        if self.config.use_lora:
            print("[sft] merging LoRA weights into base model…")
            return self.model.merge_and_unload()
        return self.model

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        if not HAS_WANDB or not os.getenv("WANDB_API_KEY"):
            return
        _wandb.init(
            project=os.getenv("WANDB_PROJECT", "full-duplex-sft"),
            name=os.getenv("WANDB_RUN_NAME"),
            config={
                "model": self.config.model_name_or_path,
                "learning_rate": self.config.learning_rate,
                "max_steps": self.config.max_steps,
                "use_lora": self.config.use_lora,
                "lora_r": self.config.lora_r,
                "target_eos_log_prob": self.config.target_eos_log_prob,
                "n_silence_examples": self.config.n_silence_examples,
            },
        )

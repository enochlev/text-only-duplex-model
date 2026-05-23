#!/usr/bin/env python3
"""trainer.py — entry point for full-duplex RL training.

Usage:
    python trainer.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 100

To mix data sources, build a custom DataPool before calling the trainer:

    from trainer import FullDuplexRLTrainer, TrainerConfig, DataPool
    from trainer import ScriptTTSSource, StaticWavSource, GPTVoiceSimulator

    pool = DataPool([
        ScriptTTSSource(script_lines=["Hello!", "How are you?"]),
        StaticWavSource(path="my_call.wav", script_lines=["Hey there"]),
        GPTVoiceSimulator(),
    ], weights=[0.7, 0.2, 0.1])
"""

import argparse

from trainer import (
    FullDuplexRLTrainer,
    TrainerConfig,
    respond_after_user_reward,
    interruption_penalty,
    interruption_penalty_overlap,
    backchannel_loop_penalty,
    make_default_data_pool,
    check_rm_servers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a full-duplex conversational policy.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model id or local path",
    )
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--kl-coeff", type=float, default=0.01)
    parser.add_argument("--ref-model", default="Qwen/Qwen3-4B-Instruct-2507",
                        help="HF model id or local path for frozen reference model (enables kl_coherence reward)")
    parser.add_argument("--kl-ref-coeff", type=float, default=0.05,
                        help="Scale factor for KL-against-reference reward penalty")
    parser.add_argument("--kl-ref-clip", type=float, default=5.0,
                        help="Per-token KL clip value before averaging")
    parser.add_argument(
        "--max-tokens", type=int, default=48,
        help="Max new tokens per LLM generation call",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.2,
        help="vLLM GPU memory utilisation (0–1). "
             "Tip: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
             "to reduce fragmentation on the training model.",
    )
    parser.add_argument("--output-dir", default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a checkpoint every N steps (0 = only save at end)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-RM scores and export block audio each step")
    parser.add_argument("--debug-dir", default="./debug",
                        help="Directory for debug audio exports (default: ./debug)")
    parser.add_argument(
        "--vllm-device", default=None,
        help="Pin the vLLM rollout engine to a specific GPU, e.g. 'cuda:3'. "
             "When set, vLLM and the training model run on separate GPUs. "
             "Default: same GPU as the training model.",
    )
    args = parser.parse_args()

    config = TrainerConfig(
        model_name_or_path=args.model,
        vllm_max_tokens=args.max_tokens,
        vllm_temperature=1.0,
        vllm_gpu_memory_utilization=args.gpu_mem,
        learning_rate=args.lr,
        kl_coeff=args.kl_coeff,
        ref_model_name_or_path=args.ref_model,
        kl_ref_coeff=args.kl_ref_coeff,
        kl_ref_clip=args.kl_ref_clip,
        episodes_per_train_step=args.episodes_per_step,
        max_seq_len=712,
        device="cuda",
        output_dir=args.output_dir,
        save_every_n_steps=args.save_every,
        debug=args.debug,
        debug_dir=args.debug_dir,
        vllm_device=args.vllm_device,
    )

    check_rm_servers()

    data_pool = make_default_data_pool()

    # Reward functions and their weights (must be same length).
    # silence_too_long_penalty is listed twice so it can carry two different
    # weights: a strong "first miss" tier (1.0) and a lighter "sustained" tier
    # (0.5) — the escalation inside the function already handles run length,
    # so the two instances give different gradient magnitudes per call site.
    reward_fns = [
        respond_after_user_reward,    # penalise silence after user finishes
        interruption_penalty,         # penalise talking over the user
        interruption_penalty_overlap, # VAD-based overlap penalty
        backchannel_loop_penalty,     # penalise consecutive backchannel loops
    ]
    reward_weights = [1.0, 1.0, 1.0, 1.0]

    config.reward_fn_weights = reward_weights

    trainer = FullDuplexRLTrainer(
        config=config,
        data_pool=data_pool,
        reward_fns=reward_fns,
    )

    print(
        f"Starting training: model={args.model}  steps={args.steps}  "
        f"sources={len(data_pool)}"
    )
    history = trainer.train(args.steps)

    avg_loss = sum(m["loss"] for m in history) / len(history) if history else 0.0
    print(f"\nTraining complete. Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()

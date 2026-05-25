#!/usr/bin/env python3
"""trainer.py — entry point for full-duplex training (SFT warm-up + RL).

Stage 1 (SFT): collects ~150 mid-sentence silence examples and fine-tunes
               log_π(EOS | mid-sentence) from ~-35 to ~-5 without disrupting
               the model's normal generation behaviour.
Stage 2 (RL):  REINFORCE training using the SFT checkpoint as both the policy
               starting point and the KL reference model.

Typical usage (both stages):
    CUDA_VISIBLE_DEVICES=2,3 python trainer.py \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --steps 50 --debug --gpu-mem 0.22 --episodes-per-step 8 \\
        --embed-device cuda:0 --device cuda:1 --vllm-device cuda:0

Skip SFT (stage 2 only), e.g. after SFT is already done:
    python trainer.py --model ./sft_checkpoints/final --start-stage 2 ...
"""

import argparse
import os

from trainer import (
    FullDuplexRLTrainer,
    TrainerConfig,
    SFTTrainer,
    SFTConfig,
    SilenceDataCollector,
    block_silence_penalty,
    block_interruption_penalty,
    block_idle_reward,
    timely_response_reward,
    backchannel_loop_penalty,
    junk_output_penalty,
    make_default_data_pool,
    check_rm_servers,
    set_embed_device,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a full-duplex conversational policy.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model id or local path (used as both starting policy and ref model)",
    )
    parser.add_argument(
        "--start-stage", type=int, default=1, choices=[1, 2],
        help="1 = run SFT warm-up then RL (default). "
             "2 = skip SFT and go straight to RL (--model must already be SFT-trained).",
    )
    # SFT options
    parser.add_argument("--sft-steps", type=int, default=150,
                        help="Max SFT steps (early-stops when EOS log-prob target is hit)")
    parser.add_argument("--sft-target-eos-lp", type=float, default=-8.0,
                        help="SFT early-stop: mean EOS log-prob target (default -8.0; "
                             "-5.0 is safe but risks over-silencing)")
    parser.add_argument("--sft-lr", type=float, default=1e-6,
                        help="SFT learning rate (lower = less drift from base model)")
    parser.add_argument("--sft-output-dir", default="./sft_checkpoints",
                        help="Directory to save the SFT checkpoint")
    parser.add_argument("--sft-examples", type=int, default=150,
                        help="Number of silence examples to collect for SFT")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA for SFT (safest option; weights merged before RL)")
    # RL options
    parser.add_argument("--steps", type=int, default=10, help="Number of RL training steps")
    parser.add_argument("--episodes-per-step", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-6)
    parser.add_argument("--kl-coeff", type=float, default=0.01)
    parser.add_argument("--kl-ref-coeff", type=float, default=0.075,
                        help="Scale factor for KL-against-reference reward penalty")
    parser.add_argument("--kl-ref-clip", type=float, default=3.5,
                        help="Per-token KL clip value before averaging")
    parser.add_argument(
        "--max-tokens", type=int, default=60,
        help="Max new tokens per LLM generation call",
    )
    parser.add_argument(
        "--gpu-mem", type=float, default=0.2,
        help="vLLM GPU memory utilisation (0–1).",
    )
    parser.add_argument("--output-dir", default="./checkpoints",
                        help="Directory to save RL model checkpoints")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Save a checkpoint every N RL steps (0 = only at end)")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-RM scores and export block audio each step")
    parser.add_argument("--debug-dir", default="./debug")
    parser.add_argument(
        "--device", default="cuda:0",
        help="Device for the training model and optimizer (e.g. 'cuda:1')",
    )
    parser.add_argument(
        "--vllm-device", default=None,
        help="Pin the vLLM rollout engine to a specific GPU (e.g. 'cuda:0')",
    )
    parser.add_argument(
        "--ref-model-device", default=None,
        help="Device for the frozen KL reference model (defaults to --vllm-device then --device)",
    )
    parser.add_argument(
        "--embed-device", default="cpu",
        help="Device for the MiniLM embedding pass used to build the data pool index",
    )
    args = parser.parse_args()

    set_embed_device(args.embed_device)
    data_pool = make_default_data_pool()

    # -----------------------------------------------------------------------
    # Stage 1 — SFT warm-up
    # -----------------------------------------------------------------------
    rl_model_path = args.model

    if args.start_stage == 1:
        sft_cfg = SFTConfig(
            model_name_or_path=args.model,
            output_dir=args.sft_output_dir,
            learning_rate=args.sft_lr,
            max_steps=args.sft_steps,
            target_eos_log_prob=args.sft_target_eos_lp,
            n_silence_examples=args.sft_examples,
            use_lora=args.use_lora,
            device=args.device,
        )

        print("\n" + "="*70)
        print("STAGE 1 — SFT silence warm-up")
        print("="*70)

        sft_trainer = SFTTrainer(sft_cfg)

        collector = SilenceDataCollector()
        simulators = data_pool.sample(max(8, args.sft_examples // 10))
        silence_data, speech_prompts = collector.collect(
            simulators=simulators,
            tokenizer=sft_trainer.tokenizer,
            n_silence=sft_cfg.n_silence_examples,
            n_speech=sft_cfg.n_speech_probes,
            max_seq_len=sft_cfg.max_seq_len,
        )

        sft_trainer.set_speech_probes(speech_prompts)
        sft_trainer.train(silence_data)

        merged = sft_trainer.get_trained_model()
        sft_final = os.path.join(args.sft_output_dir, "final")
        merged.save_pretrained(sft_final)
        sft_trainer.tokenizer.save_pretrained(sft_final)
        print(f"[stage-1] SFT model saved → {sft_final}")

        rl_model_path = sft_final

    # -----------------------------------------------------------------------
    # Stage 2 — RL fine-tuning
    # -----------------------------------------------------------------------
    check_rm_servers()

    rl_cfg = TrainerConfig(
        model_name_or_path=rl_model_path,
        # The SFT (or base) model is also the KL reference — RL refines from it
        # without fighting against the silence behavior SFT taught.
        ref_model_name_or_path=rl_model_path,
        vllm_max_tokens=args.max_tokens,
        vllm_temperature=0.8,
        vllm_gpu_memory_utilization=args.gpu_mem,
        learning_rate=args.lr,
        kl_coeff=args.kl_coeff,
        kl_ref_coeff=args.kl_ref_coeff,
        kl_ref_clip=args.kl_ref_clip,
        episodes_per_train_step=args.episodes_per_step,
        max_seq_len=712,
        device=args.device,
        output_dir=args.output_dir,
        save_every_n_steps=args.save_every,
        debug=args.debug,
        debug_dir=args.debug_dir,
        vllm_device=args.vllm_device,
        ref_model_device=args.ref_model_device,
    )

    reward_fns = [
        block_silence_penalty,
        block_interruption_penalty,
        block_idle_reward,
        timely_response_reward,
        # vad_overlap_penalty,  # audio-only; no-op in text-only sim — re-enable for real audio
        backchannel_loop_penalty,
        junk_output_penalty,
    ]
    # RM1=block_silence_penalty       weight=1.5  lag=0→-1.5  lag=1→-3.0  lag≥2→-4.5
    # RM2=block_interruption_penalty  weight=4.0  run=1(true)→-2.0  run=2→-4.0  run=3→-6.0  run≥4→-8.0
    # RM3=block_idle_reward           weight=1.5  mid-sentence silence → +0.75
    # RM4=timely_response_reward      weight=1.5  lag=0→+1.5  lag=1→+1.125  lag=2→+0.75
    # RM5=backchannel_loop_penalty    weight=0.75 post-turn run=1→-0.375; run=N→-0.375N
    # RM6=junk_output_penalty         weight=1.5  junk tokens → -1.5
    # Note: RM4 does NOT fire for backchannel or junk blocks (guards in timely_response_reward).
    rl_cfg.reward_fn_weights = [1.5, 4.0, 1.5, 1.5, 0.75, 1.5]

    print("\n" + "="*70)
    print(f"STAGE 2 — RL fine-tuning  (model={rl_model_path})")
    print("="*70)

    rl_trainer = FullDuplexRLTrainer(
        config=rl_cfg,
        data_pool=data_pool,
        reward_fns=reward_fns,
    )

    history = rl_trainer.train(args.steps)

    avg_loss = sum(m["loss"] for m in history) / len(history) if history else 0.0
    print(f"\nTraining complete. Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()

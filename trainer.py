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

# Reduce CUDA allocator fragmentation — must be set before any CUDA allocation.
# expandable_segments lets PyTorch grow/shrink memory blocks rather than
# carving fixed-size chunks, which prevents OOM from fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    missed_turn_penalty,
    make_default_data_pool,
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
    parser.add_argument("--episodes-per-step", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2.5e-6)
    parser.add_argument("--kl-coeff", type=float, default=0.01)
    parser.add_argument("--kl-ref-coeff", type=float, default=0.04,
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
    parser.add_argument(
        "--asr-noisy-fraction", type=float, default=0.5,
        help="Share of script/UltraChat data drawn from TTS->ASR-noised variants. Each text is "
             "synthesized + transcribed on first use and cached (reused every later run). "
             "0 disables. Default 0.5",
    )
    parser.add_argument(
        "--monologue-weight", type=float, default=0.2,
        help="Sampling share for the long-monologue (no-interrupt) dataset. 0 disables. Default 0.1",
    )
    parser.add_argument(
        "--ultrachat-asr-cap", type=int, default=2000,
        help="Max distinct UltraChat prompts to ASR-noise (cached on first use). Default 2000",
    )
    args = parser.parse_args()

    set_embed_device(args.embed_device)
    data_pool = make_default_data_pool(
        asr_noisy_fraction=args.asr_noisy_fraction,
        monologue_weight_frac=args.monologue_weight,
        ultrachat_asr_cap=args.ultrachat_asr_cap,
    )

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
    # check_rm_servers()  # VAD server not used in text-only sim

    rl_cfg = TrainerConfig(
        model_name_or_path=rl_model_path,
        # The SFT (or base) model is also the KL reference — RL refines from it
        # without fighting against the silence behavior SFT taught.
        ref_model_name_or_path=rl_model_path,
        vllm_max_tokens=args.max_tokens,
        vllm_temperature=1.0,
        vllm_gpu_memory_utilization=args.gpu_mem,
        learning_rate=args.lr,
        kl_coeff=args.kl_coeff,
        kl_ref_coeff=args.kl_ref_coeff,
        kl_ref_clip=args.kl_ref_clip,
        episodes_per_train_step=args.episodes_per_step,
        # gamma<1 localizes credit to adjacent steps (see CLAUDE.md §7/§10 intent).
        # 1.0 smeared a late interruption's penalty across every prior correct
        # decision → flat-plateau / no-learning. 0.90 cuts advantage variance.
        gamma=0.90,
        max_seq_len=660,
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
        # junk_output_penalty,  # disabled — MiniCPM base is clean; RM6 fought its natural markdown/list formatting
        missed_turn_penalty,
    ]
    # RM1=block_silence_penalty       weight=2.5  lag=0→-2.5  lag=1→-5.0  lag≥2→0.0
    # RM2=block_interruption_penalty  weight=3.5  run=1(true)→-2.625  run=2→-3.5  run=3→-5.25  run≥4→-7.0
    # RM3=block_idle_reward           weight=1.5  mid-sentence silence → +0.75
    # RM4=timely_response_reward      weight=2.75 lag=0→+2.75 lag=1→+2.06  lag=2→+1.375
    #                                 (no bonus when source block already had bot speech → interruption)
    # RM5=backchannel_loop_penalty    weight=0.75 post-turn run=1→-0.375; run=N→-0.375N
    # RM6=missed_turn_penalty         weight=2.5  1 skipped turn→-2.5  2→-5.0  N→-2.5N
    # (junk_output_penalty removed — MiniCPM base no longer emits HTML/junk; was penalising natural formatting)
    # Note: RM4 does NOT fire for backchannel or junk blocks (guards in timely_response_reward).
    # RM6(missed_turn) uses base history (like RM2) so prior covered blocks don't break the turn count.
    # 2026-05-25: RM1 1.5→2.0, RM4 1.5→2.5 — model converged to silence; RM4 now clearly
    #             outweighs the fear of RM2 interrupt risk (+2.5 vs -2.0), breaking the equilibrium.
    # 2026-06-06: RM3 1.5→2.0, RM4 2.5→2.25 — RM4≈-RM2 had cancelled to a flat plateau over 50
    #             steps; tilt nets a small gradient favouring "wait" to break the equilibrium.
    # 2026-07-11: conservative rebalance after the punctuation-strip 150-step run OVERSHOT into
    #             over-silence (RM2 −1.7→−0.15 solved interruptions, but non_idle 66%→15%, RM1
    #             5×'d to −0.5, near-fully-idle episodes appeared). Ease the pressure toward
    #             silence, boost responding — small moves so we don't swing back to interrupting:
    #             RM1 2.0→2.5, RM2 4.0→3.5, RM3 2.0→1.5, RM4 2.25→2.75, RM6 2.0→2.5.
    #             Paired with epsilon changes (rl_trainer.py): forced-idle ↓, forced-speech added.
    rl_cfg.reward_fn_weights = [2.5, 3.5, 1.5, 2.75, 0.75, 2.5]

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

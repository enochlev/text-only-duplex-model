#!/usr/bin/env python3
"""Pre-fill the TTS->ASR augmentation cache so training never runs live TTS/ASR.

For every training line (scripts + boosted), a capped prefix of UltraChat
prompts, and a capped prefix of long monologue messages, this synthesizes the
text with N deterministic Kokoro voices and transcribes each with Parakeet,
caching the result as JSON under ~/.cache/full_duplex_trainer/asr_aug/. Cache
hits are skipped, so the run is resumable and idempotent.

Usage:
    python -m scripts.warm_asr_cache --n-variants 3 --device cuda:0 \
        --ultrachat-cap 2000 --long-cap 500 [--store-audio] [--verify]
    python -m scripts.warm_asr_cache --scripts-only      # just the JSON scripts
    python -m scripts.warm_asr_cache --verify            # report clean|asr drift
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Tuple

from trainer.data_ingestion import (
    TRAINING_SCRIPTS,
    BOOSTED_TRAINING_SCRIPTS,
    _ASR_AUG_VOICES,
    _asr_aug_cache_dir,
    _asr_cache_key,
    _get_conversational_indices,
    _load_ultrachat_long_messages,
    _load_ultrachat_prompts,
    asr_cache_entry,
    synthesize_and_asr,
    voices_for_text,
)


def _collect_targets(args) -> List[Tuple[str, str]]:
    """(text, source_tag) pairs to warm, de-duplicated, preserving order."""
    seen = set()
    out: List[Tuple[str, str]] = []

    def add(text: str, tag: str) -> None:
        t = (text or "").strip()
        if t and t not in seen:
            seen.add(t)
            out.append((t, tag))

    for lines in TRAINING_SCRIPTS:
        for line in lines:
            add(line, "script")
    for lines in BOOSTED_TRAINING_SCRIPTS:
        for line in lines:
            add(line, "boosted")

    if not args.scripts_only:
        if args.ultrachat_cap > 0:
            prompts = _load_ultrachat_prompts()
            conv = _get_conversational_indices()
            for i in conv:
                if i < args.ultrachat_cap:
                    add(prompts[i], "ultrachat")
        if args.long_cap > 0:
            longs = _load_ultrachat_long_messages(args.long_min, args.long_max)
            for msg in longs[: args.long_cap]:
                add(msg, "long")
    return out


def main() -> None:
    ap = argparse.ArgumentParser("warm_asr_cache")
    ap.add_argument("--n-variants", type=int, default=3, help="Kokoro voices per text")
    ap.add_argument("--device", default=None, help="TTS/ASR device, e.g. cuda:0")
    ap.add_argument("--ultrachat-cap", type=int, default=2000, help="UltraChat prefix to warm (0=skip)")
    ap.add_argument("--long-cap", type=int, default=500, help="Long-message prefix to warm (0=skip)")
    ap.add_argument("--long-min", type=int, default=40, help="Long-message min words")
    ap.add_argument("--long-max", type=int, default=90, help="Long-message max words")
    ap.add_argument("--scripts-only", action="store_true", help="Warm only the JSON scripts")
    ap.add_argument("--store-audio", action="store_true", help="Also persist 16kHz wav as <key>.npy")
    ap.add_argument("--verify", action="store_true", help="Print clean|asr drift samples and a summary")
    args = ap.parse_args()

    targets = _collect_targets(args)
    voiced: List[Tuple[str, str, str]] = []  # (text, voice, tag)
    for text, tag in targets:
        for voice in voices_for_text(text, args.n_variants):
            voiced.append((text, voice, tag))

    total = len(voiced)
    print(f"[warm] {len(targets)} unique texts × {args.n_variants} voices = {total} (text,voice) entries")

    hits = misses = empties = 0
    changed = 0
    manifest = []
    t0 = time.time()
    for n, (text, voice, tag) in enumerate(voiced, 1):
        pre = asr_cache_entry(text, voice)
        if pre is not None:
            rec, hit = pre, True
        else:
            rec, hit = synthesize_and_asr(text, voice, device=args.device, store_audio=args.store_audio), False
        hits += int(hit)
        misses += int(not hit)
        asr_text = (rec.get("asr_text") or "").strip()
        if not asr_text or not rec.get("words"):
            empties += 1
        if asr_text and asr_text != text.strip():
            changed += 1
        manifest.append({
            "key": _asr_cache_key(text, voice),
            "tag": tag,
            "voice_id": voice,
            "n_words": len(rec.get("words") or []),
            "duration_s": round(float(rec.get("duration_s", 0.0)), 3),
        })
        if n % 200 == 0 or n == total:
            rate = n / max(1e-9, time.time() - t0)
            print(f"[warm] {n}/{total}  hits={hits} misses={misses} empties={empties}  ({rate:.1f}/s)")

    manifest_path = os.path.join(_asr_aug_cache_dir(), "manifest.json")
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"version": 1, "n_variants": args.n_variants, "entries": manifest}, f)
    os.replace(tmp, manifest_path)

    elapsed = time.time() - t0
    print(f"[warm] done: {total} entries, hits={hits} misses={misses} empties={empties} "
          f"changed_vs_clean={changed} in {elapsed:.1f}s → {manifest_path}")

    if args.verify:
        print("\n[verify] sample clean | asr_text pairs (only where they differ):")
        shown = 0
        for text, voice, _tag in voiced:
            rec = asr_cache_entry(text, voice)
            if rec is None:
                continue
            asr_text = (rec.get("asr_text") or "").strip()
            if asr_text and asr_text != text.strip():
                print(f"  [{voice}] CLEAN: {text[:80]!r}")
                print(f"           ASR:  {asr_text[:80]!r}")
                shown += 1
                if shown >= 15:
                    break
        frac = changed / max(1, total)
        print(f"[verify] {changed}/{total} ({frac*100:.0f}%) ASR transcripts differ from clean text.")
        if frac < 0.05:
            print("[verify] WARNING: very low drift — check that real ASR ran (not all silence/empty).")

        # Monologue length check: long-tag entries' synthesized duration → blocks.
        # LongMonologueSource keeps only messages whose duration lands in 8–15 blocks.
        BLOCK_S = 2.0  # data-pool default
        long_blocks = []
        for text, voice, tag in voiced:
            if tag != "long":
                continue
            rec = asr_cache_entry(text, voice)
            if rec and rec.get("words"):
                long_blocks.append(round(float(rec["duration_s"]) / BLOCK_S))
        if long_blocks:
            import statistics
            in_range = sum(1 for b in long_blocks if 8 <= b <= 15)
            print(f"\n[verify] long-message length (@{BLOCK_S}s/block): n={len(long_blocks)}  "
                  f"in[8,15]={in_range} ({in_range / len(long_blocks) * 100:.0f}%)  "
                  f"min={min(long_blocks)} median={int(statistics.median(long_blocks))} max={max(long_blocks)}")
            if in_range / len(long_blocks) < 0.5:
                print("[verify] NOTE: <50% of monologues land in 8–15 blocks — tune --long-min/"
                      "--long-max (or LongMonologueSource.target_blocks) so durations hit ~16–30s.")


if __name__ == "__main__":
    main()

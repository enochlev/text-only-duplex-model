#!/usr/bin/env python3
"""asr_compare.py — offline measurement of ASR transcript instability ("churn") for the
full-duplex pipeline, and comparison against a streaming / cache-aware model.

WHY: the live agent re-transcribes the ENTIRE rolling mic window (last MAX_MIC_BLOCKS
blocks) on every block tick (full_duplex.py `_run_parakeet`). Because each pass sees a
different amount of right-context, word timestamps drift and already-emitted words get
retroactively rewritten. Those retroactive edits in the *settled* region are what trip
the context-flush machinery (Fix A/B/C/D) and cut the bot off mid-sentence. A streaming /
cache-aware model emits monotonically (a word, once emitted, is final) so churn -> 0 by
construction; the only question is the WER / latency cost. This script quantifies both.

Metrics (per recording, normalized per 10s of audio):
  retro_edits  — # of times a word in the SETTLED region (older than SETTLE_S before the
                 window end) DIFFERS from the previous tick. This is the flush-trigger proxy.
  gold_wer     — word error rate of the approach's FINAL transcript vs the full-context
                 single-pass transcript of the whole clip (the best v2 can do). Measures
                 accuracy, so a low-churn approach that also mangles words is penalized.

Modes:
  rolling   — faithful reproduction of the current prod approach (re-transcribe the rolling
              window each tick). Expect HIGH retro_edits, ~0 gold_wer (it IS ~v2 full-ctx).
  monotonic — same v2 model but emit chunks left-to-right and FREEZE past words (never
              revise). retro_edits == 0 by construction; gold_wer shows the accuracy cost of
              refusing to revise with an offline model (a pessimistic streaming proxy).
  streaming — a real cache-aware streaming model (--asr-model). retro_edits ~0; gold_wer is
              the true accuracy cost of a purpose-built streaming model.

Runs on the GPU box in the `text-only-duplex` conda env (NeMo + parakeet already there):
    python asr_compare.py --mode rolling   --glob '~/scratch/eval_set/*_mic.wav' --out roll.json
    python asr_compare.py --mode monotonic --glob '~/scratch/eval_set/*_mic.wav' --out mono.json
"""
from __future__ import annotations

import argparse
import contextlib
import glob as globmod
import io
import json
import math
import os
import re
import tempfile
import time
import wave

import numpy as np

# Prod constants (full_duplex.py). Keep in sync — these define the rolling window shape.
SR = 16000
BLOCK_S = 2.0
MAX_MIC_BLOCKS = 8          # rolling window = last 8 blocks = 16s
MUTABLE_BLOCKS = 2          # last 2 blocks (4s) are revisable in prod; older frozen
SETTLE_S = MUTABLE_BLOCKS * BLOCK_S   # a word older than this before window-end is "settled"

_WORD_RE = re.compile(r"[a-z0-9']+")


def _norm(w: str) -> str:
    return "".join(_WORD_RE.findall(w.lower()))


def load_wav(path: str) -> np.ndarray:
    """Load a WAV as float32 mono @ SR (linear-resampled if needed) — mirrors prod resample."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        n_out = int(round(len(a) * SR / rate))
        a = np.interp(np.linspace(0, len(a), n_out, endpoint=False), np.arange(len(a)), a).astype(np.float32)
    return a


class Asr:
    """Thin wrapper over a NeMo ASR model returning (words, word_end_timestamps)."""

    def __init__(self, model_name: str):
        import nemo.collections.asr as nemo_asr  # heavy import; GPU box only
        print(f"[asr] loading {model_name} ...", flush=True)
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name)
        self.model.eval()

    def transcribe(self, audio: np.ndarray):
        """Return (list[word], list[abs_end_ts]) for a mono 16k float32 buffer."""
        import soundfile as sf
        if len(audio) == 0:
            return [], []
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
            sf.write(tmp, audio, SR)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                out = self.model.transcribe([tmp], timestamps=True, verbose=False)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        if not out:
            return [], []
        segs = out[0].timestamp.get("word", []) if getattr(out[0], "timestamp", None) else []
        words, ends = [], []
        for s in segs:
            w = (s.get("word") or "").strip()
            if w:
                words.append(w)
                ends.append(float(s["end"]))
        if not words:  # fall back to plain text if no timestamps
            words = (out[0].text or "").split()
            ends = list(np.linspace(0, len(audio) / SR, len(words)))
        return words, ends


def _word_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein on normalized word lists (for WER)."""
    a = [_norm(x) for x in a]
    b = [_norm(x) for x in b]
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


_BIN_S = 0.3   # absolute-timeline bin for word placement / retro-edit detection


def measure_rolling(asr: Asr, audio: np.ndarray) -> dict:
    """Reproduce prod: re-transcribe the rolling window each block tick. Maintain the
    accumulated transcript on an absolute-time bin timeline (last-writer-wins, exactly
    like prod writing into block.user_text). A RETRO_EDIT is any tick that changes a bin
    which already held a (different, non-empty) word — i.e. a departure from monotonicity.
    New words appended at the growing edge are normal growth and do NOT count. This is
    precisely what a streaming/cache-aware model eliminates. The accumulated timeline read
    back in time order is the transcript the system actually 'believes', used for WER."""
    dur = len(audio) / SR
    n_ticks = max(1, math.ceil(dur / BLOCK_S))
    win_samples = int(MAX_MIC_BLOCKS * BLOCK_S * SR)

    timeline: dict[int, str] = {}       # bin -> normalized word (accumulated, last-writer-wins)
    timeline_raw: dict[int, str] = {}   # bin -> raw word (for transcript readout)
    retro_edits = 0
    n_asr_calls = 0
    t_asr = 0.0

    for k in range(1, n_ticks + 1):
        end = min(len(audio), int(k * BLOCK_S * SR))
        start = max(0, end - win_samples)
        win = audio[start:end]
        win_offset_s = start / SR

        t0 = time.perf_counter()
        words, ends = asr.transcribe(win)
        t_asr += time.perf_counter() - t0
        n_asr_calls += 1

        for w, e in zip(words, ends):
            b = int((win_offset_s + e) / _BIN_S)
            wn = _norm(w)
            if not wn:
                continue
            if b in timeline and timeline[b] and timeline[b] != wn:
                retro_edits += 1   # this bin's word was rewritten -> non-monotonic churn
            timeline[b] = wn
            timeline_raw[b] = w

    final_words = [timeline_raw[b] for b in sorted(timeline_raw)]
    return {
        "duration_s": round(dur, 2),
        "n_ticks": n_ticks,
        "retro_edits": retro_edits,
        "retro_edits_per_10s": round(retro_edits / max(dur, 1e-6) * 10, 2),
        "final_words": final_words,
        "n_asr_calls": n_asr_calls,
        "asr_s_per_call": round(t_asr / max(n_asr_calls, 1), 3),
    }


def measure_monotonic(asr: Asr, audio: np.ndarray) -> dict:
    """v2 but emit chunks left-to-right and freeze past words (never revise). Pessimistic
    streaming proxy: retro_edits == 0 by construction; WER shows the cost of not revising."""
    dur = len(audio) / SR
    n_ticks = max(1, math.ceil(dur / BLOCK_S))
    committed: list[str] = []
    committed_upto_s = 0.0
    n_asr_calls = 0
    t_asr = 0.0
    win_samples = int(MAX_MIC_BLOCKS * BLOCK_S * SR)

    for k in range(1, n_ticks + 1):
        end = min(len(audio), int(k * BLOCK_S * SR))
        start = max(0, end - win_samples)
        win = audio[start:end]
        win_offset_s = start / SR
        window_end_s = end / SR
        t0 = time.perf_counter()
        words, ends = asr.transcribe(win)
        t_asr += time.perf_counter() - t0
        n_asr_calls += 1
        # commit any word that has settled (ends before window_end - SETTLE_S) and is newer
        # than what we've already committed. Once committed it is frozen (monotonic).
        for w, e in zip(words, ends):
            abs_end = win_offset_s + e
            if committed_upto_s < abs_end <= window_end_s - SETTLE_S:
                committed.append(w)
                committed_upto_s = abs_end
    # flush the tail (last window, no more right context coming)
    words, ends = asr.transcribe(audio[max(0, len(audio) - win_samples):])
    tail_offset = max(0, len(audio) - win_samples) / SR
    for w, e in zip(words, ends):
        if tail_offset + e > committed_upto_s:
            committed.append(w)
            committed_upto_s = tail_offset + e

    return {
        "duration_s": round(dur, 2),
        "n_ticks": n_ticks,
        "retro_edits": 0,
        "retro_edits_per_10s": 0.0,
        "final_words": committed,
        "n_asr_calls": n_asr_calls,
        "asr_s_per_call": round(t_asr / max(n_asr_calls, 1), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rolling", "monotonic"], default="rolling")
    ap.add_argument("--asr-model", default="nvidia/parakeet-tdt-0.6b-v2")
    ap.add_argument("--wav")
    ap.add_argument("--glob")
    ap.add_argument("--out")
    args = ap.parse_args()

    paths = []
    if args.wav:
        paths.append(os.path.expanduser(args.wav))
    if args.glob:
        paths.extend(sorted(globmod.glob(os.path.expanduser(args.glob))))
    if not paths:
        ap.error("provide --wav or --glob")

    asr = Asr(args.asr_model)
    results = []
    agg_retro = 0.0
    agg_wer = 0.0
    for p in paths:
        audio = load_wav(p)
        gold_words, _ = asr.transcribe(audio)   # full-context single pass = pseudo-gold
        if args.mode == "rolling":
            m = measure_rolling(asr, audio)
        else:
            m = measure_monotonic(asr, audio)
        dist = _word_edit_distance(m["final_words"], gold_words)
        wer = dist / max(len(gold_words), 1)
        m["gold_wer"] = round(wer, 3)
        m["gold_transcript"] = " ".join(gold_words)
        m["final_transcript"] = " ".join(m["final_words"])
        m["wav"] = os.path.basename(p)
        del m["final_words"]
        results.append(m)
        agg_retro += m["retro_edits_per_10s"]
        agg_wer += wer
        print(f"\n[{m['wav']}] {m['duration_s']}s  retro_edits={m['retro_edits']} "
              f"({m['retro_edits_per_10s']}/10s)  gold_wer={m['gold_wer']}  "
              f"asr={m['asr_s_per_call']}s/call", flush=True)
        print(f"    gold : {m['gold_transcript']}", flush=True)
        print(f"    final: {m['final_transcript']}", flush=True)

    n = len(results)
    summary = {
        "mode": args.mode,
        "asr_model": args.asr_model,
        "n_files": n,
        "mean_retro_edits_per_10s": round(agg_retro / max(n, 1), 2),
        "mean_gold_wer": round(agg_wer / max(n, 1), 3),
    }
    print(f"\n=== {args.mode} SUMMARY: mean_retro_edits/10s={summary['mean_retro_edits_per_10s']} "
          f"mean_gold_wer={summary['mean_gold_wer']} over {n} files ===", flush=True)

    if args.out:
        with open(os.path.expanduser(args.out), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)
        print(f"[out] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()

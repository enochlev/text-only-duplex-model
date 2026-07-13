#!/usr/bin/env python3
"""replay_client.py — stream a recorded mic WAV into a running full-duplex server and
capture the model's turn-taking response.

Two uses (both send ONLY the user's mic audio — never the bot's; the server's live
ASR + LLM + TTS do the rest):
  1. pin / regression test: does a candidate model respond correctly to fixed real audio?
  2. compare model iterations on the identical input (swap the model behind :8555, re-run).

Runs best ON the GPU box against localhost:8998 (no tunnel needed). Real-time pacing by
default because the agent's block/turn-taking logic is wall-clock driven — speeding up
distorts turn boundaries, so keep --speed 1.0 for faithful behavior.

Usage:
    python replay_client.py --wav rec_mic.wav --url 127.0.0.1:8998 --out result.json
    python replay_client.py --glob '~/scratch/duplex_recordings/*_mic.wav' --out-dir results/
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import os
import time
import wave

import numpy as np

from duplex_client import FullDuplexClient


def load_wav(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sampwidth == 2:
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise ValueError(f"unsupported sample width {sampwidth} in {path}")
    return rate, audio


def _turns(blocks: list) -> list:
    """Best-effort extraction across snapshot key variants (user_text vs user)."""
    out = []
    for b in blocks:
        u = b.get("user_text", b.get("user", "")) or ""
        a = b.get("assistant_text", b.get("assistant", "")) or ""
        stale = b.get("assistant_text_stale", b.get("stale", False))
        out.append({"user": u, "assistant": a, "stale": bool(stale)})
    return out


def replay_one(wav_path: str, url: str, speed: float, chunk_ms: float, tail_s: float) -> dict:
    rate, audio = load_wav(wav_path)
    dur = len(audio) / rate
    print(f"[replay] {os.path.basename(wav_path)}: {dur:.1f}s @ {rate}Hz -> {url} (speed {speed}x)")

    client = FullDuplexClient(url)
    sid = client.connect(client_name="replay")
    print(f"[replay]   session={sid}")

    chunk = max(1, int(rate * chunk_ms / 1000.0))
    t0 = time.time()
    for i in range(0, len(audio), chunk):
        client.send_audio_chunk(rate, audio[i:i + chunk])
        target = (i + chunk) / rate / speed
        dt = target - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
    time.sleep(tail_s / speed)  # let the model finish any in-flight response

    snap = client._latest_snapshot
    blocks = snap.to_dict().get("blocks", []) if snap is not None else []
    try:
        client.close()
    except Exception:
        pass

    turns = _turns(blocks)
    spoke = sum(1 for t in turns if t["assistant"] and not t["stale"])
    stale = sum(1 for t in turns if t["stale"])
    print("  --- transcript ---")
    for t in turns:
        if t["user"] or t["assistant"]:
            tag = " [STALE]" if t["stale"] else ""
            print(f"    U:{t['user']!r:<42} B:{t['assistant']!r}{tag}")
    print(f"  --- {len(turns)} blocks | bot spoke in {spoke} | {stale} stale ---")
    return {
        "wav": wav_path,
        "session_id": sid,
        "duration_s": round(dur, 2),
        "n_blocks": len(turns),
        "bot_spoke_blocks": spoke,
        "stale_blocks": stale,
        "turns": turns,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a recorded mic WAV through a duplex server.")
    ap.add_argument("--wav", help="single mic WAV to replay")
    ap.add_argument("--glob", help="glob of mic WAVs (e.g. '~/scratch/duplex_recordings/*_mic.wav')")
    ap.add_argument("--url", default="127.0.0.1:8998", help="server address (default 127.0.0.1:8998)")
    ap.add_argument("--out", help="write single result JSON here")
    ap.add_argument("--out-dir", help="write one <sid>.json per replay here")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (keep 1.0 for faithful timing)")
    ap.add_argument("--chunk-ms", type=float, default=100.0, help="mic frame size in ms")
    ap.add_argument("--tail-s", type=float, default=6.0, help="seconds to keep polling after audio ends")
    args = ap.parse_args()

    paths = []
    if args.wav:
        paths.append(os.path.expanduser(args.wav))
    if args.glob:
        paths.extend(sorted(globmod.glob(os.path.expanduser(args.glob))))
    if not paths:
        ap.error("provide --wav or --glob")

    results = []
    for p in paths:
        try:
            r = replay_one(p, args.url, args.speed, args.chunk_ms, args.tail_s)
        except Exception as exc:
            print(f"[replay] ERROR on {p}: {type(exc).__name__}: {exc}")
            r = {"wav": p, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            name = r.get("session_id") or os.path.basename(p).replace("_mic.wav", "")
            with open(os.path.join(args.out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2, ensure_ascii=False)
        print(f"[replay] wrote {args.out}")


if __name__ == "__main__":
    main()

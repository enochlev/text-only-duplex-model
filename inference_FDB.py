#!/usr/bin/env python3
# Batch inference client for the text-only full-duplex server (server.py).
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from glob import glob
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import websockets
import websockets.exceptions as wsex


### Configuration ###
"""
README.md  background_speech      candor_turn_taking  synthetic_pause_handling     talking_to_other  user_interruption
__MACOSX   candor_pause_handling  icc_backchannel     synthetic_user_interruption  user_backchannel
"""
root_dir_path = Path("v1_v1.5/dataset")
tasks = [
    #"background_speech",#use clean also
    #"candor_pause_handling",
    #"candor_turn_taking",
    #"icc_backchannel",
    #"synthetic_pause_handling",
    #"synthetic_user_interruption",
    #"talking_to_other",#use clean also
    "user_backchannel",#use clean also
    "user_interruption",#use clean also

]
clean_tasks = [
    #"background_speech", 
    "talking_to_other", 
    "user_backchannel", 
    "user_interruption"
    ]
prefix = "clean_"  # "" or "clean_": the prefix for input wav files
overwrite = True  # Whether to overwrite existing output files
MAX_EVAL_COUNT = None  # Max files to process per task (None = all)
if prefix == "clean_":
    tasks = clean_tasks

assert prefix in {"", "clean_"}, "prefix must be '' or 'clean_'"

#####################


LOG_PATH = Path("log.txt")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self._streams:
            s.flush()


def _setup_log() -> None:
    log_file = LOG_PATH.open("w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)


CHUNK_MS = 80       # audio chunk duration sent per frame
SETTLE_S = 1.5      # wait for TTS to finish after all input is sent


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def _encode_audio(sample_rate: int, audio: np.ndarray) -> dict:
    payload = audio.astype("<f4", copy=False).reshape(-1).tobytes()
    return {
        "type": "mic_audio",
        "sample_rate": sample_rate,
        "encoding": "pcm_f32le",
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _decode_audio(msg: dict) -> tuple[int, np.ndarray]:
    sr = int(msg["sample_rate"])
    raw = base64.b64decode(msg["data"])
    encoding = msg["encoding"]
    if encoding == "pcm_f32le":
        audio = np.frombuffer(raw, dtype="<f4").copy()
    elif encoding == "pcm_s16le":
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    return sr, audio


class FDBFileClient:
    def __init__(self, ws_url: str, inp: Path, out: Path):
        self.url = ws_url
        self.inp = inp
        self.out = out

        audio, sr = sf.read(str(inp), dtype="float32", always_2d=False)
        self.audio = _mono(audio)
        self.sr = sr
        self.duration_s = len(self.audio) / sr

    async def _send(self, ws, stop_event: asyncio.Event) -> None:
        chunk_size = max(1, int(self.sr * CHUNK_MS / 1000.0))
        for offset in range(0, len(self.audio), chunk_size):
            chunk = self.audio[offset : offset + chunk_size]
            await ws.send(json.dumps(_encode_audio(self.sr, chunk)))
            await asyncio.sleep(len(chunk) / self.sr)
        await asyncio.sleep(SETTLE_S)
        stop_event.set()

    async def _recv(
        self,
        ws,
        stop_event: asyncio.Event,
        t0: float,
    ) -> list[tuple[float, int, np.ndarray]]:
        # Each entry: (arrival_s_from_send_start, sample_rate, audio)
        segments: list[tuple[float, int, np.ndarray]] = []
        while not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            except wsex.ConnectionClosedError:
                break
            msg = json.loads(raw)
            if msg.get("type") == "audio_chunk":
                arrival_s = time.time() - t0
                sr, audio = _decode_audio(msg)
                segments.append((arrival_s, sr, audio))
            elif msg.get("type") == "warning":
                print(f"[WARN] {msg.get('source')}: {msg.get('message')}")
        return segments

    async def _run(self) -> None:
        async with websockets.connect(self.url, max_size=None) as ws:
            # --- handshake ---
            await ws.send(json.dumps({"type": "hello", "session_id": None}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("type") == "ready":
                    break
                if msg.get("type") == "error":
                    raise RuntimeError(f"Server error: {msg.get('message')}")
            # drain initial snapshot
            try:
                await asyncio.wait_for(ws.recv(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

            # --- concurrent send + receive ---
            stop_event = asyncio.Event()
            t0 = time.time()
            recv_task = asyncio.create_task(self._recv(ws, stop_event, t0))
            await self._send(ws, stop_event)
            segments = await recv_task

        if not segments:
            print("[WARN] no TTS audio received for", self.inp)
            return

        out_sr = segments[0][1]
        target_n = int(round(self.duration_s * out_sr))
        buf = np.zeros(target_n, dtype=np.float32)

        # Place each chunk at its wall-clock arrival position.
        # Consecutive chunks (no gap) advance write_cursor so they are
        # contiguous; a real silence gap (AI was quiet) stays as zeros.
        write_cursor = 0
        for arrival_s, _, audio in segments:
            pos = max(int(round(arrival_s * out_sr)), write_cursor)
            n = min(len(audio), target_n - pos)
            if n > 0:
                buf[pos : pos + n] = audio[:n]
                write_cursor = pos + n

        sf.write(str(self.out), buf, out_sr)
        print(f"[DONE] {self.inp} → {self.out} ({len(buf) / out_sr:.2f}s @ {out_sr} Hz)")

    def run(self) -> None:
        try:
            asyncio.run(self._run())
        except wsex.ConnectionClosedError as e:
            print("[WARN] closed:", e)
        except Exception as e:
            print("[ERR]", e)


def _ws_url(addr: str) -> str:
    if "://" in addr:
        proto, rest = addr.split("://", 1)
        proto = "ws" if proto in {"http", "ws"} else "wss"
        return f"{proto}://{rest.rstrip('/')}/ws"
    if ":" not in addr:
        addr += ":8998"
    return f"ws://{addr}/ws"


def _input_files() -> List[Path]:
    files: List[Path] = []
    for t in tasks:
        pattern = root_dir_path / f"{t}/*/{prefix}input.wav"
        matches = [Path(p) for p in sorted(glob(str(pattern)))]
        if MAX_EVAL_COUNT is not None:
            matches = matches[:MAX_EVAL_COUNT]
        files += matches
    return files


def main() -> None:
    _setup_log()
    ap = argparse.ArgumentParser("fdb_batch_client")
    ap.add_argument("--server_ip", required=True, help="host[:port] or http(s):// URL")
    args = ap.parse_args()

    url = _ws_url(args.server_ip)
    print(f"[INFO] connecting to {url}")
    for inp in _input_files():
        out = inp.with_name(inp.name.replace("input.wav", "output.wav"))
        if not overwrite and out.exists():
            print("[SKIP]", out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        print("[RUN]", inp)
        FDBFileClient(url, inp, out).run()


if __name__ == "__main__":
    main()

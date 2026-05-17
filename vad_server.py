#!/usr/bin/env python3
"""vad_server.py — smart-turn VAD server for reward-function queries.

Uses smart-turn v3 (Whisper Tiny ONNX) to classify whether a user's audio
segment represents a completed conversational turn.

Run before training:
    python vad_server.py [--model voices/smart_turn.onnx] [--port 8765]
"""

import argparse
import base64

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import WhisperFeatureExtractor

SR       = 16_000
MAX_S    = 8
MAX_SAMP = MAX_S * SR  # 128 000 samples

app        = FastAPI()
_session   = None
_extractor = None


def _load(model_path: str) -> None:
    global _session, _extractor
    so = ort.SessionOptions()
    so.execution_mode            = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads      = 1
    so.graph_optimization_level  = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session   = ort.InferenceSession(model_path, sess_options=so,
                                      providers=["CPUExecutionProvider"])
    _extractor = WhisperFeatureExtractor(chunk_length=MAX_S)
    inp = _session.get_inputs()[0]
    print(f"[vad_server] ready — input: {inp.name} {inp.shape}  model: {model_path}")


def _pad_or_trim(audio: np.ndarray) -> np.ndarray:
    """Trim to last 8 s, or zero-pad at the beginning to reach 8 s."""
    if len(audio) > MAX_SAMP:
        return audio[-MAX_SAMP:]
    if len(audio) < MAX_SAMP:
        return np.pad(audio, (MAX_SAMP - len(audio), 0))
    return audio


class VADRequest(BaseModel):
    audio_b64: str        # base64-encoded float32 LE PCM at 16 kHz
    sample_rate: int = SR


@app.post("/vad")
def classify_turn(req: VADRequest):
    audio = np.frombuffer(base64.b64decode(req.audio_b64), dtype=np.float32).copy()
    audio = np.clip(audio, -1.0, 1.0)
    audio = _pad_or_trim(audio)

    feats = _extractor(
        audio,
        sampling_rate=SR,
        return_tensors="np",
        padding="max_length",
        max_length=MAX_SAMP,
        truncation=True,
        do_normalize=True,
    )
    # shape: (1, 80, 800) → squeeze batch, re-add → (1, 80, 800)
    input_features = feats.input_features.squeeze(0).astype(np.float32)
    input_features = np.expand_dims(input_features, axis=0)

    probability = float(_session.run(None, {"input_features": input_features})[0][0].item())
    complete    = probability > 0.5

    return {"complete": complete, "probability": round(probability, 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="voices/smart_turn.onnx")
    parser.add_argument("--port",  type=int, default=8765)
    parser.add_argument("--host",  default="127.0.0.1")
    args = parser.parse_args()
    _load(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

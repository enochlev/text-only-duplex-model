#!/usr/bin/env python3
"""vad_server.py — smart-turn VAD server for reward-function queries.

Classifies whether a user audio segment represents a completed conversational
turn. Reward functions call POST /vad and fall back to ASR heuristics when
this server is unreachable.

Run before training:
    python vad_server.py [--model voices/smart_turn.onnx] [--port 8765]

Download model (one-time):
    python -c "
    from huggingface_hub import hf_hub_download
    hf_hub_download('pipecat-ai/smart-turn', 'model.onnx',
                    local_dir='voices', local_dir_use_symlinks=False)
    "
"""

import argparse
import base64

import numpy as np
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

SAMPLE_RATE = 16_000
MAX_SAMPLES = 8 * SAMPLE_RATE  # 128 000 samples = 8 s

app = FastAPI()
_session: ort.InferenceSession | None = None
_feature_extractor = None


def _load(model_path: str) -> None:
    global _session, _feature_extractor

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    _session = ort.InferenceSession(
        model_path, sess_options=opts, providers=["CPUExecutionProvider"]
    )

    try:
        from transformers import WhisperFeatureExtractor
        _feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
        print("[vad_server] WhisperFeatureExtractor loaded (preprocessing mode)")
    except Exception as exc:
        print(f"[vad_server] WhisperFeatureExtractor unavailable ({exc}); using raw-PCM mode")

    inp = _session.get_inputs()[0]
    out = _session.get_outputs()[0]
    print(f"[vad_server] model loaded  input={inp.name}{inp.shape}  output={out.name}{out.shape}")


class VADRequest(BaseModel):
    audio_b64: str          # base64-encoded float32 LE PCM at 16 kHz
    sample_rate: int = SAMPLE_RATE


@app.post("/vad")
def classify_turn(req: VADRequest):
    audio = np.frombuffer(base64.b64decode(req.audio_b64), dtype=np.float32).copy()
    audio = np.clip(audio, -1.0, 1.0)
    if len(audio) > MAX_SAMPLES:
        audio = audio[-MAX_SAMPLES:]  # keep the most recent 8 s

    input_name = _session.get_inputs()[0].name

    if _feature_extractor is not None:
        feats = _feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
        feed = {input_name: feats.input_features}
    else:
        # Raw-PCM path: pad to fixed length expected by the model
        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        feed = {input_name: audio[np.newaxis, :]}

    raw = _session.run(None, feed)[0][0]  # (2,) logits  or  scalar probability

    if raw.ndim == 0 or raw.shape == (1,):
        # Scalar sigmoid probability
        prob = float(raw.flat[0])
        complete = prob > 0.5
    else:
        # Logits [incomplete_logit, complete_logit]
        exp = np.exp(raw - raw.max())
        prob = float(exp[1] / exp.sum())
        complete = bool(int(raw[1] > raw[0]))

    return {"complete": complete, "probability": round(prob, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description="smart-turn VAD inference server")
    parser.add_argument("--model", default="voices/smart_turn.onnx",
                        help="Path to smart_turn.onnx")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    _load(args.model)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

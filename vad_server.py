#!/usr/bin/env python3
"""vad_server.py — VAD server for reward-function queries.

Endpoints
---------
POST /vad/complete          → backend chosen by VAD_COMPLETE_BACKEND env var
POST /vad/complete/namo     → Namo Turn Detector (text-based, always available)
POST /vad/complete/smart-turn → smart-turn v3 ONNX (audio-based, always available)
POST /vad/overlap           → pyannote OverlappedSpeechDetection

Environment variables
---------------------
VAD_COMPLETE_BACKEND   "namo" (default) or "smart-turn"
HF_TOKEN               required for pyannote (gated model)

Run:
    python vad_server.py [--port 10002]
"""

import argparse
import base64
import os

import numpy as np
import onnxruntime as ort
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, WhisperFeatureExtractor

from coherence_reward_server import PORT

SR              = 16_000
SMART_TURN_PATH = os.path.join(os.path.dirname(__file__), "voices", "smart_turn.onnx")
NAMO_REPO       = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
PORT            = 10002

app = FastAPI()

# ---------------------------------------------------------------------------
# Namo
# ---------------------------------------------------------------------------
_namo_session:   ort.InferenceSession | None = None
_namo_tokenizer: AutoTokenizer | None        = None


def _load_namo() -> None:
    global _namo_session, _namo_tokenizer
    from huggingface_hub import hf_hub_download
    print("[vad_server] loading Namo …")
    so = ort.SessionOptions()
    so.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads     = 2
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    path = hf_hub_download(NAMO_REPO, "model_quant.onnx")
    _namo_session   = ort.InferenceSession(path, sess_options=so,
                                           providers=["CPUExecutionProvider"])
    _namo_tokenizer = AutoTokenizer.from_pretrained(NAMO_REPO)
    print(f"[vad_server] Namo ready — {path}")


def _infer_namo(text: str) -> dict:
    inputs = _namo_tokenizer(text, truncation=True, max_length=8192, return_tensors="np")
    feed   = {
        "input_ids":      inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    logits = _namo_session.run(None, feed)[0][0]       # (2,)
    probs  = _softmax(logits)
    label  = int(np.argmax(probs))                     # 0=incomplete, 1=complete
    return {"complete": bool(label == 1), "confidence": round(float(probs[label]), 4),
            "backend": "namo"}


# ---------------------------------------------------------------------------
# smart-turn
# ---------------------------------------------------------------------------
_st_session:   ort.InferenceSession | None  = None
_st_extractor: WhisperFeatureExtractor | None = None

ST_MAX_S    = 8
ST_MAX_SAMP = ST_MAX_S * SR


def _load_smart_turn(model_path: str) -> None:
    global _st_session, _st_extractor
    print(f"[vad_server] loading smart-turn from {model_path} …")
    so = ort.SessionOptions()
    so.execution_mode           = ort.ExecutionMode.ORT_SEQUENTIAL
    so.inter_op_num_threads     = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _st_session   = ort.InferenceSession(model_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
    _st_extractor = WhisperFeatureExtractor(chunk_length=ST_MAX_S)
    print("[vad_server] smart-turn ready")


def _pad_or_trim(audio: np.ndarray) -> np.ndarray:
    if len(audio) > ST_MAX_SAMP:
        return audio[-ST_MAX_SAMP:]
    if len(audio) < ST_MAX_SAMP:
        return np.pad(audio, (ST_MAX_SAMP - len(audio), 0))
    return audio


def _infer_smart_turn(audio: np.ndarray) -> dict:
    audio = np.clip(audio, -1.0, 1.0)
    audio = _pad_or_trim(audio)
    feats = _st_extractor(
        audio, sampling_rate=SR, return_tensors="np",
        padding="max_length", max_length=ST_MAX_SAMP,
        truncation=True, do_normalize=True,
    )
    inp = feats.input_features.squeeze(0).astype(np.float32)[np.newaxis]
    prob     = float(_st_session.run(None, {"input_features": inp})[0][0].item())
    complete = prob > 0.5
    return {"complete": complete, "confidence": round(prob if complete else 1 - prob, 4),
            "backend": "smart-turn"}


# ---------------------------------------------------------------------------
# pyannote OSD
# ---------------------------------------------------------------------------
_osd_pipeline = None


def _load_pyannote(hf_token: str) -> None:
    global _osd_pipeline
    try:
        _orig_load = torch.load
        torch.load = lambda *a, **kw: _orig_load(*a, **{**kw, "weights_only": False})
        from pyannote.audio import Model
        from pyannote.audio.pipelines import OverlappedSpeechDetection
        print("[vad_server] loading pyannote/segmentation-3.0 …")
        model       = Model.from_pretrained("pyannote/segmentation-3.0",
                                            use_auth_token=hf_token)
        _osd_pipeline = OverlappedSpeechDetection(segmentation=model)
        _osd_pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})
        torch.load  = _orig_load
        print("[vad_server] pyannote OSD ready")
    except Exception as exc:
        try: torch.load = _orig_load        # noqa: E701
        except NameError: pass
        print(f"[vad_server] pyannote OSD unavailable — {exc!r}")
        print("[vad_server] /vad/overlap will return 503; ASR fallback active")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _decode_audio(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# /vad/complete/namo  — always Namo
# ---------------------------------------------------------------------------

class CompleteRequest(BaseModel):
    text:       str
    audio_b64:  str | None = None   # ignored by Namo, used by smart-turn
    sample_rate: int = SR


@app.post("/vad/complete/namo")
def vad_complete_namo(req: CompleteRequest):
    if _namo_session is None:
        raise HTTPException(503, "Namo not loaded")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    return _infer_namo(req.text)


# ---------------------------------------------------------------------------
# /vad/complete/smart-turn  — always smart-turn
# ---------------------------------------------------------------------------

@app.post("/vad/complete/smart-turn")
def vad_complete_smart_turn(req: CompleteRequest):
    if _st_session is None:
        raise HTTPException(503, "smart-turn not loaded")
    if not req.audio_b64:
        raise HTTPException(400, "audio_b64 required for smart-turn backend")
    return _infer_smart_turn(_decode_audio(req.audio_b64))


# ---------------------------------------------------------------------------
# /vad/complete  — routes to VAD_COMPLETE_BACKEND
# ---------------------------------------------------------------------------

_BACKEND = os.environ.get("VAD_COMPLETE_BACKEND", "namo").lower()


@app.post("/vad/complete")
def vad_complete(req: CompleteRequest):
    """Route to configured backend (VAD_COMPLETE_BACKEND env var)."""
    if _BACKEND == "smart-turn":
        return vad_complete_smart_turn(req)
    return vad_complete_namo(req)


# ---------------------------------------------------------------------------
# /vad/overlap  — pyannote OSD
# ---------------------------------------------------------------------------

class OverlapRequest(BaseModel):
    mic_b64:     str
    tts_b64:     str
    sample_rate: int = SR


@app.post("/vad/overlap")
def vad_overlap(req: OverlapRequest):
    if _osd_pipeline is None:
        raise HTTPException(503, "pyannote OSD not loaded")
    mic = np.clip(_decode_audio(req.mic_b64), -1.0, 1.0)
    tts = np.clip(_decode_audio(req.tts_b64), -1.0, 1.0)
    n   = max(len(mic), len(tts))
    mic = np.pad(mic, (0, n - len(mic)))
    tts = np.pad(tts, (0, n - len(tts)))
    mixed    = np.clip(mic + tts, -1.0, 1.0)
    waveform = torch.tensor(mixed).float().unsqueeze(0)
    osd      = _osd_pipeline({"waveform": waveform, "sample_rate": req.sample_rate})
    total_s  = n / req.sample_rate
    overlap_s = sum(seg.duration for seg in osd.get_timeline().support())
    ratio    = overlap_s / total_s if total_s > 0 else 0.0
    return {"overlap": ratio > 0.1, "overlap_ratio": round(ratio, 4)}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--smart-turn-model", default=SMART_TURN_PATH)
    args = parser.parse_args()

    _load_namo()
    _load_smart_turn(args.smart_turn_model)
    _load_pyannote(os.environ.get("HF_TOKEN", ""))

    print(f"[vad_server] /vad/complete → backend={_BACKEND!r}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

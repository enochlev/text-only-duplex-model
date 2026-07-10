"""Client-side copy of the duplex wire protocol helpers.

This is a trimmed version of the repo-root ``duplex_protocol.py``: it keeps
everything ``duplex_client.py`` needs (URL normalisation, audio payload
encode/decode, snapshot dataclasses) but drops the server-only helpers
(``snapshot_from_agent`` etc.) so it does NOT import ``full_duplex`` — the
client stays torch-free.
"""

from __future__ import annotations

import base64
import io
import os
import wave
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

DEFAULT_SERVER_PORT = int(os.getenv("SERVER_PORT", "8998"))
DEFAULT_WS_PATH = "/ws"


def server_url_from_address(address: str, path: str = DEFAULT_WS_PATH) -> str:
    value = (address or "").strip()
    if not value:
        return f"ws://127.0.0.1:{DEFAULT_SERVER_PORT}{path}"

    normalized_path = path if path.startswith("/") else f"/{path}"
    if "://" in value:
        proto, rest = value.split("://", 1)
        ws_proto = "ws" if proto in {"http", "ws"} else "wss"
        rest = rest.rstrip("/")
        if rest.endswith(normalized_path):
            return f"{ws_proto}://{rest}"
        return f"{ws_proto}://{rest}{normalized_path}"

    host = value
    if ":" not in host:
        host = f"{host}:{DEFAULT_SERVER_PORT}"
    return f"ws://{host}{normalized_path}"


def encode_audio_payload(
    sample_rate: int,
    audio_array: np.ndarray,
    *,
    encoding: str,
) -> dict[str, Any]:
    arr = np.asarray(audio_array)
    if encoding == "pcm_f32le":
        payload = arr.astype("<f4", copy=False).reshape(-1)
    elif encoding == "pcm_s16le":
        if np.issubdtype(arr.dtype, np.floating):
            payload = (arr * 32767.0).clip(-32768, 32767).astype("<i2")
        else:
            payload = arr.astype("<i2", copy=False).reshape(-1)
    else:
        raise ValueError(f"Unsupported audio encoding: {encoding}")

    return {
        "sample_rate": int(sample_rate),
        "encoding": encoding,
        "data": base64.b64encode(payload.tobytes()).decode("ascii"),
    }


def decode_audio_payload(payload: dict[str, Any]) -> tuple[int, np.ndarray]:
    sample_rate = int(payload["sample_rate"])
    encoding = payload["encoding"]
    raw = base64.b64decode(payload["data"])

    if encoding == "pcm_f32le":
        audio = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    elif encoding == "pcm_s16le":
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    else:
        raise ValueError(f"Unsupported audio encoding: {encoding}")

    return sample_rate, audio


def audio_to_data_uri(audio_array: np.ndarray, sample_rate: int) -> str:
    arr = np.asarray(audio_array)
    if np.issubdtype(arr.dtype, np.floating):
        pcm = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)
    else:
        pcm = arr.astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


@dataclass
class BlockSnapshot:
    block_id: str
    start_ts: float
    end_ts: float
    user_text: str = ""
    assistant_text: str = ""
    assistant_text_stale: bool = False
    tts_sr: int = 0
    tts_latency_s: Optional[float] = None
    asr_latency_s: Optional[float] = None
    llm_latency_s: Optional[float] = None
    total_latency_s: Optional[float] = None
    response_source_block_id: Optional[str] = None
    timeline_start_ts: Optional[float] = None
    timeline_end_ts: Optional[float] = None
    lead_silence_s: float = 0.0
    mic_audio_uri: Optional[str] = None
    tts_audio_uri: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BlockSnapshot:
        return cls(**payload)


@dataclass
class SessionSnapshot:
    session_id: str
    started_at: float
    block_count: int
    pending_word_count: int
    context_version: int
    last_llm_error: Optional[str]
    last_llm_error_seq: int
    blocks: list[BlockSnapshot]
    current_block: Optional[BlockSnapshot] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "block_count": self.block_count,
            "pending_word_count": self.pending_word_count,
            "context_version": self.context_version,
            "last_llm_error": self.last_llm_error,
            "last_llm_error_seq": self.last_llm_error_seq,
            "blocks": [block.to_dict() for block in self.blocks],
            "current_block": self.current_block.to_dict() if self.current_block else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SessionSnapshot:
        blocks = [BlockSnapshot.from_dict(block) for block in payload.get("blocks", [])]
        current_block_payload = payload.get("current_block")
        current_block = (
            BlockSnapshot.from_dict(current_block_payload)
            if current_block_payload is not None
            else None
        )
        return cls(
            session_id=payload["session_id"],
            started_at=payload["started_at"],
            block_count=payload["block_count"],
            pending_word_count=payload["pending_word_count"],
            context_version=payload["context_version"],
            last_llm_error=payload.get("last_llm_error"),
            last_llm_error_seq=payload.get("last_llm_error_seq", 0),
            blocks=blocks,
            current_block=current_block,
        )


def should_emit_audio_chunk(audio_array: np.ndarray) -> bool:
    return bool(np.any(np.asarray(audio_array)))

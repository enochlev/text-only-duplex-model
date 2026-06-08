from __future__ import annotations

import base64
import io
import os
import wave
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from full_duplex import DuplexAudioAgent, DuplexAudioBlock

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

    @classmethod
    def from_block(
        cls,
        block: DuplexAudioBlock,
        *,
        include_audio: bool,
    ) -> BlockSnapshot:
        mic_audio_uri = None
        tts_audio_uri = None
        if include_audio and block.mic_audio is not None and len(block.mic_audio) > 0:
            mic_audio_uri = audio_to_data_uri(block.mic_audio, 16000)
        if include_audio and block.tts_audio is not None and len(block.tts_audio) > 0:
            tts_audio_uri = audio_to_data_uri(block.tts_audio, block.tts_sr)

        return cls(
            block_id=block.block_id,
            start_ts=block.start_ts,
            end_ts=block.end_ts,
            user_text=block.user_text,
            assistant_text=block.assistant_text,
            assistant_text_stale=block.assistant_text_stale,
            tts_sr=block.tts_sr,
            tts_latency_s=block.tts_latency_s,
            asr_latency_s=block.asr_latency_s,
            llm_latency_s=block.llm_latency_s,
            total_latency_s=block.total_latency_s,
            response_source_block_id=block.response_source_block_id,
            timeline_start_ts=block.timeline_start_ts,
            timeline_end_ts=block.timeline_end_ts,
            lead_silence_s=block.lead_silence_s,
            mic_audio_uri=mic_audio_uri,
            tts_audio_uri=tts_audio_uri,
        )

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


def snapshot_from_agent(
    session_id: str,
    agent: DuplexAudioAgent,
    *,
    started_at: float,
    max_blocks: int = 120,
    audio_preview_blocks: int = 8,
) -> SessionSnapshot:
    block_subset = list(agent.blocks[-max_blocks:])
    audio_cutoff = max(0, len(block_subset) - audio_preview_blocks)
    blocks = [
        BlockSnapshot.from_block(block, include_audio=index >= audio_cutoff)
        for index, block in enumerate(block_subset)
    ]
    current_block = (
        BlockSnapshot.from_block(agent._current_block, include_audio=False)
        if agent._current_block is not None
        else None
    )
    return SessionSnapshot(
        session_id=session_id,
        started_at=started_at,
        block_count=len(agent.blocks),
        pending_word_count=len(agent._pending_words),
        context_version=agent.context_version,
        last_llm_error=agent.last_llm_error,
        last_llm_error_seq=agent.last_llm_error_seq,
        blocks=blocks,
        current_block=current_block,
    )


def snapshot_fingerprint(agent: DuplexAudioAgent, *, max_blocks: int = 120) -> tuple[Any, ...]:
    def block_key(block: DuplexAudioBlock) -> tuple[Any, ...]:
        return (
            block.block_id,
            block.start_ts,
            block.end_ts,
            block.user_text,
            block.assistant_text,
            block.assistant_text_stale,
            block.tts_latency_s,
            block.asr_latency_s,
            block.llm_latency_s,
            block.total_latency_s,
            block.response_source_block_id,
            block.timeline_start_ts,
            block.timeline_end_ts,
            len(block.mic_audio) if block.mic_audio is not None else 0,
            len(block.tts_audio) if block.tts_audio is not None else 0,
        )

    current_block_key = block_key(agent._current_block) if agent._current_block is not None else None
    return (
        agent.context_version,
        tuple(agent._pending_words),
        agent.last_llm_error_seq,
        tuple(block_key(block) for block in agent.blocks[-max_blocks:]),
        current_block_key,
    )


def should_emit_audio_chunk(audio_array: np.ndarray) -> bool:
    return bool(np.any(np.asarray(audio_array)))

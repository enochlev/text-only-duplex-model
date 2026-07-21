from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import soundfile as sf

try:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect as websocket_connect
except ImportError:  # pragma: no cover - exercised before uv sync installs deps
    ConnectionClosed = Exception
    websocket_connect = None

from duplex_protocol import (
    decode_audio_payload,
    encode_audio_payload,
    server_url_from_address,
    SessionSnapshot,
)


class FullDuplexClient:
    def __init__(
        self,
        server: str,
        *,
        open_timeout: float = 10.0,
    ):
        self.server_url = server_url_from_address(server)
        self.open_timeout = open_timeout
        self.session_id: Optional[str] = None

        self._ws = None
        self._recv_thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._closed_event = threading.Event()
        self._latest_snapshot: Optional[SessionSnapshot] = None
        self._snapshot_lock = threading.Lock()
        self._event_queue: queue.Queue[dict] = queue.Queue()
        self._audio_queue: queue.Queue[tuple[int, np.ndarray]] = queue.Queue()
        self._warning_queue: queue.Queue[dict] = queue.Queue()

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._closed_event.is_set()

    def connect(
        self,
        *,
        session_id: Optional[str] = None,
        client_name: str = "python-api",
        lite_snapshots: bool = False,
    ) -> str:
        if self.connected:
            return self.session_id or ""
        if websocket_connect is None:
            raise RuntimeError(
                "websockets is required for FullDuplexClient. Run `uv sync` first."
            )

        self._ws = websocket_connect(
            self.server_url,
            open_timeout=self.open_timeout,
            close_timeout=self.open_timeout,
            max_size=None,
        )
        self._closed_event.clear()
        self._ready_event.clear()
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        self._send(
            {
                "type": "hello",
                "session_id": session_id,
                "client": client_name,
                # skip base64 audio previews in periodic snapshots (30x less tunnel
                # traffic; audio_chunk frames stop queueing behind snapshot frames)
                "lite_snapshots": lite_snapshots,
            }
        )
        if not self._ready_event.wait(timeout=self.open_timeout):
            self.close()
            raise TimeoutError(f"Timed out waiting for ready event from {self.server_url}")
        return self.session_id or ""

    def _recv_loop(self) -> None:
        try:
            while True:
                raw_message = self._ws.recv()
                if raw_message is None:
                    break
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")

                message = json.loads(raw_message)
                message_type = message.get("type")
                normalized = message

                if message_type == "ready":
                    self.session_id = message.get("session_id")
                    self._ready_event.set()
                elif message_type == "snapshot":
                    snapshot = SessionSnapshot.from_dict(message["snapshot"])
                    with self._snapshot_lock:
                        self._latest_snapshot = snapshot
                    normalized = {"type": "snapshot", "snapshot": snapshot}
                elif message_type == "audio_chunk":
                    audio_chunk = decode_audio_payload(message)
                    self._audio_queue.put(audio_chunk)
                    normalized = {"type": "audio_chunk", "audio": audio_chunk}
                elif message_type in {"warning", "error"}:
                    self._warning_queue.put(message)

                self._event_queue.put(normalized)
        except ConnectionClosed:
            pass
        except Exception as exc:
            self._warning_queue.put(
                {
                    "type": "error",
                    "source": "client",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            self._closed_event.set()
            self._ready_event.set()

    def _send(self, payload: dict) -> None:
        if self._ws is None:
            raise RuntimeError("Client is not connected")
        self._ws.send(json.dumps(payload))

    def send_audio_chunk(self, sample_rate: int, audio_array: np.ndarray) -> None:
        audio = np.asarray(audio_array, dtype=np.float32).reshape(-1)
        self._send(
            {
                "type": "mic_audio",
                **encode_audio_payload(sample_rate, audio, encoding="pcm_f32le"),
            }
        )

    def send_text(self, text: str) -> None:
        self._send({"type": "user_text", "text": text})

    def request_snapshot(self) -> None:
        self._send({"type": "snapshot_request"})

    def get_event(self, timeout: Optional[float] = None) -> Optional[dict]:
        try:
            return self._event_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def iter_events(self, timeout: Optional[float] = None) -> Iterator[dict]:
        while self.connected or not self._event_queue.empty():
            event = self.get_event(timeout=timeout)
            if event is None:
                break
            yield event

    def pop_audio_chunk(self, timeout: Optional[float] = None) -> Optional[tuple[int, np.ndarray]]:
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_warnings(self) -> list[dict]:
        warnings = []
        while True:
            try:
                warnings.append(self._warning_queue.get_nowait())
            except queue.Empty:
                return warnings

    def get_latest_snapshot(self) -> Optional[SessionSnapshot]:
        with self._snapshot_lock:
            return self._latest_snapshot

    def stream_wav(
        self,
        input_path: str | Path,
        *,
        output_path: str | Path | None = None,
        chunk_ms: float = 80.0,
        realtime: bool = True,
        settle_timeout_s: float = 2.0,
    ) -> dict:
        audio, sample_rate = sf.read(str(input_path), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        chunk_size = max(1, int(sample_rate * (chunk_ms / 1000.0)))
        collected: list[tuple[int, np.ndarray]] = []

        for offset in range(0, len(audio), chunk_size):
            chunk = audio[offset : offset + chunk_size]
            self.send_audio_chunk(sample_rate, chunk)
            drained = self.pop_audio_chunk(timeout=0.01)
            if drained is not None:
                collected.append(drained)
            if realtime:
                time.sleep(len(chunk) / sample_rate)

        deadline = time.monotonic() + settle_timeout_s
        while time.monotonic() < deadline:
            drained = self.pop_audio_chunk(timeout=0.1)
            if drained is None:
                continue
            collected.append(drained)
            deadline = time.monotonic() + settle_timeout_s

        if output_path is not None and collected:
            output_sample_rate = collected[0][0]
            output_audio = np.concatenate([chunk for _, chunk in collected])
            sf.write(str(output_path), output_audio, output_sample_rate)

        return {
            "session_id": self.session_id,
            "snapshot": self.get_latest_snapshot(),
            "audio_chunks": collected,
        }

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._closed_event.set()
        if self._recv_thread is not None and self._recv_thread.is_alive():
            self._recv_thread.join(timeout=1.0)
        self._ws = None
        self._recv_thread = None

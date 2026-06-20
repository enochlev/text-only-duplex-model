from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import threading
import time
import uuid
from typing import Callable, Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from duplex_protocol import (
    decode_audio_payload,
    encode_audio_payload,
    server_url_from_address,
    should_emit_audio_chunk,
    snapshot_fingerprint,
    snapshot_from_agent,
)
import full_duplex
from full_duplex import (
    DuplexAudioAgent,
    SERVER_PORT,
    VLLM_PORT,
    cpm_generate,
    preload_duplex_models,
    warmup_duplex_models,
)

POLL_INTERVAL_S = 0.08
SESSION_TTL_S = 900.0
AUDIO_IDLE_TIMEOUT_S = 20.0


class DuplexSession:
    def __init__(
        self,
        session_id: str,
        agent_factory: Callable[[], DuplexAudioAgent],
    ):
        self.session_id = session_id
        self.started_at = time.time()
        self.last_active_at = self.started_at
        self.last_audio_activity_at = self.started_at
        self.agent = agent_factory()
        self.lock = threading.Lock()
        self._last_snapshot_key = None
        self._last_warning_seq = 0

    def _touch(self) -> None:
        self.last_active_at = time.time()

    def _mark_audio_activity(self) -> None:
        now = time.time()
        self.last_active_at = now
        self.last_audio_activity_at = now

    def audio_idle_expired(self, timeout_s: float) -> bool:
        return (time.time() - self.last_audio_activity_at) >= timeout_s

    def _warning_event(self) -> Optional[dict]:
        if (
            self.agent.last_llm_error is not None
            and self.agent.last_llm_error_seq > self._last_warning_seq
        ):
            self._last_warning_seq = self.agent.last_llm_error_seq
            return {
                "type": "warning",
                "source": "llm",
                "message": self.agent.last_llm_error,
            }
        return None

    def _snapshot_event(self, *, force: bool = False) -> Optional[dict]:
        snapshot_key = snapshot_fingerprint(self.agent)
        if not force and snapshot_key == self._last_snapshot_key:
            return None
        self._last_snapshot_key = snapshot_key
        snapshot = snapshot_from_agent(
            self.session_id,
            self.agent,
            started_at=self.started_at,
        )
        return {
            "type": "snapshot",
            "snapshot": snapshot.to_dict(),
        }

    def _audio_event(self, audio_chunk: Optional[tuple[int, np.ndarray]]) -> Optional[dict]:
        if audio_chunk is None:
            return None
        sample_rate, audio_array = audio_chunk
        if not should_emit_audio_chunk(audio_array):
            return None
        self._mark_audio_activity()
        payload = encode_audio_payload(
            sample_rate,
            audio_array,
            encoding="pcm_s16le",
        )
        return {
            "type": "audio_chunk",
            **payload,
        }

    def receive_audio(self, sample_rate: int, audio_array: np.ndarray) -> list[dict]:
        self._touch()
        if should_emit_audio_chunk(audio_array):
            self._mark_audio_activity()
        with self.lock:
            audio_chunk = self.agent.receive_mic_chunk(sample_rate, audio_array)
            events = []
            warning = self._warning_event()
            if warning is not None:
                events.append(warning)
            snapshot = self._snapshot_event()
            if snapshot is not None:
                events.append(snapshot)
            audio_event = self._audio_event(audio_chunk)
            if audio_event is not None:
                events.append(audio_event)
            return events

    def receive_text(self, text: str) -> list[dict]:
        self._touch()
        with self.lock:
            self.agent.receive_text_message(text)
            events = []
            warning = self._warning_event()
            if warning is not None:
                events.append(warning)
            snapshot = self._snapshot_event(force=True)
            if snapshot is not None:
                events.append(snapshot)
            return events

    def poll(self) -> list[dict]:
        self._touch()
        with self.lock:
            events = []
            try:
                audio_chunk = self.agent.poll()
            except Exception as exc:
                events.append(
                    {
                        "type": "warning",
                        "source": "poll",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
                audio_chunk = None
            warning = self._warning_event()
            if warning is not None:
                events.append(warning)
            snapshot = self._snapshot_event()
            if snapshot is not None:
                events.append(snapshot)
            audio_event = self._audio_event(audio_chunk)
            if audio_event is not None:
                events.append(audio_event)
            return events

    def force_snapshot(self) -> dict:
        with self.lock:
            snapshot = self._snapshot_event(force=True)
            if snapshot is None:
                snapshot = {
                    "type": "snapshot",
                    "snapshot": snapshot_from_agent(
                        self.session_id,
                        self.agent,
                        started_at=self.started_at,
                    ).to_dict(),
                }
            return snapshot


class SessionManager:
    def __init__(
        self,
        agent_factory: Callable[[], DuplexAudioAgent],
        *,
        session_ttl_s: float = SESSION_TTL_S,
    ):
        self._agent_factory = agent_factory
        self._session_ttl_s = session_ttl_s
        self._lock = threading.Lock()
        self._sessions: dict[str, DuplexSession] = {}

    def _cleanup_locked(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_active_at > self._session_ttl_s
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def remove(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def get_or_create(self, session_id: Optional[str]) -> DuplexSession:
        now = time.time()
        with self._lock:
            self._cleanup_locked(now)
            resolved_session_id = session_id or uuid.uuid4().hex[:12]
            session = self._sessions.get(resolved_session_id)
            if session is None:
                session = DuplexSession(resolved_session_id, self._agent_factory)
                self._sessions[resolved_session_id] = session
            return session


def create_app(
    *,
    agent_factory: Optional[Callable[[], DuplexAudioAgent]] = None,
    poll_interval_s: float = POLL_INTERVAL_S,
    audio_idle_timeout_s: float = AUDIO_IDLE_TIMEOUT_S,
    public_url: Optional[str] = None,
) -> FastAPI:
    resolved_factory = agent_factory or DuplexAudioAgent
    manager = SessionManager(resolved_factory)
    # When tunneled (--share), echo the public wss:// URL so clients that read the
    # ready event connect through the tunnel instead of localhost.
    resolved_server_url = server_url_from_address(public_url or "127.0.0.1")

    app = FastAPI(title="Full-Duplex Audio Server")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws")
    async def duplex_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        session = None
        stop_event = asyncio.Event()

        async def send_events(events: list[dict]) -> None:
            for event in events:
                await websocket.send_text(json.dumps(event))

        try:
            hello = json.loads(await websocket.receive_text())
            if hello.get("type") != "hello":
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "First message must be a hello event.",
                        }
                    )
                )
                return

            session = await asyncio.to_thread(
                manager.get_or_create,
                hello.get("session_id"),
            )
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "ready",
                        "session_id": session.session_id,
                        "poll_interval_s": poll_interval_s,
                        "server_url": resolved_server_url,
                    }
                )
            )
            await websocket.send_text(json.dumps(await asyncio.to_thread(session.force_snapshot)))

            async def poll_loop() -> None:
                while not stop_event.is_set():
                    events = await asyncio.to_thread(session.poll)
                    if events:
                        await send_events(events)
                    if session.audio_idle_expired(audio_idle_timeout_s):
                        stop_event.set()
                        await websocket.close(code=1000, reason="Audio idle timeout")
                        return
                    await asyncio.sleep(poll_interval_s)

            poll_task = asyncio.create_task(poll_loop())
            try:
                while True:
                    message = json.loads(await websocket.receive_text())
                    message_type = message.get("type")
                    if message_type == "mic_audio":
                        sample_rate, audio_array = decode_audio_payload(message)
                        events = await asyncio.to_thread(session.receive_audio, sample_rate, audio_array)
                        await send_events(events)
                    elif message_type == "user_text":
                        events = await asyncio.to_thread(session.receive_text, message.get("text", ""))
                        await send_events(events)
                    elif message_type == "snapshot_request":
                        await websocket.send_text(
                            json.dumps(await asyncio.to_thread(session.force_snapshot))
                        )
                    elif message_type == "ping":
                        await websocket.send_text(json.dumps({"type": "pong"}))
                    else:
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "error",
                                    "message": f"Unsupported message type: {message_type}",
                                }
                            )
                        )
            finally:
                stop_event.set()
                poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await poll_task
        except WebSocketDisconnect:
            return
        finally:
            if session is not None:
                await asyncio.to_thread(manager.remove, session.session_id)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-duplex audio websocket server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--cpm",
        "--is-cpm",
        dest="is_cpm",
        action="store_true",
        default=False,
        help="Use MiniCPM-duplex instead of the trained local model.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_PORT,
        help=f"Websocket server port that UI clients connect to (default {SERVER_PORT}).",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=VLLM_PORT,
        help=f"OpenAI-compatible model backend port (default {VLLM_PORT}).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Expose the server publicly via a Gradio FRP tunnel (*.gradio.live, expires ~1 week).",
    )
    args = parser.parse_args()

    # Point the generate functions at the chosen model backend.
    full_duplex.VLLM_PORT = args.vllm_port

    print("[boot] preloading Kokoro TTS and Parakeet ASR...")
    preload_duplex_models()
    print("[boot] warming up TTS + ASR kernels...")
    warmup_duplex_models()
    print(f"[boot] models ready, starting websocket server on ws://{args.host}:{args.port}/ws")

    public_url = None
    if args.share:
        # Same mechanism as gradio's launch(share=True): an outbound FRP tunnel to
        # Hugging Face's free *.gradio.live relay. setup_tunnel forwards any local TCP
        # port, not just a gradio app. The frpc subprocess it starts is kept alive by
        # the blocking uvicorn.run() below.
        import secrets

        from gradio.networking import setup_tunnel

        try:
            # share_server_address=None / cert=None -> auto-fetch a free relay from
            # https://api.gradio.app/v3/tunnel-request (downloads frpc on first use).
            public_url = setup_tunnel(
                args.host, args.port, secrets.token_urlsafe(32), None, None
            )
            ws_url = server_url_from_address(public_url)  # https://host -> wss://host/ws
            print(f"[share] public URL : {public_url}")
            print(f"[share] websocket  : {ws_url}")
            print("[share] expires in ~1 week; point any client (browser/Python) at the wss:// URL")
        except Exception as exc:
            public_url = None
            print(f"[share] tunnel failed ({type(exc).__name__}: {exc}); serving locally only")

    if args.is_cpm:
        print(f"[boot] CPM mode: using MiniCPM-duplex backend on port {args.vllm_port}")
        agent_factory = lambda: DuplexAudioAgent(llm_generate_fn=cpm_generate)
        app = create_app(agent_factory=agent_factory, public_url=public_url)
    else:
        print(f"[boot] local mode: using trained model backend on port {args.vllm_port}")
        app = create_app(public_url=public_url)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
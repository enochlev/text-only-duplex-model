import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from full_duplex import DuplexAudioAgent, TTS_SAMPLE_RATE
from server import create_app


def make_agent() -> DuplexAudioAgent:
    return DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda _: (TTS_SAMPLE_RATE, np.zeros(TTS_SAMPLE_RATE, dtype=np.int16)),
        asr_fn=lambda *_: None,
    )


def _recv_until_snapshot(websocket):
    while True:
        message = json.loads(websocket.receive_text())
        if message["type"] == "snapshot":
            return message


def _recv_until_snapshot_matching(websocket, predicate):
    while True:
        message = _recv_until_snapshot(websocket)
        if predicate(message["snapshot"]):
            return message


def test_server_websocket_returns_ready_and_initial_snapshot():
    app = create_app(agent_factory=make_agent, poll_interval_s=0.01)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(json.dumps({"type": "hello", "client": "pytest"}))

            ready = json.loads(websocket.receive_text())
            assert ready["type"] == "ready"
            assert ready["session_id"]

            snapshot = _recv_until_snapshot(websocket)
            assert snapshot["snapshot"]["block_count"] == 0


def test_server_user_text_message_updates_snapshot():
    app = create_app(agent_factory=make_agent, poll_interval_s=0.01)

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(json.dumps({"type": "hello", "client": "pytest"}))
            json.loads(websocket.receive_text())
            _recv_until_snapshot(websocket)

            websocket.send_text(json.dumps({"type": "user_text", "text": "hello there"}))
            snapshot = _recv_until_snapshot_matching(
                websocket,
                lambda payload: payload["context_version"] == 1,
            )

            assert snapshot["snapshot"]["context_version"] == 1
            assert snapshot["snapshot"]["current_block"]["user_text"] == "hello there"


def test_server_closes_websocket_after_audio_idle_timeout():
    app = create_app(
        agent_factory=make_agent,
        poll_interval_s=0.01,
        audio_idle_timeout_s=0.05,
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as websocket:
            websocket.send_text(json.dumps({"type": "hello", "client": "pytest"}))
            json.loads(websocket.receive_text())
            _recv_until_snapshot(websocket)

            with pytest.raises(WebSocketDisconnect):
                while True:
                    websocket.receive_text()
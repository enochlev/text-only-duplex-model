import numpy as np

from duplex_protocol import server_url_from_address, snapshot_from_agent
from full_duplex import DuplexAudioAgent, DuplexAudioBlock, TTS_SAMPLE_RATE


def make_agent() -> DuplexAudioAgent:
    return DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda _: (TTS_SAMPLE_RATE, np.zeros(TTS_SAMPLE_RATE, dtype=np.int16)),
        asr_fn=lambda *_: None,
    )


def test_server_url_from_address_normalizes_common_inputs():
    assert server_url_from_address("127.0.0.1") == "ws://127.0.0.1:8998/ws"
    assert server_url_from_address("localhost:9010") == "ws://localhost:9010/ws"
    assert server_url_from_address("http://example.com") == "ws://example.com/ws"
    assert server_url_from_address("https://example.com/api") == "wss://example.com/api/ws"


def test_snapshot_from_agent_embeds_recent_audio_previews():
    agent = make_agent()
    agent.blocks = [
        DuplexAudioBlock(
            block_id="b0",
            start_ts=0.0,
            end_ts=2.0,
            user_text="hello",
            assistant_text="world",
            mic_audio=np.ones(160, dtype=np.float32) * 0.1,
            tts_audio=np.ones(160, dtype=np.int16),
            tts_sr=TTS_SAMPLE_RATE,
            total_latency_s=0.5,
        )
    ]

    snapshot = snapshot_from_agent("session-1", agent, started_at=0.0, audio_preview_blocks=1)

    assert snapshot.session_id == "session-1"
    assert snapshot.block_count == 1
    assert snapshot.blocks[0].mic_audio_uri.startswith("data:audio/wav;base64,")
    assert snapshot.blocks[0].tts_audio_uri.startswith("data:audio/wav;base64,")
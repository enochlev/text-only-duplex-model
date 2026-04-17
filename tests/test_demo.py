import numpy as np

from demo import _build_status, _render_blocks, _render_latency_panel, _push_warning, _warning_title
from full_duplex import DuplexAudioAgent, DuplexAudioBlock, TTS_SAMPLE_RATE


def test_warning_title_uses_source_specific_labels():
    assert _warning_title("llm") == "LLM Warning"
    assert _warning_title("poll") == "Agent Warning"
    assert _warning_title("other") == "Warning"


def test_push_warning_deduplicates_repeated_messages():
    session_state = {"last_warning_key": None}

    assert _push_warning(session_state, "llm", "RuntimeError: boom") is True
    assert _push_warning(session_state, "llm", "RuntimeError: boom") is False

    assert session_state["last_warning_key"] == ("llm", "RuntimeError: boom")


def test_push_warning_accepts_new_message_after_duplicate():
    session_state = {"last_warning_key": None}

    _push_warning(session_state, "llm", "RuntimeError: boom")
    accepted = _push_warning(session_state, "poll", "RuntimeError: other")

    assert accepted is True
    assert session_state["last_warning_key"] == ("poll", "RuntimeError: other")


def test_build_status_keeps_metrics_line_stable():
    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda _: (TTS_SAMPLE_RATE, np.zeros(TTS_SAMPLE_RATE, dtype=np.int16)),
        asr_fn=lambda *_: None,
    )
    agent.blocks = [DuplexAudioBlock("b0", 0.0, 2.0)]
    agent._pending_words = ["hello", "world"]
    agent.context_version = 5

    status = _build_status(agent)

    assert "1 blocks" in status
    assert "2 pending words" in status
    assert "ctx v5" in status


def test_render_blocks_includes_asr_tts_llm_and_total_latency_labels():
    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda _: (TTS_SAMPLE_RATE, np.zeros(TTS_SAMPLE_RATE, dtype=np.int16)),
        asr_fn=lambda *_: None,
    )
    agent.blocks = [
        DuplexAudioBlock(
            "b0",
            100.0,
            102.0,
            user_text="hello",
            assistant_text="world",
            asr_latency_s=0.456,
            llm_latency_s=0.234,
            tts_latency_s=0.123,
            total_latency_s=0.789,
        )
    ]

    html = _render_blocks(agent, 100.0)

    assert "asr 0.456s" in html
    assert "llm 0.234s" in html
    assert "tts 0.123s" in html
    assert "total 0.789s" in html


def test_render_blocks_uses_timeline_overrides():
    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda _: (TTS_SAMPLE_RATE, np.zeros(TTS_SAMPLE_RATE, dtype=np.int16)),
        asr_fn=lambda *_: None,
    )
    agent.blocks = [
        DuplexAudioBlock(
            "b0",
            100.0,
            102.0,
            assistant_text="world",
            timeline_start_ts=98.5,
            timeline_end_ts=101.5,
        )
    ]

    html = _render_blocks(agent, 100.0)

    assert "[-1.50s → +1.50s  Δ3.00s]" in html


def test_render_latency_panel_reports_recent_average():
    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda _: (TTS_SAMPLE_RATE, np.zeros(TTS_SAMPLE_RATE, dtype=np.int16)),
        asr_fn=lambda *_: None,
    )
    agent.blocks = [
        DuplexAudioBlock("b0", 0.0, 2.0, assistant_text="a", total_latency_s=1.0),
        DuplexAudioBlock("b1", 2.0, 4.0, assistant_text="b", total_latency_s=2.0),
        DuplexAudioBlock("b2", 4.0, 6.0, assistant_text="c", total_latency_s=3.0),
    ]

    html = _render_latency_panel(agent)

    assert "latest" in html
    assert "3.000s" in html
    assert "avg(3)" in html
    assert "2.000s" in html
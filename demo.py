"""
demo.py — Gradio full-duplex audio demo with live block timeline.

Layout
------
  Left  : microphone input  →  agent  →  speaker output
  Right : scrolling debug panel — every timeblock in real time,
          colour-coded by content (user / assistant / both / empty)

Run
---
    python demo.py

Requires OPENAI_API_KEY in the environment (or .env file).
Piper TTS and Parakeet ASR are loaded at session start (page load).
Mic and timer are disabled until both models are ready.
"""

import base64 as _base64
import html as _html
import io as _io
import threading
import time
import wave as _wave

import gradio as gr
import numpy as np

from full_duplex import ASR_SAMPLE_RATE, DuplexAudioAgent, DuplexAudioBlock


def _audio_to_data_uri(audio: np.ndarray, sr: int) -> str:
    """Encode numpy PCM (float32 or int16, mono) → base64 WAV data URI."""
    if audio.dtype in (np.float32, np.float64):
        arr = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        arr = audio.astype(np.int16)
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(arr.tobytes())
    b64 = _base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"

POLL_INTERVAL_S = 0.08   # timer tick — 80 ms is fast enough for 2-s blocks

_LOADING_HTML = (
    "<div style='padding:20px;color:#aaa;font-family:monospace;font-size:13px'>"
    "⏳ Loading Piper TTS and Parakeet ASR — please wait…"
    "</div>"
)


def _warning_title(source: str) -> str:
    titles = {
        "llm": "LLM Warning",
        "poll": "Agent Warning",
    }
    return titles.get(source, "Warning")


def _build_status(agent: DuplexAudioAgent) -> str:
    return (
        f"_✓ {len(agent.blocks)} blocks  |  "
        f"{len(agent._pending_words)} pending words  |  "
        f"ctx v{agent.context_version}_"
    )


def _push_warning(session_state: dict, source: str, message: str) -> bool:
    message = message.strip()
    if not message:
        return False

    warning_key = (source, message)
    if session_state.get("last_warning_key") == warning_key:
        return False

    session_state["last_warning_key"] = warning_key
    return True


def _block_display_bounds(block: DuplexAudioBlock) -> tuple[float, float]:
    start_ts = block.timeline_start_ts if block.timeline_start_ts is not None else block.start_ts
    end_ts = block.timeline_end_ts if block.timeline_end_ts is not None else block.end_ts
    return start_ts, end_ts


def _render_latency_panel(agent: DuplexAudioAgent, window: int = 5) -> str:
    values = [block.total_latency_s for block in agent.blocks if block.total_latency_s is not None]
    if not values:
        return (
            "<div style='height:500px;background:#111;padding:12px;border:1px solid #2a2a2a;"
            "border-radius:6px;font-family:monospace;font-size:12px;color:#888'>"
            "No latency samples yet."
            "</div>"
        )

    recent = values[-window:]
    latest = recent[-1]
    avg = sum(recent) / len(recent)
    min_latency = min(recent)
    max_latency = max(recent)

    return (
        "<div style='height:500px;background:#111;padding:12px;border:1px solid #2a2a2a;"
        "border-radius:6px;font-family:monospace;font-size:12px;line-height:1.8;color:#ddd'>"
        "<div style='font-size:13px;color:#fff;margin-bottom:8px'>Rolling Total Latency</div>"
        f"<div><span style='color:#ffb366'>latest</span>: {latest:.3f}s</div>"
        f"<div><span style='color:#7cc7ff'>avg({len(recent)})</span>: {avg:.3f}s</div>"
        f"<div><span style='color:#aaa'>samples</span>: {len(values)}</div>"
        f"<div><span style='color:#aaa'>min</span>: {min_latency:.3f}s</div>"
        f"<div><span style='color:#aaa'>max</span>: {max_latency:.3f}s</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Debug panel renderer
# ---------------------------------------------------------------------------

def _render_blocks(agent: DuplexAudioAgent, t0: float) -> str:
    """Return auto-scrolling HTML showing all finalized + current blocks."""
    all_blocks = list(agent.blocks)
    if agent._current_block is not None:
        all_blocks.append(agent._current_block)

    rows = []
    for b in all_blocks:
        display_start, display_end = _block_display_bounds(b)
        rs = display_start - t0
        re = display_end - t0
        dur = re - rs

        if b.user_text and b.assistant_text:
            left_color = "#4a90d9"
        elif b.assistant_text:
            left_color = "#5cb85c"
        elif b.user_text:
            left_color = "#e8b84b"
        else:
            left_color = "#3a3a3a"

        user_str = _html.escape(b.user_text) if b.user_text else \
            "<em style='color:#555'>—</em>"
        asst_str = _html.escape(b.assistant_text) if b.assistant_text else \
            "<em style='color:#555'>—</em>"

        audio_html = ""
        latency_parts = []
        if b.asr_latency_s is not None:
            latency_parts.append(f'<span style="color:#e8b84b">asr {b.asr_latency_s:.3f}s</span>')
        if b.llm_latency_s is not None:
            latency_parts.append(f'<span style="color:#7cc7ff">llm {b.llm_latency_s:.3f}s</span>')
        if b.tts_latency_s is not None:
            latency_parts.append(f'<span style="color:#5cb85c">tts {b.tts_latency_s:.3f}s</span>')
        if b.total_latency_s is not None:
            latency_parts.append(f'<span style="color:#ffb366">total {b.total_latency_s:.3f}s</span>')
        latency_html = ""
        if latency_parts:
            latency_html = (
                '<span style="color:#666">  |  </span>'
                + '<span style="font-size:11px">'
                + " ".join(latency_parts)
                + "</span>"
            )

        if b.mic_audio is not None and len(b.mic_audio) > 0:
            uri = _audio_to_data_uri(b.mic_audio, ASR_SAMPLE_RATE)
            audio_html += (
                f'<span style="color:#e8b84b;font-size:11px">mic:</span> '
                f'<audio controls style="height:22px;vertical-align:middle" src="{uri}"></audio><br>'
            )
        if b.tts_audio is not None and len(b.tts_audio) > 0:
            uri = _audio_to_data_uri(b.tts_audio, b.tts_sr)
            audio_html += (
                f'<span style="color:#5cb85c;font-size:11px">tts:</span> '
                f'<audio controls style="height:22px;vertical-align:middle" src="{uri}"></audio><br>'
            )

        rows.append(
            f'<div style="'
            f'border-left:3px solid {left_color};'
            f'padding:4px 10px;margin:3px 0;'
            f'background:#1a1a1a;font-size:12px;font-family:monospace;'
            f'line-height:1.6">'
            f'<span style="color:#555">[{rs:+.2f}s → {re:+.2f}s  Δ{dur:.2f}s]</span>{latency_html}<br>'
            f'<span style="color:#e8b84b">user:</span> {user_str}<br>'
            f'<span style="color:#5cb85c">asst:</span> {asst_str}<br>'
            + audio_html +
            f'</div>'
        )

    body = "\n".join(rows) if rows else \
        "<p style='color:#555;font-size:13px;padding:8px'>No blocks yet — start speaking.</p>"

    return (
        '<div id="bp" style="height:500px;overflow-y:auto;background:#111;'
        'padding:8px;border:1px solid #2a2a2a;border-radius:6px">'
        + body
        + '</div>'
        '<script>(function(){'
        '  var p=document.getElementById("bp");'
        '  if(p) p.scrollTop=p.scrollHeight;'
        '})();</script>'
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Full-Duplex Agent") as demo:

        gr.Markdown("## Full-Duplex Audio Agent")
        gr.Markdown(
            "Piper TTS and Parakeet ASR load when the page opens. "
            "Mic and output unlock automatically once both are ready."
        )

        state = gr.State(None)

        with gr.Row():
            # --- Audio column ---
            with gr.Column(scale=1, min_width=300):
                audio_in = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone",
                    interactive=False,   # disabled until models are loaded
                )
                audio_out = gr.Audio(
                    streaming=True,
                    autoplay=True,
                    type="numpy",
                    label="Agent output",
                )
                status_md = gr.Markdown(
                    "_⏳ Loading models — microphone will unlock when ready…_"
                )

            # --- Debug column ---
            with gr.Column(scale=2):
                gr.Markdown("### Live block timeline")
                gr.Markdown(
                    "<small>"
                    "<span style='color:#e8b84b'>■</span> user only  "
                    "<span style='color:#5cb85c'>■</span> assistant only  "
                    "<span style='color:#4a90d9'>■</span> both  "
                    "<span style='color:#3a3a3a'>■</span> empty"
                    "</small>"
                )
                debug_html = gr.HTML(value=_LOADING_HTML)

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Latency")
                latency_html = gr.HTML(value=_LOADING_HTML)

        timer = gr.Timer(POLL_INTERVAL_S)

        # ------------------------------------------------------------------
        # Session init — blocks until Piper + Parakeet are loaded.
        # Returns (state, audio_in update, status update, debug update).
        # ------------------------------------------------------------------
        def on_load():
            t_start = time.time()
            # DuplexAudioAgent.__init__ eagerly calls _get_piper_voice() and
            # _get_asr_model(), so this call blocks until both are loaded.
            agent = DuplexAudioAgent()
            elapsed = time.time() - t_start
            s = {
                "agent": agent,
                "t0": time.time(),
                "lock": threading.Lock(),
                "last_warning_key": None,
                "last_llm_error_seq": 0,
            }
            status = f"_✓ Models loaded in {elapsed:.1f}s — start speaking_"
            return (
                s,
                gr.Audio(interactive=True),   # unlock microphone
                status,
                _render_blocks(agent, s["t0"]),
                _render_latency_panel(agent),
            )

        demo.load(
            on_load,
            outputs=[state, audio_in, status_md, debug_html, latency_html],
        )

        # ------------------------------------------------------------------
        # Mic stream → agent.receive_mic_chunk()
        # ------------------------------------------------------------------
        def receive_mic(audio, s):
            if s is None or audio is None:
                return s
            sr, arr = audio
            with s["lock"]:
                s["agent"].receive_mic_chunk(sr, np.array(arr, dtype=np.float32) / 32768.0)
            return s

        audio_in.stream(
            receive_mic,
            inputs=[audio_in, state],
            outputs=[state],
        )

        # ------------------------------------------------------------------
        # Timer → agent.poll() → TTS chunk + debug panel update
        # ------------------------------------------------------------------
        def poll_and_update(s):
            if s is None:
                return gr.skip(), _LOADING_HTML, "_⏳ Loading models…_", _LOADING_HTML

            with s["lock"]:
                agent = s["agent"]
                try:
                    result = agent.poll()
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    if _push_warning(s, "poll", message):
                        gr.Warning(message, duration=None, title=_warning_title("poll"))
                    result = None
                if agent.last_llm_error is not None and agent.last_llm_error_seq > s["last_llm_error_seq"]:
                    if _push_warning(s, "llm", agent.last_llm_error):
                        gr.Warning(agent.last_llm_error, duration=None, title=_warning_title("llm"))
                    s["last_llm_error_seq"] = agent.last_llm_error_seq
                html = _render_blocks(agent, s["t0"])
                latency = _render_latency_panel(agent)

            status = _build_status(agent)

            if result is not None:
                return result, html, status, latency
            return gr.skip(), html, status, latency

        timer.tick(
            poll_and_update,
            inputs=[state],
            outputs=[audio_out, debug_html, status_md, latency_html],
        )

        demo.queue()

    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft())

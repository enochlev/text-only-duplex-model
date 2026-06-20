"""
demo.py — Gradio thin client for the full-duplex audio server.

Run
---
    python server.py
    python demo.py

The Gradio process only captures microphone audio, plays returned audio, and
renders server-authored debug state. Kokoro TTS and Parakeet ASR are loaded in
the standalone server process, not in this UI process.
"""

from __future__ import annotations

import base64 as _base64
import html as _html
import io as _io
import os
import wave as _wave
from typing import Iterable, Optional

import gradio as gr
import numpy as np

from duplex_client import FullDuplexClient
from duplex_protocol import BlockSnapshot, SessionSnapshot, server_url_from_address
from full_duplex import ASR_SAMPLE_RATE, DuplexAudioAgent, DuplexAudioBlock

POLL_INTERVAL_S = 0.08

# Port of the full-duplex websocket server (server.py) this client connects to.
# Override with --port on the CLI or the FULL_DUPLEX_PORT env var.
FULL_DUPLEX_PORT = int(os.getenv("FULL_DUPLEX_PORT", "8998"))
DEFAULT_SERVER_URL = f"127.0.0.1:{FULL_DUPLEX_PORT}"

_LOADING_HTML = (
    "<div style='padding:20px;color:#aaa;font-family:monospace;font-size:13px'>"
    "Connecting to the full-duplex audio server…"
    "</div>"
)
_DISCONNECTED_HTML = (
    "<div style='padding:20px;color:#888;font-family:monospace;font-size:13px'>"
    "Enter a server URL and connect to start streaming."
    "</div>"
)

# Injected once as a <script> tag via gr.Blocks(js=).
# IMPORTANT: must be plain statements, NOT wrapped in a function — Gradio 6.x
# injects this verbatim into a <script> tag; a bare function expression would
# be evaluated and immediately discarded without ever being called.
#
# Defines window._audioEnqueue(dataUri):
#   - Schedules each WAV data-URI to start exactly when the previous chunk ends
#     using AudioContext.currentTime (hardware-locked, sample-accurate clock).
#   - Properly awaits ctx.resume() so the first chunk plays even if the browser
#     created the context in a suspended state (autoplay policy).
#   - Send "__reset__" to clear the scheduler (on connect / disconnect).
_INIT_JS = """
var _fdAudioCtx = null;
var _fdNextTime = 0;

window._audioEnqueue = function(dataUri) {
    if (!dataUri) return;
    if (dataUri === '__reset__') { _fdNextTime = 0; return; }

    if (!_fdAudioCtx || _fdAudioCtx.state === 'closed')
        _fdAudioCtx = new (window.AudioContext || window.webkitAudioContext)();

    var ctx = _fdAudioCtx;
    // Resume must be awaited — browser autoplay policy suspends new contexts
    // until a user gesture has occurred. After Connect is clicked the page is
    // activated and resume() resolves immediately on subsequent calls.
    var ready = ctx.state === 'suspended' ? ctx.resume() : Promise.resolve();
    ready
        .then(function() { return fetch(dataUri); })
        .then(function(r) { return r.arrayBuffer(); })
        .then(function(buf) { return ctx.decodeAudioData(buf); })
        .then(function(decoded) {
            var src = ctx.createBufferSource();
            src.buffer = decoded;
            src.connect(ctx.destination);
            // Math.max snaps the cursor forward after silence so a burst of
            // queued chunks drains without compounding lag.
            var when = Math.max(ctx.currentTime + 0.05, _fdNextTime);
            src.start(when);
            _fdNextTime = when + decoded.duration;
        })
        .catch(function(e) { console.error('[audio]', e); });
};
"""


def _audio_to_data_uri(audio: np.ndarray, sr: int) -> str:
    if audio.dtype in (np.float32, np.float64):
        arr = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    else:
        arr = audio.astype(np.int16)
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(arr.tobytes())
    b64 = _base64.b64encode(buf.getvalue()).decode()
    return f"data:audio/wav;base64,{b64}"


def _warning_title(source: str) -> str:
    titles = {
        "llm": "LLM Warning",
        "poll": "Agent Warning",
        "client": "Client Warning",
    }
    return titles.get(source, "Warning")


def _push_warning(session_state: dict, source: str, message: str) -> bool:
    message = message.strip()
    if not message:
        return False

    warning_key = (source, message)
    if session_state.get("last_warning_key") == warning_key:
        return False

    session_state["last_warning_key"] = warning_key
    return True


def _is_snapshot(source) -> bool:
    return isinstance(source, SessionSnapshot)


def _snapshot_blocks(source) -> list:
    if source is None:
        return []
    if isinstance(source, DuplexAudioAgent):
        blocks = list(source.blocks)
        if source._current_block is not None:
            blocks.append(source._current_block)
        return blocks
    blocks = list(source.blocks)
    if source.current_block is not None:
        blocks.append(source.current_block)
    return blocks


def _status_metrics(source) -> tuple[int, int, int]:
    if source is None:
        return 0, 0, 0
    if isinstance(source, DuplexAudioAgent):
        return len(source.blocks), len(source._pending_words), source.context_version
    return source.block_count, source.pending_word_count, source.context_version


def _build_status(source) -> str:
    block_count, pending_word_count, context_version = _status_metrics(source)
    return (
        f"_✓ {block_count} blocks  |  "
        f"{pending_word_count} pending words  |  "
        f"ctx v{context_version}_"
    )


def _block_display_bounds(block) -> tuple[float, float]:
    start_ts = block.timeline_start_ts if block.timeline_start_ts is not None else block.start_ts
    end_ts = block.timeline_end_ts if block.timeline_end_ts is not None else block.end_ts
    return start_ts, end_ts


def _block_audio_uri(block, kind: str) -> Optional[str]:
    uri_attr = f"{kind}_audio_uri"
    if hasattr(block, uri_attr):
        return getattr(block, uri_attr)

    audio_attr = f"{kind}_audio"
    audio = getattr(block, audio_attr, None)
    if audio is None or len(audio) == 0:
        return None
    sample_rate = ASR_SAMPLE_RATE if kind == "mic" else block.tts_sr
    return _audio_to_data_uri(audio, sample_rate)


def _render_latency_panel(source, window: int = 5) -> str:
    values = [block.total_latency_s for block in _snapshot_blocks(source) if block.total_latency_s is not None]
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


def _render_blocks(source, t0: Optional[float] = None) -> str:
    all_blocks = _snapshot_blocks(source)
    if not all_blocks:
        return (
            '<div id="bp" style="height:500px;overflow-y:auto;background:#111;'
            'padding:8px;border:1px solid #2a2a2a;border-radius:6px">'
            "<p style='color:#555;font-size:13px;padding:8px'>No blocks yet.</p>"
            "</div>"
        )

    if t0 is None and isinstance(source, SessionSnapshot):
        t0 = source.started_at
    if t0 is None:
        t0 = all_blocks[0].start_ts

    rows = []
    for block in all_blocks:
        display_start, display_end = _block_display_bounds(block)
        rs = display_start - t0
        re = display_end - t0
        dur = re - rs

        if block.user_text and block.assistant_text:
            left_color = "#4a90d9"
        elif block.assistant_text:
            left_color = "#5cb85c"
        elif block.user_text:
            left_color = "#e8b84b"
        else:
            left_color = "#3a3a3a"

        user_str = _html.escape(block.user_text) if block.user_text else "<em style='color:#555'>—</em>"
        asst_str = _html.escape(block.assistant_text) if block.assistant_text else "<em style='color:#555'>—</em>"

        latency_parts = []
        if block.asr_latency_s is not None:
            latency_parts.append(f'<span style="color:#e8b84b">asr {block.asr_latency_s:.3f}s</span>')
        if block.llm_latency_s is not None:
            latency_parts.append(f'<span style="color:#7cc7ff">llm {block.llm_latency_s:.3f}s</span>')
        if block.tts_latency_s is not None:
            latency_parts.append(f'<span style="color:#5cb85c">tts {block.tts_latency_s:.3f}s</span>')
        if block.total_latency_s is not None:
            latency_parts.append(f'<span style="color:#ffb366">total {block.total_latency_s:.3f}s</span>')
        latency_html = ""
        if latency_parts:
            latency_html = (
                '<span style="color:#666">  |  </span>'
                + '<span style="font-size:11px">'
                + " ".join(latency_parts)
                + "</span>"
            )

        audio_html = ""
        mic_uri = _block_audio_uri(block, "mic")
        tts_uri = _block_audio_uri(block, "tts")
        if mic_uri:
            audio_html += (
                f'<span style="color:#e8b84b;font-size:11px">mic:</span> '
                f'<audio controls style="height:22px;vertical-align:middle" data-src="{mic_uri}"></audio><br>'
            )
        if tts_uri:
            audio_html += (
                f'<span style="color:#5cb85c;font-size:11px">tts:</span> '
                f'<audio controls style="height:22px;vertical-align:middle" data-src="{tts_uri}"></audio><br>'
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
            + audio_html
            + "</div>"
        )

    return (
        '<div id="bp" style="height:500px;overflow-y:auto;background:#111;'
        'padding:8px;border:1px solid #2a2a2a;border-radius:6px">'
        + "\n".join(rows)
        + "</div>"
        '<script>(function(){'
        'var p=document.getElementById("bp");'
        'if(p) p.scrollTop=p.scrollHeight;'
        # Convert data: URIs → blob: URLs so CSP doesn't block <audio> playback.
        # fetch() handles data: URIs in all modern browsers.
        'var audios=p?p.querySelectorAll("audio[data-src]"):[];'
        'audios.forEach(function(a){'
        'var ds=a.getAttribute("data-src");'
        'if(!ds)return;'
        'a.removeAttribute("data-src");'
        'fetch(ds).then(function(r){return r.blob();})'
        '.then(function(b){a.src=URL.createObjectURL(b);})'
        '.catch(function(e){console.warn("[audio history]",e);});'
        '});'
        '})();</script>'
    )


def _render_disconnected() -> tuple[str, str]:
    return _DISCONNECTED_HTML, _LOADING_HTML


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Full-Duplex Agent Client", js=_INIT_JS) as demo:
        gr.Markdown("## Full-Duplex Audio Agent Client")
        gr.Markdown(
            "Start the standalone audio server first, then connect this Gradio client to it. "
            "All model loading and duplex state live in the server process."
        )

        state = gr.State(None)

        with gr.Row():
            server_url = gr.Textbox(
                label="Audio server",
                value=server_url_from_address(DEFAULT_SERVER_URL),
            )
            connect_btn = gr.Button("Connect", variant="primary")
            disconnect_btn = gr.Button("Disconnect", variant="secondary", interactive=False)

        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                audio_in = gr.Audio(
                    sources=["microphone"],
                    streaming=True,
                    type="numpy",
                    label="Microphone",
                    interactive=False,
                )
                audio_transport = gr.Textbox(visible=False, elem_id="audio-transport")
                status_md = gr.Markdown("_Disconnected — connect to an audio server to begin._")

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
                debug_html = gr.HTML(value=_DISCONNECTED_HTML)

            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Latency")
                latency_html = gr.HTML(value=_LOADING_HTML)

        timer = gr.Timer(POLL_INTERVAL_S)

        def on_load():
            return {
                "client": None,
                "snapshot": None,
                "last_warning_key": None,
            }

        def connect_server(server_value, session_state):
            state_value = dict(session_state or {})
            existing_client = state_value.get("client")
            if existing_client is not None:
                existing_client.close()

            client = FullDuplexClient(server_value)
            try:
                client.connect(client_name="gradio-client")
            except Exception as exc:
                state_value["client"] = None
                state_value["snapshot"] = None
                status = f"_× Unable to connect: {type(exc).__name__}: {exc}_"
                return (
                    state_value,
                    gr.Audio(interactive=False),
                    status,
                    _DISCONNECTED_HTML,
                    _LOADING_HTML,
                    gr.Button(interactive=False),  # disconnect_btn
                    gr.Button(interactive=True),   # connect_btn
                    "__reset__",                   # audio_transport: clear scheduler
                )

            snapshot = client.get_latest_snapshot()
            state_value["client"] = client
            state_value["snapshot"] = snapshot

            status = _build_status(snapshot) if snapshot is not None else "_✓ Connected — waiting for first snapshot_"
            blocks_html = _render_blocks(snapshot) if snapshot is not None else _LOADING_HTML
            latency = _render_latency_panel(snapshot) if snapshot is not None else _LOADING_HTML
            return (
                state_value,
                gr.Audio(interactive=True),
                status,
                blocks_html,
                latency,
                gr.Button(interactive=True),   # disconnect_btn
                gr.Button(interactive=True),   # connect_btn
                "__reset__",                   # audio_transport: clear scheduler
            )

        def disconnect_server(session_state):
            state_value = dict(session_state or {})
            client = state_value.get("client")
            if client is not None:
                client.close()
            state_value["client"] = None
            state_value["snapshot"] = None
            return (
                state_value,
                gr.Audio(interactive=False),
                "_Disconnected — connect to an audio server to begin._",
                _DISCONNECTED_HTML,
                _LOADING_HTML,
                gr.Button(interactive=False),  # disconnect_btn
                gr.Button(interactive=True),   # connect_btn
                "__reset__",                   # audio_transport: clear scheduler
            )

        def receive_mic(audio, session_state):
            if session_state is None or audio is None:
                return session_state
            client = session_state.get("client")
            if client is None or not client.connected:
                return session_state

            sample_rate, audio_array = audio
            audio_float = np.asarray(audio_array, dtype=np.float32) / 32768.0
            try:
                client.send_audio_chunk(sample_rate, audio_float)
            except Exception as exc:
                if _push_warning(session_state, "client", f"{type(exc).__name__}: {exc}"):
                    gr.Warning(f"{type(exc).__name__}: {exc}", duration=None, title=_warning_title("client"))
            return session_state

        def poll_and_update(session_state):
            if session_state is None:
                return gr.skip(), _DISCONNECTED_HTML, "_Disconnected._", _LOADING_HTML

            client = session_state.get("client")
            if client is None:
                return gr.skip(), _DISCONNECTED_HTML, "_Disconnected — connect to an audio server to begin._", _LOADING_HTML

            for warning in client.drain_warnings():
                message = warning.get("message", "")
                source = warning.get("source", "other")
                if _push_warning(session_state, source, message):
                    gr.Warning(message, duration=None, title=_warning_title(source))

            latest_snapshot = client.get_latest_snapshot()
            if latest_snapshot is not None:
                session_state["snapshot"] = latest_snapshot

            snapshot = session_state.get("snapshot")
            status = _build_status(snapshot) if snapshot is not None else "_✓ Connected — waiting for first snapshot_"
            blocks_html = _render_blocks(snapshot) if snapshot is not None else _LOADING_HTML
            latency = _render_latency_panel(snapshot) if snapshot is not None else _LOADING_HTML

            audio_chunk = client.pop_audio_chunk(timeout=0.0)
            if audio_chunk is not None:
                sr, audio_arr = audio_chunk
                data_uri = _audio_to_data_uri(audio_arr, sr)
                return data_uri, blocks_html, status, latency
            return gr.skip(), blocks_html, status, latency

        demo.load(on_load, outputs=[state])

        # JS bridge: when Python writes a data URI to audio_transport, the browser
        # immediately passes it to window._audioEnqueue for AudioContext scheduling.
        audio_transport.change(
            fn=None,
            inputs=[audio_transport],
            js="(uri) => { if (window._audioEnqueue) window._audioEnqueue(uri); }",
        )

        connect_btn.click(
            connect_server,
            inputs=[server_url, state],
            outputs=[state, audio_in, status_md, debug_html, latency_html, disconnect_btn, connect_btn, audio_transport],
        )

        disconnect_btn.click(
            disconnect_server,
            inputs=[state],
            outputs=[state, audio_in, status_md, debug_html, latency_html, disconnect_btn, connect_btn, audio_transport],
        )

        audio_in.stream(
            receive_mic,
            inputs=[audio_in, state],
            outputs=[state],
        )

        timer.tick(
            poll_and_update,
            inputs=[state],
            outputs=[audio_transport, debug_html, status_md, latency_html],
        )

        demo.queue()

    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gradio client for the full-duplex server")
    parser.add_argument(
        "--port",
        type=int,
        default=FULL_DUPLEX_PORT,
        help=f"Port of the full-duplex websocket server to connect to (default {FULL_DUPLEX_PORT}).",
    )
    args = parser.parse_args()
    FULL_DUPLEX_PORT = args.port
    DEFAULT_SERVER_URL = f"127.0.0.1:{FULL_DUPLEX_PORT}"

    build_demo().launch(theme=gr.themes.Soft())
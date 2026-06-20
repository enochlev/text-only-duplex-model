"""
demo.py — minimal single-model free-play voice client for the full-duplex server.

This is the "Free Play" experience from survey_demo.py, standalone: one model, no
tabs, no survey, nothing saved. The audio scheduler JS (``_JS``), WAV encoder
(``_wav_uri``), theme (``THEME``) and poll interval (``POLL_S``) are imported from
survey_demo.py so this file is only the UI + wiring.

Run
---
    python server.py
    python demo.py                                       # connect to 127.0.0.1:8998
    python demo.py --model-url https://xxx.gradio.live   # connect to a --share tunnel
"""

from __future__ import annotations

import os

import gradio as gr
import numpy as np

from duplex_client import FullDuplexClient
from duplex_protocol import server_url_from_address
from survey_demo import _JS, POLL_S, THEME, _wav_uri

# Server (server.py) this client connects to. Override the host with
# --model-url / FULL_DUPLEX_URL (e.g. a --share tunnel), or just the port with
# --port / FULL_DUPLEX_PORT. A bare host or https://host is normalised to
# wss://host/ws by server_url_from_address.
FULL_DUPLEX_PORT = int(os.getenv("FULL_DUPLEX_PORT", "8998"))
DEFAULT_SERVER_URL = os.getenv("FULL_DUPLEX_URL", f"127.0.0.1:{FULL_DUPLEX_PORT}")


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Full-Duplex Voice Client", js=_JS, theme=THEME) as demo:
        gr.Markdown(
            "## Full-Duplex Voice Client\n"
            "Open-ended voice conversation with one full-duplex model. "
            "Start `server.py` first, then connect."
        )

        state = gr.State(None)
        at = gr.Textbox(visible=False, elem_id="fd-at")  # audio transport → JS scheduler

        with gr.Row():
            server_url = gr.Textbox(
                label="Audio server",
                value=server_url_from_address(DEFAULT_SERVER_URL),
            )
            conn_btn = gr.Button("Connect", variant="primary")
            disc_btn = gr.Button("Disconnect", variant="secondary", interactive=False)

        mic = gr.Audio(
            sources=["microphone"],
            streaming=True,
            type="numpy",
            label="🎙 Microphone",
            interactive=False,
        )
        status = gr.Markdown("_Enter a server URL and click Connect._")

        tmr = gr.Timer(POLL_S, active=False)

        def connect(url, st):
            st = dict(st or {"client": None})
            old = st.get("client")
            if old:
                old.close()
            client = FullDuplexClient(url)
            try:
                client.connect(client_name="demo-client")
            except Exception as exc:
                st["client"] = None
                return (st, gr.Audio(interactive=False), f"_Connection failed: {exc}_",
                        gr.Button(interactive=True), gr.Button(interactive=False),
                        gr.update(active=False), "__reset__")
            st["client"] = client
            return (st, gr.Audio(interactive=True), "_Connected — speak freely._",
                    gr.Button(interactive=False), gr.Button(interactive=True),
                    gr.update(active=True), "__reset__")

        def disconnect(st):
            st = dict(st or {"client": None})
            c = st.get("client")
            if c:
                c.close()
            st["client"] = None
            return (st, gr.Audio(interactive=False), "_Disconnected._",
                    gr.Button(interactive=True), gr.Button(interactive=False),
                    gr.update(active=False), "__reset__")

        def on_mic(audio, st):
            if st is None or audio is None:
                return st
            c = st.get("client")
            if c is None or not c.connected:
                return st
            sr, arr = audio
            try:
                c.send_audio_chunk(sr, np.asarray(arr, dtype=np.float32) / 32768.0)
            except Exception:
                pass
            return st

        def poll(st):
            if st is None:
                return gr.skip(), gr.skip()
            c = st.get("client")
            if c is None:
                return gr.skip(), gr.skip()
            chunk = c.pop_audio_chunk(timeout=0.0)
            if chunk:
                sr, arr = chunk
                return _wav_uri(arr, sr), gr.skip()
            return gr.skip(), gr.skip()

        outs = [state, mic, status, conn_btn, disc_btn, tmr, at]
        demo.load(lambda: {"client": None}, outputs=[state])
        conn_btn.click(connect, inputs=[server_url, state], outputs=outs)
        disc_btn.click(disconnect, inputs=[state], outputs=outs)
        mic.stream(on_mic, inputs=[mic, state], outputs=[state])
        tmr.tick(poll, inputs=[state], outputs=[at, status])
        at.change(
            fn=None,
            inputs=[at],
            js="(u) => { if (window._audioEnqueue) window._audioEnqueue(u); }",
        )

        demo.queue()
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Single-model free-play client for the full-duplex server")
    parser.add_argument("--port", type=int, default=FULL_DUPLEX_PORT,
                        help=f"Server port on 127.0.0.1 (default {FULL_DUPLEX_PORT}).")
    parser.add_argument("--model-url", default=os.getenv("FULL_DUPLEX_URL"),
                        help="Full server URL/host (e.g. https://xxx.gradio.live); overrides --port.")
    parser.add_argument("--share", action="store_true",
                        help="Expose this UI publicly via a Gradio FRP tunnel (*.gradio.live, ~1 week).")
    args = parser.parse_args()
    DEFAULT_SERVER_URL = args.model_url or f"127.0.0.1:{args.port}"

    build_demo().launch(share=args.share)

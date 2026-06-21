"""
simple_web.py — minimal browser microphone client for the full-duplex server.

This is the "hook it up to anything" reference client. It is deliberately tiny:

  * a ~40-line Python static-file server (standard library only — no FastAPI,
    no Gradio, no extra pip installs), and
  * one self-contained HTML page of vanilla JavaScript that talks DIRECTLY to
    server.py's websocket.

The browser does all the streaming. The JS below is the entire integration
contract — copy it into any web app to give it a live full-duplex voice agent.

How it fits together
--------------------
    [ vLLM model backend ]  <--  server.py (TTS + ASR + /ws)  <--  THIS PAGE
         port 8555                    port 8998                  port 9000 (http)

Run
---
    # 1. model backend (see README)
    vllm serve xinrongzhang2022/MiniCPM-duplex \
        --served-model-name cpm-text-duplex --max-model-len 3000 \
        --gpu_memory_utilization 0.30 --port 8555 --trust-remote-code

    # 2. duplex audio server (loads Kokoro TTS + Parakeet ASR)
    python server.py --cpm

    # 3. this page
    python simple_web.py            # then open http://localhost:9000

Flags
-----
    --port          http port this page is served on        (default 9000)
    --duplex-host   host running server.py                  (default 127.0.0.1)
    --duplex-port   server.py websocket port (its --port)   (default 8998)

Note: microphone capture requires a "secure context". http://localhost counts
as secure, so opening http://localhost:9000 works with no TLS. If you serve
this to another machine you'll need https or a localhost SSH tunnel.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# server.py defaults (see full_duplex.SERVER_PORT) — keep in sync if you change them.
DEFAULT_HTTP_PORT = 9000
DEFAULT_DUPLEX_HOST = "127.0.0.1"
DEFAULT_DUPLEX_PORT = 8998


# The whole client. {DUPLEX_WS_URL} is filled in at request time so the browser
# knows where server.py's websocket lives.
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full-Duplex Voice Demo</title>
<style>
  body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          background:#111; color:#ddd; margin:0; padding:24px; }}
  h1 {{ font-size:18px; color:#fff; margin:0 0 4px; }}
  .sub {{ color:#888; font-size:13px; margin-bottom:16px; }}
  button {{ font:inherit; padding:10px 20px; border-radius:6px; border:0;
            cursor:pointer; }}
  #start {{ background:#4a90d9; color:#fff; }}
  #stop  {{ background:#444; color:#fff; }}
  button:disabled {{ opacity:.4; cursor:default; }}
  #status {{ margin:14px 0; font-size:13px; }}
  .dot {{ display:inline-block; width:9px; height:9px; border-radius:50%;
          background:#666; margin-right:6px; vertical-align:middle; }}
  .dot.live {{ background:#5cb85c; }}
  .dot.err  {{ background:#d9534f; }}
  #transcript {{ background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px;
                 padding:8px; height:55vh; overflow-y:auto; font-size:13px; }}
  .row {{ padding:4px 8px; margin:3px 0; border-left:3px solid #3a3a3a;
          background:#161616; line-height:1.5; }}
  .row.user {{ border-left-color:#e8b84b; }}
  .row.bot  {{ border-left-color:#5cb85c; }}
  .who {{ font-size:11px; text-transform:uppercase; opacity:.7; }}
  .who.user {{ color:#e8b84b; }} .who.bot {{ color:#5cb85c; }}
</style>
</head>
<body>
  <h1>Full-Duplex Voice Demo</h1>
  <div class="sub">Connected to <code>{DUPLEX_WS_URL}</code> &mdash;
       speak after pressing Start; the bot replies over your speakers in real time.</div>

  <button id="start">Start talking</button>
  <button id="stop" disabled>Stop</button>
  <div id="status"><span class="dot" id="dot"></span><span id="statusText">Idle.</span></div>

  <div id="transcript"></div>

<script>
// ===========================================================================
// THIS is the entire client. Three responsibilities:
//   1. open the websocket and do the hello/ready handshake
//   2. capture the mic and stream raw float32 PCM up
//   3. receive the bot's audio chunks and play them back gap-free
// ===========================================================================

const WS_URL = "{DUPLEX_WS_URL}";

let ws = null;          // the websocket to server.py
let audioCtx = null;    // single AudioContext (capture + playback)
let micStream = null;   // MediaStream from getUserMedia
let micNode = null;     // ScriptProcessorNode pulling mic samples
let playCursor = 0;     // next scheduled playback time (AudioContext clock)
// Jitter buffer head-start. The server meters audio just-in-time (one slice per
// ~1.5s tick, no look-ahead), so poll granularity (~80ms) + network jitter let
// playCursor fall behind currentTime → ~0.1-0.2s underrun gaps between blocks.
// A bigger cushion keeps playCursor ahead so consecutive slices stay glued. Costs
// this much latency only at a response's first chunk (or after an underrun), not
// per block. Tune down toward 0.12 if start latency bugs you, up if gaps persist.
const JITTER_BUFFER_S = 0.20;

const $ = (id) => document.getElementById(id);

function setStatus(text, kind) {{
  $("statusText").textContent = text;
  $("dot").className = "dot" + (kind ? " " + kind : "");
}}

// --- base64 <-> typed array (the wire format server.py uses) ----------------
function bytesToB64(u8) {{
  let s = "";
  for (let i = 0; i < u8.length; i += 0x8000)         // chunk to dodge arg limits
    s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}}
function b64ToBytes(b64) {{
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}}

// --- playback: schedule each chunk right after the previous one -------------
// server.py sends 16-bit little-endian PCM ("pcm_s16le"). We convert to float
// and queue it on the AudioContext clock so chunks play back-to-back with no
// gaps or overlap.
function playChunk(sampleRate, b64) {{
  const i16 = new Int16Array(b64ToBytes(b64).buffer);
  const f32 = new Float32Array(i16.length);
  for (let i = 0; i < i16.length; i++) f32[i] = i16[i] / 32768;

  const buf = audioCtx.createBuffer(1, f32.length, sampleRate);
  buf.copyToChannel(f32, 0);
  const src = audioCtx.createBufferSource();
  src.buffer = buf;
  src.connect(audioCtx.destination);

  // max(): if playCursor fell behind (fresh response or underrun), restart with a
  // jitter-buffer head-start instead of the stale cursor; otherwise stay glued to
  // playCursor so consecutive slices play back-to-back with no gap.
  const when = Math.max(audioCtx.currentTime + JITTER_BUFFER_S, playCursor);
  src.start(when);
  playCursor = when + buf.duration;
}}

// --- transcript: re-render from each server snapshot ------------------------
function renderSnapshot(snap) {{
  const blocks = (snap.blocks || []).slice();
  if (snap.current_block) blocks.push(snap.current_block);
  const el = $("transcript");
  el.innerHTML = "";
  for (const b of blocks) {{
    if (b.user_text) el.insertAdjacentHTML("beforeend",
      `<div class="row user"><span class="who user">you</span><br>${{escapeHtml(b.user_text)}}</div>`);
    if (b.assistant_text) el.insertAdjacentHTML("beforeend",
      `<div class="row bot"><span class="who bot">bot</span><br>${{escapeHtml(b.assistant_text)}}</div>`);
  }}
  el.scrollTop = el.scrollHeight;
}}
function escapeHtml(s) {{
  return s.replace(/[&<>"]/g, (c) =>
    ({{ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }}[c]));
}}

// --- start: open ws, then start the mic on "ready" --------------------------
async function start() {{
  $("start").disabled = true;
  setStatus("Connecting…");

  // AudioContext must be created in a click handler (browser autoplay policy).
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  await audioCtx.resume();
  playCursor = 0;

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {{
    // Handshake: the first message MUST be a "hello".
    ws.send(JSON.stringify({{ type: "hello", session_id: null, client: "browser-mic" }}));
  }};

  ws.onmessage = (ev) => {{
    const msg = JSON.parse(ev.data);
    if (msg.type === "ready") {{
      startMic();                                   // server is ready for audio
      $("stop").disabled = false;
      setStatus("Live — start speaking.", "live");
    }} else if (msg.type === "audio_chunk") {{
      playChunk(msg.sample_rate, msg.data);         // bot speech
    }} else if (msg.type === "snapshot") {{
      renderSnapshot(msg.snapshot);                 // live transcript
    }} else if (msg.type === "warning" || msg.type === "error") {{
      console.warn("[server]", msg.message);
    }}
  }};

  ws.onerror = () => setStatus("Connection error — is server.py running?", "err");
  ws.onclose = () => {{ setStatus("Disconnected.", "err"); cleanup(); }};
}}

// --- mic capture: stream native-rate float32 PCM up -------------------------
// server.py resamples to 16 kHz for ASR, so we just send whatever the
// AudioContext gives us (typically 48 kHz) as little-endian float32 ("pcm_f32le").
async function startMic() {{
  micStream = await navigator.mediaDevices.getUserMedia({{
    audio: {{ channelCount: 1, echoCancellation: true, noiseSuppression: true }},
  }});
  const source = audioCtx.createMediaStreamSource(micStream);

  // ScriptProcessorNode is deprecated but universal and the simplest thing that
  // hands you raw PCM. 4096 frames ≈ 85 ms at 48 kHz.
  micNode = audioCtx.createScriptProcessor(4096, 1, 1);
  micNode.onaudioprocess = (e) => {{
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm = new Float32Array(e.inputBuffer.getChannelData(0));   // copy out
    ws.send(JSON.stringify({{
      type: "mic_audio",
      sample_rate: audioCtx.sampleRate,
      encoding: "pcm_f32le",
      data: bytesToB64(new Uint8Array(pcm.buffer)),
    }}));
    // we don't write e.outputBuffer, so nothing is echoed to the speakers.
  }};
  source.connect(micNode);
  micNode.connect(audioCtx.destination);            // required for the node to run
}}

function cleanup() {{
  if (micNode) {{ micNode.disconnect(); micNode.onaudioprocess = null; micNode = null; }}
  if (micStream) {{ micStream.getTracks().forEach((t) => t.stop()); micStream = null; }}
  $("start").disabled = false;
  $("stop").disabled = true;
}}

function stop() {{
  if (ws) ws.close();                               // triggers onclose -> cleanup
}}

$("start").addEventListener("click", start);
$("stop").addEventListener("click", stop);
</script>
</body>
</html>
"""


def build_page(duplex_host: str, duplex_port: int) -> bytes:
    ws_url = f"ws://{duplex_host}:{duplex_port}/ws"
    return PAGE.format(DUPLEX_WS_URL=ws_url).encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal browser microphone client for the full-duplex server."
    )
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT,
                        help=f"HTTP port to serve this page on (default {DEFAULT_HTTP_PORT}).")
    parser.add_argument("--duplex-host", default=DEFAULT_DUPLEX_HOST,
                        help=f"Host running server.py (default {DEFAULT_DUPLEX_HOST}).")
    parser.add_argument("--duplex-port", type=int, default=DEFAULT_DUPLEX_PORT,
                        help=f"server.py websocket port, i.e. its --port (default {DEFAULT_DUPLEX_PORT}).")
    args = parser.parse_args()

    page_bytes = build_page(args.duplex_host, args.duplex_port)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server naming)
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page_bytes)))
            self.end_headers()
            self.wfile.write(page_bytes)

        def log_message(self, *_args):  # quiet the per-request access log
            pass

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[simple_web] serving the mic client on http://localhost:{args.port}")
    print(f"[simple_web] browser will stream to ws://{args.duplex_host}:{args.duplex_port}/ws")
    print("[simple_web] open the URL above, press 'Start talking', and speak.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[simple_web] shutting down.")
        httpd.shutdown()


if __name__ == "__main__":
    main()

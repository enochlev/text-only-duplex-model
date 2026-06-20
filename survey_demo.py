"""
survey_demo.py — Blind A/B survey comparing two duplex conversational models.

Tab 1: Structured survey (demographics → 6 voice sessions → Likert → comparison → reveal).
Tab 2: Free-play voice chat for demos.

Run after starting both servers:
    python server.py --port 8998                   # Model A on port 8998
    python server.py --cpm --port 8999 --vllm-port 8556   # Model B on port 8999
    python survey_demo.py --model-a-port 8998 --model-b-port 8999
"""
from __future__ import annotations

import base64 as _b64
import io as _io
import json
import os
import random
import threading
import time
import uuid
import wave as _wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

from duplex_client import FullDuplexClient
from duplex_protocol import server_url_from_address

# ── Config ────────────────────────────────────────────────────────────────────
# Two server.py instances are compared. Each port points at one server's
# websocket. Override with --model-a-port / --model-b-port (see __main__).
MODEL_A_PORT = int(os.getenv("MODEL_A_PORT", "8998"))
MODEL_B_PORT = int(os.getenv("MODEL_B_PORT", "8999"))
_URLS    = {
    "A": server_url_from_address(f"127.0.0.1:{MODEL_A_PORT}"),
    "B": server_url_from_address(f"127.0.0.1:{MODEL_B_PORT}"),
}
_NAMES   = {"A": "Duplex (Trained)", "B": "MiniCPM-Duplex"}
MAX_CALL = 45.0
POLL_S   = 0.08
RESULTS  = Path("survey_results")

# ── Theme ─────────────────────────────────────────────────────────────────────
# Theme pulled from the Gradio gallery: gradio.app/themes/gallery?id=harsh8001/minimal-orange
THEME = gr.Theme.from_hub("harsh8001/minimal-orange")

def _warn(msg: str) -> str:
    """High-contrast inline validation banner (dark red on warm cream)."""
    return (
        "<div style='color:#b91c1c;background:#fff7ed;border:1px solid #fdba74;"
        "border-radius:6px;padding:8px 12px;margin:6px 0;font-weight:600'>"
        f"⚠ {msg}</div>"
    )

# ── Per-model concurrency (one active call each) ──────────────────────────────
_locks     = {"A": threading.Lock(), "B": threading.Lock()}
_lock_time = {"A": 0.0, "B": 0.0}   # monotonic time of last acquire
_LOCK_TTL  = 60.0                    # release stale lock after this many seconds

def _acquire(k: str) -> bool:
    # Release stale lock from a dead session
    if _lock_time[k] > 0 and time.monotonic() - _lock_time[k] > _LOCK_TTL:
        try:
            _locks[k].release()
        except RuntimeError:
            pass
        _lock_time[k] = 0.0
    if _locks[k].acquire(blocking=False):
        _lock_time[k] = time.monotonic()
        return True
    return False

def _release(k: str) -> None:
    try:
        _locks[k].release()
    except RuntimeError:
        pass
    _lock_time[k] = 0.0

# ── Survey content ────────────────────────────────────────────────────────────
SESSIONS = [
    {
        "title":       "Session 1 — Short Questions",
        "interrupt":   False,
        "instruction": "Use these prompts or ask similar short questions. Click the mic to speak.",
        "prompts": [
            "What's 12 times 7?",
            "What's the capital of France?",
            "How many days are in February?",
            "What's the square root of 144?",
            "What year did World War II end?",
        ],
    },
    {
        "title":       "Session 2 — Long Explanations",
        "interrupt":   False,
        "instruction": "Ask questions that require detailed answers. Click the mic to speak.",
        "prompts": [
            "Can you explain how photosynthesis works in plants?",
            "Walk me through how the immune system fights off a virus.",
            "Explain the basic idea behind general relativity.",
            "How does a transistor actually work?",
            "What's going on when bread dough rises?",
        ],
    },
    {
        "title":       "Session 3 — Interruption Practice",
        "interrupt":   True,
        "instruction": (
            "Ask the question below, then INTERRUPT the bot mid-response "
            "with a follow-up or clarification."
        ),
        "prompts": [
            "Tell me about the history of the Roman Empire.",
            "Describe how machine learning models are trained.",
            "Explain what causes the seasons.",
            "Walk me through the life cycle of a star.",
            "Describe how electricity gets from a power plant to my house.",
        ],
    },
]

LIKERT_QS = [
    "The bot responded promptly when I finished speaking.",
    "The bot interrupted me while I was still talking.",
    "The bot's silences felt natural, not awkward.",
    "The bot waited too long before responding.",
    "The bot's responses were relevant to what I asked.",
    "The conversation felt natural overall.",
    "I would want to talk to this bot again.",
]

COMPARE_QS = [
    ("natural",     "Which bot felt more natural to talk to?",  ["Bot 1", "Bot 2", "No preference"]),
    ("interrupted", "Which bot interrupted you less?",          ["Bot 1", "Bot 2", "Equal"]),
    ("prompt",      "Which bot responded more promptly?",       ["Bot 1", "Bot 2", "Equal"]),
    ("prefer",      "Overall, which bot would you prefer?",     ["Bot 1", "Bot 2", "No preference"]),
]

# ── Audio util ────────────────────────────────────────────────────────────────
def _wav_uri(audio: np.ndarray, sr: int) -> str:
    a = (audio * 32767).clip(-32768, 32767).astype(np.int16) if audio.dtype != np.int16 else audio
    buf = _io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(a.tobytes())
    return "data:audio/wav;base64," + _b64.b64encode(buf.getvalue()).decode()

# ── Persistence ───────────────────────────────────────────────────────────────
def _save(s: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    ts  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"{ts}_{s['pid'][:8]}.json"
    out.write_text(json.dumps({
        "participant_id": s["pid"],
        "timestamp":      datetime.utcnow().isoformat(),
        "demographics":   s["demo"],
        "assignment":     s["assign"],
        "likert":         s["likert"],
        "comparison":     s["comp"],
        "free_text":      s["free"],
    }, indent=2))

# ── AudioContext JS (same scheduling logic as demo.py) ────────────────────────
_JS = """
var _aC = null, _aN = 0;
window._audioEnqueue = function(u) {
    if (!u) return;
    if (u === '__reset__') { _aN = 0; return; }
    if (!_aC || _aC.state === 'closed')
        _aC = new (window.AudioContext || window.webkitAudioContext)();
    var c = _aC;
    (c.state === 'suspended' ? c.resume() : Promise.resolve())
        .then(function() { return fetch(u); })
        .then(function(r) { return r.arrayBuffer(); })
        .then(function(b) { return c.decodeAudioData(b); })
        .then(function(d) {
            var s = c.createBufferSource(); s.buffer = d; s.connect(c.destination);
            var w = Math.max(c.currentTime + 0.05, _aN);
            s.start(w); _aN = w + d.duration;
        }).catch(function(e) { console.error('[audio]', e); });
};
"""

# ── Survey state ──────────────────────────────────────────────────────────────
def _new_state() -> dict:
    return {
        "step":   "welcome",     # welcome | session | likert | comparison | reveal
        "pid":    str(uuid.uuid4()),
        "demo":   {},
        "assign": {},            # {bot1: "A"/"B", bot2: "B"/"A"}
        "order":  [],            # ["A","B"] or ["B","A"]
        "mi":     0,             # model index (0 or 1)
        "si":     0,             # session index (0, 1, 2)
        "likert": {"bot1": [], "bot2": []},
        "comp":   {},
        "free":   "",
        "client": None,
        "t0":     None,          # call start time (monotonic)
    }

def _bot(s: dict, k: str) -> str:
    return "Bot 1" if s["assign"].get("bot1") == k else "Bot 2"

def _key(s: dict) -> str:
    return s["order"][s["mi"]]

def _cbot(s: dict) -> str:
    return _bot(s, _key(s))

def _disc(s: dict) -> None:
    c = s.get("client")
    if c:
        c.close()
        s["client"] = None
    if s.get("order"):
        try:
            _release(s["order"][s["mi"]])
        except Exception:
            pass
    s["t0"] = None

# ── App ───────────────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Voice Bot Study", js=_JS, theme=THEME) as app:

        # ═══════════════════════════ SURVEY TAB ══════════════════════════════
        with gr.Tab("Survey"):
            ss = gr.State(None)
            at = gr.Textbox(visible=False, elem_id="sat")  # audio transport

            # Panels — only one visible at a time
            with gr.Group(visible=True) as P_welcome:
                gr.Markdown("## Voice Assistant Evaluation Study")
                gr.Markdown(
                    "You'll have brief voice conversations with **two different bots** "
                    "and rate each one.\n\n"
                    "- **6 sessions** total — 3 per bot, up to 45 s each\n"
                    "- Likert ratings after each bot\n"
                    "- A final comparison at the end\n\n"
                    "The bots are labeled **Bot 1** and **Bot 2**. "
                    "Real identities are revealed at the end."
                )
                start_btn = gr.Button("Start Survey", variant="primary", size="lg")

            with gr.Group(visible=False) as P_demo:
                gr.Markdown("## About You")
                age_r    = gr.Radio(
                    ["18–25", "26–35", "36–45", "46–55", "56+", "Prefer not to say"],
                    label="Age range",
                )
                native_r = gr.Radio(["Yes", "No"], label="Native English speaker?")
                va_r     = gr.Radio(
                    ["Never", "Rarely", "Sometimes", "Often", "Daily"],
                    label="How often do you use voice assistants (Siri, Alexa, Google…)?",
                )
                demo_btn = gr.Button("Next →", variant="primary")
                demo_err = gr.Markdown("", visible=False)

            with gr.Group(visible=False) as P_wait:
                gr.Markdown("## One Moment…")
                gr.HTML(
                    "<div style='text-align:center;padding:30px'>"
                    "<div style='font-size:56px'>⏳</div>"
                    "<p style='color:#777;margin-top:12px'>"
                    "This bot is currently in use. Checking again every 3 s…</p></div>"
                )

            with gr.Group(visible=False) as P_session:
                s_hdr    = gr.Markdown("")
                s_instr  = gr.HTML("")
                s_prompts = gr.HTML("")
                with gr.Row():
                    mic = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="🎙 Microphone — click to start/stop recording",
                        interactive=False,
                    )
                s_timer  = gr.HTML(
                    "<div id='st' style='font-size:22px;text-align:center;"
                    "padding:8px 0;color:#555'>45 s</div>"
                )
                s_status = gr.Markdown("_Connecting…_")
                end_btn  = gr.Button("End Session & Continue →", variant="secondary")

            with gr.Group(visible=False) as P_likert:
                l_hdr  = gr.Markdown("")
                gr.Markdown(
                    "Rate each statement from **1** (strongly disagree) "
                    "to **7** (strongly agree)."
                )
                l_rs   = [gr.Radio(["1","2","3","4","5","6","7"], label=q) for q in LIKERT_QS]
                l_btn  = gr.Button("Submit Ratings", variant="primary")
                l_err  = gr.Markdown("", visible=False)

            with gr.Group(visible=False) as P_compare:
                gr.Markdown("## Final Comparison")
                gr.Markdown("Now that you've spoken with both bots, compare them:")
                c_rs   = {k: gr.Radio(choices, label=q) for k, q, choices in COMPARE_QS}
                free_t = gr.Textbox(
                    label="What specifically made you prefer one bot? (optional)",
                    placeholder="Type your thoughts here…",
                    lines=3,
                )
                c_btn  = gr.Button("Submit & See Results", variant="primary")
                c_err  = gr.Markdown("", visible=False)

            with gr.Group(visible=False) as P_reveal:
                rev_html = gr.HTML("")
                gr.Markdown(
                    "**Thank you for participating!**  \n"
                    "Your responses have been saved anonymously."
                )

            PANELS = [P_welcome, P_demo, P_wait, P_session, P_likert, P_compare, P_reveal]

            r_tmr = gr.Timer(3.0,   active=False)   # retry while waiting
            p_tmr = gr.Timer(POLL_S, active=False)  # poll during session

            # ── Helpers (closures over components) ──────────────────────────

            def _show(panel):
                return [gr.update(visible=(p is panel)) for p in PANELS]

            def _sess_html(s: dict):
                """Build session header, instruction, prompts HTML from state."""
                sess  = SESSIONS[s["si"]]
                bot   = _cbot(s)
                hdr   = (
                    f"### {bot} — {sess['title']}"
                    f"&nbsp;·&nbsp; Session {s['si']+1}/3"
                    f"&nbsp;·&nbsp; Model {s['mi']+1}/2"
                )
                if sess["interrupt"]:
                    instr = (
                        "<div style='border:2px solid #c00;background:#fff5f5;"
                        "border-radius:6px;padding:10px 14px;margin:6px 0'>"
                        "<b style='color:#c00;font-size:15px'>⚠ INTERRUPTION TASK</b><br>"
                        f"<span style='color:#b00'>{sess['instruction']}</span>"
                        "</div>"
                    )
                else:
                    instr = (
                        f"<p style='color:#555;margin:6px 0'>{sess['instruction']}</p>"
                    )
                prompts = (
                    "<div style='background:#f6f6f6;border-radius:6px;"
                    "padding:10px 14px;margin:6px 0'>"
                    "<b>Suggested prompts:</b>"
                    "<ul style='margin:6px 0 0 18px'>"
                    + "".join(f"<li style='margin:3px 0'>{p}</li>" for p in sess["prompts"])
                    + "</ul></div>"
                )
                return hdr, instr, prompts

            # Every major handler returns this 18-tuple:
            # (ss, *PANELS[7], mic, r_tmr, p_tmr,
            #  s_hdr, s_instr, s_prompts, s_status, l_hdr, rev_html, at)
            # Indices: 0  1..7  8     9      10
            #          11     12      13       14       15     16      17

            def _go_wait(s: dict):
                return (
                    s, *_show(P_wait),
                    gr.Audio(interactive=False),
                    gr.update(active=True),    # retry on
                    gr.update(active=False),   # poll off
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    "__reset__",
                )

            def _go_session(s: dict):
                # Safety: close any lingering client before opening a new one.
                # _advance/_disc should have already done this, but guard edge cases.
                old_c = s.get("client")
                if old_c is not None:
                    try:
                        old_c.close()
                    except Exception:
                        pass
                    s["client"] = None
                s["t0"] = None

                key = _key(s)
                if not _acquire(key):
                    s["step"] = "waiting"
                    return _go_wait(s)
                url = _URLS[key]
                client = FullDuplexClient(url)
                try:
                    client.connect(client_name="survey-client")
                except Exception:
                    _release(key)
                    s["step"] = "waiting"
                    return _go_wait(s)
                s["client"] = client
                s["t0"]     = time.monotonic()
                s["step"]   = "session"
                hdr, instr, prompts = _sess_html(s)
                return (
                    s, *_show(P_session),
                    gr.Audio(interactive=True),
                    gr.update(active=False),   # retry off
                    gr.update(active=True),    # poll on
                    gr.update(value=hdr),
                    gr.update(value=instr),
                    gr.update(value=prompts),
                    gr.update(value="_Connected — speak when ready._"),
                    gr.skip(), gr.skip(),
                    "__reset__",
                )

            def _go_likert(s: dict):
                bot = _cbot(s)
                return (
                    s, *_show(P_likert),
                    gr.Audio(interactive=False),
                    gr.update(active=False),
                    gr.update(active=False),
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    gr.update(value=f"### Rate {bot}"),
                    gr.skip(),
                    "__reset__",
                )

            def _advance(s: dict):
                """Disconnect and move to next session or Likert."""
                _disc(s)
                s["si"] += 1
                if s["si"] < 3:
                    return _go_session(s)
                s["si"] = 0
                return _go_likert(s)

            # ── Event handlers ───────────────────────────────────────────────

            MAJOR = [ss, *PANELS, mic, r_tmr, p_tmr,
                     s_hdr, s_instr, s_prompts, s_status, l_hdr, rev_html, at]

            app.load(lambda: _new_state(), outputs=[ss])

            start_btn.click(
                lambda _s: (_new_state(), *_show(P_demo)),
                inputs=[ss],
                outputs=[ss, *PANELS],
            )

            def on_demo(age, native, va, s):
                if not all([age, native, va]):
                    return (
                        s, *_show(P_demo),
                        gr.Audio(interactive=False),
                        gr.update(active=False), gr.update(active=False),
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        gr.skip(),
                        gr.update(value=_warn("Please answer all questions before continuing."), visible=True),
                    )
                s = dict(s)
                s["demo"] = {"age": age, "native_english": native, "va_usage": va}
                if random.random() < 0.5:
                    s["order"]  = ["A", "B"]
                    s["assign"] = {"bot1": "A", "bot2": "B"}
                else:
                    s["order"]  = ["B", "A"]
                    s["assign"] = {"bot1": "B", "bot2": "A"}
                s["mi"] = 0
                s["si"] = 0
                return _go_session(s) + (gr.update(visible=False),)

            demo_btn.click(
                on_demo,
                inputs=[age_r, native_r, va_r, ss],
                outputs=MAJOR + [demo_err],
            )

            def on_retry(s):
                if s is None or s.get("step") != "waiting":
                    return (s,) + (gr.skip(),) * 17
                return _go_session(dict(s))

            r_tmr.tick(on_retry, inputs=[ss], outputs=MAJOR)

            def on_mic(audio, s):
                if s is None or audio is None:
                    return s
                c = s.get("client")
                if c is None or not c.connected:
                    return s
                sr, arr = audio
                f32 = np.asarray(arr, dtype=np.float32) / 32768.0
                try:
                    c.send_audio_chunk(sr, f32)
                except Exception:
                    pass
                return s

            mic.stream(on_mic, inputs=[mic, ss], outputs=[ss])

            def on_poll(s):
                """Called every 80 ms during a session. Returns (MAJOR..., s_timer) = 19 items."""
                if s is None:
                    return (gr.skip(),) * 19

                c  = s.get("client")
                t0 = s.get("t0")
                timer_html = gr.skip()

                if t0 is not None:
                    elapsed   = time.monotonic() - t0
                    remaining = max(0.0, MAX_CALL - elapsed)
                    color     = "#c00" if remaining < 10 else "#333"
                    timer_html = (
                        f"<div id='st' style='font-size:22px;text-align:center;"
                        f"padding:8px 0;color:{color}'>{remaining:.0f} s</div>"
                    )
                    if elapsed >= MAX_CALL:
                        s2 = dict(s)
                        result = _advance(s2)   # 18-tuple (MAJOR)
                        return result + (timer_html,)

                if c is None:
                    return (gr.skip(),) * 18 + (timer_html,)

                chunk = c.pop_audio_chunk(timeout=0.0)
                if chunk:
                    sr, arr = chunk
                    return (
                        (gr.skip(),) * 17
                        + (_wav_uri(arr, sr),)  # audio_transport at index 17
                        + (timer_html,)         # s_timer at index 18
                    )

                return (gr.skip(),) * 17 + (gr.skip(), timer_html)

            p_tmr.tick(on_poll, inputs=[ss], outputs=MAJOR + [s_timer])

            at.change(
                fn=None,
                inputs=[at],
                js="(u) => { if (window._audioEnqueue) window._audioEnqueue(u); }",
            )

            end_btn.click(_advance, inputs=[ss], outputs=MAJOR)

            def on_likert(*args):
                s      = args[0]
                vals   = args[1:]
                if any(v is None for v in vals):
                    return (
                        s, *_show(P_likert),
                        gr.Audio(interactive=False),
                        gr.update(active=False), gr.update(active=False),
                        gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                        gr.skip(),
                        gr.update(value=_warn("Please answer all questions before submitting."), visible=True),
                    )
                s = dict(s)
                bk = "bot1" if s["mi"] == 0 else "bot2"
                s["likert"][bk] = [int(v) for v in vals]
                s["mi"] += 1
                if s["mi"] < 2:
                    s["si"] = 0
                    return _go_session(s) + (gr.update(visible=False),)
                # Both models done → comparison
                s["step"] = "comparison"
                return (
                    s, *_show(P_compare),
                    gr.Audio(interactive=False),
                    gr.update(active=False), gr.update(active=False),
                    gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(),
                    "__reset__",
                    gr.update(visible=False),
                )

            l_btn.click(on_likert, inputs=[ss, *l_rs], outputs=MAJOR + [l_err])

            def on_compare(*args):
                s      = args[0]
                cvals  = args[1:5]
                ftval  = args[5]
                if any(v is None for v in cvals):
                    return (
                        s, *_show(P_compare),
                        gr.skip(),   # rev_html unchanged
                        gr.update(value=_warn("Please answer all comparison questions."), visible=True),
                    )
                s = dict(s)
                for (k, _, _), v in zip(COMPARE_QS, cvals):
                    s["comp"][k] = v
                s["free"] = ftval or ""
                s["step"] = "reveal"
                _save(s)

                b1 = _NAMES[s["assign"]["bot1"]]
                b2 = _NAMES[s["assign"]["bot2"]]
                html = (
                    "<div style='text-align:center;padding:24px'>"
                    "<h2 style='margin-bottom:16px'>🎉 The Reveal!</h2>"
                    "<table style='margin:0 auto;border-collapse:collapse;font-size:17px'>"
                    "<tr><td style='padding:8px 20px;text-align:right'><b>Bot 1</b></td>"
                    f"<td style='padding:8px 20px;color:#ea580c'>{b1}</td></tr>"
                    "<tr><td style='padding:8px 20px;text-align:right'><b>Bot 2</b></td>"
                    f"<td style='padding:8px 20px;color:#0d9488'>{b2}</td></tr>"
                    "</table></div>"
                )
                return (
                    s, *_show(P_reveal),
                    gr.update(value=html),
                    gr.update(visible=False),
                )

            c_btn.click(
                on_compare,
                inputs=[ss, *[c_rs[k] for k, _, _ in COMPARE_QS], free_t],
                outputs=[ss, *PANELS, rev_html, c_err],
            )

        # ═══════════════════════════ FREE PLAY TAB ═══════════════════════════
        with gr.Tab("Free Play"):
            gr.Markdown(
                "## Free Play\n"
                "Open-ended voice conversation — no data is saved from this tab."
            )

            fp_state = gr.State(None)
            fp_at    = gr.Textbox(visible=False, elem_id="fp-at")

            with gr.Row():
                fp_model = gr.Dropdown(
                    choices=[
                        (f"Model A — {_NAMES['A']} (port {MODEL_A_PORT})", "A"),
                        (f"Model B — {_NAMES['B']} (port {MODEL_B_PORT})", "B"),
                    ],
                    value="A",
                    label="Model",
                )
                fp_conn_btn = gr.Button("Connect", variant="primary")
                fp_disc_btn = gr.Button("Disconnect", variant="secondary", interactive=False)

            fp_mic = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="🎙 Microphone",
                interactive=False,
            )
            fp_status = gr.Markdown("_Select a model and click Connect._")

            fp_tmr = gr.Timer(POLL_S, active=False)

            def fp_on_load():
                return {"client": None}

            def fp_connect(model_key, fstate):
                fstate = dict(fstate or {"client": None})
                c = fstate.get("client")
                if c:
                    c.close()
                client = FullDuplexClient(_URLS[model_key])
                try:
                    client.connect(client_name="freeplay-client")
                except Exception as exc:
                    fstate["client"] = None
                    return (
                        fstate,
                        gr.Audio(interactive=False),
                        f"_Connection failed: {exc}_",
                        gr.Button(interactive=True),
                        gr.Button(interactive=False),
                        gr.update(active=False),
                        "__reset__",
                    )
                fstate["client"] = client
                return (
                    fstate,
                    gr.Audio(interactive=True),
                    f"_Connected to {_NAMES[model_key]} — speak freely._",
                    gr.Button(interactive=False),
                    gr.Button(interactive=True),
                    gr.update(active=True),
                    "__reset__",
                )

            def fp_disconnect(fstate):
                fstate = dict(fstate or {"client": None})
                c = fstate.get("client")
                if c:
                    c.close()
                fstate["client"] = None
                return (
                    fstate,
                    gr.Audio(interactive=False),
                    "_Disconnected._",
                    gr.Button(interactive=True),
                    gr.Button(interactive=False),
                    gr.update(active=False),
                    "__reset__",
                )

            def fp_mic_recv(audio, fstate):
                if fstate is None or audio is None:
                    return fstate
                c = fstate.get("client")
                if c is None or not c.connected:
                    return fstate
                sr, arr = audio
                f32 = np.asarray(arr, dtype=np.float32) / 32768.0
                try:
                    c.send_audio_chunk(sr, f32)
                except Exception:
                    pass
                return fstate

            def fp_poll(fstate):
                if fstate is None:
                    return gr.skip(), gr.skip()
                c = fstate.get("client")
                if c is None:
                    return gr.skip(), gr.skip()
                chunk = c.pop_audio_chunk(timeout=0.0)
                if chunk:
                    sr, arr = chunk
                    return _wav_uri(arr, sr), gr.skip()
                return gr.skip(), gr.skip()

            _FP_CONN_OUTS = [fp_state, fp_mic, fp_status,
                             fp_conn_btn, fp_disc_btn, fp_tmr, fp_at]

            app.load(fp_on_load, outputs=[fp_state])

            fp_conn_btn.click(
                fp_connect,
                inputs=[fp_model, fp_state],
                outputs=_FP_CONN_OUTS,
            )
            fp_disc_btn.click(
                fp_disconnect,
                inputs=[fp_state],
                outputs=_FP_CONN_OUTS,
            )
            fp_mic.stream(fp_mic_recv, inputs=[fp_mic, fp_state], outputs=[fp_state])
            fp_tmr.tick(fp_poll, inputs=[fp_state], outputs=[fp_at, fp_status])

            fp_at.change(
                fn=None,
                inputs=[fp_at],
                js="(u) => { if (window._audioEnqueue) window._audioEnqueue(u); }",
            )

        app.queue()
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Blind A/B survey for two duplex servers")
    parser.add_argument(
        "--model-a-port",
        type=int,
        default=MODEL_A_PORT,
        help=f"Websocket port of the Model A server.py instance (default {MODEL_A_PORT}).",
    )
    parser.add_argument(
        "--model-b-port",
        type=int,
        default=MODEL_B_PORT,
        help=f"Websocket port of the Model B server.py instance (default {MODEL_B_PORT}).",
    )
    args = parser.parse_args()
    MODEL_A_PORT = args.model_a_port
    MODEL_B_PORT = args.model_b_port
    _URLS = {
        "A": server_url_from_address(f"127.0.0.1:{MODEL_A_PORT}"),
        "B": server_url_from_address(f"127.0.0.1:{MODEL_B_PORT}"),
    }

    build_app().launch(share=True)

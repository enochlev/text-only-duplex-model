#!/usr/bin/env python3
"""run_demo.py — clean, non-gradio survey/demo front-end for the full-duplex voice model.

Serves a single self-contained page (run_demo_ui.html) that talks DIRECTLY to one or two
running duplex model backends (server.py --share URLs) over their WebSocket protocol. No
gradio UI — just a FastAPI page exposed (optionally) through the same FRP tunnel gradio uses.

Flow: informed-consent gate (required checkbox) → blinded A/B survey (talk to each system,
one Start/Stop button, live connection status + mic/bot volume meters, then rate) →
comparison + optional demographics → responses saved to disk. `--enable_free_chat` adds a
tab to pick a system and just chat (no survey). The survey is disabled unless BOTH
--model_a_url and --model_b_url are provided.

Examples:
    # survey (A vs B), exposed publicly via FRP tunnel:
    python run_demo.py --model_a_url wss://aaa.gradio.live/ws \\
                       --model_b_url wss://bbb.gradio.live/ws --share
    # local free-chat only, no survey:
    python run_demo.py --model_a_url ws://127.0.0.1:8998/ws --enable_free_chat --port 7870
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "run_demo_ui.html")

# --- Default informed-consent text (DRAFT). Placeholders in [BRACKETS] are for the study
# team / advisor to fill in per the IRB-approved protocol. Override with --consent-file. ---
DEFAULT_CONSENT_HTML = """
<p><b>Study title:</b> Evaluation of turn-taking in a conversational voice assistant.</p>
<p><b>Investigators:</b> [PRINCIPAL INVESTIGATOR NAME], [DEPARTMENT], [INSTITUTION].
   Faculty advisor: [ADVISOR NAME]. Contact: [CONTACT EMAIL].</p>
<p><b>Purpose.</b> You are invited to take part in a research study evaluating how naturally a
   spoken-dialogue AI takes turns in conversation. Your participation will help us compare
   different versions of the system.</p>
<p><b>What you will do.</b> You will have one or two short spoken conversations (about one
   minute each) with voice-AI systems using your microphone, and then answer a few short
   questions rating your experience. The whole session takes about 5–10 minutes.</p>
<p><b>Voluntary participation.</b> Your participation is entirely voluntary. You may decline
   to answer any question and may stop at any time without penalty by simply closing the page.</p>
<p><b>Risks.</b> There are no anticipated risks beyond those of everyday conversation.</p>
<p><b>Benefits &amp; compensation.</b> There is no direct benefit to you. [COMPENSATION, IF ANY].
   Your input helps improve conversational AI.</p>
<p><b>Data &amp; confidentiality.</b> We record your questionnaire answers and basic technical
   information (browser type, timing). We do <b>not</b> collect your name or contact details,
   and audio is processed only to run the conversation and is not stored by this survey.
   Responses are stored without identifiers and reported only in aggregate.</p>
<p><b>Questions or concerns.</b> Contact the investigators at [CONTACT EMAIL], or the
   [INSTITUTION] IRB at [IRB CONTACT] regarding your rights as a participant.
   [IRB PROTOCOL #].</p>
<p>By checking the box below you confirm that you are at least 18 years old, have read and
   understood this information, and voluntarily agree to participate.</p>
"""

# --- Default questionnaire (standard turn-taking evaluation). Override with --questions-file. ---
DEFAULT_LIKERT = [
    {"id": "natural", "type": "likert", "text": "The system's timing felt natural — it waited for me and replied at the right moments.",
     "options": ["1", "2", "3", "4", "5"], "low": "Strongly disagree", "high": "Strongly agree"},
    {"id": "prompt", "type": "likert", "text": "It responded promptly once I finished speaking.",
     "options": ["1", "2", "3", "4", "5"], "low": "Strongly disagree", "high": "Strongly agree"},
    {"id": "nointerrupt", "type": "likert", "text": "It avoided interrupting me while I was still talking.",
     "options": ["1", "2", "3", "4", "5"], "low": "Strongly disagree", "high": "Strongly agree"},
    {"id": "relevant", "type": "likert", "text": "Its responses were relevant to what I said.",
     "options": ["1", "2", "3", "4", "5"], "low": "Strongly disagree", "high": "Strongly agree"},
    {"id": "overall", "type": "likert", "text": "Overall, the conversation felt smooth and human-like.",
     "options": ["1", "2", "3", "4", "5"], "low": "Strongly disagree", "high": "Strongly agree"},
]
DEFAULT_COMPARE = [
    {"id": "pref_overall", "type": "choice", "text": "Which system felt more natural to talk to overall?",
     "options": ["System 1", "System 2", "No preference"]},
    {"id": "pref_turntaking", "type": "choice", "text": "Which handled turn-taking better (waiting, not interrupting)?",
     "options": ["System 1", "System 2", "No preference"]},
    {"id": "age", "type": "choice", "text": "Your age range (optional):",
     "options": ["18–24", "25–34", "35–44", "45–54", "55+", "Prefer not to say"]},
    {"id": "native_en", "type": "choice", "text": "Are you a native English speaker? (optional)",
     "options": ["Yes", "No", "Prefer not to say"]},
    {"id": "va_experience", "type": "choice", "text": "How often do you use voice assistants? (optional)",
     "options": ["Never", "Rarely", "Sometimes", "Often", "Daily"]},
    {"id": "comments", "type": "text", "text": "Any other comments? (optional)"},
]


def _read(path: str | None, default):
    if not path:
        return default
    with open(os.path.expanduser(path), "r", encoding="utf-8") as f:
        data = f.read()
    return data if isinstance(default, str) else json.loads(data)


def create_app(args) -> FastAPI:
    app = FastAPI()
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)

    consent_html = _read(args.consent_file, DEFAULT_CONSENT_HTML)
    questions = _read(args.questions_file, {"likert": DEFAULT_LIKERT, "compare": DEFAULT_COMPARE}) \
        if args.questions_file else {"likert": DEFAULT_LIKERT, "compare": DEFAULT_COMPARE}

    survey_enabled = bool(args.model_a_url and args.model_b_url)

    with open(UI_PATH, "r", encoding="utf-8") as f:
        ui_html = f.read()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return ui_html

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/config")
    def config():
        return JSONResponse({
            "study_title": args.title,
            "model_a_url": args.model_a_url or "",
            "model_b_url": args.model_b_url or "",
            "enable_free_chat": bool(args.enable_free_chat),
            "survey_enabled": survey_enabled,
            "consent_html": consent_html,
            "likert": questions["likert"],
            "compare": questions["compare"],
        })

    @app.post("/submit")
    async def submit(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)
        rec = {"id": uuid.uuid4().hex[:12], "received_at": time.time(), **body}
        path = os.path.join(out_dir, "responses.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[submit] {rec['id']} kind={body.get('kind')} → {path}")
        return {"ok": True, "id": rec["id"]}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean survey/demo front-end for the duplex model.")
    ap.add_argument("--model_a_url", default="", help="WS URL of system A (e.g. wss://xxx.gradio.live/ws)")
    ap.add_argument("--model_b_url", default="", help="WS URL of system B (survey needs both A and B)")
    ap.add_argument("--enable_free_chat", action="store_true", help="Add a free-chat tab (pick a system, just talk)")
    ap.add_argument("--share", action="store_true", help="Expose publicly via a gradio FRP tunnel (*.gradio.live)")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--title", default="Conversational Voice AI — Research Study")
    ap.add_argument("--consent-file", default=None, help="Path to an HTML consent file (overrides the built-in draft)")
    ap.add_argument("--questions-file", default=None, help="Path to a JSON file with {likert:[...],compare:[...]}")
    ap.add_argument("--out", default="~/scratch/survey_responses", help="Directory for responses.jsonl")
    args = ap.parse_args()

    if not (args.model_a_url and args.model_b_url):
        print("[warn] survey needs BOTH --model_a_url and --model_b_url; survey tab will be disabled.")
    app = create_app(args)

    public_url = None
    if args.share:
        import secrets
        from gradio.networking import setup_tunnel
        try:
            public_url = setup_tunnel(args.host if args.host != "0.0.0.0" else "127.0.0.1",
                                      args.port, secrets.token_urlsafe(32), None, None)
            print(f"[share] public URL : {public_url}")
            print(f"[share] give participants this link. Expires ~1 week.")
        except Exception as exc:
            print(f"[share] tunnel failed ({type(exc).__name__}: {exc}); serving locally only")

    print(f"[run_demo] serving on http://{args.host}:{args.port}  (survey_enabled={bool(args.model_a_url and args.model_b_url)}, free_chat={args.enable_free_chat})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

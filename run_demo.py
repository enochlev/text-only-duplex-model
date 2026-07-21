#!/usr/bin/env python3
"""run_demo.py — in-person study front-end for the full-duplex voice model (IRB24-222 protocol).

Serves a single self-contained page (run_demo_ui.html) that talks DIRECTLY to one or two
running duplex model backends (server.py --share URLs) over their WebSocket protocol. No
gradio UI — just a FastAPI page exposed (optionally) through the same FRP tunnel gradio uses.

Participant flow (one supervised session):
  1. Informed consent (IRB24-222) — typed name + a SEPARATE agreement checkbox per system
  2. Instructions page
  3. Talk with System 1  →  Questionnaire 1: shown a 5-digit Participant ID PIN + a link to
     the Google Form (PIN pre-filled via the form's entry id)
  4. Talk with System 2  →  Questionnaire 2 (its own PIN)
  5. Debriefing statement — typed name signature
  6. Optional gift-card pickup: a third 5-digit PIN is revealed and saved, matched in person

The two systems are shown in a random blinded order chosen server-side per session. Every
step posts a checkpoint record to responses.jsonl, so a session that dies midway still
leaves its consent/PINs/order on disk.

Examples:
    # full study (A vs B), exposed publicly via FRP tunnel:
    python run_demo.py --model_a_url wss://aaa.gradio.live/ws \\
                       --model_b_url wss://bbb.gradio.live/ws --share
    # UI review without any models connected (talk steps show a not-configured notice):
    python run_demo.py --share
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse

HERE = os.path.dirname(os.path.abspath(__file__))
UI_PATH = os.path.join(HERE, "run_demo_ui.html")
DOC_DIR = os.path.join(HERE, "data", "survey")
CONSENT_PDF = os.path.join(DOC_DIR, "IRB informed consent form.pdf")
DEBRIEF_PDF = os.path.join(DOC_DIR, "IRB debriefing statement.pdf")

# Google Form questionnaire (published; first question = required "Participant ID").
# The entry id was extracted from the form's FB_PUBLIC_LOAD_DATA_; prefill link is
#   <form_url>?usp=pp_url&entry.<id>=<PIN>
DEFAULT_FORM_URL = ("https://docs.google.com/forms/d/e/"
                    "1FAIpQLSdqsfHcazIgA5yXrs028aEXxnClCfPRjeWM8wPSM2Y3eUpYdA/viewform")
DEFAULT_FORM_ENTRY = "156546644"


def _read_doc(path: str | None, default_name: str) -> str:
    """Read an HTML fragment; fall back to the checked-in data/survey/ default."""
    p = os.path.expanduser(path) if path else os.path.join(DOC_DIR, default_name)
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"<p><b>Missing document:</b> {os.path.basename(p)} — configure it on the server.</p>"


class PinAllocator:
    """5-digit PINs, unique across everything ever issued into this out-dir."""

    def __init__(self, jsonl_path: str):
        self._used: set[str] = set()
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    for k in ("pin_q1", "pin_q2", "pin_gift"):
                        if rec.get(k):
                            self._used.add(str(rec[k]))
        except FileNotFoundError:
            pass

    def take(self) -> str:
        while True:
            pin = f"{random.randint(10000, 99999)}"
            if pin not in self._used:
                self._used.add(pin)
                return pin


def create_app(args) -> FastAPI:
    app = FastAPI()
    out_dir = os.path.expanduser(args.out)
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "responses.jsonl")
    pins = PinAllocator(jsonl_path)

    consent_html = _read_doc(args.consent_file, "consent_irb24222.html")
    instructions_html = _read_doc(args.instructions_file, "instructions.html")
    debrief_html = _read_doc(args.debrief_file, "debrief_irb24222.html")

    def _load_pdf(path):
        try:
            with open(os.path.expanduser(path), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    consent_pdf = _load_pdf(args.consent_pdf)
    debrief_pdf = _load_pdf(args.debrief_pdf)

    models_configured = bool(args.model_a_url and args.model_b_url)
    # compensation differs by mode: $10 in person, $5 online (override with --gift-amount)
    gift_amount = args.gift_amount or ("$10" if args.inperson else "$5")

    with open(UI_PATH, "r", encoding="utf-8") as f:
        ui_html = f.read()

    def _append(rec: dict) -> None:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return ui_html

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    # ---- in-person mode (one supervised session per machine) --------------------
    # The wizard tells the local robot client which blinded slot to run via
    # /live_control; retico/inperson.py polls /inperson_target, connects the
    # mic→duplex→Misty pipeline to that slot's server, and pushes transcript
    # snapshots to /live_snapshot, which the wizard polls to render live.
    live = {"active_slot": None, "snapshot": None, "snapshot_ts": 0.0,
            "robot_ts": 0.0, "robot_connected": False}

    @app.post("/live_control")
    async def live_control(request: Request):
        if not args.inperson:
            return JSONResponse({"ok": False, "error": "not in in-person mode"}, status_code=400)
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)
        slot = body.get("active_slot")
        if slot not in ("A", "B", None):
            return JSONResponse({"ok": False, "error": "bad slot"}, status_code=400)
        live["active_slot"] = slot
        live["snapshot"] = None  # fresh talk step → don't show the previous system's transcript
        live["snapshot_ts"] = 0.0
        print(f"[inperson] active_slot → {slot}")
        return {"ok": True, "active_slot": slot}

    @app.get("/inperson_target")
    def inperson_target():
        slot = live["active_slot"]
        url = {"A": args.model_a_url, "B": args.model_b_url}.get(slot, "")
        return {"active_slot": slot, "url": url or ""}

    @app.post("/live_snapshot")
    async def push_live_snapshot(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)
        # every POST is a robot-client heartbeat, snapshot or not
        live["robot_ts"] = time.time()
        live["robot_connected"] = bool(body.get("connected"))
        if body.get("snapshot") is not None:
            live["snapshot"] = body["snapshot"]
            live["snapshot_ts"] = time.time()
        # echo the active slot so the robot client knows when to disconnect
        return {"ok": True, "active_slot": live["active_slot"]}

    @app.get("/live_snapshot")
    def get_live_snapshot():
        return {"snapshot": live["snapshot"], "ts": live["snapshot_ts"],
                "active_slot": live["active_slot"],
                # robot client is "alive" if it POSTed within the last 5 s
                "robot_alive": (time.time() - live["robot_ts"]) < 5.0,
                "robot_connected": live["robot_connected"]}

    @app.get("/consent.pdf")
    def consent_pdf_route():
        if consent_pdf is None:
            return JSONResponse({"error": "consent pdf not found"}, status_code=404)
        return Response(consent_pdf, media_type="application/pdf",
                        headers={"Content-Disposition": "inline; filename=consent.pdf"})

    @app.get("/debrief.pdf")
    def debrief_pdf_route():
        if debrief_pdf is None:
            return JSONResponse({"error": "debrief pdf not found"}, status_code=404)
        return Response(debrief_pdf, media_type="application/pdf",
                        headers={"Content-Disposition": "inline; filename=debrief.pdf"})

    @app.get("/config")
    def config():
        prefill = f"{args.form_url}?usp=pp_url&entry.{args.form_entry}=" if args.form_entry else ""
        return JSONResponse({
            "study_title": args.title,
            "model_a_url": args.model_a_url or "",
            "model_b_url": args.model_b_url or "",
            "models_configured": models_configured,
            "inperson": bool(args.inperson),
            "gift_amount": gift_amount,
            "enable_free_chat": bool(args.enable_free_chat),
            # PDF viewer is preferred; HTML transcription kept as a fallback when a PDF is absent.
            "consent_pdf_url": "consent.pdf" if consent_pdf is not None else "",
            "debrief_pdf_url": "debrief.pdf" if debrief_pdf is not None else "",
            "consent_html": consent_html,
            "instructions_html": instructions_html,
            "debrief_html": debrief_html,
            "form_url": args.form_url,
            "form_prefill": prefill,
        })

    @app.post("/session")
    async def new_session(request: Request):
        """Start a participant session: allocate the 3 PINs + blinded order, log it."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        sid = uuid.uuid4().hex[:12]
        order = ["A", "B"] if random.random() < 0.5 else ["B", "A"]
        rec = {
            "kind": "session_start", "session_id": sid, "ts": time.time(),
            "pin_q1": pins.take(), "pin_q2": pins.take(), "pin_gift": pins.take(),
            "order": order,  # order[0] is what the participant sees as "System 1"
            "ua": body.get("ua", ""),
        }
        _append(rec)
        print(f"[session] {sid} order={order} pins q1={rec['pin_q1']} q2={rec['pin_q2']} gift={rec['pin_gift']}")
        return {k: rec[k] for k in ("session_id", "pin_q1", "pin_q2", "pin_gift", "order")}

    @app.post("/checkpoint")
    async def checkpoint(request: Request):
        """Incremental per-step record (consent / interact / questionnaire / debrief / gift)."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"ok": False, "error": "bad json"}, status_code=400)
        rec = {"id": uuid.uuid4().hex[:12], "received_at": time.time(), **body}
        _append(rec)
        print(f"[checkpoint] session={body.get('session_id')} kind={body.get('kind')} → {jsonl_path}")
        return {"ok": True, "id": rec["id"]}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="In-person study front-end for the duplex model (IRB24-222).")
    ap.add_argument("--model_a_url", default="", help="WS URL of system A (e.g. wss://xxx.gradio.live/ws)")
    ap.add_argument("--model_b_url", default="", help="WS URL of system B")
    ap.add_argument("--enable_free_chat", action="store_true", help="Add a dev free-chat view (pick a system, just talk)")
    ap.add_argument("--inperson", action="store_true",
                    help="In-person (Misty) mode: run locally on the intern PC alongside retico/inperson.py; "
                         "the talk step shows the robot conversation's live transcript instead of the browser mic panel")
    ap.add_argument("--share", action="store_true", help="Expose publicly via a gradio FRP tunnel (*.gradio.live)")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--title", default="Interaction with Full-Duplex Large Language Models")
    ap.add_argument("--consent-file", default=None, help="HTML consent fallback (default data/survey/consent_irb24222.html)")
    ap.add_argument("--instructions-file", default=None, help="HTML instructions override (default data/survey/instructions.html)")
    ap.add_argument("--debrief-file", default=None, help="HTML debrief fallback (default data/survey/debrief_irb24222.html)")
    ap.add_argument("--consent-pdf", default=CONSENT_PDF, help="PDF shown in the consent step (default the IRB consent PDF)")
    ap.add_argument("--debrief-pdf", default=DEBRIEF_PDF, help="PDF shown in the debrief step (default the IRB debrief PDF)")
    ap.add_argument("--form-url", default=DEFAULT_FORM_URL, help="Google Form URL for the questionnaires")
    ap.add_argument("--form-entry", default=DEFAULT_FORM_ENTRY, help="Form entry id of the Participant ID question ('' = no prefill)")
    ap.add_argument("--gift-amount", default=None, help="Gift-card amount shown to participants (default: $10 in-person, $5 online)")
    ap.add_argument("--out", default="~/scratch/survey_responses", help="Directory for responses.jsonl")
    args = ap.parse_args()

    if not (args.model_a_url and args.model_b_url):
        print("[warn] no --model_a_url/--model_b_url — talk steps will show a not-configured notice (UI review mode).")
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

    print(f"[run_demo] serving on http://{args.host}:{args.port}  (models_configured={bool(args.model_a_url and args.model_b_url)}, free_chat={args.enable_free_chat})")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

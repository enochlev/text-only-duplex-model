# In-person study front-end (`run_demo.py`) — IRB24-222 protocol

A clean, non-gradio web front-end for the supervised human study of the full-duplex voice
model. It serves one self-contained page (`run_demo_ui.html`) that talks the duplex
WebSocket protocol **directly** to two running model backends (`server.py --share` URLs).
No gradio UI; optional FRP tunnel for public access.

## Participant flow (one supervised session)

1. **Informed consent** (IRB24-222, `data/survey/consent_irb24222.html`) — typed name +
   a **separate agreement checkbox per system** (System 1 / System 2), both required.
2. **Instructions** (`data/survey/instructions.html` — team-drafted, edit freely).
3. **Talk with System 1** — Start/Stop panel with connection pill + You/Bot volume meters.
4. **Questionnaire 1** — a 5-digit **Participant ID PIN** is shown, plus a link to the
   Google Form with the PIN pre-filled (`?usp=pp_url&entry.156546644=<PIN>`). The
   participant confirms submission before continuing.
5. **Talk with System 2** → **Questionnaire 2** (its own PIN).
6. **Debriefing statement** (`data/survey/debrief_irb24222.html`) — typed-name signature.
7. **Gift card** (optional) — three choices: submit an **email address** to receive the
   gift card (one per individual), "already received one", or decline. The email is
   saved in the checkpoint record. Amount shown is **$5 online / $10 in-person**
   (auto by mode; `--gift-amount` overrides).

The questionnaire PINs are invisible to the participant — they ride along in the
prefilled form link (`entry.156546644=<PIN>`) automatically. The on-screen PIN + manual
instructions only appear as a fallback if prefill is disabled (`--form-entry ''`).

The A/B order is blinded, chosen server-side per session, and recorded. All three PINs are
allocated server-side (`/session`) and guaranteed unique across the out-dir's history.

## Run it

```bash
# full study (A vs B), exposed publicly:
python run_demo.py \
  --model_a_url wss://<systemA>.gradio.live/ws \
  --model_b_url wss://<systemB>.gradio.live/ws \
  --share

# UI review without models (talk steps show a "not connected" banner):
python run_demo.py --share

# dev free-chat view (small link, top-right):
python run_demo.py --model_a_url ... --model_b_url ... --enable_free_chat --share
```

Useful flags: `--consent-file/--instructions-file/--debrief-file` (HTML overrides for the
`data/survey/` defaults), `--form-url`/`--form-entry` (Google Form + Participant-ID entry
id; `--form-entry ''` disables prefill), `--title`, `--port`, `--out`.

`?preview=[instructions|talk|quest|debrief|gift]` jumps to a step with fake PINs and posts
nothing (for screenshots/review); `?autostart` clicks Start automatically.

## Data (`--out`, default `~/scratch/survey_responses/responses.jsonl`)

One JSON line per event, all linked by `session_id`:

- `session_start` — `pin_q1`, `pin_q2`, `pin_gift`, `order` (order[0] = what the
  participant sees as "System 1"; `A`/`B` = `--model_a_url`/`--model_b_url`), `ua`.
- `consent` — `name`, `agree_system1`, `agree_system2`, `ts`.
- `interact` — `which` (1|2), `hidden_model`, `connected`, `timer_s`; online adds
  `talk_s`, `ws_sessions`, `n_conversations`; in-person adds `inperson: true`, `n_blocks`.
- `questionnaire` — `which`, `pin`, `opened_form`.
- `debrief` — `name`, `acknowledged`.
- `gift` — `gift_choice` (`email` | `already_received` | `declined`), `email` (if email).

Questionnaire answers live in the **Google Form** responses; join them to sessions on the
Participant ID field (= `pin_q1`/`pin_q2`). Each step checkpoints immediately, so a session
that dies midway still leaves consent/PINs/order on disk.

## In-person (Misty) mode — intern runbook

In-person sessions run the **same wizard locally on the intern's PC** (a local link can
reach the Misty robot on the LAN; the hosted gradio link cannot). Audio flows through the
retico pipeline (PC mic → 16 k → [hush] → Borah duplex server → Misty speaker); the wizard
shows the live transcript and controls which blinded system the robot client talks to.

On the intern PC (two terminals + a browser):

```bash
# 1. the wizard, locally, with the Borah model URLs:
python run_demo.py --inperson \
  --model_a_url wss://<base>.gradio.live/ws --model_b_url wss://<run9>.gradio.live/ws

# 2. the robot client (from the retico/ dir; set MISTY_IP in retico/.env):
cd retico && uv run inperson.py

# 3. open http://localhost:7870 for the participant
```

The robot client idles until the participant reaches a talk step, then automatically
connects to that step's (blinded) system, streams mic→server→Misty, and relays the
transcript so the participant sees it live — including on the questionnaire's review
panel. It disconnects when they click "I'm done" or the 5-minute timer fires, and waits
for the next talk step. One `inperson.py` launch covers the whole session. Per-session
stereo WAVs (L=user, R=bot) land in `retico/debug_wavs/inperson_*.wav`.

Responses save to the intern PC's `~/scratch/survey_responses/responses.jsonl` — collect
these files and concatenate with the online JSONL for analysis.

## Notes

- The consent and debrief HTML are faithful transcriptions of the IRB-approved PDFs in
  `data/survey/` — don't reword them without checking against the PDFs.
- The instructions page is a study-team draft (not an IRB document) — edit freely.
- The Google Form (`B - Cozmo`, forms.gle/jymtgBkestfN8QPK9) must keep "Participant ID" as
  its first question; if the form is rebuilt, update `--form-entry` (find the new
  `entry.<id>` in the form page source, `FB_PUBLIC_LOAD_DATA_`).
- Backends are picked with `--model_a_url/--model_b_url`; keep serving **bf16** (fp8 breaks
  the idle decision) and one vLLM per GPU.

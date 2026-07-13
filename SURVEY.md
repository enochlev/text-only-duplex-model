# Survey / demo front-end (`run_demo.py`)

A clean, non-gradio web front-end for running a human evaluation of the full-duplex voice
model. It serves one self-contained page that talks the duplex WebSocket protocol **directly**
to one or two running model backends (`server.py --share` URLs). No gradio UI; optional FRP
tunnel for public access.

## Run it

```bash
# A/B survey (both systems), exposed publicly:
python run_demo.py \
  --model_a_url wss://<systemA>.gradio.live/ws \
  --model_b_url wss://<systemB>.gradio.live/ws \
  --share

# add a free-chat tab (pick a system and just talk, no survey):
python run_demo.py --model_a_url ... --model_b_url ... --enable_free_chat --share

# local free-chat only (survey auto-disabled with <2 models):
python run_demo.py --model_a_url ws://127.0.0.1:8998/ws --enable_free_chat --port 7870
```

Give participants the printed `[share] public URL`. Backends are stood up separately with
`server.py --cpm --share` (one per system, e.g. base vs RL-trained).

| Flag | Meaning |
|---|---|
| `--model_a_url` / `--model_b_url` | WS URLs of the systems. **Survey needs both.** |
| `--enable_free_chat` | Adds a "Free chat" tab with a system picker. |
| `--share` | Expose via `*.gradio.live` FRP tunnel (no gradio UI). |
| `--out DIR` | Where `responses.jsonl` is written (default `~/scratch/survey_responses`). |
| `--consent-file FILE` | HTML consent text (overrides the built-in draft). |
| `--questions-file FILE` | JSON `{"likert":[...],"compare":[...]}` (overrides defaults). |
| `--title` | Study title shown at the top. |

Preview the UI without any backend: open `/?preview=survey`, `/?preview=rate`, or
`/?preview=freechat`.

## Participant flow

1. **Informed-consent gate** — consent text + a required checkbox; "Agree & Continue" is
   disabled until it's ticked.
2. **Blinded A/B survey** — talk to each system (order randomised, shown as "System 1/2"):
   one **Start / Stop-Reset** button, a live connection pill (Disconnected → Connecting →
   Connected), and **You / Bot volume meters**. Then rate that system.
3. **Comparison + optional demographics**, then responses are saved.

## Consent form — IMPORTANT

The built-in consent (`DEFAULT_CONSENT_HTML` in `run_demo.py`) is a **DRAFT template** with
`[BRACKETED]` placeholders. It is **not** a substitute for your IRB-approved document. Before
collecting real data, replace it with the advisor's approved consent via `--consent-file`
(and confirm human-subjects training / CITI is on file). The checkbox records voluntary,
informed agreement (18+), timestamped with each submission.

## Data collected (`responses.jsonl`, one JSON object per submission)

| Field | Meaning |
|---|---|
| `id`, `received_at`, `ts` | server id/time, client ISO time |
| `kind` | `"survey"` |
| `consent` | `true` (gate was passed) |
| `order` | e.g. `["B","A"]` — which hidden model was shown as System 1 vs 2 (blinding key) |
| `duration_s` | wall-clock length of the session |
| `sessions[]` | per system: `system_label`, `hidden_model` (A/B), `ratings` (Likert answers) |
| `comparison` | preference + optional demographics + free-text comments |
| `ua` | browser user-agent |

No names/contact info and no audio are stored by the survey; audio is only streamed to run
the conversation. Analyse by mapping `hidden_model` → actual system (base vs trained).

## Default questionnaire

Per system (1–5, strongly disagree→agree): timing felt natural · responded promptly ·
avoided interrupting · responses relevant · overall smooth/human-like. Final comparison:
which felt more natural · which handled turn-taking better · age · native English · voice-
assistant experience · comments. Edit via `--questions-file`.

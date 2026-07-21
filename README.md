# text-only-duplex-model

Real-time full-duplex audio conversation agent. The agent speaks and listens
simultaneously — words are committed to audio blocks on a rolling schedule while
Parakeet ASR transcribes microphone input and corrects historical blocks. The
turn-taking policy is trained with reinforcement learning (REINFORCE over
block-level rewards; see `CLAUDE.md` for the full architecture reference).

## Models

| Model | HuggingFace | Description |
|---|---|---|
| Base | [`enochlev/MiniCPM-duplex`](https://huggingface.co/enochlev/MiniCPM-duplex) | MiniCPM-duplex base (from `xinrongzhang2022/MiniCPM-duplex`) |
| RL fine-tuned | [`enochlev/MiniCPM-duplex-rl`](https://huggingface.co/enochlev/MiniCPM-duplex-rl) | Base + 180 steps of turn-taking RL (run 9; interrupts less, yields to barge-ins) |

Serve either with vLLM **bf16** (fp8 + greedy breaks the idle/speak decision) and
`--served-model-name cpm-text-duplex`.

## FullDuplexBench results (base vs RL)

Evaluated with [Full-Duplex-Bench](https://github.com/DanielLin94144/Full-Duplex-Bench)
(GPT-4o behavior classification). The RL model is trained toward **restraint**: it
interrupts the user less and resumes after overlaps more, at the cost of slower/less
frequent turn-taking on pause-handling tasks.

### v1.5 — behavior distribution + stop/response latency (pooled, seconds)

| Task (desired) | Model | n | RESPOND | RESUME | Stop (s) | Resp (s) |
|---|---|---|---|---|---|---|
| user_interruption (RESPOND ↑) | base | 200 | **0.65** | 0.20 | 2.17 | 1.93 |
| | RL | 175 | 0.49 | 0.36 | 2.21 | 2.60 |
| user_backchannel (RESUME ↑) | base | 98 | 0.00 | 0.52 | 0.73 | 1.93 |
| | RL | 98 | 0.00 | **0.63** | 0.67 | 2.03 |
| talking_to_other (RESUME ↑) | base | 100 | 0.47 | 0.24 | 1.43 | 1.90 |
| | RL | 100 | 0.28 | **0.43** | 1.53 | 2.18 |
| background_speech (RESUME ↑) | base | 100 | 0.63 | 0.25 | 1.21 | 2.27 |
| | RL | 98 | 0.45 | **0.31** | 1.19 | 2.32 |

The RL model wins the three tasks where the desired behavior is *staying quiet /
resuming* (backchannels, third-party speech, background speech) and trades away some
responsiveness on direct user interruptions.

### v1.0 — turn-taking dimensions

| Metric | base | RL (run9) |
|---|---|---|
| Candor Pause Handling · take-turn | 0.916 | 0.635 |
| Candor Turn Taking · take-turn / latency | 0.992 / 0.31s | 0.861 / 0.85s |
| ICC Backchannel · JSD / TOR / Freq | 0.44 / 0.71 / 0.44 | 0.69 / 0.73 / 0.15 |
| Synthetic Pause Handling · take-turn | 0.934 | 0.653 |
| Synthetic User Interruption · rating / take-turn / latency | 4.15 / 1.0 / 0.71s | 4.04 / 0.98 / 1.76s |

Base scores higher on v1.0's take-turn/latency conventions — consistent with the
v1.5 picture: RL training shifted the policy toward restraint (lower take-turn,
higher latency), which v1.0 penalizes and v1.5's RESUME-style tasks reward.

> Eval note: `v1_v1.5/evaluation/eval_user_interruption.py` upstream hard-crashed on
> samples where the model produced no output (an expected behavior); our local copy
> logs `[SKIP]` and continues (run9 skipped 1/176 samples on that task).

---

## Architecture (three layers)

```
  ┌─────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────┐
  │   vLLM backend      │ → │   server.py                  │ → │   client             │
  │   hosts the model   │   │   Kokoro TTS + Parakeet ASR  │   │   run_demo.py page / │
  │   :8555 (HTTP API)  │   │   :8998 (websocket /ws)      │   │   retico (Misty)     │
  └─────────────────────┘   └──────────────────────────────┘   └──────────────────────┘
```

One vLLM + one `server.py` per model. Clients speak a small JSON WebSocket protocol
(`hello` → `ready`, `mic_audio` base64-f32 up, `audio_chunk` + `snapshot` down) —
`duplex_client.py` is the Python reference client; the JS inside `run_demo_ui.html`
is the browser reference.

### Prerequisites

- Linux box with an NVIDIA GPU, CUDA 12.4 (`requirements.txt` pins `+cu124` wheels), Python 3.12.
- `pip install -r requirements.txt` in a venv.
- For the Misty / in-person client: `cd retico && uv sync` (see `retico/README.md`).

---

## Running the survey (IRB24-222 study)

The study compares the two models blinded (A vs B). Full protocol details, data
schema, and participant flow are in **`SURVEY.md`**. Compensation note: the survey
page shows a **$5** gift card online and **$10** in person automatically
(`--gift-amount` overrides).

### Step 1 — two vLLM backends (on the GPU box)

One GPU + one port per model. `--gpu_memory_utilization` is a fraction of the
**whole** GPU — don't stack two backends on one GPU.

```bash
# base model → GPU 0, port 8555
CUDA_VISIBLE_DEVICES=0 vllm serve enochlev/MiniCPM-duplex \
    --served-model-name cpm-text-duplex --max-model-len 3000 \
    --gpu_memory_utilization 0.30 --port 8555 --trust-remote-code

# RL model → GPU 1, port 8557
CUDA_VISIBLE_DEVICES=1 vllm serve enochlev/MiniCPM-duplex-rl \
    --served-model-name cpm-text-duplex --max-model-len 3000 \
    --gpu_memory_utilization 0.30 --port 8557 --trust-remote-code
```

Wait for `Application startup complete` on both **before** starting anything else
(other processes grabbing GPU memory first can OOM the cache allocation).

### Step 2 — two duplex servers (same GPU box)

```bash
# base frontend (ws :8997, public wss via --share)
ASR_MONOTONIC_COMMIT=1 CUDA_VISIBLE_DEVICES=0 python server.py --cpm --share \
    --vllm-port 8555 --port 8997

# RL frontend (ws :8996)
ASR_MONOTONIC_COMMIT=1 CUDA_VISIBLE_DEVICES=1 python server.py --cpm --share \
    --vllm-port 8557 --port 8996
```

Each prints its public WebSocket URL, e.g. `wss://<MODEL_A_TUNNEL>.gradio.live/ws`
and `wss://<MODEL_B_TUNNEL>.gradio.live/ws`. These rotate on every restart — copy
the fresh ones into the commands below. **Keep the mapping consistent: A = base,
B = RL.**

### Step 3, path A — in-person (Misty robot; wizard runs on the local PC)

The survey page must be a **local** link so the browser/PC can reach the Misty
robot on the LAN. Two terminals + a browser on the intern PC:

```bash
# terminal 1 — survey wizard, locally:
python run_demo.py --inperson \
    --model_a_url wss://<MODEL_A_TUNNEL>.gradio.live/ws \
    --model_b_url wss://<MODEL_B_TUNNEL>.gradio.live/ws

# terminal 2 — robot client (set MISTY_IP in retico/.env):
cd retico && uv run inperson.py

# browser — participant at http://localhost:7870
```

`inperson.py` idles until the participant reaches a talk step, then automatically
connects the PC-mic → duplex-server → Misty pipeline to that step's blinded model
and relays the live transcript into the wizard. It disconnects on "I'm done" or the
5-minute timer and waits for the next step. Responses save to the local
`~/scratch/survey_responses/responses.jsonl`; per-session stereo WAVs (L=user,
R=bot) land in `retico/debug_wavs/`.

### Step 3, path B — online (wizard runs on the GPU box)

```bash
python run_demo.py --share \
    --model_a_url wss://<MODEL_A_TUNNEL>.gradio.live/ws \
    --model_b_url wss://<MODEL_B_TUNNEL>.gradio.live/ws
```

Send participants the printed `https://<SURVEY_TUNNEL>.gradio.live` link. Audio runs
through the participant's browser mic/speakers; everything else is identical to the
in-person flow.

### Results

```bash
python survey_to_csv.py responses.jsonl [more.jsonl ...] > sessions.csv
```

One row per session; join `pin_q1`/`pin_q2` against the Google Form export's
"Participant ID" column (`system1_model`/`system2_model` unblind the order).

---

## Module overview

**`full_duplex.py`** — single module, exports:
- `DuplexAudioAgent` — the main agent class
- `DuplexAudioBlock` — finalized conversation block dataclass
- `llm_generate` — thin OpenAI wrapper

**`full-duplex.jinja2`** — system prompt template rendered for each LLM call.
Describes the `<user>/<idle>/<AI></s>` token format the agent uses.

**`trainer.py` / `trainer/`** — the RL training loop (rewards, REINFORCE trainer).
**`replay_client.py`** — replay recorded sessions against a server for offline eval.
**`inference_FDB.py`** — Full-Duplex-Bench inference driver (WS client).

## Key constants

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_WPM` | 150 | Words-per-minute used to calculate N words per block |
| `DEFAULT_BLOCK_S` | 2.0 s (training) / 1.7 s (serving `--cpm`) | Block duration |
| `TTS_SAMPLE_RATE` | 24000 Hz | Kokoro PCM output rate |
| `MIC_SAMPLE_RATE` | 16000 Hz | Parakeet input rate |
| `MAX_MIC_BLOCKS` | 10 | Rolling mic audio window (last N blocks passed to ASR) |
| `MAX_HISTORY_S` | 600 s | Blocks older than this are pruned |

## How blocks work

### historical_blocks
Each finalized block covers one scheduling interval:
- `start_ts` / `end_ts` — wall-clock boundaries
- `assistant_text` — words committed during this block (N = `ceil(WPM * block_s / 60)`)
- `user_text` — text from ASR aligned to this block's time range
- Block duration = actual TTS audio length, or `DEFAULT_BLOCK_S` when the assistant is silent

### purposed (pending) block
Words queued for the next block, not yet committed to audio:
- `_pending_words` — raw LLM output split into words
- On each `poll()`, the first N words are popped and passed to TTS
- If the assistant is silent, `_next_block_ts` advances by `DEFAULT_BLOCK_S`

## How ASR and TTS work

**TTS (Kokoro):** the whole pending response is synthesized once and sliced per
block at silence troughs, so consecutive blocks play back seamlessly.

**ASR (Parakeet):** on every block boundary the rolling mic window is re-transcribed
in a background thread and word timestamps are mapped back to blocks. With
`ASR_MONOTONIC_COMMIT=1` (used in serving) blocks older than the 2 s right-context
window are frozen and never rewritten.

## Running tests

```bash
# Unit tests — no API key, GPU, or real audio needed
pytest tests/test_full_duplex.py -v

# Integration tests — requires NeMo + Parakeet checkpoint
pytest tests/test_integration_full_duplex.py -v
```

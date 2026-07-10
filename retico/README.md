# Retico Duplex Client

Client-side retico pipeline that streams your mic to the full-duplex server on
the LAN and plays the bot's replies through a Misty robot:

```
mic (this Mac) -> resample 16k -> [hush filter, optional] -> duplex server (192.168.0.179:8998) -> Misty (192.168.0.156)
```

This folder is a self-contained [uv](https://docs.astral.sh/uv/) project — it
does **not** use the training repo's root environment (no torch needed).

## 1. Server side (the vLLM box, `192.168.0.179`)

Both must be running there (they already are if `curl http://192.168.0.179:8998/healthz` says ok):

```bash
# vLLM model backend on :8555
CUDA_VISIBLE_DEVICES=0 vllm serve xinrongzhang2022/MiniCPM-duplex \
    --served-model-name cpm-text-duplex \
    --max-model-len 3000 --port 8555 --trust-remote-code

# duplex websocket server on :8998 (must bind 0.0.0.0 to be reachable from the LAN)
python server.py --host 0.0.0.0 --port 8998 --vllm-port 8555 --cpm
```

Drop `--cpm` if vLLM is serving the locally trained checkpoint instead of
MiniCPM-duplex.

## 2. Client side (this Mac)

```bash
brew install portaudio        # once — pyaudio needs it to build
cd retico
uv sync                       # once — creates .venv from pyproject.toml
uv run test.py
```

Speak, watch the `[tap:*]` drift/RMS lines, press Enter to stop. Each tap also
records to `debug_wavs/` and the full conversation is saved to
`remote_duplex_conversation.wav` (left = you, right = bot).

### Configuration (env vars or `.env` file)

| Var | Default | Meaning |
|---|---|---|
| `DUPLEX_SERVER_URL` | `ws://192.168.0.179:8998` | duplex server (a bare host, `http(s)://`, or `ws(s)://` URL all work) |
| `MISTY_IP` | `192.168.0.156` | Misty robot REST API |
| `HUSH_CHECKPOINT` | `Hush/deployment/models/model_best.ckpt` | background-speaker filter checkpoint |

### Optional: Hush background-speaker filter

Hush runs through the prebuilt `libweya_nc` shared library (ONNX Runtime
inside, ctypes from Python) — **no torch**, ~50x realtime on an M-series CPU,
10 ms frames. Setup:

```bash
git clone https://github.com/pulp-vision/Hush ../Hush   # provides the shared lib
uv sync --extra hush                                    # huggingface-hub for model download
```

The model bundle is fetched from the HF hub on first use
(`HUSH_CHECKPOINT=weya-ai/hush`, the default) and cached. Set
`HUSH_CHECKPOINT=off` to disable the filter, `HUSH_ATTEN_LIM_DB` (default 100)
to cap suppression depth, `HUSH_INPUT_GAIN` to pre-amplify a quiet mic, and
`HUSH_GATE_RMS` (default 0.003) to hard-zero output frames below that RMS —
hush leaves a faint but ASR-transcribable residue on suppressed audio (like
quiet Misty feedback); the gate turns it into true silence. Set 0 to disable.

**What Hush does and does not fix** (measured on real recordings 2026-07-10):
it suppresses typing and background talkers *while the user is speaking* —
it keeps whoever is the dominant foreground speaker. That means it does
**not** remove Misty self-feedback: when the user is silent, Misty's playback
is the dominant voice at the mic and passes through (~-1 dB). Feedback needs
echo cancellation / playback-window gating, not speaker isolation. It also
gates very quiet far-field speech entirely (its foreground decision is
level-dependent), so keep the mic close / at a healthy level.

## Layout

```
retico/
├── pyproject.toml            # uv project (retico-core, pyaudio, websockets, ...)
├── test.py                   # the runner: mic -> [hush] -> duplex -> misty
├── duplex_client.py          # websocket client (copy of repo root)
├── duplex_protocol.py        # client-safe protocol helpers (trimmed: no full_duplex/torch import)
├── retico_minicpm/remote_duplex.py    # retico module wrapping duplex_client
├── retico_mistyrobot/misty_speaker.py # uploads WAV chunks to Misty and plays them
└── retico_hush/hush.py       # optional Hush filter module (needs Hush repo + torch)
```

Sample-rate rules (why things sound slow/fast or the agent "goes deaf") are
documented in the docstring at the top of `test.py`.

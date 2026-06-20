# text-only-duplex-model

Real-time full-duplex audio conversation agent. The agent speaks and listens simultaneously — words are committed to audio blocks on a rolling schedule while Parakeet ASR transcribes microphone input and corrects historical blocks.

---

## Quickstart (interns) — talk to the model in 3 commands

The stack is **three layers**, started in order:

```
  ┌─────────────────────┐   ┌──────────────────────────────┐   ┌─────────────────────┐
  │   vLLM backend       │ → │   server.py                  │ → │   simple_web.py     │
  │   hosts the model    │   │   Piper TTS + Parakeet ASR   │   │   browser mic page  │
  │   :8555 (HTTP API)   │   │   :8998 (websocket /ws)      │   │   :9000 (your page) │
  └─────────────────────┘   └──────────────────────────────┘   └─────────────────────┘
```

You start **one** vLLM service and connect **one** client to it — no survey
(`survey_demo.py`) and no Gradio (`demo.py`) needed for this path.

### Prerequisites
- Linux box with an **NVIDIA GPU** and **CUDA 12.4** (the `requirements.txt` pins `+cu124` wheels).
- **Python 3.12** (see `.python-version`).
- No API keys are needed for this path — the model, TTS, and ASR all run locally.

### 0. Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
> Different CUDA version? Edit the `--extra-index-url` tag and the `+cu124` suffixes at the top of `requirements.txt`.

### 1. Start the vLLM model backend (terminal 1)
Serves the public MiniCPM-duplex model over an OpenAI-compatible API — **no checkpoint download required**:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve xinrongzhang2022/MiniCPM-duplex \
    --served-model-name cpm-text-duplex \
    --max-model-len 3000 \
    --gpu_memory_utilization 0.30 \
    --port 8555 \
    --trust-remote-code
```
> **Using the locally trained checkpoint instead?** Point `vllm serve` at it but keep the same served name:
> `vllm serve ./checkpoints/final --served-model-name cpm-text-duplex --max-model-len 3000 --gpu_memory_utilization 0.30 --port 8555 --trust-remote-code`

### 2. Start the duplex audio server (terminal 2)
Loads Piper TTS + Parakeet ASR and exposes the websocket. `--cpm` selects the MiniCPM backend; the default ports already match step 1 (`--vllm-port 8555`, `--port 8998`):
```bash
CUDA_VISIBLE_DEVICES=0 python server.py --cpm
```
Wait for `models ready, starting websocket server on ws://127.0.0.1:8998/ws`.

### 3. Start the browser mic client (terminal 3)
```bash
python simple_web.py
```
Open **http://localhost:9000**, press **Start talking**, and speak. The bot replies
over your speakers in real time and the live transcript renders on the page.

### Hooking it up to your own thing
`simple_web.py` is intentionally minimal — one Python file (standard library only)
that serves one HTML page. The JavaScript in that page **is the full integration
contract**: open a websocket to `ws://<host>:8998/ws`, send a `hello`, stream
`mic_audio` frames up (base64 float32 PCM), and play the `audio_chunk` frames that
come back. Copy that JS into any web app, or speak the same JSON protocol from any
language, to give it a live full-duplex voice agent. (`duplex_client.py` is the
equivalent reference for a pure-Python client.)

---

## Module overview

**`full_duplex.py`** — single module, exports:
- `DuplexAudioAgent` — the main agent class
- `DuplexAudioBlock` — finalized conversation block dataclass
- `llm_generate` — thin OpenAI wrapper

**`full-duplex.jinja2`** — system prompt template rendered for each LLM call. Describes the `<user>/<idle>/<AI></s>` token format the agent uses.

## Key constants

| Constant | Default | Description |
|---|---|---|
| `DEFAULT_WPM` | 150 | Words-per-minute used to calculate N words per block |
| `DEFAULT_BLOCK_S` | 2.0 s | Silence block duration when the assistant says nothing |
| `TTS_SAMPLE_RATE` | 22050 Hz | Piper PCM output rate |
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
- If the final character is punctuation and the audio is shorter than `DEFAULT_BLOCK_S`, silence is padded to the block boundary

Warnings are logged when a block's TTS audio is > 4 s or < 1 s.

### purposed (pending) block
Words queued for the next block, not yet committed to audio:
- `_pending_words` — raw LLM output split into words
- On each `poll()`, the first N words are popped and passed to TTS
- TTS audio length sets `_next_block_ts` (the next scheduling deadline)
- If the assistant is silent, `_next_block_ts` advances by `DEFAULT_BLOCK_S`
- The user audio chunk for this block is not yet available (not spoken yet)

## How ASR and TTS work

**TTS (Piper):** Each block's words are synthesized to PCM. Block timing is driven by the real audio duration, not a fixed clock.

**ASR (Parakeet, lazy-loaded on first use):** On every block boundary, `_seal_mic_block` saves the current mic chunk to `_mic_rolling` (capped at `MAX_MIC_BLOCKS`). The rolling buffer is passed to Parakeet in a background thread. Word timestamps are mapped back to historical blocks by matching `abs(block.start_ts - word_abs_end) < 0.5 s`.

**Mutable window:** The last `mutable_asr_windows` (default 5) blocks can be corrected by a new ASR pass. Older blocks are frozen and their text is locked.

## Running tests

```bash
# Unit tests — no API key, GPU, or real audio needed
pytest tests/test_full_duplex.py -v

# Integration tests — requires NeMo + Parakeet checkpoint
pytest tests/test_integration_full_duplex.py -v

# Real-stack tests — requires OPENAI_API_KEY + piper-tts + NeMo
pytest tests/test_integration_real.py -v
```

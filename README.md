# text-only-duplex-model

Real-time full-duplex audio conversation agent. The agent speaks and listens simultaneously — words are committed to audio blocks on a rolling schedule while Parakeet ASR transcribes microphone input and corrects historical blocks.

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

"""
survey_demo.py - Blind A/B survey UI for comparing two duplex voice bots.

Default wiring
--------------
Model A -> the local trained-model websocket server on 127.0.0.1:$WS_DUPLEX_SERVER_PORT
Model B -> the MiniCPM websocket server on 127.0.0.1:$WS_PCM_DUPLEX_SERVER_PORT

Example run
-----------
    python server.py --port "$WS_DUPLEX_SERVER_PORT"
    python server.py --is-cpm --port "$WS_PCM_DUPLEX_SERVER_PORT"
    python survey_demo.py

Override the defaults with SURVEY_MODEL_A_NAME / SURVEY_MODEL_A_URL and
SURVEY_MODEL_B_NAME / SURVEY_MODEL_B_URL if your deployment uses different
names or ports.

The survey tab keeps the model identities hidden until the final reveal
screen; the free-play tab shows the real model names for demos and exploration.
"""

from __future__ import annotations

import base64 as _base64
from datetime import datetime, timezone
import html as _html
import io as _io
import json
import os
from pathlib import Path
import secrets
import time
import uuid
import wave as _wave
from typing import Optional

import gradio as gr
import numpy as np

from duplex_client import FullDuplexClient
from duplex_protocol import SessionSnapshot, server_url_from_address
from full_duplex import ASR_SAMPLE_RATE, WS_DUPLEX_SERVER_PORT, WS_PCM_DUPLEX_SERVER_PORT

POLL_INTERVAL_S = 0.08
RESULTS_DIR = Path(os.getenv("SURVEY_RESULTS_DIR", "survey_results"))

AGE_OPTIONS = ["18-25", "26-35", "36-45", "46-55", "56+", "prefer not to say"]
YES_NO_OPTIONS = ["yes", "no"]
VOICE_ASSISTANT_USE_OPTIONS = ["never", "rarely", "sometimes", "often", "daily"]
LIKERT_CHOICES = [str(value) for value in range(1, 8)]

LIKERT_ITEMS = [
    {
        "key": "prompt_after_finish",
        "text": "The bot responded promptly when I finished speaking.",
    },
    {
        "key": "interrupted_me",
        "text": "The bot interrupted me while I was still talking.",
    },
    {
        "key": "natural_silences",
        "text": "The bot's silences felt natural, not awkward.",
    },
    {
        "key": "waited_too_long",
        "text": "The bot waited too long before responding.",
    },
    {
        "key": "relevant_responses",
        "text": "The bot's responses were relevant to what I asked.",
    },
    {
        "key": "natural_overall",
        "text": "The conversation felt natural overall.",
    },
    {
        "key": "talk_again",
        "text": "I would want to talk to this bot again.",
    },
]

COMPARISON_OPTIONS = {
    "natural": ["Bot 1", "Bot 2", "No preference"],
    "interruptions": ["Bot 1", "Bot 2", "Equal"],
    "promptness": ["Bot 1", "Bot 2", "Equal"],
    "overall": ["Bot 1", "Bot 2", "No preference"],
}

SESSION_SPECS = [
    {
        "key": "short_questions",
        "title": "Session 1 - Short Questions",
        "subtitle": "Ask quick factual or math questions with short answers.",
        "instruction": (
            "Feel free to use these prompts or ask your own questions in a similar style. "
            "Click the microphone button to start speaking."
        ),
        "prompts": [
            "What's 12 times 7?",
            "What's the capital of France?",
            "How many days are in February?",
            "What's the square root of 144?",
            "What year did World War II end?",
        ],
    },
    {
        "key": "long_explanations",
        "title": "Session 2 - Long Explanations",
        "subtitle": "Ask for detailed explanations, and make your question longer than 10 words.",
        "instruction": (
            "Feel free to use these prompts or ask your own questions in a similar style. "
            "Click the microphone button to start speaking. Please make your question longer than 10 words."
        ),
        "prompts": [
            "Can you explain how photosynthesis works in plants?",
            "Walk me through how the immune system fights off a virus.",
            "Explain the basic idea behind general relativity.",
            "How does a transistor actually work?",
            "What's going on when bread dough rises?",
        ],
    },
    {
        "key": "interruption_practice",
        "title": "Session 3 - Interruption Practice",
        "subtitle": "Ask the question, then interrupt the bot mid-response with a follow-up clarification.",
        "instruction": (
            "Feel free to use these prompts or ask your own questions in a similar style. "
            "Click the microphone button to start speaking. Ask the question, let the bot begin answering, "
            "then interrupt it mid-response with a follow-up clarification."
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

CUSTOM_CSS = """
:root {
  --bg-cream: #f4ede1;
  --bg-warm: #fffaf2;
  --ink: #201915;
  --muted: #6b6158;
  --line: rgba(32, 25, 21, 0.12);
  --amber: #bf6c1e;
  --teal: #23685a;
  --rose: #9f4737;
  --slate: #314552;
}

.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(191, 108, 30, 0.12), transparent 28%),
    radial-gradient(circle at top right, rgba(35, 104, 90, 0.12), transparent 26%),
    linear-gradient(180deg, #f8f1e7 0%, #efe3d2 100%);
  color: var(--ink);
  font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4 {
  font-family: "Fraunces", "Georgia", serif;
  letter-spacing: -0.02em;
}

.survey-shell,
.free-shell,
.transcript-shell,
.status-card,
.survey-card,
.prompt-card,
.likert-card,
.comparison-card,
.thankyou-card {
  background: rgba(255, 250, 242, 0.88);
  border: 1px solid var(--line);
  border-radius: 20px;
  box-shadow: 0 18px 40px rgba(52, 35, 18, 0.08);
  backdrop-filter: blur(6px);
}

.survey-shell,
.free-shell {
  padding: 18px 20px;
}

.survey-card,
.prompt-card,
.likert-card,
.comparison-card,
.thankyou-card,
.status-card {
  padding: 18px 20px;
}

.eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.14em;
  font-size: 11px;
  color: var(--amber);
  font-weight: 700;
}

.lead,
.muted,
.supporting {
  color: var(--muted);
}

.progress-line {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 8px;
  color: var(--muted);
  font-size: 13px;
}

.prompt-list {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.prompt-chip {
  padding: 12px 14px;
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(191, 108, 30, 0.09), rgba(35, 104, 90, 0.06));
  border: 1px solid rgba(191, 108, 30, 0.16);
  font-size: 14px;
  line-height: 1.45;
}

.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 14px;
}

.status-pill::before {
  content: "";
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
}

.status-pill.ready {
  background: rgba(35, 104, 90, 0.10);
  color: var(--teal);
}

.status-pill.ready::before {
  background: var(--teal);
}

.status-pill.recording {
  background: rgba(159, 71, 55, 0.12);
  color: var(--rose);
}

.status-pill.recording::before {
  background: var(--rose);
  box-shadow: 0 0 0 6px rgba(159, 71, 55, 0.16);
}

.status-pill.responding {
  background: rgba(49, 69, 82, 0.10);
  color: var(--slate);
}

.status-pill.responding::before {
  background: var(--slate);
}

.interrupt-callout,
.note-callout {
  margin-top: 14px;
  padding: 14px 16px;
  border-radius: 16px;
  font-weight: 700;
  line-height: 1.5;
}

.interrupt-callout {
  background: linear-gradient(135deg, rgba(159, 71, 55, 0.14), rgba(191, 108, 30, 0.10));
  border: 1px solid rgba(159, 71, 55, 0.22);
  color: #6e2d22;
}

.note-callout {
  background: rgba(35, 104, 90, 0.09);
  border: 1px solid rgba(35, 104, 90, 0.18);
  color: var(--teal);
}

.transcript-shell {
  min-height: 320px;
  max-height: 520px;
  overflow-y: auto;
  padding: 16px;
}

.turn {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 14px;
}

.turn-label {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.turn.user .turn-label {
  color: var(--amber);
}

.turn.assistant .turn-label {
  color: var(--teal);
}

.turn-bubble {
  padding: 14px 16px;
  border-radius: 18px;
  border: 1px solid var(--line);
  line-height: 1.55;
  font-size: 14px;
}

.turn.user .turn-bubble {
  background: rgba(191, 108, 30, 0.08);
}

.turn.assistant .turn-bubble {
  background: rgba(35, 104, 90, 0.08);
}

.placeholder {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.6;
}

.scale-legend {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
  margin-top: 14px;
  font-size: 13px;
  color: var(--muted);
}

.reveal-list {
  margin: 14px 0 0;
  padding-left: 18px;
  line-height: 1.7;
}

.free-note {
  margin-top: 10px;
  color: var(--muted);
  font-size: 13px;
}

@media (max-width: 820px) {
  .survey-shell,
  .free-shell,
  .survey-card,
  .prompt-card,
  .likert-card,
  .comparison-card,
  .thankyou-card,
  .status-card {
    padding: 16px;
    border-radius: 18px;
  }

  .transcript-shell {
    min-height: 260px;
  }
}
"""

INIT_JS = """
window._fdPlayers = window._fdPlayers || {};

window._audioEnqueue = function(channel, dataUri) {
    if (!channel || !dataUri) return;

    var state = window._fdPlayers[channel];
    if (!state || !state.ctx || state.ctx.state === 'closed') {
        state = {
            ctx: new (window.AudioContext || window.webkitAudioContext)(),
            nextTime: 0,
        };
        window._fdPlayers[channel] = state;
    }

    if (dataUri === '__reset__') {
        state.nextTime = 0;
        return;
    }

    var ctx = state.ctx;
    var ready = ctx.state === 'suspended' ? ctx.resume() : Promise.resolve();
    ready
        .then(function() { return fetch(dataUri); })
        .then(function(r) { return r.arrayBuffer(); })
        .then(function(buf) { return ctx.decodeAudioData(buf); })
        .then(function(decoded) {
            var src = ctx.createBufferSource();
            src.buffer = decoded;
            src.connect(ctx.destination);
            var when = Math.max(ctx.currentTime + 0.05, state.nextTime || 0);
            src.start(when);
            state.nextTime = when + decoded.duration;
        })
        .catch(function(error) { console.error('[audio]', channel, error); });
};
"""


def _load_model_configs() -> list[dict]:
    default_a = os.getenv(
        "SURVEY_MODEL_A_URL",
        os.getenv("FULL_DUPLEX_SERVER_URL", f"127.0.0.1:{WS_DUPLEX_SERVER_PORT}"),
    )
    default_b = os.getenv("SURVEY_MODEL_B_URL", f"127.0.0.1:{WS_PCM_DUPLEX_SERVER_PORT}")
    return [
        {
            "model_key": "A",
            "name": os.getenv("SURVEY_MODEL_A_NAME", "Local duplex model"),
            "server_url": server_url_from_address(default_a),
        },
        {
            "model_key": "B",
            "name": os.getenv("SURVEY_MODEL_B_NAME", "MiniCPM duplex"),
            "server_url": server_url_from_address(default_b),
        },
    ]


MODEL_CONFIGS = _load_model_configs()
MODEL_BY_KEY = {config["model_key"]: config for config in MODEL_CONFIGS}
FREE_PLAY_CHOICES = [(config["name"], config["model_key"]) for config in MODEL_CONFIGS]


def _audio_to_data_uri(audio: np.ndarray, sample_rate: int) -> str:
    arr = np.asarray(audio)
    if np.issubdtype(arr.dtype, np.floating):
        pcm = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)
    else:
        pcm = arr.astype(np.int16)

    buffer = _io.BytesIO()
    with _wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())
    encoded = _base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{encoded}"


def _warning_title(source: str) -> str:
    titles = {
        "llm": "LLM Warning",
        "poll": "Agent Warning",
        "client": "Client Warning",
        "survey": "Survey Warning",
    }
    return titles.get(source, "Warning")


def _push_warning(state: dict, source: str, message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False

    warning_key = (source, text)
    if state.get("last_warning_key") == warning_key:
        return False

    state["last_warning_key"] = warning_key
    return True


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _participant_id() -> str:
    return f"p-{uuid.uuid4().hex[:10]}"


def _empty_audio_segments() -> dict:
    return {
        "mic_segments": [],
        "assistant_segments": [],
    }


def _append_audio_segment(segments: list[dict], sample_rate: int, audio: np.ndarray) -> None:
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return

    resolved_rate = int(sample_rate)
    if segments and segments[-1]["sample_rate"] == resolved_rate:
        segments[-1]["chunks"].append(arr)
        return

    segments.append({
        "sample_rate": resolved_rate,
        "chunks": [arr],
    })


def _serialize_audio_segments(segments: list[dict]) -> list[dict]:
    serialized = []
    for segment in segments:
        chunks = segment.get("chunks", [])
        if not chunks:
            continue
        audio = np.concatenate(chunks)
        sample_rate = int(segment["sample_rate"])
        serialized.append(
            {
                "sample_rate": sample_rate,
                "duration_s": round(len(audio) / sample_rate, 4),
                "wav_data_uri": _audio_to_data_uri(audio, sample_rate),
            }
        )
    return serialized


def _normalize_input_audio(audio_array: np.ndarray) -> np.ndarray:
    arr = np.asarray(audio_array)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        return arr.astype(np.float32) / scale
    return arr.astype(np.float32)


def _snapshot_blocks(snapshot: Optional[SessionSnapshot]) -> list:
    if snapshot is None:
        return []
    blocks = list(snapshot.blocks)
    if snapshot.current_block is not None:
        blocks.append(snapshot.current_block)
    return blocks


def _turns_from_snapshot(snapshot: Optional[SessionSnapshot]) -> list[dict]:
    turns: list[dict] = []
    for block in _snapshot_blocks(snapshot):
        if block.user_text:
            text = block.user_text.strip()
            if text:
                if turns and turns[-1]["speaker"] == "user":
                    turns[-1]["text"] = f"{turns[-1]['text']} {text}".strip()
                else:
                    turns.append({"speaker": "user", "text": text})
        if block.assistant_text:
            text = block.assistant_text.strip()
            if text:
                if turns and turns[-1]["speaker"] == "assistant":
                    turns[-1]["text"] = f"{turns[-1]['text']} {text}".strip()
                else:
                    turns.append({"speaker": "assistant", "text": text})
    return turns


def _current_bot(state: dict) -> Optional[dict]:
    sequence = state.get("bot_sequence", [])
    index = state.get("current_bot_index", 0)
    if 0 <= index < len(sequence):
        return sequence[index]
    return None


def _current_session_spec(state: dict) -> Optional[dict]:
    index = state.get("current_session_index", 0)
    if 0 <= index < len(SESSION_SPECS):
        return SESSION_SPECS[index]
    return None


def _current_likert_item(state: dict) -> Optional[dict]:
    index = state.get("likert_index", 0)
    if 0 <= index < len(LIKERT_ITEMS):
        return LIKERT_ITEMS[index]
    return None


def _default_survey_state() -> dict:
    return {
        "phase": "intro",
        "participant_id": None,
        "result_path": None,
        "started_at": None,
        "completed_at": None,
        "backend_ready": False,
        "backend_message": "Checking the hidden survey backends…",
        "assignment_order": None,
        "bot_sequence": [],
        "bots": {},
        "current_bot_index": 0,
        "current_session_index": 0,
        "likert_index": 0,
        "demographics": {},
        "comparison": {},
        "client": None,
        "snapshot": None,
        "last_warning_key": None,
        "last_mic_at": 0.0,
        "last_bot_audio_at": 0.0,
        "active_audio": _empty_audio_segments(),
        "active_session": None,
    }


def _default_free_play_state() -> dict:
    return {
        "client": None,
        "snapshot": None,
        "current_model_key": MODEL_CONFIGS[0]["model_key"],
        "last_warning_key": None,
        "last_mic_at": 0.0,
        "last_bot_audio_at": 0.0,
    }


def _ensure_survey_state(state: Optional[dict]) -> dict:
    return state if state else _default_survey_state()


def _ensure_free_play_state(state: Optional[dict]) -> dict:
    return state if state else _default_free_play_state()


def _close_client(client: Optional[FullDuplexClient]) -> None:
    if client is None:
        return
    with np.errstate(all="ignore"):
        client.close()


def _reset_survey_runtime(state: dict) -> None:
    _close_client(state.get("client"))
    state["client"] = None
    state["snapshot"] = None
    state["last_warning_key"] = None
    state["last_mic_at"] = 0.0
    state["last_bot_audio_at"] = 0.0
    state["active_audio"] = _empty_audio_segments()
    state["active_session"] = None


def _survey_result_path(participant_id: str, started_at: str) -> str:
    timestamp = started_at.replace(":", "").replace("-", "")
    timestamp = timestamp.replace("+0000", "Z").replace("+00:00", "Z")
    timestamp = timestamp.replace(".", "_")
    return str(RESULTS_DIR / f"survey_{timestamp}_{participant_id}.json")


def _build_results_payload(state: dict, *, final: bool) -> dict:
    bots_payload = {}
    for entry in state.get("bot_sequence", []):
        bot_label = entry["bot_label"]
        bot_state = state.get("bots", {}).get(bot_label, {})
        bots_payload[bot_label] = {
            "model_key": bot_state.get("model_key"),
            "model_name": bot_state.get("model_name"),
            "server_url": bot_state.get("server_url"),
            "sessions": bot_state.get("sessions", []),
            "likert": bot_state.get("likert", {}),
        }

    return {
        "participant_id": state.get("participant_id"),
        "started_at": state.get("started_at"),
        "completed_at": state.get("completed_at"),
        "is_complete": final,
        "assignment_order": state.get("assignment_order"),
        "bot_mapping": {
            entry["bot_label"]: {
                "model_key": entry["model_key"],
                "model_name": entry["model_name"],
                "server_url": entry["server_url"],
            }
            for entry in state.get("bot_sequence", [])
        },
        "demographics": state.get("demographics", {}),
        "bots": bots_payload,
        "comparison": state.get("comparison", {}),
    }


def _save_progress(state: dict, *, final: bool = False) -> None:
    result_path = state.get("result_path")
    if not result_path:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = _build_results_payload(state, final=final)
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _probe_hidden_backends() -> tuple[bool, str]:
    try:
        for config in MODEL_CONFIGS:
            client = FullDuplexClient(config["server_url"], open_timeout=2.0)
            try:
                client.connect(client_name="survey-preflight")
            finally:
                client.close()
    except Exception:
        return (
            False,
            "The hidden survey backends are not reachable yet. Start both websocket servers, then reload or try again.",
        )
    return True, "Two hidden survey backends are ready. The survey can begin."


def _conversation_status(state: dict) -> tuple[str, str]:
    client = state.get("client")
    if client is None or not client.connected:
        return "ready", "Preparing the session. The microphone will activate when the bot connects."

    now = time.time()
    if now - state.get("last_mic_at", 0.0) < 0.8:
        return "recording", "Recording now. Keep speaking in a normal voice."
    if now - state.get("last_bot_audio_at", 0.0) < 1.6:
        spec = _current_session_spec(state)
        if spec and spec["key"] == "interruption_practice":
            return "responding", "Bot responding now. Interrupt with a follow-up clarification during this section."
        return "responding", "Bot responding now. You can interrupt if you need to refine your question."
    return "ready", "Ready for you. Click the microphone button and start speaking."


def _free_play_status(state: dict) -> tuple[str, str]:
    client = state.get("client")
    if client is None or not client.connected:
        return "ready", "Pick a model, connect, and start talking. Nothing from this tab is saved."

    now = time.time()
    if now - state.get("last_mic_at", 0.0) < 0.8:
        return "recording", "Recording now. Keep speaking."
    if now - state.get("last_bot_audio_at", 0.0) < 1.6:
        return "responding", "Bot responding now."
    return "ready", "Connected and ready for another turn."


def _render_intro_status(state: dict) -> str:
    message = _html.escape(state.get("backend_message", ""))
    readiness = "ready" if state.get("backend_ready") else "responding"
    return (
        "<div class='status-card'>"
        "<div class='eyebrow'>Survey Setup</div>"
        f"<div class='status-pill {readiness}'>{message}</div>"
        "<div class='free-note'>This tab hides the two model identities until the final thank-you screen.</div>"
        "</div>"
    )


def _render_presurvey_header(state: dict) -> str:
    participant_id = _html.escape(state.get("participant_id") or "pending")
    return (
        "<div class='survey-card'>"
        "<div class='eyebrow'>Pre-Survey</div>"
        "<h2>Tell us a little about the participant</h2>"
        "<p class='lead'>This study compares two hidden duplex bots across short questions, long explanations, and interruption handling.</p>"
        f"<div class='free-note'>Anonymous participant ID: {participant_id}</div>"
        "</div>"
    )


def _render_conversation_header(state: dict) -> str:
    bot = _current_bot(state)
    spec = _current_session_spec(state)
    if bot is None or spec is None:
        return (
            "<div class='survey-card'><div class='eyebrow'>Survey Session</div>"
            "<h2>Waiting for the survey session to start</h2></div>"
        )

    conversation_step = state.get("current_bot_index", 0) * len(SESSION_SPECS) + state.get("current_session_index", 0) + 1
    total_steps = max(1, len(state.get("bot_sequence", [])) * len(SESSION_SPECS))
    return (
        "<div class='survey-card'>"
        f"<div class='eyebrow'>{_html.escape(bot['bot_label'])}</div>"
        f"<h2>{_html.escape(spec['title'])}</h2>"
        f"<p class='lead'>{_html.escape(spec['subtitle'])}</p>"
        "<div class='progress-line'>"
        f"<span>Conversation step {conversation_step} of {total_steps}</span>"
        f"<span>Session {state.get('current_session_index', 0) + 1} of {len(SESSION_SPECS)} for {_html.escape(bot['bot_label'])}</span>"
        "</div>"
        "</div>"
    )


def _render_prompt_panel(state: dict) -> str:
    spec = _current_session_spec(state)
    if spec is None:
        return (
            "<div class='prompt-card'><p class='placeholder'>Prompt suggestions will appear here once the session starts.</p></div>"
        )

    prompt_html = "".join(
        f"<div class='prompt-chip'>{_html.escape(prompt)}</div>"
        for prompt in spec["prompts"]
    )
    extra_html = ""
    if spec["key"] == "long_explanations":
        extra_html = (
            "<div class='note-callout'>"
            "Please make your question longer than 10 words so the bot has to respond to a richer prompt."
            "</div>"
        )
    elif spec["key"] == "interruption_practice":
        extra_html = (
            "<div class='interrupt-callout'>"
            "Important: interrupt the bot on purpose in this section. Ask the question, let it begin answering, then jump in with a follow-up clarification."
            "</div>"
        )

    return (
        "<div class='prompt-card'>"
        "<div class='eyebrow'>Suggested Prompts</div>"
        f"<p class='supporting'>{_html.escape(spec['instruction'])}</p>"
        f"{extra_html}"
        f"<div class='prompt-list'>{prompt_html}</div>"
        "</div>"
    )


def _render_indicator(state: dict, *, survey: bool, speaker_label: str) -> str:
    kind, message = _conversation_status(state) if survey else _free_play_status(state)
    subtitle = (
        f"Live with {_html.escape(speaker_label)}"
        if state.get("client") is not None
        else "Connect to start audio streaming"
    )
    return (
        "<div class='status-card'>"
        "<div class='eyebrow'>Live Audio</div>"
        f"<div class='status-pill {kind}'>{_html.escape(message)}</div>"
        f"<div class='free-note'>{subtitle}</div>"
        "</div>"
    )


def _render_transcript(snapshot: Optional[SessionSnapshot], speaker_label: str) -> str:
    turns = _turns_from_snapshot(snapshot)
    if not turns:
        return (
            "<div class='transcript-shell'>"
            "<p class='placeholder'>The live transcript will appear here as you and the bot talk.</p>"
            "</div>"
        )

    rows = []
    for turn in turns[-24:]:
        label = "You" if turn["speaker"] == "user" else speaker_label
        rows.append(
            "<div class='turn {speaker}'>".format(speaker=turn["speaker"])
            + f"<div class='turn-label'>{_html.escape(label)}</div>"
            + f"<div class='turn-bubble'>{_html.escape(turn['text'])}</div>"
            + "</div>"
        )
    return "<div class='transcript-shell'>" + "".join(rows) + "</div>"


def _render_likert_header(state: dict) -> str:
    bot = _current_bot(state)
    item = _current_likert_item(state)
    if bot is None or item is None:
        return (
            "<div class='likert-card'><p class='placeholder'>The rating prompt will appear here after the conversation sessions.</p></div>"
        )

    item_number = state.get("likert_index", 0) + 1
    return (
        "<div class='likert-card'>"
        f"<div class='eyebrow'>{_html.escape(bot['bot_label'])} rating {item_number} of {len(LIKERT_ITEMS)}</div>"
        f"<h2>{_html.escape(item['text'])}</h2>"
        "<p class='lead'>Rate this statement from 1 (strongly disagree) to 7 (strongly agree).</p>"
        "<div class='scale-legend'><span>1 = strongly disagree</span><span>7 = strongly agree</span></div>"
        "</div>"
    )


def _render_comparison_header(state: dict) -> str:
    return (
        "<div class='comparison-card'>"
        "<div class='eyebrow'>Final Comparison</div>"
        "<h2>Compare Bot 1 and Bot 2</h2>"
        "<p class='lead'>Answer the forced-choice questions below, then add any brief comment that helps explain your preference.</p>"
        f"<div class='free-note'>Participant ID: {_html.escape(state.get('participant_id') or 'pending')}</div>"
        "</div>"
    )


def _render_thankyou(state: dict) -> str:
    if state.get("phase") != "thankyou":
        return (
            "<div class='thankyou-card'><p class='placeholder'>The model identities will be revealed here after submission.</p></div>"
        )

    reveals = []
    for entry in state.get("bot_sequence", []):
        reveals.append(
            f"<li>{_html.escape(entry['bot_label'])} was {_html.escape(entry['model_name'])}.</li>"
        )
    return (
        "<div class='thankyou-card'>"
        "<div class='eyebrow'>Thank You</div>"
        "<h2>The survey is complete</h2>"
        "<p class='lead'>Your responses have been saved under an anonymous participant ID.</p>"
        f"<div class='free-note'>Participant ID: {_html.escape(state.get('participant_id') or 'pending')}</div>"
        f"<ul class='reveal-list'>{''.join(reveals)}</ul>"
        "</div>"
    )


def _render_free_play_status_card(state: dict) -> str:
    model_name = MODEL_BY_KEY.get(state.get("current_model_key"), MODEL_CONFIGS[0])["name"]
    return (
        "<div class='free-shell'>"
        f"<div class='eyebrow'>{_html.escape(model_name)}</div>"
        f"{_render_indicator(state, survey=False, speaker_label=model_name)}"
        "<div class='free-note'>Nothing from this tab is written to the survey results.</div>"
        "</div>"
    )


def _current_likert_value(state: dict) -> Optional[str]:
    bot = _current_bot(state)
    item = _current_likert_item(state)
    if bot is None or item is None:
        return None
    bot_state = state.get("bots", {}).get(bot["bot_label"], {})
    saved = bot_state.get("likert", {}).get(item["key"])
    return None if saved is None else str(saved)


def _session_has_content(state: dict) -> bool:
    turns = _turns_from_snapshot(state.get("snapshot"))
    if turns:
        return True
    active_audio = state.get("active_audio", {})
    return bool(active_audio.get("mic_segments"))


def _start_survey_session(state: dict) -> tuple[bool, Optional[str]]:
    bot = _current_bot(state)
    spec = _current_session_spec(state)
    if bot is None or spec is None:
        return False, "Survey session metadata is incomplete."

    _reset_survey_runtime(state)

    client = FullDuplexClient(bot["server_url"], open_timeout=5.0)
    try:
        client.connect(client_name=f"survey-{bot['bot_label'].replace(' ', '-').lower()}-{spec['key']}")
    except Exception as exc:
        client.close()
        return False, f"{type(exc).__name__}: {exc}"

    state["client"] = client
    state["snapshot"] = client.get_latest_snapshot()
    state["active_session"] = {
        "bot_label": bot["bot_label"],
        "session_key": spec["key"],
        "session_title": spec["title"],
        "started_at": _utc_now_iso(),
        "warnings": [],
    }
    state["phase"] = "conversation"
    return True, None


def _finalize_survey_session(state: dict) -> None:
    active_session = state.get("active_session")
    if active_session is None:
        _reset_survey_runtime(state)
        return

    bot_label = active_session["bot_label"]
    bot_state = state.get("bots", {}).get(bot_label)
    if bot_state is None:
        _reset_survey_runtime(state)
        return

    snapshot = state.get("snapshot")
    session_spec = next(spec for spec in SESSION_SPECS if spec["key"] == active_session["session_key"])
    session_record = {
        "session_key": active_session["session_key"],
        "session_title": active_session["session_title"],
        "started_at": active_session["started_at"],
        "ended_at": _utc_now_iso(),
        "instruction": session_spec["instruction"],
        "prompts": session_spec["prompts"],
        "turns": _turns_from_snapshot(snapshot),
        "snapshot": snapshot.to_dict() if snapshot is not None else None,
        "audio": {
            "microphone": _serialize_audio_segments(state.get("active_audio", {}).get("mic_segments", [])),
            "assistant": _serialize_audio_segments(state.get("active_audio", {}).get("assistant_segments", [])),
        },
        "warnings": active_session.get("warnings", []),
    }
    bot_state.setdefault("sessions", []).append(session_record)
    _reset_survey_runtime(state)


def _store_likert_value(state: dict, value: Optional[str]) -> bool:
    if not value:
        return False
    bot = _current_bot(state)
    item = _current_likert_item(state)
    if bot is None or item is None:
        return False
    bot_state = state.get("bots", {}).get(bot["bot_label"])
    if bot_state is None:
        return False
    bot_state.setdefault("likert", {})[item["key"]] = int(value)
    return True


def _initialize_survey_assignment(state: dict) -> None:
    assignment_order = "A-first" if secrets.randbelow(2) == 0 else "B-first"
    model_order = ["A", "B"] if assignment_order == "A-first" else ["B", "A"]
    bot_sequence = []
    bots = {}
    for index, model_key in enumerate(model_order, start=1):
        config = MODEL_BY_KEY[model_key]
        bot_label = f"Bot {index}"
        entry = {
            "bot_label": bot_label,
            "model_key": model_key,
            "model_name": config["name"],
            "server_url": config["server_url"],
        }
        bot_sequence.append(entry)
        bots[bot_label] = {
            **entry,
            "sessions": [],
            "likert": {},
        }
    state["assignment_order"] = assignment_order
    state["bot_sequence"] = bot_sequence
    state["bots"] = bots
    state["current_bot_index"] = 0
    state["current_session_index"] = 0
    state["likert_index"] = 0


def _survey_render(state: dict, *, transport_value=gr.skip()):
    phase = state.get("phase")
    intro_visible = phase == "intro"
    presurvey_visible = phase == "presurvey"
    conversation_visible = phase == "conversation"
    likert_visible = phase == "likert"
    comparison_visible = phase == "comparison"
    thankyou_visible = phase == "thankyou"

    current_bot = _current_bot(state)
    speaker_label = current_bot["bot_label"] if current_bot is not None else "Bot"

    comparison = state.get("comparison", {})
    demographics = state.get("demographics", {})

    return (
        state,
        gr.update(visible=intro_visible),
        gr.update(visible=presurvey_visible),
        gr.update(visible=conversation_visible),
        gr.update(visible=likert_visible),
        gr.update(visible=comparison_visible),
        gr.update(visible=thankyou_visible),
        _render_intro_status(state),
        _render_presurvey_header(state),
        _render_conversation_header(state),
        _render_prompt_panel(state),
        _render_indicator(state, survey=True, speaker_label=speaker_label),
        _render_transcript(state.get("snapshot"), speaker_label),
        gr.update(interactive=conversation_visible and state.get("client") is not None),
        transport_value,
        _render_likert_header(state),
        gr.update(choices=LIKERT_CHOICES, value=_current_likert_value(state)),
        _render_comparison_header(state),
        _render_thankyou(state),
        gr.update(choices=AGE_OPTIONS, value=demographics.get("age_range")),
        gr.update(choices=YES_NO_OPTIONS, value=demographics.get("native_english")),
        gr.update(choices=VOICE_ASSISTANT_USE_OPTIONS, value=demographics.get("voice_assistant_frequency")),
        gr.update(choices=COMPARISON_OPTIONS["natural"], value=comparison.get("natural")),
        gr.update(choices=COMPARISON_OPTIONS["interruptions"], value=comparison.get("interruptions")),
        gr.update(choices=COMPARISON_OPTIONS["promptness"], value=comparison.get("promptness")),
        gr.update(choices=COMPARISON_OPTIONS["overall"], value=comparison.get("overall")),
        gr.update(value=comparison.get("free_text", "")),
    )


def _free_play_render(state: dict, *, transport_value=gr.skip()):
    model_name = MODEL_BY_KEY.get(state.get("current_model_key"), MODEL_CONFIGS[0])["name"]
    return (
        state,
        gr.update(interactive=state.get("client") is not None),
        _render_free_play_status_card(state),
        _render_transcript(state.get("snapshot"), model_name),
        gr.update(interactive=state.get("client") is None),
        gr.update(interactive=state.get("client") is not None),
        transport_value,
    )


def initialize_app():
    survey_state = _default_survey_state()
    survey_state["backend_ready"], survey_state["backend_message"] = _probe_hidden_backends()
    free_play_state = _default_free_play_state()
    return survey_state, _render_intro_status(survey_state), free_play_state


def start_survey(session_state):
    previous_state = _ensure_survey_state(session_state)
    _close_client(previous_state.get("client"))

    state = _default_survey_state()
    state["backend_ready"], state["backend_message"] = _probe_hidden_backends()
    if not state["backend_ready"]:
        gr.Warning(
            "The hidden survey backends are not ready yet. Please ask the researcher for help.",
            duration=None,
            title="Survey Setup",
        )
        return _survey_render(state, transport_value="__reset__")

    state["participant_id"] = _participant_id()
    state["started_at"] = _utc_now_iso()
    state["result_path"] = _survey_result_path(state["participant_id"], state["started_at"])
    state["phase"] = "presurvey"
    _save_progress(state)
    return _survey_render(state, transport_value="__reset__")


def submit_presurvey(age_range, native_english, voice_assistant_frequency, session_state):
    state = _ensure_survey_state(session_state)
    if not age_range or not native_english or not voice_assistant_frequency:
        gr.Warning(
            "Please answer all three pre-survey questions before continuing.",
            duration=None,
            title="Survey",
        )
        return _survey_render(state)

    state["demographics"] = {
        "age_range": age_range,
        "native_english": native_english,
        "voice_assistant_frequency": voice_assistant_frequency,
    }
    _initialize_survey_assignment(state)
    success, error = _start_survey_session(state)
    if not success:
        state["phase"] = "presurvey"
        if _push_warning(state, "survey", error or "backend unavailable"):
            gr.Warning(
                "The survey backend could not start this session. Please ask the researcher for help.",
                duration=None,
                title="Survey Warning",
            )
        return _survey_render(state, transport_value="__reset__")

    _save_progress(state)
    return _survey_render(state, transport_value="__reset__")


def receive_survey_mic(audio, session_state):
    state = _ensure_survey_state(session_state)
    if audio is None or state.get("client") is None:
        return state

    sample_rate, audio_array = audio
    audio_float = _normalize_input_audio(audio_array)
    state["last_mic_at"] = time.time()
    _append_audio_segment(state["active_audio"]["mic_segments"], sample_rate, audio_float)
    try:
        state["client"].send_audio_chunk(sample_rate, audio_float)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        active_session = state.get("active_session")
        if active_session is not None:
            active_session.setdefault("warnings", []).append({
                "source": "client",
                "message": message,
                "ts": _utc_now_iso(),
            })
        if _push_warning(state, "client", message):
            gr.Warning(
                "The survey session had trouble sending microphone audio. Please ask the researcher if it keeps happening.",
                duration=None,
                title=_warning_title("survey"),
            )
    return state


def poll_survey(session_state):
    state = _ensure_survey_state(session_state)
    client = state.get("client")
    speaker_label = (_current_bot(state) or {}).get("bot_label", "Bot")
    if client is None:
        return state, gr.skip(), _render_indicator(state, survey=True, speaker_label=speaker_label), _render_transcript(state.get("snapshot"), speaker_label)

    active_session = state.get("active_session")
    for warning in client.drain_warnings():
        if active_session is not None:
            active_session.setdefault("warnings", []).append({
                "source": warning.get("source", "other"),
                "message": warning.get("message", ""),
                "ts": _utc_now_iso(),
            })
        message = warning.get("message", "")
        source = warning.get("source", "other")
        if _push_warning(state, source, message):
            gr.Warning(
                "This survey session reported a backend warning. You can continue, but ask the researcher if it persists.",
                duration=None,
                title=_warning_title("survey"),
            )

    latest_snapshot = client.get_latest_snapshot()
    if latest_snapshot is not None:
        state["snapshot"] = latest_snapshot

    audio_chunk = client.pop_audio_chunk(timeout=0.0)
    indicator_html = _render_indicator(state, survey=True, speaker_label=speaker_label)
    transcript_html = _render_transcript(state.get("snapshot"), speaker_label)
    if audio_chunk is None:
        return state, gr.skip(), indicator_html, transcript_html

    sample_rate, audio_array = audio_chunk
    state["last_bot_audio_at"] = time.time()
    _append_audio_segment(state["active_audio"]["assistant_segments"], sample_rate, audio_array)
    return state, _audio_to_data_uri(audio_array, sample_rate), indicator_html, transcript_html


def finish_survey_session(session_state):
    state = _ensure_survey_state(session_state)
    if not _session_has_content(state):
        gr.Warning(
            "Please try at least one prompt before finishing this session.",
            duration=None,
            title="Survey",
        )
        return _survey_render(state)

    _finalize_survey_session(state)
    if state.get("current_session_index", 0) < len(SESSION_SPECS) - 1:
        state["current_session_index"] += 1
        success, error = _start_survey_session(state)
        if not success and _push_warning(state, "survey", error or "backend unavailable"):
            gr.Warning(
                "The next survey session could not start. Please ask the researcher for help.",
                duration=None,
                title="Survey Warning",
            )
    else:
        state["phase"] = "likert"
        state["likert_index"] = 0

    _save_progress(state)
    return _survey_render(state, transport_value="__reset__")


def likert_back(current_value, session_state):
    state = _ensure_survey_state(session_state)
    _store_likert_value(state, current_value)
    if state.get("likert_index", 0) > 0:
        state["likert_index"] -= 1
    _save_progress(state)
    return _survey_render(state)


def likert_next(current_value, session_state):
    state = _ensure_survey_state(session_state)
    if not _store_likert_value(state, current_value):
        gr.Warning(
            "Please select a rating before continuing.",
            duration=None,
            title="Survey",
        )
        return _survey_render(state)

    if state.get("likert_index", 0) < len(LIKERT_ITEMS) - 1:
        state["likert_index"] += 1
        _save_progress(state)
        return _survey_render(state)

    if state.get("current_bot_index", 0) < len(state.get("bot_sequence", [])) - 1:
        state["current_bot_index"] += 1
        state["current_session_index"] = 0
        state["likert_index"] = 0
        success, error = _start_survey_session(state)
        if not success and _push_warning(state, "survey", error or "backend unavailable"):
            gr.Warning(
                "The next bot session could not start. Please ask the researcher for help.",
                duration=None,
                title="Survey Warning",
            )
    else:
        state["phase"] = "comparison"

    _save_progress(state)
    return _survey_render(state, transport_value="__reset__")


def submit_comparison(natural, interruptions, promptness, overall, free_text, session_state):
    state = _ensure_survey_state(session_state)
    if not natural or not interruptions or not promptness or not overall:
        gr.Warning(
            "Please answer all forced-choice comparison questions before submitting.",
            duration=None,
            title="Survey",
        )
        return _survey_render(state)

    state["comparison"] = {
        "natural": natural,
        "interruptions": interruptions,
        "promptness": promptness,
        "overall": overall,
        "free_text": (free_text or "").strip(),
    }
    state["completed_at"] = _utc_now_iso()
    state["phase"] = "thankyou"
    _reset_survey_runtime(state)
    _save_progress(state, final=True)
    return _survey_render(state, transport_value="__reset__")


def connect_free_play(model_key, free_play_state):
    state = _ensure_free_play_state(free_play_state)
    _close_client(state.get("client"))

    resolved_key = model_key or MODEL_CONFIGS[0]["model_key"]
    config = MODEL_BY_KEY[resolved_key]
    client = FullDuplexClient(config["server_url"], open_timeout=5.0)
    try:
        client.connect(client_name=f"free-play-{resolved_key.lower()}")
    except Exception as exc:
        client.close()
        state["client"] = None
        state["snapshot"] = None
        state["current_model_key"] = resolved_key
        if _push_warning(state, "client", f"{type(exc).__name__}: {exc}"):
            gr.Warning(
                f"Unable to connect to {config['name']}: {type(exc).__name__}: {exc}",
                duration=None,
                title=_warning_title("client"),
            )
        return _free_play_render(state, transport_value="__reset__")

    state["client"] = client
    state["snapshot"] = client.get_latest_snapshot()
    state["current_model_key"] = resolved_key
    state["last_warning_key"] = None
    state["last_mic_at"] = 0.0
    state["last_bot_audio_at"] = 0.0
    return _free_play_render(state, transport_value="__reset__")


def disconnect_free_play(free_play_state):
    state = _ensure_free_play_state(free_play_state)
    _close_client(state.get("client"))
    state["client"] = None
    state["snapshot"] = None
    state["last_warning_key"] = None
    state["last_mic_at"] = 0.0
    state["last_bot_audio_at"] = 0.0
    return _free_play_render(state, transport_value="__reset__")


def receive_free_play_mic(audio, free_play_state):
    state = _ensure_free_play_state(free_play_state)
    if audio is None or state.get("client") is None:
        return state

    sample_rate, audio_array = audio
    audio_float = _normalize_input_audio(audio_array)
    state["last_mic_at"] = time.time()
    try:
        state["client"].send_audio_chunk(sample_rate, audio_float)
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        if _push_warning(state, "client", message):
            gr.Warning(message, duration=None, title=_warning_title("client"))
    return state


def poll_free_play(free_play_state):
    state = _ensure_free_play_state(free_play_state)
    client = state.get("client")
    if client is None:
        return state, gr.skip(), _render_free_play_status_card(state), _render_transcript(state.get("snapshot"), MODEL_BY_KEY[state.get("current_model_key")]["name"])

    for warning in client.drain_warnings():
        message = warning.get("message", "")
        source = warning.get("source", "other")
        if _push_warning(state, source, message):
            gr.Warning(message, duration=None, title=_warning_title(source))

    latest_snapshot = client.get_latest_snapshot()
    if latest_snapshot is not None:
        state["snapshot"] = latest_snapshot

    model_name = MODEL_BY_KEY[state.get("current_model_key")]["name"]
    status_html = _render_free_play_status_card(state)
    transcript_html = _render_transcript(state.get("snapshot"), model_name)

    audio_chunk = client.pop_audio_chunk(timeout=0.0)
    if audio_chunk is None:
        return state, gr.skip(), status_html, transcript_html

    sample_rate, audio_array = audio_chunk
    state["last_bot_audio_at"] = time.time()
    return state, _audio_to_data_uri(audio_array, sample_rate), status_html, transcript_html


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Blind Duplex Survey", css=CUSTOM_CSS, js=INIT_JS) as demo:
        gr.Markdown("# Duplex Voice Survey")
        gr.Markdown(
            "Use the survey tab for blinded A/B data collection. Use the free-play tab for open-ended demos with named models."
        )

        survey_state = gr.State(None)
        free_play_state = gr.State(None)

        with gr.Tab("Survey"):
            survey_timer = gr.Timer(POLL_INTERVAL_S)

            survey_intro = gr.Column(visible=True)
            with survey_intro:
                gr.HTML(
                    "<div class='survey-shell'>"
                    "<div class='eyebrow'>Research Mode</div>"
                    "<h2>Blind A/B conversation survey</h2>"
                    "<p class='lead'>Participants will talk to two hidden duplex bots, complete structured sessions, rate each bot, and then make a final comparison.</p>"
                    "</div>"
                )
                intro_status_html = gr.HTML()
                start_survey_btn = gr.Button("Start Survey", variant="primary")

            presurvey_screen = gr.Column(visible=False)
            with presurvey_screen:
                presurvey_header_html = gr.HTML()
                age_range = gr.Radio(choices=AGE_OPTIONS, label="Age range")
                native_english = gr.Radio(choices=YES_NO_OPTIONS, label="Native English speaker")
                voice_frequency = gr.Radio(
                    choices=VOICE_ASSISTANT_USE_OPTIONS,
                    label="How often does the participant use voice assistants?",
                )
                presurvey_continue_btn = gr.Button("Continue to the first hidden bot", variant="primary")

            conversation_screen = gr.Column(visible=False)
            with conversation_screen:
                conversation_header_html = gr.HTML()
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        session_prompt_html = gr.HTML()
                    with gr.Column(scale=1, min_width=280):
                        survey_indicator_html = gr.HTML()
                        survey_audio = gr.Audio(
                            sources=["microphone"],
                            streaming=True,
                            type="numpy",
                            label="Survey microphone",
                            interactive=False,
                        )
                        survey_audio_transport = gr.Textbox(visible=False, elem_id="survey-audio-transport")
                        finish_session_btn = gr.Button("Finish This Session", variant="primary")
                survey_transcript_html = gr.HTML(
                    value=_render_transcript(None, "Bot")
                )

            likert_screen = gr.Column(visible=False)
            with likert_screen:
                likert_header_html = gr.HTML()
                likert_radio = gr.Radio(choices=LIKERT_CHOICES, label="Rating")
                with gr.Row():
                    likert_back_btn = gr.Button("Back")
                    likert_next_btn = gr.Button("Next", variant="primary")

            comparison_screen = gr.Column(visible=False)
            with comparison_screen:
                comparison_header_html = gr.HTML()
                natural_choice = gr.Radio(
                    choices=COMPARISON_OPTIONS["natural"],
                    label="Which bot felt more natural to talk to?",
                )
                interruption_choice = gr.Radio(
                    choices=COMPARISON_OPTIONS["interruptions"],
                    label="Which bot interrupted you less?",
                )
                promptness_choice = gr.Radio(
                    choices=COMPARISON_OPTIONS["promptness"],
                    label="Which bot responded more promptly?",
                )
                overall_choice = gr.Radio(
                    choices=COMPARISON_OPTIONS["overall"],
                    label="Overall, which bot would you prefer to use?",
                )
                comparison_free_text = gr.Textbox(
                    lines=4,
                    label="What specifically made you prefer one bot over the other?",
                )
                comparison_submit_btn = gr.Button("Submit Survey", variant="primary")

            thankyou_screen = gr.Column(visible=False)
            with thankyou_screen:
                thankyou_html = gr.HTML()

        with gr.Tab("Free Play"):
            free_play_timer = gr.Timer(POLL_INTERVAL_S)

            gr.HTML(
                "<div class='free-shell'>"
                "<div class='eyebrow'>Demo Mode</div>"
                "<h2>Open-ended free play</h2>"
                "<p class='lead'>Choose a named model, connect, and have an unrestricted voice conversation. Nothing from this tab is saved.</p>"
                "</div>"
            )

            with gr.Row():
                free_play_model = gr.Dropdown(
                    choices=FREE_PLAY_CHOICES,
                    value=MODEL_CONFIGS[0]["model_key"],
                    label="Model",
                )
                free_play_connect_btn = gr.Button("Connect", variant="primary")
                free_play_disconnect_btn = gr.Button("Disconnect", interactive=False)

            free_play_status_html = gr.HTML(
                value=_render_free_play_status_card(_default_free_play_state())
            )
            free_play_audio = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Free-play microphone",
                interactive=False,
            )
            free_play_audio_transport = gr.Textbox(visible=False, elem_id="free-play-audio-transport")
            free_play_transcript_html = gr.HTML(
                value=_render_transcript(None, MODEL_CONFIGS[0]["name"])
            )

        demo.load(
            initialize_app,
            outputs=[survey_state, intro_status_html, free_play_state],
        )

        survey_audio_transport.change(
            fn=None,
            inputs=[survey_audio_transport],
            js="(uri) => { if (window._audioEnqueue) window._audioEnqueue('survey', uri); }",
        )

        free_play_audio_transport.change(
            fn=None,
            inputs=[free_play_audio_transport],
            js="(uri) => { if (window._audioEnqueue) window._audioEnqueue('free-play', uri); }",
        )

        survey_outputs = [
            survey_state,
            survey_intro,
            presurvey_screen,
            conversation_screen,
            likert_screen,
            comparison_screen,
            thankyou_screen,
            intro_status_html,
            presurvey_header_html,
            conversation_header_html,
            session_prompt_html,
            survey_indicator_html,
            survey_transcript_html,
            survey_audio,
            survey_audio_transport,
            likert_header_html,
            likert_radio,
            comparison_header_html,
            thankyou_html,
            age_range,
            native_english,
            voice_frequency,
            natural_choice,
            interruption_choice,
            promptness_choice,
            overall_choice,
            comparison_free_text,
        ]

        start_survey_btn.click(
            start_survey,
            inputs=[survey_state],
            outputs=survey_outputs,
        )

        presurvey_continue_btn.click(
            submit_presurvey,
            inputs=[age_range, native_english, voice_frequency, survey_state],
            outputs=survey_outputs,
        )

        survey_audio.stream(
            receive_survey_mic,
            inputs=[survey_audio, survey_state],
            outputs=[survey_state],
        )

        survey_timer.tick(
            poll_survey,
            inputs=[survey_state],
            outputs=[survey_state, survey_audio_transport, survey_indicator_html, survey_transcript_html],
        )

        finish_session_btn.click(
            finish_survey_session,
            inputs=[survey_state],
            outputs=survey_outputs,
        )

        likert_back_btn.click(
            likert_back,
            inputs=[likert_radio, survey_state],
            outputs=survey_outputs,
        )

        likert_next_btn.click(
            likert_next,
            inputs=[likert_radio, survey_state],
            outputs=survey_outputs,
        )

        comparison_submit_btn.click(
            submit_comparison,
            inputs=[natural_choice, interruption_choice, promptness_choice, overall_choice, comparison_free_text, survey_state],
            outputs=survey_outputs,
        )

        free_play_outputs = [
            free_play_state,
            free_play_audio,
            free_play_status_html,
            free_play_transcript_html,
            free_play_connect_btn,
            free_play_disconnect_btn,
            free_play_audio_transport,
        ]

        free_play_connect_btn.click(
            connect_free_play,
            inputs=[free_play_model, free_play_state],
            outputs=free_play_outputs,
        )

        free_play_disconnect_btn.click(
            disconnect_free_play,
            inputs=[free_play_state],
            outputs=free_play_outputs,
        )

        free_play_audio.stream(
            receive_free_play_mic,
            inputs=[free_play_audio, free_play_state],
            outputs=[free_play_state],
        )

        free_play_timer.tick(
            poll_free_play,
            inputs=[free_play_state],
            outputs=[free_play_state, free_play_audio_transport, free_play_status_html, free_play_transcript_html],
        )

        demo.queue()

    return demo


if __name__ == "__main__":
    build_demo().launch(theme=gr.themes.Soft())
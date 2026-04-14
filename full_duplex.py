import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

_client: Optional[OpenAI] = None

load_dotenv()


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when using llm_generate")

    _client = OpenAI(api_key=api_key)
    return _client

AGENT_POLL_MS = int(os.getenv("DUPLEX_AGENT_POLL_MS", "800"))
GRADIO_CHECK_MS = int(os.getenv("DUPLEX_GRADIO_CHECK_MS", "10"))

_template_dir = os.path.dirname(os.path.abspath(__file__)) or "."
_template_env = Environment(loader=FileSystemLoader(_template_dir))
_system_prompt_template = _template_env.get_template("full-duplex.jinja2")


def llm_generate(system_prompt: str, user_message: str) -> str:
    client = _get_client()
    response = client.responses.create(
        model="gpt-5.2",
        instructions=system_prompt,
        input=[
            {"role": "user", "content": user_message},
        ],
        reasoning={"effort": "none"},
        max_output_tokens=16,
    )
    return response.output_text


# ---------------------------------------------------------------------------
# Duration per word for the default (no-TTS) timing estimator.
# ---------------------------------------------------------------------------
_DEFAULT_WORD_DURATION_S = 0.25


def _default_word_timing_fn(text: str) -> List[tuple[str, float, float]]:
    """
    Returns (word, start_offset_s, end_offset_s) relative to TTS start.
    Default: uniform 0.25 s per word, back-to-back.
    Injected callers supply real TTS durations from the audio talker module.
    """
    words = text.split()
    result: List[tuple[str, float, float]] = []
    cursor = 0.0
    for word in words:
        result.append((word, cursor, cursor + _DEFAULT_WORD_DURATION_S))
        cursor += _DEFAULT_WORD_DURATION_S
    return result


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DuplexAgentTimeBlock:
    block_id: str
    start_ts: float
    end_ts: float
    user_text: str = ""
    assistant_text: str = ""
    context_version: int = 0


@dataclass
class AsrAlignedWord:
    text: str
    end_time: float


@dataclass
class AsrTimestampWindow:
    window_id: str
    start_ts: float
    end_ts: float
    words: List[AsrAlignedWord]
    revision: int = 0
    frozen: bool = False


@dataclass
class PurposedWord:
    """
    One proposed agent word with predicted absolute TTS timestamps.

    A word is considered *spoken* when ``end_time <= now``.  It belongs to
    whichever ``DuplexAgentTimeBlock`` whose ``[start_ts, end_ts)`` window
    contains ``end_time`` — regardless of which tick is currently active.
    """
    text: str
    start_time: float   # absolute epoch seconds when TTS begins this word
    end_time: float     # absolute epoch seconds when TTS finishes this word


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class TextOnlyDuplexAgent:
    def __init__(
        self,
        llm_generate_fn: Callable[[str, str], str] = llm_generate,
        agent_poll_ms: int = AGENT_POLL_MS,
        max_prompt_blocks: int = 20,
        word_timing_fn: Optional[Callable[[str], List[tuple[str, float, float]]]] = None,
    ):
        self.llm_generate_fn = llm_generate_fn
        self.agent_poll_ms = agent_poll_ms
        self.interval_s = agent_poll_ms / 1000.0
        self.max_prompt_blocks = max_prompt_blocks
        self.word_timing_fn = word_timing_fn or _default_word_timing_fn

        self.blocks: List[DuplexAgentTimeBlock] = []
        self.current_block: Optional[DuplexAgentTimeBlock] = None
        self.context_version = 0
        self.llm_in_flight = False

        # Proposed agent audio queue — words with predicted TTS timestamps.
        self.purposed_audio: List[PurposedWord] = []
        # Text of words already committed to history (spoken) in the current
        # utterance. Reset when a new user message starts a new utterance.
        # Used by _update_purposed_queue to know how many proposal words to skip.
        self._committed_words_current_utterance: List[str] = []

        self.next_scheduler_ts = 0.0
        self.max_asr_windows = 20
        self.mutable_asr_windows = 10
        self.asr_windows: List[AsrTimestampWindow] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

    @staticmethod
    def _window_id(start_ts: float, end_ts: float) -> str:
        return f"{start_ts:.3f}-{end_ts:.3f}"

    def _now(self) -> float:
        return time.time()

    def _ensure_current_block(self):
        if self.current_block is None:
            now = self._now()
            self.current_block = DuplexAgentTimeBlock(
                block_id=self._new_id(),
                start_ts=now,
                end_ts=now + self.interval_s,
                context_version=self.context_version,
            )

    # ------------------------------------------------------------------
    # User input
    # ------------------------------------------------------------------

    def receive_user_message(self, message: str, ts: Optional[float] = None):
        text = message.strip()
        if not text:
            return

        self.context_version += 1
        self._committed_words_current_utterance = []
        self._ensure_current_block()
        print(f"[user] ctx={self.context_version} {repr(text)}")

        if self.current_block.user_text:
            self.current_block.user_text += " " + text
        else:
            self.current_block.user_text = text

        self.current_block.context_version = self.context_version

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        return (
            text
            .replace("\u2018", "'").replace("\u2019", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "--")
        )

    def _normalize_word(self, text: str) -> str:
        return self._normalize(text).strip()

    # ------------------------------------------------------------------
    # PurposedWord queue management
    # ------------------------------------------------------------------

    def _create_purposed_words(self, text: str, tts_start_ts: float) -> List[PurposedWord]:
        """
        Convert ``word_timing_fn(text)`` relative offsets to absolute timestamps.
        """
        raw = self.word_timing_fn(text)
        result: List[PurposedWord] = []
        for word, start_off, end_off in raw:
            normalized = self._normalize_word(word)
            if not normalized:
                continue
            result.append(PurposedWord(
                text=normalized,
                start_time=tts_start_ts + start_off,
                end_time=tts_start_ts + end_off,
            ))
        return result

    def _update_purposed_queue(self, proposal_words: List[str]) -> None:
        """
        Reconcile ``purposed_audio`` against a new LLM proposal.

        Two cases depending on whether words are currently queued:

        **Queue non-empty** (words still being spoken):
          The LLM may echo the committed prefix before continuing. Walk
          ``zip(committed, proposal)`` to strip any echoed prefix, then
          compare the remaining proposal tail against the unspoken queue.

        **Queue empty** (agent just finished speaking):
          The LLM returns a complete proposed utterance that may include words
          already committed to history. Walk the proposal sequentially,
          consuming committed words one-by-one until the first divergence,
          then treat everything from the divergence point forward as new.
        """
        committed = self._committed_words_current_utterance

        if self.purposed_audio:
            # --- Queue non-empty: find how many committed words LLM echoes ---
            committed_overlap = 0
            for cw, pw in zip(committed, proposal_words):
                if self._normalize_word(pw) == cw:
                    committed_overlap += 1
                else:
                    break
            proposal_tail_raw = proposal_words[committed_overlap:]
        else:
            # --- Queue empty: strip already-spoken words from proposal head ---
            matched = 0
            proposal_tail_raw = []
            for i, pw in enumerate(proposal_words):
                if matched < len(committed) and self._normalize_word(pw) == committed[matched]:
                    matched += 1
                else:
                    proposal_tail_raw = proposal_words[i:]
                    break
            committed_overlap = matched

        proposal_tail_norm = [self._normalize_word(w) for w in proposal_tail_raw]

        # Find first mismatch against the unspoken queue.
        mismatch_idx = min(len(self.purposed_audio), len(proposal_tail_norm))
        for i, (pw, pt) in enumerate(zip(self.purposed_audio, proposal_tail_norm)):
            if pw.text != pt:
                mismatch_idx = i
                break

        retained = self.purposed_audio[:mismatch_idx]
        new_tail_raw = proposal_tail_raw[mismatch_idx:]

        new_tts_start = retained[-1].end_time if retained else self._now()
        new_purposed = (
            self._create_purposed_words(" ".join(new_tail_raw), new_tts_start)
            if new_tail_raw else []
        )

        self.purposed_audio = retained + new_purposed
        print(
            f"[queue] echo={committed_overlap}/{len(committed)} "
            f"retained={[w.text for w in retained]} "
            f"new={[w.text for w in new_purposed]} "
            f"→ full={[w.text for w in self.purposed_audio]}"
        )

    def _compute_forced_purposed_words(self, next_block_end_ts: float) -> List[str]:
        """
        Words whose ``end_time`` falls in ``[now, next_block_end_ts)`` will
        finish speaking during the next block and must appear verbatim at the
        start of the LLM's next output.
        """
        now = self._now()
        return [
            pw.text
            for pw in self.purposed_audio
            if now <= pw.end_time < next_block_end_ts
        ]

    # ------------------------------------------------------------------
    # History block helpers
    # ------------------------------------------------------------------

    def _find_block_for_ts(self, ts: float) -> Optional[DuplexAgentTimeBlock]:
        """Return the finalized block whose ``[start_ts, end_ts)`` contains ts."""
        for block in self.blocks:
            if block.start_ts <= ts < block.end_ts:
                return block
        if self.current_block is not None:
            if self.current_block.start_ts <= ts < self.current_block.end_ts:
                return self.current_block
        return None

    def _commit_spoken_words(self, now: float) -> None:
        """
        Flush all PurposedWords with ``end_time <= now`` into their correct
        DuplexAgentTimeBlock (aligned by word end_time, not current tick).
        """
        newly_spoken = [pw for pw in self.purposed_audio if pw.end_time <= now]

        for pw in newly_spoken:
            target = self._find_block_for_ts(pw.end_time)
            if target is None:
                # Fallback: most recent finalized block or current block.
                target = self.blocks[-1] if self.blocks else self.current_block
            if target is None:
                continue
            word = self._normalize(pw.text)
            if target.assistant_text:
                target.assistant_text += " " + word
            else:
                target.assistant_text = word
            self._committed_words_current_utterance.append(word)

        if newly_spoken:
            print(f"[commit] {[w.text for w in newly_spoken]} | committed={self._committed_words_current_utterance}")

        self.purposed_audio = [pw for pw in self.purposed_audio if pw.end_time > now]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _format_timeblocks(self, forced_words: Optional[List[str]] = None) -> str:
        parts = []
        for block in self.blocks[-self.max_prompt_blocks:]:
            inner = ""
            if block.user_text:
                inner += f'<user>"{block.user_text}"'
            if block.assistant_text:
                inner += f'<assistant>"{block.assistant_text}"'
            parts.append(f"<timeblock>{inner}<endtimeblock>")
        parts.append("<new_assistant_message>")
        if forced_words:
            parts.append(" ".join(forced_words))
        return "".join(parts)

    def _build_prompt(self) -> tuple[str, str]:
        next_block_end_ts = self._now() + self.interval_s
        forced_words = self._compute_forced_purposed_words(next_block_end_ts)
        system_prompt = _system_prompt_template.render(
            has_forced_words=bool(forced_words)
        )
        user_message = self._format_timeblocks(forced_words)
        return system_prompt, user_message

    # ------------------------------------------------------------------
    # ASR window management (user speech path — unchanged)
    # ------------------------------------------------------------------

    def _commit_frozen_window(self, window: AsrTimestampWindow):
        if window.frozen:
            return
        committed_words = [w.text for w in window.words if w.text]
        if committed_words:
            self.receive_user_message(" ".join(committed_words), ts=window.end_ts)
        window.frozen = True

    def _apply_asr_window_policy(self):
        self.asr_windows.sort(key=lambda window: window.end_ts)

        mutable_start = max(0, len(self.asr_windows) - self.mutable_asr_windows)
        for index, window in enumerate(self.asr_windows):
            if index < mutable_start:
                self._commit_frozen_window(window)

        if len(self.asr_windows) > self.max_asr_windows:
            overflow = len(self.asr_windows) - self.max_asr_windows
            to_drop = self.asr_windows[:overflow]
            for window in to_drop:
                self._commit_frozen_window(window)
            self.asr_windows = self.asr_windows[overflow:]

    def _is_window_mutable(self, window_id: str) -> bool:
        self.asr_windows.sort(key=lambda window: window.end_ts)
        mutable_ids = {
            window.window_id
            for window in self.asr_windows[-self.mutable_asr_windows:]
        }
        return window_id in mutable_ids

    def ingest_parakeet_window(
        self,
        start_ts: float,
        end_ts: float,
        words: List[tuple[str, float]],
        window_id: Optional[str] = None,
    ) -> bool:
        resolved_window_id = window_id or self._window_id(start_ts, end_ts)
        normalized_words: List[AsrAlignedWord] = []

        for text, word_end_time in words:
            normalized = self._normalize_word(text)
            if not normalized:
                continue
            normalized_words.append(
                AsrAlignedWord(
                    text=normalized,
                    end_time=float(word_end_time),
                )
            )

        normalized_words.sort(key=lambda word: word.end_time)

        existing = None
        for idx, window in enumerate(self.asr_windows):
            if window.window_id == resolved_window_id:
                existing = (idx, window)
                break

        if existing is None:
            self.asr_windows.append(
                AsrTimestampWindow(
                    window_id=resolved_window_id,
                    start_ts=float(start_ts),
                    end_ts=float(end_ts),
                    words=normalized_words,
                )
            )
            self._apply_asr_window_policy()
            return True

        _, existing_window = existing
        if not self._is_window_mutable(existing_window.window_id):
            return False

        existing_window.start_ts = float(start_ts)
        existing_window.end_ts = float(end_ts)
        existing_window.words = normalized_words
        existing_window.revision += 1
        self._apply_asr_window_policy()
        return True

    def get_asr_window_state(self) -> List[dict]:
        self.asr_windows.sort(key=lambda window: window.end_ts)
        return [
            {
                "window_id": window.window_id,
                "start_ts": window.start_ts,
                "end_ts": window.end_ts,
                "words": [word.text for word in window.words],
                "word_end_times": [word.end_time for word in window.words],
                "revision": window.revision,
                "frozen": window.frozen,
            }
            for window in self.asr_windows
        ]

    # ------------------------------------------------------------------
    # Main poll loop
    # ------------------------------------------------------------------

    def poll(self) -> Optional[DuplexAgentTimeBlock]:
        now = self._now()
        if now < self.next_scheduler_ts:
            return None
        self.next_scheduler_ts = now + self.interval_s

        self._ensure_current_block()

        # Flush words whose predicted TTS end time has passed.
        self._commit_spoken_words(now)

        finalized = self.current_block
        finalized.end_ts = finalized.start_ts + self.interval_s
        self.blocks.append(finalized)
        self.current_block = None

        if self.llm_in_flight:
            return finalized

        has_user_input = any(block.user_text for block in self.blocks)
        if not has_user_input:
            return finalized

        self.llm_in_flight = True
        generation_context_version = self.context_version

        try:
            system_prompt, user_message = self._build_prompt()
            history_summary = " | ".join(
                f"[{b.user_text!r}/{b.assistant_text!r}]" for b in self.blocks[-3:]
            )
            print(f"[llm→] ctx={generation_context_version} history(last3)={history_summary} prompt_tail={repr(user_message[-80:])}")

            raw = self.llm_generate_fn(system_prompt, user_message).strip()

            if generation_context_version != self.context_version:
                print(f"[llm←] stale (ctx {generation_context_version} < {self.context_version}), discarding {repr(raw)}")
                return finalized

            cleaned = self._normalize(raw.strip())
            print(f"[llm←] {repr(cleaned)}")
            if not cleaned:
                self.purposed_audio = []
                return finalized

            self._update_purposed_queue(cleaned.split())
            return finalized
        finally:
            self.llm_in_flight = False

    # ------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------

    def get_chat_history(self):
        history: list[dict] = []

        all_blocks = list(self.blocks)
        if self.current_block is not None:
            all_blocks.append(self.current_block)

        for block in all_blocks:
            if block.user_text:
                if history and history[-1]["role"] == "user":
                    history[-1]["content"] += " " + block.user_text
                else:
                    history.append({"role": "user", "content": block.user_text})

            if block.assistant_text:
                if history and history[-1]["role"] == "assistant":
                    history[-1]["content"] += " " + block.assistant_text
                else:
                    history.append({"role": "assistant", "content": block.assistant_text})

        return history

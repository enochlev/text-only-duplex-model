import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is required when using llm_generate")
    _client = OpenAI(api_key=OPENAI_API_KEY)
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


@dataclass
class DuplexAgentTimeBlock:
    block_id: str
    start_ts: float
    end_ts: float
    user_text: str = ""
    assistant_text: str = ""
    context_version: int = 0


class TextOnlyDuplexAgent:
    def __init__(
        self,
        llm_generate_fn: Callable[[str, str], str] = llm_generate,
        agent_poll_ms: int = AGENT_POLL_MS,
        max_prompt_blocks: int = 20,
    ):
        self.llm_generate_fn = llm_generate_fn
        self.agent_poll_ms = agent_poll_ms
        self.interval_s = agent_poll_ms / 1000.0
        self.max_prompt_blocks = max_prompt_blocks

        self.blocks: List[DuplexAgentTimeBlock] = []
        self.current_block: Optional[DuplexAgentTimeBlock] = None
        self.context_version = 0
        self.llm_in_flight = False
        self.pending_assistant_words: List[str] = []
        self.next_scheduler_ts = 0.0
        self.spoken_assistant_words_current_utterance: List[str] = []

    def _new_id(self) -> str:
        return str(uuid.uuid4())[:8]

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

    def receive_user_message(self, message: str, ts: Optional[float] = None):
        text = message.strip()
        if not text:
            return

        self.context_version += 1
        self._ensure_current_block()

        if self.current_block.user_text:
            self.current_block.user_text += " " + text
        else:
            self.current_block.user_text = text

        self.current_block.context_version = self.context_version

    def _format_timeblocks(self) -> str:
        parts = []
        for block in self.blocks[-self.max_prompt_blocks:]:
            inner = ""
            if block.user_text:
                inner += f'<user>"{block.user_text}"'
            if block.assistant_text:
                inner += f'<assistant>"{block.assistant_text}"'
            parts.append(f"<timeblock>{inner}<endtimeblock>")
        return "".join(parts)

    def _build_prompt(self) -> tuple[str, str]:
        system_prompt = _system_prompt_template.render()
        user_message = self._format_timeblocks()
        return system_prompt, user_message

    def _strip_spoken_prefix(self, proposal_words: List[str]) -> List[str]:
        spoken_words = self.spoken_assistant_words_current_utterance
        overlap = 0
        limit = min(len(spoken_words), len(proposal_words))
        while overlap < limit and proposal_words[overlap] == spoken_words[overlap]:
            overlap += 1
        self.spoken_assistant_words_current_utterance = spoken_words[:overlap]
        return proposal_words[overlap:]

    def _set_pending_assistant_message(self, text: str):
        cleaned = text.strip()
        if not cleaned:
            self.pending_assistant_words = []
            return

        proposal_words = cleaned.split()
        remaining_words = self._strip_spoken_prefix(proposal_words)
        self.pending_assistant_words = remaining_words

    def _speak_next_assistant_word(self):
        if not self.pending_assistant_words:
            return

        self._ensure_current_block()
        next_word = self.pending_assistant_words.pop(0)
        if self.current_block.assistant_text:
            self.current_block.assistant_text += " " + next_word
        else:
            self.current_block.assistant_text = next_word
        self.spoken_assistant_words_current_utterance.append(next_word)

    def poll(self) -> Optional[DuplexAgentTimeBlock]:
        now = self._now()
        if now < self.next_scheduler_ts:
            return None
        self.next_scheduler_ts = now + self.interval_s

        self._ensure_current_block()
        self._speak_next_assistant_word()

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
            raw = self.llm_generate_fn(system_prompt, user_message).strip()

            if generation_context_version != self.context_version:
                return finalized

            self._set_pending_assistant_message(raw)
            return finalized
        finally:
            self.llm_in_flight = False

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

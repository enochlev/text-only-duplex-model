import time
from typing import Optional

import gradio as gr

from full_duplex import AGENT_POLL_MS, GRADIO_CHECK_MS, TextOnlyDuplexAgent
from full_duplex2 import DuplexAudioAgent


# ---------------------------------------------------------------------------
# Text tab handlers
# ---------------------------------------------------------------------------

def push_completed_words(text, agent: Optional[TextOnlyDuplexAgent]):
    if agent is None:
        agent = TextOnlyDuplexAgent()

    if not text or " " not in text:
        return agent.get_chat_history(), text, agent

    parts = text.split(" ")
    completed = [w for w in parts[:-1] if w]
    remainder = parts[-1]

    if completed:
        committed_text = " ".join(completed)
        agent.receive_user_message(committed_text, ts=time.time())

    return agent.get_chat_history(), remainder, agent


def maybe_agent_reply(agent: Optional[TextOnlyDuplexAgent]):
    if agent is None:
        agent = TextOnlyDuplexAgent()

    agent.poll()
    return agent.get_chat_history(), agent


# ---------------------------------------------------------------------------
# Audio tab handlers
# ---------------------------------------------------------------------------

def handle_mic_stream(audio_chunk, session: Optional[DuplexAudioAgent]):
    if session is None:
        session = DuplexAudioAgent()
    if audio_chunk is None:
        return None, session.get_chat_history(), session
    sr, arr = audio_chunk
    tts_chunk = session.receive_mic_chunk(sr, arr)
    return tts_chunk, session.get_chat_history(), session


def handle_audio_tick(session: Optional[DuplexAudioAgent]):
    if session is None:
        return None, [], None
    tts_chunk = session.poll()
    return tts_chunk, session.get_chat_history(), session


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        f"## Duplex Agent Demo\n"
        f"Scheduler tick every **{AGENT_POLL_MS} ms**\n"
        f"Gradio checks every **{GRADIO_CHECK_MS} ms**"
    )

    with gr.Tab("Text"):
        chat = gr.Chatbot(label="Duplex Agent")
        box = gr.Textbox(label="Type here", autofocus=True)
        agent_state = gr.State(TextOnlyDuplexAgent())
        timer = gr.Timer(GRADIO_CHECK_MS / 1000.0)

        box.input(
            fn=push_completed_words,
            inputs=[box, agent_state],
            outputs=[chat, box, agent_state],
            queue=False,
            show_progress="hidden",
        )

        timer.tick(
            fn=maybe_agent_reply,
            inputs=[agent_state],
            outputs=[chat, agent_state],
            queue=False,
            show_progress="hidden",
        )

    with gr.Tab("Audio"):
        with gr.Row():
            mic_in = gr.Audio(
                sources=["microphone"],
                streaming=True,
                type="numpy",
                label="Your Mic",
            )
            agent_out = gr.Audio(
                label="Agent Voice",
                autoplay=True,
                streaming=True,
                type="numpy",
            )
        audio_chat = gr.Chatbot(label="Transcript")
        audio_state = gr.State(None)
        audio_timer = gr.Timer(GRADIO_CHECK_MS / 1000.0)

        mic_in.stream(
            fn=handle_mic_stream,
            inputs=[mic_in, audio_state],
            outputs=[agent_out, audio_chat, audio_state],
            show_progress="hidden",
        )

        audio_timer.tick(
            fn=handle_audio_tick,
            inputs=[audio_state],
            outputs=[agent_out, audio_chat, audio_state],
            show_progress="hidden",
        )

if __name__ == "__main__":
    demo.launch()

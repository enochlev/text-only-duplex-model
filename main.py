import time
from typing import Optional

import gradio as gr

from full_duplex import AGENT_POLL_MS, GRADIO_CHECK_MS, TextOnlyDuplexAgent


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


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------

with gr.Blocks() as demo:
    gr.Markdown(
        f"## Duplex Agent Demo\n"
        f"Scheduler tick every **{AGENT_POLL_MS} ms**\n"
        f"Gradio checks every **{GRADIO_CHECK_MS} ms**"
    )

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

if __name__ == "__main__":
    demo.launch()
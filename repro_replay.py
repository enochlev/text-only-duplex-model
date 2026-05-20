import math
import numpy as np
import full_duplex
from full_duplex import DuplexAudioAgent, TTS_SAMPLE_RATE, DuplexAudioBlock

def make_agent(llm_fn, wpm=60, block_s=1.0):
    def mock_tts(text):
        return TTS_SAMPLE_RATE, np.zeros(100, dtype=np.int16)
    agent = DuplexAudioAgent(wpm=wpm, default_block_s=block_s, llm_generate_fn=llm_fn, tts_fn=mock_tts)
    agent._seal_mic_block = lambda start, end: None
    agent._frozen_time = 1000.0
    agent._now = lambda: agent._frozen_time
    return agent

resp_idx = 0
responses = ["Yeah here.", "yeah here, what is up?"]
def llm_fn(p, u):
    return responses[resp_idx]

# N = 1 word per block (60 wpm, 1s block)
agent = make_agent(llm_fn, wpm=60, block_s=1.0)

# Need some history for LLM to trigger
agent.blocks.append(DuplexAudioBlock(block_id="b1", start_ts=990.0, end_ts=991.0, user_text="Hello"))
agent._latest_user_source_block_id = "b1"
agent.context_version = 1

print("\n--- Step 1: LLM returns 'Yeah here.' ---")
agent.poll() 
# poll() calls _maybe_run_llm -> _pending_words = ['Yeah', 'here.']
# then calls _commit_block_words -> consumes 'Yeah', puts into blocks.append(finalized)
# Wait, poll() appends the block it just committed.

print(f"Blocks after Step 1: {[b.assistant_text for b in agent.blocks if b.assistant_text]}")
print(f"Committed words: {agent._committed_words}")
print(f"Pending words: {agent._pending_words}")

print("\n--- Step 2: LLM returns 'yeah here, what is up?' (triggering overlap failure) ---")
resp_idx = 1
agent.context_version = 2
agent._next_block_ts = 1001.0 # Advance time for next poll
agent._frozen_time = 1001.1
agent.poll()

print(f"Blocks after Step 2: {[b.assistant_text for b in agent.blocks if b.assistant_text]}")
print(f"Committed words: {agent._committed_words}")
print(f"Pending words: {agent._pending_words}")

print("\n--- Step 3: Advance to see next committed block ---")
agent._next_block_ts = 1002.0
agent._frozen_time = 1002.1
agent.poll()

print("\n--- Result Summary ---")
for i, b in enumerate(agent.blocks):
    if b.assistant_text:
        print(f"Block {i}: {repr(b.assistant_text)}")

print(f"Current Pending: {agent._pending_words}")

# Count occurrences of variants of "yeah"
total_text = " ".join([b.assistant_text for b in agent.blocks if b.assistant_text] + agent._pending_words)
print(f"\nWord list in output: {total_text.split()}")

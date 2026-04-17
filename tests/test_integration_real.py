"""
Integration tests — real Piper TTS + real Parakeet ASR (+ optional LLM).

Sections:
  1. Parakeet ASR       — lazy model load, mic pipeline
  2. Piper TTS          — PCM output, duration, latency
  3. Real LLM           — requires OPENAI_API_KEY env var
  4. TTS → ASR roundtrip
  5. Full poll() loop   — requires OPENAI_API_KEY
  6. Speech then silence — block alignment end-to-end

The entire module is skipped when piper-tts or NeMo is not installed.

Run:
    pytest tests/test_integration_real.py -v
"""

import math
import os
import time

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Dependency guards
# ---------------------------------------------------------------------------

piper_voice_mod = pytest.importorskip("piper.voice", reason="piper-tts not installed")
nemo_asr_mod    = pytest.importorskip("nemo.collections.asr", reason="NeMo not installed")

from full_duplex import (
    DEFAULT_BLOCK_S,
    ASR_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    DuplexAudioAgent,
    _prompt_template,
    _resample,
    llm_generate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def require_openai_key():
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        pytest.skip("OPENAI_API_KEY not set")


@pytest.fixture(scope="session")
def tts_agent():
    """A single DuplexAudioAgent with real Piper TTS, shared across TTS tests."""
    agent = DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        asr_fn=lambda *_: None,
    )
    return agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _force_block(agent: DuplexAudioAgent):
    agent._next_block_ts = 0.0
    return agent.poll()


def _minimal_llm_prompt() -> tuple:
    """Return (system, user) matching exactly what DuplexAudioAgent._build_prompt() produces."""
    system = _prompt_template.render()
    user = "<user>hello<AI>"
    return system, user


# ---------------------------------------------------------------------------
# Section 1 — Parakeet ASR (NeMo)
# ---------------------------------------------------------------------------

def _make_asr_agent() -> DuplexAudioAgent:
    return DuplexAudioAgent(
        llm_generate_fn=lambda *_: "",
        tts_fn=lambda t: (TTS_SAMPLE_RATE, np.zeros(int(2.0 * TTS_SAMPLE_RATE), dtype=np.int16)),
    )


class TestParakeetASR:
    def test_asr_model_loads_on_demand(self):
        """_get_asr_model() lazy-loads and returns a non-None model."""
        agent = _make_asr_agent()
        assert agent._get_asr_model() is not None

    def test_asr_model_is_singleton(self):
        """Two agents share the exact same model object."""
        a = _make_asr_agent()
        b = _make_asr_agent()
        assert a._get_asr_model() is b._get_asr_model()

    def test_run_parakeet_empty_rolling(self):
        """Empty rolling list returns immediately without error."""
        agent = _make_asr_agent()
        agent._run_parakeet([])

    def test_receive_mic_chunk_and_seal(self):
        """Streaming mic audio accumulates then triggers real ASR when a block is sealed."""
        agent = _make_asr_agent()
        agent._frozen_time = 1000.0
        agent._now = lambda: agent._frozen_time

        silence = np.zeros(int(0.25 * ASR_SAMPLE_RATE), dtype=np.float32)
        for _ in range(4):
            agent.receive_mic_chunk(ASR_SAMPLE_RATE, silence)

        agent._next_block_ts = 0.0
        agent.poll()
        agent._executor.shutdown(wait=True)

        assert len(agent._mic_rolling) == 1

    def test_asr_window_state_after_poll(self):
        """After a block + ASR completes, get_asr_window_state() returns a list."""
        agent = _make_asr_agent()
        agent._frozen_time = 2000.0
        agent._now = lambda: agent._frozen_time

        agent.receive_mic_chunk(ASR_SAMPLE_RATE, np.zeros(int(2.0 * ASR_SAMPLE_RATE), dtype=np.float32))
        agent._next_block_ts = 0.0
        agent.poll()
        agent._executor.shutdown(wait=True)

        assert isinstance(agent.get_asr_window_state(), list)


# ---------------------------------------------------------------------------
# Section 2 — Piper TTS
# ---------------------------------------------------------------------------

class TestPiperTTS:
    def test_piper_tts_returns_nonzero_pcm(self, tts_agent):
        """_generate_tts returns int16 PCM with at least some non-zero samples."""
        sr, arr, _ = tts_agent._generate_tts("hello")
        assert isinstance(sr, int)
        assert sr > 0
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.int16
        assert len(arr) > 0
        assert np.any(arr != 0), "TTS returned all-zero audio — Piper produced silence"

    def test_piper_tts_sample_rate_matches_voice_config(self, tts_agent):
        """Returned sample rate equals voice.config.sample_rate."""
        sr, _, _ = tts_agent._generate_tts("test")
        assert sr == tts_agent._get_piper_voice().config.sample_rate

    def test_piper_tts_duration_is_realistic(self, tts_agent):
        """A five-word phrase produces between 1 and 6 seconds of audio."""
        sr, arr, _ = tts_agent._generate_tts("hello world how are you")
        duration_s = len(arr) / sr
        assert 1.0 <= duration_s <= 6.0, (
            f"TTS duration {duration_s:.2f}s is outside expected range [1, 6]s"
        )

    def test_piper_tts_latency_under_2s(self, tts_agent):
        """Synthesizing a short phrase completes in under 2 seconds."""
        start = time.perf_counter()
        tts_agent._generate_tts("hello there")
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"TTS took {elapsed:.2f}s — too slow for real-time use"


# ---------------------------------------------------------------------------
# Section 3 — Real LLM
# ---------------------------------------------------------------------------

class TestRealLLM:
    def test_llm_returns_nonempty_string(self, require_openai_key):
        """llm_generate returns a non-empty string."""
        system, user = _minimal_llm_prompt()
        result = llm_generate(system, user)
        assert isinstance(result, str)
        assert len(result.strip()) > 0, "LLM returned an empty string"

    def test_llm_output_is_short(self, require_openai_key):
        """max_output_tokens=16 keeps the response to at most ~20 words."""
        system, user = _minimal_llm_prompt()
        result = llm_generate(system, user)
        word_count = len(result.split())
        assert word_count <= 20, (
            f"LLM returned {word_count} words — max_output_tokens=16 should cap this"
        )

    def test_llm_responds_with_alphabetic_content(self, require_openai_key):
        """LLM response contains at least one alphabetic character."""
        system, user = _minimal_llm_prompt()
        result = llm_generate(system, user)
        assert any(c.isalpha() for c in result), (
            f"LLM response {result!r} contains no alphabetic characters"
        )


# ---------------------------------------------------------------------------
# Section 4 — TTS → ASR round-trip
# ---------------------------------------------------------------------------

class TestTTSASRRoundtrip:
    def _synthesize_as_mic_audio(self, agent: DuplexAudioAgent, text: str) -> np.ndarray:
        """Synthesize text with Piper, convert to float32 at ASR_SAMPLE_RATE."""
        sr, int16_arr, _ = agent._generate_tts(text)
        float32_arr = int16_arr.astype(np.float32) / 32768.0
        return _resample(float32_arr, sr, ASR_SAMPLE_RATE)

    def test_tts_asr_roundtrip_does_not_crash(self, tts_agent):
        """Full TTS→ASR pipeline completes without raising."""
        audio = self._synthesize_as_mic_audio(tts_agent, "hello")
        # _run_parakeet is synchronous — call directly
        tts_agent._run_parakeet([(1000.0, 1000.0 + len(audio) / ASR_SAMPLE_RATE, audio)])
        # No assertion needed beyond no exception

    def test_tts_asr_roundtrip_sentence_recognized(self, tts_agent):
        """Parakeet transcribes Piper-synthesized speech and recovers some words."""
        phrase = "the weather is nice today"
        original_words = set(phrase.lower().split())

        audio = self._synthesize_as_mic_audio(tts_agent, phrase)
        duration = len(audio) / ASR_SAMPLE_RATE
        start_ts, end_ts = 2000.0, 2000.0 + duration

        # Run synchronously
        tts_agent._run_parakeet([(start_ts, end_ts, audio)])

        # Gather all words from any windows that were created for this timestamp range
        recognized = set()
        for w in tts_agent.asr_windows:
            if w.start_ts >= start_ts - 0.1:
                for word in w.words:
                    recognized.add(word.text.lower().strip(".,!?"))

        overlap = recognized & original_words
        assert len(overlap) >= 2, (
            f"Expected ≥2 original words in ASR output. "
            f"Original: {original_words}, Recognized: {recognized}"
        )


# ---------------------------------------------------------------------------
# Section 5 — Full poll() loop with real LLM + real TTS
# ---------------------------------------------------------------------------

class TestFullPollLoop:
    def _real_agent(self) -> DuplexAudioAgent:
        """Real LLM + real TTS; ASR disabled to avoid mic dependency."""
        return DuplexAudioAgent(asr_fn=lambda *_: None)

    def test_poll_loop_produces_assistant_text_and_audio(self, require_openai_key):
        """After a user message, poll() drives a real LLM call and returns real TTS audio."""
        agent = self._real_agent()

        # Block 1: receive user input, run LLM, queue pending words
        agent.receive_text_message("hello")
        _force_block(agent)  # LLM runs here; pending_words should fill

        assert agent._pending_words, "LLM returned nothing — no words pending after first poll"

        # Block 2: commit first N words → generate real TTS
        agent._next_block_ts = 0.0
        result = agent.poll()

        assert agent.blocks[-1].assistant_text != "", (
            "No assistant text committed to block after second poll"
        )
        assert result is not None, "poll() returned None — no audio produced"
        sr, arr = result
        assert arr.dtype == np.int16
        assert np.any(arr != 0), "TTS returned silence for real assistant text"

    def test_block_timestamps_are_realistic(self, require_openai_key):
        """Block start_ts and end_ts fall within the real wall-clock window."""
        agent = self._real_agent()
        wall_start = time.time()

        agent.receive_text_message("hi there")
        _force_block(agent)
        agent._next_block_ts = 0.0
        agent.poll()

        wall_end = time.time()

        for block in agent.blocks:
            assert block.start_ts >= wall_start - 0.1, (
                f"block.start_ts {block.start_ts:.3f} predates wall_start {wall_start:.3f}"
            )
            assert block.end_ts <= wall_end + 1.0, (
                f"block.end_ts {block.end_ts:.3f} is too far in the future"
            )

    def test_tts_duration_sets_next_block_ts(self, require_openai_key):
        """next_block_ts is set to now + actual TTS audio duration (within 0.2s)."""
        agent = self._real_agent()

        agent.receive_text_message("hello how are you doing today")
        _force_block(agent)                   # LLM fills pending words

        t_before = time.time()
        agent._next_block_ts = 0.0
        result = agent.poll()                 # commits words + generates TTS
        t_after = time.time()

        assert result is not None, "No audio returned — cannot check timing"
        sr, arr = result
        expected_duration = len(arr) / sr
        # next_block_ts should be approximately t_before + expected_duration
        assert agent._next_block_ts >= t_before + expected_duration - 0.2
        assert agent._next_block_ts <= t_after  + expected_duration + 0.2, (
            f"next_block_ts={agent._next_block_ts:.3f} is not close to "
            f"t_before+duration={t_before + expected_duration:.3f}"
        )


# ---------------------------------------------------------------------------
# Section 6 — "hey how are you" + 5 s silence: LLM behavior + block alignment
# ---------------------------------------------------------------------------

class TestSpeechThenSilenceAlignment:
    """
    Synthesize 'hey how are you' with real Piper TTS, feed it through the mic
    pipeline to real Parakeet ASR, then run 5 seconds of silence blocks.

    Checks:
    - User speech text lands in a block (ASR → block alignment)
    - LLM receives the context and produces assistant output
    - All timeblock timestamps chain without gaps
    - No block has start_ts > end_ts
    """

    @staticmethod
    def _make_agent() -> DuplexAudioAgent:
        """Real Piper TTS + real Parakeet ASR; deterministic mock LLM."""
        call_log = []

        def llm_fn(_, user_msg):
            call_log.append(user_msg[-120:])  # capture last 120 chars of prompt
            return "I am doing well thank you"

        agent = DuplexAudioAgent(llm_generate_fn=llm_fn)
        agent._call_log = call_log  # expose for assertions
        return agent

    @staticmethod
    def _to_mic(agent: DuplexAudioAgent, text: str) -> np.ndarray:
        """Synthesize text with real Piper; return float32 at ASR_SAMPLE_RATE."""
        sr, pcm, _ = agent._generate_tts(text)
        return _resample(pcm.astype(np.float32) / 32768.0, sr, ASR_SAMPLE_RATE)

    def test_speech_then_silence(self):
        """
        Feed synthesized user speech then silence. Verify block timestamps chain
        correctly and ASR output is aligned to the right block.
        """
        agent = self._make_agent()
        t0 = 1000.0
        agent._frozen_time = t0
        agent._now = lambda: agent._frozen_time

        # --- Synthesize user speech as mic audio ---
        speech_audio = self._to_mic(agent, "hey how are you")
        speech_s = len(speech_audio) / ASR_SAMPLE_RATE
        print(f"\n[test] speech audio: {speech_s:.2f}s  ({len(speech_audio)} samples @ {ASR_SAMPLE_RATE} Hz)")

        block_samples = int(DEFAULT_BLOCK_S * ASR_SAMPLE_RATE)
        total_s = speech_s + 5.0
        n_blocks = math.ceil(total_s / DEFAULT_BLOCK_S) + 2  # +2 for initial zero-dur + margin

        # --- Drive the poll loop ---
        for i in range(n_blocks):
            # Feed the chunk of audio that belongs to this block window
            offset = i * block_samples
            raw = speech_audio[offset : offset + block_samples]
            if len(raw) > 0:
                chunk = raw.astype(np.float32)
            else:
                chunk = np.zeros(block_samples, dtype=np.float32)
            agent.receive_mic_chunk(ASR_SAMPLE_RATE, chunk)

            agent._next_block_ts = 0.0
            agent.poll()
            agent._frozen_time += DEFAULT_BLOCK_S

        # Wait for all background ASR jobs to finish
        agent._executor.shutdown(wait=True)

        # --- Print block timeline for visual inspection ---
        print(f"\n[test] {len(agent.blocks)} blocks produced:")
        for b in agent.blocks:
            print(
                f"  [{b.start_ts - t0:+.2f}s → {b.end_ts - t0:+.2f}s]"
                f"  user={repr(b.user_text):<35}"
                f"  assistant={repr(b.assistant_text)}"
            )

        if agent._call_log:
            print(f"\n[test] LLM prompt tail (last call): {agent._call_log[-1]!r}")

        blocks = agent.blocks
        assert len(blocks) >= 3, f"Expected ≥3 blocks, got {len(blocks)}"

        # 1. All blocks have valid boundaries (start_ts ≤ end_ts)
        for b in blocks:
            assert b.start_ts <= b.end_ts, (
                f"Block {b.block_id}: inverted timestamps "
                f"[{b.start_ts:.3f}, {b.end_ts:.3f}]"
            )

        # 2. Timestamps chain without gaps
        for i in range(len(blocks) - 1):
            gap = abs(blocks[i].end_ts - blocks[i + 1].start_ts)
            assert gap < 0.01, (
                f"Gap of {gap:.4f}s between block {i} (end={blocks[i].end_ts:.3f}) "
                f"and block {i+1} (start={blocks[i+1].start_ts:.3f})"
            )

        # 3. User speech text is assigned to at least one block
        all_user = " ".join(b.user_text for b in blocks if b.user_text).lower()
        recognized = set(all_user.split())
        expected_words = {"hey", "how", "are", "you"}
        matched = recognized & expected_words
        assert len(matched) >= 2, (
            f"Expected ≥2 words from 'hey how are you' in block user_text.\n"
            f"All user text across blocks: {all_user!r}"
        )

        # 4. LLM responded with assistant text in at least one block
        all_assistant = " ".join(b.assistant_text for b in blocks if b.assistant_text)
        assert all_assistant.strip(), (
            "LLM produced no assistant text — check that LLM is being called"
        )

        # 5. User text is not smeared across too many blocks (aligned, not scattered)
        blocks_with_user_text = [b for b in blocks if b.user_text]
        assert len(blocks_with_user_text) <= math.ceil(speech_s / DEFAULT_BLOCK_S) + 2, (
            f"User text spread across {len(blocks_with_user_text)} blocks — "
            f"expected ≤{math.ceil(speech_s / DEFAULT_BLOCK_S) + 2}"
        )

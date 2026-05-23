"""test_rewards.py — unit tests for all reward functions in trainer/rewards.py.

Covers:
  - interruption_penalty        (pure logic, no external deps)
  - respond_after_user_reward   (mocks _user_finished_in to avoid VAD server)
  - interruption_penalty_overlap (mocks _vad_overlap_score to avoid VAD server)

No VAD server or GPU required.
"""

import numpy as np
import pytest

from full_duplex import DuplexAudioBlock
from trainer.rewards import (
    backchannel_loop_penalty,
    interruption_penalty,
    interruption_penalty_overlap,
    respond_after_user_reward,
)
import trainer.rewards as _rewards_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block(
    user_text: str = "",
    assistant_text: str = "",
    mic_audio: np.ndarray | None = None,
    tts_audio: np.ndarray | None = None,
    block_id: str = "b",
) -> DuplexAudioBlock:
    b = DuplexAudioBlock.__new__(DuplexAudioBlock)
    b.block_id = block_id
    b.start_ts = 0.0
    b.end_ts = 2.0
    b.user_text = user_text
    b.assistant_text = assistant_text
    b.assistant_text_stale = False
    b.mic_audio = mic_audio
    b.tts_audio = tts_audio
    b.tts_sr = 24_000
    b.tts_latency_s = None
    b.asr_latency_s = None
    b.llm_latency_s = None
    b.total_latency_s = None
    b.asr_started_perf_s = None
    b.response_source_block_id = None
    b.timeline_start_ts = None
    b.timeline_end_ts = None
    b.lead_silence_s = 0.0
    return b


def _nonsilent_audio(n: int = 1600) -> np.ndarray:
    """Float32 PCM with RMS well above _RMS_SILENCE (1e-4)."""
    return np.full(n, 0.1, dtype=np.float32)


# ---------------------------------------------------------------------------
# interruption_penalty
# (pure logic — no mocking required)
# ---------------------------------------------------------------------------

class TestInterruptionPenalty:

    def test_no_user_text_returns_zero(self):
        block = _block(assistant_text="hello")
        assert interruption_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_no_bot_text_returns_zero(self):
        block = _block(user_text="hello")
        assert interruption_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_both_empty_returns_zero(self):
        block = _block()
        assert interruption_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_first_overlap_run1_is_free(self):
        """run=1: first simultaneous block gets no penalty."""
        block = _block(user_text="hi", assistant_text="hello")
        assert interruption_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_second_consecutive_overlap_run2(self):
        """run=2: second consecutive simultaneous block → -0.5."""
        history = [_block(user_text="a", assistant_text="b")]
        block = _block(user_text="c", assistant_text="d")
        assert interruption_penalty(block, history, False) == -0.5

    def test_third_consecutive_overlap_run3(self):
        """run=3 → -1.0."""
        history = [
            _block(user_text="a", assistant_text="b"),
            _block(user_text="c", assistant_text="d"),
        ]
        block = _block(user_text="e", assistant_text="f")
        assert interruption_penalty(block, history, False) == -1.0

    def test_fourth_consecutive_overlap_run4_plus(self):
        """run=4+ → -2.0."""
        history = [
            _block(user_text="a", assistant_text="b"),
            _block(user_text="c", assistant_text="d"),
            _block(user_text="e", assistant_text="f"),
        ]
        block = _block(user_text="g", assistant_text="h")
        assert interruption_penalty(block, history, False) == -2.0

    def test_fifth_consecutive_capped_at_minus_two(self):
        """run=5 stays at -2.0."""
        history = [_block(user_text="x", assistant_text="y") for _ in range(4)]
        block = _block(user_text="a", assistant_text="b")
        assert interruption_penalty(block, history, False) == -2.0

    def test_break_in_overlap_resets_run(self):
        """A non-overlap block in history breaks the run → next overlap is run=1 → 0.0."""
        history = [
            _block(user_text="a", assistant_text="b"),
            _block(user_text="c"),             # user only — breaks run
        ]
        block = _block(user_text="e", assistant_text="f")
        assert interruption_penalty(block, history, False) == pytest.approx(0.0, abs=1e-9)

    def test_bot_only_block_in_history_breaks_run(self):
        history = [
            _block(user_text="a", assistant_text="b"),
            _block(assistant_text="c"),        # bot only — breaks run
        ]
        block = _block(user_text="e", assistant_text="f")
        assert interruption_penalty(block, history, False) == pytest.approx(0.0, abs=1e-9)

    def test_empty_history_overlap_is_free(self):
        block = _block(user_text="hi", assistant_text="hello")
        assert interruption_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_escalation_matches_log_data(self):
        """
        Validates the penalty schedule seen in real reward logs:
          run=2 → -0.5,  run=3 → -1.0,  run=4 → -2.0
        """
        def run_penalty(run_length: int) -> float:
            history = [
                _block(user_text="u", assistant_text="a")
                for _ in range(run_length - 1)
            ]
            return interruption_penalty(
                _block(user_text="u", assistant_text="a"), history, False
            )

        assert run_penalty(1) == pytest.approx(0.0, abs=1e-9)
        assert run_penalty(2) == -0.5
        assert run_penalty(3) == -1.0
        assert run_penalty(4) == -2.0


# ---------------------------------------------------------------------------
# respond_after_user_reward
# (_user_finished_in is mocked to isolate from VAD server)
# ---------------------------------------------------------------------------

class TestRespondAfterUserReward:

    @pytest.fixture(autouse=True)
    def mock_user_finished(self, monkeypatch):
        """By default, any block with user_text is considered a complete turn."""
        monkeypatch.setattr(
            _rewards_mod, "_user_finished_in",
            lambda b: bool(b.user_text)
        )

    def test_block_with_user_text_returns_zero(self):
        """Bot silent, but user is still speaking → no penalty."""
        block = _block(user_text="hi")
        assert respond_after_user_reward(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_block_with_bot_text_returns_zero(self):
        """Bot already responding → no penalty."""
        block = _block(assistant_text="hello")
        history = [_block(user_text="hi")]
        assert respond_after_user_reward(block, history, False) == pytest.approx(0.0, abs=1e-9)

    def test_block_with_both_texts_returns_zero(self):
        block = _block(user_text="hi", assistant_text="hello")
        assert respond_after_user_reward(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_no_prior_user_turn_returns_zero(self):
        """No user has spoken yet → no penalty for silence."""
        block = _block()
        assert respond_after_user_reward(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_bot_already_responded_in_history_returns_zero(self):
        """Bot replied before the silent block → silence is voluntary, no penalty."""
        history = [
            _block(user_text="hi"),
            _block(assistant_text="hello"),  # bot already spoke
        ]
        block = _block()
        assert respond_after_user_reward(block, history, False) == pytest.approx(0.0, abs=1e-9)

    def test_lag1_first_silent_block_after_user(self):
        """First silent block immediately after user finishes → -1.0."""
        history = [_block(user_text="hi")]
        block = _block()
        assert respond_after_user_reward(block, history, False) == -1.0

    def test_lag2_second_silent_block(self):
        """Second consecutive silent block → -2.0."""
        history = [
            _block(user_text="hi"),
            _block(),  # first silent block
        ]
        block = _block()
        assert respond_after_user_reward(block, history, False) == -2.0

    def test_lag3_third_silent_block(self):
        """Third consecutive silent block → -3.0."""
        history = [
            _block(user_text="hi"),
            _block(),
            _block(),
        ]
        block = _block()
        assert respond_after_user_reward(block, history, False) == -3.0

    def test_lag4_capped_at_minus_three(self):
        """lag=4+ is capped at -3.0."""
        history = [
            _block(user_text="hi"),
            _block(),
            _block(),
            _block(),
        ]
        block = _block()
        assert respond_after_user_reward(block, history, False) == -3.0

    def test_incomplete_user_turn_no_penalty(self, monkeypatch):
        """If user's turn is not yet complete, no penalty for bot silence."""
        monkeypatch.setattr(_rewards_mod, "_user_finished_in", lambda b: False)
        history = [_block(user_text="and then I")]
        block = _block()
        assert respond_after_user_reward(block, history, False) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# interruption_penalty_overlap
# (_vad_overlap_score is mocked; zero-audio fast-path is tested without mocking)
# ---------------------------------------------------------------------------

class TestInterruptionPenaltyOverlap:

    def test_no_user_text_returns_zero(self):
        block = _block(
            assistant_text="hello",
            mic_audio=_nonsilent_audio(),
            tts_audio=_nonsilent_audio(),
        )
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_no_bot_text_returns_zero(self):
        block = _block(
            user_text="hello",
            mic_audio=_nonsilent_audio(),
            tts_audio=_nonsilent_audio(),
        )
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_both_empty_text_returns_zero(self):
        block = _block(mic_audio=_nonsilent_audio(), tts_audio=_nonsilent_audio())
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_no_audio_returns_zero(self):
        block = _block(user_text="hi", assistant_text="hello")
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_empty_audio_arrays_return_zero(self):
        block = _block(
            user_text="hi",
            assistant_text="hello",
            mic_audio=np.array([], dtype=np.float32),
            tts_audio=np.array([], dtype=np.int16),
        )
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_silent_audio_returns_zero(self):
        """Zero-amplitude audio skips the VAD call and returns 0.0."""
        block = _block(
            user_text="hi",
            assistant_text="hello",
            mic_audio=np.zeros(1600, dtype=np.float32),
            tts_audio=np.zeros(1600, dtype=np.int16),
        )
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_vad_server_failure_returns_zero(self, monkeypatch):
        """If VAD server is down, falls back to 0.0."""
        monkeypatch.setattr(_rewards_mod, "_vad_overlap_score", lambda *_: None)
        block = _block(
            user_text="hi",
            assistant_text="hello",
            mic_audio=_nonsilent_audio(),
            tts_audio=_nonsilent_audio(),
        )
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_overlap_ratio_negated(self, monkeypatch):
        """Penalty = -overlap_ratio."""
        monkeypatch.setattr(_rewards_mod, "_vad_overlap_score", lambda *_: 0.73)
        block = _block(
            user_text="hi",
            assistant_text="hello",
            mic_audio=_nonsilent_audio(),
            tts_audio=_nonsilent_audio(),
        )
        result = interruption_penalty_overlap(block, [], False)
        assert abs(result - (-0.73)) < 1e-6

    def test_zero_overlap_ratio_returns_zero(self, monkeypatch):
        """VAD reports no overlap → 0.0 penalty."""
        monkeypatch.setattr(_rewards_mod, "_vad_overlap_score", lambda *_: 0.0)
        block = _block(
            user_text="hi",
            assistant_text="hello",
            mic_audio=_nonsilent_audio(),
            tts_audio=_nonsilent_audio(),
        )
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_full_overlap_ratio_returns_minus_one(self, monkeypatch):
        """VAD reports full overlap → -1.0 penalty."""
        monkeypatch.setattr(_rewards_mod, "_vad_overlap_score", lambda *_: 1.0)
        block = _block(
            user_text="hi",
            assistant_text="hello",
            mic_audio=_nonsilent_audio(),
            tts_audio=_nonsilent_audio(),
        )
        assert interruption_penalty_overlap(block, [], False) == pytest.approx(-1.0)

    def test_penalty_range_is_zero_to_minus_one(self, monkeypatch):
        """Penalty is always in [-1.0, 0.0]."""
        for ratio in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            monkeypatch.setattr(_rewards_mod, "_vad_overlap_score", lambda *_, r=ratio: r)
            block = _block(
                user_text="hi",
                assistant_text="hello",
                mic_audio=_nonsilent_audio(),
                tts_audio=_nonsilent_audio(),
            )
            result = interruption_penalty_overlap(block, [], False)
            assert -1.0 <= result <= 0.0, f"ratio={ratio} → penalty={result} out of range"


# ---------------------------------------------------------------------------
# backchannel_loop_penalty
# (pure logic — no mocking required)
# ---------------------------------------------------------------------------

class TestBackchannelLoopPenalty:

    def test_empty_assistant_text_returns_zero(self):
        block = _block()
        assert backchannel_loop_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_non_backchannel_text_returns_zero(self):
        block = _block(assistant_text="That's a great question.")
        assert backchannel_loop_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_first_backchannel_no_history_returns_zero(self):
        block = _block(assistant_text="yeah")
        assert backchannel_loop_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_first_backchannel_after_non_backchannel_returns_zero(self):
        history = [_block(assistant_text="So the answer is five.")]
        block = _block(assistant_text="right")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(0.0, abs=1e-9)

    def test_two_consecutive_backchannels(self):
        history = [_block(assistant_text="yeah")]
        block = _block(assistant_text="uh-huh")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(-0.5)

    def test_three_consecutive_backchannels(self):
        history = [
            _block(assistant_text="yeah"),
            _block(assistant_text="uh-huh"),
        ]
        block = _block(assistant_text="i know")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(-1.0)

    def test_four_consecutive_backchannels(self):
        history = [
            _block(assistant_text="yeah"),
            _block(assistant_text="uh-huh"),
            _block(assistant_text="right"),
        ]
        block = _block(assistant_text="okay")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(-1.5)

    def test_non_backchannel_in_history_breaks_run(self):
        history = [
            _block(assistant_text="yeah"),
            _block(assistant_text="Let me think about that."),  # breaks run
        ]
        block = _block(assistant_text="right")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(0.0, abs=1e-9)

    def test_idle_bot_block_in_history_breaks_run(self):
        history = [
            _block(assistant_text="yeah"),
            _block(),  # silence — breaks run
        ]
        block = _block(assistant_text="right")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(0.0, abs=1e-9)

    def test_case_normalized(self):
        history = [_block(assistant_text="Yeah")]
        block = _block(assistant_text="OK")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(-0.5)

    def test_trailing_punctuation_normalized(self):
        history = [_block(assistant_text="yeah.")]
        block = _block(assistant_text="right,")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(-0.5)

    def test_long_text_containing_backchannel_word_not_matched(self):
        """'right so the answer is five' should NOT match — full text, not substring."""
        block = _block(assistant_text="right so the answer is five")
        assert backchannel_loop_penalty(block, [], False) == pytest.approx(0.0, abs=1e-9)

    def test_two_word_backchannel_phrase_matched(self):
        history = [_block(assistant_text="yeah")]
        block = _block(assistant_text="I know")
        assert backchannel_loop_penalty(block, history, False) == pytest.approx(-0.5)

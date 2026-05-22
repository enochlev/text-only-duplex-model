"""trainer — full-duplex RL training package.

Must be used from the repository root so that `full_duplex` (in the root) is
importable. Run: `python trainer.py` from the repo root.
"""

from .rl_trainer import (
    FullDuplexRLTrainer,
    TrainerConfig,
    StepRecord,
    Episode,
    llm_generate_train,
    VirtualSimulationConnection,
)
from .rewards import (
    RewardFn,
    respond_after_user_reward,
    coherence_reward,
    interruption_penalty,
    interruption_penalty_overlap,
    silence_too_long_penalty,
    monologue_too_long_penalty,
    check_rm_servers,
)
from .data_ingestion import (
    EpisodeData,
    DataPool,
    StaticWavSource,
    ScriptTTSSource,
    GPTVoiceSimulator,
    PlaybackSimulator,
    TRAINING_SCRIPTS,
    make_default_data_pool,
)

__all__ = [
    # rl_trainer
    "FullDuplexRLTrainer",
    "TrainerConfig",
    "StepRecord",
    "Episode",
    "llm_generate_train",
    "VirtualSimulationConnection",
    # rewards
    "RewardFn",
    "latency_reward",
    "idle_penalty",
    "response_length_reward",
    "respond_after_user_reward",
    "coherence_reward",
    "interruption_penalty",
    "interruption_penalty_overlap",
    "silence_too_long_penalty",
    "monologue_too_long_penalty",
    "check_rm_servers",
    # data_ingestion
    "EpisodeData",
    "DataPool",
    "StaticWavSource",
    "ScriptTTSSource",
    "GPTVoiceSimulator",
    "PlaybackSimulator",
    "TRAINING_SCRIPTS",
    "make_default_data_pool",
]

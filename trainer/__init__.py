"""trainer — full-duplex RL training package.

Must be used from the repository root so that `full_duplex` (in the root) is
importable. Run: `python trainer.py` from the repo root.
"""

from .sft_trainer import SFTTrainer, SFTConfig, SilenceDataCollector
from .training_utils import load_hf_model
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
    block_silence_penalty,
    block_interruption_penalty,
    block_idle_reward,
    timely_response_reward,
    vad_overlap_penalty,
    backchannel_loop_penalty,
    junk_output_penalty,
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
    set_embed_device,
)

__all__ = [
    # sft_trainer
    "SFTTrainer",
    "SFTConfig",
    "SilenceDataCollector",
    # training_utils
    "load_hf_model",
    # rl_trainer
    "FullDuplexRLTrainer",
    "TrainerConfig",
    "StepRecord",
    "Episode",
    "llm_generate_train",
    "VirtualSimulationConnection",
    # rewards
    "RewardFn",
    "block_silence_penalty",
    "block_interruption_penalty",
    "block_idle_reward",
    "timely_response_reward",
    "vad_overlap_penalty",
    "backchannel_loop_penalty",
    "junk_output_penalty",
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
    "set_embed_device",
]

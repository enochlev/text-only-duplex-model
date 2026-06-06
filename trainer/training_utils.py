"""training_utils.py — shared helpers used by both SFTTrainer and FullDuplexRLTrainer."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


def load_hf_model(
    model_name_or_path: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = True,
) -> AutoModelForCausalLM:
    """Load a causal-LM model in training mode with optional gradient checkpointing."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,  # MiniCPM ships custom modeling code
    )
    model.to(device)
    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model

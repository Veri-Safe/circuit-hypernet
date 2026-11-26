"""Locked configuration and learning-rate math for delta-4 HF/DDP training."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration drifts from the locked protocol."""


LOCKED_VALUES: dict[str, dict[str, Any]] = {
    "model": {
        "parameter_dtype": "float32",
        "compute_dtype": "bfloat16",
        "attention_implementation": "flash_attention_2",
        "gradient_checkpointing": False,
        "max_sequence_length": 262_144,
        "expected_parameters": 3_633_552_896,
    },
    "training": {
        "seed": 24,
        "global_batch_size": 64,
        "micro_batch_size": 1,
        "logical_epochs": 50,
        "steps_per_epoch": 25,
        "max_steps": 1_250,
        "loss_normalization": "global_assistant_tokens",
    },
    "optimizer": {
        "name": "AdamW",
        "beta1": 0.9,
        "beta2": 0.98,
        "epsilon": 1.0e-8,
        "weight_decay": 0.1,
        "clip_grad_norm": 0.5,
        "fused": True,
    },
    "scheduler": {
        "name": "linear_warmup_decay",
        "warmup_steps": 25,
        "warmup_initial_lr": 1.0e-7,
        "min_lr": 1.0e-7,
        "decay_steps": 1_250,
    },
    "validation": {
        "at_start": True,
        "period": 25,
        "micro_batch_size": 2,
    },
    "checkpointing": {
        "save_period": 50,
        "save_dtype": "bfloat16",
        "safe_serialization": True,
        "max_shard_size": "5GB",
        "save_optimizer": False,
    },
    "data": {"pad_to_multiple_of": 64, "num_workers": 0},
}


def configure_process_batch(training: dict[str, Any]) -> None:
    """Resolve launch topology and preserve the configured logical batch."""
    workers = int(os.environ.get("WORLD_SIZE", "1"))
    local_workers = int(os.environ.get("LOCAL_WORLD_SIZE", str(workers)))
    micro = int(training["micro_batch_size"])
    size = int(training["global_batch_size"])
    if not 0 < local_workers <= workers or micro <= 0 or size % (workers * micro):
        raise ConfigError("Logical batch must divide evenly across launched workers")
    training.update(world_size=workers, gpus_per_node=local_workers,
                    gradient_accumulation_steps=size // (workers * micro))


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    if payload.get("format_version") != "nvarc_delta4_hf_ddp_v1":
        raise ConfigError("Unexpected HF/DDP configuration format")
    for section, expected_values in LOCKED_VALUES.items():
        observed_section = payload.get(section)
        if not isinstance(observed_section, dict):
            raise ConfigError(f"Missing configuration section: {section}")
        for key, expected in expected_values.items():
            observed = observed_section.get(key)
            if observed != expected:
                raise ConfigError(
                    f"Locked configuration changed: {section}.{key}="
                    f"{observed!r}, expected {expected!r}"
                )
    training = payload["training"]
    configure_process_batch(training)
    if (
        training["world_size"]
        * training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        != training["global_batch_size"]
    ):
        raise ConfigError("world_size * micro_batch * accumulation must equal global batch")
    if training["steps_per_epoch"] * training["logical_epochs"] != training["max_steps"]:
        raise ConfigError("logical epoch count does not match max_steps")
    return payload


def learning_rate_for_step(
    step: int,
    *,
    peak_lr: float,
    warmup_steps: int,
    warmup_initial_lr: float,
    decay_steps: int,
    min_lr: float,
) -> float:
    """Return the optimizer LR for one-based ``step``."""

    if not 1 <= step <= decay_steps:
        raise ValueError(f"step must be in [1, {decay_steps}], got {step}")
    if not 0 < warmup_steps < decay_steps:
        raise ValueError("warmup_steps must be between zero and decay_steps")
    if not 0.0 <= min_lr <= peak_lr or not 0.0 <= warmup_initial_lr <= peak_lr:
        raise ValueError("initial/minimum learning rates must not exceed peak_lr")
    if step <= warmup_steps:
        fraction = (step - 1) / max(1, warmup_steps - 1)
        return warmup_initial_lr + fraction * (peak_lr - warmup_initial_lr)
    fraction = (step - warmup_steps) / (decay_steps - warmup_steps)
    return peak_lr - fraction * (peak_lr - min_lr)

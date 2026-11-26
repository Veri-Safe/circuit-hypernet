"""Learning-rate schedules shared by Stage-B SFT tuning rounds."""

from __future__ import annotations

import math


CONSTANT_SCHEDULE = "linear_warmup_then_constant"
COSINE_SCHEDULE = "linear_warmup_then_cosine_decay"
LINEAR_DECAY_SCHEDULE = "linear_warmup_then_linear_decay"
SUPPORTED_SCHEDULES = (
    CONSTANT_SCHEDULE,
    COSINE_SCHEDULE,
    LINEAR_DECAY_SCHEDULE,
)


def learning_rate_for_step(
    step: int,
    *,
    learning_rate: float,
    warmup_steps: int,
    total_steps: int | None = None,
    decay_end_step: int | None = None,
    schedule: str = CONSTANT_SCHEDULE,
    final_learning_rate_ratio: float = 0.1,
) -> float:
    """Return the exact optimizer LR for a one-indexed training step."""

    if step < 1:
        raise ValueError("step must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if schedule not in SUPPORTED_SCHEDULES:
        raise ValueError(f"unsupported learning-rate schedule: {schedule}")
    if not 0.0 <= final_learning_rate_ratio <= 1.0:
        raise ValueError("final_learning_rate_ratio must be in [0, 1]")
    if warmup_steps and step < warmup_steps:
        return learning_rate * step / warmup_steps
    if schedule == CONSTANT_SCHEDULE:
        return learning_rate
    effective_decay_end = decay_end_step if decay_end_step is not None else total_steps
    if effective_decay_end is None or effective_decay_end <= warmup_steps:
        raise ValueError("decay schedules require total_steps > warmup_steps")

    progress = (step - warmup_steps) / (effective_decay_end - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    if schedule == COSINE_SCHEDULE:
        decay_fraction = 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        decay_fraction = 1.0 - progress
    multiplier = final_learning_rate_ratio + (
        1.0 - final_learning_rate_ratio
    ) * decay_fraction
    return learning_rate * multiplier

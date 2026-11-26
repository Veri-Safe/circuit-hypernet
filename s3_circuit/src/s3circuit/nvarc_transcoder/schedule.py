from __future__ import annotations

import math

import torch


def continuation_lr_lambda(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float = 0.02,
    cooldown_fraction: float = 0.20,
) -> float:
    if total_steps <= 0 or step < 0:
        raise ValueError("Invalid continuation schedule position")
    warmup_steps = max(1, math.ceil(total_steps * warmup_fraction))
    cooldown_start = max(warmup_steps, math.floor(total_steps * (1.0 - cooldown_fraction)))
    if step < warmup_steps:
        return min(1.0, float(step + 1) / warmup_steps)
    if step < cooldown_start:
        return 1.0
    if step >= total_steps:
        return 0.0
    return float(total_steps - step) / max(1, total_steps - cooldown_start)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_fraction: float,
    cooldown_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: continuation_lr_lambda(
            step,
            total_steps=total_steps,
            warmup_fraction=warmup_fraction,
            cooldown_fraction=cooldown_fraction,
        ),
    )


def percentage_targets(total_steps: int, interval_percent: int) -> dict[int, int]:
    if total_steps <= 0 or interval_percent <= 0 or 100 % interval_percent:
        raise ValueError("Invalid percentage checkpoint schedule")
    return {
        percent: max(1, round(total_steps * percent / 100))
        for percent in range(interval_percent, 101, interval_percent)
    }

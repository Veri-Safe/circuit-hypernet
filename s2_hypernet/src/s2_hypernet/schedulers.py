from __future__ import annotations

import math
from typing import Callable

import torch
from transformers import get_scheduler


def resolve_warmup_steps(total_steps: int, warmup_ratio: float, warmup_steps: int) -> int:
    if warmup_steps >= 0:
        return min(total_steps, warmup_steps)
    return min(total_steps, max(0, int(total_steps * warmup_ratio)))


def cosine_to_floor_lambda(
    *,
    total_steps: int,
    warmup_steps: int,
    floor_step: int,
    final_decay_steps: int,
    floor_ratio: float,
) -> Callable[[int], float]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive.")
    if not 0.0 <= floor_ratio <= 1.0:
        raise ValueError(f"floor_ratio must be in [0, 1], got {floor_ratio}.")
    if warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError(f"warmup_steps must be in [0, {total_steps - 1}], got {warmup_steps}.")
    if floor_step <= warmup_steps or floor_step > total_steps:
        raise ValueError(
            f"floor_step must be greater than warmup_steps and at most total_steps; got {floor_step}."
        )
    if final_decay_steps < 0:
        raise ValueError("final_decay_steps must be non-negative.")
    if final_decay_steps == 0 and floor_step != total_steps:
        raise ValueError("final_decay_steps=0 requires floor_step=total_steps.")
    decay_start = total_steps - final_decay_steps
    if decay_start < floor_step:
        raise ValueError(
            f"Final decay starts at step {decay_start}, before floor_step={floor_step}."
        )

    def lr_lambda(current_step: int) -> float:
        step = max(0, min(int(current_step), total_steps))
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if step <= floor_step:
            progress = (step - warmup_steps) / (floor_step - warmup_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return floor_ratio + (1.0 - floor_ratio) * cosine
        if step <= decay_start:
            return floor_ratio
        progress = (step - decay_start) / final_decay_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor_ratio * cosine

    return lr_lambda


def build_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    learning_rate: float,
    total_steps: int,
    warmup_ratio: float,
    warmup_steps: int,
    lr_floor: float,
    lr_floor_step: int,
    lr_final_decay_steps: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler, int]:
    resolved_warmup_steps = resolve_warmup_steps(total_steps, warmup_ratio, warmup_steps)
    if scheduler_type == "cosine_to_floor":
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive for cosine_to_floor.")
        lr_lambda = cosine_to_floor_lambda(
            total_steps=total_steps,
            warmup_steps=resolved_warmup_steps,
            floor_step=lr_floor_step,
            final_decay_steps=lr_final_decay_steps,
            floor_ratio=lr_floor / learning_rate,
        )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), resolved_warmup_steps

    scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=resolved_warmup_steps,
        num_training_steps=total_steps,
    )
    return scheduler, resolved_warmup_steps

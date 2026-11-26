"""Loss and Generated-LoRA magnitude contracts for S5 HyperNet training."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


IGNORE_INDEX = -100


def lora_gate_warmup_steps(total_steps: int, warmup_ratio: float) -> int:
    """Return the deterministic gate horizon recorded in the run manifest."""

    if int(total_steps) <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < float(warmup_ratio) < 1.0:
        raise ValueError("gate warmup ratio must be in (0,1)")
    return max(1, int(round(float(total_steps) * float(warmup_ratio))))


def lora_gate_at_step(
    step: int,
    *,
    total_steps: int,
    gate_max: float,
    warmup_ratio: float,
) -> float:
    """Linearly warm the shared Generated-LoRA gate from zero to ``gate_max``."""

    if int(step) < 0:
        raise ValueError("gate step must be non-negative")
    if not math.isfinite(float(gate_max)) or not 0.0 < float(gate_max) <= 1.0:
        raise ValueError("gate_max must be finite and in (0,1]")
    warmup = lora_gate_warmup_steps(total_steps, warmup_ratio)
    progress = min(float(step) / float(warmup), 1.0)
    return float(gate_max) * progress


def scale_generated_lora_state(
    state: Mapping[str, Tensor], multiplier: float
) -> dict[str, Tensor]:
    """Multiply only the effective LoRA scale while preserving A/B directions."""

    factor = float(multiplier)
    if not math.isfinite(factor) or factor < 0.0:
        raise ValueError("Generated-LoRA multiplier must be finite and non-negative")
    if "scale" not in state:
        raise ValueError("Generated-LoRA state has no scale")
    result = dict(state)
    result["scale"] = state["scale"] * factor
    return result


def causal_loss_stats_per_example(
    logits: Tensor, labels: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Return FP32 next-token loss sum, token mean, and count for every row."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("logits and labels sequence shapes differ")
    shifted_logits = logits[:, :-1].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(IGNORE_INDEX)
    token_count = valid.sum(dim=1)
    if bool((token_count <= 0).any()):
        raise ValueError("every event must contain at least one supervised token")
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape_as(shifted_labels)
    loss_sum = (token_loss * valid).sum(dim=1)
    mean = loss_sum / token_count.to(dtype=loss_sum.dtype)
    if not bool(torch.isfinite(mean).all()):
        raise FloatingPointError("per-event causal CE is non-finite")
    return loss_sum, mean, token_count


def base_preservation_hinge(adapted_ce: Tensor, frozen_base_ce: Tensor) -> Tensor:
    """One-sided per-event penalty; improvements never cancel regressions."""

    if adapted_ce.ndim != 1 or frozen_base_ce.shape != adapted_ce.shape:
        raise ValueError("adapted and frozen Base CE must have shape [B]")
    base = frozen_base_ce.to(device=adapted_ce.device, dtype=adapted_ce.dtype)
    if not bool(torch.isfinite(base).all()):
        raise ValueError("Frozen Base CE contains non-finite values")
    return (adapted_ce - base).clamp_min(0.0)


def effective_delta_l2_per_example(state: Mapping[str, Tensor]) -> Tensor:
    """Mean normalized ``||scale * B @ A||_F^2`` over Generated-LoRA targets.

    The low-rank Gram identity avoids materializing dense updates and makes
    the penalty invariant to reciprocal A/B factor rescaling.
    """

    if "scale" not in state:
        raise ValueError("Generated-LoRA state has no scale")
    scale = torch.as_tensor(state["scale"]).float()
    if scale.numel() != 1:
        raise ValueError("Generated-LoRA scale must be scalar")
    terms: list[Tensor] = []
    batch_size: int | None = None
    for key, lora_a in state.items():
        if not key.endswith(".lora_A"):
            continue
        lora_b = state.get(f"{key.removesuffix('.lora_A')}.lora_B")
        if lora_b is None:
            raise ValueError(f"missing B factor for {key}")
        if lora_a.ndim != 3 or lora_b.ndim != 3:
            raise ValueError("Generated-LoRA factors must be batched rank-3 tensors")
        if lora_a.shape[0] != lora_b.shape[0] or lora_a.shape[1] != lora_b.shape[2]:
            raise ValueError("Generated-LoRA A/B shapes are incompatible")
        if batch_size is None:
            batch_size = int(lora_a.shape[0])
        elif batch_size != int(lora_a.shape[0]):
            raise ValueError("Generated-LoRA factor batches differ")
        a = lora_a.float()
        b = lora_b.float()
        gram_a = torch.bmm(a, a.transpose(1, 2))
        gram_b = torch.bmm(b.transpose(1, 2), b)
        squared = (gram_a * gram_b).sum(dim=(1, 2)).clamp_min(0.0)
        normalized = squared * scale.to(device=squared.device).square()
        normalized = normalized / float(lora_a.shape[-1] * lora_b.shape[-2])
        terms.append(normalized)
    if not terms:
        raise ValueError("Generated-LoRA state contains no A/B pairs")
    result = torch.stack(terms, dim=0).mean(dim=0)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("effective-Delta-W L2 is non-finite")
    return result


def scale_microbatch_objective_for_ddp(
    *,
    loss_sum: Tensor,
    hinge_sum: Tensor,
    effective_delta_sum: Tensor,
    world_size: int,
    global_token_count: Tensor | int,
    global_event_count: int,
    base_preservation_coefficient: float,
    effective_delta_coefficient: float,
) -> Tensor:
    """Scale one microbatch so DDP averaging recovers the global objective."""

    if int(world_size) <= 0 or int(global_event_count) <= 0:
        raise ValueError("world_size and global_event_count must be positive")
    tokens = torch.as_tensor(
        global_token_count, device=loss_sum.device, dtype=loss_sum.dtype
    )
    if tokens.numel() != 1 or float(tokens.item()) <= 0:
        raise ValueError("global_token_count must be a positive scalar")
    event_scale = float(world_size) / float(global_event_count)
    return (
        loss_sum * float(world_size) / tokens
        + float(base_preservation_coefficient) * hinge_sum * event_scale
        + float(effective_delta_coefficient) * effective_delta_sum * event_scale
    )


__all__ = [
    "IGNORE_INDEX",
    "base_preservation_hinge",
    "causal_loss_stats_per_example",
    "effective_delta_l2_per_example",
    "lora_gate_at_step",
    "lora_gate_warmup_steps",
    "scale_generated_lora_state",
    "scale_microbatch_objective_for_ddp",
]

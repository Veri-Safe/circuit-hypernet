from __future__ import annotations

import torch
from torch import Tensor
from typing import Mapping


IGNORE_INDEX = -100


def causal_loss_sum(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return response-only next-token CE numerator and supervised token count."""

    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected logits [B,T,V] and labels [B,T]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("Logit and label sequence shapes do not match")
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    token_count = shift_labels.ne(IGNORE_INDEX).sum()
    if int(token_count.item()) <= 0:
        raise ValueError("Prediction event has no supervised next-token targets")
    loss_sum = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum, token_count


def ddp_scaled_token_loss(
    local_loss_sum: Tensor,
    global_token_count: Tensor | int,
    *,
    world_size: int,
) -> Tensor:
    """Scale one local loss numerator for DDP's gradient averaging.

    Summing this value over local turn chunks and allowing DDP to average
    gradients yields global_loss_sum / global_token_count.
    """

    if world_size <= 0:
        raise ValueError("world_size must be positive")
    denominator = torch.as_tensor(
        global_token_count,
        device=local_loss_sum.device,
        dtype=local_loss_sum.dtype,
    )
    if float(denominator.item()) <= 0:
        raise ValueError("global_token_count must be positive")
    return local_loss_sum * float(world_size) / denominator


def distributed_token_count(local_token_count: int, device: torch.device) -> Tensor:
    """All-reduce a precomputed optimizer-window token count."""

    count = torch.tensor(local_token_count, device=device, dtype=torch.long)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
    return count


def generated_lora_factor_l2_per_query(state: Mapping[str, Tensor]) -> Tensor:
    """Return the mean squared A/B factor magnitude for each generated LoRA.

    The mean is taken over every generated factor scalar for one query, so the
    regularizer remains comparable across target modules with different shapes.
    Scalar control entries such as ``scale`` are deliberately excluded.
    """

    factors = tuple(
        value
        for key, value in state.items()
        if key.endswith((".lora_A", ".lora_B"))
    )
    if not factors:
        raise ValueError("Generated LoRA state contains no A/B factors")
    batch_size = factors[0].shape[0]
    if batch_size <= 0 or any(value.dim() < 2 for value in factors):
        raise ValueError("Generated LoRA factors need a non-empty batch dimension")
    if any(value.shape[0] != batch_size for value in factors):
        raise ValueError("Generated LoRA factors have inconsistent batch sizes")
    squared_sum = torch.zeros(
        batch_size,
        device=factors[0].device,
        dtype=torch.float32,
    )
    scalar_count = 0
    for value in factors:
        flattened = value.float().reshape(batch_size, -1)
        squared_sum = squared_sum + flattened.square().sum(dim=1)
        scalar_count += flattened.shape[1]
    if scalar_count <= 0:
        raise ValueError("Generated LoRA factors contain no scalars")
    return squared_sum / float(scalar_count)

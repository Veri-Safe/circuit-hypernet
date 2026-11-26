from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .contracts import VECTOR_DIM


GLOBAL_SLICE = slice(0, 11)
LAYER_SLICE = slice(11, 47)
TOPOLOGY_SLICE = slice(47, 59)


@dataclass(frozen=True)
class Circuit59Parts:
    global_values: Tensor
    layer_values: Tensor
    topology_values: Tensor
    layer_mask: Tensor
    circuit_present: Tensor


def validate_raw_vector59(vectors: Tensor) -> None:
    if vectors.ndim != 2 or vectors.shape[1] != VECTOR_DIM:
        raise ValueError(f"Circuit vectors must have shape [B, {VECTOR_DIM}]")
    if vectors.dtype != torch.float32:
        raise ValueError("Circuit vectors must remain raw FP32")
    if not torch.isfinite(vectors).all():
        raise ValueError("Circuit vectors contain non-finite values")


def split_vector59(
    vectors: Tensor,
    *,
    circuit_present: Tensor | None = None,
) -> Circuit59Parts:
    validate_raw_vector59(vectors)
    batch_size = vectors.shape[0]
    if circuit_present is None:
        present = torch.ones(batch_size, device=vectors.device, dtype=torch.bool)
    else:
        if circuit_present.shape != (batch_size,):
            raise ValueError("circuit_present must have shape [B]")
        present = circuit_present.to(device=vectors.device, dtype=torch.bool)
    return Circuit59Parts(
        global_values=vectors[:, GLOBAL_SLICE],
        layer_values=vectors[:, LAYER_SLICE].unsqueeze(-1),
        topology_values=vectors[:, TOPOLOGY_SLICE],
        layer_mask=present[:, None].expand(-1, 36),
        circuit_present=present,
    )

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ObjectiveResult:
    loss: Tensor
    reconstruction: Tensor
    sparsity: Tensor
    token_count: int
    l0_mean: Tensor


def transcoder_objective(
    y_hat: Tensor,
    y: Tensor,
    features: Tensor,
    decoder_weight: Tensor,
    *,
    sparsity_coeff: float = 8.0,
    c_coeff: float = 4.0,
) -> ObjectiveResult:
    if y_hat.shape != y.shape or y_hat.ndim != 2:
        raise ValueError("y_hat and y must be equal [tokens, d_model] tensors")
    if features.ndim != 2 or features.shape[0] != y.shape[0]:
        raise ValueError("features must align with the token dimension")
    if decoder_weight.shape != (features.shape[1], y.shape[1]):
        raise ValueError("decoder weight does not match feature/output dimensions")
    if y.shape[0] == 0:
        raise ValueError("Objective received no valid tokens")

    reconstruction_per_token = (y_hat.float() - y.float()).square().sum(dim=-1)
    decoder_norms = torch.linalg.vector_norm(decoder_weight.float(), dim=1)
    sparsity_per_token = float(sparsity_coeff) * torch.tanh(
        float(c_coeff) * features.float() * decoder_norms.unsqueeze(0)
    ).sum(dim=-1)
    reconstruction = reconstruction_per_token.mean()
    sparsity = sparsity_per_token.mean()
    return ObjectiveResult(
        loss=reconstruction + sparsity,
        reconstruction=reconstruction,
        sparsity=sparsity,
        token_count=int(y.shape[0]),
        l0_mean=features.gt(0).sum(dim=-1).float().mean(),
    )

"""Rectangular CodeCircuit-compatible ReLU Transcoders for S5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TranscoderObjective:
    loss: Tensor
    reconstruction: Tensor
    sparsity: Tensor
    l0_mean: Tensor
    token_count: int


class RectangularReLUTranscoder(nn.Module):
    """Map an S5 2080D MLP input to a sparse reconstruction of its 2048D output."""

    def __init__(self, d_in: int, d_out: int, d_feature: int) -> None:
        super().__init__()
        if min(d_in, d_out, d_feature) <= 0:
            raise ValueError("Transcoder dimensions must be positive")
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.d_feature = int(d_feature)
        self.W_enc = nn.Parameter(torch.empty(self.d_feature, self.d_in))
        self.b_enc = nn.Parameter(torch.empty(self.d_feature))
        self.W_dec = nn.Parameter(torch.empty(self.d_feature, self.d_out))
        self.b_dec = nn.Parameter(torch.empty(self.d_out))

    def reset_parameters(self, *, seed: int) -> None:
        devices = [self.W_enc.device] if self.W_enc.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(int(seed))
            nn.init.kaiming_uniform_(self.W_enc, mode="fan_in", nonlinearity="linear")
            self.W_enc.data.div_(
                self.W_enc.data.norm(dim=1, keepdim=True).clamp_min(1e-12)
            )
            nn.init.kaiming_uniform_(self.W_dec, mode="fan_in", nonlinearity="linear")
            self.W_dec.data.div_(
                self.W_dec.data.norm(dim=1, keepdim=True).clamp_min(1e-12)
            )
            self.W_dec.data.mul_(0.01)
            nn.init.uniform_(self.b_enc, a=-0.5, b=0.0)
            nn.init.uniform_(self.b_dec, a=-0.01, b=0.01)

    def encode(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.d_in:
            raise ValueError(f"expected input width {self.d_in}, got {x.shape[-1]}")
        return torch.relu(F.linear(x, self.W_enc, self.b_enc))

    def decode(self, features: Tensor) -> Tensor:
        if features.shape[-1] != self.d_feature:
            raise ValueError(
                f"expected feature width {self.d_feature}, got {features.shape[-1]}"
            )
        return features @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encode(x)
        return self.decode(features), features

    @classmethod
    def from_state_dict(
        cls, state: Mapping[str, Tensor]
    ) -> "RectangularReLUTranscoder":
        d_feature, d_in = state["W_enc"].shape
        d_feature_decoder, d_out = state["W_dec"].shape
        if d_feature_decoder != d_feature:
            raise ValueError("encoder and decoder feature dimensions differ")
        model = cls(d_in=d_in, d_out=d_out, d_feature=d_feature)
        model.load_state_dict(dict(state), strict=True)
        return model


def transcoder_objective(
    y_hat: Tensor,
    y: Tensor,
    features: Tensor,
    decoder_weight: Tensor,
    *,
    sparsity_coeff: float = 8.0,
    c_coeff: float = 4.0,
) -> TranscoderObjective:
    if y_hat.shape != y.shape or y.ndim != 2:
        raise ValueError("y_hat and y must be equal [tokens, d_out] tensors")
    if features.ndim != 2 or features.shape[0] != y.shape[0]:
        raise ValueError("features must align with the token dimension")
    if decoder_weight.shape != (features.shape[1], y.shape[1]):
        raise ValueError("decoder weight does not align with features and outputs")
    if y.shape[0] == 0:
        raise ValueError("Transcoder objective received no tokens")
    error = (y_hat.float() - y.float()).square().sum(dim=-1)
    decoder_norms = torch.linalg.vector_norm(decoder_weight.float(), dim=1)
    sparse = float(sparsity_coeff) * torch.tanh(
        float(c_coeff) * features.float() * decoder_norms.unsqueeze(0)
    ).sum(dim=-1)
    return TranscoderObjective(
        loss=error.mean() + sparse.mean(),
        reconstruction=error.mean(),
        sparsity=sparse.mean(),
        l0_mean=features.gt(0).sum(dim=-1).float().mean(),
        token_count=int(y.shape[0]),
    )


def canonicalize_transcoder(
    transcoder: RectangularReLUTranscoder, *, min_norm: float = 1e-8
) -> tuple[RectangularReLUTranscoder, Tensor]:
    with torch.no_grad():
        norms = torch.linalg.vector_norm(transcoder.W_dec.float(), dim=1)
        invalid = ~torch.isfinite(norms) | norms.le(float(min_norm))
        if invalid.any():
            ids = invalid.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"cannot canonicalize decoder rows: {ids[:20]}")
        output = RectangularReLUTranscoder(
            transcoder.d_in, transcoder.d_out, transcoder.d_feature
        ).to(device=transcoder.W_enc.device, dtype=transcoder.W_enc.dtype)
        output.W_enc.copy_(transcoder.W_enc * norms[:, None])
        output.b_enc.copy_(transcoder.b_enc * norms)
        output.W_dec.copy_(transcoder.W_dec / norms[:, None])
        output.b_dec.copy_(transcoder.b_dec)
    return output, norms

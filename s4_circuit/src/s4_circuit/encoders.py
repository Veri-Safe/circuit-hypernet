from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .representations import Circuit59Parts, split_vector59


def _zero_final_projection(module: nn.Sequential) -> None:
    final = module[-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Circuit encoder must end in a Linear projection")
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


def _mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    *,
    zero_final: bool = True,
) -> nn.Sequential:
    module = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, output_dim),
    )
    if zero_final:
        _zero_final_projection(module)
    return module


@dataclass(frozen=True)
class EncodedCircuit:
    global_context: Tensor
    topology_context: Tensor
    layer_context: Tensor
    layer_mask: Tensor
    circuit_present: Tensor


class SingleCircuitEncoder(nn.Module):
    def __init__(self, *, hidden_dim: int = 128, output_dim: int = 1_280) -> None:
        super().__init__()
        self.network = _mlp(59, hidden_dim, output_dim)

    def forward(self, vectors: Tensor) -> Tensor:
        return self.network(vectors.to(dtype=torch.float32))


class StructuredCircuitEncoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        output_dim: int = 1_280,
        zero_memory_outputs: bool = True,
        input_mode: str = "raw_v1",
    ) -> None:
        super().__init__()
        if input_mode not in {
            "raw_v1",
            "dual_raw_signed_log_layernorm_v1",
        }:
            raise ValueError(f"Unsupported structured circuit input mode: {input_mode}")
        self.input_mode = input_mode
        self.global_encoder = _mlp(
            11,
            hidden_dim,
            output_dim,
            zero_final=zero_memory_outputs,
        )
        self.topology_encoder = _mlp(
            12,
            hidden_dim,
            output_dim,
            zero_final=zero_memory_outputs,
        )
        self.layer_encoder = _mlp(1, hidden_dim, output_dim)
        if input_mode == "dual_raw_signed_log_layernorm_v1":
            self.normalized_global_encoder: nn.Sequential | None = _mlp(
                11, hidden_dim, output_dim, zero_final=True
            )
            self.normalized_topology_encoder: nn.Sequential | None = _mlp(
                12, hidden_dim, output_dim, zero_final=True
            )
            self.normalized_layer_encoder: nn.Sequential | None = _mlp(
                1, hidden_dim, output_dim, zero_final=True
            )
        else:
            self.normalized_global_encoder = None
            self.normalized_topology_encoder = None
            self.normalized_layer_encoder = None

    @staticmethod
    def _signed_log_layer_norm(values: Tensor) -> Tensor:
        transformed = torch.sign(values) * torch.log1p(torch.abs(values))
        return F.layer_norm(transformed, (transformed.shape[-1],))

    def forward(
        self,
        vectors: Tensor,
        *,
        circuit_present: Tensor | None = None,
    ) -> EncodedCircuit:
        parts: Circuit59Parts = split_vector59(
            vectors,
            circuit_present=circuit_present,
        )
        present = parts.circuit_present.to(dtype=torch.float32)[:, None]
        layer_present = parts.layer_mask.to(dtype=torch.float32)[:, :, None]
        global_context = self.global_encoder(parts.global_values)
        topology_context = self.topology_encoder(parts.topology_values)
        layer_context = self.layer_encoder(parts.layer_values)
        if self.normalized_global_encoder is not None:
            assert self.normalized_topology_encoder is not None
            assert self.normalized_layer_encoder is not None
            normalized_global = self._signed_log_layer_norm(parts.global_values)
            normalized_topology = self._signed_log_layer_norm(parts.topology_values)
            layer_values = parts.layer_values.squeeze(-1)
            normalized_layers = self._signed_log_layer_norm(layer_values).unsqueeze(-1)
            global_context = global_context + self.normalized_global_encoder(
                normalized_global
            )
            topology_context = topology_context + self.normalized_topology_encoder(
                normalized_topology
            )
            layer_context = layer_context + self.normalized_layer_encoder(
                normalized_layers
            )
        return EncodedCircuit(
            global_context=global_context * present,
            topology_context=topology_context * present,
            layer_context=layer_context * layer_present,
            layer_mask=parts.layer_mask,
            circuit_present=parts.circuit_present,
        )

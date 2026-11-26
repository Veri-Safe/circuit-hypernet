from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor, nn


_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.")
_NUM_SLOTS = 37
_LM_HEAD_SLOT = 36


def validate_lora_scale_multiplier(value: float) -> float:
    multiplier = float(value)
    if not math.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("LoRA scale multiplier must be finite and non-negative")
    return multiplier


def scale_generated_lora_state(
    state: Mapping[str, Tensor], multiplier: float
) -> dict[str, Tensor]:
    """Scale only the effective LoRA update, without changing its A/B direction."""

    factor = validate_lora_scale_multiplier(multiplier)
    if "scale" not in state:
        raise ValueError("Generated LoRA state has no scale entry")
    result = dict(state)
    result["scale"] = state["scale"] * factor
    return result


def generated_lora_effective_delta_l2_per_query(
    state: Mapping[str, Tensor],
) -> Tensor:
    """Mean normalized ||scale * B @ A||_F^2 per generated query.

    This is invariant to reciprocal rescaling of the two LoRA factors and uses
    the low-rank Gram identity, so no full delta-weight matrix is materialized.
    """

    if "scale" not in state:
        raise ValueError("Generated LoRA state has no scale entry")
    terms: list[Tensor] = []
    batch_size: int | None = None
    scale = state["scale"].float()
    for key, lora_a in state.items():
        if not key.endswith(".lora_A"):
            continue
        lora_b = state.get(f"{key.removesuffix('.lora_A')}.lora_B")
        if lora_b is None:
            raise ValueError(f"Generated LoRA state is missing B for {key}")
        if lora_a.ndim != 3 or lora_b.ndim != 3:
            raise ValueError("Generated LoRA A/B factors must be rank-3")
        if lora_a.shape[0] != lora_b.shape[0] or lora_a.shape[1] != lora_b.shape[2]:
            raise ValueError("Generated LoRA A/B factors are incompatible")
        if batch_size is None:
            batch_size = int(lora_a.shape[0])
        elif batch_size != int(lora_a.shape[0]):
            raise ValueError("Generated LoRA factors have inconsistent batches")
        a = lora_a.float()
        b = lora_b.float()
        gram_a = torch.bmm(a, a.transpose(1, 2))
        gram_b = torch.bmm(b.transpose(1, 2), b)
        norm_squared = (gram_a * gram_b).sum(dim=(1, 2)).clamp_min(0.0)
        normalized = norm_squared * scale.square()
        normalized = normalized / float(lora_a.shape[-1] * lora_b.shape[-2])
        terms.append(normalized)
    if not terms:
        raise ValueError("Generated LoRA state contains no A/B pairs")
    return torch.stack(terms, dim=0).mean(dim=0)


def exact_batched_delta_frobenius(
    lora_a: Tensor,
    lora_b: Tensor,
    *,
    scale: Tensor | float,
) -> Tensor:
    """Compute ||scale * B @ A||_F exactly without materializing B @ A."""

    if lora_a.ndim != 3 or lora_b.ndim != 3:
        raise ValueError("LoRA A/B factors must have shape [batch, ..., ...]")
    if lora_a.shape[0] != lora_b.shape[0] or lora_a.shape[1] != lora_b.shape[2]:
        raise ValueError("LoRA A/B factors have incompatible batch or rank dimensions")
    a = lora_a.float()
    b = lora_b.float()
    gram_a = torch.bmm(a, a.transpose(1, 2))
    gram_b = torch.bmm(b.transpose(1, 2), b)
    squared = (gram_a * gram_b).sum(dim=(1, 2)).clamp_min(0.0)
    scale_tensor = torch.as_tensor(scale, device=squared.device, dtype=squared.dtype)
    return squared.sqrt() * scale_tensor.abs()


def _slot_for_target(target_key: str) -> int:
    if target_key == "lm_head":
        return _LM_HEAD_SLOT
    match = _LAYER_KEY.match(target_key)
    if match is None:
        raise ValueError(f"Unsupported Generated LoRA target key: {target_key}")
    layer = int(match.group(1))
    if not 0 <= layer < _LM_HEAD_SLOT:
        raise ValueError(f"Generated LoRA layer is outside 0..35: {target_key}")
    return layer


@dataclass
class _MomentTensors:
    total: Tensor
    squared_total: Tensor
    maximum: Tensor


class GeneratedLoRANormAccumulator:
    """Aggregate per-layer factor and exact materialized-delta norms."""

    def __init__(
        self,
        model: nn.Module,
        *,
        target_keys: Sequence[str],
        device: torch.device,
        max_queries: int | None = None,
    ) -> None:
        self.target_keys = tuple(target_keys)
        if not self.target_keys:
            raise ValueError("LoRA norm diagnostics require non-empty target keys")
        self.device = device
        if max_queries is not None and max_queries <= 0:
            raise ValueError("LoRA norm max_queries must be positive")
        self.max_queries = max_queries
        self.local_query_count = 0
        self.base_squared = torch.zeros(_NUM_SLOTS, device=device, dtype=torch.float32)
        for target_key in self.target_keys:
            module = model.get_submodule(target_key)
            weight = getattr(module, "weight", None)
            if not isinstance(weight, Tensor):
                raise TypeError(
                    f"Generated LoRA target has no weight tensor: {target_key}"
                )
            slot = _slot_for_target(target_key)
            base_norm = torch.linalg.vector_norm(weight.detach().float())
            self.base_squared[slot] += base_norm.square()
        if torch.any(self.base_squared <= 0):
            missing = torch.nonzero(self.base_squared <= 0).flatten().tolist()
            raise ValueError(
                f"Base-weight norm diagnostics have empty slots: {missing}"
            )
        self.query_count = torch.zeros((), device=device, dtype=torch.float32)
        self.a = self._empty_moments()
        self.b = self._empty_moments()
        self.raw_delta = self._empty_moments()
        self.raw_relative_delta = self._empty_moments()

    def _empty_moments(self) -> _MomentTensors:
        zeros = torch.zeros(_NUM_SLOTS, device=self.device, dtype=torch.float32)
        return _MomentTensors(
            total=zeros.clone(),
            squared_total=zeros.clone(),
            maximum=zeros.clone(),
        )

    @staticmethod
    def _update_moments(target: _MomentTensors, values: Tensor) -> None:
        target.total += values.sum(dim=0)
        target.squared_total += values.square().sum(dim=0)
        target.maximum.copy_(torch.maximum(target.maximum, values.max(dim=0).values))

    def update(self, state: Mapping[str, Tensor]) -> None:
        if "scale" not in state:
            raise ValueError("Generated LoRA state has no scale entry")
        first_key = f"{self.target_keys[0]}.lora_A"
        raw_batch_size = int(state[first_key].shape[0])
        batch_size = raw_batch_size
        if self.max_queries is not None:
            remaining = self.max_queries - self.local_query_count
            if remaining <= 0:
                return
            batch_size = min(batch_size, remaining)
        a_squared = torch.zeros(
            batch_size, _NUM_SLOTS, device=self.device, dtype=torch.float32
        )
        b_squared = torch.zeros_like(a_squared)
        raw_delta_squared = torch.zeros_like(a_squared)
        raw_scale = state["scale"]
        for target_key in self.target_keys:
            slot = _slot_for_target(target_key)
            a = state[f"{target_key}.lora_A"]
            b = state[f"{target_key}.lora_B"]
            if a.shape[0] != raw_batch_size or b.shape[0] != raw_batch_size:
                raise ValueError("Generated LoRA diagnostic batch dimensions changed")
            a = a[:batch_size]
            b = b[:batch_size]
            a_norm = torch.linalg.vector_norm(a.float(), dim=(1, 2))
            b_norm = torch.linalg.vector_norm(b.float(), dim=(1, 2))
            a_squared[:, slot] += a_norm.square()
            b_squared[:, slot] += b_norm.square()
            delta = exact_batched_delta_frobenius(a, b, scale=raw_scale)
            raw_delta_squared[:, slot] += delta.square()
        a_norm = a_squared.sqrt()
        b_norm = b_squared.sqrt()
        raw_delta_norm = raw_delta_squared.sqrt()
        raw_relative = raw_delta_norm / self.base_squared.sqrt().unsqueeze(0)
        self.query_count += float(batch_size)
        self.local_query_count += batch_size
        self._update_moments(self.a, a_norm)
        self._update_moments(self.b, b_norm)
        self._update_moments(self.raw_delta, raw_delta_norm)
        self._update_moments(self.raw_relative_delta, raw_relative)

    @staticmethod
    def _all_reduce(value: Tensor, *, op: dist.ReduceOp) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        backend = dist.get_backend()
        backend_name = getattr(backend, "value", backend)
        if str(backend_name).lower() == "gloo" and value.device.type != "cpu":
            # Gloo only accepts host tensors.  S4's evaluation control group is
            # deliberately Gloo, while the diagnostic accumulators live on the
            # model device; the vectors are tiny, so this copy is negligible.
            host_value = value.detach().cpu()
            dist.all_reduce(host_value, op=op)
            with torch.no_grad():
                value.copy_(host_value)
            return
        dist.all_reduce(value, op=op)

    @classmethod
    def _reduce_moments(cls, value: _MomentTensors) -> None:
        cls._all_reduce(value.total, op=dist.ReduceOp.SUM)
        cls._all_reduce(value.squared_total, op=dist.ReduceOp.SUM)
        cls._all_reduce(value.maximum, op=dist.ReduceOp.MAX)

    @staticmethod
    def _summary(value: _MomentTensors, count: float, slot: int) -> dict[str, float]:
        return {
            "mean": float(value.total[slot].item() / count),
            "rms": float(math.sqrt(value.squared_total[slot].item() / count)),
            "max": float(value.maximum[slot].item()),
        }

    def finalize(self, *, lora_scale_multiplier: float) -> dict[str, Any]:
        multiplier = validate_lora_scale_multiplier(lora_scale_multiplier)
        self._all_reduce(self.query_count, op=dist.ReduceOp.SUM)
        for value in (self.a, self.b, self.raw_delta, self.raw_relative_delta):
            self._reduce_moments(value)
        count = float(self.query_count.item())
        if count <= 0:
            raise ValueError("LoRA norm diagnostics received no generated queries")
        slots = []
        for slot in range(_NUM_SLOTS):
            raw_delta = self._summary(self.raw_delta, count, slot)
            raw_relative = self._summary(self.raw_relative_delta, count, slot)
            slots.append(
                {
                    "slot": slot,
                    "kind": "layer" if slot < _LM_HEAD_SLOT else "lm_head",
                    "base_weight_frobenius": float(
                        self.base_squared[slot].sqrt().item()
                    ),
                    "lora_a_frobenius": self._summary(self.a, count, slot),
                    "lora_b_frobenius": self._summary(self.b, count, slot),
                    "raw_delta_frobenius": raw_delta,
                    "raw_relative_delta": raw_relative,
                    "effective_delta_frobenius": {
                        key: value * multiplier for key, value in raw_delta.items()
                    },
                    "effective_relative_delta": {
                        key: value * multiplier for key, value in raw_relative.items()
                    },
                }
            )
        return {
            "format": "nvarc_s4_generated_lora_norm_diagnostics_v1",
            "status": "complete",
            "query_count": int(count),
            "max_queries_per_rank": self.max_queries,
            "lora_scale_multiplier": multiplier,
            "delta_norm_method": "exact_low_rank_gram_v1",
            "slots": slots,
        }

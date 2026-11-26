"""Target layout for the S5 Lean-Circuit Generated-LoRA protocol.

The S5 Base is a GPT-style model with positional features concatenated to the
token representation.  Consequently, ``attn.c_attn`` and ``mlp.c_fc`` really
consume 2080 features (2048 model features plus 32 positional features).  This
module keeps that fact in the serialized target contract instead of silently
projecting those inputs down to 2048.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True)
class TargetModuleSpec:
    """One linear Base module represented by a pair of LoRA factors."""

    name: str
    in_features: int
    out_features: int


S5_LEAN_CIRCUIT_MODULES = (
    TargetModuleSpec("attn.c_attn", 2080, 6144),
    TargetModuleSpec("attn.c_proj", 2048, 2048),
    TargetModuleSpec("mlp.c_fc", 2080, 8192),
    TargetModuleSpec("mlp.c_proj", 8192, 2048),
)

# Public aliases make the protocol map easy to consume from launchers and
# audits while keeping the canonical source as the typed module tuple above.
S5_TARGET_MODULES = S5_LEAN_CIRCUIT_MODULES
S5_TARGET_MAP: dict[str, dict[str, int]] = {
    module.name: {
        "lora_A_dim": module.in_features,
        "lora_B_dim": module.out_features,
    }
    for module in S5_LEAN_CIRCUIT_MODULES
}
S5_PG_MAPPING = S5_TARGET_MAP


@dataclass(frozen=True)
class S5TargetLayout:
    """Immutable, hashable description of the S5 Generated-LoRA targets."""

    rank: int = 4
    alpha: int = 8
    token_dim: int = 1024
    dim_accumulation: int = 4
    num_layers: int = 8
    prefix: str = "transformer.h."
    modules: tuple[TargetModuleSpec, ...] = S5_LEAN_CIRCUIT_MODULES

    def __post_init__(self) -> None:
        if min(self.rank, self.alpha, self.token_dim, self.dim_accumulation) <= 0:
            raise ValueError("LoRA and token layout dimensions must be positive")
        if self.rank % self.dim_accumulation:
            raise ValueError("rank must be divisible by dim_accumulation")
        if self.num_layers != 8:
            raise ValueError("S5 Lean-Circuit Base layout requires exactly 8 layers")
        if self.prefix != "transformer.h.":
            raise ValueError("S5 Lean-Circuit target prefix must be 'transformer.h.'")
        if self.modules != S5_LEAN_CIRCUIT_MODULES:
            raise ValueError("S5 Lean-Circuit target module layout is protocol-fixed")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.modules)

    @property
    def generator_slot_count(self) -> int:
        return self.num_layers

    @property
    def pg_mapping(self) -> dict[str, dict[str, int]]:
        return {
            module.name: {
                "lora_A_dim": module.in_features,
                "lora_B_dim": module.out_features,
            }
            for module in self.modules
        }

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(
            f"{self.prefix}{layer_index}.{module.name}"
            for layer_index in range(self.num_layers)
            for module in self.modules
        )

    @property
    def generated_scalar_count(self) -> int:
        per_layer = sum(
            self.rank * (module.in_features + module.out_features)
            for module in self.modules
        )
        return self.num_layers * per_layer

    def _token_groups(self, *, factor: str) -> int:
        token_area = self.rank * self.token_dim
        if factor == "A":
            sizes = (self.rank * module.in_features for module in self.modules)
        elif factor == "B":
            sizes = (self.rank * module.out_features for module in self.modules)
        else:
            raise ValueError(f"Unknown LoRA factor: {factor!r}")
        return sum(math.ceil(size / token_area) for size in sizes)

    @property
    def lora_a_token_groups_per_slot(self) -> int:
        return self._token_groups(factor="A")

    @property
    def lora_b_token_groups_per_slot(self) -> int:
        return self._token_groups(factor="B")

    @property
    def internal_parameter_tokens_per_slot(self) -> int:
        groups = self.lora_a_token_groups_per_slot + self.lora_b_token_groups_per_slot
        return groups * (self.rank // self.dim_accumulation)

    @property
    def internal_parameter_tokens_total(self) -> int:
        return self.generator_slot_count * self.internal_parameter_tokens_per_slot

    def manifest(self) -> dict[str, Any]:
        return {
            "format": "s5_lean_circuit_generated_lora_target_layout_v1",
            "runtime": "s5_upstream_gpt",
            "prefix": self.prefix,
            "num_layers": self.num_layers,
            "generator_slot_count": self.generator_slot_count,
            "rank": self.rank,
            "alpha": self.alpha,
            "scale": self.scale,
            "token_dim": self.token_dim,
            "dim_accumulation": self.dim_accumulation,
            "modules": [
                {
                    "name": module.name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
                for module in self.modules
            ],
            "generated_scalar_count": self.generated_scalar_count,
            "internal_parameter_tokens_total": self.internal_parameter_tokens_total,
        }

    @property
    def layout_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_model(self, model: nn.Module) -> list[dict[str, Any]]:
        """Fail closed unless every Base target has its exact linear shape."""

        expected = {module.name: module for module in self.modules}
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for target_key in self.target_keys:
            try:
                module = model.get_submodule(target_key)
            except AttributeError:
                errors.append(f"missing {target_key}")
                continue

            base_layer = getattr(module, "base_layer", module)
            in_features = int(getattr(base_layer, "in_features", -1))
            out_features = int(getattr(base_layer, "out_features", -1))
            layer_and_module = target_key[len(self.prefix) :]
            module_name = layer_and_module.split(".", 1)[1]
            spec = expected[module_name]
            ok = (
                in_features == spec.in_features
                and out_features == spec.out_features
            )
            rows.append(
                {
                    "key": target_key,
                    "in_features": in_features,
                    "out_features": out_features,
                    "expected_in_features": spec.in_features,
                    "expected_out_features": spec.out_features,
                    "ok": ok,
                }
            )
            if not ok:
                errors.append(
                    f"{target_key}: expected ({spec.in_features}, {spec.out_features}), "
                    f"found ({in_features}, {out_features})"
                )

        if errors:
            raise ValueError(
                "S5 Generated-LoRA target layout validation failed: "
                + "; ".join(errors[:10])
            )
        return rows


def build_s5_target_layout(
    *, rank: int = 4, alpha: int = 8, token_dim: int = 1024, dim_accumulation: int = 4
) -> S5TargetLayout:
    """Build the protocol-default S5 target layout."""

    return S5TargetLayout(
        rank=rank,
        alpha=alpha,
        token_dim=token_dim,
        dim_accumulation=dim_accumulation,
    )


DEFAULT_S5_TARGET_LAYOUT = build_s5_target_layout()


__all__ = [
    "DEFAULT_S5_TARGET_LAYOUT",
    "S5_LEAN_CIRCUIT_MODULES",
    "S5_PG_MAPPING",
    "S5_TARGET_MAP",
    "S5_TARGET_MODULES",
    "S5TargetLayout",
    "TargetModuleSpec",
    "build_s5_target_layout",
]

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True)
class TargetModuleSpec:
    name: str
    in_features: int
    out_features: int


QWEN3_4B_MODULES = (
    TargetModuleSpec("self_attn.q_proj", 2560, 4096),
    TargetModuleSpec("self_attn.k_proj", 2560, 1024),
    TargetModuleSpec("self_attn.v_proj", 2560, 1024),
    TargetModuleSpec("self_attn.o_proj", 4096, 2560),
    TargetModuleSpec("mlp.gate_proj", 2560, 9728),
    TargetModuleSpec("mlp.up_proj", 2560, 9728),
    TargetModuleSpec("mlp.down_proj", 9728, 2560),
)

NVARC_GLOBAL_MODULES = (TargetModuleSpec("lm_head", 2560, 16),)


@dataclass(frozen=True)
class QwenTargetLayout:
    rank: int
    alpha: int
    token_dim: int
    dim_accumulation: int
    num_layers: int = 36
    prefix: str = "model.layers."
    modules: tuple[TargetModuleSpec, ...] = QWEN3_4B_MODULES
    global_modules: tuple[TargetModuleSpec, ...] = NVARC_GLOBAL_MODULES

    def __post_init__(self) -> None:
        if min(self.rank, self.alpha, self.token_dim, self.dim_accumulation) <= 0:
            raise ValueError("LoRA and token layout dimensions must be positive")
        if self.rank % self.dim_accumulation:
            raise ValueError("rank must be divisible by dim_accumulation")
        if self.num_layers != 36:
            raise ValueError("NVARC Qwen3-4B HF layout requires exactly 36 layers")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.modules)

    @property
    def global_module_names(self) -> tuple[str, ...]:
        return tuple(module.name for module in self.global_modules)

    @property
    def generator_slot_count(self) -> int:
        return self.num_layers + len(self.global_modules)

    @property
    def all_generator_modules(self) -> tuple[TargetModuleSpec, ...]:
        return self.modules + self.global_modules

    @property
    def pg_mapping(self) -> dict[str, dict[str, int]]:
        return {
            module.name: {
                "lora_A_dim": module.in_features,
                "lora_B_dim": module.out_features,
            }
            for module in self.all_generator_modules
        }

    @property
    def target_keys(self) -> tuple[str, ...]:
        layer_keys = tuple(
            f"{self.prefix}{layer_index}.{module.name}"
            for layer_index in range(self.num_layers)
            for module in self.modules
        )
        return layer_keys + self.global_module_names

    @property
    def generated_scalar_count(self) -> int:
        per_layer = sum(
            self.rank * (module.in_features + module.out_features)
            for module in self.modules
        )
        global_count = sum(
            self.rank * (module.in_features + module.out_features)
            for module in self.global_modules
        )
        return self.num_layers * per_layer + global_count

    @property
    def emitted_scalar_count(self) -> int:
        per_slot = sum(
            self.rank * (module.in_features + module.out_features)
            for module in self.all_generator_modules
        )
        return self.generator_slot_count * per_slot

    def _token_groups(self, *, factor: str) -> int:
        token_area = self.rank * self.token_dim
        if factor == "A":
            sizes = (
                self.rank * module.in_features
                for module in self.all_generator_modules
            )
        elif factor == "B":
            sizes = (
                self.rank * module.out_features
                for module in self.all_generator_modules
            )
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
            "format": "nvarc_qwen3_4b_hf_target_layout_v2",
            "runtime": "hf_separated",
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
            "global_modules": [
                {
                    "name": module.name,
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
                for module in self.global_modules
            ],
            "generated_scalar_count": self.generated_scalar_count,
            "emitted_scalar_count": self.emitted_scalar_count,
            "internal_parameter_tokens_total": self.internal_parameter_tokens_total,
        }

    @property
    def layout_sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_model(self, model: nn.Module) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        for target_key in self.target_keys:
            try:
                module = model.get_submodule(target_key)
            except AttributeError:
                errors.append(f"missing {target_key}")
                continue
            in_features = int(getattr(module, "in_features", -1))
            out_features = int(getattr(module, "out_features", -1))
            short_name = target_key.split(".", 3)[-1]
            expected = next(
                item for item in self.all_generator_modules if item.name == short_name
            )
            ok = (
                in_features == expected.in_features
                and out_features == expected.out_features
            )
            rows.append(
                {
                    "key": target_key,
                    "in_features": in_features,
                    "out_features": out_features,
                    "ok": ok,
                }
            )
            if not ok:
                errors.append(
                    f"{target_key}: expected ({expected.in_features}, {expected.out_features}), "
                    f"found ({in_features}, {out_features})"
                )
        if errors:
            raise ValueError("Qwen target layout validation failed: " + "; ".join(errors[:10]))
        return rows


def build_qwen3_4b_target_layout(
    *, rank: int, alpha: int, token_dim: int, dim_accumulation: int
) -> QwenTargetLayout:
    return QwenTargetLayout(
        rank=rank,
        alpha=alpha,
        token_dim=token_dim,
        dim_accumulation=dim_accumulation,
    )

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from torch import nn
from transformers import PretrainedConfig


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


gemma2_s2_pg_mapping: dict[str, dict[str, int]] = {
    "self_attn.q_proj": {"lora_A_dim": 2304, "lora_B_dim": 2048},
    "self_attn.k_proj": {"lora_A_dim": 2304, "lora_B_dim": 1024},
    "self_attn.v_proj": {"lora_A_dim": 2304, "lora_B_dim": 1024},
    "self_attn.o_proj": {"lora_A_dim": 2048, "lora_B_dim": 2304},
    "mlp.gate_proj": {"lora_A_dim": 2304, "lora_B_dim": 9216},
    "mlp.up_proj": {"lora_A_dim": 2304, "lora_B_dim": 9216},
    "mlp.down_proj": {"lora_A_dim": 9216, "lora_B_dim": 2304},
}


class ParameterGeneratorConfig(PretrainedConfig):
    model_type = "s2_parameter_generator"

    def __init__(
        self,
        token_dim: int = 1024,
        rank: int = 16,
        alpha: int = 32,
        input_dim: int = 2304,
        dim_accumulation: int = 4,
        d_model: int = 2304,
        head_dim: int = 256,
        num_base_model_layers: int = 26,
        output_dim: int = 1024,
        num_pg_layers: int = 24,
        prefix: str = "model.layers.",
        pg_mapping: dict[str, dict[str, int]] | None = None,
        target_layer_indices: list[int] | None = None,
        near_zero_b_init_std: float = 0.0,
        condition_injection_mode: str = "none",
        condition_summary_scale_init: float = 0.01,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        if target_layer_indices is None:
            target_layer_indices = list(range(num_base_model_layers))
        else:
            target_layer_indices = [int(idx) for idx in target_layer_indices]
            num_base_model_layers = len(target_layer_indices)
        self.token_dim = token_dim
        self.rank = rank
        self.alpha = alpha
        self.input_dim = input_dim
        self.dim_accumulation = dim_accumulation
        self.d_model = d_model
        self.head_dim = head_dim
        self.num_base_model_layers = num_base_model_layers
        self.output_dim = output_dim
        self.num_pg_layers = num_pg_layers
        self.prefix = prefix
        self.pg_mapping = pg_mapping if pg_mapping is not None else dict(gemma2_s2_pg_mapping)
        self.target_layer_indices = target_layer_indices
        self.near_zero_b_init_std = near_zero_b_init_std
        self.condition_injection_mode = condition_injection_mode
        self.condition_summary_scale_init = float(condition_summary_scale_init)

        if not self.target_layer_indices:
            raise ValueError("target_layer_indices must not be empty.")
        if self.rank <= 0:
            raise ValueError("rank must be positive.")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")
        if self.dim_accumulation <= 0:
            raise ValueError("dim_accumulation must be positive.")
        if self.rank % self.dim_accumulation != 0:
            raise ValueError("rank must be divisible by dim_accumulation.")
        if self.d_model % self.dim_accumulation != 0:
            raise ValueError("d_model must be divisible by dim_accumulation.")
        if self.d_model % self.head_dim != 0:
            raise ValueError("d_model must be divisible by head_dim.")
        if self.condition_injection_mode not in {"none", "additive", "film"}:
            raise ValueError("condition_injection_mode must be one of: none, additive, film.")
        if not math.isfinite(float(self.condition_summary_scale_init)):
            raise ValueError("condition_summary_scale_init must be finite.")
        if self.condition_summary_scale_init < 0:
            raise ValueError("condition_summary_scale_init must be non-negative.")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def num_heads(self) -> int:
        return self.d_model // self.head_dim


@dataclass
class S2RunConfig:
    model_path: str
    metamathqa_path: str
    output_dir: str
    log_dir: str
    run_name: str
    static_adapter_path: str | None = None
    base_only: bool = True
    static_null_smoke: bool = False
    max_seq_length: int = 1536
    pg_rank: int = 16
    pg_alpha: int = 32
    pg_token_dim: int = 1024
    pg_num_layers: int = 24
    pg_dim_accumulation: int = 4

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_layer_pg_mapping(config: ParameterGeneratorConfig) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for layer_idx in config.target_layer_indices:
        for module_name in config.pg_mapping:
            key = f"{config.prefix}{layer_idx}.{module_name}"
            mapping[key] = key
    return mapping


def get_linear_io_features(module: nn.Module) -> tuple[int, int] | None:
    if hasattr(module, "in_features") and hasattr(module, "out_features"):
        return int(module.in_features), int(module.out_features)
    base_layer = getattr(module, "base_layer", None)
    if base_layer is not None and hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
        return int(base_layer.in_features), int(base_layer.out_features)
    return None


def find_module_by_suffix(model: nn.Module, suffix: str) -> tuple[str, nn.Module] | None:
    for name, module in model.named_modules():
        if name == suffix or name.endswith("." + suffix):
            return name, module
    return None


def validate_gemma_target_map(
    model: nn.Module,
    config: ParameterGeneratorConfig,
    *,
    require_all: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    mismatched: list[str] = []

    for layer_idx in config.target_layer_indices:
        for module_name, dims in config.pg_mapping.items():
            pg_key = f"{config.prefix}{layer_idx}.{module_name}"
            found = find_module_by_suffix(model, pg_key)
            if found is None:
                missing.append(pg_key)
                continue

            actual_name, module = found
            features = get_linear_io_features(module)
            if features is None:
                mismatched.append(f"{pg_key}: target has no linear in/out features")
                continue
            in_features, out_features = features
            expected_in = int(dims["lora_A_dim"])
            expected_out = int(dims["lora_B_dim"])
            ok = in_features == expected_in and out_features == expected_out
            if not ok:
                mismatched.append(
                    f"{pg_key}: expected ({expected_in}, {expected_out}), got ({in_features}, {out_features})"
                )
            rows.append(
                {
                    "pg_key": pg_key,
                    "actual_name": actual_name,
                    "in_features": in_features,
                    "out_features": out_features,
                    "expected_in_features": expected_in,
                    "expected_out_features": expected_out,
                    "ok": ok,
                }
            )

    if require_all and (missing or mismatched):
        detail = []
        if missing:
            detail.append("missing=" + ", ".join(missing[:10]))
        if mismatched:
            detail.append("mismatched=" + "; ".join(mismatched[:10]))
        raise ValueError("Gemma target map validation failed: " + " | ".join(detail))

    return rows

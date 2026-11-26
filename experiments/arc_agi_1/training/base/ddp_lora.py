#!/usr/bin/env python3
"""Small, dependency-light LoRA implementation for the NVARC DDP trainer."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F


TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
SHARED_A_GROUPS = (
    ("q_proj", "k_proj", "v_proj"),
    ("gate_proj", "up_proj"),
)


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable ``alpha / rank * B @ A`` update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        adapter_dtype: torch.dtype = torch.float32,
        shared_A: nn.Parameter | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        if shared_A is None:
            self.lora_A = nn.Parameter(
                torch.empty(
                    rank,
                    base_layer.in_features,
                    device=base_layer.weight.device,
                    dtype=adapter_dtype,
                )
            )
            initialize_A = True
        else:
            expected_shape = (rank, base_layer.in_features)
            if tuple(shared_A.shape) != expected_shape:
                raise ValueError(
                    f"Shared LoRA A has shape {tuple(shared_A.shape)}, "
                    f"expected {expected_shape}"
                )
            self.lora_A = shared_A
            initialize_A = False
        self.lora_B = nn.Parameter(
            torch.empty(
                base_layer.out_features,
                rank,
                device=base_layer.weight.device,
                dtype=adapter_dtype,
            )
        )
        self.reset_lora_parameters(initialize_A=initialize_A)

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def reset_lora_parameters(self, *, initialize_A: bool = True) -> None:
        # a=sqrt(5) gives U(-1/sqrt(d_in), 1/sqrt(d_in)), the standard PEFT init.
        if initialize_A:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(inputs)
        adapter_inputs = self.dropout(inputs)
        if (
            adapter_inputs.dtype != self.lora_A.dtype
            and not torch.is_autocast_enabled()
        ):
            adapter_inputs = adapter_inputs.to(self.lora_A.dtype)
        adapter_hidden = F.linear(adapter_inputs, self.lora_A)
        adapter_output = F.linear(adapter_hidden, self.lora_B)
        return base_output + adapter_output.to(base_output.dtype) * self.scaling


def _parent_and_child(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    target_modules: Iterable[str] = TARGET_MODULES,
    shared_A_groups: Iterable[Iterable[str]] = SHARED_A_GROUPS,
    adapter_dtype: torch.dtype = torch.float32,
) -> list[str]:
    """Freeze ``model`` and replace every requested linear projection with LoRA."""

    targets = frozenset(target_modules)
    if not targets:
        raise ValueError("target_modules must not be empty")
    model.requires_grad_(False)

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in targets:
            replacements.append((name, module))
    if not replacements:
        raise ValueError(f"No linear layers matched target modules {sorted(targets)}")

    group_by_target = {
        target: group_index
        for group_index, group in enumerate(shared_A_groups)
        for target in group
    }
    shared_parameters: dict[tuple[str, int], nn.Parameter] = {}
    replaced_names: list[str] = []
    for name, base_layer in replacements:
        target_name = name.rsplit(".", 1)[-1]
        shared_A = None
        if target_name in group_by_target:
            parent_scope = name.rsplit(".", 1)[0]
            group_key = (parent_scope, group_by_target[target_name])
            shared_A = shared_parameters.get(group_key)
            if shared_A is None:
                shared_A = nn.Parameter(
                    torch.empty(
                        rank,
                        base_layer.in_features,
                        device=base_layer.weight.device,
                        dtype=adapter_dtype,
                    )
                )
                nn.init.kaiming_uniform_(shared_A, a=math.sqrt(5))
                shared_parameters[group_key] = shared_A
        parent, child_name = _parent_and_child(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(
                base_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                adapter_dtype=adapter_dtype,
                shared_A=shared_A,
            ),
        )
        replaced_names.append(name)
    return replaced_names


def iter_lora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def select_lora_probe_modules(
    model: nn.Module,
    *,
    layer_indices: Iterable[int],
    target_modules: Iterable[str],
) -> dict[str, LoRALinear]:
    """Select an exact layer/target grid for sparse dynamics diagnostics."""

    layers = {int(layer) for layer in layer_indices}
    targets = {str(target) for target in target_modules}
    selected: dict[str, LoRALinear] = {}
    found_pairs: set[tuple[int, str]] = set()
    for name, module in iter_lora_modules(model):
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        target = name.rsplit(".", 1)[-1]
        if match is None or target not in targets:
            continue
        layer = int(match.group(1))
        if layer not in layers:
            continue
        selected[name] = module
        found_pairs.add((layer, target))
    expected_pairs = {(layer, target) for layer in layers for target in targets}
    if found_pairs != expected_pairs:
        missing = sorted(expected_pairs - found_pairs)
        extra = sorted(found_pairs - expected_pairs)
        raise ValueError(
            f"LoRA dynamics probe selection mismatch: missing={missing}, extra={extra}"
        )
    return selected


@torch.inference_mode()
def estimate_effective_update_statistics(
    module: LoRALinear,
    *,
    seed: int,
    trace_probes: int = 4,
    power_iterations: int = 6,
) -> dict[str, float]:
    """Estimate norms of ``(alpha / rank) * B @ A`` without materializing it."""

    if trace_probes <= 0 or power_iterations <= 0:
        raise ValueError("trace_probes and power_iterations must be positive")
    A = module.lora_A.detach().float()
    B = module.lora_B.detach().float()
    generator = torch.Generator(device=A.device)
    generator.manual_seed(int(seed))

    signs = torch.randint(
        0,
        2,
        (A.shape[1], trace_probes),
        generator=generator,
        device=A.device,
        dtype=torch.int64,
    ).to(dtype=A.dtype)
    signs.mul_(2).sub_(1)
    projected = B @ (A @ signs)
    frobenius_sq = float(
        projected.square().sum(dim=0).mean().item() * module.scaling**2
    )

    vector = torch.randn(
        A.shape[1], 1, generator=generator, device=A.device, dtype=A.dtype
    )
    vector.div_(vector.norm().clamp_min(torch.finfo(vector.dtype).tiny))
    for _ in range(power_iterations):
        left = B @ (A @ vector)
        left_norm = left.norm()
        if float(left_norm.item()) == 0.0:
            break
        left.div_(left_norm)
        vector = A.T @ (B.T @ left)
        vector_norm = vector.norm()
        if float(vector_norm.item()) == 0.0:
            break
        vector.div_(vector_norm)
    spectral = float((B @ (A @ vector)).norm().item() * abs(module.scaling))
    frobenius = math.sqrt(max(frobenius_sq, 0.0))
    stable_rank = frobenius_sq / spectral**2 if spectral > 0.0 else 0.0
    base_frobenius = float(module.base_layer.weight.detach().float().norm().item())
    return {
        "effective_update_frobenius_estimate": frobenius,
        "effective_update_spectral_estimate": spectral,
        "effective_update_stable_rank_estimate": stable_rank,
        "base_weight_frobenius": base_frobenius,
        "effective_update_to_base_frobenius": (
            frobenius / base_frobenius if base_frobenius > 0.0 else float("nan")
        ),
    }


def trainable_parameter_summary(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
    }


def adapter_state_dict(
    model: nn.Module, *, peft_compatible: bool = True
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in iter_lora_modules(model):
        prefix = f"base_model.model.{name}" if peft_compatible else name
        state[f"{prefix}.lora_A.weight"] = (
            module.lora_A.detach().cpu().contiguous().clone()
        )
        state[f"{prefix}.lora_B.weight"] = (
            module.lora_B.detach().cpu().contiguous().clone()
        )
    if not state:
        raise ValueError("Model has no LoRA modules")
    return state


def save_adapter(
    model: nn.Module,
    output_dir: Path,
    *,
    base_model_name_or_path: str,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Iterable[str] = TARGET_MODULES,
    shared_A_groups: Iterable[Iterable[str]] = SHARED_A_GROUPS,
) -> None:
    """Write a PEFT-shaped adapter without requiring PEFT at training time."""

    from safetensors.torch import save_file

    normalized_targets = tuple(target_modules)
    normalized_shared_groups = tuple(tuple(group) for group in shared_A_groups)
    training_parametrization = (
        "independent_per_projection"
        if not normalized_shared_groups
        else "fused_projection_equivalent_shared_A"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        adapter_state_dict(model, peft_compatible=True),
        str(output_dir / "adapter_model.safetensors"),
    )
    config: dict[str, Any] = {
        "alpha_pattern": {},
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": None,
        "loftq_config": {},
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(normalized_targets),
        "task_type": "CAUSAL_LM",
        "use_dora": False,
        "use_rslora": False,
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    adapter_manifest = {
        "format_version": 1,
        "training_parametrization": training_parametrization,
        "shared_A_groups": [list(group) for group in normalized_shared_groups],
        "target_modules": list(normalized_targets),
        "wrapped_module_count": sum(1 for _ in iter_lora_modules(model)),
        "trainable_parameters": trainable_parameter_summary(model)[
            "trainable_parameters"
        ],
        "peft_inference_compatible": True,
        "native_resume_required_to_preserve_shared_A": bool(normalized_shared_groups),
    }
    (output_dir / "hypercombine_adapter_manifest.json").write_text(
        json.dumps(adapter_manifest, indent=2) + "\n", encoding="utf-8"
    )


def load_adapter(model: nn.Module, adapter_file: Path) -> None:
    from safetensors.torch import load_file

    tensors = load_file(str(adapter_file), device="cpu")
    missing: list[str] = []
    loaded_shared: dict[int, torch.Tensor] = {}
    for name, module in iter_lora_modules(model):
        prefix = f"base_model.model.{name}"
        for suffix, parameter in (
            ("lora_A.weight", module.lora_A),
            ("lora_B.weight", module.lora_B),
        ):
            key = f"{prefix}.{suffix}"
            if key not in tensors:
                missing.append(key)
                continue
            value = tensors[key]
            if value.shape != parameter.shape:
                raise ValueError(
                    f"Adapter shape mismatch for {key}: expected {tuple(parameter.shape)}, "
                    f"got {tuple(value.shape)}"
                )
            converted = value.to(device=parameter.device, dtype=parameter.dtype)
            parameter_id = id(parameter)
            if parameter_id in loaded_shared:
                if not torch.equal(loaded_shared[parameter_id], converted):
                    raise ValueError(
                        f"Adapter contains different values for shared parameter {key}"
                    )
            else:
                parameter.data.copy_(converted)
                loaded_shared[parameter_id] = converted
    if missing:
        raise ValueError(
            f"Adapter is missing {len(missing)} tensors; first={missing[0]}"
        )

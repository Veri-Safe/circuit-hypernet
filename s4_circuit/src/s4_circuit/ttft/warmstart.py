"""Copy one generated LoRA into a trainable PEFT adapter without merging."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from s4_circuit.lora_scale import validate_lora_scale_multiplier


ADAPTER = "default"


def _finite(tensor: Tensor, label: str) -> None:
    if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{label} must contain finite floating-point values")


def _validate_layer(module: nn.Module, name: str) -> None:
    if getattr(module, "merged", False) or getattr(module, "disable_adapters", False):
        raise ValueError(f"Warmstart requires an unmerged active adapter: {name}")
    if tuple(module.active_adapters) != (ADAPTER,):
        raise ValueError(f"Warmstart requires only the default active adapter: {name}")
    if module.use_dora.get(ADAPTER, False) or getattr(module, "fan_in_fan_out", False):
        raise ValueError(f"Warmstart supports ordinary linear/embedding LoRA only: {name}")
    if getattr(module.lora_dropout[ADAPTER], "p", 0.0) != 0.0:
        raise ValueError(f"Warmstart requires zero LoRA dropout: {name}")


def initialize_peft_from_generated(
    model: nn.Module, generated_state: Mapping[str, Tensor], *, gate: float
) -> dict[str, Any]:
    """Initialize default PEFT linear factors from a single-query PG state.

    For every covered target, ``s_peft * B_new @ A_new`` equals
    ``gate * s_pg * B_pg @ A_pg`` up to the destination dtype's precision.
    Expanded A rows retain PEFT's random initialization and expanded B columns
    are zero, so those extra ranks have a learning signal. The caller must reset
    the stock adapter before each task/LR; generated embedding factors are not
    supported, and untouched embeddings must still have a zero initial delta.
    All validation happens before copying parameters. No base weight, requires-
    grad flag, generated tensor, or optimizer state is changed.
    """

    gate = validate_lora_scale_multiplier(gate)
    if "scale" not in generated_state:
        raise ValueError("Generated LoRA state is missing scale")
    scale = generated_state["scale"]
    if not isinstance(scale, Tensor) or scale.numel() != 1:
        raise ValueError("Generated LoRA scale must be a scalar tensor")
    _finite(scale, "Generated LoRA scale")
    source_scale = float(scale.detach().item())
    if source_scale <= 0.0:
        raise ValueError("Generated LoRA scale must be positive")
    effective_scale = gate * source_scale
    if not math.isfinite(effective_scale):
        raise ValueError("Gated generated LoRA scale is not finite")

    pairs: dict[str, dict[str, Tensor]] = {}
    for key, tensor in generated_state.items():
        if key == "scale":
            continue
        if not isinstance(key, str) or not key.endswith((".lora_A", ".lora_B")):
            raise ValueError(f"Unexpected generated LoRA key: {key}")
        stem, factor = key.rsplit(".", 1)
        if not isinstance(tensor, Tensor):
            raise ValueError(f"Generated LoRA factor is not a tensor: {key}")
        _finite(tensor, key)
        if tensor.ndim != 3 or tensor.shape[0] != 1:
            raise ValueError(f"Generated LoRA factor must have batch size one: {key}")
        pairs.setdefault(stem, {})[factor] = tensor
    if not pairs or any(set(pair) != {"lora_A", "lora_B"} for pair in pairs.values()):
        raise ValueError("Generated LoRA state has missing A/B pairs")

    linear_layers = {}
    embedding_names = []
    adapter_parameters: set[int] = set()
    for name, module in model.named_modules():
        if ADAPTER in getattr(module, "lora_A", {}):
            if ADAPTER not in module.lora_B:
                raise ValueError(f"PEFT target is missing default B: {name}")
            _validate_layer(module, name)
            a_layer, b_layer = module.lora_A[ADAPTER], module.lora_B[ADAPTER]
            if a_layer.bias is not None or b_layer.bias is not None:
                raise ValueError(f"Warmstart does not support adapter biases: {name}")
            linear_layers[name] = module
            adapter_parameters.update((id(a_layer.weight), id(b_layer.weight)))
        if ADAPTER in getattr(module, "lora_embedding_A", {}):
            _validate_layer(module, name)
            a, b = module.lora_embedding_A[ADAPTER], module.lora_embedding_B[ADAPTER]
            _finite(a, name + ".lora_embedding_A")
            _finite(b, name + ".lora_embedding_B")
            if not (bool((a == 0).all()) or bool((b == 0).all())):
                raise ValueError(f"Uncovered embedding must have its stock zero delta: {name}")
            embedding_names.append(name)
            adapter_parameters.update((id(a), id(b)))
    if not linear_layers:
        raise ValueError("Model has no default PEFT linear adapters")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad != (id(parameter) in adapter_parameters):
            raise ValueError(f"Only default adapter parameters may require gradients: {name}")

    prepared = []
    matched_names = set()
    ranks = set()
    reports = []
    for stem, pair in sorted(pairs.items()):
        matches = [name for name in linear_layers if name == stem or name.endswith("." + stem)]
        if len(matches) != 1 or matches[0] in matched_names:
            raise ValueError(f"Generated LoRA key does not uniquely match a PEFT target: {stem}")
        name = matches[0]
        matched_names.add(name)
        module = linear_layers[name]
        a, b = module.lora_A[ADAPTER].weight, module.lora_B[ADAPTER].weight
        source_a, source_b = pair["lora_A"], pair["lora_B"]
        rank = source_a.shape[1]
        if (
            a.ndim != 2 or b.ndim != 2 or rank <= 0 or rank > a.shape[0]
            or source_b.shape[2] != rank or b.shape[1] != a.shape[0]
            or source_a.shape[2] != a.shape[1] or source_b.shape[1] != b.shape[0]
        ):
            raise ValueError(f"Generated/PEFT LoRA factor shape mismatch: {stem}")
        ranks.add(rank)
        _finite(a, name + ".lora_A")
        _finite(b, name + ".lora_B")
        if rank < a.shape[0] and not bool(a[rank:].detach().ne(0).any(dim=1).all()):
            raise ValueError(f"Expanded PEFT A rows must retain their random initialization: {name}")
        target_scale = float(module.scaling[ADAPTER])
        if not math.isfinite(target_scale) or target_scale <= 0:
            raise ValueError(f"Invalid PEFT LoRA scaling: {name}")
        multiplier = effective_scale / target_scale
        new_a = source_a[0].detach().to(device=a.device, dtype=a.dtype)
        new_b = source_b[0].detach().to(device=b.device, dtype=b.dtype) * multiplier
        _finite(new_a, name + ".copied_A")
        _finite(new_b, name + ".copied_B")
        prepared.append((a, b, new_a, new_b, rank))
        reports.append({
            "generated_target": stem, "peft_target": name,
            "generated_rank": rank, "peft_rank": a.shape[0],
            "peft_scale": target_scale, "copied_B_multiplier": multiplier,
            "extra_trainable_ranks": a.shape[0] - rank,
        })
    if matched_names != set(linear_layers):
        missing = sorted(set(linear_layers) - matched_names)
        raise ValueError(f"Generated LoRA does not cover all PEFT linear targets: {missing}")
    if len(ranks) != 1:
        raise ValueError("Generated LoRA targets must share one rank")

    with torch.no_grad():
        for a, b, new_a, new_b, rank in prepared:
            a[:rank].copy_(new_a)
            b.zero_()
            b[:, :rank].copy_(new_b)
    return {
        "format": "sft_ttft_generated_lora_warmstart_v1",
        "adapter": ADAPTER, "gate": gate, "generated_scale": source_scale,
        "effective_generated_scale": effective_scale,
        "generated_rank": next(iter(ranks)), "initialized_module_count": len(reports),
        "modules": reports, "untouched_embeddings": sorted(embedding_names),
        "extra_rank_initialization": "preserve_random_A_zero_B",
    }

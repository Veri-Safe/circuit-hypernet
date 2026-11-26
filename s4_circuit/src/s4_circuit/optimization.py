from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from s2_hypernet.schedulers import cosine_to_floor_lambda
from s2_hypernet.nvarc_pg.runtime import (
    build_optimizer_and_scheduler as build_default_optimizer_and_scheduler,
)

from .config import ContinuationConfig, OptimizationConfig


@dataclass(frozen=True)
class ParameterPartition:
    trunk: tuple[nn.Parameter, ...]
    circuit: tuple[nn.Parameter, ...]


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def partition_parameter_generator_parameters(
    module: nn.Module, *, require_circuit: bool = True
) -> ParameterPartition:
    model = _unwrap(module)
    circuit_modules = (
        model.fusion,
        model.circuit_attentions,
        model.circuit_output_projections,
    )
    circuit: list[nn.Parameter] = []
    circuit_ids: set[int] = set()
    for component in circuit_modules:
        for parameter in component.parameters():
            if not parameter.requires_grad or id(parameter) in circuit_ids:
                continue
            circuit.append(parameter)
            circuit_ids.add(id(parameter))
    if model.global_lora_scale_control is not None:
        for parameter in model.global_lora_scale_control.parameters():
            if parameter.requires_grad and id(parameter) not in circuit_ids:
                circuit.append(parameter)
                circuit_ids.add(id(parameter))

    all_trainable = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    trunk = tuple(
        parameter for parameter in all_trainable if id(parameter) not in circuit_ids
    )
    partition_ids = {id(parameter) for parameter in trunk + tuple(circuit)}
    all_ids = {id(parameter) for parameter in all_trainable}
    if len(partition_ids) != len(trunk) + len(circuit) or partition_ids != all_ids:
        raise RuntimeError(
            "S4 optimizer parameter partition is incomplete or overlapping"
        )
    if require_circuit and not circuit:
        raise RuntimeError("S4 optimizer requires non-empty circuit parameters")
    return ParameterPartition(trunk=trunk, circuit=tuple(circuit))


def build_optimizer_and_scheduler(
    s2_config: Any,
    parameter_generator: nn.Module,
    optimization: OptimizationConfig,
    continuation: ContinuationConfig | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    # Preserve the exact one-group optimizer layout used by existing production
    # checkpoints. Split groups are needed only when a sweep changes LR ratios.
    if optimization.is_default:
        optimizer, scheduler = build_default_optimizer_and_scheduler(
            s2_config, parameter_generator
        )
        if continuation is None:
            return optimizer, scheduler
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )

    if optimization.freeze_trunk:
        model = _unwrap(parameter_generator)
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
    partition = partition_parameter_generator_parameters(parameter_generator)
    trunk_peak_lr = s2_config.training.peak_lr * optimization.global_lr_multiplier
    circuit_peak_lr = trunk_peak_lr * optimization.circuit_lr_multiplier
    kwargs: dict[str, Any] = {
        "betas": (s2_config.training.beta1, s2_config.training.beta2),
        "weight_decay": s2_config.training.weight_decay,
    }
    fused = os.environ.get("NVARC_USE_FUSED_ADAMW", "1").strip().lower()
    if fused not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError("NVARC_USE_FUSED_ADAMW has an unsupported value")
    if torch.cuda.is_available() and fused in {"1", "true", "yes", "on"}:
        kwargs["fused"] = True
    groups = []
    if partition.trunk:
        groups.append({"params": partition.trunk, "lr": trunk_peak_lr, "name": "trunk"})
    groups.append(
        {"params": partition.circuit, "lr": circuit_peak_lr, "name": "circuit"}
    )
    optimizer = torch.optim.AdamW(groups, **kwargs)
    lr_lambda = (
        (lambda _: 1.0)
        if continuation is not None
        else cosine_to_floor_lambda(
            total_steps=s2_config.training.max_steps,
            warmup_steps=s2_config.training.warmup_steps,
            floor_step=s2_config.training.max_steps,
            final_decay_steps=0,
            floor_ratio=s2_config.training.min_lr / s2_config.training.peak_lr,
        )
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=tuple(lr_lambda for _ in groups),
    )
    return optimizer, scheduler


def apply_continuation_learning_rates(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    continuation: ContinuationConfig,
) -> dict[str, dict[str, float]]:
    """Scale restored parent LRs and rebase a constant continuation scheduler."""

    scales = {
        "trunk": continuation.trunk_lr_scale,
        "circuit": continuation.circuit_lr_scale,
    }
    names = [str(group.get("name", "")) for group in optimizer.param_groups]
    if set(names) != set(scales) or len(names) != len(scales):
        raise ValueError(
            "S4 continuation requires exactly named trunk and circuit optimizer groups"
        )
    result: dict[str, dict[str, float]] = {}
    target_lrs: list[float] = []
    for group, name in zip(optimizer.param_groups, names, strict=True):
        parent_lr = float(group["lr"])
        target_lr = parent_lr * scales[name]
        if target_lr <= 0.0:
            raise ValueError("S4 continuation produced a non-positive learning rate")
        group["lr"] = target_lr
        group["initial_lr"] = target_lr
        target_lrs.append(target_lr)
        result[name] = {
            "parent_lr": parent_lr,
            "scale": scales[name],
            "target_lr": target_lr,
        }
    scheduler.base_lrs = target_lrs
    scheduler._last_lr = target_lrs
    return result


def learning_rates(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    rates = {
        str(group.get("name", f"group_{index}")): float(group["lr"])
        for index, group in enumerate(optimizer.param_groups)
    }
    if len(optimizer.param_groups) == 1 and "group_0" in rates:
        value = rates["group_0"]
        return {"trunk": value, "circuit": value}
    if "trunk" not in rates:
        rates["trunk"] = 0.0
    if "circuit" not in rates:
        rates["circuit"] = rates.get("trunk", 0.0)
    return rates

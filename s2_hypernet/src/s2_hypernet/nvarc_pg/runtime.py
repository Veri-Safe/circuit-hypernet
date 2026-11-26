from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoModelForCausalLM, AutoTokenizer

from s2_hypernet.schedulers import cosine_to_floor_lambda

from .config import NvarcPGConfig
from .model import FrozenQwenPerQueryLoRA
from .parameter_generator import NvarcParameterGenerator

DEFAULT_ATTENTION_IMPLEMENTATION = "flash_attention_2"
DEFAULT_USE_FUSED_ADAMW = True


def _configure_runtime_from_environment() -> dict[str, Any] | None:
    """Apply an explicitly requested CUDA runtime before model construction.

    S2 keeps its historical runtime by default. S4 launchers opt into the
    shared strict runtime through the environment so unrelated experiments do
    not silently change.
    """

    mode = os.environ.get("NVARC_RUNTIME_MODE")
    if not mode:
        return None
    from s3circuit.nvarc_circuit.deterministic_runtime import configure_runtime

    settings = configure_runtime(mode)
    _emit_runtime_event("runtime_configured", **settings)
    return settings


def _attention_implementation() -> str:
    return os.environ.get(
        "NVARC_ATTENTION_IMPLEMENTATION", DEFAULT_ATTENTION_IMPLEMENTATION
    )


def _use_fused_adamw() -> bool:
    value = os.environ.get("NVARC_USE_FUSED_ADAMW")
    if value is None:
        return DEFAULT_USE_FUSED_ADAMW
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        "NVARC_USE_FUSED_ADAMW must be one of 0/1, true/false, yes/no, or on/off"
    )


def _validate_process_group_timeout_seconds(value: float, *, source: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{source} must be a finite positive number of seconds")
    return seconds


def resolve_process_group_timeout_seconds(
    *,
    env_name: str = "NVARC_PROCESS_GROUP_TIMEOUT_SECONDS",
    default: float | None = None,
) -> float | None:
    """Resolve an optional process-group timeout without changing old defaults."""

    raw = os.environ.get(env_name)
    if raw is None or not raw.strip():
        if default is None:
            return None
        return _validate_process_group_timeout_seconds(default, source=env_name)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be a finite positive number of seconds"
        ) from exc
    return _validate_process_group_timeout_seconds(value, source=env_name)


def _emit_runtime_event(stage: str, **fields: Any) -> None:
    payload = {
        "event": "nvarc_pg_runtime",
        "stage": stage,
        "rank": os.environ.get("RANK", "0"),
        "local_rank": os.environ.get("LOCAL_RANK", "0"),
        **fields,
    }
    print(json.dumps(payload, sort_keys=True, default=str), flush=True)


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_rank_zero(self) -> bool:
        return self.rank == 0


def setup_distributed(
    expected_world_size: int,
    *,
    process_group_timeout_seconds: float | None = None,
    backend: str = "nccl",
) -> DistributedContext:
    _configure_runtime_from_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("NVARC PG training requires CUDA")
    backend = str(backend).strip().lower()
    if not backend:
        raise ValueError("backend must be a non-empty torch.distributed backend")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    _emit_runtime_event(
        "distributed_env",
        expected_world_size=expected_world_size,
        world_size=world_size,
        backend=backend,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    if world_size != expected_world_size:
        raise RuntimeError(
            f"Expected world size {expected_world_size}, found {world_size}"
        )
    torch.cuda.set_device(local_rank)
    _emit_runtime_event(
        "cuda_device_bound",
        device_index=local_rank,
        device_name=torch.cuda.get_device_name(local_rank),
    )
    timeout_seconds = (
        _validate_process_group_timeout_seconds(
            process_group_timeout_seconds,
            source="process_group_timeout_seconds",
        )
        if process_group_timeout_seconds is not None
        else resolve_process_group_timeout_seconds()
    )
    if not dist.is_initialized():
        _emit_runtime_event(
            "process_group_init_start",
            backend=backend,
            timeout_seconds=timeout_seconds,
        )
        init_kwargs = (
            {}
            if timeout_seconds is None
            else {"timeout": timedelta(seconds=timeout_seconds)}
        )
        dist.init_process_group(backend=backend, **init_kwargs)
        _emit_runtime_event(
            "process_group_init_complete",
            backend=backend,
            timeout_seconds=timeout_seconds,
        )
    else:
        _emit_runtime_event(
            "process_group_already_initialized",
            backend=str(dist.get_backend()),
        )
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def set_reproducible_seed(seed: int, *, rank: int = 0) -> None:
    effective_seed = int(seed) + int(rank)
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    torch.manual_seed(effective_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_seed)


def load_tokenizer(config: NvarcPGConfig) -> Any:
    _emit_runtime_event("tokenizer_load_start", tokenizer=config.base.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base.tokenizer,
        use_fast=True,
        local_files_only=True,
    )
    if len(tokenizer) != 16:
        raise ValueError(
            f"Expected the NVARC 16-token tokenizer, found {len(tokenizer)} tokens"
        )
    _emit_runtime_event(
        "tokenizer_load_complete", tokenizer=config.base.tokenizer, tokens=16
    )
    return tokenizer


def load_frozen_qwen(
    config: NvarcPGConfig,
    *,
    device: torch.device,
    for_training: bool = True,
) -> torch.nn.Module:
    _emit_runtime_event(
        "frozen_model_load_start",
        hf_checkpoint=config.base.hf_checkpoint,
        device=str(device),
        for_training=for_training,
        attention_implementation=_attention_implementation(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.base.hf_checkpoint,
        torch_dtype=torch.bfloat16,
        attn_implementation=_attention_implementation(),
        local_files_only=True,
    )
    if float(getattr(model.config, "attention_dropout", 0.0)) != 0.0:
        raise ValueError(
            "Frozen Qwen must have zero attention dropout for deterministic conditions"
        )
    model.config.use_cache = not for_training
    if for_training and config.training.base_gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    config.target_layout.validate_model(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.train(for_training)
    _emit_runtime_event(
        "frozen_model_load_complete",
        hf_checkpoint=config.base.hf_checkpoint,
        device=str(device),
        for_training=for_training,
    )
    return model


def build_parameter_generator(
    config: NvarcPGConfig,
    *,
    context: DistributedContext,
) -> torch.nn.Module:
    parameter_generator = NvarcParameterGenerator(
        config.build_parameter_generator_config()
    )
    if config.training.pg_gradient_checkpointing:
        parameter_generator.enable_gradient_checkpointing()
    parameter_generator.to(device=context.device, dtype=torch.bfloat16)
    parameter_generator.train()
    return DistributedDataParallel(
        parameter_generator,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def build_system(
    config: NvarcPGConfig,
    *,
    context: DistributedContext,
) -> tuple[FrozenQwenPerQueryLoRA, Any]:
    tokenizer = load_tokenizer(config)
    base_model = load_frozen_qwen(config, device=context.device)
    parameter_generator = build_parameter_generator(config, context=context)
    system = FrozenQwenPerQueryLoRA(
        base_model,
        parameter_generator,
        target_keys=config.target_layout.target_keys,
    )
    base_model.train()
    return system, tokenizer


def build_evaluation_system(
    config: NvarcPGConfig,
    *,
    context: DistributedContext,
    parameter_generator_checkpoint: str | Path | None,
) -> tuple[FrozenQwenPerQueryLoRA, Any]:
    _emit_runtime_event(
        "evaluation_system_build_start",
        rank=context.rank,
        local_rank=context.local_rank,
        pg_checkpoint=str(parameter_generator_checkpoint)
        if parameter_generator_checkpoint is not None
        else None,
    )
    tokenizer = load_tokenizer(config)
    base_model = load_frozen_qwen(config, device=context.device, for_training=False)
    if parameter_generator_checkpoint is None:
        parameter_generator: torch.nn.Module = torch.nn.Identity()
    else:
        _emit_runtime_event(
            "parameter_generator_load_start",
            rank=context.rank,
            local_rank=context.local_rank,
            pg_checkpoint=str(parameter_generator_checkpoint),
        )
        parameter_generator = NvarcParameterGenerator(
            config.build_parameter_generator_config()
        )
        state = torch.load(
            Path(parameter_generator_checkpoint) / "parameter_generator.pt",
            map_location="cpu",
            weights_only=True,
        )
        parameter_generator.load_state_dict(state, strict=True)
        parameter_generator.to(device=context.device, dtype=torch.bfloat16)
        parameter_generator.eval()
        _emit_runtime_event(
            "parameter_generator_load_complete",
            rank=context.rank,
            local_rank=context.local_rank,
            pg_checkpoint=str(parameter_generator_checkpoint),
        )
    system = FrozenQwenPerQueryLoRA(
        base_model,
        parameter_generator,
        target_keys=config.target_layout.target_keys,
    )
    base_model.eval()
    _emit_runtime_event(
        "evaluation_system_build_complete",
        rank=context.rank,
        local_rank=context.local_rank,
    )
    return system, tokenizer


def build_optimizer_and_scheduler(
    config: NvarcPGConfig,
    parameter_generator: torch.nn.Module,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    kwargs: dict[str, Any] = {
        "lr": config.training.peak_lr,
        "betas": (config.training.beta1, config.training.beta2),
        "weight_decay": config.training.weight_decay,
    }
    if torch.cuda.is_available() and _use_fused_adamw():
        kwargs["fused"] = True
    optimizer = torch.optim.AdamW(parameter_generator.parameters(), **kwargs)
    lr_lambda = cosine_to_floor_lambda(
        total_steps=config.training.max_steps,
        warmup_steps=config.training.warmup_steps,
        floor_step=config.training.max_steps,
        final_decay_steps=0,
        floor_ratio=config.training.min_lr / config.training.peak_lr,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def require_local_path(path: str | Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from s2_hypernet.nvarc_pg.config import NvarcPGConfig
from s2_hypernet.nvarc_pg.runtime import (
    DistributedContext,
    load_frozen_qwen,
    load_tokenizer,
)
from torch.nn.parallel import DistributedDataParallel

from .checkpoint import load_checkpoint
from .config import S4R1Config
from .initialization import load_fresh_model
from .model import FrozenQwenCircuitPerQueryLoRA


def _emit_training_stage(context: DistributedContext, stage: str) -> None:
    print(
        json.dumps(
            {
                "event": "s4_training_runtime",
                "stage": stage,
                "rank": context.rank,
                "local_rank": context.local_rank,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_training_system(
    config: S4R1Config,
    *,
    context: DistributedContext,
) -> tuple[FrozenQwenCircuitPerQueryLoRA, Any, NvarcPGConfig, dict[str, Any]]:
    s2 = config.load_s2_config(apply_training_override=True)
    tokenizer = load_tokenizer(s2)
    base_model = load_frozen_qwen(s2, device=context.device, for_training=True)
    _emit_training_stage(context, "fresh_model_load_start")
    model, init_manifest = load_fresh_model(config)
    _emit_training_stage(context, "fresh_model_load_complete")
    if config.runtime_optimization.pg_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    else:
        model.disable_gradient_checkpointing()
    _emit_training_stage(context, "fresh_model_prepare_runtime_start")
    model.prepare_runtime(device=context.device, pg_dtype=torch.bfloat16)
    _emit_training_stage(context, "fresh_model_prepare_runtime_complete")
    # DDP inspects the trainable parameter set while it is constructed.  The
    # optimizer also enforces this setting, but doing it only there leaves
    # freeze_trunk variants wrapped with frozen-after-DDP parameters; the next
    # reducer pass then reports an unused-parameter failure.  Apply the
    # declared optimization contract before DDP sees the model.
    if config.optimization.freeze_trunk:
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
    model.train()
    _emit_training_stage(context, "ddp_wrap_start")
    # The trainable global scale is returned as a scalar and consumed later by
    # GeneratedLoRAController forward hooks on the frozen Qwen model.  That
    # consumer is outside the DDP-wrapped module's forward call, so let the
    # reducer discover the scale parameter from the returned autograd graph.
    # Fixed-scale variants retain the cheaper, strict all-parameters path.
    find_unused_parameters = bool(config.optimization.trainable_global_lora_scale)
    distributed = DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        output_device=context.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
        # Every rank loads the immutable common trunk and deterministically
        # constructs the variant fusion weights. Re-broadcasting this multi-GB
        # state is redundant and can stall startup on distributed workers.
        init_sync=False,
    )
    _emit_training_stage(context, "ddp_wrap_complete")
    system = FrozenQwenCircuitPerQueryLoRA(
        base_model,
        distributed,
        target_keys=s2.target_layout.target_keys,
        loss_backend=config.runtime_optimization.loss_backend,
    )
    # The Base supplies frozen condition states and adapted logits, but it is
    # never itself trained. Keep its runtime mode fixed across all variants.
    base_model.eval()
    return system, tokenizer, s2, init_manifest


def build_evaluation_system(
    config: S4R1Config,
    *,
    context: DistributedContext,
    checkpoint: str | Path,
    expected_checkpoint_identity: Any,
) -> tuple[FrozenQwenCircuitPerQueryLoRA, Any, NvarcPGConfig, Any]:
    s2 = config.load_s2_config()
    tokenizer = load_tokenizer(s2)
    base_model = load_frozen_qwen(s2, device=context.device, for_training=False)
    model, _ = load_fresh_model(config)
    load_checkpoint(
        checkpoint,
        parameter_generator=model,
        optimizer=None,
        scheduler=None,
        expected_identity=expected_checkpoint_identity,
        rank=None,
    )
    model.prepare_runtime(device=context.device, pg_dtype=torch.bfloat16)
    model.eval()
    system = FrozenQwenCircuitPerQueryLoRA(
        base_model,
        model,
        target_keys=s2.target_layout.target_keys,
        loss_backend=config.runtime_optimization.loss_backend,
    )
    base_model.eval()
    return system, tokenizer, s2, model

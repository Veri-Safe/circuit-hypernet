from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from s2_hypernet.nvarc_pg.checkpoint import inspect_frozen_hf_checkpoint
from s2_hypernet.nvarc_pg.data import NvarcRowDataset, load_training_sources
from s2_hypernet.nvarc_pg.runtime import (
    cleanup_distributed,
    set_reproducible_seed,
    setup_distributed,
)

from .artifacts import load_vector_bank
from .batches import rank_rows_for_step, validate_logical_batch_manifest
from .checkpoint import (
    S4TrainerState,
    attention_backend_identity,
    finalize_checkpoint,
    load_checkpoint,
    load_model_initialization,
    load_optimizer_fork_checkpoint,
    resolve_latest,
    save_checkpoint_core,
    save_rank_rng,
)
from .checkpoint_schedule import RecoveryCheckpointSchedule
from .config import load_config
from .contracts import (
    CHECKPOINT_FORMAT,
    TRAIN_RUN_FORMAT,
    semantic_sha256,
    sha256_file,
    validate_upstream_api_contract,
)
from .join import load_join_bundle
from .optimization import (
    apply_continuation_learning_rates,
    build_optimizer_and_scheduler,
    learning_rates,
)
from .runtime import build_training_system
from .training import (
    TrainingCircuitIndex,
    selected_event_indices,
    train_optimizer_window,
)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _inspect_base_checkpoint_once(context: Any, s2: Any) -> dict[str, Any]:
    """Hash the immutable multi-GB Base checkpoint once per distributed job."""

    payload: list[dict[str, Any] | None] = [None]
    if context.is_rank_zero:
        print("base_checkpoint_inspection=start rank=0", flush=True)
        try:
            payload[0] = {
                "status": "ok",
                "manifest": inspect_frozen_hf_checkpoint(
                    hf_checkpoint=s2.base.hf_checkpoint,
                    tokenizer=s2.base.tokenizer,
                    source_run_manifest=s2.base.source_run_manifest,
                    profile=s2.profile,
                    layout=s2.target_layout,
                ),
            }
        except Exception as exc:  # Propagate one deterministic failure to every rank.
            payload[0] = {
                "status": "error",
                "exception": type(exc).__name__,
                "message": str(exc),
            }
    dist.broadcast_object_list(payload, src=0, device=context.device)
    result = payload[0]
    if result is None or result.get("status") != "ok":
        raise RuntimeError(f"Rank-zero Base checkpoint inspection failed: {result}")
    manifest = result.get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("Rank-zero Base checkpoint inspection returned no manifest")
    if context.is_rank_zero:
        print("base_checkpoint_inspection=complete rank=0", flush=True)
    return manifest


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _reduce_sum(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _reduce_int(value: int, device: torch.device, *, op: dist.ReduceOp) -> int:
    tensor = torch.tensor(value, device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=op)
    return int(tensor.item())


def _tensor_storage_bytes(
    value: Any,
    *,
    seen: set[tuple[str, int, int]] | None = None,
) -> int:
    visited = seen if seen is not None else set()
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        identity = (str(value.device), int(storage.data_ptr()), int(storage.nbytes()))
        if identity in visited:
            return 0
        visited.add(identity)
        return int(storage.nbytes())
    if isinstance(value, dict):
        return sum(_tensor_storage_bytes(item, seen=visited) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_storage_bytes(item, seen=visited) for item in value)
    return 0


def _estimated_checkpoint_bytes(
    parameter_generator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    raw_model = getattr(parameter_generator, "module", parameter_generator)
    model_bytes = _tensor_storage_bytes(raw_model.state_dict())
    optimizer_bytes = _tensor_storage_bytes(optimizer.state_dict())
    return int((model_bytes + optimizer_bytes) * 1.10) + 16 * 1024**2


def _identity(
    *,
    config: Any,
    s2: Any,
    base_manifest_sha256: str,
    join_metadata: dict[str, Any],
    batch_manifest_sha256: str,
    fresh_init_sha256: str,
    global_step: int,
) -> S4TrainerState:
    return S4TrainerState(
        format=CHECKPOINT_FORMAT,
        global_step=global_step,
        rows_consumed=global_step * s2.training.global_row_batch,
        variant_id=config.variant_id,
        # Evaluation artifacts may be finalized after training. They are
        # deliberately excluded from the immutable checkpoint lineage.
        training_config_sha256=config.training_semantic_sha256,
        base_manifest_sha256=base_manifest_sha256,
        layout_sha256=s2.target_layout.layout_sha256,
        train_root_manifest_sha256=config.train_artifacts.root_manifest_sha256,
        train_join_sha256=str(join_metadata["join_sha256"]),
        batch_manifest_sha256=batch_manifest_sha256,
        fresh_init_sha256=fresh_init_sha256,
        attention_backend=attention_backend_identity(),
    )


def run_training(
    *,
    config_path: str | Path,
    join_path: str | Path,
    output_dir: str | Path | None = None,
    log_dir: str | Path,
    resume_from: str | Path | None = None,
    initialize_from: str | Path | None = None,
    auto_resume: bool = False,
    stop_after_step: int | None = None,
) -> None:
    config = load_config(config_path)
    s2_static = config.load_s2_config(apply_training_override=True)
    logical_steps = config.diagnostic_logical_step_indices or tuple(
        range(1, s2_static.training.max_steps + 1)
    )
    active_max_steps = len(logical_steps)
    checkpoint_schedule = RecoveryCheckpointSchedule.from_environment(
        max_steps=s2_static.training.max_steps,
        default_steady_every=s2_static.training.save_every,
    )
    final_step = active_max_steps if stop_after_step is None else int(stop_after_step)
    if not 1 <= final_step <= active_max_steps:
        raise ValueError("stop_after_step is outside the active S4 schedule")
    output_root = Path(output_dir or (Path(config.output_root) / "train")).resolve()
    log_root = Path(log_dir).resolve()
    if auto_resume and resume_from is not None:
        raise ValueError("Set resume_from or auto_resume, not both")
    if auto_resume:
        resume_from = resolve_latest(output_root)
    if initialize_from is not None and resume_from is not None:
        raise ValueError("initialize_from and resume_from are mutually exclusive")
    if config.continuation is not None and initialize_from is not None:
        raise ValueError("S4 continuation cannot use model-only initialization")

    context = setup_distributed(s2_static.training.world_size)
    system = None
    try:
        set_reproducible_seed(config.seed, rank=context.rank)
        validate_upstream_api_contract()
        base_manifest = _inspect_base_checkpoint_once(context, s2_static)
        base_manifest_sha256 = semantic_sha256(base_manifest)
        join_records, join_metadata = load_join_bundle(join_path)
        if (
            join_metadata.get("artifact_contract_sha256")
            != config.artifact_contract_sha256
        ):
            raise ValueError("Training join belongs to another S4 artifact contract")
        if (
            join_metadata.get("final_root_manifest_sha256")
            != config.train_artifacts.root_manifest_sha256
        ):
            raise ValueError("Training join belongs to another S3 final root")
        if len(join_records) != config.train_artifacts.expected_events:
            raise ValueError("Training join coverage is incomplete")
        row_indices, batch_metadata = validate_logical_batch_manifest(
            config.batch_manifest, config=config
        )
        batch_manifest_sha256 = sha256_file(config.batch_manifest)
        vector_bank = load_vector_bank(config.train_artifacts)
        circuits = TrainingCircuitIndex(join_records, vector_bank)

        system, tokenizer, s2, fresh_manifest = build_training_system(
            config,
            context=context,
        )
        if config.replication is not None:
            # All ranks load the same initial weights. Restore rank-specific
            # run RNG after that deterministic construction; resume later
            # restores each rank's saved RNG instead.
            set_reproducible_seed(config.seed, rank=context.rank)
        optimizer, scheduler = build_optimizer_and_scheduler(
            s2,
            system.parameter_generator,
            config.optimization,
            config.continuation,
        )
        expected = _identity(
            config=config,
            s2=s2,
            base_manifest_sha256=base_manifest_sha256,
            join_metadata=join_metadata,
            batch_manifest_sha256=batch_manifest_sha256,
            fresh_init_sha256=str(fresh_manifest["aggregate_sha256"]),
            global_step=0,
        )
        start_step = 0
        if initialize_from is not None:
            parent = load_model_initialization(
                initialize_from,
                parameter_generator=system.parameter_generator,
                expected_identity=expected,
            )
            if context.is_rank_zero:
                print(
                    f"training_initialization=parent checkpoint={Path(initialize_from).resolve()} "
                    f"parent_step={parent.global_step} optimizer=reset rng=reset",
                    flush=True,
                )
        continuation_lrs: dict[str, dict[str, float]] | None = None
        if config.continuation is not None and resume_from is None:
            parent_config = load_config(config.continuation.source_config)
            if parent_config.continuation is not None:
                raise ValueError("S4 continuation source config must be a base run")
            parent = load_optimizer_fork_checkpoint(
                config.continuation.source_checkpoint,
                parameter_generator=system.parameter_generator,
                optimizer=optimizer,
                expected_identity=expected,
                expected_parent_training_config_sha256=(
                    parent_config.training_semantic_sha256
                ),
                expected_parent_step=config.continuation.source_step,
                rank=context.rank,
            )
            continuation_lrs = apply_continuation_learning_rates(
                optimizer,
                scheduler,
                config.continuation,
            )
            start_step = parent.global_step
            if context.is_rank_zero:
                print(
                    "training_initialization=optimizer_preserving_continuation "
                    f"checkpoint={Path(config.continuation.source_checkpoint).resolve()} "
                    f"parent_step={parent.global_step} scheduler=constant_lr_v1 "
                    f"learning_rates={json.dumps(continuation_lrs, sort_keys=True)}",
                    flush=True,
                )
        if resume_from is not None:
            restored = load_checkpoint(
                resume_from,
                parameter_generator=system.parameter_generator,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_identity=expected,
                rank=context.rank,
            )
            start_step = restored.global_step
            if start_step >= final_step:
                if context.is_rank_zero:
                    print(
                        f"training_status=already_complete step={start_step}",
                        flush=True,
                    )
                return
        if start_step >= final_step:
            raise ValueError(
                f"S4 continuation start step {start_step} must precede {final_step}"
            )

        source_dataset = load_training_sources(s2.data.train_paths).combined
        if len(source_dataset) != 3_200:
            raise ValueError("S4 training dataset row count changed")
        assistant_counts = tuple(
            sum(
                str(message.get("role", "")).strip().lower() == "assistant"
                for message in source_dataset[index]["messages"]
            )
            for index in range(len(source_dataset))
        )
        source_event_count = sum(assistant_counts)
        if source_event_count != config.train_artifacts.expected_events:
            raise ValueError(
                "S4 source assistant-event coverage changed: "
                f"{source_event_count} != {config.train_artifacts.expected_events}"
            )
        start_index = config.loss_protocol.supervised_assistant_start_index
        if any(count <= start_index for count in assistant_counts):
            raise ValueError("S4 loss protocol leaves a source row unsupervised")
        supervised_event_count = sum(
            len(
                selected_event_indices(
                    count,
                    supervised_assistant_start_index=start_index,
                    objective_weighting_mode=config.objective_weighting.mode,
                )
            )
            for count in assistant_counts
        )
        if supervised_event_count != config.expected_supervised_train_events:
            raise ValueError(
                "S4 supervised assistant-event coverage changed: "
                f"{supervised_event_count} != "
                f"{config.expected_supervised_train_events}"
            )
        if context.is_rank_zero:
            print(
                "loss_protocol=verified "
                f"name={config.loss_protocol.name} "
                f"objective_weighting={config.objective_weighting.mode} "
                f"total_events={source_event_count} "
                f"supervised_events={supervised_event_count} "
                f"skipped_events={source_event_count - supervised_event_count}",
                flush=True,
            )

        if context.is_rank_zero:
            if (
                output_root.exists()
                and any(output_root.iterdir())
                and resume_from is None
            ):
                raise FileExistsError(f"S4 training output is not empty: {output_root}")
            output_root.mkdir(parents=True, exist_ok=True)
            log_root.mkdir(parents=True, exist_ok=True)
            _write_json_atomic(
                output_root / "base_checkpoint_manifest.json", base_manifest
            )
            _write_json_atomic(
                output_root / "run_manifest.json",
                {
                    "format": TRAIN_RUN_FORMAT,
                    "status": "running",
                    "variant_id": config.variant_id,
                    "training_config_sha256": config.training_semantic_sha256,
                    "join_sha256": join_metadata["join_sha256"],
                    "batch_manifest_sha256": batch_manifest_sha256,
                    "batch_manifest_semantic_sha256": batch_metadata[
                        "manifest_semantic_sha256"
                    ],
                    "fresh_init_sha256": fresh_manifest["aggregate_sha256"],
                    "attention_backend": attention_backend_identity(),
                    "base_manifest_sha256": base_manifest_sha256,
                    "start_step": start_step,
                    "target_step": final_step,
                    "world_size": context.world_size,
                    **(
                        {"replication": config.semantic_dict["replication"]}
                        if config.replication else {}
                    ),
                    "micro_row_batch": s2.training.micro_row_batch,
                    "micro_row_batch_override": config.training_micro_row_batch_override,
                    "max_grad_norm": s2.training.max_grad_norm,
                    "max_grad_norm_override": (
                        config.training_max_grad_norm_override
                    ),
                    "runtime_optimization": {
                        "loss_backend": config.runtime_optimization.loss_backend,
                        "batching_mode": config.runtime_optimization.batching_mode,
                        "max_micro_rows": config.runtime_optimization.max_micro_rows,
                        "padded_token_budget": (
                            config.runtime_optimization.padded_token_budget
                        ),
                        "pg_gradient_checkpointing": (
                            config.runtime_optimization.pg_gradient_checkpointing
                        ),
                    },
                    "loss_protocol": {
                        "name": config.loss_protocol.name,
                        "supervised_assistant_start_index": (
                            config.loss_protocol.supervised_assistant_start_index
                        ),
                        "expected_supervised_train_events": (
                            config.expected_supervised_train_events
                        ),
                        "expected_skipped_train_events": (
                            config.train_artifacts.expected_events
                            - config.expected_supervised_train_events
                        ),
                    },
                    "objective_weighting": {
                        "mode": config.objective_weighting.mode,
                        "expected_selected_events": (
                            config.objective_weighting.expected_selected_events
                        ),
                        "depth_min_weight": (
                            config.objective_weighting.depth_min_weight
                        ),
                        "depth_max_weight": (
                            config.objective_weighting.depth_max_weight
                        ),
                    },
                    "circuit_training": {
                        "negative_mode": config.circuit_training.negative_mode,
                        "negative_fraction": (
                            config.circuit_training.negative_fraction
                        ),
                        "negative_coefficient": (
                            config.circuit_training.negative_coefficient
                        ),
                        "negative_margin": config.circuit_training.negative_margin,
                        "negative_seed": config.circuit_training.negative_seed,
                        "negative_assignment_sha256": (
                            circuits.negative_assignment_sha256
                        ),
                    },
                    "checkpoint_schedule": checkpoint_schedule.as_dict(),
                    "checkpoint_min_free_gib": float(
                        os.environ.get("S4_CHECKPOINT_MIN_FREE_GIB", "16")
                    ),
                    "diagnostic_logical_step_indices": list(
                        config.diagnostic_logical_step_indices
                    ),
                    "optimization": {
                        "global_lr_multiplier": config.optimization.global_lr_multiplier,
                        "circuit_lr_multiplier": config.optimization.circuit_lr_multiplier,
                        "trunk_peak_lr": (
                            s2.training.peak_lr
                            * config.optimization.global_lr_multiplier
                        ),
                        "trunk_min_lr": (
                            s2.training.min_lr
                            * config.optimization.global_lr_multiplier
                        ),
                        "circuit_peak_lr": (
                            s2.training.peak_lr
                            * config.optimization.global_lr_multiplier
                            * config.optimization.circuit_lr_multiplier
                        ),
                        "circuit_min_lr": (
                            s2.training.min_lr
                            * config.optimization.global_lr_multiplier
                            * config.optimization.circuit_lr_multiplier
                        ),
                        "factor_l2": config.optimization.factor_l2,
                        "effective_delta_l2": config.optimization.effective_delta_l2,
                        "lora_gate_max": config.optimization.lora_gate_max,
                        "lora_gate_warmup_steps": (
                            config.optimization.lora_gate_warmup_steps
                        ),
                        "trainable_global_lora_scale": (
                            config.optimization.trainable_global_lora_scale
                        ),
                        "global_lora_scale_initial": (
                            config.optimization.global_lora_scale_initial
                        ),
                        "global_lora_scale_max": (
                            config.optimization.global_lora_scale_max
                        ),
                        "base_preservation_coefficient": (
                            config.optimization.base_preservation_coefficient
                        ),
                    },
                    "continuation": (
                        None
                        if config.continuation is None
                        else {
                            "source_checkpoint": (
                                config.continuation.source_checkpoint
                            ),
                            "source_config": config.continuation.source_config,
                            "source_step": config.continuation.source_step,
                            "scheduler": config.continuation.scheduler,
                            "trunk_lr_scale": (config.continuation.trunk_lr_scale),
                            "circuit_lr_scale": (config.continuation.circuit_lr_scale),
                            "fork_learning_rates": continuation_lrs,
                        }
                    ),
                    "trainable_parameters": sum(
                        value.numel()
                        for value in system.parameter_generator.parameters()
                        if value.requires_grad
                    ),
                    "missing_circuits": 0,
                    "corrupt_circuits": 0,
                },
            )
        _barrier()

        dataset = NvarcRowDataset(
            source_dataset,
            tokenizer,
            max_sequence_length=s2.data.max_sequence_length,
        )
        base_loss_cache: dict[tuple[int, int], float] | None = (
            {}
            if config.optimization.base_preservation_coefficient
            else None
        )
        training_started = time.perf_counter()
        for global_step in range(start_step + 1, final_step + 1):
            logical_step = logical_steps[global_step - 1]
            local_indices = rank_rows_for_step(
                row_indices,
                step_index=logical_step - 1,
                rank=context.rank,
                world_size=context.world_size,
                micro_batch_size=s2.training.micro_row_batch,
            )
            rows = tuple(dataset[index] for index in local_indices)
            local_total_events = sum(len(row.events) for row in rows)
            local_events = sum(
                len(
                    selected_event_indices(
                        len(row.events),
                        supervised_assistant_start_index=(
                            config.loss_protocol.supervised_assistant_start_index
                        ),
                        objective_weighting_mode=config.objective_weighting.mode,
                    )
                )
                for row in rows
            )
            if context.is_rank_zero:
                print(
                    f"training_step=start global_step={global_step}/{final_step} "
                    f"logical_step={logical_step}/{s2.training.max_steps} "
                    f"local_rows={len(rows)} supervised_events={local_events} "
                    f"total_events={local_total_events}",
                    flush=True,
                )
            if context.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(context.device)
                torch.cuda.synchronize(context.device)
            step_started = time.perf_counter()
            metrics = train_optimizer_window(
                system,
                rows=rows,
                circuits=circuits,
                optimizer=optimizer,
                scheduler=scheduler,
                device=context.device,
                world_size=context.world_size,
                max_grad_norm=s2.training.max_grad_norm,
                factor_l2=config.optimization.factor_l2,
                effective_delta_l2=config.optimization.effective_delta_l2,
                lora_gate_multiplier=config.optimization.lora_gate_at_step(
                    logical_step
                ),
                base_preservation_coefficient=(
                    config.optimization.base_preservation_coefficient
                ),
                base_loss_cache=base_loss_cache,
                circuit_negative_mode=config.circuit_training.negative_mode,
                circuit_negative_fraction=(
                    config.circuit_training.negative_fraction
                ),
                circuit_negative_coefficient=(
                    config.circuit_training.negative_coefficient
                ),
                circuit_negative_margin=config.circuit_training.negative_margin,
                circuit_negative_seed=config.circuit_training.negative_seed,
                micro_batch_size=s2.training.micro_row_batch,
                batching_mode=config.runtime_optimization.batching_mode,
                max_micro_rows=config.runtime_optimization.max_micro_rows,
                padded_token_budget=(config.runtime_optimization.padded_token_budget),
                supervised_assistant_start_index=(
                    config.loss_protocol.supervised_assistant_start_index
                ),
                objective_weighting_mode=config.objective_weighting.mode,
                objective_depth_min_weight=(
                    config.objective_weighting.depth_min_weight
                ),
                objective_depth_max_weight=(
                    config.objective_weighting.depth_max_weight
                ),
                progress_callback=(
                    (
                        lambda complete, total: print(
                            f"training_step=progress global_step={global_step}/{final_step} "
                            f"local_rows={complete}/{total}",
                            flush=True,
                        )
                    )
                    if context.is_rank_zero
                    else None
                ),
            )
            if any(value.grad is not None for value in system.base_model.parameters()):
                raise RuntimeError("Frozen Base unexpectedly received gradients")
            torch.cuda.synchronize(context.device)
            compute_step_seconds = time.perf_counter() - step_started
            global_loss_sum = _reduce_sum(
                float(metrics["local_loss_sum"]), context.device
            )
            global_main_objective_sum = _reduce_sum(
                float(metrics["local_main_objective_sum"]),
                context.device,
            )
            global_factor_l2_sum = _reduce_sum(
                float(metrics["local_factor_l2_sum"]), context.device
            )
            global_effective_delta_l2_sum = _reduce_sum(
                float(metrics["local_effective_delta_l2_sum"]), context.device
            )
            global_factor_l2_mean = global_factor_l2_sum / int(
                metrics["global_event_count"]
            )
            global_effective_delta_l2_mean = (
                global_effective_delta_l2_sum / int(metrics["global_event_count"])
            )
            global_base_preservation_hinge_sum = _reduce_sum(
                float(metrics["local_base_preservation_hinge_sum"]),
                context.device,
            )
            global_base_preservation_hinge_mean = (
                global_base_preservation_hinge_sum
                / float(metrics["global_event_weight_sum"])
            )
            global_base_preservation_active_events = _reduce_int(
                int(metrics["local_base_preservation_active_events"]),
                context.device,
                op=dist.ReduceOp.SUM,
            )
            global_negative_margin_sum = _reduce_sum(
                float(metrics["local_negative_margin_sum"]),
                context.device,
            )
            global_negative_active_events = _reduce_int(
                int(metrics["local_negative_active_events"]),
                context.device,
                op=dist.ReduceOp.SUM,
            )
            global_negative_margin_mean = (
                global_negative_margin_sum
                / int(metrics["global_negative_event_count"])
                if int(metrics["global_negative_event_count"])
                else 0.0
            )
            global_token_loss = global_loss_sum / int(metrics["global_token_count"])
            global_loss = global_main_objective_sum / float(
                metrics["global_objective_weight_sum"]
            )
            lrs = learning_rates(optimizer)
            metrics.update(
                {
                    "global_step": global_step,
                    "logical_schedule_step": logical_step,
                    "loss_backend": config.runtime_optimization.loss_backend,
                    "pg_gradient_checkpointing": (
                        config.runtime_optimization.pg_gradient_checkpointing
                    ),
                    "global_loss": global_loss,
                    "global_token_loss": global_token_loss,
                    "global_factor_l2_mean": global_factor_l2_mean,
                    "global_effective_delta_l2_mean": (
                        global_effective_delta_l2_mean
                    ),
                    "global_base_preservation_hinge_mean": (
                        global_base_preservation_hinge_mean
                    ),
                    "global_base_preservation_active_events": (
                        global_base_preservation_active_events
                    ),
                    "global_negative_margin_mean": global_negative_margin_mean,
                    "global_negative_active_events": (
                        global_negative_active_events
                    ),
                    "global_objective": (
                        global_loss
                        + config.optimization.factor_l2 * global_factor_l2_mean
                        + config.optimization.effective_delta_l2
                        * global_effective_delta_l2_mean
                        + config.optimization.base_preservation_coefficient
                        * global_base_preservation_hinge_mean
                        + config.circuit_training.negative_coefficient
                        * global_negative_margin_mean
                    ),
                    "learning_rate": lrs["trunk"],
                    "trunk_learning_rate": lrs["trunk"],
                    "circuit_learning_rate": lrs["circuit"],
                    "max_grad_norm": s2.training.max_grad_norm,
                    "compute_step_seconds": compute_step_seconds,
                    "gpu_active_fraction_of_compute": min(
                        1.0,
                        float(metrics["gpu_active_seconds"])
                        / max(compute_step_seconds, 1.0e-12),
                    ),
                    "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(
                        context.device
                    )
                    / 1024**3,
                    "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(
                        context.device
                    )
                    / 1024**3,
                }
            )

            should_save = checkpoint_schedule.should_save(
                global_step,
                target_step=final_step,
                milestone_steps=s2.evaluation.milestone_steps,
            )
            checkpoint_started = time.perf_counter()
            checkpoint_bytes = 0
            checkpoint_estimated_bytes = 0
            disk_free_before_checkpoint = 0
            if should_save:
                checkpoint_estimated_bytes = _estimated_checkpoint_bytes(
                    system.parameter_generator,
                    optimizer,
                )
                disk_free_before_checkpoint = shutil.disk_usage(output_root).free
                checkpoint_estimated_bytes = _reduce_int(
                    checkpoint_estimated_bytes,
                    context.device,
                    op=dist.ReduceOp.MAX,
                )
                disk_free_before_checkpoint = _reduce_int(
                    disk_free_before_checkpoint,
                    context.device,
                    op=dist.ReduceOp.MIN,
                )
                reserve_bytes = int(
                    float(os.environ.get("S4_CHECKPOINT_MIN_FREE_GIB", "16")) * 1024**3
                )
                if (
                    disk_free_before_checkpoint
                    < checkpoint_estimated_bytes + reserve_bytes
                ):
                    raise OSError(
                        "Insufficient free space for an atomic S4 checkpoint: "
                        f"free={disk_free_before_checkpoint} "
                        f"estimated_checkpoint={checkpoint_estimated_bytes} "
                        f"required_reserve={reserve_bytes}"
                    )
                _barrier()
                state = _identity(
                    config=config,
                    s2=s2,
                    base_manifest_sha256=base_manifest_sha256,
                    join_metadata=join_metadata,
                    batch_manifest_sha256=batch_manifest_sha256,
                    fresh_init_sha256=str(fresh_manifest["aggregate_sha256"]),
                    global_step=global_step,
                )
                checkpoint = output_root / "checkpoints" / f"step_{global_step:08d}"
                if context.is_rank_zero:
                    checkpoint = save_checkpoint_core(
                        output_root=output_root,
                        parameter_generator=system.parameter_generator,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        trainer_state=state,
                    )
                _barrier()
                save_rank_rng(checkpoint, rank=context.rank)
                _barrier()
                if context.is_rank_zero:
                    finalize_checkpoint(
                        checkpoint,
                        output_root=output_root,
                        world_size=context.world_size,
                    )
                    checkpoint_bytes = sum(
                        path.stat().st_size
                        for path in checkpoint.rglob("*")
                        if path.is_file()
                    )
                _barrier()

            checkpoint_seconds = (
                time.perf_counter() - checkpoint_started if should_save else 0.0
            )
            step_seconds = time.perf_counter() - step_started
            elapsed = time.perf_counter() - training_started
            completed = global_step - start_step
            eta = elapsed / completed * (final_step - global_step)
            metrics.update(
                {
                    "checkpoint_saved": int(should_save),
                    "checkpoint_seconds": checkpoint_seconds,
                    "checkpoint_bytes": checkpoint_bytes,
                    "checkpoint_estimated_bytes": checkpoint_estimated_bytes,
                    "disk_free_before_checkpoint": disk_free_before_checkpoint,
                    "step_seconds": step_seconds,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "rows_per_second": s2.training.global_row_batch / step_seconds,
                    "events_per_second": int(metrics["global_event_count"])
                    / step_seconds,
                    "supervised_tokens_per_second": int(metrics["global_token_count"])
                    / step_seconds,
                }
            )
            if context.is_rank_zero:
                _append_jsonl(log_root / "train_metrics.jsonl", metrics)
                print(
                    f"training_step=complete global_step={global_step}/{final_step} "
                    f"logical_step={logical_step} loss={metrics['global_loss']:.8f} "
                    f"compute_seconds={compute_step_seconds:.3f} "
                    f"checkpoint_seconds={checkpoint_seconds:.3f} "
                    f"step_seconds={step_seconds:.3f} eta_seconds={eta:.1f}",
                    flush=True,
                )

        if context.is_rank_zero:
            manifest = json.loads(
                (output_root / "run_manifest.json").read_text(encoding="utf-8")
            )
            manifest.update(
                status="complete",
                completed_step=final_step,
                elapsed_seconds=time.perf_counter() - training_started,
            )
            _write_json_atomic(output_root / "run_manifest.json", manifest)
    finally:
        if system is not None:
            system.close()
        cleanup_distributed()

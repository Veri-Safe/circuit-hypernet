from __future__ import annotations

import contextlib
import datetime as dt
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .activations import LayerActivationCapture, load_frozen_step399
from .blocks import ActivationBlockBuffer, deterministic_block_permutation
from .checkpoint import (
    TrainingCursor,
    checkpoint_code_identity_compatibility,
    load_resumable_checkpoint,
    save_resumable_checkpoint,
    save_weight_snapshot,
)
from .config import NvarcTranscoderConfig
from .identity import (
    load_or_create_base_identity,
    package_code_identity,
    semantic_sha256,
    sha256_file,
)
from .model import load_public_transcoder
from .objective import transcoder_objective
from .schedule import build_scheduler, percentage_targets
from .snapshot import (
    iter_snapshot_records,
    load_snapshot_manifest,
    resolve_snapshot_source_root,
    validate_materialized_snapshot_lineage,
)
from .traces import iter_trace_records, load_trace_manifest, write_json_atomic


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=True) + "\n")
        handle.flush()


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _truncate_metrics(path: Path, *, max_update: int) -> None:
    if not path.is_file():
        return
    retained = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if int(value["update"]) <= max_update:
            retained.append(json.dumps(value, sort_keys=True, ensure_ascii=True))
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        ("\n".join(retained) + "\n") if retained else "",
        encoding="utf-8",
    )
    temporary.replace(path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_row_selection(path: Path | None) -> tuple[set[int] | None, dict[str, Any]]:
    if path is None:
        return None, {"format": "nvarc_training_rows_v1", "scope": "all_rows"}
    resolved = path.resolve()
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if value.get("format") not in {
        "nvarc_lr_audit_row_sample_v1",
        "nvarc_snapshot_lr_audit_row_sample_v1",
    }:
        raise ValueError(f"Unsupported training-row selection: {resolved}")
    indices = [int(item) for item in value["dataset_indices"]]
    if len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("Training-row selection must contain unique sorted indices")
    if len(indices) != int(value.get("rows", -1)):
        raise ValueError("Training-row selection count does not match its manifest")
    result = {
        "format": value["format"],
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "rows": len(indices),
        "puzzles": int(value["puzzles"]),
        "fraction": float(value["fraction"]),
        "config_sha256": str(value["config_sha256"]),
    }
    if value["format"] == "nvarc_snapshot_lr_audit_row_sample_v1":
        result["snapshot_sha256"] = str(value["snapshot_sha256"])
    return set(indices), result


def _autocast_context(matrix_compute: str, device: torch.device) -> Any:
    if matrix_compute == "bf16_autocast":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _verify_candidate_snapshot(
    destination: Path,
    *,
    expected_manifest: dict[str, Any],
    progress_percent: int,
    update_index: int,
) -> None:
    manifest_path = destination / "manifest.json"
    if not (destination / "COMPLETE").is_file() or not manifest_path.is_file():
        raise ValueError(f"Candidate snapshot is incomplete: {destination}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        **expected_manifest,
        "format": "nvarc_transcoder_candidate_v1",
        "progress_percent": progress_percent,
        "update_index": update_index,
    }
    mismatches: dict[str, Any] = {}
    for key, item in expected.items():
        actual = value.get(key)
        if key == "code_identity":
            try:
                checkpoint_code_identity_compatibility(actual, item)
            except ValueError:
                mismatches[key] = {"expected": item, "actual": actual}
            continue
        if actual != item:
            mismatches[key] = {"expected": item, "actual": actual}
    if mismatches:
        raise ValueError(f"Existing candidate snapshot mismatch: {mismatches}")


def _process_block(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    block_index: int,
    update_index: int,
    total_updates: int,
    config: NvarcTranscoderConfig,
    transcoder: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    metrics_path: Path,
    candidate_targets: dict[int, int],
    candidate_root: Path,
    snapshot_manifest: dict[str, Any],
    microbatch_tokens: int,
    matrix_compute: str,
    timing_state: dict[str, float],
) -> int:
    permutation = deterministic_block_permutation(
        x.shape[0], seed=config.data.seed, block_index=block_index
    ).to(device)
    x = x.index_select(0, permutation)
    y = y.index_select(0, permutation)
    target_to_percent = {target: percent for percent, target in candidate_targets.items()}
    batch_size = config.training.effective_tokens
    for start in range(0, x.shape[0], batch_size):
        if update_index >= total_updates:
            break
        effective_x = x[start : start + batch_size]
        effective_y = y[start : start + batch_size]
        effective_count = int(effective_x.shape[0])
        update_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        metric_sums = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "sparsity": 0.0,
            "l0_mean": 0.0,
        }
        for micro_start in range(0, effective_count, microbatch_tokens):
            batch_x = effective_x[micro_start : micro_start + microbatch_tokens]
            batch_y = effective_y[micro_start : micro_start + microbatch_tokens]
            if matrix_compute == "fp32_tf32":
                batch_x = batch_x.float()
                batch_y = batch_y.float()
            with _autocast_context(matrix_compute, device):
                y_hat, features = transcoder(batch_x)
                objective = transcoder_objective(
                    y_hat,
                    batch_y,
                    features,
                    transcoder.W_dec,
                    sparsity_coeff=config.model.sparsity_coeff,
                    c_coeff=config.model.c_coeff,
                )
            weight = objective.token_count / effective_count
            (objective.loss * weight).backward()
            for key in metric_sums:
                metric_sums[key] += float(getattr(objective, key).detach().cpu()) * weight
            del y_hat, features, objective
        grad_norm = torch.nn.utils.clip_grad_norm_(
            transcoder.parameters(), config.training.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        update_index += 1
        update_seconds = time.perf_counter() - update_started
        timing_state["processed_tokens"] += effective_count
        session_seconds = time.perf_counter() - timing_state["started"]
        end_to_end_tps = timing_state["processed_tokens"] / max(session_seconds, 1e-9)
        remaining_tokens = max(
            0,
            total_updates * config.training.effective_tokens
            - timing_state["completed_before_session"]
            - timing_state["processed_tokens"],
        )
        grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu())
        _append_jsonl(
            metrics_path,
            {
                "update": update_index,
                "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "block": block_index,
                "tokens": effective_count,
                **metric_sums,
                "grad_norm": grad_norm_value,
                "grad_clipped": grad_norm_value > config.training.max_grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                "update_seconds": update_seconds,
                "optimizer_tokens_per_second": effective_count / max(update_seconds, 1e-9),
                "session_end_to_end_tokens_per_second": end_to_end_tps,
                "estimated_remaining_seconds": remaining_tokens / max(end_to_end_tps, 1e-9),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
        )
        if update_index in target_to_percent:
            percent = target_to_percent[update_index]
            destination = candidate_root / f"p{percent:03d}"
            if (destination / "COMPLETE").is_file():
                _verify_candidate_snapshot(
                    destination,
                    expected_manifest=snapshot_manifest,
                    progress_percent=percent,
                    update_index=update_index,
                )
            else:
                save_weight_snapshot(
                    destination,
                    transcoder=transcoder,
                    manifest={
                        **snapshot_manifest,
                        "format": "nvarc_transcoder_candidate_v1",
                        "progress_percent": percent,
                        "update_index": update_index,
                    },
                )
    return update_index


def train_layer(
    config: NvarcTranscoderConfig,
    *,
    layer: int,
    learning_rate: float,
    device: torch.device,
    resume: bool = True,
    metrics_path: Path | None = None,
    run_id: str = "production",
    row_indices_path: Path | None = None,
    microbatch_tokens: int = 8192,
    matrix_compute: str | None = None,
    trace_snapshot_path: Path | None = None,
    source_trace_snapshot_path: Path | None = None,
    output_root_override: Path | None = None,
    resumable_percent_override: int | None = None,
) -> dict[str, Any]:
    if not 0 <= layer < config.model.num_layers:
        raise ValueError(f"Layer must be in 0..{config.model.num_layers - 1}")
    if learning_rate not in config.training.learning_rate_candidates:
        raise ValueError(
            "Learning rate must be one of the audited candidates; pass the selected value"
        )
    if device.type != "cuda":
        raise RuntimeError("Production NVARC transcoder training requires CUDA")
    if microbatch_tokens not in {2048, 4096, 8192}:
        raise ValueError("microbatch_tokens must be one of 2048, 4096, or 8192")
    matrix_compute = matrix_compute or config.training.matrix_compute
    if matrix_compute not in {"fp32_tf32", "bf16_autocast"}:
        raise ValueError("Unsupported matrix compute mode")
    if matrix_compute == "fp32_tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
    resumable_percent = (
        int(resumable_percent_override)
        if resumable_percent_override is not None
        else config.training.resumable_percent
    )
    if resumable_percent <= 0 or 100 % resumable_percent:
        raise ValueError("resumable_percent must be a positive divisor of 100")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, digits, dot, underscore, and dash")
    _set_seed(config.data.seed)

    output_root = (
        output_root_override.resolve()
        if output_root_override is not None
        else Path(config.model.output_root).resolve()
    )
    layer_root = output_root / "runs" / run_id / f"layer_{layer:02d}"
    layer_root.mkdir(parents=True, exist_ok=True)
    base_identity = load_or_create_base_identity(
        config, destination=output_root / "base_identity.json"
    )
    base_identity_sha = semantic_sha256(base_identity)
    if trace_snapshot_path is None:
        trace_manifest = load_trace_manifest(
            config.traces.train_root,
            expected_rows=config.data.expected_train_rows,
            expected_events=config.data.expected_train_events,
            expected_base_identity_sha256=base_identity_sha,
            expected_config_sha256=config.semantic_sha256,
        )
        trace_source_path = str(Path(config.traces.train_root).resolve())
        trace_source_kind = "finalized_full"
        trace_rows = config.data.expected_train_rows
        trace_puzzles = config.data.expected_train_puzzles
        final_cursor_index = config.data.expected_train_rows

        def training_records() -> Any:
            return iter_trace_records(config.traces.train_root)

    else:
        resolved_snapshot = trace_snapshot_path.resolve()
        trace_manifest = load_snapshot_manifest(resolved_snapshot)
        materialized_selection = trace_manifest.get("row_selection")
        source_snapshot_manifest = None
        if materialized_selection is not None:
            if source_trace_snapshot_path is None:
                raise ValueError(
                    "Materialized snapshot training requires its source snapshot path"
                )
            source_snapshot_manifest = load_snapshot_manifest(
                source_trace_snapshot_path
            )
            validate_materialized_snapshot_lineage(
                trace_manifest, source_snapshot_manifest
            )
        expected_snapshot = {
            "source_config_sha256": config.semantic_sha256,
            "base_identity_sha256": base_identity_sha,
        }
        actual_snapshot = {
            "source_config_sha256": trace_manifest.get("source_config_sha256"),
            "base_identity_sha256": trace_manifest.get("base_identity_sha256"),
        }
        identity_root_manifest = source_snapshot_manifest or trace_manifest
        expected_snapshot["source_trace_root"] = str(
            Path(config.traces.train_root).resolve()
        )
        actual_snapshot["source_trace_root"] = str(
            resolve_snapshot_source_root(identity_root_manifest)
        )
        snapshot_mismatches = {
            key: {"expected": value, "actual": actual_snapshot.get(key)}
            for key, value in expected_snapshot.items()
            if actual_snapshot.get(key) != value
        }
        if snapshot_mismatches:
            raise ValueError(
                f"Frozen trace snapshot identity mismatch: {snapshot_mismatches}"
            )
        trace_source_path = str(resolved_snapshot)
        trace_source_kind = "frozen_snapshot"
        trace_rows = int(trace_manifest["rows"])
        trace_puzzles = int(trace_manifest["puzzles"])
        final_cursor_index = int(trace_manifest["cursor_end"])

        def training_records() -> Any:
            return iter_snapshot_records(trace_manifest)

    selected_indices, row_selection = _load_row_selection(row_indices_path)
    materialized_selection = trace_manifest.get("row_selection")
    if materialized_selection is not None:
        if selected_indices is not None:
            raise ValueError(
                "A materialized row-selection snapshot cannot take row_indices_path"
            )
        if not isinstance(materialized_selection, dict):
            raise ValueError("Frozen snapshot row_selection must be a mapping")
        row_selection = dict(materialized_selection)
        eligible_tokens = int(trace_manifest["eligible_tokens"])
    elif selected_indices is None:
        eligible_tokens = int(trace_manifest["eligible_tokens"])
    else:
        expected_audit = {
            "rows": round(trace_rows * 0.05),
            "puzzles": trace_puzzles,
            "fraction": 0.05,
            "config_sha256": config.semantic_sha256,
        }
        if trace_source_kind == "frozen_snapshot":
            expected_audit.update(
                {
                    "format": "nvarc_snapshot_lr_audit_row_sample_v1",
                    "snapshot_sha256": trace_manifest["snapshot_sha256"],
                }
            )
        else:
            expected_audit["format"] = "nvarc_lr_audit_row_sample_v1"
        mismatches = {
            key: {"expected": value, "actual": row_selection.get(key)}
            for key, value in expected_audit.items()
            if row_selection.get(key) != value
        }
        if mismatches:
            raise ValueError(f"LR-audit row sample contract mismatch: {mismatches}")
        if (
            not selected_indices
            or min(selected_indices) < 0
            or max(selected_indices) >= config.data.expected_train_rows
        ):
            raise ValueError("Training-row selection contains an out-of-range index")
        found_indices: set[int] = set()
        eligible_tokens = 0
        for record in training_records():
            if record.dataset_index in selected_indices:
                found_indices.add(record.dataset_index)
                eligible_tokens += sum(record.eligible_mask)
        if found_indices != selected_indices:
            missing = sorted(selected_indices - found_indices)
            raise ValueError(
                "Training-row selection is not contained in the frozen trace source: "
                f"{missing[:20]}"
            )
    total_updates = math.ceil(eligible_tokens / config.training.effective_tokens)
    public_path = Path(config.model.public_transcoder_root) / f"layer_{layer}.safetensors"
    transcoder, imported = load_public_transcoder(
        public_path,
        layer=layer,
        d_model=config.model.d_model,
        d_feature=config.model.d_feature,
    )
    transcoder.to(device=device, dtype=torch.float32).train()
    optimizer_kwargs: dict[str, Any] = {
        "lr": learning_rate,
        "betas": (config.training.beta1, config.training.beta2),
        "eps": config.training.eps,
        "weight_decay": config.training.weight_decay,
    }
    if torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(transcoder.parameters(), **optimizer_kwargs)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_updates,
        warmup_fraction=config.training.warmup_fraction,
        cooldown_fraction=config.training.cooldown_fraction,
    )
    identity = {
        "layer": layer,
        "learning_rate": learning_rate,
        "config_sha256": config.semantic_sha256,
        "base_identity_sha256": base_identity_sha,
        "trace_manifest_sha256": semantic_sha256(trace_manifest),
        "trace_source": {
            "kind": trace_source_kind,
            "path": trace_source_path,
            "rows": trace_rows,
            "puzzles": trace_puzzles,
            "snapshot_sha256": trace_manifest.get("snapshot_sha256"),
        },
        "public_transcoder_path": str(public_path.resolve()),
        "public_transcoder_sha256": sha256_file(public_path),
        "public_source_keys": imported.source_keys,
        "total_updates": total_updates,
        "eligible_tokens": eligible_tokens,
        "run_id": run_id,
        "row_selection": row_selection,
        "microbatch_tokens": microbatch_tokens,
        "matrix_compute": matrix_compute,
        "resumable_percent": resumable_percent,
        "code_identity": package_code_identity(),
    }
    cursor = TrainingCursor(0, 0, 0, 0)
    latest = layer_root / "latest"
    if resume and (latest / "COMPLETE").is_file():
        cursor = load_resumable_checkpoint(
            latest,
            transcoder=transcoder,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_manifest=identity,
        )
        transcoder.to(device)
        _optimizer_to_device(optimizer, device)

    candidate_root = layer_root / "candidates"
    candidate_root.mkdir(exist_ok=True)
    p000 = candidate_root / "p000"
    if cursor.update_index == 0:
        if (p000 / "COMPLETE").is_file():
            _verify_candidate_snapshot(
                p000,
                expected_manifest=identity,
                progress_percent=0,
                update_index=0,
            )
        else:
            save_weight_snapshot(
                p000,
                transcoder=transcoder,
                manifest={
                    **identity,
                    "format": "nvarc_transcoder_candidate_v1",
                    "progress_percent": 0,
                    "update_index": 0,
                },
            )
    else:
        _verify_candidate_snapshot(
            p000,
            expected_manifest=identity,
            progress_percent=0,
            update_index=0,
        )

    model, _tokenizer = load_frozen_step399(
        config.base.hf_checkpoint, config.base.tokenizer, device=device
    )
    capture = LayerActivationCapture(model, layer)
    buffer = ActivationBlockBuffer(
        capacity=config.training.tokens_per_block,
        d_model=config.model.d_model,
        device=device,
        dtype=torch.bfloat16,
    )
    candidate_targets = percentage_targets(total_updates, config.training.candidate_percent)
    resumable_targets = {
        percent: target
        for percent, target in percentage_targets(total_updates, resumable_percent).items()
        if percent < 100
    }
    for percent, target in candidate_targets.items():
        if target <= cursor.update_index:
            _verify_candidate_snapshot(
                candidate_root / f"p{percent:03d}",
                expected_manifest=identity,
                progress_percent=percent,
                update_index=target,
            )
    metrics_path = (
        metrics_path.resolve()
        if metrics_path is not None
        else (layer_root / "train_metrics.jsonl").resolve()
    )
    if resume:
        _truncate_metrics(metrics_path, max_update=cursor.update_index)
    elif metrics_path.exists():
        raise FileExistsError(
            f"Training metrics already exist and resume is disabled: {metrics_path}"
        )
    update_index = cursor.update_index
    block_index = cursor.block_index
    next_cursor = cursor
    timing_state = {
        "started": time.perf_counter(),
        "processed_tokens": 0.0,
        "completed_before_session": float(
            cursor.update_index * config.training.effective_tokens
        ),
    }

    for record in training_records():
        if record.dataset_index < cursor.dataset_index:
            continue
        if selected_indices is not None and record.dataset_index not in selected_indices:
            continue
        captured = capture.capture(record, device=device)
        x = captured.mlp_input[captured.eligible_mask]
        y = captured.mlp_output[captured.eligible_mask]
        offset = cursor.eligible_offset if record.dataset_index == cursor.dataset_index else 0
        if offset > x.shape[0]:
            raise ValueError("Resume eligible offset exceeds the captured row")
        while offset < x.shape[0]:
            consumed = buffer.append(x, y, start=offset)
            offset += consumed
            if not buffer.full:
                break
            block_x, block_y = buffer.values()
            update_index = _process_block(
                x=block_x,
                y=block_y,
                block_index=block_index,
                update_index=update_index,
                total_updates=total_updates,
                config=config,
                transcoder=transcoder,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                metrics_path=metrics_path,
                candidate_targets=candidate_targets,
                candidate_root=candidate_root,
                snapshot_manifest=identity,
                microbatch_tokens=microbatch_tokens,
                matrix_compute=matrix_compute,
                timing_state=timing_state,
            )
            block_index += 1
            buffer.reset()
            next_cursor = TrainingCursor(
                dataset_index=(record.dataset_index if offset < x.shape[0] else record.dataset_index + 1),
                eligible_offset=(offset if offset < x.shape[0] else 0),
                block_index=block_index,
                update_index=update_index,
            )
            crossed = any(
                cursor.update_index < target <= update_index
                for target in resumable_targets.values()
            )
            if crossed:
                save_resumable_checkpoint(
                    latest,
                    transcoder=transcoder,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    cursor=next_cursor,
                    manifest=identity,
                )
                cursor = next_cursor
        if offset >= x.shape[0]:
            next_cursor = TrainingCursor(
                record.dataset_index + 1, 0, block_index, update_index
            )
        del captured, x, y

    if buffer.count:
        block_x, block_y = buffer.values()
        update_index = _process_block(
            x=block_x,
            y=block_y,
            block_index=block_index,
            update_index=update_index,
            total_updates=total_updates,
            config=config,
            transcoder=transcoder,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            metrics_path=metrics_path,
            candidate_targets=candidate_targets,
            candidate_root=candidate_root,
            snapshot_manifest=identity,
            microbatch_tokens=microbatch_tokens,
            matrix_compute=matrix_compute,
            timing_state=timing_state,
        )
        block_index += 1
        buffer.reset()
    final_cursor = TrainingCursor(final_cursor_index, 0, block_index, update_index)
    if update_index != total_updates:
        raise RuntimeError(f"Training ended at update {update_index}, expected {total_updates}")
    save_resumable_checkpoint(
        latest,
        transcoder=transcoder,
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=final_cursor,
        manifest=identity,
    )
    result = {
        "format": "nvarc_transcoder_layer_training_v1",
        **identity,
        "status": "complete",
        "final_cursor": final_cursor.__dict__,
        "candidate_root": str(candidate_root),
        "latest": str(latest),
    }
    write_json_atomic(layer_root / "training_manifest.json", result)
    return result

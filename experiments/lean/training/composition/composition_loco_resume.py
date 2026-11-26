"""Verify complete LOCO recovery state and preserve discarded replay suffixes."""

from __future__ import annotations

import json
import math
import os
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import torch

from .composition_loco import TOTAL_STEPS, VALIDATION_STEPS, sha256_file
from .composition_loco_config import CompositionLOCOConfig
from .composition_loco_vector422_spec import VECTOR422_PROTOCOL_ID, VECTOR422_SOURCE_KEYS


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"resume has missing/malformed {name}")
    return value


def validate_full_resume_state(
    payload: Mapping[str, Any],
    *,
    config: CompositionLOCOConfig,
    expected_step: int | None = None,
    world_size: int | None = None,
    require_cuda_rng: bool = True,
) -> dict[str, Any]:
    """Check a CPU-loaded full checkpoint without changing process RNG state.

    This is a structural/content check; the trainer additionally performs
    strict model/optimizer loading and validates selection files on disk.
    Legacy checkpoints do not contain skip counters: unknown stays unknown.
    """
    payload = _mapping(payload, "payload")
    run_total_steps = int(config.training.get("total_steps", TOTAL_STEPS))
    validation_steps = tuple(
        int(value) for value in config.training.get("validation_steps", VALIDATION_STEPS)
    )
    if (
        payload.get("format") != "s5_composition_loco_resume_v1"
        or payload.get("config_sha256") != config.computed_sha256
        or payload.get("route") != config.route
        or payload.get("fold_id") != config.fold_id
    ):
        raise ValueError("composition LOCO resume identity changed")
    step = int(payload.get("global_step", -1))
    if not 0 < step <= run_total_steps or (expected_step is not None and step != expected_step):
        raise ValueError("composition LOCO resume step changed")
    model = _mapping(payload.get("model_state_dict"), "model_state_dict")
    for key, tensor in model.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("resume model state contains a non-tensor entry")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"resume model state is non-finite: {key}")
    optimizer = _mapping(payload.get("optimizer_state_dict"), "optimizer_state_dict")
    states = _mapping(optimizer.get("state"), "optimizer moments")
    groups = optimizer.get("param_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("resume optimizer param_groups missing")
    parameter_ids = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("params"), list):
            raise ValueError("resume optimizer parameter group malformed")
        parameter_ids.extend(group["params"])
        if not math.isfinite(float(group.get("lr", float("nan")))):
            raise ValueError("resume optimizer learning rate missing/non-finite")
    if len(set(parameter_ids)) != len(parameter_ids) or set(states) != set(parameter_ids):
        raise ValueError("resume optimizer moment coverage changed")
    for value in states.values():
        state = _mapping(value, "optimizer parameter state")
        mean, variance = state.get("exp_avg"), state.get("exp_avg_sq")
        if (
            not isinstance(mean, torch.Tensor)
            or not isinstance(variance, torch.Tensor)
            or mean.shape != variance.shape
            or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(variance).all())
            or not 0 < float(state.get("step", -1)) <= step
        ):
            raise ValueError("resume Adam moments/step missing or invalid")
    scheduler = _mapping(payload.get("scheduler_state_dict"), "scheduler_state_dict")
    if (
        int(scheduler.get("last_epoch", -1)) != step
        or int(scheduler.get("_step_count", -1)) != step + 1
        or not isinstance(scheduler.get("base_lrs"), list)
        or len(scheduler["base_lrs"]) != len(groups)
        or scheduler.get("_last_lr") != [group["lr"] for group in groups]
        or not isinstance(scheduler.get("lr_lambdas"), list)
        or len(scheduler["lr_lambdas"]) != len(groups)
    ):
        raise ValueError("resume scheduler state/cursor disagrees with optimizer")
    if config.route == "direct_sft":
        scaler = _mapping(payload.get("scaler_state_dict"), "scaler_state_dict")
        for key in ("scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"):
            if key not in scaler or not math.isfinite(float(scaler[key])):
                raise ValueError(f"resume GradScaler state missing/invalid: {key}")
        if float(scaler["scale"]) <= 0:
            raise ValueError("resume GradScaler scale must be positive")
    world = int(os.environ.get("WORLD_SIZE", "1")) if world_size is None else int(world_size)
    rng = payload.get("rng_by_rank")
    if not isinstance(rng, list) or len(rng) != world:
        raise ValueError("resume RNG rank coverage changed")
    for value in rng:
        state = _mapping(value, "rank RNG state")
        if set(("torch", "python", "numpy", "cuda")) - set(state):
            raise ValueError("resume RNG components missing")
        try:
            torch.Generator(device="cpu").set_state(state["torch"])
            random.Random().setstate(state["python"])
            np.random.RandomState().set_state(state["numpy"])
        except (TypeError, ValueError, RuntimeError, KeyError) as exc:
            raise ValueError("resume CPU RNG state invalid") from exc
        cuda = state["cuda"]
        if require_cuda_rng and (not isinstance(cuda, list) or len(cuda) != world):
            raise ValueError("resume CUDA RNG device coverage changed")
        if cuda is not None:
            if not isinstance(cuda, list) or any(
                not isinstance(item, torch.Tensor)
                or item.dtype != torch.uint8 or item.ndim != 1 or item.numel() == 0
                for item in cuda
            ):
                raise ValueError("resume CUDA RNG state invalid")
    selection = _mapping(payload.get("selection_state"), "selection_state")
    if (
        selection.get("format") != "s5_composition_loco_selection_state_v1"
        or selection.get("config_sha256") != config.computed_sha256
        or selection.get("route") != config.route
        or selection.get("fold_id") != config.fold_id
        or selection.get("heldout_task") != config.heldout_task
        or selection.get("visible_tasks") != list(config.train_tasks)
        or selection.get("validation_steps") != list(validation_steps)
    ):
        raise ValueError("resume selection identity changed")
    if config.protocol_id in {
        "s5_composition_loco_direct_retention4_v1",
        "s5_composition_loco_direct_retention32_12k_v1",
        VECTOR422_PROTOCOL_ID,
    }:
        retention_label = (
            "vector422" if config.protocol_id == VECTOR422_PROTOCOL_ID
            else "direct_retention32" if "retention32" in config.protocol_id
            else "direct_retention4"
        )
        retention_format = f"s5_composition_loco_{retention_label}_resume_v1"
        retention_state = _mapping(payload.get("retention_state"), "retention_state")
        if (
            retention_state.get("format") != retention_format
            or retention_state.get("protocol_id") != config.protocol_id
        ):
            raise ValueError(f"{retention_label} resume provenance is missing or malformed")
        cursor = _mapping(retention_state.get("cursor"), "retention cursor")
        if (
            cursor.get("format") != "s5_composition_loco_retention_cursor_v1"
            or int(cursor.get("next_global_step", -1)) != step
        ):
            raise ValueError(f"{retention_label} resume sampler cursor disagrees with step")
        # During training retention records only the step-zero reference;
        # the selected-checkpoint atomic evaluation is performed after the
        # visible selection is frozen and therefore is not part of a resume
        # checkpoint's history.
        if selection.get("atomic_validation_steps") != [0]:
            raise ValueError(f"{retention_label} resume atomic-validation schedule changed")
        if config.protocol_id == VECTOR422_PROTOCOL_ID:
            sources = _mapping(retention_state.get("vector422_sources"), "Vector422 condition identities")
            if set(sources) != set(VECTOR422_SOURCE_KEYS):
                raise ValueError("Vector422 resume condition identity coverage changed")
            for key in VECTOR422_SOURCE_KEYS:
                _mapping(sources[key], f"Vector422 {key} identity")
    _mapping(selection.get("validation_bank_identity"), "selection validation bank identity")
    evaluations = selection.get("evaluations")
    expected = [value for value in validation_steps if value <= step]
    if not isinstance(evaluations, list) or [row.get("step") for row in evaluations] != expected:
        raise ValueError("resume selection history missing/out of order")
    # The full recorded counts and checkpoint hashes are checked against the
    # preserved files by _restore_selection_state before any archive mutation.
    for key in ("best_step", "best_score", "best_correct"):
        if key not in selection:
            raise ValueError(f"resume selection missing {key}")
    skips = payload.get("nonfinite_gradient_steps")
    since = int(payload.get("nonfinite_gradient_steps_known_since", step if skips is None else 0))
    observed = int(payload.get("nonfinite_gradient_steps_since_known", 0 if skips is None else skips))
    if not 0 <= since <= step or not 0 <= observed <= step - since:
        raise ValueError("resume optimizer skip counter range changed")
    if skips is not None and (since != 0 or int(skips) != observed):
        raise ValueError("resume known optimizer skip counter inconsistent")
    return {
        "global_step": step,
        "world_size": world,
        "model_tensor_count": len(model),
        "optimizer_state_count": len(states),
        "nonfinite_gradient_steps": None if skips is None else int(skips),
        "nonfinite_gradient_steps_known_since": since,
        "nonfinite_gradient_steps_since_known": observed,
        "structural_state_valid": True,
    }


def validate_optimizer_layout(payload: Mapping[str, Any], optimizer: torch.optim.Optimizer) -> None:
    """Bind the saved Adam state to the freshly constructed run's parameters."""
    saved = payload["optimizer_state_dict"]
    groups = saved["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("resume optimizer group count differs from current model")
    for previous, current in zip(groups, optimizer.param_groups, strict=True):
        if len(previous["params"]) != len(current["params"]):
            raise ValueError("resume optimizer parameter count differs from current model")
        for field in ("betas", "weight_decay", "eps", "amsgrad", "maximize", "initial_lr", "name"):
            if previous.get(field) != current.get(field):
                raise ValueError(f"resume optimizer configuration changed: {field}")
        for key, parameter in zip(previous["params"], current["params"], strict=True):
            state = saved["state"][key]
            if state["exp_avg"].shape != parameter.shape or state["exp_avg_sq"].shape != parameter.shape:
                raise ValueError("resume optimizer moment shape differs from current parameter")


def _inside(path: Path, root: Path) -> Path:
    actual = path.resolve()
    if actual == root or not actual.is_relative_to(root):
        raise ValueError(f"recovery path escapes run directory: {path}")
    return actual


def load_verified_resume(
    checkpoint_path: Path,
    *,
    config: CompositionLOCOConfig,
    output_dir: Path | None = None,
    require_cuda_rng: bool = True,
    expected_cache_identity: Mapping[str, Any] | None = None,
    expected_validation_identity: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Read/verify recorded file SHA, complete CPU payload and data provenance."""
    output = Path(config.output_dir if output_dir is None else output_dir).resolve()
    checkpoint = _inside(Path(checkpoint_path), output)
    if checkpoint.parent != output / "checkpoints" or not checkpoint.is_file():
        raise ValueError("resume must be an existing checkpoint in this run")
    match = re.fullmatch(r"resume_step(\d+)\.pt", checkpoint.name)
    if match is None:
        raise ValueError("resume must name a full resume_step checkpoint")
    manifest = output / "checkpoint_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    matches = [row for row in records if row.get("kind") == "full_resume" and row.get("path") == checkpoint.name]
    if not matches:
        raise ValueError("resume has no recorded checkpoint manifest identity")
    digest, size = sha256_file(checkpoint), checkpoint.stat().st_size
    for record in matches:
        if (
            record.get("payload_sha256") != digest
            or record.get("payload_bytes") != size
            or record.get("config_sha256") != config.computed_sha256
            or int(record.get("global_step", -1)) != int(match[1])
        ):
            raise ValueError("resume checkpoint manifest hash/size/identity mismatch")
    resolved = json.loads((output / "resolved_config.json").read_text(encoding="utf-8"))
    if (
        resolved.get("config_sha256") != config.computed_sha256
        or resolved.get("route") != config.route
        or resolved.get("fold_id") != config.fold_id
        or resolved.get("data") != dict(config.data)
    ):
        raise ValueError("resume resolved config/data provenance changed")
    cache = _mapping(resolved.get("cache_identity"), "resolved cache identity")
    validation = _mapping(resolved.get("validation_bank_identity"), "resolved validation identity")
    if expected_cache_identity is not None and cache != dict(expected_cache_identity):
        raise ValueError("resume cache identity changed")
    if expected_validation_identity is not None and validation != dict(expected_validation_identity):
        raise ValueError("resume validation identity changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    evidence = validate_full_resume_state(
        payload, config=config, expected_step=int(match[1]), require_cuda_rng=require_cuda_rng
    )
    if payload["selection_state"]["validation_bank_identity"] != validation:
        raise ValueError("resume checkpoint validation identity differs from resolved config")
    evidence.update({"path": str(checkpoint), "payload_sha256": digest, "payload_bytes": size})
    return payload, evidence


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def archive_replay_suffix(
    output_dir: Path,
    *,
    resume_step: int,
    resume_sha256: str,
    total_steps: int = TOTAL_STEPS,
) -> Path:
    """Preserve old files before replay; never delete artifacts to free space.

    Call only after model/optimizer/RNG/selection restoration succeeds on all
    ranks. Moves are individual files on the same filesystem; <=resume model
    and validation files, including the selected model, stay in place.
    """
    output = output_dir.resolve()
    if not output.is_dir() or not 0 < int(resume_step) <= int(total_steps):
        raise ValueError("invalid recovery output directory/step")
    archive_root = _inside(output / "recovery_archive", output)
    if archive_root.exists():
        for prior in archive_root.glob("*/archive.json"):
            if json.loads(prior.read_text(encoding="utf-8")).get("status") != "complete":
                raise ValueError(f"incomplete previous recovery archive requires inspection: {prior}")
    moves: list[Path] = []
    for folder, pattern in (
        ("checkpoints", r"(?:model|resume)_step(\d+)\.pt"),
        ("evaluations", r"visible_step(\d+)\.json"),
    ):
        directory = _inside(output / folder, output)
        if directory.exists():
            for path in directory.iterdir():
                match = re.fullmatch(pattern, path.name)
                temporary = re.fullmatch(r"\.(?:model|resume)_step(\d+)\.pt\.tmp\.\d+", path.name)
                if (match and int(match[1]) > resume_step) or (temporary and int(temporary[1]) > resume_step):
                    if not path.is_file():
                        raise ValueError(f"recovery suffix is not a file: {path}")
                    moves.append(_inside(path, output))
    for name in ("complete.json", "selection.json"):
        path = _inside(output / name, output)
        if path.exists():
            moves.append(path)
    snapshots: dict[Path, str | None] = {}
    for name, field in (
        ("train_metrics.jsonl", "step"),
        ("optimizer_skip_events.jsonl", "step"),
        ("checkpoint_manifest.jsonl", "global_step"),
    ):
        path = _inside(output / name, output)
        if path.is_file():
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            keep = []
            for line in lines:
                if not line.strip():
                    continue
                row = json.loads(line)
                if field not in row:
                    raise ValueError(f"recovery history has no {field}: {path}")
                if int(row[field]) <= resume_step:
                    keep.append(line if line.endswith("\n") else line + "\n")
            snapshots[path] = "".join(keep)
    resolved = _inside(output / "resolved_config.json", output)
    if resolved.is_file():
        snapshots[resolved] = None
    archive = _inside(archive_root / f"resume_step{resume_step:06d}_{uuid4().hex}", output)
    archive.mkdir(parents=True, exist_ok=False)
    plan = {
        "format": "s5_composition_loco_replay_archive_v1", "status": "in_progress",
        "resume_step": int(resume_step), "resume_sha256": resume_sha256,
        "moved_files": [str(path.relative_to(output)) for path in moves],
        "snapshot_files": [str(path.relative_to(output)) for path in snapshots],
        "storage_freed_bytes": 0,
    }
    _write_json(archive / "archive.json", plan)
    # Snapshot evidence first: a quota failure here cannot remove the source.
    for path in snapshots:
        target = _inside(archive / path.relative_to(output), output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in moves:
        target = _inside(archive / path.relative_to(output), output)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"recovery archive target already exists: {target}")
        path.rename(target)
    for path, content in snapshots.items():
        if content is not None:
            temporary = path.with_name(f".{path.name}.recovery.tmp.{os.getpid()}")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)
    _write_json(archive / "archive.json", {**plan, "status": "complete"})
    return archive

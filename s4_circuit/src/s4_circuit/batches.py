from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from nvarc_qwen3.rearc_scale_schedule import build_rearc_schedule, indices_sha256

from s3circuit.nvarc_circuit.storage import load_safetensors_bundle, write_safetensors_bundle

from .config import S4R1Config
from .contracts import BATCH_MANIFEST_FORMAT, EXPECTED_SCHEDULE_SHA256, semantic_sha256


def build_logical_batch_manifest(
    config: S4R1Config,
    *,
    puzzle_names: Sequence[str],
) -> dict[str, Any]:
    from s2_hypernet.nvarc_pg.config import load_experiment_config

    s2 = load_experiment_config(config.s2_config)
    schedule = build_rearc_schedule(
        puzzle_names,
        rows_per_puzzle=8,
        seed=config.effective_data_seed,
        logical_epochs=s2.profile.train_epochs,
        global_batch_size=s2.training.global_row_batch,
    )
    if schedule.schedule_sha256 != config.expected_schedule_sha256:
        raise ValueError(
            f"Logical RE-ARC schedule changed: {schedule.schedule_sha256}"
        )
    indices = torch.from_numpy(schedule.indices.copy()).to(dtype=torch.int32)
    expected = s2.training.max_steps * s2.training.global_row_batch
    if indices.numel() != expected:
        raise ValueError(f"Logical schedule has {indices.numel()} rows, expected {expected}")
    indices = indices.reshape(s2.training.max_steps, s2.training.global_row_batch)
    identity = {
        "format": BATCH_MANIFEST_FORMAT,
        "s4_seed": config.seed,
        "schedule_sha256": schedule.schedule_sha256,
        "max_steps": s2.training.max_steps,
        "global_row_batch": s2.training.global_row_batch,
    }
    if config.replication is not None:
        identity["replication"] = config.semantic_dict["replication"]
    semantic = semantic_sha256(identity)
    return write_safetensors_bundle(
        config.batch_manifest,
        tensors={"row_indices": indices},
        metadata={
            "format": BATCH_MANIFEST_FORMAT,
            "status": "complete",
            "s4_seed": config.seed,
            "s2_profile": s2.profile.profile_id,
            "schedule_sha256": schedule.schedule_sha256,
            "max_steps": s2.training.max_steps,
            "global_row_batch": s2.training.global_row_batch,
            "rows": int(indices.numel()),
            "manifest_semantic_sha256": semantic,
            **({"replication": identity["replication"]} if config.replication else {}),
        },
    )


def validate_logical_batch_manifest(
    path: str | Path,
    *,
    expected_steps: int = 650,
    expected_batch: int = 256,
    config: S4R1Config | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    tensors, metadata = load_safetensors_bundle(
        path,
        expected_format=BATCH_MANIFEST_FORMAT,
    )
    rows = tensors.get("row_indices")
    if rows is None or rows.dtype != torch.int32 or rows.shape != (
        expected_steps,
        expected_batch,
    ):
        raise ValueError("Logical batch manifest tensor shape or dtype changed")
    expected_schedule = (
        EXPECTED_SCHEDULE_SHA256 if config is None else config.expected_schedule_sha256
    )
    if metadata.get("schedule_sha256") != expected_schedule:
        raise ValueError("Logical batch schedule identity changed")
    if config is not None:
        if metadata.get("s4_seed") != config.seed:
            raise ValueError("Logical batch initialization seed changed")
        expected_replication = config.semantic_dict.get("replication")
        if metadata.get("replication") != expected_replication:
            raise ValueError("Logical batch replication identity changed")
        if config.replication is not None:
            actual_schedule = indices_sha256(rows.to(dtype=torch.int64).numpy())
            if actual_schedule != expected_schedule:
                raise ValueError("Logical batch rows do not match the declared seed schedule")
    identity = {
        key: metadata.get(key)
        for key in ("format", "s4_seed", "schedule_sha256", "max_steps", "global_row_batch")
    }
    if metadata.get("replication") is not None:
        identity["replication"] = metadata["replication"]
    if semantic_sha256(identity) != metadata.get("manifest_semantic_sha256"):
        raise ValueError("Logical batch manifest semantic identity changed")
    return rows, metadata


def rank_rows_for_step(
    row_indices: torch.Tensor,
    *,
    step_index: int,
    rank: int,
    world_size: int,
    micro_batch_size: int,
) -> tuple[int, ...]:
    """Recreate S2's [accumulation, world, micro] sharding exactly."""

    if row_indices.ndim != 2 or not 0 <= step_index < row_indices.shape[0]:
        raise ValueError("Logical batch step is outside the manifest")
    if not 0 <= rank < world_size or micro_batch_size <= 0:
        raise ValueError("Invalid rank topology")
    global_batch = int(row_indices.shape[1])
    local_width = world_size * micro_batch_size
    if global_batch % local_width:
        raise ValueError("Logical batch is not divisible by world size * micro batch")
    shaped = row_indices[step_index].reshape(-1, world_size, micro_batch_size)
    return tuple(int(value) for value in shaped[:, rank, :].reshape(-1).tolist())

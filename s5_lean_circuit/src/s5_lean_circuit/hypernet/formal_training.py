"""Formal S5 HyperNet A/B paired-matrix training runner."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import (
    ATOMIC_SPECIALIZATION_GROUPS,
    CIRCUIT_HEAVY80_GROUPS,
    FORMAL_GRID_GROUPS,
    FORMAL_QC_SWEEP_GROUPS,
    FORMAL_PREFIX,
    HYBRID_APPEND6_GROUPS,
    PAIRED_GROUPS,
    PROTOCOL_ID,
    S5HyperNetRunConfig,
    load_run_config,
    write_run_config,
)
from .alignment import ALIGNMENT_MODES
from ..circuit.model_io import load_tokenizer, sha256_file
from .circuit_normalizer import FixedLog1pVector397
from .circuit_projection import retained_indices_from_dead_mask
from .losses import (
    lora_gate_at_step,
    lora_gate_warmup_steps,
    scale_microbatch_objective_for_ddp,
)
from .formal_evaluation import (
    MatchedValidationBank,
    evaluate_matched_allnine,
    load_matched_validation_bank,
)
from .packed_cache import (
    PackedTokenizedVector397Dataset,
    load_base_ce_artifact,
    packed_cache_identity,
)
from .vector397_source import (
    Vector397FingerprintSource,
    load_vector397_fingerprint_source,
)
from .raw_weight import S5RawWeightCache
from .runtime import (
    S5HyperNet,
    load_s5_base_model,
    resolve_common_initialization_root,
    validate_candidate_initialization,
    validate_common_initialization,
)


PADDING_REPEATS = 128
STEPS_PER_LOGICAL_EPOCH = 2_344
SMOKE_STEPS = 100
SMOKE_WARMUP_STEPS = 20
FORMAL_MILESTONE_STEPS = (7_629, 15_258, 38_000)
ATOMIC_SPECIALIZATION_MATRIX_ID = (
    "s5_hypernet_query_only_atomic_specialization_v1"
)
ATOMIC_SPECIALIZATION_MILESTONE_STEPS = (
    7_629,
    11_443,
    15_258,
    22_887,
    30_516,
    38_000,
)
ATOMIC_VALIDATION_TASKS = ("S_and", "S_or_l", "S_or_r", "S_id")
CIRCUIT_HEAVY_MILESTONE_STEPS = (
    7_629,
    11_443,
    15_258,
    22_887,
    30_516,
    38_000,
)
CIRCUIT_HEAVY80_CELL_IDS = {
    *(f"M{index:02d}" for index in range(24)),
    *(f"L{index:02d}" for index in range(8)),
    *(f"D{index:02d}" for index in range(8)),
}
RAW_ABLATION_MATRIX_IDS = {
    "s5_hypernet_circuit_local20_v1",
    "s5_hypernet_circuit_dual16_v1",
    "s5_hypernet_raw_query_cross16_v1",
    "s5_hypernet_circuit_p2e16_v1",
}
HYBRID_APPEND6_MATRIX_ID = "s5_hypernet_circuit_raw_append6_v1"
FORMAL32_MATRIX_IDS = {
    "s5_hypernet_formal32_lr_gate_v1",
    "s5_hypernet_formal32_lr_gate_zscore_v1",
    "s5_hypernet_formal32_qc_lr_v1",
}
SAME_TASK_SHUFFLE_SOURCE_VERSION = "same_task_query_length_derangement_v2"
SAME_TASK_QUERY_LENGTH_FALLBACK_VERSION = "singleton_donor_cycle_v1"
GLOBAL_SHUFFLE_SOURCE_VERSION = "global_half_offset_derangement_v1"


def _milestone_steps_for_matrix(matrix_id: str) -> tuple[int, ...]:
    if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID:
        return ATOMIC_SPECIALIZATION_MILESTONE_STEPS
    if matrix_id in {
        "s5_hypernet_circuit_heavy120_v1",
        "s5_hypernet_circuit_heavy80_v2",
        *RAW_ABLATION_MATRIX_IDS,
    }:
        return CIRCUIT_HEAVY_MILESTONE_STEPS
    return FORMAL_MILESTONE_STEPS


def _gradient_reachability_groups(
    condition_source: str,
    completed_step: int,
) -> tuple[str, ...] | None:
    """Return the route groups that must be reachable at a checkpoint step.

    The formal S5 routes use a zero-initialized LoRA-B output. Consequently,
    the first backward pass can update that B head, while a branch injected
    before the B head only becomes reachable on the following backward pass.
    Raw-Weight has one additional zero-initialized projection after PG block 1:
    ``raw_output`` is reachable on backward step 2 and ``raw_encoder`` only
    after that projection has been updated, on backward step 3.
    """

    source = str(condition_source).lower()
    if source not in {
        "circuit",
        "query_only",
        "raw_weight",
        "circuit_raw_append",
    }:
        raise ValueError(f"unknown S5 condition source: {condition_source}")
    step = int(completed_step)
    if step == 2:
        required = ["pg", "query"]
        if source == "circuit":
            required.append("circuit")
        elif source == "raw_weight":
            required.append("raw_output")
        elif source == "circuit_raw_append":
            required.extend(("circuit_global", "raw_output"))
        return tuple(required)
    if step == 3 and source == "raw_weight":
        return ("pg", "query", "raw_encoder", "raw_output")
    if step == 3 and source == "circuit_raw_append":
        return (
            "pg",
            "query",
            "circuit_global",
            "raw_encoder",
            "raw_output",
        )
    return None


SUPPORTED_MATRIX_IDS = {
    "s5_hypernet_ab_g0_g3_paired_v1",
    "s5_hypernet_runtime_sweep_mb_gc_v1",
    "s5_hypernet_execution_trial_v1",
    *FORMAL32_MATRIX_IDS,
    "s5_hypernet_circuit_heavy120_v1",
    "s5_hypernet_circuit_heavy80_v2",
    "s5_hypernet_a_shuffle_sweep_v1",
    "s5_hypernet_a_alignment_sweep_v1",
    "s5_hypernet_query_only_formal_v1",
    ATOMIC_SPECIALIZATION_MATRIX_ID,
    "s5_hypernet_raw_weight_formal_v1",
    HYBRID_APPEND6_MATRIX_ID,
    *RAW_ABLATION_MATRIX_IDS,
}
ALIGNMENT_SWEEP_CELLS: dict[str, tuple[str, float, bool, float, float]] = {
    "QCA00": ("none", 0.0, False, 0.0, 0.0),
    "QCA01": ("global_inbatch_infonce_v1", 0.005, False, 0.0, 0.0),
    "QCA02": ("global_inbatch_infonce_v1", 0.01, False, 0.0, 0.0),
    "QCA03": ("global_inbatch_infonce_v1", 0.02, False, 0.0, 0.0),
    "QCA04": ("global_inbatch_infonce_v1", 0.05, False, 0.0, 0.0),
    "QCA05": ("same_task_inbatch_infonce_v1", 0.02, False, 0.0, 0.0),
    "QCA06": ("hybrid_inbatch_infonce_v1", 0.02, False, 0.0, 0.0),
    "QCA07": ("none", 0.0, True, 0.1, 0.01),
    "QCA08": ("hybrid_inbatch_infonce_v1", 0.02, True, 0.1, 0.01),
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) & 0xFFFFFFFF)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _init_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("formal S5 HyperNet training requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl", init_method="env://")
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return rank, world_size, local_rank, device


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _all_reduce(value: Tensor) -> Tensor:
    result = value.detach().clone()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
    return result


def _gather_objects(value: Any, world_size: int) -> list[Any]:
    if world_size == 1:
        return [value]
    result: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(result, value)
    return result


def balanced_epoch_order(size: int, *, seed: int, epoch: int) -> Tensor:
    """Shuffle all unique events and append a rotating 128-event pad window."""

    if int(size) != FORMAL_PREFIX:
        raise ValueError("balanced formal epochs require exactly 1.2M events")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + int(epoch) * 1_000_003)
    unique = torch.randperm(size, generator=generator)
    repeat_start = (int(epoch) * PADDING_REPEATS) % int(size)
    repeats = (torch.arange(PADDING_REPEATS) + repeat_start) % int(size)
    order = torch.cat((unique, repeats), dim=0)
    if order.numel() != STEPS_PER_LOGICAL_EPOCH * 512:
        raise RuntimeError("balanced epoch cardinality changed")
    return order


def rank_indices_for_step(
    global_step: int,
    *,
    dataset_size: int,
    seed: int,
    rank: int,
    world_size: int,
    cached_epoch: tuple[int, Tensor] | None = None,
) -> tuple[Tensor, tuple[int, Tensor]]:
    """Return one rank's deterministic slice for a zero-based step."""

    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    epoch, batch_in_epoch = divmod(global_step, STEPS_PER_LOGICAL_EPOCH)
    if cached_epoch is None or cached_epoch[0] != epoch:
        cached_epoch = (
            epoch,
            balanced_epoch_order(dataset_size, seed=seed, epoch=epoch),
        )
    order = cached_epoch[1]
    global_indices = order[batch_in_epoch * 512 : (batch_in_epoch + 1) * 512]
    if global_indices.numel() != 512 or 512 % world_size:
        raise RuntimeError("formal global batch sharding changed")
    local = 512 // world_size
    return global_indices[rank * local : (rank + 1) * local], cached_epoch


@dataclass(frozen=True)
class SameTaskShufflePlan:
    """A deterministic training source permutation and its protocol metadata."""

    sources: Tensor
    metadata: dict[str, Any]


def build_same_task_shuffle_plan(
    dataset: PackedTokenizedVector397Dataset,
) -> SameTaskShufflePlan:
    """Build a same-task derangement, recording unavoidable length fallbacks.

    Most groups are cycled independently by ``(task_id, query_length)``.  A
    singleton length group cannot be deranged on its own.  When a task has
    exactly one such singleton, the smallest available non-singleton group is
    used as a deterministic donor: a three-cycle is formed with two donor
    records (and any remaining donor records are cycled separately).  If the
    donor has exactly three records, all four records are put in one cycle so
    that no leftover singleton is created.  The formal 1.2M bank takes the
    three-cycle path.
    """

    task_ids = dataset.payload["task_ids"].to(dtype=torch.long)
    query_lengths = dataset.payload["query_lengths"].to(dtype=torch.long)
    keys = task_ids * 256 + query_lengths
    order = torch.argsort(keys, stable=True)
    sources = torch.full((len(dataset),), -1, dtype=torch.long)
    groups_by_task: dict[int, list[Tensor]] = {}
    singletons_by_task: dict[int, list[int]] = {}
    start = 0
    while start < order.numel():
        key = int(keys[order[start]].item())
        end = start + 1
        while end < order.numel() and int(keys[order[end]].item()) == key:
            end += 1
        group = order[start:end]
        task = int(task_ids[group[0]].item())
        groups_by_task.setdefault(task, []).append(group)
        if group.numel() >= 2:
            sources[group] = torch.roll(group, shifts=1, dims=0)
        else:
            singletons_by_task.setdefault(task, []).append(int(group.item()))
        start = end

    fallback_records: list[dict[str, Any]] = []
    for task in sorted(singletons_by_task):
        singleton = singletons_by_task[task]
        if len(singleton) >= 2:
            singleton_tensor = torch.tensor(singleton, dtype=torch.long)
            sources[singleton_tensor] = torch.roll(
                singleton_tensor, shifts=1, dims=0
            )
            fallback_records.append(
                {
                    "task_id": task,
                    "singleton_indices": singleton,
                    "singleton_query_lengths": [
                        int(query_lengths[index].item()) for index in singleton
                    ],
                    "mode": "singleton_cycle",
                }
            )
            continue

        singleton_index = singleton[0]
        donor_candidates = [
            group
            for group in groups_by_task.get(task, ())
            if group.numel() >= 2
        ]
        if not donor_candidates:
            raise RuntimeError("same-task shuffled training derangement is impossible")
        # Use the smallest donor group.  A two-record donor gives the intended
        # minimal three-cycle; larger donors are handled without leaving a
        # fixed point in the unconsumed remainder.
        donor = min(
            donor_candidates,
            key=lambda group: (
                int(group.numel()),
                int(group[0].item()),
            ),
        )
        donor_values = [int(value) for value in donor.tolist()]
        if donor.numel() == 3:
            cycle = torch.tensor(
                [singleton_index, *donor_values], dtype=torch.long
            )
            sources[cycle] = torch.roll(cycle, shifts=1, dims=0)
            fallback_mode = "four_cycle"
        else:
            first, second = donor_values[:2]
            cycle = torch.tensor(
                [singleton_index, first, second], dtype=torch.long
            )
            sources[cycle] = torch.tensor(
                [first, second, singleton_index], dtype=torch.long
            )
            remainder = donor[2:]
            if remainder.numel() >= 2:
                sources[remainder] = torch.roll(remainder, shifts=1, dims=0)
            fallback_mode = "three_cycle"
        fallback_records.append(
            {
                "task_id": task,
                "singleton_index": singleton_index,
                "singleton_query_length": int(query_lengths[singleton_index].item()),
                "donor_query_length": int(query_lengths[donor[0]].item()),
                "donor_size": int(donor.numel()),
                "mode": fallback_mode,
            }
        )
    if bool((sources < 0).any()) or bool((sources == torch.arange(len(dataset))).any()):
        raise RuntimeError("same-task shuffled training source coverage changed")
    if bool((task_ids[sources] != task_ids).any()):
        raise RuntimeError("same-task shuffled training task identity changed")
    if sources.unique().numel() != len(dataset):
        raise RuntimeError("same-task shuffled training mapping is not bijective")
    metadata = {
        "version": SAME_TASK_SHUFFLE_SOURCE_VERSION,
        "grouping": "task_id_query_length",
        "query_length_fallback_version": SAME_TASK_QUERY_LENGTH_FALLBACK_VERSION,
        # The operation count is useful for provenance; the record count is
        # the number of target rows whose source length differs.
        "query_length_fallback_count": len(fallback_records),
        "query_length_fallback_record_count": int(
            (query_lengths[sources] != query_lengths).sum().item()
        ),
        "query_length_fallback_records": fallback_records,
    }
    return SameTaskShufflePlan(sources=sources, metadata=metadata)


def build_same_task_shuffle_sources(
    dataset: PackedTokenizedVector397Dataset,
) -> Tensor:
    """Backward-compatible tensor-only wrapper for the shuffle plan helper."""

    return build_same_task_shuffle_plan(dataset).sources


def build_global_half_offset_shuffle_plan(
    dataset: PackedTokenizedVector397Dataset,
) -> SameTaskShufflePlan:
    """Build the globally mismatched S4-style half-offset assignment."""

    size = len(dataset)
    if size < 2:
        raise ValueError("global shuffled training requires at least two events")
    sources = torch.roll(torch.arange(size, dtype=torch.long), shifts=size // 2)
    if bool((sources == torch.arange(size)).any()):
        raise RuntimeError("global shuffled training produced a fixed point")
    if sources.unique().numel() != size:
        raise RuntimeError("global shuffled training mapping is not bijective")
    return SameTaskShufflePlan(
        sources=sources,
        metadata={
            "version": GLOBAL_SHUFFLE_SOURCE_VERSION,
            "grouping": "global_event_order",
            "offset": size // 2,
            "query_length_fallback_version": None,
            "query_length_fallback_count": 0,
            "query_length_fallback_record_count": 0,
            "query_length_fallback_records": [],
        },
    )


def build_negative_event_mask(size: int, *, fraction: float, seed: int) -> Tensor:
    """Build a deterministic reciprocal-fraction event selection mask."""

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("negative event fraction must be in (0,1]")
    period = round(1.0 / float(fraction))
    if period < 1 or abs(float(fraction) - 1.0 / period) > 1.0e-9:
        raise ValueError("negative event fraction must be a reciprocal integer")
    indices = torch.arange(int(size), dtype=torch.long)
    return (indices + int(seed)) % period == 0


def _unwrap(model: nn.Module) -> S5HyperNet:
    return model.module if hasattr(model, "module") else model  # type: ignore[return-value]


def _rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    torch.set_rng_state(value["torch"])
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    if torch.cuda.is_available() and value.get("cuda") is not None:
        torch.cuda.set_rng_state_all(value["cuda"])


def _checkpoint_identity(
    *,
    config: S5HyperNetRunConfig,
    cache_metadata: Mapping[str, Any],
    base_ce_metadata: Mapping[str, Any],
    common_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any] | None,
    world_size: int,
    vector_source_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "format": "s5_hypernet_formal_checkpoint_v1",
        "protocol_id": PROTOCOL_ID,
        "config_sha256": config.config_sha256,
        "candidate": config.model.candidate,
        "group": config.model.group,
        "planned_total_steps": config.model.total_steps,
        "world_size": world_size,
        "micro_batch_size": config.model.micro_batch_size,
        "packed_cache_identity_sha256": cache_metadata["identity_sha256"],
        "packed_cache_payload_sha256": cache_metadata["payload_sha256"],
        "base_ce_payload_sha256": base_ce_metadata["payload_sha256"],
        "common_init_sha256": common_manifest["aggregate_sha256"],
        "candidate_init_sha256": (
            candidate_manifest["aggregate_sha256"]
            if candidate_manifest is not None
            else None
        ),
        "condition_source": str(config.extra.get("condition_source", "circuit")),
        "fusion_mode": str(config.extra.get("fusion_mode", "append")),
        "circuit_residual_scale": float(config.extra.get("circuit_residual_scale", 1.0)),
        "circuit_global_residual_scale": float(
            config.extra.get("circuit_global_residual_scale", 1.0)
        ),
        "circuit_block_residual_scale": float(
            config.extra.get("circuit_block_residual_scale", 1.0)
        ),
        "fusion_injection_blocks": list(
            int(value) for value in config.extra.get("fusion_injection_blocks", ())
        ),
        "circuit_lr_multipliers": dict(
            config.extra.get("circuit_lr_multipliers", {})
        ),
        "raw_weight_cache_sha256": (
            str(config.extra.get("raw_weight_cache_sha256", ""))
            if config.extra.get("condition_source")
            in {"raw_weight", "circuit_raw_append"}
            else None
        ),
        "raw_fusion_mode": str(config.extra.get("raw_fusion_mode", "p2e")),
    }
    # Keep historical fixed-log1p checkpoint identities byte-compatible; any
    # normalized run carries an explicit condition-source identity.
    if config.data.input_treatment != "fixed_log1p_v1" or vector_source_identity is not None:
        identity.update(
            {
                "input_treatment": str(config.data.input_treatment),
                "input_treatment_identity": config.input_treatment_identity,
            }
        )
        if vector_source_identity is not None:
            identity["vector397_source"] = dict(vector_source_identity)
        if config.data.fingerprint_normalizer_artifact_sha256 is not None:
            identity["fingerprint_normalizer_artifact_sha256"] = (
                config.data.fingerprint_normalizer_artifact_sha256
            )
    return identity


def _save_model_only(
    path: Path,
    *,
    core: S5HyperNet,
    global_step: int,
    identity: Mapping[str, Any],
    write_integrity: bool = False,
) -> dict[str, Any] | None:
    payload = {
        **dict(identity),
        "kind": "model_only",
        "global_step": int(global_step),
        "model_state_dict": core.trainable_state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    if not write_integrity:
        return None
    sidecar_body = {
        "format": "s5_hypernet_model_checkpoint_integrity_v1",
        "status": "complete",
        "kind": "model_only",
        "global_step": int(global_step),
        "payload_path": path.name,
        "payload_sha256": _file_sha256(path),
        "payload_bytes": path.stat().st_size,
        "checkpoint_identity": dict(identity),
    }
    sidecar = {
        **sidecar_body,
        "sidecar_identity_sha256": hashlib.sha256(
            json.dumps(
                sidecar_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }
    _atomic_json(path.with_suffix(path.suffix + ".json"), sidecar)
    return sidecar


def _save_resume(
    path: Path,
    *,
    core: S5HyperNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    global_step: int,
    identity: Mapping[str, Any],
    rng_by_rank: Sequence[Mapping[str, Any]],
    runtime_by_rank: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        **dict(identity),
        "kind": "full_resume",
        "global_step": int(global_step),
        "model_state_dict": core.trainable_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_by_rank": list(rng_by_rank),
        "runtime_by_rank": list(runtime_by_rank),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_resume(
    path: str | Path,
    *,
    core: S5HyperNet,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    identity: Mapping[str, Any],
    rank: int,
    world_size: int,
    runtime_state_out: dict[str, Any] | None = None,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"resume identity mismatch for {key}: {payload.get(key)!r} != {expected!r}"
            )
    if payload.get("kind") != "full_resume":
        raise ValueError("resume path is not a full-resume checkpoint")
    states = payload.get("rng_by_rank")
    if not isinstance(states, list) or len(states) != world_size:
        raise ValueError("resume checkpoint rank RNG coverage changed")
    runtime_states = payload.get("runtime_by_rank")
    if runtime_states is not None and (
        not isinstance(runtime_states, list) or len(runtime_states) != world_size
    ):
        raise ValueError("resume checkpoint rank runtime coverage changed")
    core.load_trainable_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_rng(states[rank])
    if runtime_state_out is not None:
        runtime_state_out.clear()
        runtime_state_out.update(
            {
                "runtime_by_rank": runtime_states,
                "global_step": int(payload["global_step"]),
            }
        )
    return int(payload["global_step"])


@torch.no_grad()
def _validate_step0_parity(
    core: S5HyperNet,
    dataset: PackedTokenizedVector397Dataset,
    base_ce: Tensor,
    *,
    device: torch.device,
) -> None:
    batch = dataset.collate_indices(
        torch.arange(2), device=device, frozen_base_ce=base_ce
    )
    base_logits = core.base_logits(batch["input_ids"])
    modes: list[tuple[str, Tensor, Tensor, tuple[int, ...] | None]] = [
        (
            "matched",
            batch["vector397"],
            torch.ones(2, device=device, dtype=torch.bool),
            None,
        ),
        (
            "absent",
            batch["vector397"],
            torch.zeros(2, device=device, dtype=torch.bool),
            None,
        ),
        (
            "shuffled",
            batch["vector397"].flip(0),
            torch.ones(2, device=device, dtype=torch.bool),
            None,
        ),
    ]
    if core.candidate in {"B", "AB"}:
        modes.append(
            (
                "block_permuted",
                batch["vector397"],
                torch.ones(2, device=device, dtype=torch.bool),
                (1, 2, 3, 4, 0),
            )
        )
    for mode, vectors, present, permutation in modes:
        state = core.generate_lora(
            batch["query_ids"],
            batch["query_mask"],
            vectors,
            circuit_present=present,
            block_permutation=permutation,
            gate_multiplier=0.0,
        )
        with core.controller.activate(state):
            adapted = core.base_logits(batch["input_ids"])
        if not bool(torch.equal(base_logits, adapted)):
            difference = float(
                (base_logits.float() - adapted.float()).abs().max().item()
            )
            raise RuntimeError(
                f"step-0 {mode} Base parity failed: max_abs={difference}"
            )


def _scheduler(optimizer: torch.optim.Optimizer, config: S5HyperNetRunConfig) -> Any:
    total_steps = config.model.total_steps
    model_warmup_steps = max(1, int(round(total_steps * config.model.warmup_ratio)))
    raw_warmup_steps = max(
        1,
        int(config.extra.get("raw_warmup_steps", model_warmup_steps)),
    )

    def multiplier(scheduler_step: int, *, warmup_steps: int) -> float:
        position = int(scheduler_step) + 1
        if position <= warmup_steps:
            return float(position) / float(warmup_steps)
        progress = min(
            1.0,
            float(position - warmup_steps)
            / float(max(1, total_steps - warmup_steps)),
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = config.model.cosine_min_lr_ratio
        return floor + (1.0 - floor) * cosine

    lambdas = [
        (
            lambda step, warmup_steps=(
                raw_warmup_steps
                if str(group.get("name", "")).startswith("raw_")
                else model_warmup_steps
            ): multiplier(step, warmup_steps=warmup_steps)
        )
        for group in optimizer.param_groups
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def _resolve_stop_after_step(
    planned_total_steps: int,
    stop_after_step: int | None,
    *,
    max_steps: int | None = None,
) -> int:
    """Keep formal target execution independent of the legacy smoke horizon."""

    if stop_after_step is None:
        return int(planned_total_steps if max_steps is None else max_steps)
    if max_steps is not None:
        raise ValueError("--stop-after-step cannot be combined with --max-steps smoke mode")
    if not 0 < int(stop_after_step) <= int(planned_total_steps):
        raise ValueError("stop_after_step must be within the planned training horizon")
    return int(stop_after_step)


def _write_completion_marker(
    output: Path,
    marker: Mapping[str, Any],
    *,
    complete: bool,
    stop_after_step: int | None,
) -> dict[str, Any]:
    """Publish readiness only after both target checkpoint artifacts exist."""

    record = dict(marker)
    if stop_after_step is not None:
        actual_step = int(record["global_step"])
        if actual_step != int(stop_after_step):
            raise RuntimeError("cannot mark an unreached training target ready")
        checkpoint_paths = {
            "model_checkpoint": output / "checkpoints" / f"model_step{actual_step:07d}.pt",
            "resume_checkpoint": output / "checkpoints" / f"resume_step{actual_step:07d}.pt",
        }
        checkpoint_records = {}
        for name, path in checkpoint_paths.items():
            if not path.is_file():
                raise RuntimeError(f"target reached without {name}: {path}")
            checkpoint_records[name] = {
                "path": str(path),
                "step": actual_step,
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        record.update(
            status="target_reached",
            stop_after_step=int(stop_after_step),
            actual_steps=actual_step,
            **checkpoint_records,
        )
        _atomic_json(output / "target_reached_summary.json", record)
        _atomic_json(output / "target_reached.ready", record)
    elif complete:
        record["status"] = "complete"
        _atomic_json(output / "final.ready", record)
    else:
        record["status"] = "smoke_complete"
        _atomic_json(output / "smoke_summary.json", record)
        _atomic_json(output / "smoke.ready", record)
    return record


def train_formal_run(
    config: S5HyperNetRunConfig,
    *,
    resume: str | None = None,
    max_steps: int | None = None,
    stop_after_step: int | None = None,
) -> dict[str, Any]:
    stop_step = _resolve_stop_after_step(
        config.model.total_steps, stop_after_step, max_steps=max_steps
    )
    rank, world_size, _local_rank, device = _init_distributed()
    if world_size != config.model.expected_world_size:
        raise ValueError("runtime world size does not match the approved config")
    if max_steps is not None and not 0 < int(max_steps) <= config.model.total_steps:
        raise ValueError("max_steps is outside the planned token budget")
    extra = dict(config.extra)
    matrix_id = extra.get("matrix")
    if matrix_id not in SUPPORTED_MATRIX_IDS:
        raise ValueError("formal runner matrix identity changed")
    if matrix_id == "s5_hypernet_ab_g0_g3_paired_v1":
        if (
            config.model.group not in PAIRED_GROUPS
        ):
            raise ValueError("unknown legacy paired A/B group")
    if matrix_id == "s5_hypernet_runtime_sweep_mb_gc_v1":
        if (
            config.model.group != "G0"
            or config.model.total_tokens != 1_000_000_000
            or extra.get("benchmark_kind") != "runtime_sweep"
            or extra.get("benchmark_warmup_steps") != SMOKE_WARMUP_STEPS
            or not isinstance(extra.get("pg_gradient_checkpointing"), bool)
        ):
            raise ValueError("runtime-sweep config contract changed")
    if matrix_id == "s5_hypernet_execution_trial_v1":
        if (
            config.model.group != "G0"
            or config.model.total_tokens != 1_000_000_000
            or extra.get("benchmark_kind") != "execution_trial"
            or extra.get("benchmark_warmup_steps") != SMOKE_WARMUP_STEPS
            or extra.get("pg_gradient_checkpointing") is not False
        ):
            raise ValueError("execution trial config contract changed")
    if matrix_id in FORMAL32_MATRIX_IDS:
        milestone_steps = tuple(int(value) for value in extra.get("milestone_steps", ()))
        validation = dict(extra.get("matched_validation", {}))
        is_qc_sweep = matrix_id == "s5_hypernet_formal32_qc_lr_v1"
        allowed_groups = FORMAL_QC_SWEEP_GROUPS if is_qc_sweep else FORMAL_GRID_GROUPS
        allowed_query_multipliers = {0.25, 0.5, 1.0} if is_qc_sweep else {0.5}
        allowed_circuit_multipliers = {1.0, 2.0, 4.0} if is_qc_sweep else {2.0}
        anchor_run_id, anchor_group = (
            ("a_p03", "P03") if config.model.candidate == "A" else ("b_p02", "P02")
        )
        qc_sweep = dict(extra.get("qc_lr_sweep", {}))
        qc_contract_ok = (not is_qc_sweep) or (
            qc_sweep.get("format") == matrix_id
            and qc_sweep.get("anchor_run_id") == anchor_run_id
            and qc_sweep.get("anchor_group") == anchor_group
            and qc_sweep.get("query_lr_multiplier") == config.model.query_lr_multiplier
            and qc_sweep.get("circuit_lr_multiplier") == config.model.circuit_lr_multiplier
            and float(qc_sweep.get("query_lr", -1.0))
            == float(config.model.pg_lr) * float(config.model.query_lr_multiplier)
            and float(qc_sweep.get("circuit_lr", -1.0))
            == float(config.model.pg_lr) * float(config.model.circuit_lr_multiplier)
            and qc_sweep.get("grid_index") == extra.get("registration_order")
            and 0 <= int(extra.get("registration_order", -1)) < 8
        )
        if (
            config.model.group not in allowed_groups
            or config.model.total_tokens != 2_490_368_000
            or config.model.query_lr_multiplier not in allowed_query_multipliers
            or config.model.circuit_lr_multiplier not in allowed_circuit_multipliers
            or not qc_contract_ok
            or extra.get("pg_gradient_checkpointing") is not False
            or milestone_steps != FORMAL_MILESTONE_STEPS
            or validation.get("condition") != "matched"
            or validation.get("samples_per_task") != 50
            or validation.get("event_count") != 450
            or validation.get("seed_base") != 888_888_000
            or validation.get("max_new_tokens") != 50
            or not all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        ):
            raise ValueError("formal 32-cell LR/gate matrix contract changed")
    if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID:
        milestone_steps = tuple(int(value) for value in extra.get("milestone_steps", ()))
        validation = dict(extra.get("matched_validation", {}))
        if (
            config.model.candidate != "A"
            or config.model.group not in ATOMIC_SPECIALIZATION_GROUPS
            or config.model.total_tokens != 2_490_368_000
            or config.model.circuit_lr_multiplier != 2.0
            or extra.get("condition_source") != "query_only"
            or extra.get("pg_gradient_checkpointing") is not False
            or milestone_steps != ATOMIC_SPECIALIZATION_MILESTONE_STEPS
            or validation.get("condition") != "matched"
            or validation.get("task_types") != list(ATOMIC_VALIDATION_TASKS)
            or validation.get("samples_per_task") != 50
            or validation.get("event_count") != 200
            or validation.get("seed_base") != 888_888_000
            or validation.get("max_new_tokens") != 50
            or not all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        ):
            raise ValueError("Query-only atomic-specialization matrix contract changed")
    if matrix_id == "s5_hypernet_circuit_heavy120_v1":
        milestone_steps = tuple(int(value) for value in extra.get("milestone_steps", ()))
        validation = dict(extra.get("matched_validation", {}))
        fusion_mode = str(extra.get("fusion_mode", ""))
        cell_id = str(extra.get("cell_id", ""))
        if (
            config.model.candidate not in {"A", "B"}
            or config.model.group not in FORMAL_GRID_GROUPS
            or config.model.total_tokens != 2_490_368_000
            or config.model.circuit_lr_multiplier != 2.0
            or extra.get("condition_source") != "circuit"
            or fusion_mode not in {"f2", "f3i"}
            or cell_id not in {
                f"C{index:02d}"
                for index in range(16)
                if index != 13
            }
            or extra.get("pg_gradient_checkpointing") is not False
            or milestone_steps != CIRCUIT_HEAVY_MILESTONE_STEPS
            or validation.get("condition") != "matched"
            or validation.get("samples_per_task") != 50
            or validation.get("event_count") != 450
            or validation.get("seed_base") != 888_888_000
            or validation.get("max_new_tokens") != 50
            or not all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        ):
            raise ValueError("Circuit-heavy120 fusion matrix contract changed")
    if matrix_id == "s5_hypernet_circuit_heavy80_v2":
        milestone_steps = tuple(int(value) for value in extra.get("milestone_steps", ()))
        validation = dict(extra.get("matched_validation", {}))
        fusion_mode = str(extra.get("fusion_mode", ""))
        cell_id = str(extra.get("cell_id", ""))
        lr_map = dict(extra.get("circuit_lr_multipliers", {}))
        allowed_groups = {**FORMAL_GRID_GROUPS, **CIRCUIT_HEAVY80_GROUPS}
        injection_blocks = tuple(int(value) for value in extra.get("fusion_injection_blocks", ()))
        residual_scale = float(extra.get("circuit_residual_scale", -1.0))
        if (
            config.model.candidate != "AB"
            or config.model.group not in allowed_groups
            or config.model.total_tokens != 2_490_368_000
            or config.model.circuit_lr_multiplier != 2.0
            or extra.get("condition_source") != "circuit"
            or fusion_mode not in {"ab_joint_f3i", "split_f3i"}
            or cell_id not in CIRCUIT_HEAVY80_CELL_IDS
            or extra.get("pg_gradient_checkpointing") is not False
            or milestone_steps != CIRCUIT_HEAVY_MILESTONE_STEPS
            or set(lr_map) != {"global", "block", "fusion"}
            or any(float(value) not in {1.5, 2.0, 3.0, 4.0} for value in lr_map.values())
            or not injection_blocks
            or any(value < 1 or value >= 12 for value in injection_blocks)
            or not math.isfinite(residual_scale)
            or residual_scale <= 0.0
            or validation.get("condition") != "matched"
            or validation.get("samples_per_task") != 50
            or validation.get("event_count") != 450
            or validation.get("seed_base") != 888_888_000
            or validation.get("max_new_tokens") != 50
            or not all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        ):
            raise ValueError("Circuit-heavy80 fusion matrix contract changed")
    if matrix_id in {
        "s5_hypernet_query_only_formal_v1",
        "s5_hypernet_raw_weight_formal_v1",
    }:
        validation = dict(extra.get("matched_validation", {}))
        expected_source = (
            "query_only"
            if matrix_id == "s5_hypernet_query_only_formal_v1"
            else "raw_weight"
        )
        if (
            config.model.candidate != "A"
            or config.model.group not in FORMAL_GRID_GROUPS
            or config.model.total_tokens != 2_490_368_000
            or extra.get("pg_gradient_checkpointing") is not False
            or extra.get("condition_source") != expected_source
            or tuple(int(value) for value in extra.get("milestone_steps", ()))
            != FORMAL_MILESTONE_STEPS
            or validation.get("condition") != "matched"
            or validation.get("samples_per_task") != 50
            or validation.get("event_count") != 450
            or validation.get("seed_base") != 888_888_000
            or validation.get("max_new_tokens") != 50
            or not all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        ):
            raise ValueError("S5 baseline formal matrix contract changed")
        if expected_source == "raw_weight":
            raw_cache_path = str(extra.get("raw_weight_cache_path", ""))
            if (
                not raw_cache_path
                or len(str(extra.get("raw_weight_cache_sha256", ""))) != 64
                or float(extra.get("raw_encoder_lr", -1.0)) != 4.0e-4
                or float(extra.get("raw_output_lr", -1.0)) != 1.0e-3
                or int(extra.get("raw_warmup_steps", -1)) != 20
            ):
                raise ValueError("Raw-Weight formal cache/LR contract changed")
    if matrix_id in RAW_ABLATION_MATRIX_IDS:
        validation = dict(extra.get("matched_validation", {}))
        milestone_steps = tuple(int(value) for value in extra.get("milestone_steps", ()))
        fusion_mode = str(extra.get("fusion_mode", "append"))
        condition_source = str(extra.get("condition_source", "circuit"))
        cell_id = str(extra.get("cell_id", ""))
        lr_map = dict(extra.get("circuit_lr_multipliers", {}))
        common_contract = (
            config.model.total_tokens == 2_490_368_000



            and extra.get("pg_gradient_checkpointing") is False
            and milestone_steps == CIRCUIT_HEAVY_MILESTONE_STEPS
            and validation.get("condition") == "matched"
            and validation.get("samples_per_task") == 50
            and validation.get("event_count") == 450
            and validation.get("seed_base") == 888_888_000
            and validation.get("max_new_tokens") == 50
            and all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        )
        if not common_contract:
            raise ValueError("S5 model_ablation common training contract changed")
        if matrix_id == "s5_hypernet_circuit_local20_v1":
            valid = (
                config.model.candidate == "AB"
                and config.model.group in {"P02", "P03"}
                and condition_source == "circuit"
                and fusion_mode == "ab_joint_f3i"
                and extra.get("allow_mode_specific_fusion_init") is True
                and cell_id in {f"LS{index:02d}" for index in range(20)}
                and tuple(int(value) for value in extra.get("fusion_injection_blocks", ()))
                == tuple(range(1, 12))
                and float(extra.get("circuit_residual_scale", -1.0)) == 1.0
                and set(lr_map) == {"global", "block", "fusion"}
                and all(float(value) in {1.5, 2.0, 3.0, 4.0} for value in lr_map.values())
            )
        elif matrix_id == "s5_hypernet_circuit_dual16_v1":
            valid = (
                config.model.candidate == "AB"
                and config.model.group in {"P02", "P03"}
                and condition_source == "circuit"
                and fusion_mode == "dual_f3i"
                and extra.get("allow_mode_specific_fusion_init") is True
                and cell_id in {f"DF{index:02d}" for index in range(16)}
                and tuple(int(value) for value in extra.get("fusion_injection_blocks", ()))
                == tuple(range(1, 12))
                and float(extra.get("circuit_global_residual_scale", -1.0)) in {0.5, 1.0}
                and float(extra.get("circuit_block_residual_scale", -1.0)) in {0.25, 0.5}
                and set(lr_map) == {"global", "block", "fusion"}
                and float(lr_map.get("global", -1.0)) == 2.0
                and float(lr_map.get("block", -1.0)) in {2.0, 4.0}
                and float(lr_map.get("fusion", -1.0)) in {2.0, 4.0}
            )
        elif matrix_id == "s5_hypernet_circuit_p2e16_v1":
            valid = (
                config.model.candidate == "AB"
                and config.model.group in FORMAL_GRID_GROUPS
                and condition_source == "circuit"
                and fusion_mode == "circuit_p2e"
                and extra.get("allow_mode_specific_fusion_init") is True
                and cell_id in {f"CP{index:02d}" for index in range(16)}
                and tuple(int(value) for value in extra.get("fusion_injection_blocks", ())) == (0,)
                and set(lr_map) == {"global", "block", "fusion"}
                and {key: float(value) for key, value in lr_map.items()}
                == {"global": 2.0, "block": 4.0, "fusion": 4.0}
            )
        else:
            valid = (
                config.model.candidate == "A"
                and config.model.group in FORMAL_GRID_GROUPS
                and condition_source == "raw_weight"
                and fusion_mode == "append"
                and str(extra.get("raw_fusion_mode", "")) == "query_cross"
                and cell_id in {f"RQ{index:02d}" for index in range(16)}
                and bool(str(extra.get("raw_weight_cache_path", "")))
                and len(str(extra.get("raw_weight_cache_sha256", ""))) == 64
                and float(extra.get("raw_encoder_lr", -1.0)) == 4.0e-4
                and float(extra.get("raw_output_lr", -1.0)) == 1.0e-3
                and int(extra.get("raw_warmup_steps", -1)) == 20
            )
        if not valid:
            raise ValueError(f"{matrix_id} route-specific contract changed")
    if matrix_id == HYBRID_APPEND6_MATRIX_ID:
        validation = dict(extra.get("matched_validation", {}))
        milestone_steps = tuple(int(value) for value in extra.get("milestone_steps", ()))
        cell_id = str(extra.get("cell_id", "")).upper()
        is_smoke = max_steps is not None
        expected_group = "H00" if is_smoke else cell_id
        expected_raw_lrs = {
            group: (4.0e-4, 1.0e-3) for group in HYBRID_APPEND6_GROUPS
        }
        raw_lrs = expected_raw_lrs.get(expected_group)
        valid = (
            config.model.candidate == "A"
            and config.model.group == expected_group
            and (
                (is_smoke and cell_id == "H00")
                or (not is_smoke and cell_id in HYBRID_APPEND6_GROUPS)
            )
            and config.model.total_tokens == 2_490_368_000
            and config.model.global_batch_size == 512
            and config.model.circuit_lr_multiplier == 2.0
            and extra.get("pg_gradient_checkpointing") is False
            and extra.get("condition_source") == "circuit_raw_append"
            and str(extra.get("fusion_mode", "")) == "append"
            and str(extra.get("raw_fusion_mode", "")) == "p2e"
            and bool(str(extra.get("raw_weight_cache_path", "")))
            and len(str(extra.get("raw_weight_cache_sha256", ""))) == 64
            and raw_lrs is not None
            and (
                float(extra.get("raw_encoder_lr", -1.0)),
                float(extra.get("raw_output_lr", -1.0)),
            )
            == raw_lrs
            and int(extra.get("raw_warmup_steps", -1)) == 20
            and milestone_steps == FORMAL_MILESTONE_STEPS
            and validation.get("condition") == "matched"
            and validation.get("samples_per_task") == 50
            and validation.get("event_count") == 450
            and validation.get("seed_base") == 888_888_000
            and validation.get("max_new_tokens") == 50
            and all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        )
        if not valid:
            raise ValueError("Circuit+Raw append6 matrix contract changed")
    if matrix_id == "s5_hypernet_a_shuffle_sweep_v1":
        mismatch = dict(extra.get("mismatch_training", {}))
        validation = dict(extra.get("matched_validation", {}))
        if (
            config.model.candidate != "A"
            or config.model.total_tokens != 2_490_368_000
            or extra.get("pg_gradient_checkpointing") is not False
            or tuple(int(value) for value in extra.get("milestone_steps", ()))
            != FORMAL_MILESTONE_STEPS
            or validation.get("condition") != "matched"
            or validation.get("samples_per_task") != 50
            or validation.get("event_count") != 450
            or validation.get("seed_base") != 888_888_000
            or validation.get("max_new_tokens") != 50
            or not all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
            or mismatch.get("mode") != "same_task_query_length_nll_rank"
            or not isinstance(mismatch.get("enabled"), bool)
            or float(mismatch.get("fraction", 0.25)) != 0.25
            or not 0.0 <= float(mismatch.get("coefficient", -1.0)) <= 1.0
            or not 0.0 <= float(mismatch.get("margin", -1.0)) <= 1.0
            or int(mismatch.get("seed", -1)) != config.assignment_seed
        ):
            raise ValueError("A shuffled-dependence sweep contract changed")
    if matrix_id == "s5_hypernet_a_alignment_sweep_v1":
        alignment = dict(extra.get("alignment_training", {}))
        mismatch = dict(extra.get("mismatch_training", {}))
        validation = dict(extra.get("matched_validation", {}))
        alignment_mode = str(alignment.get("mode", ""))
        alignment_coefficient = float(alignment.get("coefficient", -1.0))
        mismatch_coefficient = float(mismatch.get("coefficient", -1.0))
        cell_id = str(extra.get("alignment_cell_id", ""))
        declared_cell = (
            alignment_mode,
            alignment_coefficient,
            bool(mismatch.get("enabled", False)),
            mismatch_coefficient,
            float(mismatch.get("margin", -1.0)),
        )
        if (
            config.model.candidate != "A"
            or config.model.group != "P03"
            or config.model.total_tokens != 2_490_368_000
            or extra.get("pg_gradient_checkpointing") is not False
            or tuple(int(value) for value in extra.get("milestone_steps", ()))
            != FORMAL_MILESTONE_STEPS
            or validation.get("condition") != "matched"
            or validation.get("samples_per_task") != 50
            or validation.get("event_count") != 450
            or validation.get("seed_base") != 888_888_000
            or validation.get("max_new_tokens") != 50
            or alignment_mode not in ALIGNMENT_MODES
            or not 0.0 <= alignment_coefficient <= 1.0
            or float(alignment.get("temperature", -1.0)) != 0.1
            or int(alignment.get("projection_width", -1)) != 128
            or (alignment_mode == "none") != (alignment_coefficient == 0.0)
            or mismatch.get("mode") != "global_half_offset_nll_rank"
            or not isinstance(mismatch.get("enabled"), bool)
            or bool(mismatch.get("enabled")) != (mismatch_coefficient > 0.0)
            or float(mismatch.get("fraction", 0.25)) != 0.25
            or not 0.0 <= mismatch_coefficient <= 1.0
            or float(mismatch.get("margin", -1.0)) not in {0.0, 0.01}
            or int(mismatch.get("seed", -1)) != config.assignment_seed
            or ALIGNMENT_SWEEP_CELLS.get(cell_id) != declared_cell
            or not all(
                isinstance(validation.get(name), str) and validation.get(name)
                for name in (
                    "corpus_manifest",
                    "vector_manifest",
                    "binding_path",
                    "suite_sha256",
                    "tokenizer_sha256",
                )
            )
        ):
            raise ValueError("A query/circuit alignment sweep contract changed")
    expected_mode = "smoke" if max_steps is not None else "formal"
    if extra.get("mode") != expected_mode:
        raise ValueError("config mode and runner horizon disagree")
    if expected_mode == "formal" and matrix_id not in ({
        *FORMAL32_MATRIX_IDS,
        "s5_hypernet_circuit_heavy120_v1",
        "s5_hypernet_circuit_heavy80_v2",
        "s5_hypernet_a_shuffle_sweep_v1",
        "s5_hypernet_a_alignment_sweep_v1",
        "s5_hypernet_query_only_formal_v1",
        ATOMIC_SPECIALIZATION_MATRIX_ID,
        "s5_hypernet_raw_weight_formal_v1",
        HYBRID_APPEND6_MATRIX_ID,
    } | RAW_ABLATION_MATRIX_IDS):
        raise RuntimeError(
            "formal long training is fail-closed until the all-nine Circuit "
            "greedy evaluator and 5,000-step cadence are implemented"
        )
    if expected_mode == "smoke" and int(max_steps or -1) != SMOKE_STEPS:
        raise ValueError("approved HyperNet smoke matrices require exactly 100 steps")
    if max_steps is not None and int(max_steps) != int(extra.get("max_steps", -1)):
        raise ValueError("smoke max_steps differs from the config contract")
    fixed_treatment_identity = FixedLog1pVector397().artifact_sha256
    if config.data.input_treatment == "fixed_log1p_v1":
        if config.input_treatment_identity != fixed_treatment_identity:
            raise ValueError("run config fixed-log1p identity changed")
    elif config.data.input_treatment in {"train_zscore_v1", "train_zscore_compact381_v1"}:
        if not isinstance(config.input_treatment_identity, str) or not config.input_treatment_identity:
            raise ValueError("normalized run is missing input_treatment_identity")
        declared_artifact = config.data.fingerprint_normalizer_artifact_sha256
        if declared_artifact is not None and config.input_treatment_identity != declared_artifact:
            raise ValueError("run config z-score normalizer identity changed")
    else:
        raise ValueError("run config input treatment is unsupported")
    condition_source = str(extra.get("condition_source", "circuit"))
    if condition_source not in {
        "circuit",
        "query_only",
        "raw_weight",
        "circuit_raw_append",
    }:
        raise ValueError("unknown S5 condition_source")
    fusion_mode = str(extra.get("fusion_mode", "append")).lower()
    if fusion_mode not in {
        "append",
        "f2",
        "f3i",
        "ab_joint_f3i",
        "split_f3i",
        "dual_f3i",
        "circuit_p2e",
    }:
        raise ValueError("unknown S5 fusion_mode")
    if condition_source not in {"circuit", "circuit_raw_append"} and fusion_mode != "append":
        raise ValueError("Circuit fusion requires condition_source=circuit")
    input_treatment = str(config.data.input_treatment)
    if input_treatment not in {
        "fixed_log1p_v1",
        "train_zscore_v1",
        "train_zscore_compact381_v1",
    }:
        raise ValueError("unknown Vector397 input treatment")
    if input_treatment == "fixed_log1p_v1":
        if config.input_treatment_identity != FixedLog1pVector397().artifact_sha256:
            raise ValueError("run config fixed-log1p identity changed")
    elif condition_source not in {"circuit", "circuit_raw_append"}:
        raise ValueError(
            f"{input_treatment} is only valid for a Circuit-conditioned route"
        )
    _seed_everything(config.model.seed + rank)
    output = Path(config.output_dir).resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        if (output / "final.ready").exists() or (output / "smoke.ready").exists():
            raise FileExistsError("run output already contains a completion marker")
        if (output / "target_reached.ready").exists() and resume is None:
            raise FileExistsError("target run already reached its target; use --resume")
        write_run_config(output / "resolved_config.json", config)
    _barrier()

    cache_identity = (
        packed_cache_identity(
            archive_manifest=config.data.archive_manifest,
            tokenizer_path=config.data.tokenizer_path,
            prefix=config.data.prefix,
            n_ctx=config.data.n_ctx,
            metadata_root=config.data.metadata_root,
        )
        if rank == 0
        else None
    )
    raw_archive_manifest = json.loads(
        Path(config.data.archive_manifest).resolve().read_text(encoding="utf-8")
    )
    raw_archive_identity = str(raw_archive_manifest.get("archive_identity_sha256", ""))
    if len(raw_archive_identity) != 64:
        raise ValueError("formal raw Vector397 archive identity is missing")
    include_vector = condition_source in {"circuit", "circuit_raw_append"}
    fingerprint_manifest = config.data.resolved_fingerprint_manifest
    fingerprint_normalizer = config.data.resolved_fingerprint_normalizer
    fingerprint_scope = config.data.fingerprint_scope
    fingerprint_manifest_body: Mapping[str, Any] | None = None
    direct_normalizer = False
    if input_treatment in {"train_zscore_v1", "train_zscore_compact381_v1"}:
        if not fingerprint_manifest or not fingerprint_normalizer:
            raise ValueError(
                f"{input_treatment} requires a fingerprint manifest and normalizer path"
            )
        fingerprint_manifest_body = json.loads(
            Path(fingerprint_manifest).resolve().read_text(encoding="utf-8")
        )
        if fingerprint_manifest_body.get("source_archive_identity_sha256") != raw_archive_identity:
            raise ValueError("fingerprint source is bound to a different raw archive")
        # A root manifest plus its persisted normalizer is sufficient for the
        # cache-backed path used by the original z-score protocol.  A materialized
        # archive uses the newer Vector397FingerprintSource path below.
        direct_normalizer = (
            fingerprint_manifest_body.get("format")
            == "s5_vector397_fingerprint_bank_manifest_v1"
        )
        if input_treatment == "train_zscore_compact381_v1" and not direct_normalizer:
            # The archive path is required for compact381 only when the source
            # is not the canonical root manifest; its loader validates every shard.
            direct_normalizer = False
    dataset = PackedTokenizedVector397Dataset(
        config.data.cache_path or "",
        expected_identity=cache_identity,
        verify_payload_checksum=(rank == 0),
        include_vector=include_vector,
        fingerprint_manifest=(
            fingerprint_manifest if direct_normalizer else None
        ),
        fingerprint_normalizer=(
            fingerprint_normalizer if direct_normalizer else None
        ),
        fingerprint_scope=(
            fingerprint_scope
            if direct_normalizer
            else None
        ),
        fingerprint_normalizer_artifact_sha256=(
            config.data.fingerprint_normalizer_artifact_sha256
            if direct_normalizer
            else None
        ),
    )
    vector_source: Vector397FingerprintSource | None = None
    vector_source_identity: Mapping[str, Any] | None = None
    normalizer_for_validation: Any | None = None
    if input_treatment in {"train_zscore_v1", "train_zscore_compact381_v1"}:
        if direct_normalizer:
            normalizer_for_validation = dataset.fingerprint_normalizer
            normalizer_metadata = dataset.fingerprint_normalizer_metadata or {}
            if normalizer_for_validation is None:
                raise ValueError("direct fingerprint normalizer was not loaded")
            if config.input_treatment_identity != normalizer_metadata.get("artifact_sha256"):
                raise ValueError("run config z-score normalizer identity changed")
            vector_source_identity = {
                "format": fingerprint_manifest_body.get("format")
                if fingerprint_manifest_body is not None
                else None,
                "fingerprint_manifest_sha256": sha256_file(Path(fingerprint_manifest)),
                "fingerprint_manifest_identity_sha256": (
                    fingerprint_manifest_body.get("manifest_identity_sha256")
                    if fingerprint_manifest_body is not None
                    else None
                ),
                "fingerprint_normalizer_artifact_sha256": normalizer_metadata.get(
                    "artifact_sha256"
                ),
                "fingerprint_normalizer_sha256": normalizer_metadata.get(
                    "normalizer_sha256"
                ),
                "source_archive_identity_sha256": raw_archive_identity,
                "source_cache_identity_sha256": dataset.metadata["identity_sha256"],
                "fingerprint_events": len(dataset),
                "fingerprint_prefix": config.data.prefix,
                "fingerprint_scope": normalizer_metadata.get("scope"),
                "fingerprint_heldout_task": normalizer_metadata.get("heldout_task"),
                "condition_dim": (
                    381 if input_treatment == "train_zscore_compact381_v1" else 397
                ),
                "compact381": input_treatment == "train_zscore_compact381_v1",
            }
        else:
            vector_source = load_vector397_fingerprint_source(
                manifest_path=fingerprint_manifest,
                normalizer_path=fingerprint_normalizer,
                expected_prefix=config.data.prefix,
                expected_source_archive_identity_sha256=raw_archive_identity,
                expected_source_cache_identity_sha256=str(
                    dataset.metadata["identity_sha256"]
                ),
                compact381=input_treatment == "train_zscore_compact381_v1",
            )
            normalizer_for_validation = vector_source.normalizer
            if config.input_treatment_identity != vector_source.normalizer.artifact_sha256:
                raise ValueError("run config z-score normalizer identity changed")
            dataset.attach_condition_source(vector_source)
            vector_source_identity = vector_source.identity
    circuit_input_indices = None
    if input_treatment == "train_zscore_compact381_v1":
        if normalizer_for_validation is None:
            raise ValueError("compact381 treatment has no validated normalizer")
        circuit_input_indices = retained_indices_from_dead_mask(
            normalizer_for_validation.dead_mask
        )
        projection = extra.get("circuit_input_projection")
        if not isinstance(projection, Mapping):
            raise ValueError(
                "compact381 treatment requires circuit_input_projection metadata"
            )
        if (
            projection.get("format")
            != "s5_hypernet_vector397_dead_coordinate_projection_v1"
            or int(projection.get("input_dim", -1)) != len(circuit_input_indices)
            or int(projection.get("dead_coordinate_count", -1)) != 16
            or projection.get("normalizer_artifact_sha256")
            != normalizer_for_validation.artifact_sha256
        ):
            raise ValueError("compact381 Circuit projection identity changed")
    _barrier()
    mismatch_training = dict(extra.get("mismatch_training", {}))
    alignment_training = dict(extra.get("alignment_training", {}))
    alignment_mode = str(alignment_training.get("mode", "none"))
    alignment_coefficient = float(alignment_training.get("coefficient", 0.0))
    alignment_temperature = float(alignment_training.get("temperature", 0.1))
    alignment_projection_width = int(
        alignment_training.get("projection_width", 128)
    )
    alignment_enabled = bool(
        matrix_id == "s5_hypernet_a_alignment_sweep_v1"
        and alignment_mode != "none"
        and alignment_coefficient > 0.0
    )
    mismatch_fraction = float(mismatch_training.get("fraction", 0.25))
    mismatch_enabled = bool(
        matrix_id in {
            "s5_hypernet_a_shuffle_sweep_v1",
            "s5_hypernet_a_alignment_sweep_v1",
        }
        and mismatch_training.get("enabled", False)
        and float(mismatch_training.get("coefficient", 0.0)) > 0.0
    )
    shuffle_plan = (
        build_global_half_offset_shuffle_plan(dataset)
        if mismatch_enabled
        else None
    )
    shuffle_sources = None if shuffle_plan is None else shuffle_plan.sources
    negative_event_mask = (
        build_negative_event_mask(
            len(dataset),
            fraction=mismatch_fraction,
            seed=int(mismatch_training.get("seed", config.assignment_seed)),
        )
        if mismatch_enabled
        else None
    )
    base_ce, base_ce_metadata = load_base_ce_artifact(
        config.data.base_ce_path,
        expected_count=len(dataset),
        expected_cache_identity_sha256=dataset.metadata["identity_sha256"],
        expected_cache_payload_sha256=dataset.metadata["payload_sha256"],
        expected_base_identity=asdict(config.base),
    )
    base, base_identity = load_s5_base_model(
        config.base.architecture_config_path,
        config.base.checkpoint_path,
        device=device,
        dtype=torch.bfloat16,
        expected_identity=config.base,
    )
    raw_weight_cache = None
    if condition_source in {"raw_weight", "circuit_raw_append"}:
        raw_cache_path = Path(str(extra.get("raw_weight_cache_path", ""))).resolve()
        if not raw_cache_path.is_file():
            raise FileNotFoundError(f"S5 Raw-Weight cache is missing: {raw_cache_path}")
        raw_weight_cache = S5RawWeightCache.load(
            raw_cache_path,
            expected_sha256=str(extra.get("raw_weight_cache_sha256", "")),
            expected_base_identity=asdict(base_identity),
        )
    core = S5HyperNet(
        base,
        candidate=config.model.candidate,
        seed=config.model.seed,
        condition_source=condition_source,
        fusion_mode=fusion_mode,
        circuit_residual_scale=float(extra.get("circuit_residual_scale", 1.0)),
        circuit_global_residual_scale=float(
            extra.get("circuit_global_residual_scale", 1.0)
        ),
        circuit_block_residual_scale=float(
            extra.get("circuit_block_residual_scale", 1.0)
        ),
        fusion_injection_blocks=(
            tuple(int(value) for value in extra["fusion_injection_blocks"])
            if "fusion_injection_blocks" in extra
            else None
        ),
        circuit_lr_multipliers=(
            dict(extra["circuit_lr_multipliers"])
            if "circuit_lr_multipliers" in extra
            else None
        ),
        raw_weight_cache=raw_weight_cache,
        raw_fusion_mode=str(extra.get("raw_fusion_mode", "p2e")),
        circuit_input_indices=circuit_input_indices,
    )
    if not config.common_init_manifest:
        raise ValueError("formal runs require a prepared common init manifest")
    common_root = resolve_common_initialization_root(config.common_init_manifest)
    common_manifest = validate_common_initialization(
        common_root,
        expected_base_identity=asdict(base_identity),
        expected_seed=config.model.seed,
        expected_layout_sha256=core.layout.layout_sha256,
    )
    common_state = torch.load(
        common_root / str(common_manifest["state_path"]),
        map_location="cpu",
        weights_only=True,
    )
    core.load_common_state_dict(
        common_state,
        allow_mode_specific_fusion_init=bool(
            extra.get("allow_mode_specific_fusion_init", False)
        ),
    )
    candidate_manifest = None
    if condition_source in {"circuit", "circuit_raw_append"}:
        if not config.candidate_init_manifest:
            raise ValueError("Circuit formal runs require a candidate init manifest")
        candidate_root = resolve_common_initialization_root(config.candidate_init_manifest)
        candidate_manifest = validate_candidate_initialization(
            candidate_root,
            expected_base_identity=asdict(base_identity),
            expected_seed=config.model.seed,
            expected_candidate=config.model.candidate,
            expected_layout_sha256=core.layout.layout_sha256,
        )
        candidate_state = torch.load(
            candidate_root / str(candidate_manifest["state_path"]),
            map_location="cpu",
            weights_only=True,
        )
        core.load_circuit_state_dict(candidate_state)
    core.to(device=device)
    if core.raw_weight_conditioner is not None:
        core.raw_weight_conditioner.restore_fp32_output_path()
    core.validate_step0_initialization()
    _validate_step0_parity(core, dataset, base_ce, device=device)
    validation_bank: MatchedValidationBank | None = None
    validation_tokenizer: Any | None = None
    evaluation_steps: set[int] = set()
    if matrix_id in ({
        *FORMAL32_MATRIX_IDS,
        "s5_hypernet_circuit_heavy120_v1",
        "s5_hypernet_circuit_heavy80_v2",
        "s5_hypernet_a_shuffle_sweep_v1",
        "s5_hypernet_a_alignment_sweep_v1",
        "s5_hypernet_query_only_formal_v1",
        ATOMIC_SPECIALIZATION_MATRIX_ID,
        "s5_hypernet_raw_weight_formal_v1",
        HYBRID_APPEND6_MATRIX_ID,
    } | RAW_ABLATION_MATRIX_IDS):
        validation_config = dict(extra["matched_validation"])
        tokenizer_path = Path(config.data.tokenizer_path).resolve()
        if sha256_file(tokenizer_path) != validation_config["tokenizer_sha256"]:
            raise ValueError("formal validation tokenizer identity changed")
        validation_tokenizer = load_tokenizer(
            tokenizer_path,
            expected_sha256=validation_config["tokenizer_sha256"],
        )
        validation_bank = load_matched_validation_bank(
            corpus_manifest=validation_config["corpus_manifest"],
            vector_manifest=validation_config["vector_manifest"],
            binding_path=validation_config["binding_path"],
            expected_suite_sha256=validation_config["suite_sha256"],
            load_vectors=condition_source in {"circuit", "circuit_raw_append"},
            vector397_treatment=normalizer_for_validation,
            compact381=input_treatment == "train_zscore_compact381_v1",
        )
        evaluation_steps = set(_milestone_steps_for_matrix(matrix_id))
    pg_gradient_checkpointing = extra.get("pg_gradient_checkpointing", True)
    if not isinstance(pg_gradient_checkpointing, bool):
        raise ValueError("pg_gradient_checkpointing must be boolean")
    if pg_gradient_checkpointing:
        core.enable_gradient_checkpointing()
    else:
        core.disable_gradient_checkpointing()
    model: nn.Module
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            core,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        model = core
    core = _unwrap(model)
    optimizer = torch.optim.AdamW(
        core.parameter_groups(
            pg_lr=config.model.pg_lr,
            query_lr_multiplier=config.model.query_lr_multiplier,
            circuit_lr_multiplier=config.model.circuit_lr_multiplier,
            circuit_lr_multipliers=(
                dict(extra["circuit_lr_multipliers"])
                if "circuit_lr_multipliers" in extra
                else None
            ),
            raw_encoder_lr=(
                float(extra["raw_encoder_lr"])
                if "raw_encoder_lr" in extra
                else None
            ),
            raw_output_lr=(
                float(extra["raw_output_lr"])
                if "raw_output_lr" in extra
                else None
            ),
            weight_decay=config.model.weight_decay,
        ),
        betas=(config.model.adam_beta1, config.model.adam_beta2),
        weight_decay=0.0,
    )
    scheduler = _scheduler(optimizer, config)
    checkpoint_identity = _checkpoint_identity(
        config=config,
        cache_metadata=dataset.metadata,
        base_ce_metadata=base_ce_metadata,
        common_manifest=common_manifest,
        candidate_manifest=candidate_manifest,
        world_size=world_size,
        vector_source_identity=vector_source_identity,
    )
    global_step = 0
    resume_runtime_state: dict[str, Any] = {}
    if resume is not None:
        global_step = _load_resume(
            resume,
            core=core,
            optimizer=optimizer,
            scheduler=scheduler,
            identity=checkpoint_identity,
            rank=rank,
            world_size=world_size,
            runtime_state_out=resume_runtime_state,
        )
    if stop_after_step is not None and global_step >= stop_step:
        raise ValueError("resume checkpoint has already reached the requested stop step")
    if global_step > stop_step or (
        global_step == stop_step
        and matrix_id
        not in ({
            *FORMAL32_MATRIX_IDS,
            "s5_hypernet_circuit_heavy120_v1",
            "s5_hypernet_circuit_heavy80_v2",
            "s5_hypernet_a_shuffle_sweep_v1",
            "s5_hypernet_a_alignment_sweep_v1",
            "s5_hypernet_query_only_formal_v1",
            ATOMIC_SPECIALIZATION_MATRIX_ID,
            "s5_hypernet_raw_weight_formal_v1",
            HYBRID_APPEND6_MATRIX_ID,
        } | RAW_ABLATION_MATRIX_IDS)
    ):
        raise ValueError("resume checkpoint has already reached the requested stop step")
    if stop_after_step is not None and rank == 0:
        (output / "target_reached.ready").unlink(missing_ok=True)
    gate_warmup = lora_gate_warmup_steps(
        config.model.total_steps, config.model.gate_warmup_ratio
    )
    if rank == 0:
        _atomic_json(
            output / "run_identity.json",
            {
                **checkpoint_identity,
                "base_identity": asdict(base_identity),
                "vector397_source": (
                    None
                    if vector_source_identity is None
                    else dict(vector_source_identity)
                ),
                "target_layout": core.layout.manifest(),
                "condition_source": condition_source,
                "input_treatment": config.data.input_treatment,
                "input_treatment_identity": config.input_treatment_identity,
                "fingerprint_normalizer": (
                    {
                        "manifest": config.data.resolved_fingerprint_manifest,
                        "normalizer": config.data.resolved_fingerprint_normalizer,
                        "scope": config.data.fingerprint_scope,
                        "artifact_sha256": (
                            config.data.fingerprint_normalizer_artifact_sha256
                            or (
                                None
                                if normalizer_for_validation is None
                                else normalizer_for_validation.artifact_sha256
                            )
                        ),
                    }
                    if input_treatment
                    in {"train_zscore_v1", "train_zscore_compact381_v1"}
                    else None
                ),
                "fusion_mode": fusion_mode,
                "raw_fusion_mode": str(extra.get("raw_fusion_mode", "p2e")),
                "circuit_global_residual_scale": float(
                    extra.get("circuit_global_residual_scale", 1.0)
                ),
                "circuit_block_residual_scale": float(
                    extra.get("circuit_block_residual_scale", 1.0)
                ),
                "raw_weight_cache": (
                    {
                        "path": str(Path(str(extra["raw_weight_cache_path"])).resolve()),
                        "sha256": str(extra.get("raw_weight_cache_sha256", "")),
                    }
                    if condition_source in {"raw_weight", "circuit_raw_append"}
                    else None
                ),
                "total_tokens": config.model.total_tokens,
                "gate_max": config.model.gate_max,
                "gate_warmup_steps": gate_warmup,
                "gradient_accumulation_steps": config.model.gradient_accumulation_steps,
                "trainable_parameter_count": int(
                    sum(
                        parameter.numel()
                        for parameter in core.parameters()
                        if parameter.requires_grad
                    )
                ),
                "optimizer_group_parameter_counts": {
                    str(group.get("name", index)): int(
                        sum(parameter.numel() for parameter in group["params"])
                    )
                    for index, group in enumerate(optimizer.param_groups)
                },
                "pg_gradient_checkpointing": pg_gradient_checkpointing,
                "deployed_commit": os.environ.get("S5_DEPLOY_COMMIT"),
                "max_steps": max_steps,
                **(
                    {
                        "planned_total_steps": config.model.total_steps,
                        "stop_after_step": stop_after_step,
                    }
                    if stop_after_step is not None
                    else {}
                ),
                "milestone_steps": sorted(evaluation_steps),
                "matched_validation_bank": (
                    dict(validation_bank.identity)
                    if validation_bank is not None
                    else None
                ),
                "mismatch_training": (
                    {
                        **mismatch_training,
                        "enabled": mismatch_enabled,
                        "shuffle_source": (
                            None
                            if shuffle_plan is None
                            else shuffle_plan.metadata["version"]
                        ),
                        "shuffle_metadata": (
                            None
                            if shuffle_plan is None
                            else shuffle_plan.metadata
                        ),
                        "matched_reference_gradient": "stop_gradient",
                        "fraction": mismatch_fraction,
                    }
                    if matrix_id in {
                        "s5_hypernet_a_shuffle_sweep_v1",
                        "s5_hypernet_a_alignment_sweep_v1",
                    }
                    else None
                ),
                "alignment_training": (
                    {
                        **alignment_training,
                        "enabled": alignment_enabled,
                        "coefficient_warmup_steps": gate_warmup,
                        "representation": "shared_pg_condition_memory_v1",
                    }
                    if matrix_id == "s5_hypernet_a_alignment_sweep_v1"
                    else None
                ),
                "shuffle_source_count": (
                    None if shuffle_sources is None else int(shuffle_sources.numel())
                ),
            },
        )
    metrics_path = output / "train_metrics.jsonl"
    evaluation_root = output / "evaluations"
    evaluated_steps: set[int] = set()

    def milestone_checkpoint_reference(step: int) -> dict[str, Any]:
        checkpoint_path = output / "checkpoints" / f"model_step{step:07d}.pt"
        sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
        if rank != 0 and torch.distributed.is_initialized():
            values: list[Any] = [None]
            torch.distributed.broadcast_object_list(values, src=0)
            if not isinstance(values[0], dict):
                raise RuntimeError("rank zero did not broadcast checkpoint integrity")
            return dict(values[0])
        if not checkpoint_path.is_file() or not sidecar_path.is_file():
            raise FileNotFoundError(f"milestone checkpoint/integrity is missing: {step}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in sidecar.items()
            if key != "sidecar_identity_sha256"
        }
        expected_sidecar_sha = hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        if (
            sidecar.get("format") != "s5_hypernet_model_checkpoint_integrity_v1"
            or sidecar.get("status") != "complete"
            or sidecar.get("kind") != "model_only"
            or sidecar.get("global_step") != step
            or sidecar.get("payload_path") != checkpoint_path.name
            or sidecar.get("payload_bytes") != checkpoint_path.stat().st_size
            or sidecar.get("payload_sha256") != _file_sha256(checkpoint_path)
            or sidecar.get("checkpoint_identity") != dict(checkpoint_identity)
            or sidecar.get("sidecar_identity_sha256") != expected_sidecar_sha
        ):
            raise ValueError("milestone checkpoint integrity changed")
        reference = {
            "path": checkpoint_path.name,
            "payload_sha256": sidecar["payload_sha256"],
            "payload_bytes": sidecar["payload_bytes"],
            "sidecar_identity_sha256": sidecar["sidecar_identity_sha256"],
            "global_step": step,
            "config_sha256": config.config_sha256,
        }
        if torch.distributed.is_initialized():
            values = [reference]
            torch.distributed.broadcast_object_list(values, src=0)
        return reference

    def evaluate_if_due(step: int, *, allow_new: bool = True) -> None:
        if step not in evaluation_steps:
            return
        if validation_bank is None or validation_tokenizer is None:
            raise RuntimeError("formal milestone evaluation bank is unavailable")
        evaluation_prefix = (
            "atomic" if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID else "matched"
        )
        path = evaluation_root / f"{evaluation_prefix}_step{step:07d}.json"
        checkpoint_reference = milestone_checkpoint_reference(step)
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if (
                value.get("format")
                != (
                    "s5_hypernet_atomic_validation_v1"
                    if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID
                    else "s5_hypernet_matched_allnine_validation_v1"
                )
                or value.get("config_sha256") != config.config_sha256
                or value.get("step") != step
                or value.get("condition") != "matched"
                or value.get("validation_bank") != dict(validation_bank.identity)
                or value.get("model_checkpoint") != checkpoint_reference
                or (
                    matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID
                    and value.get("task_types") != list(ATOMIC_VALIDATION_TASKS)
                )
            ):
                raise ValueError("existing matched validation artifact changed")
            evaluated_steps.add(step)
            _barrier()
            return
        if not allow_new:
            raise FileNotFoundError(
                f"resume crossed milestone {step} without matched evaluation"
            )
        result = evaluate_matched_allnine(
            core,
            validation_tokenizer,
            validation_bank,
            device=device,
            rank=rank,
            world_size=world_size,
            gate_multiplier=config.model.gate_max,
            n_ctx=config.model.n_ctx,
            max_new_tokens=50,
            task_types=(
                ATOMIC_VALIDATION_TASKS
                if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID
                else None
            ),
        )
        if rank == 0:
            if result is None:
                raise RuntimeError("rank zero matched validation returned no result")
            _atomic_json(
                path,
                {
                    "format": (
                        "s5_hypernet_atomic_validation_v1"
                        if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID
                        else "s5_hypernet_matched_allnine_validation_v1"
                    ),
                    "config_sha256": config.config_sha256,
                    "candidate": config.model.candidate,
                    "group": config.model.group,
                    "step": step,
                    "model_checkpoint": checkpoint_reference,
                    **(
                        {"task_types": list(ATOMIC_VALIDATION_TASKS)}
                        if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID
                        else {}
                    ),
                    **result,
                },
            )
        _barrier()
        if not path.is_file():
            raise RuntimeError("matched validation artifact was not published")
        evaluated_steps.add(step)

    micro = config.model.micro_batch_size
    local_logical = config.model.global_batch_size // world_size
    if local_logical % micro:
        raise RuntimeError("microbatch no longer divides the local logical batch")
    previous_runtime_by_rank = resume_runtime_state.get("runtime_by_rank")
    previous_runtime = (
        dict(previous_runtime_by_rank[rank])
        if isinstance(previous_runtime_by_rank, list)
        else {}
    )
    step_times = [float(value) for value in previous_runtime.get("step_times", [])]
    model_checkpoint_times = [
        float(value) for value in previous_runtime.get("model_checkpoint_times", [])
    ]
    resume_checkpoint_times = [
        float(value) for value in previous_runtime.get("resume_checkpoint_times", [])
    ]
    previous_peak_memory_allocated = int(
        previous_runtime.get("peak_memory_allocated", 0)
    )
    previous_peak_memory_reserved = int(previous_runtime.get("peak_memory_reserved", 0))
    if resume is not None and previous_runtime_by_rank is not None and len(step_times) != global_step:
        raise ValueError("resume runtime step-time coverage changed")
    epoch_cache: tuple[int, Tensor] | None = None
    model.train()
    for previous_milestone in sorted(
        step for step in evaluation_steps if step <= global_step
    ):
        evaluate_if_due(
            previous_milestone,
            allow_new=previous_milestone == global_step,
        )
    while global_step < stop_step:
        local_indices, epoch_cache = rank_indices_for_step(
            global_step,
            dataset_size=len(dataset),
            seed=config.assignment_seed,
            rank=rank,
            world_size=world_size,
            cached_epoch=epoch_cache,
        )
        local_lengths = dataset.payload["input_lengths"].index_select(
            0, local_indices
        ).long()
        local_token_count = (local_lengths - 1).sum().to(device=device)
        global_token_count = _all_reduce(local_token_count)
        if int(global_token_count.item()) <= 0:
            raise RuntimeError("logical batch has no full-sequence targets")
        gate = lora_gate_at_step(
            global_step + 1,
            total_steps=config.model.total_steps,
            gate_max=config.model.gate_max,
            warmup_ratio=config.model.gate_warmup_ratio,
        )
        alignment_weight = (
            alignment_coefficient
            * min(float(global_step + 1) / float(max(1, gate_warmup)), 1.0)
            if alignment_enabled
            else 0.0
        )
        optimizer.zero_grad(set_to_none=True)
        local_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_hinge_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_delta_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_shuffle_rank_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_alignment_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_alignment_positive_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_alignment_negative_sum = torch.zeros((), device=device, dtype=torch.float64)
        local_alignment_retrieval_correct = torch.zeros(
            (), device=device, dtype=torch.float64
        )
        local_alignment_event_count = torch.zeros((), device=device, dtype=torch.long)
        local_alignment_negative_count = torch.zeros((), device=device, dtype=torch.long)
        local_alignment_same_task_negative_sum = torch.zeros(
            (), device=device, dtype=torch.float64
        )
        local_alignment_same_task_negative_count = torch.zeros(
            (), device=device, dtype=torch.long
        )
        step_started = time.perf_counter()
        micro_slices = list(local_indices.split(micro))
        for micro_index, micro_indices in enumerate(micro_slices):
            batch = dataset.collate_indices(
                micro_indices, device=device, frozen_base_ce=base_ce
            )
            sync_context = (
                nullcontext()
                if world_size == 1 or micro_index + 1 == len(micro_slices)
                else model.no_sync()  # type: ignore[attr-defined]
            )
            with sync_context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(
                        query_ids=batch["query_ids"],
                        query_mask=batch["query_mask"],
                        input_ids=batch["input_ids"],
                        labels=batch["labels"],
                        vector397=batch["vector397"],
                        frozen_base_ce=batch["frozen_base_ce"],
                        gate_multiplier=gate,
                        alignment_mode=(alignment_mode if alignment_enabled else "none"),
                        alignment_temperature=alignment_temperature,
                        alignment_projection_width=alignment_projection_width,
                        task_ids=batch["task_ids"],
                    )
                hinge_sum = result["base_preservation_hinge"].sum()
                delta_sum = result["effective_delta_l2"].sum()
                loss = scale_microbatch_objective_for_ddp(
                    loss_sum=result["loss_sum"],
                    hinge_sum=hinge_sum,
                    effective_delta_sum=delta_sum,
                    world_size=world_size,
                    global_token_count=global_token_count,
                    global_event_count=config.model.global_batch_size,
                    base_preservation_coefficient=(
                        config.model.base_preservation_coefficient
                    ),
                    effective_delta_coefficient=config.model.effective_delta_l2,
                )
                if alignment_enabled:
                    loss = loss + alignment_weight * (
                        result["alignment_loss_sum"]
                        / result["alignment_event_count"].to(
                            dtype=result["alignment_loss_sum"].dtype
                        )
                    )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite formal HyperNet loss at step {global_step + 1}"
                    )
                # Keep the matched path as the ordinary CE/regularization
                # update.  The detached per-event CE is used as the reference
                # for the second pass so both complete graphs never coexist.
                matched_loss_sum = result["loss_sum"].detach()
                matched_hinge_sum = hinge_sum.detach()
                matched_delta_sum = delta_sum.detach()
                matched_per_event_ce = result["per_event_ce"].detach()
                alignment_loss_sum = result["alignment_loss_sum"].detach()
                alignment_positive_sum = result[
                    "alignment_positive_similarity_sum"
                ].detach()
                alignment_negative_sum = result[
                    "alignment_negative_similarity_sum"
                ].detach()
                alignment_event_count = result["alignment_event_count"].detach()
                alignment_negative_count = result[
                    "alignment_negative_similarity_count"
                ].detach()
                alignment_retrieval_correct = result[
                    "alignment_retrieval_correct_sum"
                ].detach()
                alignment_same_task_negative_sum = result[
                    "alignment_same_task_negative_similarity_sum"
                ].detach()
                alignment_same_task_negative_count = result[
                    "alignment_same_task_negative_similarity_count"
                ].detach()
                loss.backward()
                del result, loss, hinge_sum, delta_sum

                shuffle_rank_metric = torch.zeros(
                    (), device=device, dtype=torch.float64
                )
                if mismatch_enabled:
                    if shuffle_sources is None:
                        raise RuntimeError(
                            "mismatch training enabled without shuffle sources"
                        )
                    if negative_event_mask is None:
                        raise RuntimeError(
                            "mismatch training enabled without negative event mask"
                        )
                    selected = negative_event_mask.index_select(0, micro_indices)
                    negative_indices = micro_indices[selected]
                    if negative_indices.numel() > 0:
                        shuffled_indices = shuffle_sources.index_select(
                            0, negative_indices
                        )
                        shuffled_batch = dataset.collate_indices(
                            negative_indices,
                            device=device,
                            frozen_base_ce=base_ce,
                        )
                        shuffled_batch["vector397"] = dataset.condition_vectors(
                            shuffled_indices
                        ).to(
                            device=device, non_blocking=True
                        )
                        shuffle_result = model(
                            query_ids=shuffled_batch["query_ids"],
                            query_mask=shuffled_batch["query_mask"],
                            input_ids=shuffled_batch["input_ids"],
                            labels=shuffled_batch["labels"],
                            vector397=shuffled_batch["vector397"],
                            frozen_base_ce=shuffled_batch["frozen_base_ce"],
                            gate_multiplier=gate,
                        )
                        matched_selected_ce = matched_per_event_ce[selected]
                        shuffle_rank_sum = F.relu(
                            float(mismatch_training["margin"])
                            + matched_selected_ce
                            - shuffle_result["per_event_ce"]
                        ).sum()
                        rank_loss = float(mismatch_training["coefficient"]) * (
                            shuffle_rank_sum
                            * float(world_size)
                            / float(config.model.global_batch_size)
                        )
                        if not bool(torch.isfinite(rank_loss)):
                            raise FloatingPointError(
                                "non-finite formal HyperNet shuffled ranking loss "
                                f"at step {global_step + 1}"
                            )
                        shuffle_rank_metric = shuffle_rank_sum.detach()
                        rank_loss.backward()
            local_loss_sum += matched_loss_sum.double()
            local_hinge_sum += matched_hinge_sum.double()
            local_delta_sum += matched_delta_sum.double()
            local_alignment_loss_sum += alignment_loss_sum.double()
            local_alignment_positive_sum += alignment_positive_sum.double()
            local_alignment_negative_sum += alignment_negative_sum.double()
            local_alignment_retrieval_correct += alignment_retrieval_correct.double()
            local_alignment_event_count += alignment_event_count.long()
            local_alignment_negative_count += alignment_negative_count.long()
            local_alignment_same_task_negative_sum += (
                alignment_same_task_negative_sum.double()
            )
            local_alignment_same_task_negative_count += (
                alignment_same_task_negative_count.long()
            )
            if mismatch_enabled:
                local_shuffle_rank_sum += shuffle_rank_metric
        grad_groups = core.group_gradient_norms()
        required_groups = _gradient_reachability_groups(
            condition_source,
            global_step + 1,
        )
        if required_groups is not None:
            unreachable = [
                name
                for name in required_groups
                if not math.isfinite(float(grad_groups.get(name, 0.0)))
                or float(grad_groups.get(name, 0.0)) <= 0.0
            ]
            if unreachable:
                raise RuntimeError(
                    f"gradient reachability failed at step {global_step + 1} for: "
                    + ", ".join(unreachable)
                )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in core.parameters() if parameter.requires_grad],
            config.model.max_grad_norm,
        )
        if not bool(torch.isfinite(torch.as_tensor(grad_norm))):
            raise FloatingPointError(f"non-finite gradient norm at step {global_step + 1}")
        optimizer.step()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - step_started
        step_times.append(step_seconds)
        global_step += 1

        global_loss_sum = _all_reduce(local_loss_sum)
        global_hinge_sum = _all_reduce(local_hinge_sum)
        global_delta_sum = _all_reduce(local_delta_sum)
        global_shuffle_rank_sum = _all_reduce(local_shuffle_rank_sum)
        global_alignment_loss_sum = _all_reduce(local_alignment_loss_sum)
        global_alignment_positive_sum = _all_reduce(local_alignment_positive_sum)
        global_alignment_negative_sum = _all_reduce(local_alignment_negative_sum)
        global_alignment_retrieval_correct = _all_reduce(
            local_alignment_retrieval_correct
        )
        global_alignment_event_count = _all_reduce(local_alignment_event_count)
        global_alignment_negative_count = _all_reduce(local_alignment_negative_count)
        global_alignment_same_task_negative_sum = _all_reduce(
            local_alignment_same_task_negative_sum
        )
        global_alignment_same_task_negative_count = _all_reduce(
            local_alignment_same_task_negative_count
        )
        if rank == 0 and (
            global_step % 50 == 0 or global_step == 1 or global_step == stop_step
        ):
            _append_jsonl(
                metrics_path,
                {
                    "step": global_step,
                    "full_sequence_ce": float(
                        global_loss_sum.item() / float(global_token_count.item())
                    ),
                    "base_preservation_hinge": float(
                        global_hinge_sum.item() / config.model.global_batch_size
                    ),
                    "effective_delta_l2": float(
                        global_delta_sum.item() / config.model.global_batch_size
                    ),
                    "shuffle_rank_hinge": float(
                        global_shuffle_rank_sum.item()
                        / config.model.global_batch_size
                    ),
                    "mismatch_coefficient": float(
                        mismatch_training.get("coefficient", 0.0)
                    ),
                    "alignment_mode": alignment_mode,
                    "alignment_coefficient": alignment_weight,
                    "alignment_loss": (
                        float(
                            global_alignment_loss_sum.item()
                            / global_alignment_event_count.item()
                        )
                        if int(global_alignment_event_count.item())
                        else 0.0
                    ),
                    "alignment_positive_similarity": (
                        float(
                            global_alignment_positive_sum.item()
                            / global_alignment_event_count.item()
                        )
                        if int(global_alignment_event_count.item())
                        else 0.0
                    ),
                    "alignment_negative_similarity": (
                        float(
                            global_alignment_negative_sum.item()
                            / global_alignment_negative_count.item()
                        )
                        if int(global_alignment_negative_count.item())
                        else 0.0
                    ),
                    "alignment_retrieval_top1": (
                        float(
                            global_alignment_retrieval_correct.item()
                            / global_alignment_event_count.item()
                        )
                        if int(global_alignment_event_count.item())
                        else 0.0
                    ),
                    "alignment_same_task_negative_similarity": (
                        float(
                            global_alignment_same_task_negative_sum.item()
                            / global_alignment_same_task_negative_count.item()
                        )
                        if int(global_alignment_same_task_negative_count.item())
                        else 0.0
                    ),
                    "gate": gate,
                    "grad_norm": float(torch.as_tensor(grad_norm).float().item()),
                    "gradient_groups": grad_groups,
                    "lr": {
                        str(group["name"]): float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "global_token_count": int(global_token_count.item()),
                    "step_seconds": step_seconds,
                    "samples_per_second": config.model.global_batch_size / step_seconds,
                    "padded_tokens_per_second": (
                        config.model.global_batch_size * config.model.n_ctx / step_seconds
                    ),
                    "max_memory_allocated": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                },
            )

        model_due = (
            global_step % 1_000 == 0
            or global_step in evaluation_steps
            or global_step == stop_step
        )
        resume_due = global_step % 5_000 == 0 or global_step == stop_step
        if model_due:
            checkpoint_started = time.perf_counter()
            _barrier()
            if rank == 0:
                _save_model_only(
                    output / "checkpoints" / f"model_step{global_step:07d}.pt",
                    core=core,
                    global_step=global_step,
                    identity=checkpoint_identity,
                    write_integrity=global_step in evaluation_steps,
                )
            _barrier()
            model_checkpoint_times.append(time.perf_counter() - checkpoint_started)
        if resume_due:
            checkpoint_started = time.perf_counter()
            rng_by_rank = _gather_objects(_rng_state(), world_size)
            runtime_by_rank = _gather_objects(
                {
                    "step_times": step_times,
                    "model_checkpoint_times": model_checkpoint_times,
                    "resume_checkpoint_times": resume_checkpoint_times,
                    "peak_memory_allocated": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                    "peak_memory_reserved": (
                        int(torch.cuda.max_memory_reserved(device))
                        if device.type == "cuda"
                        else 0
                    ),
                },
                world_size,
            )
            if rank == 0:
                _save_resume(
                    output / "checkpoints" / f"resume_step{global_step:07d}.pt",
                    core=core,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    identity=checkpoint_identity,
                    rng_by_rank=rng_by_rank,
                    runtime_by_rank=runtime_by_rank,
                )
            _barrier()
            resume_checkpoint_times.append(time.perf_counter() - checkpoint_started)
        evaluate_if_due(global_step)

    benchmark_warmup_steps = int(
        extra.get(
            "benchmark_warmup_steps",
            SMOKE_WARMUP_STEPS if stop_step == SMOKE_STEPS else 0,
        )
    )
    if not 0 <= benchmark_warmup_steps < len(step_times):
        raise ValueError("benchmark_warmup_steps must leave at least one steady step")
    steady = step_times[benchmark_warmup_steps:]
    local_runtime = {
        "rank": rank,
        "world_size": world_size,
        "steps": len(step_times),
        "benchmark_warmup_steps": benchmark_warmup_steps,
        "steady_steps": len(steady),
        "micro_batch_size": config.model.micro_batch_size,
        "gradient_accumulation_steps": config.model.gradient_accumulation_steps,
        "pg_gradient_checkpointing": pg_gradient_checkpointing,
        "mean_step_seconds": statistics.fmean(steady) if steady else None,
        "median_step_seconds": statistics.median(steady) if steady else None,
        "p90_step_seconds": (
            float(np.quantile(np.asarray(steady), 0.90)) if steady else None
        ),
        "steady_mean_samples_per_second": (
            config.model.global_batch_size / statistics.fmean(steady)
            if steady
            else None
        ),
        "steady_median_samples_per_second": (
            config.model.global_batch_size / statistics.median(steady)
            if steady
            else None
        ),
        "steady_p90_time_samples_per_second": (
            config.model.global_batch_size
            / float(np.quantile(np.asarray(steady), 0.90))
            if steady
            else None
        ),
        "steady_mean_padded_tokens_per_second": (
            config.model.global_batch_size
            * config.model.n_ctx
            / statistics.fmean(steady)
            if steady
            else None
        ),
        "steady_median_padded_tokens_per_second": (
            config.model.global_batch_size
            * config.model.n_ctx
            / statistics.median(steady)
            if steady
            else None
        ),
        "steady_p90_time_padded_tokens_per_second": (
            config.model.global_batch_size
            * config.model.n_ctx
            / float(np.quantile(np.asarray(steady), 0.90))
            if steady
            else None
        ),
        "model_checkpoint_seconds": model_checkpoint_times,
        "resume_checkpoint_seconds": resume_checkpoint_times,
        "peak_memory_allocated": (
            max(
                previous_peak_memory_allocated,
                int(torch.cuda.max_memory_allocated(device)),
            )
            if device.type == "cuda"
            else previous_peak_memory_allocated
        ),
        "peak_memory_reserved": (
            max(
                previous_peak_memory_reserved,
                int(torch.cuda.max_memory_reserved(device)),
            )
            if device.type == "cuda"
            else previous_peak_memory_reserved
        ),
    }
    runtime_by_rank = _gather_objects(local_runtime, world_size)
    if rank == 0:
        complete = global_step == config.model.total_steps
        evaluation_prefix = (
            "atomic" if matrix_id == ATOMIC_SPECIALIZATION_MATRIX_ID else "matched"
        )
        milestone_steps = _milestone_steps_for_matrix(matrix_id)
        if stop_after_step is not None:
            milestone_steps = tuple(step for step in milestone_steps if step <= stop_step)
        evaluation_artifacts = [
            {
                "step": step,
                "path": f"{evaluation_prefix}_step{step:07d}.json",
                "sha256": _file_sha256(
                    output / "evaluations" / f"{evaluation_prefix}_step{step:07d}.json"
                ),
                "bytes": (
                    output / "evaluations" / f"{evaluation_prefix}_step{step:07d}.json"
                ).stat().st_size,
            }
            for step in sorted(evaluated_steps)
        ]
        requires_milestone_coverage = (
            complete or matrix_id != HYBRID_APPEND6_MATRIX_ID
        )
        if requires_milestone_coverage and matrix_id in ({
            *FORMAL32_MATRIX_IDS,
            "s5_hypernet_circuit_heavy120_v1",
            "s5_hypernet_circuit_heavy80_v2",
            "s5_hypernet_a_shuffle_sweep_v1",
            "s5_hypernet_a_alignment_sweep_v1",
            "s5_hypernet_query_only_formal_v1",
            ATOMIC_SPECIALIZATION_MATRIX_ID,
            "s5_hypernet_raw_weight_formal_v1",
            HYBRID_APPEND6_MATRIX_ID,
        } | RAW_ABLATION_MATRIX_IDS) and (
            tuple(sorted(evaluated_steps))
            != milestone_steps
            or any(
                not (
                    output
                    / "checkpoints"
                    / f"model_step{step:07d}.pt.json"
                ).is_file()
                or not (
                    output
                    / "evaluations"
                    / f"{evaluation_prefix}_step{step:07d}.json"
                ).is_file()
                for step in milestone_steps
            )
        ):
            raise RuntimeError("formal32 milestone checkpoint/evaluation coverage changed")
        completion_status = (
            "target_reached"
            if stop_after_step is not None
            else ("complete" if complete else "smoke_complete")
        )
        marker = {
            "status": completion_status,
            "protocol_id": PROTOCOL_ID,
            "config_sha256": config.config_sha256,
            "candidate": config.model.candidate,
            "condition_source": condition_source,
            "fusion_mode": fusion_mode,
            "group": config.model.group,
            "global_step": global_step,
            "planned_total_steps": config.model.total_steps,
            "world_size": world_size,
            "micro_batch_size": config.model.micro_batch_size,
            "gradient_accumulation_steps": config.model.gradient_accumulation_steps,
            "pg_gradient_checkpointing": pg_gradient_checkpointing,
            "evaluated_steps": sorted(evaluated_steps),
            "evaluation_artifacts": evaluation_artifacts,
            "runtime_by_rank": runtime_by_rank,
        }
        _write_completion_marker(
            output, marker, complete=complete, stop_after_step=stop_after_step
        )
    _barrier()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    return {
        "status": (
            "target_reached"
            if stop_after_step is not None
            else ("complete" if global_step == config.model.total_steps else "smoke_complete")
        ),
        "global_step": global_step,
        "output_dir": str(output),
        **(
            {
                "planned_total_steps": config.model.total_steps,
                "stop_after_step": int(stop_after_step),
                "actual_steps": global_step,
            }
            if stop_after_step is not None
            else {}
        ),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    train_formal_run(
        load_run_config(args.config),
        resume=args.resume,
        max_steps=args.max_steps,
        stop_after_step=args.stop_after_step,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SMOKE_STEPS",
    "balanced_epoch_order",
    "rank_indices_for_step",
    "train_formal_run",
]

"""Sealed 20-cell Query-only atomic-retention protocol.

The protocol mirrors the Circuit-A retention20 budget and sampler contract,
but trains only the Query encoder and parameter generator.  It intentionally
does not expose a Circuit encoder/token or a Raw-weight branch.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .composition_loco import ATOMIC_TASKS, COMPOSITION_TASKS, GLOBAL_BATCH_SIZE, N_CTX, task_fold


QUERY_RETENTION20_PROTOCOL_ID = "s5_composition_loco_query_retention20_v1"
QUERY_RETENTION20_OUTPUT_ROOT = (
    "outputs/s5_lean_circuit/composition_loco_query_retention20_v1"
)
QUERY_RETENTION20_CONFIG_ROOT = (
    "s5_lean_circuit/configs/composition_loco_query_retention20_v1"
)
QUERY_RETENTION20_YAML_ROOT = (
    "4server/0914/s5_composition_loco_query_retention20_v1"
)
QUERY_RETENTION20_CONTROL_ROOT = f"{QUERY_RETENTION20_OUTPUT_ROOT}/control"

QUERY_RETENTION20_HELDOUT_TASKS = tuple(COMPOSITION_TASKS)
QUERY_RETENTION20_GROUP_IDS = ("M12", "M25", "Q12", "Q25")
QUERY_RETENTION20_ROUTES = ("query_only",)
QUERY_RETENTION20_FOLD_COUNT = len(QUERY_RETENTION20_HELDOUT_TASKS)
QUERY_RETENTION20_JOB_COUNT = QUERY_RETENTION20_FOLD_COUNT * len(QUERY_RETENTION20_GROUP_IDS)
QUERY_RETENTION20_TRAIN_STEPS = 12_000
QUERY_RETENTION20_SCHEDULER_STEPS = 38_000
QUERY_RETENTION20_WARMUP_STEPS = 1_140
QUERY_RETENTION20_VALIDATION_STEPS = (
    0,
    1_000,
    2_000,
    4_000,
    6_000,
    8_000,
    10_000,
    12_000,
)
QUERY_RETENTION20_ATOMIC_STEPS = (0,)
QUERY_RETENTION20_KD_TEMPERATURE = 2.0


_GROUPS: dict[str, dict[str, Any]] = {
    "M12": {
        "label": "moderate PG/Query LR, 12.5 percent replay",
        "pg_lr": 2.5e-5,
        "query_lr_multiplier": 0.5,
        "composition_per_batch": 448,
        "atomic_per_batch": 64,
        "composition_fraction": 0.875,
        "atomic_fraction": 0.125,
        "composition_per_task": 112,
        "atomic_per_task": 16,
        "kd_lambda": 0.25,
    },
    "M25": {
        "label": "moderate PG/Query LR, 25 percent replay",
        "pg_lr": 2.5e-5,
        "query_lr_multiplier": 0.5,
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "composition_per_task": 96,
        "atomic_per_task": 32,
        "kd_lambda": 0.25,
    },
    "Q12": {
        "label": "historical P10 PG/Query LR, 12.5 percent replay",
        "pg_lr": 5.0e-5,
        "query_lr_multiplier": 1.0,
        "composition_per_batch": 448,
        "atomic_per_batch": 64,
        "composition_fraction": 0.875,
        "atomic_fraction": 0.125,
        "composition_per_task": 112,
        "atomic_per_task": 16,
        "kd_lambda": 0.25,
    },
    "Q25": {
        "label": "historical P10 PG/Query LR, 25 percent replay",
        "pg_lr": 5.0e-5,
        "query_lr_multiplier": 1.0,
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "composition_per_task": 96,
        "atomic_per_task": 32,
        "kd_lambda": 0.25,
    },
}


def query_retention20_group_values(group_id: str) -> dict[str, Any]:
    try:
        return deepcopy(_GROUPS[str(group_id)])
    except KeyError as exc:
        raise ValueError(f"invalid Query retention20 group: {group_id}") from exc


def query_retention20_job_index(group_id: str, heldout_task: str) -> int:
    try:
        fold_index = QUERY_RETENTION20_HELDOUT_TASKS.index(str(heldout_task))
        group_index = QUERY_RETENTION20_GROUP_IDS.index(str(group_id))
    except ValueError as exc:
        raise ValueError(f"invalid Query retention20 row: {group_id}/{heldout_task}") from exc
    return fold_index * len(QUERY_RETENTION20_GROUP_IDS) + group_index


def query_retention20_expected_output(group_id: str, heldout_task: str) -> str:
    query_retention20_job_index(group_id, heldout_task)
    return f"{QUERY_RETENTION20_OUTPUT_ROOT}/runs/{heldout_task}/{group_id}/query_only"


def query_retention20_train_tasks(heldout_task: str) -> tuple[str, ...]:
    if str(heldout_task) not in QUERY_RETENTION20_HELDOUT_TASKS:
        raise ValueError(f"unknown Query retention20 held-out task: {heldout_task}")
    return task_fold(str(heldout_task))


if set(QUERY_RETENTION20_HELDOUT_TASKS) != set(COMPOSITION_TASKS):
    raise RuntimeError("Query retention20 must cover all five composition folds")
if set(QUERY_RETENTION20_GROUP_IDS) != set(_GROUPS) or len(_GROUPS) != 4:
    raise RuntimeError("Query retention20 must contain four groups")
if QUERY_RETENTION20_JOB_COUNT != 20:
    raise RuntimeError("Query retention20 must contain exactly twenty jobs")
if QUERY_RETENTION20_WARMUP_STEPS != round(0.03 * QUERY_RETENTION20_SCHEDULER_STEPS):
    raise RuntimeError("Query retention20 warmup is not three percent of the 38k horizon")
if QUERY_RETENTION20_VALIDATION_STEPS[-1] != QUERY_RETENTION20_TRAIN_STEPS:
    raise RuntimeError("Query retention20 validation schedule must end at 12k")
if set(ATOMIC_TASKS) != {"S_and", "S_or_l", "S_or_r", "S_id"}:
    raise RuntimeError("atomic task contract changed")
if any(
    values["composition_per_batch"] + values["atomic_per_batch"] != GLOBAL_BATCH_SIZE
    or len(task_fold("T_and_or")) * values["composition_per_task"] != values["composition_per_batch"]
    or len(ATOMIC_TASKS) * values["atomic_per_task"] != values["atomic_per_batch"]
    or values["composition_fraction"] + values["atomic_fraction"] != 1.0
    for values in _GROUPS.values()
):
    raise RuntimeError("Query retention20 batch quotas are inconsistent")


__all__ = [
    "QUERY_RETENTION20_PROTOCOL_ID",
    "QUERY_RETENTION20_OUTPUT_ROOT",
    "QUERY_RETENTION20_CONFIG_ROOT",
    "QUERY_RETENTION20_YAML_ROOT",
    "QUERY_RETENTION20_CONTROL_ROOT",
    "QUERY_RETENTION20_HELDOUT_TASKS",
    "QUERY_RETENTION20_GROUP_IDS",
    "QUERY_RETENTION20_ROUTES",
    "QUERY_RETENTION20_JOB_COUNT",
    "QUERY_RETENTION20_TRAIN_STEPS",
    "QUERY_RETENTION20_SCHEDULER_STEPS",
    "QUERY_RETENTION20_WARMUP_STEPS",
    "QUERY_RETENTION20_VALIDATION_STEPS",
    "QUERY_RETENTION20_ATOMIC_STEPS",
    "QUERY_RETENTION20_KD_TEMPERATURE",
    "query_retention20_group_values",
    "query_retention20_job_index",
    "query_retention20_expected_output",
    "query_retention20_train_tasks",
]

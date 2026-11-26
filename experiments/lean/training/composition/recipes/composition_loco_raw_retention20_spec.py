"""Sealed Raw-Weight query-cross retention sweep: five folds by four recipes."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .composition_loco import ATOMIC_TASKS, COMPOSITION_TASKS, GLOBAL_BATCH_SIZE, N_CTX, task_fold


RAW_RETENTION20_PROTOCOL_ID = "s5_composition_loco_raw_retention20_v1"
RAW_RETENTION20_OUTPUT_ROOT = (
    "outputs/s5_lean_circuit/composition_loco_raw_retention20_v1"
)
RAW_RETENTION20_CONFIG_ROOT = (
    "s5_lean_circuit/configs/composition_loco_raw_retention20_v1"
)
RAW_RETENTION20_YAML_ROOT = (
    "4server/0914/s5_composition_loco_raw_retention20_v1"
)
RAW_RETENTION20_CONTROL_ROOT = f"{RAW_RETENTION20_OUTPUT_ROOT}/control"

RAW_RETENTION20_HELDOUT_TASKS = tuple(COMPOSITION_TASKS)
RAW_RETENTION20_GROUP_IDS = ("M12", "M25", "R12", "R25")
RAW_RETENTION20_ROUTES = ("raw_weight",)
RAW_RETENTION20_FOLD_COUNT = len(RAW_RETENTION20_HELDOUT_TASKS)
RAW_RETENTION20_JOB_COUNT = RAW_RETENTION20_FOLD_COUNT * len(RAW_RETENTION20_GROUP_IDS)
RAW_RETENTION20_TRAIN_STEPS = 12_000
RAW_RETENTION20_SCHEDULER_STEPS = 38_000
RAW_RETENTION20_WARMUP_STEPS = 1_140
RAW_RETENTION20_VALIDATION_STEPS = (
    0,
    1_000,
    2_000,
    4_000,
    6_000,
    8_000,
    10_000,
    12_000,
)
RAW_RETENTION20_ATOMIC_STEPS = (0,)
RAW_RETENTION20_KD_TEMPERATURE = 2.0


_GROUPS: dict[str, dict[str, Any]] = {
    "M12": {
        "label": "moderate PG/Query LR, 12.5 percent replay",
        "pg_lr": 2.5e-5,
        "query_lr_multiplier": 0.5,
        "raw_encoder_lr": 2.0e-4,
        "raw_output_lr": 5.0e-4,
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
        "raw_encoder_lr": 2.0e-4,
        "raw_output_lr": 5.0e-4,
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "composition_per_task": 96,
        "atomic_per_task": 32,
        "kd_lambda": 0.25,
    },
    "R12": {
        "label": "historical P02 PG/Query LR, 12.5 percent replay",
        "pg_lr": 1.875e-5,
        "query_lr_multiplier": 0.5,
        "raw_encoder_lr": 2.0e-4,
        "raw_output_lr": 5.0e-4,
        "composition_per_batch": 448,
        "atomic_per_batch": 64,
        "composition_fraction": 0.875,
        "atomic_fraction": 0.125,
        "composition_per_task": 112,
        "atomic_per_task": 16,
        "kd_lambda": 0.25,
    },
    "R25": {
        "label": "historical P02 PG/Query LR, 25 percent replay",
        "pg_lr": 1.875e-5,
        "query_lr_multiplier": 0.5,
        "raw_encoder_lr": 2.0e-4,
        "raw_output_lr": 5.0e-4,
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "composition_per_task": 96,
        "atomic_per_task": 32,
        "kd_lambda": 0.25,
    },
}


def raw_retention20_group_values(group_id: str) -> dict[str, Any]:
    try:
        return deepcopy(_GROUPS[str(group_id)])
    except KeyError as exc:
        raise ValueError(f"invalid Raw retention20 group: {group_id}") from exc


def raw_retention20_job_index(group_id: str, heldout_task: str) -> int:
    try:
        fold_index = RAW_RETENTION20_HELDOUT_TASKS.index(str(heldout_task))
        group_index = RAW_RETENTION20_GROUP_IDS.index(str(group_id))
    except ValueError as exc:
        raise ValueError(f"invalid Raw retention20 row: {group_id}/{heldout_task}") from exc
    return fold_index * len(RAW_RETENTION20_GROUP_IDS) + group_index


def raw_retention20_expected_output(group_id: str, heldout_task: str) -> str:
    raw_retention20_job_index(group_id, heldout_task)
    return f"{RAW_RETENTION20_OUTPUT_ROOT}/runs/{heldout_task}/{group_id}/raw_weight"


def raw_retention20_train_tasks(heldout_task: str) -> tuple[str, ...]:
    if str(heldout_task) not in RAW_RETENTION20_HELDOUT_TASKS:
        raise ValueError(f"unknown Raw retention20 held-out task: {heldout_task}")
    return task_fold(str(heldout_task))


if set(RAW_RETENTION20_HELDOUT_TASKS) != set(COMPOSITION_TASKS):
    raise RuntimeError("Raw retention20 must cover all five composition folds")
if set(RAW_RETENTION20_GROUP_IDS) != set(_GROUPS) or len(_GROUPS) != 4:
    raise RuntimeError("Raw retention20 must contain four groups")
if RAW_RETENTION20_JOB_COUNT != 20:
    raise RuntimeError("Raw retention20 must contain exactly twenty jobs")
if RAW_RETENTION20_WARMUP_STEPS != round(0.03 * RAW_RETENTION20_SCHEDULER_STEPS):
    raise RuntimeError("Raw retention20 warmup is not three percent of the 38k horizon")
if RAW_RETENTION20_VALIDATION_STEPS[-1] != RAW_RETENTION20_TRAIN_STEPS:
    raise RuntimeError("Raw retention20 validation schedule must end at 12k")
if set(ATOMIC_TASKS) != {"S_and", "S_or_l", "S_or_r", "S_id"}:
    raise RuntimeError("atomic task contract changed")
if any(
    values["composition_per_batch"] + values["atomic_per_batch"] != GLOBAL_BATCH_SIZE
    or len(task_fold("T_and_or")) * values["composition_per_task"] != values["composition_per_batch"]
    or len(ATOMIC_TASKS) * values["atomic_per_task"] != values["atomic_per_batch"]
    or values["composition_fraction"] + values["atomic_fraction"] != 1.0
    for values in _GROUPS.values()
):
    raise RuntimeError("Raw retention20 batch quotas are inconsistent")


__all__ = [
    "RAW_RETENTION20_PROTOCOL_ID",
    "RAW_RETENTION20_OUTPUT_ROOT",
    "RAW_RETENTION20_CONFIG_ROOT",
    "RAW_RETENTION20_YAML_ROOT",
    "RAW_RETENTION20_CONTROL_ROOT",
    "RAW_RETENTION20_HELDOUT_TASKS",
    "RAW_RETENTION20_GROUP_IDS",
    "RAW_RETENTION20_ROUTES",
    "RAW_RETENTION20_JOB_COUNT",
    "RAW_RETENTION20_TRAIN_STEPS",
    "RAW_RETENTION20_SCHEDULER_STEPS",
    "RAW_RETENTION20_WARMUP_STEPS",
    "RAW_RETENTION20_VALIDATION_STEPS",
    "RAW_RETENTION20_ATOMIC_STEPS",
    "RAW_RETENTION20_KD_TEMPERATURE",
    "raw_retention20_group_values",
    "raw_retention20_job_index",
    "raw_retention20_expected_output",
    "raw_retention20_train_tasks",
]

RAW_RETENTION20_CHECKPOINT = "outputs/s5_lean_circuit/hypernet_raw_atomic_seed42/raw_query16/rq14_p14/checkpoints/model_step0030516.pt"
RAW_RETENTION20_CHECKPOINT_SHA256 = "94dfb66fc1a050a90ac542482820d626cb5f8ef2e98554c8ac13806221bf4e63"
RAW_RETENTION20_CACHE_PATH = "outputs/s5_lean_circuit/hypernet_raw_weight_formal_1p2m_seed42/shared/raw_weight_cache.pt"
RAW_RETENTION20_CACHE_SHA256 = "886479c121f743184f5bb89eef2af30afb86a0278cb0b94dbcebfc0c98428334"
RAW_RETENTION20_HISTORICAL_CACHE_SHA256 = "a70eb76788bcef9c2ca301ad4a4d3febc94eb247ce1f02cf41ebdccda8bb87e1"

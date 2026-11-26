"""Scientific configuration for this composition training sweep."""

from __future__ import annotations

from typing import Any

from .composition_loco import ATOMIC_TASKS, COMPOSITION_TASKS, task_fold
from .composition_loco_retention import (
    DIRECT_RETENTION32_GROUPS,
    DIRECT_RETENTION32_PROTOCOL_ID,
    RETENTION_GLOBAL_BATCH_SIZE,
    RETENTION_KD_TEMPERATURE,
)


DIRECT_RETENTION32_OUTPUT_ROOT = (
    "outputs/s5_lean_circuit/composition_loco_direct_retention32_12k_v1"
)
DIRECT_RETENTION32_CONFIG_ROOT = (
    "s5_lean_circuit/configs/composition_loco_direct_retention32_12k_v1"
)
DIRECT_RETENTION32_YAML_ROOT = (
    "4server/0913/s5_composition_loco_direct_retention32_12k_v1"
)
DIRECT_RETENTION32_CONTROL_ROOT = f"{DIRECT_RETENTION32_OUTPUT_ROOT}/control"

DIRECT_RETENTION32_HELDOUT_TASKS = tuple(
    task for task in COMPOSITION_TASKS if task != "T_and_or"
)
DIRECT_RETENTION32_GROUP_IDS = tuple(DIRECT_RETENTION32_GROUPS)
DIRECT_RETENTION32_ROUTES = ("direct_sft",)
DIRECT_RETENTION32_JOB_COUNT = (
    len(DIRECT_RETENTION32_HELDOUT_TASKS) * len(DIRECT_RETENTION32_GROUP_IDS)
)
DIRECT_RETENTION32_TRAIN_STEPS = 12_000
DIRECT_RETENTION32_SCHEDULER_STEPS = 38_000
DIRECT_RETENTION32_WARMUP_STEPS = 1_140
DIRECT_RETENTION32_GLOBAL_BATCH_SIZE = RETENTION_GLOBAL_BATCH_SIZE
DIRECT_RETENTION32_VALIDATION_STEPS = (
    0, 1_000, 2_000, 4_000, 6_000, 8_000, 10_000, 12_000
)
DIRECT_RETENTION32_ATOMIC_STEPS = (0,)
DIRECT_RETENTION32_KD_TEMPERATURE = RETENTION_KD_TEMPERATURE


def direct_retention32_group_values(group_id: str) -> dict[str, Any]:
    try:
        return dict(DIRECT_RETENTION32_GROUPS[str(group_id)])
    except KeyError as exc:
        raise ValueError(f"invalid direct retention32 group: {group_id}") from exc


def direct_retention32_job_index(group_id: str, heldout_task: str) -> int:
    try:
        group_index = DIRECT_RETENTION32_GROUP_IDS.index(str(group_id))
        fold_index = DIRECT_RETENTION32_HELDOUT_TASKS.index(str(heldout_task))
    except ValueError as exc:
        raise ValueError(
            f"invalid direct retention32 row identity: {group_id}/{heldout_task}"
        ) from exc
    return fold_index * len(DIRECT_RETENTION32_GROUP_IDS) + group_index


def direct_retention32_expected_output(group_id: str, heldout_task: str) -> str:
    direct_retention32_job_index(group_id, heldout_task)
    return (
        f"{DIRECT_RETENTION32_OUTPUT_ROOT}/runs/{heldout_task}/{group_id}/direct_sft"
    )


def direct_retention32_train_tasks(heldout_task: str) -> tuple[str, ...]:
    if str(heldout_task) not in DIRECT_RETENTION32_HELDOUT_TASKS:
        raise ValueError(f"heldout task is outside retention32: {heldout_task}")
    return task_fold(str(heldout_task))


if len(DIRECT_RETENTION32_GROUP_IDS) != 8:
    raise RuntimeError("direct_retention32 must contain exactly eight groups")
if DIRECT_RETENTION32_JOB_COUNT != 32:
    raise RuntimeError("direct_retention32 must contain exactly 32 jobs")
if len(DIRECT_RETENTION32_HELDOUT_TASKS) != 4:
    raise RuntimeError("direct_retention32 must contain four held-out folds")
if DIRECT_RETENTION32_VALIDATION_STEPS[-1] != DIRECT_RETENTION32_TRAIN_STEPS:
    raise RuntimeError("retention32 validation schedule must end at train budget")
if DIRECT_RETENTION32_SCHEDULER_STEPS <= DIRECT_RETENTION32_TRAIN_STEPS:
    raise RuntimeError("retention32 scheduler horizon must exceed the short train budget")
if DIRECT_RETENTION32_WARMUP_STEPS != round(0.03 * DIRECT_RETENTION32_SCHEDULER_STEPS):
    raise RuntimeError("retention32 warmup is not the approved 3% of 38k horizon")
if set(DIRECT_RETENTION32_HELDOUT_TASKS) | {"T_and_or"} != set(COMPOSITION_TASKS):
    raise RuntimeError("retention32 held-out fold set is incomplete")
if set(ATOMIC_TASKS) != {"S_and", "S_or_l", "S_or_r", "S_id"}:
    raise RuntimeError("atomic task contract changed")


__all__ = [
    "DIRECT_RETENTION32_PROTOCOL_ID",
    "DIRECT_RETENTION32_OUTPUT_ROOT",
    "DIRECT_RETENTION32_CONFIG_ROOT",
    "DIRECT_RETENTION32_YAML_ROOT",
    "DIRECT_RETENTION32_CONTROL_ROOT",
    "DIRECT_RETENTION32_HELDOUT_TASKS",
    "DIRECT_RETENTION32_GROUPS",
    "DIRECT_RETENTION32_GROUP_IDS",
    "DIRECT_RETENTION32_ROUTES",
    "DIRECT_RETENTION32_JOB_COUNT",
    "DIRECT_RETENTION32_TRAIN_STEPS",
    "DIRECT_RETENTION32_SCHEDULER_STEPS",
    "DIRECT_RETENTION32_WARMUP_STEPS",
    "DIRECT_RETENTION32_GLOBAL_BATCH_SIZE",
    "DIRECT_RETENTION32_VALIDATION_STEPS",
    "DIRECT_RETENTION32_ATOMIC_STEPS",
    "DIRECT_RETENTION32_KD_TEMPERATURE",
    "direct_retention32_group_values",
    "direct_retention32_job_index",
    "direct_retention32_expected_output",
    "direct_retention32_train_tasks",
]

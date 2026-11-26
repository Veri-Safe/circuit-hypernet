"""Sealed 20-cell Circuit-A atomic-retention protocol.

The protocol is deliberately separate from the historical Circuit sweep and
the Direct retention protocols.  It keeps the existing Vector397/Circuit-A
architecture, restarts every cell from the original A-P03 endpoint, and
changes only the declared replay/KD and optimizer-group settings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .composition_loco import ATOMIC_TASKS, COMPOSITION_TASKS, N_CTX, GLOBAL_BATCH_SIZE, task_fold


CIRCUIT_RETENTION20_PROTOCOL_ID = "s5_composition_loco_circuit_retention20_v1"
CIRCUIT_RETENTION20_OUTPUT_ROOT = (
    "outputs/s5_lean_circuit/composition_loco_circuit_retention20_v2"
)
CIRCUIT_RETENTION20_CONFIG_ROOT = (
    "s5_lean_circuit/configs/composition_loco_circuit_retention20_v2"
)
CIRCUIT_RETENTION20_YAML_ROOT = (
    "4server/0914/s5_composition_loco_circuit_retention20_v2"
)
CIRCUIT_RETENTION20_CONTROL_ROOT = f"{CIRCUIT_RETENTION20_OUTPUT_ROOT}/control_repair"

CIRCUIT_RETENTION20_HELDOUT_TASKS = tuple(COMPOSITION_TASKS)
CIRCUIT_RETENTION20_GROUP_IDS = ("M12", "M25", "C12", "C25")
CIRCUIT_RETENTION20_ROUTES = ("circuit_a",)
CIRCUIT_RETENTION20_FOLD_COUNT = len(CIRCUIT_RETENTION20_HELDOUT_TASKS)
CIRCUIT_RETENTION20_JOB_COUNT = (
    CIRCUIT_RETENTION20_FOLD_COUNT * len(CIRCUIT_RETENTION20_GROUP_IDS)
)
CIRCUIT_RETENTION20_TRAIN_STEPS = 12_000
CIRCUIT_RETENTION20_SCHEDULER_STEPS = 38_000
CIRCUIT_RETENTION20_WARMUP_STEPS = 1_140
CIRCUIT_RETENTION20_VALIDATION_STEPS = (
    0,
    1_000,
    2_000,
    4_000,
    6_000,
    8_000,
    10_000,
    12_000,
)
CIRCUIT_RETENTION20_ATOMIC_STEPS = (0,)
CIRCUIT_RETENTION20_KD_TEMPERATURE = 2.0


_GROUPS: dict[str, dict[str, Any]] = {
    "M12": {
        "label": "moderate LR, 12.5 percent replay",
        "pg_lr": 2.5e-5,
        "query_lr_multiplier": 0.5,
        "circuit_lr_multiplier": 2.0,
        "composition_per_batch": 448,
        "atomic_per_batch": 64,
        "composition_fraction": 0.875,
        "atomic_fraction": 0.125,
        "composition_per_task": 112,
        "atomic_per_task": 16,
        "kd_lambda": 0.25,
    },
    "M25": {
        "label": "moderate LR, 25 percent replay",
        "pg_lr": 2.5e-5,
        "query_lr_multiplier": 0.5,
        "circuit_lr_multiplier": 2.0,
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "composition_per_task": 96,
        "atomic_per_task": 32,
        "kd_lambda": 0.25,
    },
    "C12": {
        "label": "slow query, fast circuit, 12.5 percent replay",
        "pg_lr": 2.5e-5,
        "query_lr_multiplier": 0.25,
        "circuit_lr_multiplier": 4.0,
        "composition_per_batch": 448,
        "atomic_per_batch": 64,
        "composition_fraction": 0.875,
        "atomic_fraction": 0.125,
        "composition_per_task": 112,
        "atomic_per_task": 16,
        "kd_lambda": 0.25,
    },
    "C25": {
        "label": "slow query, fast circuit, 25 percent replay",
        "pg_lr": 2.5e-5,
        "query_lr_multiplier": 0.25,
        "circuit_lr_multiplier": 4.0,
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "composition_per_task": 96,
        "atomic_per_task": 32,
        "kd_lambda": 0.25,
    },
}

# Row order is ``heldout_task`` major, then ``M12/M25/C12/C25``.  Keeping the
# assignment explicit makes the five-YAML topology auditable and prevents a
# later change in list arithmetic from silently moving a child to another
# server task.


def circuit_retention20_group_values(group_id: str) -> dict[str, Any]:
    try:
        return deepcopy(_GROUPS[str(group_id)])
    except KeyError as exc:
        raise ValueError(f"invalid Circuit retention20 group: {group_id}") from exc


def circuit_retention20_job_index(group_id: str, heldout_task: str) -> int:
    try:
        fold_index = CIRCUIT_RETENTION20_HELDOUT_TASKS.index(str(heldout_task))
        group_index = CIRCUIT_RETENTION20_GROUP_IDS.index(str(group_id))
    except ValueError as exc:
        raise ValueError(f"invalid Circuit retention20 row: {group_id}/{heldout_task}") from exc
    return fold_index * len(CIRCUIT_RETENTION20_GROUP_IDS) + group_index


def circuit_retention20_expected_output(group_id: str, heldout_task: str) -> str:
    circuit_retention20_job_index(group_id, heldout_task)
    return (
        f"{CIRCUIT_RETENTION20_OUTPUT_ROOT}/runs/"
        f"{heldout_task}/{group_id}/circuit_a"
    )


def circuit_retention20_train_tasks(heldout_task: str) -> tuple[str, ...]:
    if str(heldout_task) not in CIRCUIT_RETENTION20_HELDOUT_TASKS:
        raise ValueError(f"unknown Circuit retention20 held-out task: {heldout_task}")
    return task_fold(str(heldout_task))


if set(CIRCUIT_RETENTION20_HELDOUT_TASKS) != set(COMPOSITION_TASKS):
    raise RuntimeError("Circuit retention20 must cover all five composition folds")
if len(CIRCUIT_RETENTION20_GROUP_IDS) != 4 or len(_GROUPS) != 4:
    raise RuntimeError("Circuit retention20 must contain four groups")
if CIRCUIT_RETENTION20_JOB_COUNT != 20:
    raise RuntimeError("Circuit retention20 must contain exactly twenty jobs")
if CIRCUIT_RETENTION20_WARMUP_STEPS != round(0.03 * CIRCUIT_RETENTION20_SCHEDULER_STEPS):
    raise RuntimeError("Circuit retention20 warmup is not three percent of the 38k horizon")
if CIRCUIT_RETENTION20_VALIDATION_STEPS[-1] != CIRCUIT_RETENTION20_TRAIN_STEPS:
    raise RuntimeError("Circuit retention20 validation schedule must end at 12k")
if set(ATOMIC_TASKS) != {"S_and", "S_or_l", "S_or_r", "S_id"}:
    raise RuntimeError("atomic task contract changed")
if any(
    values["composition_per_batch"] + values["atomic_per_batch"] != GLOBAL_BATCH_SIZE
    or len(task_fold("T_and_or")) * values["composition_per_task"] != values["composition_per_batch"]
    or len(ATOMIC_TASKS) * values["atomic_per_task"] != values["atomic_per_batch"]
    or values["composition_fraction"] + values["atomic_fraction"] != 1.0
    for values in _GROUPS.values()
):
    raise RuntimeError("Circuit retention20 batch quotas are inconsistent")


__all__ = [
    "CIRCUIT_RETENTION20_PROTOCOL_ID",
    "CIRCUIT_RETENTION20_OUTPUT_ROOT",
    "CIRCUIT_RETENTION20_CONFIG_ROOT",
    "CIRCUIT_RETENTION20_YAML_ROOT",
    "CIRCUIT_RETENTION20_CONTROL_ROOT",
    "CIRCUIT_RETENTION20_HELDOUT_TASKS",
    "CIRCUIT_RETENTION20_GROUP_IDS",
    "CIRCUIT_RETENTION20_ROUTES",
    "CIRCUIT_RETENTION20_JOB_COUNT",
    "CIRCUIT_RETENTION20_TRAIN_STEPS",
    "CIRCUIT_RETENTION20_SCHEDULER_STEPS",
    "CIRCUIT_RETENTION20_WARMUP_STEPS",
    "CIRCUIT_RETENTION20_VALIDATION_STEPS",
    "CIRCUIT_RETENTION20_ATOMIC_STEPS",
    "CIRCUIT_RETENTION20_KD_TEMPERATURE",
    "circuit_retention20_group_values",
    "circuit_retention20_job_index",
    "circuit_retention20_expected_output",
    "circuit_retention20_train_tasks",
]

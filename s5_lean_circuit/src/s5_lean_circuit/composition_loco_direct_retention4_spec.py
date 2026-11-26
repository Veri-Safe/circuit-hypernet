"""Sealed four-cell Direct-SFT atomic-retention protocol."""

from __future__ import annotations

from typing import Any

from .composition_loco import ATOMIC_TASKS, COMPOSITION_TASKS, task_fold
from .composition_loco_retention import (
    DIRECT_RETENTION4_CONFIG_ROOT,
    DIRECT_RETENTION4_GROUPS,
    DIRECT_RETENTION4_OUTPUT_ROOT,
    DIRECT_RETENTION4_PROTOCOL_ID,
    DIRECT_RETENTION4_YAML_ROOT,
    RETENTION_COMPOSITION_TASKS,
    RETENTION_GLOBAL_BATCH_SIZE,
    RETENTION_HELDOUT_TASK,
    RETENTION_KD_TEMPERATURE,
    RETENTION_TOTAL_STEPS,
)


DIRECT_RETENTION4_CONTROL_ROOT = (
    f"{DIRECT_RETENTION4_OUTPUT_ROOT}/control"
)
# A conservative four-job estimate derived from the measured Direct-LR32
# model/resume checkpoint envelope.  It is used only by the controller's
# storage preflight and never changes training behavior.
DIRECT_RETENTION4_GROUP_IDS = tuple(DIRECT_RETENTION4_GROUPS)
DIRECT_RETENTION4_ROUTES = ("direct_sft",)
DIRECT_RETENTION4_JOB_COUNT = len(DIRECT_RETENTION4_GROUP_IDS)
DIRECT_RETENTION4_VALIDATION_STEPS = (0, 1_000, 2_000, 4_000, 8_000, 16_000, 24_000, 32_000, 38_000)
DIRECT_RETENTION4_ATOMIC_STEPS = (0, RETENTION_TOTAL_STEPS)


def direct_retention4_group_values(group_id: str) -> dict[str, Any]:
    try:
        return dict(DIRECT_RETENTION4_GROUPS[str(group_id)])
    except KeyError as exc:
        raise ValueError(f"invalid direct retention4 group: {group_id}") from exc


def direct_retention4_expected_output(group_id: str) -> str:
    if str(group_id) not in DIRECT_RETENTION4_GROUP_IDS:
        raise ValueError(f"invalid direct retention4 group: {group_id}")
    return f"{DIRECT_RETENTION4_OUTPUT_ROOT}/runs/{group_id}/{RETENTION_HELDOUT_TASK}/direct_sft"


def direct_retention4_job_index(group_id: str) -> int:
    try:
        return DIRECT_RETENTION4_GROUP_IDS.index(str(group_id))
    except ValueError as exc:
        raise ValueError(f"invalid direct retention4 group: {group_id}") from exc


__all__ = [
    "DIRECT_RETENTION4_PROTOCOL_ID",
    "DIRECT_RETENTION4_OUTPUT_ROOT",
    "DIRECT_RETENTION4_CONFIG_ROOT",
    "DIRECT_RETENTION4_YAML_ROOT",
    "DIRECT_RETENTION4_GROUPS",
    "DIRECT_RETENTION4_GROUP_IDS",
    "DIRECT_RETENTION4_ROUTES",
    "DIRECT_RETENTION4_CONTROL_ROOT",
    "DIRECT_RETENTION4_JOB_COUNT",
    "DIRECT_RETENTION4_VALIDATION_STEPS",
    "DIRECT_RETENTION4_ATOMIC_STEPS",
    "RETENTION_COMPOSITION_TASKS",
    "RETENTION_GLOBAL_BATCH_SIZE",
    "RETENTION_HELDOUT_TASK",
    "RETENTION_KD_TEMPERATURE",
    "RETENTION_TOTAL_STEPS",
    "direct_retention4_group_values",
    "direct_retention4_expected_output",
    "direct_retention4_job_index",
]

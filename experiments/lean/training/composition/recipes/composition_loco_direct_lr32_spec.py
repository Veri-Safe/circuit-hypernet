"""Frozen 32-profile Direct-SFT learning-rate sweep.

This protocol is deliberately separate from the historical/recreated Direct
continuations.  Every row starts from the same Stage-A Base checkpoint at
step 38000 and consumes the same leave-one-task-out 1.2M-event training view.
The profile table varies only ordinary Direct-SFT optimizer settings; OOD
results are post-selection diagnostics and never choose a checkpoint.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .composition_loco import COMPOSITION_TASKS


DIRECT_LR32_PROTOCOL_ID = "s5_composition_loco_direct_lr32_v1"
DIRECT_LR32_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_direct_lr32_v1"
DIRECT_LR32_CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_direct_lr32_v1"
DIRECT_LR32_YAML_ROOT = "4server/0910/s5_composition_loco_direct_lr32_v1"
DIRECT_LR32_CONTROL_ROOT = f"{DIRECT_LR32_OUTPUT_ROOT}/control"

DIRECT_LR32_PROFILES = tuple(f"p{index:02d}" for index in range(1, 33))
DIRECT_LR32_ROUTES = ("direct_sft",)
DIRECT_LR32_JOB_COUNT = len(DIRECT_LR32_PROFILES) * len(COMPOSITION_TASKS)


DIRECT_LR32_LEARNING_RATES = (
    5.0e-6,
    7.5e-6,
    1.0e-5,
    1.5e-5,
    2.0e-5,
    3.0e-5,
    4.0e-5,
    6.0e-5,
)
DIRECT_LR32_WARMUPS = (0.03, 0.10)
DIRECT_LR32_WEIGHT_DECAYS = (0.01, 0.10)


def direct_lr32_job_index(profile_id: str, heldout_task: str) -> int:
    try:
        profile_index = DIRECT_LR32_PROFILES.index(profile_id)
        fold_index = COMPOSITION_TASKS.index(heldout_task)
    except ValueError as exc:
        raise ValueError(f"invalid direct_lr32 row identity: {profile_id}/{heldout_task}") from exc
    return profile_index * len(COMPOSITION_TASKS) + fold_index


def direct_lr32_profile_values(profile_id: str) -> dict[str, Any]:
    """Return the exact settings for one of the 32 profiles."""

    if profile_id not in DIRECT_LR32_PROFILES:
        raise ValueError(f"invalid direct_lr32 profile: {profile_id}")
    index = DIRECT_LR32_PROFILES.index(profile_id)
    block = index // 4
    within = index % 4
    values = {
        "learning_rate": DIRECT_LR32_LEARNING_RATES[block],
        "warmup_ratio": DIRECT_LR32_WARMUPS[within // 2],
        "weight_decay": DIRECT_LR32_WEIGHT_DECAYS[within % 2],
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "max_grad_norm": 1.0,
    }
    return deepcopy(values)


def direct_lr32_profile_label(profile_id: str) -> str:
    values = direct_lr32_profile_values(profile_id)
    return (
        f"lr={values['learning_rate']:.6g};"
        f"warmup={values['warmup_ratio']:.2f};"
        f"wd={values['weight_decay']:.2f}"
    )


__all__ = [
    "DIRECT_LR32_PROTOCOL_ID",
    "DIRECT_LR32_OUTPUT_ROOT",
    "DIRECT_LR32_CONFIG_ROOT",
    "DIRECT_LR32_YAML_ROOT",
    "DIRECT_LR32_CONTROL_ROOT",
    "DIRECT_LR32_PROFILES",
    "DIRECT_LR32_ROUTES",
    "DIRECT_LR32_JOB_COUNT",
    "DIRECT_LR32_LEARNING_RATES",
    "DIRECT_LR32_WARMUPS",
    "DIRECT_LR32_WEIGHT_DECAYS",
    "direct_lr32_job_index",
    "direct_lr32_profile_values",
    "direct_lr32_profile_label",
]

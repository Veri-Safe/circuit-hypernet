"""Scientific configuration for this composition training sweep."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .protocol import COMPOSITION_TASKS


MANUAL160_PROTOCOL_ID = "s5_composition_loco_manual160_v2"
MANUAL160_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_manual160_v2"
MANUAL160_CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_manual160_v2"
MANUAL160_YAML_ROOT = "4server/0909/s5_composition_loco_manual160_v2"
MANUAL160_CONTROL_ROOT = f"{MANUAL160_OUTPUT_ROOT}/control"
MANUAL160_FOLD_COUNT = len(COMPOSITION_TASKS)
MANUAL160_JOB_COUNT = 160
MANUAL160_PROFILES = tuple(f"{index:02d}" for index in range(1, 9))
MANUAL160_ROUTES = ("direct_sft", "query_only", "raw_weight", "circuit_a")


def manual160_job_index(profile_id: str, heldout_task: str, route: str) -> int:
    """Return the canonical row index used to assign a scheduling wave."""

    try:
        profile_index = MANUAL160_PROFILES.index(profile_id)
        fold_index = COMPOSITION_TASKS.index(heldout_task)
        route_index = MANUAL160_ROUTES.index(route)
    except ValueError as exc:
        raise ValueError(f"invalid manual160 row identity: {profile_id}/{heldout_task}/{route}") from exc
    return (profile_index * MANUAL160_FOLD_COUNT + fold_index) * len(MANUAL160_ROUTES) + route_index


def _route_anchor(route: str) -> dict[str, Any]:
    if route == "direct_sft":
        return {
            "learning_rate": 1.5e-5,
            "weight_decay": 0.1,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
        }
    if route == "query_only":
        return {
            "pg_lr": 7.5e-5,
            "query_lr_multiplier": 0.5,
            "gate_max": 0.5,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.5,
            "base_preservation_coefficient": 0.5,
            "effective_delta_l2": 1.0e-4,
        }
    if route == "raw_weight":
        return {
            "pg_lr": 3.75e-5,
            "query_lr_multiplier": 0.5,
            "raw_encoder_lr": 2.0e-4,
            "raw_output_lr": 5.0e-4,
            "gate_max": 0.375,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.5,
            "base_preservation_coefficient": 0.5,
            "effective_delta_l2": 1.0e-4,
            "raw_fusion_mode": "query_cross",
        }
    if route == "circuit_a":
        return {
            "pg_lr": 2.5e-5,
            "query_lr_multiplier": 0.5,
            "circuit_lr_multiplier": 2.0,
            "gate_max": 0.5,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "warmup_ratio": 0.03,
            "max_grad_norm": 0.5,
            "base_preservation_coefficient": 0.5,
            "effective_delta_l2": 1.0e-4,
        }
    raise ValueError(f"unknown manual160 route: {route}")


def profile_values(route: str, profile_id: str) -> dict[str, Any]:
    """Return the exact, serialisable values for one route/profile pair."""

    if route not in MANUAL160_ROUTES or profile_id not in MANUAL160_PROFILES:
        raise ValueError(f"invalid manual160 route/profile: {route}/{profile_id}")
    values = deepcopy(_route_anchor(route))
    if profile_id == "02":
        for key in ("learning_rate", "pg_lr", "raw_encoder_lr", "raw_output_lr"):
            if key in values:
                values[key] *= 0.5
    elif profile_id == "03":
        for key in ("learning_rate", "pg_lr", "raw_encoder_lr", "raw_output_lr"):
            if key in values:
                values[key] *= 2.0
    elif profile_id == "04":
        values["warmup_ratio"] = 0.10
    elif profile_id == "05":
        values["weight_decay"] = 0.01
    elif profile_id == "06":
        for key in ("learning_rate", "pg_lr", "raw_encoder_lr", "raw_output_lr"):
            if key in values:
                values[key] *= 0.5
        values["weight_decay"] = 0.0
        if route != "direct_sft":
            values["gate_max"] = 0.25
    elif profile_id == "07":
        values["adam_beta2"] = 0.999
        if route == "direct_sft":
            # Direct profile 07 is explicitly D02 + beta2 probe.
            values["learning_rate"] *= 0.5
        elif route == "query_only":
            values["pg_lr"] = 3.75e-5
            values["query_lr_multiplier"] = 1.0
        elif route == "raw_weight":
            values["pg_lr"] = 1.875e-5
            values["query_lr_multiplier"] = 1.0
        elif route == "circuit_a":
            values["pg_lr"] = 1.25e-5
            values["query_lr_multiplier"] = 1.0
            values["circuit_lr_multiplier"] = 4.0
    elif profile_id == "08":
        if route == "direct_sft":
            values["weight_decay"] = 0.01
            values["warmup_ratio"] = 0.10
        else:
            values["base_preservation_coefficient"] = 0.25
            values["effective_delta_l2"] = 3.0e-5
    return values


def profile_label(route: str, profile_id: str) -> str:
    """Human-readable provenance for reports and resolved configs."""

    labels = {
        "01": "route anchor",
        "02": "all route LRs x0.5",
        "03": "all route LRs x2",
        "04": "warmup 0.10",
        "05": "weight decay 0.01",
        "06": "half-LR low-gate (HN) / half-LR zero-WD (Direct)",
        "07": "beta2 0.999 with route-specific ratio probe",
        "08": "regularization stress (Direct WD+warmup; HN preserve/delta)",
    }
    if route not in MANUAL160_ROUTES or profile_id not in MANUAL160_PROFILES:
        raise ValueError(f"invalid manual160 route/profile: {route}/{profile_id}")
    return labels[profile_id]


__all__ = [
    "MANUAL160_PROTOCOL_ID",
    "MANUAL160_OUTPUT_ROOT",
    "MANUAL160_CONFIG_ROOT",
    "MANUAL160_YAML_ROOT",
    "MANUAL160_CONTROL_ROOT",
    "MANUAL160_FOLD_COUNT",
    "MANUAL160_JOB_COUNT",
    "MANUAL160_PROFILES",
    "MANUAL160_ROUTES",
    "manual160_job_index",
    "profile_label",
    "profile_values",
]

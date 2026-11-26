"""Scientific configuration for this composition training sweep."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .composition_loco import COMPOSITION_TASKS
from .composition_loco_query_circuit_spec import (
    QUERY_CIRCUIT_PROFILES,
    QUERY_CIRCUIT_EVAL_BATCH_SIZE,
    circuit_profile_for_query_circuit,
    initial_checkpoint as query_circuit_initial_checkpoint,
    profile_label as query_circuit_profile_label,
    profile_values as query_circuit_profile_values,
)
from .composition_loco_query_weight_spec import (
    QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256,
    QUERY_WEIGHT_RAW_CACHE_PATH,
    QUERY_WEIGHT_RAW_CACHE_SHA256,
    QUERY_WEIGHT_PROFILES,
    profile_label as query_weight_profile_label,
    profile_values as query_weight_profile_values,
    QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT,
    QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256,
)


WEIGHT_CIRCUIT_PROTOCOL_ID = "s5_composition_loco_weight_circuit_v1"
WEIGHT_CIRCUIT_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_weight_circuit_v1"
WEIGHT_CIRCUIT_CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_weight_circuit_v1"
WEIGHT_CIRCUIT_YAML_ROOT = "4server/0912/s5_composition_loco_weight_circuit"
WEIGHT_CIRCUIT_CONTROL_ROOT = f"{WEIGHT_CIRCUIT_OUTPUT_ROOT}/control"
WEIGHT_CIRCUIT_PROFILES = tuple(QUERY_WEIGHT_PROFILES)
WEIGHT_CIRCUIT_ROUTES = ("raw_weight", "circuit_a")
WEIGHT_CIRCUIT_HELDOUT_TASKS = {
    "raw_weight": "T_and_or_and",
    "circuit_a": "T_nested_and",
}
WEIGHT_CIRCUIT_JOB_COUNT = len(WEIGHT_CIRCUIT_PROFILES) * len(WEIGHT_CIRCUIT_ROUTES)
WEIGHT_CIRCUIT_EVAL_BATCH_SIZE = QUERY_CIRCUIT_EVAL_BATCH_SIZE


def weight_circuit_job_index(profile_id: str, route: str) -> int:
    try:
        profile_index = WEIGHT_CIRCUIT_PROFILES.index(profile_id)
        route_index = WEIGHT_CIRCUIT_ROUTES.index(route)
    except ValueError as exc:
        raise ValueError(f"invalid weight_circuit row identity: {profile_id}/{route}") from exc
    return profile_index * len(WEIGHT_CIRCUIT_ROUTES) + route_index


def heldout_task(route: str) -> str:
    try:
        return WEIGHT_CIRCUIT_HELDOUT_TASKS[route]
    except KeyError as exc:
        raise ValueError(f"unknown weight_circuit route: {route}") from exc


def profile_values(route: str, profile_id: str) -> dict[str, Any]:
    """Return the predeclared QW/QC values for one WEIGHT_CIRCUIT row."""

    if route not in WEIGHT_CIRCUIT_ROUTES or profile_id not in WEIGHT_CIRCUIT_PROFILES:
        raise ValueError(f"invalid weight_circuit route/profile: {route}/{profile_id}")
    if route == "raw_weight":
        values = deepcopy(query_weight_profile_values("raw_weight", profile_id))
    else:
        values = deepcopy(query_circuit_profile_values("circuit_a", profile_id))
    values["eval_batch_size"] = WEIGHT_CIRCUIT_EVAL_BATCH_SIZE
    return values


def profile_label(route: str, profile_id: str) -> str:
    if route not in WEIGHT_CIRCUIT_ROUTES or profile_id not in WEIGHT_CIRCUIT_PROFILES:
        raise ValueError(f"invalid weight_circuit route/profile: {route}/{profile_id}")
    if route == "raw_weight":
        return f"Raw/{query_weight_profile_label(profile_id)}"
    label = query_circuit_profile_label(route, profile_id)
    if label.startswith("C/"):
        label = label[2:]
    return f"Circuit-A/{label}"


def initial_checkpoint(route: str) -> tuple[str, str, int]:
    if route == "raw_weight":
        return QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT, QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256, 30_516
    if route == "circuit_a":
        return query_circuit_initial_checkpoint("circuit_a")
    raise ValueError(f"unknown weight_circuit route: {route}")


if set(WEIGHT_CIRCUIT_PROFILES) != set(QUERY_CIRCUIT_PROFILES):
    raise RuntimeError("WEIGHT_CIRCUIT must reuse the exact 24-profile QW/QC table")
if set(WEIGHT_CIRCUIT_HELDOUT_TASKS) != set(WEIGHT_CIRCUIT_ROUTES):
    raise RuntimeError("WEIGHT_CIRCUIT held-out route table is incomplete")
if any(task not in COMPOSITION_TASKS for task in WEIGHT_CIRCUIT_HELDOUT_TASKS.values()):
    raise RuntimeError("WEIGHT_CIRCUIT held-out task is not a composition task")
for _route in WEIGHT_CIRCUIT_ROUTES:
    for _profile in WEIGHT_CIRCUIT_PROFILES:
        _values = profile_values(_route, _profile)
        _expected_gate = 0.375 if _route == "raw_weight" else 0.5
        if float(_values["gate_max"]) != _expected_gate:
            raise RuntimeError("WEIGHT_CIRCUIT gate must remain fixed at the route anchor")


__all__ = [
    "WEIGHT_CIRCUIT_PROTOCOL_ID",
    "WEIGHT_CIRCUIT_OUTPUT_ROOT",
    "WEIGHT_CIRCUIT_CONFIG_ROOT",
    "WEIGHT_CIRCUIT_YAML_ROOT",
    "WEIGHT_CIRCUIT_CONTROL_ROOT",
    "WEIGHT_CIRCUIT_PROFILES",
    "WEIGHT_CIRCUIT_ROUTES",
    "WEIGHT_CIRCUIT_HELDOUT_TASKS",
    "WEIGHT_CIRCUIT_JOB_COUNT",
    "WEIGHT_CIRCUIT_EVAL_BATCH_SIZE",
    "QUERY_WEIGHT_RAW_CACHE_PATH",
    "QUERY_WEIGHT_RAW_CACHE_SHA256",
    "QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256",
    "weight_circuit_job_index",
    "heldout_task",
    "profile_values",
    "profile_label",
    "initial_checkpoint",
    "circuit_profile_for_query_circuit",
]

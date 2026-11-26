"""Scientific configuration for this composition training sweep."""

from __future__ import annotations

from .composition_loco import COMPOSITION_TASKS
from .composition_loco_query_weight_spec import (
    QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256,
    QUERY_WEIGHT_RAW_CACHE_PATH,
    QUERY_WEIGHT_RAW_CACHE_SHA256,
    QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT,
    QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256,
    QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT,
    QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256,
    QUERY_WEIGHT_PROFILES,
    QUERY_WEIGHT_ROUTES,
    profile_label as query_weight_profile_label,
    profile_values as query_weight_profile_values,
)


NESTED_QUERY_WEIGHT_PROTOCOL_ID = "s5_composition_loco_t_nested_and_query_weight_v1"
NESTED_QUERY_WEIGHT_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_t_nested_and_query_weight_v1"
NESTED_QUERY_WEIGHT_CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_t_nested_and_query_weight_v1"
NESTED_QUERY_WEIGHT_YAML_ROOT = "4server/0913/s5_composition_loco_t_nested_and_query_weight"
NESTED_QUERY_WEIGHT_CONTROL_ROOT = f"{NESTED_QUERY_WEIGHT_OUTPUT_ROOT}/control"
NESTED_QUERY_WEIGHT_HELDOUT_TASK = "T_nested_and"
NESTED_QUERY_WEIGHT_FOLD_COUNT = 1
NESTED_QUERY_WEIGHT_PROFILES = tuple(f"P{index:02d}" for index in range(1, 17))
NESTED_QUERY_WEIGHT_ROUTES = tuple(QUERY_WEIGHT_ROUTES)
NESTED_QUERY_WEIGHT_JOB_COUNT = len(NESTED_QUERY_WEIGHT_PROFILES) * len(NESTED_QUERY_WEIGHT_ROUTES)
NESTED_QUERY_WEIGHT_EVAL_BATCH_SIZE = 16


def nested_query_weight_job_index(profile_id: str, route: str) -> int:
    try:
        profile_index = NESTED_QUERY_WEIGHT_PROFILES.index(profile_id)
        route_index = NESTED_QUERY_WEIGHT_ROUTES.index(route)
    except ValueError as exc:
        raise ValueError(f"invalid nested_query_weight row identity: {profile_id}/{route}") from exc
    return profile_index * len(NESTED_QUERY_WEIGHT_ROUTES) + route_index


def profile_values(route: str, profile_id: str) -> dict[str, object]:
    """Return the exact QUERY_WEIGHT values, with the fixed batched eval contract."""

    if route not in NESTED_QUERY_WEIGHT_ROUTES or profile_id not in NESTED_QUERY_WEIGHT_PROFILES:
        raise ValueError(f"invalid nested_query_weight route/profile: {route}/{profile_id}")
    values = dict(query_weight_profile_values(route, profile_id))
    values["eval_batch_size"] = NESTED_QUERY_WEIGHT_EVAL_BATCH_SIZE
    return values


def profile_label(route: str, profile_id: str) -> str:
    if route not in NESTED_QUERY_WEIGHT_ROUTES or profile_id not in NESTED_QUERY_WEIGHT_PROFILES:
        raise ValueError(f"invalid nested_query_weight route/profile: {route}/{profile_id}")
    return f"{route}/{query_weight_profile_label(profile_id)}"


def initial_checkpoint(route: str) -> tuple[str, str, int]:
    if route == "query_only":
        return QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT, QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256, 30_516
    if route == "raw_weight":
        return QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT, QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256, 30_516
    raise ValueError(f"unknown nested_query_weight route: {route}")


if NESTED_QUERY_WEIGHT_HELDOUT_TASK not in COMPOSITION_TASKS:
    raise RuntimeError("NESTED_QUERY_WEIGHT held-out task is not a composition task")
if NESTED_QUERY_WEIGHT_ROUTES != ("query_only", "raw_weight"):
    raise RuntimeError("NESTED_QUERY_WEIGHT must contain exactly Query-only and Raw-weight")
if NESTED_QUERY_WEIGHT_EVAL_BATCH_SIZE <= 1:
    raise RuntimeError("NESTED_QUERY_WEIGHT evaluation must be batched")
for _profile in NESTED_QUERY_WEIGHT_PROFILES:
    for _route in NESTED_QUERY_WEIGHT_ROUTES:
        _values = profile_values(_route, _profile)
        _expected_gate = 0.5 if _route == "query_only" else 0.375
        if _values.get("gate_max") != _expected_gate:
            raise RuntimeError("NESTED_QUERY_WEIGHT gate must remain at the QUERY_WEIGHT route anchor")


__all__ = [
    "NESTED_QUERY_WEIGHT_PROTOCOL_ID",
    "NESTED_QUERY_WEIGHT_OUTPUT_ROOT",
    "NESTED_QUERY_WEIGHT_CONFIG_ROOT",
    "NESTED_QUERY_WEIGHT_YAML_ROOT",
    "NESTED_QUERY_WEIGHT_CONTROL_ROOT",
    "NESTED_QUERY_WEIGHT_HELDOUT_TASK",
    "NESTED_QUERY_WEIGHT_FOLD_COUNT",
    "NESTED_QUERY_WEIGHT_PROFILES",
    "NESTED_QUERY_WEIGHT_ROUTES",
    "NESTED_QUERY_WEIGHT_JOB_COUNT",
    "NESTED_QUERY_WEIGHT_EVAL_BATCH_SIZE",
    "QUERY_WEIGHT_RAW_CACHE_PATH",
    "QUERY_WEIGHT_RAW_CACHE_SHA256",
    "QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256",
    "nested_query_weight_job_index",
    "profile_values",
    "profile_label",
    "initial_checkpoint",
]

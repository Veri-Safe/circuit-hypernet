"""Scientific configuration for this composition training sweep."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .composition_loco import COMPOSITION_TASKS
from .composition_loco_circuit_sweep40_spec import (
    CIRCUIT_SWEEP40_INITIAL_CHECKPOINT,
    CIRCUIT_SWEEP40_INITIAL_CHECKPOINT_SHA256,
    circuit_sweep40_profile_label,
    circuit_sweep40_profile_values,
)
from .composition_loco_query_weight_spec import (
    QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT,
    QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256,
    profile_label as query_weight_profile_label,
    profile_values as query_weight_profile_values,
)


QUERY_CIRCUIT_PROTOCOL_ID = "s5_composition_loco_t_and_or_and_query_circuit_v1"
QUERY_CIRCUIT_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_t_and_or_and_query_circuit_v1"
QUERY_CIRCUIT_CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_t_and_or_and_query_circuit_v1"
QUERY_CIRCUIT_YAML_ROOT = "4server/0912/s5_composition_loco_t_and_or_and_query_circuit"
QUERY_CIRCUIT_CONTROL_ROOT = f"{QUERY_CIRCUIT_OUTPUT_ROOT}/control"
QUERY_CIRCUIT_HELDOUT_TASK = "T_and_or_and"
QUERY_CIRCUIT_FOLD_COUNT = 1
QUERY_CIRCUIT_PROFILES = tuple(f"P{index:02d}" for index in range(1, 25))
QUERY_CIRCUIT_ROUTES = ("query_only", "circuit_a")
QUERY_CIRCUIT_JOB_COUNT = len(QUERY_CIRCUIT_PROFILES) * len(QUERY_CIRCUIT_ROUTES)
QUERY_CIRCUIT_EVAL_BATCH_SIZE = 16

# The post-selection artifacts are large; this is a preflight estimate only.
# It is intentionally conservative and does not trigger cleanup.

# Circuit-A profile IDs are selected from the already documented C40 table.
# The selection covers LR balance, preservation/delta regularization, training
# dynamics, and three predeclared interactions without adding a new search
# space or using held-out-task scores.
QUERY_CIRCUIT_CIRCUIT_PROFILES = (
    "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
    "C09", "C10", "C11", "C12", "C14", "C15", "C16", "C17",
    "C18", "C19", "C24", "C26", "C29", "C32", "C34", "C39",
)


def query_circuit_job_index(profile_id: str, route: str) -> int:
    """Return the canonical row index used by the one-wave matrix."""

    try:
        profile_index = QUERY_CIRCUIT_PROFILES.index(profile_id)
        route_index = QUERY_CIRCUIT_ROUTES.index(route)
    except ValueError as exc:
        raise ValueError(f"invalid query_circuit row identity: {profile_id}/{route}") from exc
    return profile_index * len(QUERY_CIRCUIT_ROUTES) + route_index


def circuit_profile_for_query_circuit(profile_id: str) -> str:
    """Map the 24 paired IDs to the predeclared Circuit-A C40 rows."""

    if profile_id not in QUERY_CIRCUIT_PROFILES:
        raise ValueError(f"invalid query_circuit profile: {profile_id}")
    return QUERY_CIRCUIT_CIRCUIT_PROFILES[QUERY_CIRCUIT_PROFILES.index(profile_id)]


def profile_values(route: str, profile_id: str) -> dict[str, Any]:
    """Return the complete serialisable values for one route/profile pair."""

    if route not in QUERY_CIRCUIT_ROUTES or profile_id not in QUERY_CIRCUIT_PROFILES:
        raise ValueError(f"invalid query_circuit route/profile: {route}/{profile_id}")
    if route == "query_only":
        values = deepcopy(query_weight_profile_values("query_only", profile_id))
    else:
        values = deepcopy(
            circuit_sweep40_profile_values(circuit_profile_for_query_circuit(profile_id))
        )
    values["eval_batch_size"] = QUERY_CIRCUIT_EVAL_BATCH_SIZE
    return values


def profile_label(route: str, profile_id: str) -> str:
    if route not in QUERY_CIRCUIT_ROUTES or profile_id not in QUERY_CIRCUIT_PROFILES:
        raise ValueError(f"invalid query_circuit route/profile: {route}/{profile_id}")
    if route == "query_only":
        return f"Q/{query_weight_profile_label(profile_id)}"
    circuit_id = circuit_profile_for_query_circuit(profile_id)
    return f"C/{circuit_id} {circuit_sweep40_profile_label(circuit_id)}"


def initial_checkpoint(route: str) -> tuple[str, str, int]:
    if route == "query_only":
        return QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT, QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256, 30_516
    if route == "circuit_a":
        return CIRCUIT_SWEEP40_INITIAL_CHECKPOINT, CIRCUIT_SWEEP40_INITIAL_CHECKPOINT_SHA256, 15_258
    raise ValueError(f"unknown query_circuit route: {route}")


if QUERY_CIRCUIT_HELDOUT_TASK not in COMPOSITION_TASKS:
    raise RuntimeError("QUERY_CIRCUIT held-out task is not a composition task")
if len(QUERY_CIRCUIT_CIRCUIT_PROFILES) != len(QUERY_CIRCUIT_PROFILES):
    raise RuntimeError("QUERY_CIRCUIT Circuit profile selection must contain 24 rows")
for _profile in QUERY_CIRCUIT_PROFILES:
    if profile_values("query_only", _profile)["gate_max"] != 0.5:
        raise RuntimeError("QUERY_CIRCUIT Query-only gate must remain fixed at 0.5")
    if profile_values("circuit_a", _profile)["gate_max"] != 0.5:
        raise RuntimeError("QUERY_CIRCUIT Circuit-A gate must remain fixed at 0.5")


__all__ = [
    "QUERY_CIRCUIT_PROTOCOL_ID",
    "QUERY_CIRCUIT_OUTPUT_ROOT",
    "QUERY_CIRCUIT_CONFIG_ROOT",
    "QUERY_CIRCUIT_YAML_ROOT",
    "QUERY_CIRCUIT_CONTROL_ROOT",
    "QUERY_CIRCUIT_HELDOUT_TASK",
    "QUERY_CIRCUIT_FOLD_COUNT",
    "QUERY_CIRCUIT_PROFILES",
    "QUERY_CIRCUIT_ROUTES",
    "QUERY_CIRCUIT_JOB_COUNT",
    "QUERY_CIRCUIT_EVAL_BATCH_SIZE",
    "QUERY_CIRCUIT_CIRCUIT_PROFILES",
    "query_circuit_job_index",
    "circuit_profile_for_query_circuit",
    "profile_values",
    "profile_label",
    "initial_checkpoint",
]

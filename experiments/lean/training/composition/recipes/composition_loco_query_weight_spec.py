"""Scientific configuration for this composition training sweep."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .protocol import COMPOSITION_TASKS


QUERY_WEIGHT_PROTOCOL_ID = "s5_composition_loco_t_and_or_query_weight_v1"
QUERY_WEIGHT_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_t_and_or_query_weight_v1"
QUERY_WEIGHT_CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_t_and_or_query_weight_v1"
QUERY_WEIGHT_YAML_ROOT = "4server/0912/s5_composition_loco_t_and_or_query_weight"
QUERY_WEIGHT_CONTROL_ROOT = f"{QUERY_WEIGHT_OUTPUT_ROOT}/control"
QUERY_WEIGHT_HELDOUT_TASK = "T_and_or"
QUERY_WEIGHT_FOLD_COUNT = 1
QUERY_WEIGHT_PROFILES = tuple(f"P{index:02d}" for index in range(1, 25))
QUERY_WEIGHT_ROUTES = ("query_only", "raw_weight")
QUERY_WEIGHT_JOB_COUNT = len(QUERY_WEIGHT_PROFILES) * len(QUERY_WEIGHT_ROUTES)

QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT = (
    "outputs/s5_lean_circuit/hypernet_query_only_atomic_specialization_seed42/"
    "qo02/checkpoints/model_step0030516.pt"
)
QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256 = (
    "80b51d000c9e48e1d5a8abe3bcb2a8e9ce9c86481db1977021dd6a0876c366d4"
)
QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT = (
    "outputs/s5_lean_circuit/hypernet_raw_atomic_seed42/raw_query16/"
    "rq14_p14/checkpoints/model_step0030516.pt"
)
QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256 = (
    "94dfb66fc1a050a90ac542482820d626cb5f8ef2e98554c8ac13806221bf4e63"
)
QUERY_WEIGHT_RAW_CACHE_PATH = (
    "outputs/s5_lean_circuit/hypernet_raw_weight_formal_1p2m_seed42/"
    "shared/raw_weight_cache.pt"
)
QUERY_WEIGHT_RAW_CACHE_SHA256 = (
    "886479c121f743184f5bb89eef2af30afb86a0278cb0b94dbcebfc0c98428334"
)
QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256 = (
    "a70eb76788bcef9c2ca301ad4a4d3febc94eb247ce1f02cf41ebdccda8bb87e1"
)


def query_weight_job_index(profile_id: str, route: str) -> int:
    """Return the canonical row index used by the one-wave matrix."""

    try:
        profile_index = QUERY_WEIGHT_PROFILES.index(profile_id)
        route_index = QUERY_WEIGHT_ROUTES.index(route)
    except ValueError as exc:
        raise ValueError(f"invalid query_weight row identity: {profile_id}/{route}") from exc
    return profile_index * len(QUERY_WEIGHT_ROUTES) + route_index


def _route_anchor(route: str) -> dict[str, Any]:
    if route == "query_only":
        return {
            "pg_lr": 7.5e-5,
            "query_lr_multiplier": 0.5,
            "gate_max": 0.5,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,
            "warmup_ratio": 0.03,
            "cosine_min_lr_ratio": 0.1,
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
            "cosine_min_lr_ratio": 0.1,
            "max_grad_norm": 0.5,
            "base_preservation_coefficient": 0.5,
            "effective_delta_l2": 1.0e-4,
            "raw_fusion_mode": "query_cross",
        }
    raise ValueError(f"unknown query_weight route: {route}")


def profile_values(route: str, profile_id: str) -> dict[str, Any]:
    """Return the exact serialisable values for one route/profile pair."""

    if route not in QUERY_WEIGHT_ROUTES or profile_id not in QUERY_WEIGHT_PROFILES:
        raise ValueError(f"invalid query_weight route/profile: {route}/{profile_id}")
    values = deepcopy(_route_anchor(route))

    # P01 is the route anchor.  P02--P06 give a compact monotone PG-learning
    # rate bracket shared by both routes; route-specific encoder/output rates
    # remain at their anchor values in this block.
    if profile_id == "P02":
        values["pg_lr"] = 3.75e-5 if route == "query_only" else 1.875e-5
    elif profile_id == "P03":
        values["pg_lr"] = 5.0e-5 if route == "query_only" else 2.5e-5
    elif profile_id == "P04":
        values["pg_lr"] = 6.0e-5 if route == "query_only" else 3.0e-5
    elif profile_id == "P05":
        values["pg_lr"] = 1.0e-4 if route == "query_only" else 5.0e-5
    elif profile_id == "P06":
        values["pg_lr"] = 1.5e-4 if route == "query_only" else 7.5e-5
    elif profile_id == "P07":
        values["query_lr_multiplier"] = 0.25
    elif profile_id == "P08":
        values["query_lr_multiplier"] = 1.0
    elif profile_id == "P09":
        values["query_lr_multiplier"] = 2.0
    elif profile_id == "P10":
        if route == "query_only":
            values.update({"pg_lr": 5.0e-5, "query_lr_multiplier": 1.0})
        else:
            values.update({"raw_encoder_lr": 1.0e-4, "raw_output_lr": 5.0e-4})
    elif profile_id == "P11":
        if route == "query_only":
            values.update({"pg_lr": 1.0e-4, "query_lr_multiplier": 0.25})
        else:
            values.update({"raw_encoder_lr": 4.0e-4, "raw_output_lr": 5.0e-4})
    elif profile_id == "P12":
        if route == "query_only":
            values.update({"pg_lr": 3.75e-5, "query_lr_multiplier": 1.5})
        else:
            values.update({"raw_encoder_lr": 2.0e-4, "raw_output_lr": 2.5e-4})
    elif profile_id == "P13":
        if route == "query_only":
            values.update({"pg_lr": 1.0e-4, "query_lr_multiplier": 1.0})
        else:
            values.update({"raw_encoder_lr": 2.0e-4, "raw_output_lr": 1.0e-3})
    elif profile_id == "P14":
        values["base_preservation_coefficient"] = 0.1
    elif profile_id == "P15":
        values["base_preservation_coefficient"] = 0.25
    elif profile_id == "P16":
        values["base_preservation_coefficient"] = 1.0
    elif profile_id == "P17":
        values["effective_delta_l2"] = 0.0
    elif profile_id == "P18":
        values["effective_delta_l2"] = 3.0e-5
    elif profile_id == "P19":
        values["effective_delta_l2"] = 3.0e-4
    elif profile_id == "P20":
        values["warmup_ratio"] = 0.10
    elif profile_id == "P21":
        values["cosine_min_lr_ratio"] = 0.01
    elif profile_id == "P22":
        values["adam_beta2"] = 0.999
    elif profile_id == "P23":
        if route == "query_only":
            values.update({
                "pg_lr": 3.75e-5,
                "base_preservation_coefficient": 0.25,
                "effective_delta_l2": 3.0e-5,
                "cosine_min_lr_ratio": 0.01,
            })
        else:
            values.update({
                "pg_lr": 1.875e-5,
                "raw_encoder_lr": 1.0e-4,
                "raw_output_lr": 2.5e-4,
                "base_preservation_coefficient": 0.25,
                "effective_delta_l2": 3.0e-5,
                "cosine_min_lr_ratio": 0.01,
            })
    elif profile_id == "P24":
        if route == "query_only":
            values.update({
                "pg_lr": 5.0e-5,
                "query_lr_multiplier": 1.0,
                "base_preservation_coefficient": 0.25,
                "effective_delta_l2": 3.0e-5,
                "cosine_min_lr_ratio": 0.01,
            })
        else:
            values.update({
                "pg_lr": 2.5e-5,
                "raw_encoder_lr": 4.0e-4,
                "raw_output_lr": 1.0e-3,
                "base_preservation_coefficient": 0.25,
                "effective_delta_l2": 3.0e-5,
                "cosine_min_lr_ratio": 0.01,
            })
    return values


_PROFILE_LABELS = {
    "P01": "route anchor",
    "P02": "PG LR lower bracket",
    "P03": "PG LR 5e-5/2.5e-5",
    "P04": "PG LR 6e-5/3e-5",
    "P05": "PG LR higher bracket",
    "P06": "PG LR high bracket",
    "P07": "query multiplier 0.25",
    "P08": "query multiplier 1.0",
    "P09": "query multiplier 2.0",
    "P10": "moderate PG/query or raw-encoder probe",
    "P11": "high PG/query or raw-encoder probe",
    "P12": "high query / low raw-output probe",
    "P13": "high PG/query or raw-output probe",
    "P14": "preservation 0.10",
    "P15": "preservation 0.25",
    "P16": "preservation 1.00",
    "P17": "delta L2 zero",
    "P18": "delta L2 3e-5",
    "P19": "delta L2 3e-4",
    "P20": "warmup 0.10",
    "P21": "cosine floor 0.01",
    "P22": "Adam beta2 0.999",
    "P23": "conservative combined probe",
    "P24": "best-bet combined probe",
}


def profile_label(profile_id: str) -> str:
    if profile_id not in QUERY_WEIGHT_PROFILES:
        raise ValueError(f"invalid query_weight profile: {profile_id}")
    return _PROFILE_LABELS[profile_id]


# Import-time checks keep accidental edits to the matrix obvious in tests.
if QUERY_WEIGHT_HELDOUT_TASK not in COMPOSITION_TASKS:
    raise RuntimeError("QUERY_WEIGHT held-out task is not a composition task")
for _profile in QUERY_WEIGHT_PROFILES:
    for _route in QUERY_WEIGHT_ROUTES:
        _profile_values = profile_values(_route, _profile)
        if _profile_values.get("gate_max") != _route_anchor(_route)["gate_max"]:
            raise RuntimeError("QUERY_WEIGHT gate value must remain fixed")


__all__ = [
    "QUERY_WEIGHT_PROTOCOL_ID",
    "QUERY_WEIGHT_OUTPUT_ROOT",
    "QUERY_WEIGHT_CONFIG_ROOT",
    "QUERY_WEIGHT_YAML_ROOT",
    "QUERY_WEIGHT_CONTROL_ROOT",
    "QUERY_WEIGHT_HELDOUT_TASK",
    "QUERY_WEIGHT_FOLD_COUNT",
    "QUERY_WEIGHT_PROFILES",
    "QUERY_WEIGHT_ROUTES",
    "QUERY_WEIGHT_JOB_COUNT",
    "QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT",
    "QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256",
    "QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT",
    "QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256",
    "QUERY_WEIGHT_RAW_CACHE_PATH",
    "QUERY_WEIGHT_RAW_CACHE_SHA256",
    "QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256",
    "query_weight_job_index",
    "profile_values",
    "profile_label",
]

"""Scientific configuration for this composition training sweep."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .composition_loco import COMPOSITION_TASKS


CIRCUIT_SWEEP40_PROTOCOL_ID = "s5_composition_loco_circuit_sweep40_v1"
CIRCUIT_SWEEP40_OUTPUT_ROOT = (
    "outputs/s5_lean_circuit/composition_loco_circuit_sweep40_v1"
)
CIRCUIT_T_AND_OR40_OUTPUT_ROOT = (
    "outputs/s5_lean_circuit/composition_loco_circuit_t_and_or40_v1"
)
CIRCUIT_T_AND_OR40_CONFIG_ROOT = (
    "s5_lean_circuit/configs/composition_loco_circuit_t_and_or40_v1"
)
CIRCUIT_SWEEP40_CONFIG_ROOT = (
    "s5_lean_circuit/configs/composition_loco_circuit_sweep40_v1"
)
CIRCUIT_SWEEP40_YAML_ROOT = "4server/0910/s5_composition_loco_circuit_sweep40_v1"
CIRCUIT_SWEEP40_CONTROL_ROOT = f"{CIRCUIT_SWEEP40_OUTPUT_ROOT}/control"
CIRCUIT_SWEEP40_PROFILES = tuple(f"C{index:02d}" for index in range(1, 41))
CIRCUIT_SWEEP40_ROUTES = ("circuit_a",)
CIRCUIT_SWEEP40_FOLD_COUNT = len(COMPOSITION_TASKS)
CIRCUIT_SWEEP40_JOB_COUNT = len(CIRCUIT_SWEEP40_PROFILES)
CIRCUIT_SWEEP40_CELL_COUNT = CIRCUIT_SWEEP40_JOB_COUNT * CIRCUIT_SWEEP40_FOLD_COUNT

# Every profile starts from the same frozen A-P03 endpoint.  Keeping this
# identity here lets the validator reject an accidental continuation from a
# smoke checkpoint without embedding a filesystem dependency in the module.
CIRCUIT_SWEEP40_INITIAL_CHECKPOINT = (
    "outputs/s5_lean_circuit/hypernet_ab_formal_1p2m_seed42/"
    "formal_lr_gate_r4a8/a_p03/checkpoints/model_step0015258.pt"
)
CIRCUIT_SWEEP40_INITIAL_CHECKPOINT_SHA256 = (
    "d948792b7206d36f1a06697428fcb928aefaf7db120558529fae3c38ab957cde"
)
CIRCUIT_SWEEP40_INITIAL_STEP = 15_258
CIRCUIT_SWEEP40_BASE_STEP = 38_000


_C01: dict[str, Any] = {
    "pg_lr": 2.5e-5,
    "query_lr_multiplier": 0.5,
    "circuit_lr_multiplier": 2.0,
    "base_preservation_coefficient": 0.5,
    "effective_delta_l2": 1.0e-4,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "max_grad_norm": 0.5,
    "cosine_min_lr_ratio": 0.1,
    "gate_max": 0.5,
}


# Values are complete deviations from C01.  The table deliberately uses
# ordinary values and small, interpretable perturbations; omitted keys inherit
# C01 and are never inherited from the previous row.
_OVERRIDES: dict[str, dict[str, Any]] = {
    "C01": {},
    "C02": {"pg_lr": 1.25e-5},
    "C03": {"pg_lr": 5.0e-5},
    "C04": {"pg_lr": 7.5e-5},
    "C05": {"query_lr_multiplier": 0.25},
    "C06": {"query_lr_multiplier": 1.0},
    "C07": {"circuit_lr_multiplier": 1.0},
    "C08": {"circuit_lr_multiplier": 4.0},
    "C09": {"pg_lr": 1.25e-5, "query_lr_multiplier": 0.25, "circuit_lr_multiplier": 4.0},
    "C10": {"pg_lr": 1.25e-5, "query_lr_multiplier": 1.0, "circuit_lr_multiplier": 4.0},
    "C11": {"pg_lr": 5.0e-5, "query_lr_multiplier": 0.25, "circuit_lr_multiplier": 4.0},
    "C12": {"pg_lr": 5.0e-5, "query_lr_multiplier": 1.0, "circuit_lr_multiplier": 1.0},
    "C13": {"base_preservation_coefficient": 0.0},
    "C14": {"base_preservation_coefficient": 0.1},
    "C15": {"base_preservation_coefficient": 0.25},
    "C16": {"base_preservation_coefficient": 1.0},
    "C17": {"effective_delta_l2": 0.0},
    "C18": {"effective_delta_l2": 3.0e-5},
    "C19": {"effective_delta_l2": 3.0e-4},
    "C20": {"effective_delta_l2": 1.0e-3},
    "C21": {"weight_decay": 0.01},
    "C22": {"weight_decay": 0.1},
    "C23": {"warmup_ratio": 0.01},
    "C24": {"warmup_ratio": 0.1},
    "C25": {"adam_beta2": 0.95},
    "C26": {"adam_beta2": 0.999},
    "C27": {"max_grad_norm": 0.25},
    "C28": {"max_grad_norm": 1.0},
    "C29": {"cosine_min_lr_ratio": 0.01},
    "C30": {"cosine_min_lr_ratio": 0.3},
    "C31": {
        "pg_lr": 1.25e-5,
        "query_lr_multiplier": 0.25,
        "circuit_lr_multiplier": 4.0,
        "base_preservation_coefficient": 0.25,
        "effective_delta_l2": 3.0e-5,
    },
    "C32": {
        "query_lr_multiplier": 0.25,
        "circuit_lr_multiplier": 4.0,
        "base_preservation_coefficient": 0.25,
        "effective_delta_l2": 3.0e-5,
    },
    "C33": {
        "base_preservation_coefficient": 0.25,
        "effective_delta_l2": 3.0e-5,
    },
    "C34": {
        "circuit_lr_multiplier": 4.0,
        "base_preservation_coefficient": 0.25,
        "effective_delta_l2": 3.0e-5,
    },
    "C35": {
        "pg_lr": 5.0e-5,
        "query_lr_multiplier": 0.25,
        "effective_delta_l2": 3.0e-4,
        "warmup_ratio": 0.1,
    },
    "C36": {
        "pg_lr": 5.0e-5,
        "base_preservation_coefficient": 1.0,
        "effective_delta_l2": 3.0e-4,
    },
    "C37": {"weight_decay": 0.01, "warmup_ratio": 0.1, "adam_beta2": 0.999},
    "C38": {
        "pg_lr": 5.0e-5,
        "query_lr_multiplier": 0.25,
        "circuit_lr_multiplier": 4.0,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
    },
    "C39": {
        "pg_lr": 1.25e-5,
        "query_lr_multiplier": 1.0,
        "circuit_lr_multiplier": 4.0,
        "base_preservation_coefficient": 0.25,
        "effective_delta_l2": 3.0e-5,
    },
    "C40": {
        "base_preservation_coefficient": 0.25,
        "effective_delta_l2": 3.0e-5,
        "weight_decay": 0.01,
        "adam_beta2": 0.95,
    },
}


_LABELS: dict[str, str] = {
    "C01": "original LR reference",
    "C02": "main LR x0.5",
    "C03": "main LR x2",
    "C04": "higher main LR",
    "C05": "slow query branch",
    "C06": "fast query branch",
    "C07": "slow circuit branch",
    "C08": "fast circuit branch",
    "C09": "low LR, slow query, fast circuit",
    "C10": "low LR, both encoders faster",
    "C11": "high LR, slow query, fast circuit",
    "C12": "high LR, equal encoder rates",
    "C13": "remove base-preservation term",
    "C14": "weak base-preservation term",
    "C15": "half base-preservation term",
    "C16": "strong base-preservation term",
    "C17": "remove delta regularizer",
    "C18": "weak delta regularizer",
    "C19": "strong delta regularizer",
    "C20": "very strong delta regularizer",
    "C21": "small AdamW weight decay",
    "C22": "large AdamW weight decay",
    "C23": "short warmup",
    "C24": "long warmup",
    "C25": "faster beta2 response",
    "C26": "smoother beta2",
    "C27": "stronger gradient clipping",
    "C28": "wider gradient clipping",
    "C29": "lower terminal LR",
    "C30": "higher terminal LR",
    "C31": "low LR, circuit-heavy, weak regularization",
    "C32": "circuit-heavy, weak regularization",
    "C33": "balanced update, weak regularization",
    "C34": "fast circuit, weak regularization",
    "C35": "high LR, long warmup, delta regularization",
    "C36": "high LR, strong preservation and delta regularization",
    "C37": "long warmup, smooth beta2, small WD",
    "C38": "high LR, fast circuit, long warmup, wide clip",
    "C39": "low LR, fast encoders, weak regularization",
    "C40": "weak regularization, fast response, small WD",
}


def circuit_sweep40_profile_values(profile_id: str) -> dict[str, Any]:
    """Return a fresh complete parameter mapping for one profile."""

    if profile_id not in CIRCUIT_SWEEP40_PROFILES:
        raise ValueError(f"invalid Circuit sweep40 profile: {profile_id}")
    values = deepcopy(_C01)
    values.update(_OVERRIDES[profile_id])
    return values


def circuit_sweep40_profile_label(profile_id: str) -> str:
    if profile_id not in CIRCUIT_SWEEP40_PROFILES:
        raise ValueError(f"invalid Circuit sweep40 profile: {profile_id}")
    return _LABELS[profile_id]


def circuit_sweep40_job_index(profile_id: str) -> int:
    try:
        return CIRCUIT_SWEEP40_PROFILES.index(profile_id)
    except ValueError as exc:
        raise ValueError(f"invalid Circuit sweep40 profile: {profile_id}") from exc


def circuit_sweep40_optimizer_peak_lrs(profile_id: str) -> dict[str, float]:
    values = circuit_sweep40_profile_values(profile_id)
    pg = float(values["pg_lr"])
    return {
        "pg": pg,
        "query": pg * float(values["query_lr_multiplier"]),
        "circuit": pg * float(values["circuit_lr_multiplier"]),
    }


def _validate_table() -> None:
    if len(_OVERRIDES) != len(CIRCUIT_SWEEP40_PROFILES) or set(_OVERRIDES) != set(CIRCUIT_SWEEP40_PROFILES):
        raise RuntimeError("Circuit sweep40 profile table is incomplete")
    if set(_LABELS) != set(CIRCUIT_SWEEP40_PROFILES):
        raise RuntimeError("Circuit sweep40 profile labels are incomplete")
    for profile_id in CIRCUIT_SWEEP40_PROFILES:
        values = circuit_sweep40_profile_values(profile_id)
        if not 0 < float(values["pg_lr"]):
            raise RuntimeError("Circuit sweep40 PG learning rates must be positive")
        if not 0 < float(values["query_lr_multiplier"]):
            raise RuntimeError("Circuit sweep40 query multipliers must be positive")
        if not 0 < float(values["circuit_lr_multiplier"]):
            raise RuntimeError("Circuit sweep40 circuit multipliers must be positive")
        if float(values["gate_max"]) != 0.5:
            raise RuntimeError("Circuit sweep40 gate must remain fixed at 0.5")


_validate_table()


__all__ = [
    "CIRCUIT_SWEEP40_PROTOCOL_ID",
    "CIRCUIT_SWEEP40_OUTPUT_ROOT",
    "CIRCUIT_T_AND_OR40_OUTPUT_ROOT",
    "CIRCUIT_T_AND_OR40_CONFIG_ROOT",
    "CIRCUIT_SWEEP40_CONFIG_ROOT",
    "CIRCUIT_SWEEP40_YAML_ROOT",
    "CIRCUIT_SWEEP40_CONTROL_ROOT",
    "CIRCUIT_SWEEP40_PROFILES",
    "CIRCUIT_SWEEP40_ROUTES",
    "CIRCUIT_SWEEP40_FOLD_COUNT",
    "CIRCUIT_SWEEP40_JOB_COUNT",
    "CIRCUIT_SWEEP40_CELL_COUNT",
    "CIRCUIT_SWEEP40_INITIAL_CHECKPOINT",
    "CIRCUIT_SWEEP40_INITIAL_CHECKPOINT_SHA256",
    "CIRCUIT_SWEEP40_INITIAL_STEP",
    "CIRCUIT_SWEEP40_BASE_STEP",
    "circuit_sweep40_profile_values",
    "circuit_sweep40_profile_label",
    "circuit_sweep40_job_index",
    "circuit_sweep40_optimizer_peak_lrs",
]

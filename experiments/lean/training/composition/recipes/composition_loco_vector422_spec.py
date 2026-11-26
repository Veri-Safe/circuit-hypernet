"""Five paired Vector422 continuations of the seen-selected Vector397 profiles.

Profile choice is historical seen validation only. Every new student and
teacher starts from the original atomic A-P03, never a composition endpoint.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .composition_loco_circuit_retention20_spec import CIRCUIT_RETENTION20_CONFIG_ROOT


VECTOR422_PROTOCOL_ID = "s5_composition_loco_vector422_v1"
VECTOR422_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_vector422_v1"
VECTOR422_CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_vector422_v1"
VECTOR422_MIGRATION = "zero_prefix25_copy_tail397_v1"
VECTOR422_PROFILE_BY_HELDOUT = {
    "T_and_or": "C25",
    "T_or_and": "M25",
    "T_and_or_and": "M12",
    "T_nested_and": "C25",
    "T_nested_or": "M25",
}
VECTOR422_SOURCE_KEYS = ("composition", "atomic_replay", "validation", "atomic_bank")
VECTOR422_JOB_COUNT = 5


def vector422_expected_output(heldout_task: str) -> str:
    group = VECTOR422_PROFILE_BY_HELDOUT[str(heldout_task)]
    return f"{VECTOR422_OUTPUT_ROOT}/runs/{heldout_task}/{group}/circuit_a"


def build_vector422_config(heldout_task: str) -> dict[str, Any]:
    """Copy the exact historical profile; change only the declared input view."""
    from .composition_loco_config import load_config

    try:
        group = VECTOR422_PROFILE_BY_HELDOUT[str(heldout_task)]
    except KeyError as exc:
        raise ValueError(f"unknown Vector422 heldout task: {heldout_task}") from exc
    root = Path(__file__).resolve().parents[3]
    source = root / CIRCUIT_RETENTION20_CONFIG_ROOT / heldout_task / group / "circuit_a.json"
    # Validate the original protocol and its checksum before deriving a new one.
    value = deepcopy(load_config(source).to_dict())
    value["protocol_id"] = VECTOR422_PROTOCOL_ID
    value["output_dir"] = vector422_expected_output(heldout_task)
    value["data"]["vector422_sources"] = {
        key: f"{VECTOR422_OUTPUT_ROOT}/conditions/{key}/manifest.json"
        for key in VECTOR422_SOURCE_KEYS
    }
    value["model"]["circuit_input_dim"] = 422
    value["model"]["initialization_migration"] = VECTOR422_MIGRATION
    # Final is outside this round. Do not create or consume its 422 condition bank.
    value["evaluation"]["final_authorized"] = False
    value["evaluation"]["ood_conditions"] = ["matched", "shuffled", "absent"]
    return value


def validate_vector422_config(config: Any) -> None:
    """Fail closed on any undeclared optimizer, sampler, init or selector change."""
    if config.protocol_id != VECTOR422_PROTOCOL_ID:
        raise ValueError("Vector422 protocol changed")
    expected = build_vector422_config(config.heldout_task)
    if config.to_dict() != expected:
        raise ValueError(
            "Vector422 config must preserve the sealed seen-selected profile, "
            "atomic A-P03 initialization and four identity-bound condition sources"
        )


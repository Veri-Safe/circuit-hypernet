from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from nvarc_qwen3.arc_eval_protocol import (
    ARC_AGI1_INTERNAL_CONFIG_SHA256,
    ARC_AGI1_INTERNAL_PROTOCOL_VERSION,
    ARC_AGI1_PUBLIC_VALIDATION_CONFIG_SHA256,
    ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION,
    ARC_AGI1_TASK_COUNT,
    ARC_AGI1_TEST_COUNT,
    ARC_AGI2_TASK_COUNT,
    ARC_AGI2_TEST_COUNT,
    INTERNAL_CONFIG_SHA256,
    INTERNAL_PROTOCOL_VERSION,
    load_frozen_config,
    validate_arc_agi1_identity,
    validate_arc_agi2_identity,
)

from .config import EvaluationConfig


@dataclass(frozen=True)
class EvalProtocolContract:
    protocol_kind: str
    protocol_version: str
    config_sha256: str
    task_count: int
    test_count: int


_CONTRACTS = {
    "arc_agi1_internal": EvalProtocolContract(
        protocol_kind="arc_agi1_internal",
        protocol_version=ARC_AGI1_INTERNAL_PROTOCOL_VERSION,
        config_sha256=ARC_AGI1_INTERNAL_CONFIG_SHA256,
        task_count=ARC_AGI1_TASK_COUNT,
        test_count=ARC_AGI1_TEST_COUNT,
    ),
    "arc_agi1_public_validation": EvalProtocolContract(
        protocol_kind="arc_agi1_public_validation",
        protocol_version=ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION,
        config_sha256=ARC_AGI1_PUBLIC_VALIDATION_CONFIG_SHA256,
        task_count=ARC_AGI1_TASK_COUNT,
        test_count=ARC_AGI1_TEST_COUNT,
    ),
    "arc_agi2_internal": EvalProtocolContract(
        protocol_kind="arc_agi2_internal",
        protocol_version=INTERNAL_PROTOCOL_VERSION,
        config_sha256=INTERNAL_CONFIG_SHA256,
        task_count=ARC_AGI2_TASK_COUNT,
        test_count=ARC_AGI2_TEST_COUNT,
    ),
}


def protocol_contract(protocol_kind: str) -> EvalProtocolContract:
    try:
        return _CONTRACTS[protocol_kind]
    except KeyError as error:
        raise ValueError(f"Unsupported NVARC PG eval protocol: {protocol_kind!r}") from error


def load_profile_eval_protocol(
    config: EvaluationConfig,
) -> tuple[dict[str, Any], str]:
    contract = protocol_contract(config.protocol_kind)
    return load_frozen_config(
        config.protocol_config,
        contract.protocol_version,
        contract.config_sha256,
    )


def validate_profile_dataset_identity(
    protocol_kind: str,
    challenges: Mapping[str, Any],
    solutions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if protocol_kind in {"arc_agi1_internal", "arc_agi1_public_validation"}:
        return validate_arc_agi1_identity(challenges, solutions)
    if protocol_kind == "arc_agi2_internal":
        return validate_arc_agi2_identity(challenges, solutions)
    raise ValueError(f"Unsupported NVARC PG eval protocol: {protocol_kind!r}")

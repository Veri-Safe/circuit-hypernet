from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


CONFIG_FORMAT = "nvarc_s4_r1_config_v1"
JOIN_FORMAT = "nvarc_s4_r1_event_join_v1"
EVAL_JOIN_FORMAT = "nvarc_s4_r1_eval_join_v1"
FRESH_INIT_FORMAT = "nvarc_s4_r1_fresh_init_v1"
BATCH_MANIFEST_FORMAT = "nvarc_s4_r1_batch_manifest_v1"
PREFLIGHT_FORMAT = "nvarc_s4_r1_g3_preflight_v1"
CHECKPOINT_FORMAT = "nvarc_s4_r1_checkpoint_v2"
TRAIN_RUN_FORMAT = "nvarc_s4_r1_train_run_v1"
EVALUATION_FORMAT = "nvarc_s4_r1_evaluation_v1"
SELECTION_FORMAT = "nvarc_s4_r1_checkpoint_selection_v1"
SUMMARY_FORMAT = "nvarc_s4_r1_summary_v1"
FULL_EVAL_TASK_COUNT = 400
FULL_EVAL_QUERY_COUNT = 7_123
CANONICAL_VALIDATION_QUERY_COUNT = 419
AUTHOR4_VALIDATION_QUERY_COUNT = 1_676

VECTOR_SCHEMA = "nvarc_qwen3_36l_joint_signed_span_mean_59d_v1"
VECTOR_DIM = 59
FINAL_SHARD_FORMAT = "nvarc_rearc_circuit_final_shard_v1"
FINAL_ROOT_FORMAT = "nvarc_rearc_circuit_final_root_v1"
CATALOG_FORMAT = "nvarc_rearc_event_catalog_v1"

FUSION_VARIANTS = (
    "f0_single_token",
    "f1_layer_shared_memory",
    "f2_fully_direct",
    "f3_shared_attention",
    "f3_independent_attention",
)

EXPECTED_S2_PROFILE = "arc1_rearc8_s42_step130"
EXPECTED_BASE_AGGREGATE_SHA256 = (
    "b74bcb895fe162511ab8d047648a9d3cbba110fead5f3d58b9a24e9456d0084c"
)
EXPECTED_ROWS = 3_200
EXPECTED_PUZZLES = 400
EXPECTED_EVENTS = 20_826
EXPECTED_LAYERS = 36
EXPECTED_GENERATOR_SLOTS = 37
EXPECTED_MAX_STEPS = 650
EXPECTED_GLOBAL_ROW_BATCH = 256
EXPECTED_SCHEDULE_SHA256 = (
    "2a98e42bb28f5be78e6c48d852d461d84b2ad4f3980100043ac3c2582f650868"
)
EXPECTED_UPSTREAM_API_CONTRACT_SHA256 = (
    "9fc0a94cac0e2066c3ad539b8cbfa04312d1f8da972db0c5e51ecbdcad1de886"
)

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def validate_full_evaluation_request(max_eval_tasks: int | None) -> None:
    if max_eval_tasks is not None:
        raise ValueError(
            "Task-subset evaluation is disabled. Run the complete 400-task "
            "ARC/NVARC evaluation; max_eval_tasks must be omitted."
        )


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(values: tuple[int, ...] | list[int]) -> str:
    digest = hashlib.sha256()
    for item in values:
        token_id = int(item)
        if not -(2**15) <= token_id < 2**15:
            raise ValueError(f"Token ID is outside signed int16: {token_id}")
        digest.update(token_id.to_bytes(2, byteorder="little", signed=True))
    return digest.hexdigest()


def expand_environment(value: Any) -> Any:
    """Expand known ${VAR}s while retaining unresolved pins for static checks."""

    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(0)),
            value,
        )
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def is_sha256(value: str) -> bool:
    return bool(_SHA256_PATTERN.fullmatch(value))


def require_sha256(value: str, *, label: str) -> str:
    if not is_sha256(value):
        raise ValueError(f"{label} must be pinned to a lowercase SHA256, found {value!r}")
    return value


def callable_signature(value: Any) -> str:
    return str(inspect.signature(value))


def upstream_api_contract() -> Mapping[str, str]:
    """Record the narrow S2/S3 API surface imported by S4."""

    from s2_hypernet.nvarc_pg.config import load_experiment_config
    from s2_hypernet.nvarc_pg.events import build_row_events
    from s2_hypernet.nvarc_pg.model import FrozenQwenPerQueryLoRA
    from s2_hypernet.nvarc_pg.parameter_generator import NvarcParameterGenerator
    from s3circuit.nvarc_circuit.events import read_catalog_chunk
    from s3circuit.nvarc_circuit.finalize import validate_final_shard

    values = {
        "s2.load_experiment_config": callable_signature(load_experiment_config),
        "s2.build_row_events": callable_signature(build_row_events),
        "s2.FrozenQwenPerQueryLoRA.condition_batch": callable_signature(
            FrozenQwenPerQueryLoRA.condition_batch
        ),
        "s2.NvarcParameterGenerator.forward": callable_signature(
            NvarcParameterGenerator.forward
        ),
        "s3.read_catalog_chunk": callable_signature(read_catalog_chunk),
        "s3.validate_final_shard": callable_signature(validate_final_shard),
    }
    return dict(sorted(values.items()))


def upstream_api_contract_sha256() -> str:
    return semantic_sha256(upstream_api_contract())


def validate_upstream_api_contract() -> Mapping[str, str]:
    contract = upstream_api_contract()
    actual = semantic_sha256(contract)
    if actual != EXPECTED_UPSTREAM_API_CONTRACT_SHA256:
        raise ValueError(
            "Imported S2/S3 API signatures changed: "
            f"{actual} != {EXPECTED_UPSTREAM_API_CONTRACT_SHA256}"
        )
    return contract

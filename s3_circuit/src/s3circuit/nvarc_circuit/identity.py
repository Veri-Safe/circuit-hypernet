from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


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


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        value = int(token_id)
        if not -(2**15) <= value < 2**15:
            raise ValueError(
                f"Token ID is outside the signed-int16 identity range: {value}"
            )
        digest.update(value.to_bytes(2, byteorder="little", signed=True))
    return digest.hexdigest()


def event_key(dataset_snapshot_sha256: str, dataset_index: int, pair_index: int) -> str:
    if len(dataset_snapshot_sha256) != 64:
        raise ValueError("dataset_snapshot_sha256 must be a SHA256 hex digest")
    if dataset_index < 0 or pair_index < 0:
        raise ValueError("Event indices must be nonnegative")
    return semantic_sha256(
        {
            "format": "nvarc_rearc_event_key_v1",
            "dataset_snapshot_sha256": dataset_snapshot_sha256,
            "dataset_index": int(dataset_index),
            "pair_index": int(pair_index),
        }
    )


def artifact_id(
    *,
    event_key_value: str,
    condition_token_ids: Sequence[int],
    generated_token_ids: Sequence[int],
    model_identity_sha256: str,
    tokenizer_identity_sha256: str,
    transcoder_identity_sha256: str,
    operator_identity: str,
    backend_identity: str,
    production_policy_sha256: str,
    payload_schema: str,
    vector_schema: str,
    trail_schema: str,
) -> str:
    hashes = {
        "event_key": event_key_value,
        "model_identity_sha256": model_identity_sha256,
        "tokenizer_identity_sha256": tokenizer_identity_sha256,
        "transcoder_identity_sha256": transcoder_identity_sha256,
        "production_policy_sha256": production_policy_sha256,
    }
    invalid = {key: value for key, value in hashes.items() if len(value) != 64}
    if invalid:
        raise ValueError(f"Artifact identity contains invalid SHA256 values: {invalid}")
    return semantic_sha256(
        {
            "format": "nvarc_rearc_circuit_artifact_id_v1",
            **hashes,
            "condition_token_ids_sha256": token_ids_sha256(condition_token_ids),
            "generated_token_ids_sha256": token_ids_sha256(generated_token_ids),
            "operator_identity": operator_identity,
            "backend_identity": backend_identity,
            "payload_schema": payload_schema,
            "vector_schema": vector_schema,
            "trail_schema": trail_schema,
        }
    )


def dataset_snapshot_identity(
    *,
    fingerprint: str,
    rows: int,
    puzzles: int,
    events: int,
    schema: Mapping[str, Any] | None = None,
) -> str:
    if not fingerprint:
        raise ValueError("Dataset fingerprint is required")
    if min(rows, puzzles, events) <= 0:
        raise ValueError("Dataset identity counts must be positive")
    return semantic_sha256(
        {
            "format": "nvarc_rearc_dataset_snapshot_v1",
            "fingerprint": fingerprint,
            "rows": int(rows),
            "puzzles": int(puzzles),
            "events": int(events),
            "schema": dict(schema or {}),
        }
    )


def model_identity_from_base_identity(value: Mapping[str, Any]) -> str:
    if value.get("format") == "nvarc_rearc_pinned_hf_base_identity_v2":
        fields = (
            "format",
            "label",
            "run_name",
            "source_step",
            "hf_aggregate_sha256",
            "model_config_sha256",
            "hf_artifacts",
        )
        missing = [field for field in fields if field not in value]
        if missing:
            raise ValueError(
                f"Pinned HF identity is missing model content fields: {missing}"
            )
        return semantic_sha256(
            {
                "format": "nvarc_rearc_pinned_hf_model_content_identity_v1",
                **{field: value[field] for field in fields},
            }
        )
    fields = (
        "format",
        "run_name",
        "source_step",
        "run_manifest_sha256",
        "model_config_sha256",
        "hf_artifacts",
    )
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"Base identity is missing model content fields: {missing}")
    return semantic_sha256(
        {
            "format": "nvarc_rearc_step399_model_content_identity_v1",
            **{field: value[field] for field in fields},
        }
    )


def tokenizer_identity_from_base_identity(value: Mapping[str, Any]) -> str:
    fields = ("tokenizer_config_sha256", "token_mapping_sha256")
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(
            f"Base identity is missing tokenizer content fields: {missing}"
        )
    return semantic_sha256(
        {
            "format": "nvarc_rearc_16t_tokenizer_content_identity_v1",
            **{field: value[field] for field in fields},
        }
    )

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .finalize import validate_final_shard
from .identity import sha256_file


PAYLOAD_WORK_UNIT_FORMAT = "nvarc_rearc_payload_work_unit_v1"
FINAL_WORK_UNIT_FORMAT = "nvarc_rearc_circuit_final_work_unit_v1"


def read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {source}")
    return value


def _require_fields(
    value: dict[str, Any], expected: dict[str, Any], *, label: str
) -> None:
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"{label} identity changed: {mismatches}")


def load_ready_payload_manifest(
    root: str | Path,
    *,
    work_unit: int,
    expected_events: int,
    config_sha256: str,
    production_policy_sha256: str,
) -> dict[str, Any] | None:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return None
    value = read_json_object(path)
    if value.get("status") != "complete":
        return None
    _require_fields(
        value,
        {
            "format": PAYLOAD_WORK_UNIT_FORMAT,
            "status": "complete",
            "config_sha256": config_sha256,
            "production_policy_sha256": production_policy_sha256,
            "work_unit": work_unit,
            "expected_events": expected_events,
            "completed_events": expected_events,
            "failures": 0,
        },
        label=f"Payload work unit {work_unit}",
    )
    return value


def load_complete_final_manifest(
    root: str | Path,
    *,
    work_unit: int,
    expected_events: int,
    config_sha256: str,
    production_policy_sha256: str,
    require_payload_deleted: bool,
) -> dict[str, Any] | None:
    path = Path(root) / "manifest.json"
    if not path.is_file():
        return None
    value = read_json_object(path)
    if value.get("status") != "complete":
        return None
    expected = {
        "format": FINAL_WORK_UNIT_FORMAT,
        "status": "complete",
        "config_sha256": config_sha256,
        "production_policy_sha256": production_policy_sha256,
        "work_unit": work_unit,
        "events": expected_events,
        "unique_event_keys": expected_events,
    }
    _require_fields(value, expected, label=f"Final work unit {work_unit}")
    if require_payload_deleted and value.get("payload_deleted") is not True:
        return None
    return value


def validate_finalized_work_unit_for_payload_skip(
    *,
    payload_root: str | Path,
    final_root: str | Path,
    work_unit: int,
    expected_event_keys: Iterable[str],
    config_sha256: str,
    production_policy_sha256: str,
) -> bool:
    """Return true only when a deleted payload has a fully validated replacement."""

    expected_keys = set(expected_event_keys)
    final = load_complete_final_manifest(
        final_root,
        work_unit=work_unit,
        expected_events=len(expected_keys),
        config_sha256=config_sha256,
        production_policy_sha256=production_policy_sha256,
        require_payload_deleted=True,
    )
    if final is None:
        return False
    payload = load_ready_payload_manifest(
        payload_root,
        work_unit=work_unit,
        expected_events=len(expected_keys),
        config_sha256=config_sha256,
        production_policy_sha256=production_policy_sha256,
    )
    if payload is None:
        raise ValueError(
            f"Final work unit {work_unit} deleted payloads without retaining "
            "the committed payload manifest"
        )
    expected_source_hashes = {
        str(chunk["path"]): str(chunk["tensor_sha256"])
        for chunk in payload.get("chunks", [])
    }
    if final.get("source_payload_sha256") != expected_source_hashes:
        raise ValueError(
            f"Final work unit {work_unit} no longer identifies the deleted payload"
        )

    observed_keys: set[str] = set()
    final_path = Path(final_root)
    for shard in final.get("shards", []):
        path = final_path / str(shard["path"])
        if sha256_file(path) != shard.get("tensor_sha256"):
            raise ValueError(f"Final shard checksum changed before GPU resume: {path}")
        validation = validate_final_shard(
            path,
            expected_config_sha256=config_sha256,
            expected_policy_sha256=production_policy_sha256,
        )
        keys = set(validation["event_keys"])
        if observed_keys.intersection(keys):
            raise ValueError(f"Final work unit {work_unit} repeats event keys")
        observed_keys.update(keys)
    if observed_keys != expected_keys:
        raise ValueError(
            f"Final work unit {work_unit} cannot replace its deleted payload: "
            f"observed={len(observed_keys)} expected={len(expected_keys)}"
        )
    return True

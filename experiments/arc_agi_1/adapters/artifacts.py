from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor

from s3circuit.nvarc_circuit.events import CircuitEvent, read_catalog_chunk
from s3circuit.nvarc_circuit.finalize import validate_final_shard
from s3circuit.nvarc_circuit.storage import load_safetensors_bundle

from .config import ArtifactBankConfig
from .contracts import (
    CATALOG_FORMAT,
    FINAL_ROOT_FORMAT,
    FINAL_SHARD_FORMAT,
    VECTOR_DIM,
    VECTOR_SCHEMA,
    require_sha256,
    sha256_file,
)


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _contained_path(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Artifact path escapes its declared root: {relative_path}")
    return resolved


def _decode_ascii_hashes(value: Tensor, *, label: str) -> tuple[str, ...]:
    if value.ndim != 2 or value.shape[1] != 64 or value.dtype != torch.uint8:
        raise ValueError(f"{label} must be a uint8 [N,64] ASCII table")
    result = tuple(bytes(row.tolist()).decode("ascii") for row in value)
    if any(len(item) != 64 for item in result):
        raise ValueError(f"{label} contains an invalid identity")
    return result


@dataclass(frozen=True)
class VectorRecord:
    event_key: str
    artifact_id: str
    shard_relative_path: str
    row_in_shard: int
    vector: Tensor


class CircuitVectorBank:
    def __init__(
        self,
        records: Mapping[str, VectorRecord],
        *,
        root_manifest: Mapping[str, Any],
        final_root: Path,
    ) -> None:
        self._records = dict(records)
        self.root_manifest = dict(root_manifest)
        self.final_root = final_root

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, event_key: str) -> bool:
        return event_key in self._records

    def record(self, event_key: str) -> VectorRecord:
        try:
            return self._records[event_key]
        except KeyError as error:
            raise KeyError(f"Circuit vector is missing for event {event_key}") from error

    def vectors(self, event_keys: list[str] | tuple[str, ...]) -> Tensor:
        return torch.stack([self.record(key).vector for key in event_keys]).to(
            dtype=torch.float32
        )


def load_catalog_events(
    catalog_root: str | Path,
    *,
    expected_events: int,
) -> tuple[dict[str, CircuitEvent], dict[str, Any]]:
    root = Path(catalog_root)
    manifest_path = root / "manifest.json"
    manifest = _read_json_object(manifest_path)
    expected = {
        "format": CATALOG_FORMAT,
        "status": "complete",
        "events": expected_events,
        "unique_event_keys": expected_events,
    }
    mismatches = {
        key: {"expected": wanted, "actual": manifest.get(key)}
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"Circuit catalog identity changed: {mismatches}")
    result: dict[str, CircuitEvent] = {}
    for chunk in manifest.get("chunks", []):
        path = _contained_path(root, str(chunk["path"]))
        if sha256_file(path) != chunk.get("tensor_sha256"):
            raise ValueError(f"Circuit catalog checksum changed: {path}")
        events = read_catalog_chunk(
            path,
            expected_identity={
                "dataset_snapshot_sha256": manifest["dataset_snapshot_sha256"],
                "config_sha256": manifest["config_sha256"],
            },
        )
        for event in events:
            if event.event_key in result:
                raise ValueError(f"Duplicate catalog event: {event.event_key}")
            result[event.event_key] = event
    if len(result) != expected_events:
        raise ValueError(
            f"Circuit catalog coverage is {len(result)}, expected {expected_events}"
        )
    return result, manifest


def _iter_root_shards(
    manifest: Mapping[str, Any],
) -> Iterator[Mapping[str, Any]]:
    for work_unit in manifest.get("work_units", []):
        for shard in work_unit.get("shards", []):
            yield shard


def load_vector_bank(config: ArtifactBankConfig) -> CircuitVectorBank:
    final_root = Path(config.final_root)
    manifest_path = Path(config.root_manifest)
    pinned_sha = require_sha256(
        config.root_manifest_sha256,
        label="circuit root manifest",
    )
    if sha256_file(manifest_path) != pinned_sha:
        raise ValueError(f"Circuit root manifest checksum changed: {manifest_path}")
    manifest = _read_json_object(manifest_path)
    expected = {
        "format": FINAL_ROOT_FORMAT,
        "status": "complete",
        "events": config.expected_events,
        "unique_event_keys": config.expected_events,
        "unique_artifact_ids": config.expected_events,
    }
    mismatches = {
        key: {"expected": wanted, "actual": manifest.get(key)}
        for key, wanted in expected.items()
        if manifest.get(key) != wanted
    }
    if mismatches:
        raise ValueError(f"Circuit final root identity changed: {mismatches}")

    work_units = manifest.get("work_units")
    if not isinstance(work_units, list) or not work_units:
        raise ValueError("Circuit final root has no work units")
    unit_ids = [int(item["work_unit"]) for item in work_units]
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("Circuit final root repeats a work unit")
    if set(unit_ids) != set(range(len(unit_ids))):
        raise ValueError("Circuit final root work-unit IDs are not contiguous")
    if sum(int(item["events"]) for item in work_units) != config.expected_events:
        raise ValueError("Circuit final root work-unit coverage changed")
    for item in work_units:
        unit_root = _contained_path(
            final_root,
            f"work_unit_{int(item['work_unit']):04d}",
        )
        unit_manifest = unit_root / "manifest.json"
        if sha256_file(unit_manifest) != item.get("manifest_sha256"):
            raise ValueError(f"Circuit work-unit manifest checksum changed: {unit_manifest}")

    records: dict[str, VectorRecord] = {}
    artifact_ids: set[str] = set()
    for shard in _iter_root_shards(manifest):
        relative_path = str(shard["path"])
        path = _contained_path(final_root, relative_path)
        if sha256_file(path) != shard.get("tensor_sha256"):
            raise ValueError(f"Circuit final shard checksum changed: {path}")
        validation = validate_final_shard(
            path,
            expected_config_sha256=str(manifest["config_sha256"]),
            expected_policy_sha256=str(manifest["production_policy_sha256"]),
        )
        tensors, metadata = load_safetensors_bundle(
            path,
            expected_format=FINAL_SHARD_FORMAT,
        )
        if metadata.get("vector_schema") != VECTOR_SCHEMA:
            raise ValueError(f"Circuit vector schema changed: {path}")
        vectors = tensors["vectors"]
        if vectors.shape != (int(shard["events"]), VECTOR_DIM):
            raise ValueError(f"Circuit vector shard shape changed: {path}")
        event_keys = _decode_ascii_hashes(tensors["event_key_ascii"], label="event keys")
        local_artifacts = _decode_ascii_hashes(
            tensors["artifact_id_ascii"], label="artifact IDs"
        )
        if tuple(validation["event_keys"]) != event_keys:
            raise ValueError(f"Circuit validation order changed: {path}")
        for row_index, (event_key, artifact_id) in enumerate(
            zip(event_keys, local_artifacts, strict=True)
        ):
            if event_key in records or artifact_id in artifact_ids:
                raise ValueError("Circuit final root repeats an event or artifact")
            artifact_ids.add(artifact_id)
            records[event_key] = VectorRecord(
                event_key=event_key,
                artifact_id=artifact_id,
                shard_relative_path=relative_path,
                row_in_shard=row_index,
                vector=vectors[row_index].clone().to(dtype=torch.float32),
            )
    if len(records) != config.expected_events:
        raise ValueError(
            f"Circuit vector coverage is {len(records)}, expected {config.expected_events}"
        )
    return CircuitVectorBank(records, root_manifest=manifest, final_root=final_root)

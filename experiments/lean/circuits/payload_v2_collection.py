"""Canonical collection indices over worker-local S5 Payload v2 archives."""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .payload_v2 import (
    PAYLOAD_V2_FORMAT,
    PAYLOAD_V2_SCHEMA_VERSION,
    payload_artifact_id,
    semantic_sha256,
    sha256_file,
)


PAYLOAD_COLLECTION_FORMAT = "s5_payload_v2_collection_index_v1"
WORKER_SPECIFIC_TRACE_KEYS = {"worker_id", "prompt_keys", "event_keys"}
FORBIDDEN_CONDITION_KEYS = {
    "task",
    "task_type",
    "task_family",
    "seed",
    "sample_seed",
    "generator_seed",
    "gold",
    "gold_proof",
    "gold_tactic",
    "ground_truth",
    "answer",
    "response",
    "supervision",
    "correct",
    "correctness",
    "accuracy",
    "pass@1",
    "label",
    "labels",
}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _relative(parent: Path, value: Path) -> str:
    return Path(os.path.relpath(value.resolve(), parent.resolve())).as_posix()


def assert_condition_safe_archive_identity(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in FORBIDDEN_CONDITION_KEYS:
                raise ValueError(
                    f"Payload collection archive identity contains forbidden key: {key}"
                )
            assert_condition_safe_archive_identity(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_condition_safe_archive_identity(item)


def common_worker_archive_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only worker assignment fields from one Payload archive identity."""

    result = deepcopy(dict(identity))
    trace = result.get("trace_corpus_identity")
    if not isinstance(trace, dict):
        raise ValueError("Payload worker archive omits trace_corpus_identity")
    result["trace_corpus_identity"] = {
        key: value for key, value in trace.items() if key not in WORKER_SPECIFIC_TRACE_KEYS
    }
    result.pop("trace_corpus_identity_sha256", None)
    return result


def _worker_manifest_record(
    manifest_path: Path,
    *,
    expected_worker_count: int,
) -> tuple[int, dict[str, Any], list[str], list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != PAYLOAD_V2_FORMAT
        or int(manifest.get("schema_version", -1)) != PAYLOAD_V2_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"Payload worker manifest is incomplete: {manifest_path}")
    identity = manifest.get("archive_identity")
    if not isinstance(identity, dict):
        raise ValueError("Payload worker manifest omits archive identity")
    assert_condition_safe_archive_identity(identity)
    archive_sha = str(manifest.get("archive_identity_sha256", ""))
    if semantic_sha256(identity) != archive_sha:
        raise ValueError("Payload worker archive identity checksum changed")
    trace = identity.get("trace_corpus_identity")
    if not isinstance(trace, dict):
        raise ValueError("Payload worker trace identity is invalid")
    if semantic_sha256(trace) != identity.get("trace_corpus_identity_sha256"):
        raise ValueError("Payload worker trace identity checksum changed")
    worker_id = int(trace.get("worker_id", -1))
    worker_count = int(trace.get("worker_count", -1))
    if worker_count != int(expected_worker_count) or not 0 <= worker_id < worker_count:
        raise ValueError("Payload worker routing identity changed")
    event_keys = [str(value) for value in trace.get("event_keys", [])]
    prompt_keys = [str(value) for value in trace.get("prompt_keys", [])]
    total_events = int(manifest.get("total_events", -1))
    if total_events < 0 or len(event_keys) != total_events or len(prompt_keys) != total_events:
        raise ValueError("Payload worker event/prompt coverage changed")
    if manifest.get("event_keys_sha256") != semantic_sha256(event_keys):
        raise ValueError("Payload worker event ordering changed")
    artifact_ids = [
        payload_artifact_id(
            event_key=event_key,
            archive_identity_sha256=archive_sha,
        )
        for event_key in event_keys
    ]
    if manifest.get("payload_artifact_ids_sha256") != semantic_sha256(artifact_ids):
        raise ValueError("Payload worker artifact ordering changed")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or sum(int(row.get("records", -1)) for row in shards) != total_events:
        raise ValueError("Payload worker shard coverage changed")
    for row in shards:
        if row.get("archive_identity_sha256") != archive_sha:
            raise ValueError("Payload worker shard mixes archive identities")
    return worker_id, manifest, event_keys, artifact_ids


def build_payload_v2_collection_index(
    worker_manifest_paths: Sequence[str | Path],
    *,
    output_path: str | Path,
    expected_workers: int,
    expected_events: int,
) -> dict[str, Any]:
    """Index worker archives and restore canonical round-robin event ordering."""

    if int(expected_workers) <= 0 or int(expected_events) <= 0:
        raise ValueError("Payload collection expected coverage must be positive")
    paths = sorted({Path(value).resolve() for value in worker_manifest_paths})
    if len(paths) != int(expected_workers):
        raise ValueError(
            f"Payload collection expected {expected_workers} worker manifests, got {len(paths)}"
        )
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"Payload collection index already exists: {output}")
    worker_rows: dict[int, dict[str, Any]] = {}
    worker_events: dict[int, list[str]] = {}
    worker_artifacts: dict[int, list[str]] = {}
    common_identity: dict[str, Any] | None = None
    common_sha: str | None = None
    for path in paths:
        worker_id, manifest, event_keys, artifact_ids = _worker_manifest_record(
            path,
            expected_worker_count=expected_workers,
        )
        if worker_id in worker_rows:
            raise ValueError("Payload collection contains duplicate worker IDs")
        current_common = common_worker_archive_identity(manifest["archive_identity"])
        current_common_sha = semantic_sha256(current_common)
        if common_identity is None:
            common_identity = current_common
            common_sha = current_common_sha
        elif current_common != common_identity or current_common_sha != common_sha:
            raise ValueError("Payload collection mixes model/bank/policy/code identities")
        worker_events[worker_id] = event_keys
        worker_artifacts[worker_id] = artifact_ids
        worker_rows[worker_id] = {
            "worker_id": worker_id,
            "manifest_path": _relative(output.parent, path),
            "manifest_sha256": sha256_file(path),
            "archive_identity_sha256": manifest["archive_identity_sha256"],
            "events": int(manifest["total_events"]),
            "event_keys_sha256": manifest["event_keys_sha256"],
            "payload_artifact_ids_sha256": manifest[
                "payload_artifact_ids_sha256"
            ],
            "shards": [
                {
                    "records": int(row["records"]),
                    "bytes": int(row["bytes"]),
                    "tensor_sha256": str(row["tensor_sha256"]),
                    "sidecar_sha256": str(row["sidecar_sha256"]),
                }
                for row in manifest["shards"]
            ],
        }
    if set(worker_rows) != set(range(int(expected_workers))):
        raise ValueError("Payload collection worker IDs are not exactly canonical")
    if sum(len(value) for value in worker_events.values()) != int(expected_events):
        raise ValueError("Payload collection total event coverage changed")

    canonical_events: list[str | None] = [None] * int(expected_events)
    canonical_artifacts: list[str | None] = [None] * int(expected_events)
    for worker_id in range(int(expected_workers)):
        for local_row, (event_key, artifact_id_value) in enumerate(
            zip(
                worker_events[worker_id],
                worker_artifacts[worker_id],
                strict=True,
            )
        ):
            global_index = worker_id + local_row * int(expected_workers)
            if global_index >= int(expected_events):
                raise ValueError("Payload worker/local routing exceeds canonical coverage")
            if canonical_events[global_index] is not None:
                raise ValueError("Payload collection canonical index is duplicated")
            canonical_events[global_index] = event_key
            canonical_artifacts[global_index] = artifact_id_value
    if any(value is None for value in canonical_events) or any(
        value is None for value in canonical_artifacts
    ):
        raise ValueError("Payload collection canonical coverage has holes")
    event_keys = [str(value) for value in canonical_events]
    artifact_ids = [str(value) for value in canonical_artifacts]
    if len(set(event_keys)) != len(event_keys) or len(set(artifact_ids)) != len(
        artifact_ids
    ):
        raise ValueError("Payload collection contains duplicate event/artifact IDs")
    if common_identity is None or common_sha is None:
        raise AssertionError("Payload collection has no common identity")
    identity = {
        "format": PAYLOAD_COLLECTION_FORMAT,
        "payload_format": PAYLOAD_V2_FORMAT,
        "payload_schema_version": PAYLOAD_V2_SCHEMA_VERSION,
        "worker_count": int(expected_workers),
        "events": int(expected_events),
        "routing": "global_index=worker_id+local_row*worker_count",
        "common_worker_archive_identity_sha256": common_sha,
        "canonical_event_keys_sha256": semantic_sha256(event_keys),
        "canonical_payload_artifact_ids_sha256": semantic_sha256(artifact_ids),
        "worker_manifest_inventory_sha256": semantic_sha256(
            [
                {
                    key: value
                    for key, value in worker_rows[worker_id].items()
                    if key != "manifest_path"
                }
                for worker_id in range(int(expected_workers))
            ]
        ),
    }
    value = {
        **identity,
        "status": "complete",
        "index_identity_sha256": semantic_sha256(identity),
        "common_worker_archive_identity": common_identity,
        "workers": [worker_rows[index] for index in range(int(expected_workers))],
        "verification": {
            "worker_manifest_identity": "verified",
            "physical_payload_shards": "deferred_to_streaming_vectorization",
        },
    }
    _atomic_json(output, value)
    reopened = read_payload_v2_collection_index(output)
    if reopened != value:
        raise RuntimeError("Payload collection index changed during publication")
    return value


def read_payload_v2_collection_index(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("format") != PAYLOAD_COLLECTION_FORMAT or value.get(
        "status"
    ) != "complete":
        raise ValueError("Payload collection index is incomplete")
    workers = value.get("workers")
    worker_count = int(value.get("worker_count", -1))
    if not isinstance(workers, list) or len(workers) != worker_count:
        raise ValueError("Payload collection worker coverage changed")
    if [int(row.get("worker_id", -1)) for row in workers] != list(
        range(worker_count)
    ):
        raise ValueError("Payload collection worker order changed")
    recomputed_inventory_sha = semantic_sha256(
        [
            {key: item for key, item in row.items() if key != "manifest_path"}
            for row in workers
        ]
    )
    if recomputed_inventory_sha != value.get("worker_manifest_inventory_sha256"):
        raise ValueError("Payload collection worker inventory changed")
    identity_keys = (
        "format",
        "payload_format",
        "payload_schema_version",
        "worker_count",
        "events",
        "routing",
        "common_worker_archive_identity_sha256",
        "canonical_event_keys_sha256",
        "canonical_payload_artifact_ids_sha256",
        "worker_manifest_inventory_sha256",
    )
    identity = {key: value[key] for key in identity_keys}
    if semantic_sha256(identity) != value.get("index_identity_sha256"):
        raise ValueError("Payload collection index semantic identity changed")
    common = value.get("common_worker_archive_identity")
    if not isinstance(common, dict) or semantic_sha256(common) != value.get(
        "common_worker_archive_identity_sha256"
    ):
        raise ValueError("Payload collection common identity changed")
    assert_condition_safe_archive_identity(common)
    if sum(int(row.get("events", -1)) for row in workers) != int(value["events"]):
        raise ValueError("Payload collection event totals changed")
    return value


def resolve_worker_manifest(
    collection_index_path: str | Path,
    worker_record: Mapping[str, Any],
) -> Path:
    source = Path(collection_index_path).resolve()
    path = _resolve(source.parent, str(worker_record["manifest_path"]))
    if sha256_file(path) != worker_record.get("manifest_sha256"):
        raise ValueError("Payload collection worker-manifest checksum changed")
    return path

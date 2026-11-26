"""Durable worker and canonical archives for raw S5 397D descriptors."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .descriptor_v2_397 import DESCRIPTOR_DIM, SCHEMA_397, vectorize_payload_v2_397
from .payload_v2 import (
    PAYLOAD_V2_FORMAT,
    read_payload_v2_shard,
    semantic_sha256,
    sha256_file,
)
from .payload_v2_collection import (
    read_payload_v2_collection_index,
    resolve_worker_manifest,
)


WORKER_DESCRIPTOR_FORMAT = "s5_circuit_descriptor397_worker_v1"
DESCRIPTOR_ARCHIVE_FORMAT = "s5_circuit_descriptor397_archive_v1"
SHARD_OVERHEAD_RESERVE = 64 * 1024


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


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


def _quarantine_orphan(path: Path) -> Path:
    destination = path.with_name(
        f".{path.name}.orphan.{os.getpid()}.{time.time_ns()}"
    )
    os.replace(path, destination)
    return destination


def _hex_to_bytes(values: Sequence[str]) -> Tensor:
    if any(len(value) != 64 for value in values):
        raise ValueError("descriptor identity is not SHA256")
    array = np.frombuffer(b"".join(bytes.fromhex(value) for value in values), dtype=np.uint8)
    return torch.from_numpy(array.copy().reshape(len(values), 32))


def _bytes_to_hex(value: Tensor) -> list[str]:
    if value.dtype != torch.uint8 or value.ndim != 2 or value.shape[1] != 32:
        raise ValueError("descriptor identity tensor must be uint8[N,32]")
    return [bytes(row.tolist()).hex() for row in value.cpu()]


def _semantic_string_list_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        first = False
        digest.update(json.dumps(str(value), ensure_ascii=True).encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def _digest_add_string(
    digest: Any,
    value: str,
    *,
    first: bool,
) -> bool:
    if not first:
        digest.update(b",")
    digest.update(json.dumps(str(value), ensure_ascii=True).encode("utf-8"))
    return False


def _load_payload_worker_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != PAYLOAD_V2_FORMAT or value.get("status") != "complete":
        raise ValueError("Payload worker manifest is incomplete")
    identity = value.get("archive_identity")
    if not isinstance(identity, dict) or semantic_sha256(identity) != value.get(
        "archive_identity_sha256"
    ):
        raise ValueError("Payload worker archive identity changed")
    return value


def _iter_worker_payloads(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    full_verify: bool,
):
    archive_sha = str(manifest["archive_identity_sha256"])
    for raw_record in manifest["shards"]:
        record = dict(raw_record)
        record["path"] = str(_resolve(manifest_path.parent, str(record["path"])))
        record["sidecar"] = str(_resolve(manifest_path.parent, str(record["sidecar"])))
        if full_verify:
            sidecar = json.loads(
                Path(record["sidecar"]).read_text(encoding="utf-8")
            )
            if (
                sha256_file(record["sidecar"]) != record.get("sidecar_sha256")
                or sidecar.get("tensor_sha256") != record.get("tensor_sha256")
                or sidecar.get("archive_identity_sha256") != archive_sha
                or record.get("archive_identity_sha256") != archive_sha
            ):
                raise ValueError("Payload worker shard manifest/sidecar identity changed")
        payloads = read_payload_v2_shard(record["path"])
        if full_verify and (
            len(payloads) != int(record["records"])
            or semantic_sha256([payload.event_key for payload in payloads])
            != record.get("event_keys_sha256")
            or semantic_sha256(
                [payload.payload_artifact_id for payload in payloads]
            )
            != record.get("payload_artifact_ids_sha256")
        ):
            raise ValueError("Payload worker shard coverage changed")
        yield from payloads


def write_worker_descriptor397(
    *,
    collection_index_path: str | Path,
    worker_id: int,
    output_path: str | Path,
    full_payload_verify: bool = True,
) -> dict[str, Any]:
    """Vectorize one Payload worker archive exactly once in local-row order."""

    collection_path = Path(collection_index_path).resolve()
    collection = read_payload_v2_collection_index(collection_path)
    worker_count = int(collection["worker_count"])
    if not 0 <= int(worker_id) < worker_count:
        raise ValueError("descriptor worker_id is outside collection coverage")
    worker_record = collection["workers"][int(worker_id)]
    manifest_path = resolve_worker_manifest(collection_path, worker_record)
    manifest = _load_payload_worker_manifest(manifest_path)
    if manifest["archive_identity_sha256"] != worker_record["archive_identity_sha256"]:
        raise ValueError("descriptor worker source archive identity changed")
    identity = manifest["archive_identity"]
    n_layers = int(identity["n_layers"])
    d_feature = int(identity["d_feature"])
    archive_sha = str(manifest["archive_identity_sha256"])
    destination = Path(output_path).resolve()
    sidecar = _sidecar(destination)
    if destination.exists() != sidecar.exists():
        _quarantine_orphan(destination if destination.exists() else sidecar)
    if destination.is_file() and sidecar.is_file():
        _existing_tensors, existing = read_worker_descriptor397(destination)
        if (
            int(existing["source_worker_id"]) != int(worker_id)
            or existing["payload_collection_identity_sha256"]
            != collection["index_identity_sha256"]
            or existing["source_worker_manifest_sha256"]
            != worker_record["manifest_sha256"]
            or existing["source_payload_archive_identity_sha256"] != archive_sha
        ):
            raise ValueError("existing descriptor worker artifact is stale")
        if bool(full_payload_verify) and not bool(existing["full_payload_verify"]):
            raise ValueError(
                "formal descriptor worker resume cannot reuse a trusted-fast artifact"
            )
        return {
            "path": str(destination),
            "sidecar": str(sidecar),
            "worker_id": int(worker_id),
            "records": int(existing["records"]),
            "artifact_id": existing["artifact_id"],
            "tensor_sha256": existing["tensor_sha256"],
            "sidecar_sha256": sha256_file(sidecar),
            "bytes": destination.stat().st_size + sidecar.stat().st_size,
            "seconds": float(existing["seconds"]),
            "resumed": True,
        }
    if destination.exists() or sidecar.exists():
        raise FileExistsError(f"descriptor worker artifact already exists: {destination}")
    started = time.perf_counter()
    vectors: list[Tensor] = []
    event_keys: list[str] = []
    artifact_ids: list[str] = []
    for payload in _iter_worker_payloads(
        manifest_path,
        manifest,
        full_verify=full_payload_verify,
    ):
        vectors.append(
            vectorize_payload_v2_397(
                payload,
                n_layers=n_layers,
                d_feature=d_feature,
                archive_identity_sha256=archive_sha,
            )
        )
        event_keys.append(payload.event_key)
        artifact_ids.append(payload.payload_artifact_id)
    expected_records = int(worker_record["events"])
    if len(vectors) != expected_records:
        raise ValueError("descriptor worker Payload coverage changed")
    if semantic_sha256(event_keys) != worker_record["event_keys_sha256"] or semantic_sha256(
        artifact_ids
    ) != worker_record["payload_artifact_ids_sha256"]:
        raise ValueError("descriptor worker Payload ordering changed")
    global_index = torch.tensor(
        [int(worker_id) + row * worker_count for row in range(expected_records)],
        dtype=torch.int64,
    )
    tensors = {
        "global_index": global_index,
        "event_key": _hex_to_bytes(event_keys),
        "payload_artifact_id": _hex_to_bytes(artifact_ids),
        "descriptor_raw": (
            torch.stack(vectors).contiguous()
            if vectors
            else torch.empty((0, DESCRIPTOR_DIM), dtype=torch.float32)
        ),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary_sidecar = _sidecar(temporary)
    save_file(tensors, str(temporary))
    tensor_sha = sha256_file(temporary)
    identity_value = {
        "format": WORKER_DESCRIPTOR_FORMAT,
        "descriptor_schema_sha256": SCHEMA_397.sha256,
        "payload_collection_identity_sha256": collection["index_identity_sha256"],
        "source_worker_id": int(worker_id),
        "source_worker_manifest_sha256": worker_record["manifest_sha256"],
        "source_payload_archive_identity_sha256": archive_sha,
        "records": expected_records,
        "event_keys_sha256": semantic_sha256(event_keys),
        "payload_artifact_ids_sha256": semantic_sha256(artifact_ids),
        "global_indices_sha256": semantic_sha256(global_index.tolist()),
        "full_payload_verify": bool(full_payload_verify),
        "tensor_sha256": tensor_sha,
    }
    metadata = {
        **identity_value,
        "artifact_id": semantic_sha256(identity_value),
        "descriptor_schema": SCHEMA_397.identity(),
        "seconds": time.perf_counter() - started,
    }
    temporary_sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    loaded = load_file(str(temporary), device="cpu")
    if set(loaded) != set(tensors) or any(
        not torch.equal(loaded[name], value) for name, value in tensors.items()
    ):
        raise RuntimeError("descriptor worker temporary reopen parity failed")
    os.replace(temporary, destination)
    os.replace(temporary_sidecar, sidecar)
    read_worker_descriptor397(destination)
    return {
        "path": str(destination),
        "sidecar": str(sidecar),
        "worker_id": int(worker_id),
        "records": expected_records,
        "artifact_id": metadata["artifact_id"],
        "tensor_sha256": tensor_sha,
        "sidecar_sha256": sha256_file(sidecar),
        "bytes": destination.stat().st_size + sidecar.stat().st_size,
        "seconds": metadata["seconds"],
    }


def read_worker_descriptor397(
    path: str | Path,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    source = Path(path).resolve()
    sidecar = _sidecar(source)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format") != WORKER_DESCRIPTOR_FORMAT:
        raise ValueError("descriptor worker format changed")
    if metadata.get("descriptor_schema") != SCHEMA_397.identity() or metadata.get(
        "descriptor_schema_sha256"
    ) != SCHEMA_397.sha256:
        raise ValueError("descriptor worker schema identity changed")
    if sha256_file(source) != metadata.get("tensor_sha256"):
        raise ValueError("descriptor worker tensor checksum changed")
    tensors = load_file(str(source), device="cpu")
    required = {"global_index", "event_key", "payload_artifact_id", "descriptor_raw"}
    if set(tensors) != required:
        raise ValueError("descriptor worker tensor schema changed")
    count = int(metadata["records"])
    if (
        tensors["global_index"].dtype != torch.int64
        or tensors["global_index"].shape != (count,)
        or tensors["event_key"].dtype != torch.uint8
        or tensors["event_key"].shape != (count, 32)
        or tensors["payload_artifact_id"].dtype != torch.uint8
        or tensors["payload_artifact_id"].shape != (count, 32)
        or tensors["descriptor_raw"].dtype != torch.float32
        or tensors["descriptor_raw"].shape != (count, DESCRIPTOR_DIM)
        or not bool(torch.isfinite(tensors["descriptor_raw"]).all())
    ):
        raise ValueError("descriptor worker tensors are invalid")
    event_keys = _bytes_to_hex(tensors["event_key"])
    artifact_ids = _bytes_to_hex(tensors["payload_artifact_id"])
    if semantic_sha256(event_keys) != metadata.get("event_keys_sha256") or semantic_sha256(
        artifact_ids
    ) != metadata.get("payload_artifact_ids_sha256"):
        raise ValueError("descriptor worker identity ordering changed")
    if semantic_sha256(tensors["global_index"].tolist()) != metadata.get(
        "global_indices_sha256"
    ):
        raise ValueError("descriptor worker global-index ordering changed")
    identity_keys = (
        "format",
        "descriptor_schema_sha256",
        "payload_collection_identity_sha256",
        "source_worker_id",
        "source_worker_manifest_sha256",
        "source_payload_archive_identity_sha256",
        "records",
        "event_keys_sha256",
        "payload_artifact_ids_sha256",
        "global_indices_sha256",
        "full_payload_verify",
        "tensor_sha256",
    )
    identity = {key: metadata[key] for key in identity_keys}
    if semantic_sha256(identity) != metadata.get("artifact_id"):
        raise ValueError("descriptor worker semantic identity changed")
    return tensors, metadata


def _write_descriptor_shard(
    destination: Path,
    *,
    start: int,
    raw: np.ndarray,
    event_key: np.ndarray,
    payload_artifact_id_value: np.ndarray,
    source_identity_sha256: str,
) -> dict[str, Any]:
    count = int(raw.shape[0])
    tensors = {
        "global_index": torch.arange(start, start + count, dtype=torch.int64),
        "event_key": torch.from_numpy(np.array(event_key, copy=True)),
        "payload_artifact_id": torch.from_numpy(
            np.array(payload_artifact_id_value, copy=True)
        ),
        "descriptor_raw": torch.from_numpy(np.array(raw, copy=True)),
    }
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary_sidecar = _sidecar(temporary)
    sidecar = _sidecar(destination)
    save_file(tensors, str(temporary))
    tensor_sha = sha256_file(temporary)
    event_keys = _bytes_to_hex(tensors["event_key"])
    artifact_ids = _bytes_to_hex(tensors["payload_artifact_id"])
    metadata = {
        "format": DESCRIPTOR_ARCHIVE_FORMAT,
        "descriptor_schema_sha256": SCHEMA_397.sha256,
        "source_identity_sha256": source_identity_sha256,
        "global_start": int(start),
        "global_end": int(start + count),
        "records": count,
        "event_keys_sha256": semantic_sha256(event_keys),
        "payload_artifact_ids_sha256": semantic_sha256(artifact_ids),
        "tensor_sha256": tensor_sha,
    }
    temporary_sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    reopened = load_file(str(temporary), device="cpu")
    if any(not torch.equal(reopened[name], value) for name, value in tensors.items()):
        raise RuntimeError("descriptor shard temporary reopen parity failed")
    os.replace(temporary, destination)
    os.replace(temporary_sidecar, sidecar)
    return {
        "path": destination.name,
        "sidecar": sidecar.name,
        "records": count,
        "bytes": destination.stat().st_size + sidecar.stat().st_size,
        "global_start": int(start),
        "global_end": int(start + count),
        "tensor_sha256": tensor_sha,
        "sidecar_sha256": sha256_file(sidecar),
        "event_keys_sha256": metadata["event_keys_sha256"],
        "payload_artifact_ids_sha256": metadata[
            "payload_artifact_ids_sha256"
        ],
    }


def _shard_identity_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "records": int(record["records"]),
        "bytes": int(record["bytes"]),
        "global_start": int(record["global_start"]),
        "global_end": int(record["global_end"]),
        "tensor_sha256": str(record["tensor_sha256"]),
        "sidecar_sha256": str(record["sidecar_sha256"]),
        "event_keys_sha256": str(record["event_keys_sha256"]),
        "payload_artifact_ids_sha256": str(
            record["payload_artifact_ids_sha256"]
        ),
    }


def _verify_descriptor_shard(
    record: Mapping[str, Any],
    *,
    parent: Path,
    source_identity_sha256: str,
    expected_raw: np.ndarray | None = None,
    expected_event_key: np.ndarray | None = None,
    expected_payload_id: np.ndarray | None = None,
) -> dict[str, Tensor]:
    tensor_path = _resolve(parent, str(record["path"]))
    sidecar_path = _resolve(parent, str(record["sidecar"]))
    if (
        sha256_file(tensor_path) != record.get("tensor_sha256")
        or sha256_file(sidecar_path) != record.get("sidecar_sha256")
    ):
        raise ValueError("descriptor397 shard checksum changed")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        sidecar.get("source_identity_sha256") != source_identity_sha256
        or sidecar.get("tensor_sha256") != record.get("tensor_sha256")
        or int(sidecar.get("global_start", -1))
        != int(record.get("global_start", -2))
        or int(sidecar.get("global_end", -1))
        != int(record.get("global_end", -2))
        or sidecar.get("event_keys_sha256")
        != record.get("event_keys_sha256")
        or sidecar.get("payload_artifact_ids_sha256")
        != record.get("payload_artifact_ids_sha256")
    ):
        raise ValueError("descriptor397 shard identity changed")
    tensors = load_file(str(tensor_path), device="cpu")
    count = int(record["records"])
    start = int(record["global_start"])
    if (
        set(tensors)
        != {
            "global_index",
            "event_key",
            "payload_artifact_id",
            "descriptor_raw",
        }
        or tensors["global_index"].tolist() != list(range(start, start + count))
        or tensors["event_key"].dtype != torch.uint8
        or tensors["event_key"].shape != (count, 32)
        or tensors["payload_artifact_id"].dtype != torch.uint8
        or tensors["payload_artifact_id"].shape != (count, 32)
        or tensors["descriptor_raw"].dtype != torch.float32
        or tensors["descriptor_raw"].shape != (count, DESCRIPTOR_DIM)
        or not bool(torch.isfinite(tensors["descriptor_raw"]).all())
        or semantic_sha256(_bytes_to_hex(tensors["event_key"]))
        != record["event_keys_sha256"]
        or semantic_sha256(_bytes_to_hex(tensors["payload_artifact_id"]))
        != record["payload_artifact_ids_sha256"]
    ):
        raise ValueError("descriptor397 shard tensor coverage changed")
    if expected_raw is not None and not np.array_equal(
        tensors["descriptor_raw"].numpy(), expected_raw
    ):
        raise ValueError("resumed descriptor397 shard raw values changed")
    if expected_event_key is not None and not np.array_equal(
        tensors["event_key"].numpy(), expected_event_key
    ):
        raise ValueError("resumed descriptor397 shard event identities changed")
    if expected_payload_id is not None and not np.array_equal(
        tensors["payload_artifact_id"].numpy(), expected_payload_id
    ):
        raise ValueError("resumed descriptor397 shard Payload identities changed")
    return tensors


def _existing_descriptor_shard_record(destination: Path) -> dict[str, Any]:
    sidecar_path = _sidecar(destination)
    metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    return {
        "path": destination.name,
        "sidecar": sidecar_path.name,
        "records": int(metadata["records"]),
        "bytes": destination.stat().st_size + sidecar_path.stat().st_size,
        "global_start": int(metadata["global_start"]),
        "global_end": int(metadata["global_end"]),
        "tensor_sha256": str(metadata["tensor_sha256"]),
        "sidecar_sha256": sha256_file(sidecar_path),
        "event_keys_sha256": str(metadata["event_keys_sha256"]),
        "payload_artifact_ids_sha256": str(
            metadata["payload_artifact_ids_sha256"]
        ),
    }


def finalize_descriptor397_archive(
    worker_artifact_paths: Sequence[str | Path],
    *,
    collection_index_path: str | Path,
    output_dir: str | Path,
    max_shard_bytes: int = 512 * 1024**2,
) -> dict[str, Any]:
    """Canonicalize verified worker vectors into byte-bounded raw shards."""

    collection_path = Path(collection_index_path).resolve()
    collection = read_payload_v2_collection_index(collection_path)
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    running_manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("status") == "complete":
            verified = read_descriptor397_archive(
                manifest_path,
                full_tensor_verify=True,
            )
            if verified["payload_collection_identity_sha256"] != collection[
                "index_identity_sha256"
            ]:
                raise ValueError("existing descriptor archive uses a stale collection")
            return {
                "path": str(output),
                "manifest": str(manifest_path),
                "archive_identity_sha256": verified[
                    "archive_identity_sha256"
                ],
                "events": int(verified["events"]),
                "shards": len(verified["shards"]),
                "bytes": sum(int(row["bytes"]) for row in verified["shards"])
                + manifest_path.stat().st_size,
                "resumed": True,
            }
        if existing.get("status") != "running":
            raise ValueError("existing descriptor archive manifest is neither running nor complete")
        running_manifest = existing
    elif output.exists() and any(
        not path.name.startswith(".") for path in output.iterdir()
    ):
        raise FileExistsError(
            f"descriptor archive has files but no resumable manifest: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    worker_paths = sorted({Path(value).resolve() for value in worker_artifact_paths})
    worker_count = int(collection["worker_count"])
    event_count = int(collection["events"])
    if len(worker_paths) != worker_count:
        raise ValueError("descriptor worker artifact coverage changed")
    bytes_per_event = DESCRIPTOR_DIM * 4 + 32 + 32 + 8
    if int(max_shard_bytes) < bytes_per_event + SHARD_OVERHEAD_RESERVE:
        raise ValueError("descriptor shard byte cap is smaller than one event")
    events_per_shard = max(
        1,
        (int(max_shard_bytes) - SHARD_OVERHEAD_RESERVE) // bytes_per_event,
    )

    raw_path = output / f".descriptor_raw.tmp.{os.getpid()}.mmap"
    event_path = output / f".event_key.tmp.{os.getpid()}.mmap"
    artifact_path = output / f".payload_artifact_id.tmp.{os.getpid()}.mmap"
    coverage_path = output / f".coverage.tmp.{os.getpid()}.mmap"
    raw = np.memmap(
        raw_path,
        mode="w+",
        dtype=np.float32,
        shape=(event_count, DESCRIPTOR_DIM),
    )
    event_key = np.memmap(
        event_path,
        mode="w+",
        dtype=np.uint8,
        shape=(event_count, 32),
    )
    payload_id = np.memmap(
        artifact_path, mode="w+", dtype=np.uint8, shape=(event_count, 32)
    )
    coverage = np.memmap(coverage_path, mode="w+", dtype=np.uint8, shape=(event_count,))
    coverage[:] = 0
    memmaps = (raw, event_key, payload_id, coverage)
    try:
        seen_workers: set[int] = set()
        worker_inventory = []
        for path in worker_paths:
            tensors, metadata = read_worker_descriptor397(path)
            worker_id = int(metadata["source_worker_id"])
            if worker_id in seen_workers or not 0 <= worker_id < worker_count:
                raise ValueError("descriptor archive worker identity is duplicated")
            seen_workers.add(worker_id)
            if metadata["payload_collection_identity_sha256"] != collection[
                "index_identity_sha256"
            ]:
                raise ValueError("descriptor worker uses a different Payload collection")
            if not bool(metadata.get("full_payload_verify")):
                raise ValueError(
                    "formal descriptor finalize requires fully verified worker artifacts"
                )
            expected_worker = collection["workers"][worker_id]
            if (
                metadata["source_worker_manifest_sha256"]
                != expected_worker["manifest_sha256"]
                or metadata["source_payload_archive_identity_sha256"]
                != expected_worker["archive_identity_sha256"]
            ):
                raise ValueError("descriptor worker source identity changed")
            indices = tensors["global_index"].numpy().astype(np.int64, copy=False)
            if indices.size and (indices.min() < 0 or indices.max() >= event_count):
                raise ValueError("descriptor worker global index is out of range")
            if np.any(coverage[indices] != 0):
                raise ValueError("descriptor archive canonical coverage is duplicated")
            raw[indices] = tensors["descriptor_raw"].numpy()
            event_key[indices] = tensors["event_key"].numpy()
            payload_id[indices] = tensors["payload_artifact_id"].numpy()
            coverage[indices] = 1
            worker_inventory.append(
                {
                    "worker_id": worker_id,
                    "artifact_id": metadata["artifact_id"],
                    "tensor_sha256": metadata["tensor_sha256"],
                    "sidecar_sha256": sha256_file(_sidecar(path)),
                    "records": int(metadata["records"]),
                    "event_keys_sha256": metadata["event_keys_sha256"],
                    "payload_artifact_ids_sha256": metadata[
                        "payload_artifact_ids_sha256"
                    ],
                    "global_indices_sha256": metadata[
                        "global_indices_sha256"
                    ],
                }
            )
        if seen_workers != set(range(worker_count)) or not np.all(coverage == 1):
            raise ValueError("descriptor archive canonical coverage is incomplete")
        raw.flush()
        event_key.flush()
        payload_id.flush()
        event_digest = _semantic_string_list_sha256(
            bytes(event_key[index].tolist()).hex() for index in range(event_count)
        )
        artifact_digest = _semantic_string_list_sha256(
            bytes(payload_id[index].tolist()).hex() for index in range(event_count)
        )
        if (
            event_digest != collection["canonical_event_keys_sha256"]
            or artifact_digest
            != collection["canonical_payload_artifact_ids_sha256"]
        ):
            raise ValueError("descriptor archive canonical identity join changed")
        worker_inventory = sorted(
            worker_inventory, key=lambda row: int(row["worker_id"])
        )
        source_identity = {
            "format": DESCRIPTOR_ARCHIVE_FORMAT,
            "descriptor_schema_sha256": SCHEMA_397.sha256,
            "payload_collection_identity_sha256": collection[
                "index_identity_sha256"
            ],
            "events": event_count,
            "event_keys_sha256": event_digest,
            "payload_artifact_ids_sha256": artifact_digest,
            "worker_artifact_inventory_sha256": semantic_sha256(worker_inventory),
        }
        source_sha = semantic_sha256(source_identity)
        manifest_base = {
            **source_identity,
            "source_identity_sha256": source_sha,
            "descriptor_schema": SCHEMA_397.identity(),
            "max_shard_bytes": int(max_shard_bytes),
            "worker_artifacts": worker_inventory,
        }
        if running_manifest is not None:
            for key, expected in manifest_base.items():
                if running_manifest.get(key) != expected:
                    raise ValueError(
                        f"running descriptor archive identity changed at {key}"
                    )
            shards = [dict(row) for row in running_manifest.get("shards", [])]
        else:
            shards = []
            _atomic_json(
                manifest_path,
                {**manifest_base, "status": "running", "shards": []},
            )
        planned = list(range(0, event_count, events_per_shard))
        if len(shards) > len(planned):
            raise ValueError("running descriptor archive has excess shards")
        for shard_index, start in enumerate(planned):
            end = min(event_count, start + events_per_shard)
            if shard_index < len(shards):
                record = shards[shard_index]
                if (
                    int(record.get("global_start", -1)) != start
                    or int(record.get("global_end", -1)) != end
                ):
                    raise ValueError("running descriptor shard boundaries changed")
                _verify_descriptor_shard(
                    record,
                    parent=output,
                    source_identity_sha256=source_sha,
                    expected_raw=raw[start:end],
                    expected_event_key=event_key[start:end],
                    expected_payload_id=payload_id[start:end],
                )
                continue
            destination = output / f"descriptor_raw_{shard_index:05d}.safetensors"
            destination_sidecar = _sidecar(destination)
            if destination.exists() != destination_sidecar.exists():
                _quarantine_orphan(
                    destination if destination.exists() else destination_sidecar
                )
            if destination.is_file() and destination_sidecar.is_file():
                record = _existing_descriptor_shard_record(destination)
                if (
                    int(record["global_start"]) != start
                    or int(record["global_end"]) != end
                ):
                    raise ValueError("orphan descriptor shard boundaries changed")
                _verify_descriptor_shard(
                    record,
                    parent=output,
                    source_identity_sha256=source_sha,
                    expected_raw=raw[start:end],
                    expected_event_key=event_key[start:end],
                    expected_payload_id=payload_id[start:end],
                )
                shards.append(record)
                _atomic_json(
                    manifest_path,
                    {**manifest_base, "status": "running", "shards": shards},
                )
                continue
            record = _write_descriptor_shard(
                destination,
                start=start,
                raw=raw[start:end],
                event_key=event_key[start:end],
                payload_artifact_id_value=payload_id[start:end],
                source_identity_sha256=source_sha,
            )
            if int(record["bytes"]) > int(max_shard_bytes):
                raise RuntimeError(
                    "descriptor shard exceeded the declared byte cap after serialization"
                )
            shards.append(record)
            _atomic_json(
                manifest_path,
                {**manifest_base, "status": "running", "shards": shards},
            )
        publication_identity = {
            **source_identity,
            "worker_artifacts": worker_inventory,
            "shards": [_shard_identity_record(row) for row in shards],
        }
        archive_sha = semantic_sha256(publication_identity)
        complete = {
            **manifest_base,
            "status": "complete",
            "archive_identity_sha256": archive_sha,
            "shards": shards,
        }
        candidate = output / f".manifest.complete.{os.getpid()}.{time.time_ns()}.json"
        _atomic_json(candidate, complete)
        read_descriptor397_archive(candidate, full_tensor_verify=True)
        for attempt in range(20):
            try:
                os.replace(candidate, manifest_path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
        for value in memmaps:
            value._mmap.close()
        for path in (raw_path, event_path, artifact_path, coverage_path):
            path.unlink(missing_ok=True)
        return {
            "path": str(output),
            "manifest": str(manifest_path),
            "archive_identity_sha256": archive_sha,
            "events": event_count,
            "shards": len(shards),
            "bytes": sum(int(row["bytes"]) for row in shards)
            + manifest_path.stat().st_size,
        }
    except Exception:
        for value in memmaps:
            value._mmap.close()
        raise


def read_descriptor397_archive(
    path: str | Path,
    *,
    full_tensor_verify: bool = False,
) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if value.get("format") != DESCRIPTOR_ARCHIVE_FORMAT or value.get(
        "status"
    ) != "complete":
        raise ValueError("descriptor397 archive is incomplete")
    if value.get("descriptor_schema") != SCHEMA_397.identity() or value.get(
        "descriptor_schema_sha256"
    ) != SCHEMA_397.sha256:
        raise ValueError("descriptor397 archive schema changed")
    worker_inventory = value.get("worker_artifacts")
    if not isinstance(worker_inventory, list) or semantic_sha256(
        worker_inventory
    ) != value.get("worker_artifact_inventory_sha256"):
        raise ValueError("descriptor397 worker artifact inventory changed")
    source_identity_keys = (
        "format",
        "descriptor_schema_sha256",
        "payload_collection_identity_sha256",
        "events",
        "event_keys_sha256",
        "payload_artifact_ids_sha256",
        "worker_artifact_inventory_sha256",
    )
    source_identity = {key: value[key] for key in source_identity_keys}
    source_sha = semantic_sha256(source_identity)
    if source_sha != value.get("source_identity_sha256"):
        raise ValueError("descriptor397 source identity changed")
    shards = value.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("descriptor397 archive has no shards")
    cursor = 0
    event_digest = hashlib.sha256()
    artifact_digest = hashlib.sha256()
    event_first = True
    artifact_first = True
    if full_tensor_verify:
        event_digest.update(b"[")
        artifact_digest.update(b"[")
    for record in shards:
        if int(record.get("global_start", -1)) != cursor:
            raise ValueError("descriptor397 shard coverage is not contiguous")
        tensor_path = _resolve(source.parent, str(record["path"]))
        sidecar_path = _resolve(source.parent, str(record["sidecar"]))
        if (
            sha256_file(tensor_path) != record.get("tensor_sha256")
            or sha256_file(sidecar_path) != record.get("sidecar_sha256")
        ):
            raise ValueError("descriptor397 shard checksum changed")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            sidecar.get("source_identity_sha256") != source_sha
            or sidecar.get("tensor_sha256") != record.get("tensor_sha256")
            or int(sidecar.get("global_start", -1))
            != int(record.get("global_start", -2))
            or int(sidecar.get("global_end", -1))
            != int(record.get("global_end", -2))
            or sidecar.get("event_keys_sha256")
            != record.get("event_keys_sha256")
            or sidecar.get("payload_artifact_ids_sha256")
            != record.get("payload_artifact_ids_sha256")
        ):
            raise ValueError("descriptor397 shard identity changed")
        if full_tensor_verify:
            tensors = load_file(str(tensor_path), device="cpu")
            count = int(record["records"])
            if (
                set(tensors)
                != {
                    "global_index",
                    "event_key",
                    "payload_artifact_id",
                    "descriptor_raw",
                }
                or tensors["event_key"].shape != (count, 32)
                or tensors["payload_artifact_id"].shape != (count, 32)
                or tensors["global_index"].tolist()
                != list(range(cursor, cursor + count))
                or tensors["descriptor_raw"].dtype != torch.float32
                or tensors["descriptor_raw"].shape != (count, DESCRIPTOR_DIM)
                or not bool(torch.isfinite(tensors["descriptor_raw"]).all())
                or semantic_sha256(_bytes_to_hex(tensors["event_key"]))
                != record["event_keys_sha256"]
                or semantic_sha256(_bytes_to_hex(tensors["payload_artifact_id"]))
                != record["payload_artifact_ids_sha256"]
            ):
                raise ValueError("descriptor397 shard tensor coverage changed")
            for event_key_value in _bytes_to_hex(tensors["event_key"]):
                event_first = _digest_add_string(
                    event_digest,
                    event_key_value,
                    first=event_first,
                )
            for artifact_id_value in _bytes_to_hex(tensors["payload_artifact_id"]):
                artifact_first = _digest_add_string(
                    artifact_digest,
                    artifact_id_value,
                    first=artifact_first,
                )
        cursor = int(record["global_end"])
    if cursor != int(value["events"]):
        raise ValueError("descriptor397 archive total coverage changed")
    if full_tensor_verify:
        event_digest.update(b"]")
        artifact_digest.update(b"]")
        if (
            event_digest.hexdigest() != value["event_keys_sha256"]
            or artifact_digest.hexdigest()
            != value["payload_artifact_ids_sha256"]
        ):
            raise ValueError("descriptor397 archive canonical identity digest changed")
    publication_identity = {
        **source_identity,
        "worker_artifacts": worker_inventory,
        "shards": [_shard_identity_record(row) for row in shards],
    }
    if semantic_sha256(publication_identity) != value.get(
        "archive_identity_sha256"
    ):
        raise ValueError("descriptor397 archive semantic identity changed")
    return value

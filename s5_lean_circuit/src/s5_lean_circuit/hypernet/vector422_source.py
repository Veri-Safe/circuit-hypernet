"""Immutable 25D sidecars joined to existing, unchanged fixed-log1p397 caches.

Only retained Payload-v2 graphs can supply the prefix.  Completion requires
full event/payload coverage; NaNs in an unfinished mmap are never a condition.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ..circuit.descriptor_422 import DESCRIPTOR422_SCHEMA_SHA256
from ..circuit.descriptor_legacy25 import (
    LEGACY25_SCHEMA_SHA256, LEGACY25_LOG1P_INDICES,
    LEGACY25_SIGNED_LOG1P_INDICES, compute_legacy25, fixed_log1p_legacy25,
)
from ..circuit.descriptor_v2_397 import (
    DESCRIPTOR_397_SCHEMA_SHA256, fixed_log1p_397, vectorize_payload_v2_397,
)
from ..circuit.payload_v2 import (
    PAYLOAD_V2_FORMAT, semantic_sha256, sha256_file,
    verify_payload_v2_shard_record,
)
from ..circuit.payload_v2_collection import (
    PAYLOAD_COLLECTION_FORMAT, read_payload_v2_collection_index, resolve_worker_manifest,
    assert_condition_safe_archive_identity,
)


SIDECAR_FORMAT = "s5_vector422_condition_sidecar_v1"
TRANSFORM = {
    "format": "fixed_log1p25_plus_unchanged_fixed_log1p397_v1",
    "nonnegative_log1p_indices": list(LEGACY25_LOG1P_INDICES),
    "signed_log1p_indices": list(LEGACY25_SIGNED_LOG1P_INDICES),
    "path_sentinels": "use_payload_diagnostics_to_preserve_invalid_minus_one",
    "tail": "exact_original_cache_vectors_no_second_transform",
}
_SOURCE_FORMATS = {
    "s5_composition_loco_vector397_cache_v1", "s5_composition_loco_eval_bank_v1",
    "s5_hypernet_formal_packed_cache_v1",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _array_sha(value: Tensor | np.ndarray) -> str:
    array = value.detach().cpu().numpy() if isinstance(value, Tensor) else value
    digest = hashlib.sha256()
    for start in range(0, len(array), 8192):
        digest.update(np.ascontiguousarray(array[start:start + 8192]).tobytes())
    return digest.hexdigest()


def _identity_array(value: Tensor | Sequence[str], count: int) -> np.ndarray:
    if isinstance(value, Tensor):
        if value.dtype != torch.uint8 or tuple(value.shape) != (count, 32):
            raise ValueError("event/payload identities must be uint8[N,32]")
        return value.detach().cpu().numpy()
    try:
        array = np.asarray([list(bytes.fromhex(v)) for v in value], dtype=np.uint8)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid event/payload identities") from exc
    if array.shape != (count, 32):
        raise ValueError("event/payload identity coverage changed")
    return array


def _load_source(path: Path) -> tuple[Mapping[str, Tensor], dict[str, Any], dict[str, Any]]:
    sidecar = path.with_suffix(path.suffix + ".json")
    metadata = _read_json(sidecar)
    if metadata.get("status") != "complete" or metadata.get("format") not in _SOURCE_FORMATS:
        raise ValueError("422 source must be a complete existing 397 cache/bank")
    if metadata.get("corpus_name") == "final_test":
        raise ValueError("422 comparison preparation excludes the final-test bank")
    measured_sha = sha256_file(path)
    if metadata.get("payload_sha256") != measured_sha:
        raise ValueError("422 source cache payload checksum changed")
    treatment = metadata.get("input_treatment", {})
    if not isinstance(treatment, Mapping) or treatment.get("format") != "s5_hypernet_vector397_fixed_log1p_v1":
        raise ValueError("422 source must contain the original fixed-log1p397 treatment")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    vectors = payload.get("vectors")
    count = int(metadata.get("event_count", metadata.get("prefix", -1)))
    if not isinstance(vectors, Tensor) or vectors.dtype != torch.float32 or tuple(vectors.shape) != (count, 397):
        raise ValueError("422 source vector coverage/shape changed")
    binding = {
        "sha256": measured_sha, "sidecar_sha256": sha256_file(sidecar),
        "identity_sha256": metadata.get("identity_sha256"), "records": count,
        "vectors_sha256": _array_sha(vectors),
    }
    return payload, metadata, binding


def validate_vector422_sidecar_manifest(
    manifest_path: str | Path, *, source_path: str | Path,
) -> dict[str, Any]:
    """Read-only pre-submit check without torch-loading a multi-GB cache.

    File checksums remain mandatory. The training loader additionally compares
    row identities and the actual tail tensor supplied by its dataset.
    """
    path, source = Path(manifest_path).resolve(), Path(source_path).resolve()
    manifest = _read_json(path)
    body = {k: v for k, v in manifest.items() if k != "identity_sha256"}
    if manifest.get("format") != SIDECAR_FORMAT or manifest.get("status") != "complete" or manifest.get("identity_sha256") != semantic_sha256(body):
        raise ValueError("422 condition manifest is incomplete or its identity changed")
    if manifest.get("schema_sha256") != DESCRIPTOR422_SCHEMA_SHA256 or manifest.get("legacy25_schema_sha256") != LEGACY25_SCHEMA_SHA256 or manifest.get("tail397_schema_sha256") != DESCRIPTOR_397_SCHEMA_SHA256 or manifest.get("transform") != TRANSFORM:
        raise ValueError("422 schema/transform changed")
    count = int(manifest.get("records", -1))
    source_meta = _read_json(source.with_suffix(source.suffix + ".json"))
    if source_meta.get("corpus_name") == "final_test":
        raise ValueError("422 comparison preparation excludes the final-test bank")
    binding = manifest.get("source", {})
    if source_meta.get("status") != "complete" or source_meta.get("format") not in _SOURCE_FORMATS or count <= 0 or int(source_meta.get("event_count", source_meta.get("prefix", -1))) != count or int(binding.get("records", -1)) != count or source_meta.get("identity_sha256") != binding.get("identity_sha256") or source_meta.get("payload_sha256") != binding.get("sha256") or sha256_file(source) != binding.get("sha256") or sha256_file(source.with_suffix(source.suffix + ".json")) != binding.get("sidecar_sha256"):
        raise ValueError("422 source cache identity changed")
    if source_meta.get("input_treatment", {}).get("format") != "s5_hypernet_vector397_fixed_log1p_v1":
        raise ValueError("422 original source treatment changed")
    for name, shape, dtype in (("prefix", (count, 25), np.float32), ("row_identities", (count, 64), np.uint8)):
        record = manifest[name]
        file = _resolve(path.parent, record["path"])
        if not file.is_relative_to(path.parent):
            raise ValueError("422 sidecar tensor path escapes its artifact directory")
        if sha256_file(file) != record["sha256"]:
            raise ValueError(f"422 {name} checksum changed")
        value = np.load(file, mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"422 {name} coverage/shape changed")
        if name == "prefix":
            for start in range(0, count, 8192):
                if not np.isfinite(value[start:start + 8192]).all():
                    raise ValueError("422 prefix is incomplete/nonfinite")
    return manifest


def _resolve(parent: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (parent / candidate).resolve()


def _payload_workers(path: Path, bindings: list[dict[str, str]], seen: set[Path]):
    path = path.resolve()
    if path in seen:
        raise ValueError("duplicate/cyclic Payload manifest")
    seen.add(path)
    data = _read_json(path)
    bindings.append({"path": str(path), "sha256": sha256_file(path)})
    if data.get("format") == PAYLOAD_V2_FORMAT:
        if data.get("status") != "complete":
            raise ValueError("incomplete Payload worker manifest")
        identity = data.get("archive_identity", {})
        if semantic_sha256(identity) != data.get("archive_identity_sha256"):
            raise ValueError("Payload archive identity changed")
        assert_condition_safe_archive_identity(identity)
        if int(identity.get("n_layers", -1)) != 8 or int(identity.get("d_feature", -1)) <= 0:
            raise ValueError("422 requires the frozen eight-layer S5 Payload graph")
        yield path, data
    elif data.get("format") == PAYLOAD_COLLECTION_FORMAT:
        collection = read_payload_v2_collection_index(path)
        for worker in collection["workers"]:
            yield from _payload_workers(resolve_worker_manifest(path, worker), bindings, seen)
    else:
        children = data.get("payload_manifests") or data.get("source_payload_manifests")
        if not isinstance(children, list) or not children:
            raise ValueError(f"unsupported Payload manifest: {path}")
        for child in children:
            if not isinstance(child, str):
                raise ValueError("nested Payload manifest entries must be paths")
            yield from _payload_workers(_resolve(path.parent, child), bindings, seen)


def _source_identities(payload, metadata, count, vector_manifest):
    if "event_key" in payload and "payload_artifact_id" in payload:
        return np.concatenate((
            _identity_array(payload["event_key"], count),
            _identity_array(payload["payload_artifact_id"], count),
        ), axis=1), None
    if vector_manifest is None:
        raise ValueError("packed cache lacks row identities; --vector397-manifest is required")
    path = Path(vector_manifest).resolve()
    if sha256_file(path) != metadata.get("archive_manifest_sha256"):
        raise ValueError("packed cache original Vector397 manifest checksum changed")
    archive = _read_json(path)
    if archive.get("archive_identity_sha256") != metadata.get("archive_identity_sha256"):
        raise ValueError("packed cache original Vector397 archive identity changed")
    # Existing reader verifies the archive and canonical indices; no supervision
    # or generated text is opened to recover condition identities.
    from ..composition_loco import _iter_vector_archive_rows_generic
    rows = np.empty((count, 64), dtype=np.uint8)
    seen = 0
    for index, event, artifact, raw in _iter_vector_archive_rows_generic(path, count):
        if index != seen:
            raise ValueError("packed cache Vector397 canonical row order changed")
        fixed = fixed_log1p_397(torch.from_numpy(raw))
        if not torch.equal(fixed, payload["vectors"][index]):
            raise ValueError("packed cache Vector397 row does not match its source archive")
        rows[index] = np.frombuffer(bytes.fromhex(event + artifact), dtype=np.uint8)
        seen += 1
    if seen != count:
        raise ValueError("packed cache identity coverage is incomplete")
    return rows, {"path": str(path), "sha256": sha256_file(path)}


class Vector422Sidecar:
    def __init__(self, manifest_path: Path, metadata, prefix, vectors):
        self.manifest_path = manifest_path
        self.metadata = metadata
        self.identity = metadata
        self.prefix = prefix
        self.vectors = vectors
        self.size = int(vectors.shape[0])
        self.condition_dim = 422

    def condition_vectors(self, indices: Tensor | Sequence[int]) -> Tensor:
        index = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if not index.numel() or bool(((index < 0) | (index >= self.size)).any()):
            raise IndexError("422 condition indices are invalid")
        prefix = torch.from_numpy(np.array(self.prefix[index.numpy()], copy=True))
        return torch.cat((prefix, self.vectors.index_select(0, index)), dim=1)

    index_select = condition_vectors


def load_vector422_sidecar(
    manifest_path: str | Path, *, source_path: str | Path, vectors: Tensor,
    event_keys: Tensor | Sequence[str] | None = None,
    payload_artifact_ids: Tensor | Sequence[str] | None = None,
) -> Vector422Sidecar:
    """Validate full coverage and return a lazy 25+397 condition source."""
    path = Path(manifest_path).resolve()
    manifest = validate_vector422_sidecar_manifest(path, source_path=source_path)
    binding = manifest["source"]
    count = int(manifest["records"])
    if vectors.dtype != torch.float32 or tuple(vectors.shape) != (count, 397) or _array_sha(vectors) != binding["vectors_sha256"]:
        raise ValueError("422 original tail tensor changed")
    arrays = {}
    for name, shape, dtype in (("prefix", (count, 25), np.float32), ("row_identities", (count, 64), np.uint8)):
        record = manifest[name]
        file = _resolve(path.parent, record["path"])
        value = np.load(file, mmap_mode="r", allow_pickle=False)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"422 {name} coverage/shape changed")
        arrays[name] = value
    if (event_keys is None) != (payload_artifact_ids is None):
        raise ValueError("both event and payload identities are required together")
    if event_keys is None:
        # Only needed for callers that do not already pass the identity tensors.
        # mmap leaves token/answer tensors untouched and avoids copying a cache.
        original = torch.load(Path(source_path).resolve(), map_location="cpu", weights_only=True, mmap=True)
        if "event_key" in original:
            event_keys, payload_artifact_ids = original["event_key"], original["payload_artifact_id"]
    if event_keys is not None:
        if not np.array_equal(arrays["row_identities"][:, :32], _identity_array(event_keys, count)) or not np.array_equal(arrays["row_identities"][:, 32:], _identity_array(payload_artifact_ids, count)):
            raise ValueError("422 event/payload row order changed")
    return Vector422Sidecar(path, manifest, arrays["prefix"], vectors)


def prepare_vector422_sidecar(
    *, source_path: str | Path, payload_manifest: str | Path,
    output_dir: str | Path, vector397_manifest: str | Path | None = None,
    chunk_size: int = 64, max_new_rows: int | None = None,
) -> dict[str, Any]:
    """Stream graph shards, checkpoint verified chunks, publish after full join.

    A failed/interrupted run retains NaNs for unfinished rows and immutable
    per-chunk digests. Resume checks every completed chunk before reuse.
    """
    started = time.perf_counter()
    source = Path(source_path).resolve()
    graph_manifest = Path(payload_manifest).resolve()
    output = Path(output_dir).resolve()
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    if max_new_rows is not None and int(max_new_rows) <= 0:
        raise ValueError("max_new_rows must be positive")
    if output == source or output == source.parent or output in source.parents or output == graph_manifest.parent or output in graph_manifest.parents:
        raise ValueError("422 output must be isolated from original source artifacts")
    payload, metadata, binding = _load_source(source)
    count = binding["records"]
    output.mkdir(parents=True, exist_ok=True)
    complete = output / "manifest.json"
    if complete.exists():
        loaded = load_vector422_sidecar(complete, source_path=source, vectors=payload["vectors"])
        recorded_payload = loaded.metadata.get("payload_manifests", [])
        if not recorded_payload or sha256_file(graph_manifest) != recorded_payload[0]["sha256"]:
            raise ValueError("422 completed preparation Payload manifest changed")
        if vector397_manifest is not None and (loaded.metadata.get("vector397_manifest") or {}).get("sha256") != sha256_file(Path(vector397_manifest)):
            raise ValueError("422 completed preparation Vector397 manifest changed")
        return dict(loaded.metadata)
    lock = output / ".prepare.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("422 preparation lock exists; inspect the previous process before removing it") from exc
    os.close(fd)
    try:
        workers_bindings: list[dict[str, str]] = []
        workers = list(_payload_workers(graph_manifest, workers_bindings, set()))
        ids, archive_binding = _source_identities(payload, metadata, count, vector397_manifest)
        # A full 64-byte key binds BOTH event and payload identity.  Mapping is
        # numeric/identity metadata only; graph tensors remain shard-bounded.
        positions = {row.tobytes(): index for index, row in enumerate(ids)}
        if len(positions) != count or len({row[:32].tobytes() for row in ids}) != count:
            raise ValueError("source event/payload identities contain duplicates")
        identity = {
            "format": SIDECAR_FORMAT, "records": count, "source": binding,
            "payload_manifests": workers_bindings, "vector397_manifest": archive_binding,
            "schema_sha256": DESCRIPTOR422_SCHEMA_SHA256,
            "legacy25_schema_sha256": LEGACY25_SCHEMA_SHA256,
            "tail397_schema_sha256": DESCRIPTOR_397_SCHEMA_SHA256, "transform": TRANSFORM,
            "row_identities_sha256": _array_sha(ids),
            "implementation": {
                "descriptor_legacy25_sha256": sha256_file(Path(__file__).parents[1] / "circuit" / "descriptor_legacy25.py"),
                "descriptor397_sha256": sha256_file(Path(__file__).parents[1] / "circuit" / "descriptor_v2_397.py"),
            },
        }
        progress_path = output / "preparation.json"
        prefix_path, ids_path = output / "prefix25_fixed.npy", output / "row_identities.npy"
        if progress_path.exists():
            if _read_json(progress_path) != identity:
                raise ValueError("422 resume source/schema/payload identities changed")
            if not np.array_equal(np.load(ids_path, mmap_mode="r", allow_pickle=False), ids):
                raise ValueError("422 resume event/payload identities changed")
            prefix = np.load(prefix_path, mmap_mode="r+", allow_pickle=False)
            if prefix.shape != (count, 25) or prefix.dtype != np.float32:
                raise ValueError("422 resume prefix shape changed")
        else:
            unexpected = [p for p in output.iterdir() if p != lock]
            if unexpected:
                raise ValueError("new 422 output is not empty")
            prefix = np.lib.format.open_memmap(prefix_path, mode="w+", dtype=np.float32, shape=(count, 25))
            prefix[:] = np.nan
            prefix.flush()
            np.save(ids_path, ids, allow_pickle=False)
            _atomic_json(progress_path, identity)
        chunks = output / "chunks"
        chunks.mkdir(exist_ok=True)
        done = np.zeros(count, dtype=bool)
        for receipt in sorted(chunks.glob("chunk_*.json")):
            record = _read_json(receipt)
            index = np.asarray(record["indices"], dtype=np.int64)
            if not len(index) or np.any(index < 0) or np.any(index >= count) or len(np.unique(index)) != len(index) or done[index].any():
                raise ValueError("422 resume chunk coverage overlaps/changed")
            if record.get("preparation_sha256") != semantic_sha256(identity) or record.get("values_sha256") != _array_sha(prefix[index]):
                raise ValueError("422 resume chunk checksum changed")
            done[index] = True
        pending_indices, pending_values = [], []
        chunk_number = len(list(chunks.glob("chunk_*.json")))

        def flush():
            nonlocal chunk_number
            if not pending_indices:
                return
            index = np.asarray(pending_indices, dtype=np.int64)
            prefix[index] = np.asarray(pending_values, dtype=np.float32)
            prefix.flush()
            _atomic_json(chunks / f"chunk_{chunk_number:08d}.json", {
                "preparation_sha256": semantic_sha256(identity),
                "indices": pending_indices, "values_sha256": _array_sha(prefix[index]),
            })
            done[index] = True
            chunk_number += 1
            pending_indices.clear()
            pending_values.clear()

        encountered: set[int] = set()
        new_rows = 0
        computation_seconds = 0.0
        for worker_path, worker in workers:
            archive_sha = worker["archive_identity_sha256"]
            dims = worker["archive_identity"]
            for original_record in worker["shards"]:
                record = {**original_record,
                    "path": str(_resolve(worker_path.parent, original_record["path"])),
                    "sidecar": str(_resolve(worker_path.parent, original_record["sidecar"])),
                }
                # Full checksums and Payload validation precede any numeric use.
                graphs = verify_payload_v2_shard_record(record, expected_archive_identity_sha256=archive_sha)
                for graph in graphs:
                    index = positions.get(bytes.fromhex(graph.event_key + graph.payload_artifact_id))
                    if index is None:
                        continue
                    if index in encountered:
                        raise ValueError("duplicate event/payload in source graph archive")
                    encountered.add(index)
                    if done[index]:
                        continue
                    row_started = time.perf_counter()
                    fixed_tail = fixed_log1p_397(vectorize_payload_v2_397(
                        graph, n_layers=int(dims["n_layers"]),
                        d_feature=int(dims["d_feature"]), archive_identity_sha256=archive_sha,
                    ))
                    if not torch.equal(fixed_tail, payload["vectors"][index]):
                        raise ValueError("retained Payload graph differs from the original 397 cache row")
                    result = compute_legacy25(graph, n_layers=int(dims["n_layers"]), d_feature=int(dims["d_feature"]), archive_identity_sha256=archive_sha)
                    fixed = fixed_log1p_legacy25(result.vector, diagnostics=result.diagnostics)
                    pending_indices.append(index)
                    pending_values.append(fixed.numpy())
                    new_rows += 1
                    computation_seconds += time.perf_counter() - row_started
                    if len(pending_indices) >= chunk_size:
                        flush()
                    if max_new_rows is not None and new_rows >= max_new_rows:
                        flush()
                        partial = {
                            "format": SIDECAR_FORMAT, "status": "partial",
                            "records": count, "new_records": new_rows,
                            "completed_records": int(done.sum()),
                            "elapsed_seconds": time.perf_counter() - started,
                            "computation_seconds": computation_seconds,
                            "identity_sha256": semantic_sha256(identity),
                            "note": "bounded throughput prefix; no complete manifest published",
                        }
                        _atomic_json(output / "last_partial.json", partial)
                        return partial
                del graphs
        flush()
        if len(encountered) != count or not done.all():
            raise ValueError(f"422 graph join coverage incomplete: {len(encountered)}/{count}")
        # Check source files again before publication; unchanged old caches are
        # the only source for the 397D tail.
        if sha256_file(source) != binding["sha256"] or sha256_file(source.with_suffix(source.suffix + ".json")) != binding["sidecar_sha256"]:
            raise ValueError("422 source changed while preparing")
        for worker_binding in workers_bindings:
            if sha256_file(Path(worker_binding["path"])) != worker_binding["sha256"]:
                raise ValueError("Payload manifest changed while preparing")
        result = {**identity, "status": "complete",
            "prefix": {"path": prefix_path.name, "sha256": sha256_file(prefix_path)},
            "row_identities": {"path": ids_path.name, "sha256": sha256_file(ids_path)},
        }
        result["identity_sha256"] = semantic_sha256(result)
        _atomic_json(complete, result)
        load_vector422_sidecar(complete, source_path=source, vectors=payload["vectors"])
        return result
    finally:
        lock.unlink(missing_ok=True)

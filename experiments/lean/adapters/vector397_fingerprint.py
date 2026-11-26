"""Production train-only z-score fingerprints for the S5 Vector397 bank.

The raw descriptor archive and the historical fixed-log1p composition cache
remain authoritative inputs.  This module adds a separately versioned,
checksum-bound derived view.  It intentionally does not put task labels,
prompt identifiers, rewards, or gold outputs into the numeric fingerprint.

The implementation is streaming over safetensors shards.  It can fit one
global all-five normalizer and five leave-one-composition-out (LOCO)
normalizers in one pass, then optionally materialize independent normalized
archives for downstream consumers that need a disk-backed loader.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ..circuit.descriptor_v2_397_artifact import read_descriptor397_archive
from .circuit_normalizer import (
    DEAD_STD_THRESHOLD,
    Vector397Normalizer,
    _array_sha256,
    _fixed_log1p_mask,
    _file_sha256,
    read_vector397_normalizer,
    write_vector397_normalizer,
)
from .data import VECTOR397_DIM, VECTOR397_SCHEMA_SHA256, StrictDataError


FINGERPRINT_MANIFEST_FORMAT = "s5_vector397_fingerprint_bank_manifest_v1"
FINGERPRINT_ARCHIVE_FORMAT = "s5_vector397_fingerprint_archive_v1"
FINGERPRINT_VECTOR_NAME = "vector397_fingerprint"
SOURCE_CACHE_FORMAT = "s5_composition_loco_vector397_cache_v1"
FORMAL_SOURCE_CACHE_FORMAT = "s5_hypernet_formal_packed_cache_v1"
FINGERPRINT_VERSION = "fixed_log1p_then_train_only_population_zscore_v1"

COMPOSITION_TASKS: tuple[str, ...] = (
    "T_and_or",
    "T_or_and",
    "T_and_or_and",
    "T_nested_and",
    "T_nested_or",
)
ATOMIC_TASKS: tuple[str, ...] = ("S_and", "S_or_l", "S_or_r", "S_id")
TASK_TO_ID = {name: index for index, name in enumerate(COMPOSITION_TASKS)}

DEFAULT_EVENT_COUNT = 1_500_000
DEFAULT_ARCHIVE_IDENTITY_SHA256 = (
    "000aad6433c23d445ff1554dc57077fdaf2d45f7af1da92bca8af5d528b9c100"
)
DEFAULT_CACHE_IDENTITY_SHA256 = (
    "24f9642f186cde72b516447fe51674b4ab7777f780ce6e6e48f20df3b70d73bc"
)
DEFAULT_ATOMIC_EVENT_COUNT = 1_200_000
DEFAULT_ATOMIC_ARCHIVE_IDENTITY_SHA256 = (
    "371f8bde1cb4cf41c7cc456a5a6651e6d19be1f9c8cdf8f48a34e1eab4794cf0"
)
DEFAULT_ATOMIC_CACHE_IDENTITY_SHA256 = (
    "101aca4ccfb844eeeab30fde7d2ac4545536d5f7a4fe22b91e9e252c0c4147cb"
)
_FINGERPRINT_TENSOR_NAMES = frozenset(
    {"global_index", "event_key", "payload_artifact_id", FINGERPRINT_VECTOR_NAME}
)


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _sha256_hex(value: Any, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise StrictDataError(f"{field} is not a lowercase SHA256")
    return text


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _fixed_log1p_batch(raw: np.ndarray) -> np.ndarray:
    """Apply the frozen Vector397 mask with the same FP32 arithmetic as S5."""

    value = np.asarray(raw, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != VECTOR397_DIM:
        raise StrictDataError("raw Vector397 batch must have shape [N,397]")
    if not np.isfinite(value).all():
        raise StrictDataError("raw Vector397 batch is non-finite")
    mask = _fixed_log1p_mask()
    if bool(np.any(value[:, mask] < 0.0)):
        raise StrictDataError("raw Vector397 log1p coordinate is negative")
    result = np.array(value, dtype=np.float32, copy=True, order="C")
    result[:, mask] = torch.log1p(torch.from_numpy(result[:, mask])).numpy()
    if not np.isfinite(result).all():
        raise StrictDataError("fixed-log1p Vector397 batch is non-finite")
    return result


def _hex_tensor(value: np.ndarray, field: str) -> list[str]:
    array = np.asarray(value)
    if array.dtype != np.uint8 or array.ndim != 2 or array.shape[1] != 32:
        raise StrictDataError(f"{field} must be uint8[N,32]")
    return [bytes(row.tolist()).hex() for row in array]


@dataclass(frozen=True)
class FingerprintSource:
    """Validated raw archive/cache pair used by every fitting/materialization pass."""

    archive_path: Path
    cache_path: Path
    archive: Mapping[str, Any]
    cache_metadata: Mapping[str, Any]
    cache_payload: Mapping[str, torch.Tensor]
    event_count: int
    task_names: tuple[str, ...] = COMPOSITION_TASKS
    source_kind: str = "composition"

    @property
    def archive_identity_sha256(self) -> str:
        return str(self.archive["archive_identity_sha256"])

    @property
    def cache_identity_sha256(self) -> str:
        return str(self.cache_metadata["identity_sha256"])


def load_fingerprint_source(
    *,
    archive_manifest: str | Path,
    cache_path: str | Path,
    expected_archive_identity_sha256: str | None = DEFAULT_ARCHIVE_IDENTITY_SHA256,
    expected_cache_identity_sha256: str | None = DEFAULT_CACHE_IDENTITY_SHA256,
    expected_event_count: int | None = DEFAULT_EVENT_COUNT,
    expected_task_names: Sequence[str] | None = None,
    expected_cache_format: str = SOURCE_CACHE_FORMAT,
    expected_protocol_id: str = "s5_composition_loco_sft_v1",
    source_kind: str = "composition",
) -> FingerprintSource:
    """Open the raw archive and fixed-log1p cache with fail-closed identity checks."""
    task_names = tuple(str(value) for value in (expected_task_names or COMPOSITION_TASKS))
    if not task_names or len(set(task_names)) != len(task_names):
        raise ValueError("expected_task_names must be non-empty and unique")

    archive_path = Path(archive_manifest).resolve()
    cache_file = Path(cache_path).resolve()
    cache_sidecar = _sidecar(cache_file)
    if not archive_path.is_file():
        raise FileNotFoundError(f"raw Vector397 manifest is missing: {archive_path}")
    if not cache_file.is_file() or not cache_sidecar.is_file():
        raise FileNotFoundError(f"composition cache or sidecar is missing: {cache_file}")
    archive = read_descriptor397_archive(archive_path, full_tensor_verify=False)
    if archive.get("descriptor_schema_sha256") != VECTOR397_SCHEMA_SHA256:
        raise StrictDataError("raw Vector397 schema identity changed")
    archive_identity = _sha256_hex(
        archive.get("archive_identity_sha256"), "archive_identity_sha256"
    )
    if expected_archive_identity_sha256 is not None and archive_identity != expected_archive_identity_sha256:
        raise StrictDataError("raw Vector397 archive identity differs from the frozen source")
    event_count = int(archive.get("events", -1))
    if event_count <= 0 or (expected_event_count is not None and event_count != int(expected_event_count)):
        raise StrictDataError("raw Vector397 event count differs from the frozen source")
    try:
        cache_metadata = json.loads(cache_sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictDataError("composition cache sidecar is unreadable") from error
    if (
        not isinstance(cache_metadata, dict)
        or cache_metadata.get("format") != expected_cache_format
        or cache_metadata.get("protocol_id") != expected_protocol_id
    ):
        raise StrictDataError("unsupported Vector397 source cache format")
    if cache_metadata.get("status") != "complete":
        raise StrictDataError("composition Vector397 cache is incomplete")
    cache_identity = _sha256_hex(cache_metadata.get("identity_sha256"), "cache identity_sha256")
    if expected_cache_identity_sha256 is not None and cache_identity != expected_cache_identity_sha256:
        raise StrictDataError("composition cache identity differs from the frozen source")
    cache_event_count = int(cache_metadata.get("event_count", cache_metadata.get("prefix", -1)))
    if cache_event_count != event_count:
        raise StrictDataError("raw/cache event count differs")
    metadata_tasks = cache_metadata.get("tasks")
    if metadata_tasks is not None:
        if tuple(str(value) for value in metadata_tasks) != task_names:
            raise StrictDataError("source cache task order changed")
    else:
        expected_task_to_id = {task: index for index, task in enumerate(task_names)}
        if cache_metadata.get("task_to_id") != expected_task_to_id:
            raise StrictDataError("source cache task mapping changed")
    expected_task_counts = {task: event_count // len(task_names) for task in task_names}
    metadata_task_counts = cache_metadata.get("task_counts")
    if metadata_task_counts is not None and (
        event_count % len(task_names) != 0 or metadata_task_counts != expected_task_counts
    ):
        raise StrictDataError("source cache task balance changed")
    cache_manifest_sha = cache_metadata.get("vector_manifest_sha256") or cache_metadata.get("archive_manifest_sha256")
    if cache_manifest_sha != _file_sha256(archive_path):
        raise StrictDataError("source cache is bound to a different raw manifest")
    cache_archive_identity = cache_metadata.get("vector_archive_identity_sha256") or cache_metadata.get("archive_identity_sha256")
    if cache_archive_identity != archive_identity:
        raise StrictDataError("source cache is bound to a different raw archive")
    payload_path = cache_metadata.get("payload_path")
    if (
        (payload_path is not None and payload_path != cache_file.name)
        or int(cache_metadata.get("payload_bytes", -1)) != cache_file.stat().st_size
        or cache_metadata.get("payload_sha256") != _file_sha256(cache_file)
    ):
        raise StrictDataError("composition cache payload checksum changed")
    cache_identity_fields = {
        key: value
        for key, value in cache_metadata.items()
        if key not in {
            "status",
            "payload_path",
            "payload_sha256",
            "payload_bytes",
            "identity_sha256",
            "build_wall_seconds",
        }
    }
    if semantic_sha256(cache_identity_fields) != cache_identity:
        raise StrictDataError("composition cache semantic identity changed")
    payload = torch.load(cache_file, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(payload, Mapping):
        raise StrictDataError("composition cache payload is not a mapping")
    required = {"vectors", "task_ids"}
    if not required.issubset(payload):
        raise StrictDataError("source cache is missing fingerprint source tensors")
    vectors = payload["vectors"]
    task_ids = payload["task_ids"]
    event_key = payload.get("event_key")
    payload_id = payload.get("payload_artifact_id")
    if (
        not isinstance(vectors, torch.Tensor)
        or vectors.dtype != torch.float32
        or tuple(vectors.shape) != (event_count, VECTOR397_DIM)
        or not isinstance(task_ids, torch.Tensor)
        or task_ids.dtype not in (torch.uint8, torch.int64, torch.int32)
        or tuple(task_ids.shape) != (event_count,)
    ):
        raise StrictDataError("source cache fingerprint tensor schema changed")
    if (event_key is None) != (payload_id is None):
        raise StrictDataError("source cache identity tensors are incomplete")
    if event_key is not None and (
        not isinstance(event_key, torch.Tensor)
        or event_key.dtype != torch.uint8
        or tuple(event_key.shape) != (event_count, 32)
        or not isinstance(payload_id, torch.Tensor)
        or payload_id.dtype != torch.uint8
        or tuple(payload_id.shape) != (event_count, 32)
    ):
        raise StrictDataError("source cache identity tensor schema changed")
    if not bool(torch.isfinite(vectors).all()):
        raise StrictDataError("source cache fixed-log1p vectors are non-finite")
    if bool(((task_ids < 0) | (task_ids >= len(task_names))).any()):
        raise StrictDataError("source cache task ids are invalid")
    actual_task_counts = {
        task: int((task_ids == task_id).sum().item())
        for task_id, task in enumerate(task_names)
    }
    if actual_task_counts != expected_task_counts:
        raise StrictDataError("source cache tensor task balance changed")
    return FingerprintSource(
        archive_path=archive_path,
        cache_path=cache_file,
        archive=archive,
        cache_metadata=cache_metadata,
        cache_payload=payload,
        event_count=event_count,
        task_names=task_names,
        source_kind=str(source_kind),
    )


def iter_source_batches(
    source: FingerprintSource,
    *,
    batch_rows: int = 8192,
    verify_cache_parity: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``(indices,event,payload,raw,fixed,task_ids)`` in canonical order."""

    if int(batch_rows) <= 0:
        raise ValueError("batch_rows must be positive")
    cursor = 0
    for shard_number, raw_record in enumerate(source.archive["shards"]):
        if not isinstance(raw_record, Mapping):
            raise StrictDataError(f"raw Vector397 shard {shard_number} is malformed")
        start = int(raw_record.get("global_start", -1))
        end = int(raw_record.get("global_end", -1))
        count = int(raw_record.get("records", -1))
        if start != cursor or end - start != count or count <= 0:
            raise StrictDataError(f"raw Vector397 shard {shard_number} coverage changed")
        path = _resolve(source.archive_path.parent, str(raw_record.get("path", "")))
        if not path.is_file():
            raise StrictDataError(f"raw Vector397 shard {shard_number} is missing")
        with safe_open(str(path), framework="np", device="cpu") as handle:
            if set(handle.keys()) != {"global_index", "event_key", "payload_artifact_id", "descriptor_raw"}:
                raise StrictDataError("raw Vector397 shard tensor schema changed")
            index_slice = handle.get_slice("global_index")
            event_slice = handle.get_slice("event_key")
            payload_slice = handle.get_slice("payload_artifact_id")
            raw_slice = handle.get_slice("descriptor_raw")
            if (
                index_slice.get_dtype() != "I64"
                or event_slice.get_dtype() != "U8"
                or payload_slice.get_dtype() != "U8"
                or raw_slice.get_dtype() != "F32"
                or index_slice.get_shape() != [count]
                or event_slice.get_shape() != [count, 32]
                or payload_slice.get_shape() != [count, 32]
                or raw_slice.get_shape() != [count, VECTOR397_DIM]
            ):
                raise StrictDataError("raw Vector397 shard vector shape changed")
            for offset in range(0, count, int(batch_rows)):
                length = min(int(batch_rows), count - offset)
                indices = np.asarray(index_slice[offset : offset + length], dtype=np.int64)
                event_ids = np.asarray(event_slice[offset : offset + length], dtype=np.uint8)
                payload_ids = np.asarray(payload_slice[offset : offset + length], dtype=np.uint8)
                raw = np.asarray(raw_slice[offset : offset + length], dtype=np.float32)
                expected_indices = np.arange(start + offset, start + offset + length, dtype=np.int64)
                if not np.array_equal(indices, expected_indices):
                    raise StrictDataError("raw Vector397 global index ordering changed")
                fixed = _fixed_log1p_batch(raw)
                cache_start = start + offset
                cache_end = cache_start + length
                cached_fixed = source.cache_payload["vectors"][cache_start:cache_end].numpy()
                cached_events = source.cache_payload.get("event_key")
                cached_payloads = source.cache_payload.get("payload_artifact_id")
                if (cached_events is None) != (cached_payloads is None):
                    raise StrictDataError("source cache identity tensors are incomplete")
                if cached_events is not None and (
                    not np.array_equal(event_ids, cached_events[cache_start:cache_end].numpy())
                    or not np.array_equal(payload_ids, cached_payloads[cache_start:cache_end].numpy())
                ):
                    raise StrictDataError("raw/cache event or Payload identity binding changed")
                if verify_cache_parity:
                    delta = np.abs(fixed.astype(np.float64) - cached_fixed.astype(np.float64))
                    if float(delta.max(initial=0.0)) > 1.0e-6:
                        raise StrictDataError("raw/cache fixed-log1p parity changed")
                task_ids = source.cache_payload["task_ids"][cache_start:cache_end].numpy().astype(np.int64, copy=False)
                yield expected_indices, event_ids, payload_ids, raw, fixed, task_ids
        cursor = end
    if cursor != source.event_count:
        raise StrictDataError("raw Vector397 archive coverage is incomplete")


def _update_welford(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    values: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != VECTOR397_DIM:
        raise StrictDataError("Welford batch shape changed")
    size = int(values.shape[0])
    if size == 0:
        return count, mean, m2
    batch = values.astype(np.float64, copy=False)
    batch_mean = batch.mean(axis=0, dtype=np.float64)
    centered = batch - batch_mean
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
    if count == 0:
        return size, np.array(batch_mean, copy=True), np.array(batch_m2, copy=True)
    total = count + size
    delta = batch_mean - mean
    mean = mean + delta * (size / total)
    m2 = m2 + batch_m2 + delta * delta * (count * size / total)
    return total, mean, m2


class _Accumulator:
    def __init__(
        self,
        *,
        scope: str,
        heldout_task: str | None,
        archive_identity_sha256: str,
        task_names: Sequence[str],
    ):
        self.scope = str(scope)
        self.heldout_task = heldout_task
        self.archive_identity_sha256 = archive_identity_sha256
        self.task_names = tuple(str(value) for value in task_names)
        self.count = 0
        self.mean = np.zeros(VECTOR397_DIM, dtype=np.float64)
        self.m2 = np.zeros(VECTOR397_DIM, dtype=np.float64)
        self.index_digest = hashlib.sha256()
        self.identity_digest = hashlib.sha256()
        self.task_counts: Counter[str] = Counter()

    def update(
        self,
        indices: np.ndarray,
        event_ids: np.ndarray,
        payload_ids: np.ndarray,
        fixed: np.ndarray,
        task_ids: np.ndarray,
    ) -> None:
        task_names = np.asarray([self.task_names[int(value)] for value in task_ids], dtype=object)
        selected = np.ones(indices.shape[0], dtype=np.bool_)
        if self.heldout_task is not None:
            selected = task_names != self.heldout_task
        if not bool(selected.any()):
            return
        selected_indices = np.asarray(indices[selected], dtype=np.int64)
        selected_events = np.asarray(event_ids[selected], dtype=np.uint8)
        selected_payloads = np.asarray(payload_ids[selected], dtype=np.uint8)
        selected_fixed = np.asarray(fixed[selected], dtype=np.float32)
        self.count, self.mean, self.m2 = _update_welford(
            self.count, self.mean, self.m2, selected_fixed
        )
        for index, event_id, payload_id, task_name in zip(
            selected_indices.tolist(), selected_events, selected_payloads, task_names[selected].tolist()
        ):
            self.index_digest.update(np.int64(index).tobytes())
            self.identity_digest.update(np.int64(index).tobytes())
            self.identity_digest.update(bytes(event_id.tolist()))
            self.identity_digest.update(bytes(payload_id.tolist()))
            self.task_counts[str(task_name)] += 1

    def finish(self) -> Vector397Normalizer:
        if self.count <= 0:
            raise StrictDataError(f"normalizer {self.scope} has no training rows")
        variance = np.maximum(self.m2 / float(self.count), 0.0)
        std = np.sqrt(variance, dtype=np.float64)
        dead = std < DEAD_STD_THRESHOLD
        return Vector397Normalizer(
            descriptor_schema_sha256=VECTOR397_SCHEMA_SHA256,
            training_event_count=self.count,
            training_identity_sha256=self.identity_digest.hexdigest(),
            mean=np.ascontiguousarray(self.mean),
            std=np.ascontiguousarray(std),
            dead_mask=np.ascontiguousarray(dead),
            archive_identity_sha256=self.archive_identity_sha256,
            training_prefix_global_index_sha256=self.index_digest.hexdigest(),
        )

    def selection_metadata(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "heldout_task": self.heldout_task,
            "training_event_count": int(self.count),
            "training_task_counts": dict(self.task_counts),
            "training_identity_scheme": "sha256(index_i64_le||event_key_raw32||payload_artifact_id_raw32)",
            "training_index_digest_scheme": "sha256(concatenated_canonical_global_index_i64_le)",
            "training_index_sha256": self.index_digest.hexdigest(),
            "training_identity_sha256": self.identity_digest.hexdigest(),
        }


def fit_fingerprint_normalizers(
    source: FingerprintSource,
    *,
    batch_rows: int = 8192,
    global_name: str | None = None,
    include_loco: bool = True,
) -> dict[str, tuple[Vector397Normalizer, dict[str, Any]]]:
    """Fit a global normalizer and optional leave-one-task-out views."""

    task_names = tuple(source.task_names)
    global_name = global_name or (
        "all_five_global_train"
        if task_names == COMPOSITION_TASKS
        else f"all_{source.source_kind}_global_train"
    )
    global_scope = "all-five/global-train" if task_names == COMPOSITION_TASKS else f"{source.source_kind}/global-train"
    accumulators: dict[str, _Accumulator] = {
        global_name: _Accumulator(
            scope=global_scope,
            heldout_task=None,
            archive_identity_sha256=source.archive_identity_sha256,
            task_names=task_names,
        )
    }
    if include_loco:
        for task in task_names:
            accumulators[f"loco_holdout_{task}"] = _Accumulator(
                scope="LOCO",
                heldout_task=task,
                archive_identity_sha256=source.archive_identity_sha256,
                task_names=task_names,
            )
    for indices, event_ids, payload_ids, _raw, fixed, task_ids in iter_source_batches(
        source, batch_rows=batch_rows, verify_cache_parity=True
    ):
        for accumulator in accumulators.values():
            accumulator.update(indices, event_ids, payload_ids, fixed, task_ids)
    return {
        key: (accumulator.finish(), accumulator.selection_metadata())
        for key, accumulator in accumulators.items()
    }


def _publish_normalizer(
    path: Path,
    normalizer: Vector397Normalizer,
    selection_metadata: Mapping[str, Any],
    *,
    source_cache_identity_sha256: str,
) -> dict[str, Any]:
    """Write/reopen a normalizer and bind scope/cache provenance in its sidecar."""

    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise StrictDataError("normalizer tensor/sidecar pair is incomplete")
        existing, metadata = read_vector397_normalizer(
            path,
            expected_archive_identity_sha256=normalizer.archive_identity_sha256,
            expected_training_prefix_global_index_sha256=normalizer.training_prefix_global_index_sha256,
        )
        if existing.normalizer_sha256 != normalizer.normalizer_sha256:
            raise StrictDataError(f"existing normalizer differs: {path}")
        for field, expected in selection_metadata.items():
            if metadata.get(field) != expected:
                raise StrictDataError(f"existing normalizer {field} differs: {path}")
        if metadata.get("source_cache_identity_sha256") != source_cache_identity_sha256:
            raise StrictDataError(f"existing normalizer cache identity differs: {path}")
        return metadata
    write_vector397_normalizer(path, normalizer)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.update(
        {
            **dict(selection_metadata),
            "source_cache_identity_sha256": source_cache_identity_sha256,
            "fingerprint_version": FINGERPRINT_VERSION,
            "raw_vector_dtype": "float32[397]",
            "normalized_vector_dtype": "float32[397]",
        }
    )
    _atomic_json(sidecar, metadata)
    reopened, _ = read_vector397_normalizer(
        path,
        expected_archive_identity_sha256=normalizer.archive_identity_sha256,
        expected_training_prefix_global_index_sha256=normalizer.training_prefix_global_index_sha256,
    )
    if reopened.normalizer_sha256 != normalizer.normalizer_sha256:
        raise StrictDataError("normalizer changed during publication")
    return json.loads(sidecar.read_text(encoding="utf-8"))


def write_fingerprint_manifest(
    *,
    output_root: str | Path,
    source: FingerprintSource,
    normalizers: Mapping[str, tuple[Vector397Normalizer, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Publish the normalizer index after every tensor/sidecar has reopened."""

    root = Path(output_root).resolve()
    normalizer_root = root / "normalizers"
    normalizer_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for name, (normalizer, selection) in normalizers.items():
        path = normalizer_root / f"{name}.npz"
        metadata = _publish_normalizer(
            path,
            normalizer,
            selection,
            source_cache_identity_sha256=source.cache_identity_sha256,
        )
        records[name] = {
            "name": name,
            "path": str(path.relative_to(root)).replace(os.sep, "/"),
            "sidecar": str(_sidecar(path).relative_to(root)).replace(os.sep, "/"),
            "artifact_sha256": metadata.get("artifact_sha256"),
            "normalizer_sha256": metadata.get("normalizer_sha256"),
            "training_event_count": int(metadata["training_event_count"]),
            "training_identity_sha256": metadata["training_identity_sha256"],
            "training_index_sha256": metadata["training_index_sha256"],
            "scope": metadata["scope"],
            "heldout_task": metadata.get("heldout_task"),
            "training_task_counts": metadata["training_task_counts"],
        }
    identity = {
        "format": FINGERPRINT_MANIFEST_FORMAT,
        "fingerprint_version": FINGERPRINT_VERSION,
        "vector_dim": VECTOR397_DIM,
        "vector_dtype": "float32",
        "descriptor_schema_sha256": VECTOR397_SCHEMA_SHA256,
        "source_archive_manifest_sha256": _file_sha256(source.archive_path),
        "source_archive_identity_sha256": source.archive_identity_sha256,
        "source_cache_sha256": _file_sha256(source.cache_path),
        "source_cache_identity_sha256": source.cache_identity_sha256,
        "event_count": int(source.event_count),
        "source_kind": source.source_kind,
        "tasks": list(source.task_names),
        "normalizers": records,
        "dead_std_threshold": DEAD_STD_THRESHOLD,
        "fixed_log1p_mask_sha256": _array_sha256(_fixed_log1p_mask()),
    }
    manifest = {**identity, "status": "complete", "manifest_identity_sha256": semantic_sha256(identity)}
    _atomic_json(root / "manifest.json", manifest)
    return manifest


def read_fingerprint_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictDataError("fingerprint manifest is unreadable") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FINGERPRINT_MANIFEST_FORMAT:
        raise StrictDataError("unsupported Vector397 fingerprint manifest")
    if manifest.get("status") != "complete":
        raise StrictDataError("Vector397 fingerprint manifest is incomplete")
    identity = {key: value for key, value in manifest.items() if key != "manifest_identity_sha256" and key != "status"}
    if semantic_sha256(identity) != manifest.get("manifest_identity_sha256"):
        raise StrictDataError("Vector397 fingerprint manifest identity changed")
    if manifest.get("descriptor_schema_sha256") != VECTOR397_SCHEMA_SHA256 or int(manifest.get("vector_dim", -1)) != VECTOR397_DIM:
        raise StrictDataError("Vector397 fingerprint schema changed")
    tasks = manifest.get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or any(not isinstance(task, str) or not task for task in tasks)
        or len(set(tasks)) != len(tasks)
    ):
        raise StrictDataError("Vector397 fingerprint task index changed")
    normalizers = manifest.get("normalizers")
    if not isinstance(normalizers, Mapping):
        raise StrictDataError("Vector397 fingerprint normalizer index changed")
    global_names = [name for name in normalizers if str(name).endswith("_global_train")]
    if len(global_names) != 1:
        raise StrictDataError("Vector397 fingerprint global normalizer index changed")
    global_name = str(global_names[0])
    loco_names = {f"loco_holdout_{task}" for task in tasks}
    present_loco_names = {str(name) for name in normalizers if str(name).startswith("loco_holdout_")}
    if present_loco_names and present_loco_names != loco_names:
        raise StrictDataError("Vector397 fingerprint LOCO normalizer index changed")
    expected_names = {global_name} | present_loco_names
    if set(str(name) for name in normalizers) != expected_names:
        raise StrictDataError("Vector397 fingerprint normalizer index changed")
    for name in sorted(expected_names):
        record = normalizers[name]
        if not isinstance(record, Mapping):
            raise StrictDataError(f"Vector397 fingerprint normalizer record {name} is malformed")
        normalizer_path = _resolve(source.parent, str(record.get("path", "")))
        if not normalizer_path.is_file():
            raise StrictDataError(f"Vector397 fingerprint normalizer {name} is missing")
        normalizer, metadata = read_vector397_normalizer(
            normalizer_path,
            expected_archive_identity_sha256=manifest.get("source_archive_identity_sha256"),
            expected_training_prefix_global_index_sha256=record.get("training_index_sha256"),
        )
        if (
            record.get("artifact_sha256") != metadata.get("artifact_sha256")
            or record.get("normalizer_sha256") != metadata.get("normalizer_sha256")
            or record.get("training_event_count") != metadata.get("training_event_count")
            or record.get("training_identity_sha256") != metadata.get("training_identity_sha256")
            or record.get("training_index_sha256") != metadata.get("training_prefix_global_index_sha256")
            or record.get("scope") != metadata.get("scope")
            or record.get("heldout_task") != metadata.get("heldout_task")
            or record.get("training_task_counts") != metadata.get("training_task_counts")
            or normalizer.descriptor_schema_sha256 != VECTOR397_SCHEMA_SHA256
        ):
            raise StrictDataError(f"Vector397 fingerprint normalizer {name} identity changed")
    materialized = manifest.get("materialized_banks")
    if materialized is not None:
        if not isinstance(materialized, Mapping) or set(materialized) != expected_names:
            raise StrictDataError("Vector397 fingerprint materialized-bank index changed")
        for name in sorted(expected_names):
            record = materialized[name]
            if not isinstance(record, Mapping):
                raise StrictDataError(f"Vector397 fingerprint bank record {name} is malformed")
            bank_manifest = _resolve(source.parent, str(record.get("path", ""))) / "manifest.json"
            bank = read_fingerprint_archive(
                bank_manifest,
                full_tensor_verify=False,
                expected_normalizer_artifact_sha256=normalizers[name].get("artifact_sha256"),
            )
            if (
                int(record.get("events", -1)) != int(bank.get("events", -2))
                or record.get("manifest_identity_sha256") != bank.get("manifest_identity_sha256")
            ):
                raise StrictDataError(f"Vector397 fingerprint bank {name} identity changed")
    return manifest


def load_fingerprint_normalizer(
    manifest_path: str | Path,
    scope: str,
    *,
    expected_source_archive_identity_sha256: str | None = None,
    expected_source_cache_identity_sha256: str | None = None,
    expected_event_count: int | None = None,
    expected_heldout_task: str | None = None,
) -> tuple[Vector397Normalizer, dict[str, Any], dict[str, Any]]:
    """Load one published scope and bind it to the caller's source cache.

    The manifest is checked before the selected normalizer is opened.  This
    keeps training configs from accidentally selecting a normalizer from a
    neighboring archive or from a different LOCO fold.
    """

    manifest_file = Path(manifest_path).resolve()
    manifest = read_fingerprint_manifest(manifest_file)
    if expected_source_archive_identity_sha256 is not None and manifest.get(
        "source_archive_identity_sha256"
    ) != expected_source_archive_identity_sha256:
        raise StrictDataError("fingerprint manifest source archive identity changed")
    if expected_source_cache_identity_sha256 is not None and manifest.get(
        "source_cache_identity_sha256"
    ) != expected_source_cache_identity_sha256:
        raise StrictDataError("fingerprint manifest source cache identity changed")
    if expected_event_count is not None and int(manifest.get("event_count", -1)) != int(
        expected_event_count
    ):
        raise StrictDataError("fingerprint manifest source event count changed")
    records = manifest["normalizers"]
    record = records.get(str(scope))
    if not isinstance(record, Mapping):
        raise StrictDataError(f"fingerprint normalizer scope is missing: {scope}")
    normalizer_path = _resolve(manifest_file.parent, str(record.get("path", "")))
    normalizer, metadata = read_vector397_normalizer(
        normalizer_path,
        expected_archive_identity_sha256=manifest.get("source_archive_identity_sha256"),
        expected_training_prefix_global_index_sha256=record.get("training_index_sha256"),
    )
    if (
        metadata.get("artifact_sha256") != record.get("artifact_sha256")
        or metadata.get("normalizer_sha256") != record.get("normalizer_sha256")
        or metadata.get("training_event_count") != record.get("training_event_count")
        or metadata.get("training_identity_sha256") != record.get("training_identity_sha256")
        or metadata.get("training_prefix_global_index_sha256") != record.get("training_index_sha256")
        or metadata.get("scope") != record.get("scope")
        or metadata.get("heldout_task") != record.get("heldout_task")
        or metadata.get("training_task_counts") != record.get("training_task_counts")
        or metadata.get("source_cache_identity_sha256")
        != manifest.get("source_cache_identity_sha256")
    ):
        raise StrictDataError(f"fingerprint normalizer scope identity changed: {scope}")
    if expected_heldout_task is not None and metadata.get("heldout_task") != expected_heldout_task:
        raise StrictDataError("fingerprint normalizer held-out task differs from the fold")
    if expected_heldout_task is None and metadata.get("heldout_task") is not None:
        raise StrictDataError("global fingerprint normalizer unexpectedly has a held-out task")
    return normalizer, metadata, manifest


def _write_fingerprint_shard(
    destination: Path,
    *,
    start: int,
    event_ids: np.ndarray,
    payload_ids: np.ndarray,
    vectors: np.ndarray,
    source_archive_identity_sha256: str,
    source_cache_identity_sha256: str,
    normalizer_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    count = int(vectors.shape[0])
    if vectors.dtype != np.float32 or vectors.shape != (count, VECTOR397_DIM):
        raise StrictDataError("fingerprint shard vector shape/dtype changed")
    tensors = {
        "global_index": torch.arange(start, start + count, dtype=torch.int64),
        "event_key": torch.from_numpy(np.ascontiguousarray(event_ids, dtype=np.uint8)),
        "payload_artifact_id": torch.from_numpy(np.ascontiguousarray(payload_ids, dtype=np.uint8)),
        FINGERPRINT_VECTOR_NAME: torch.from_numpy(np.ascontiguousarray(vectors, dtype=np.float32)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary_sidecar = _sidecar(temporary)
    save_file(tensors, str(temporary))
    tensor_sha = _file_sha256(temporary)
    event_hex = _hex_tensor(event_ids, "event_key")
    payload_hex = _hex_tensor(payload_ids, "payload_artifact_id")
    metadata = {
        "format": FINGERPRINT_ARCHIVE_FORMAT,
        "fingerprint_version": FINGERPRINT_VERSION,
        "descriptor_schema_sha256": VECTOR397_SCHEMA_SHA256,
        "source_archive_identity_sha256": source_archive_identity_sha256,
        "source_cache_identity_sha256": source_cache_identity_sha256,
        "normalizer_artifact_sha256": normalizer_metadata.get("artifact_sha256"),
        "normalizer_sha256": normalizer_metadata.get("normalizer_sha256"),
        "scope": normalizer_metadata.get("scope"),
        "heldout_task": normalizer_metadata.get("heldout_task"),
        "global_start": int(start),
        "global_end": int(start + count),
        "records": count,
        "event_keys_sha256": semantic_sha256(event_hex),
        "payload_artifact_ids_sha256": semantic_sha256(payload_hex),
        "tensor_sha256": tensor_sha,
    }
    temporary_sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    reopened = load_file(str(temporary), device="cpu")
    if set(reopened) != _FINGERPRINT_TENSOR_NAMES or any(not torch.equal(reopened[name], value) for name, value in tensors.items()):
        raise StrictDataError("fingerprint shard reopen parity failed")
    os.replace(temporary, destination)
    os.replace(temporary_sidecar, _sidecar(destination))
    return {
        **metadata,
        "path": destination.name,
        "sidecar": _sidecar(destination).name,
        "bytes": destination.stat().st_size + _sidecar(destination).stat().st_size,
        "sidecar_sha256": _file_sha256(_sidecar(destination)),
    }


def materialize_fingerprint_archive(
    source: FingerprintSource,
    *,
    normalizer_path: str | Path,
    output_dir: str | Path,
    batch_rows: int = 8192,
    shard_rows: int = 250_000,
) -> dict[str, Any]:
    """Materialize one normalized archive without touching the raw archive."""

    if int(shard_rows) <= 0:
        raise ValueError("shard_rows must be positive")
    normalizer_file = Path(normalizer_path).resolve()
    normalizer, normalizer_metadata = read_vector397_normalizer(
        normalizer_file,
        expected_archive_identity_sha256=source.archive_identity_sha256,
    )
    if normalizer_metadata.get("source_cache_identity_sha256") != source.cache_identity_sha256:
        raise StrictDataError("normalizer/cache identity differs")
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = read_fingerprint_archive(manifest_path, full_tensor_verify=True)
        if (
            existing.get("source_archive_identity_sha256") != source.archive_identity_sha256
            or existing.get("source_cache_identity_sha256") != source.cache_identity_sha256
            or existing.get("normalizer_artifact_sha256") != normalizer_metadata.get("artifact_sha256")
        ):
            raise StrictDataError("existing fingerprint archive is bound to different sources")
        return existing
    if output.exists() and any(not child.name.startswith(".") for child in output.iterdir()):
        raise FileExistsError(f"fingerprint archive has files but no complete manifest: {output}")
    output.mkdir(parents=True, exist_ok=True)
    shard_records: list[dict[str, Any]] = []
    vector_parts: list[np.ndarray] = []
    event_parts: list[np.ndarray] = []
    payload_parts: list[np.ndarray] = []
    buffered = 0
    next_start = 0

    def flush() -> None:
        nonlocal vector_parts, event_parts, payload_parts, buffered, next_start
        if buffered == 0:
            return
        vectors = np.concatenate(vector_parts, axis=0).astype(np.float32, copy=False)
        events = np.concatenate(event_parts, axis=0).astype(np.uint8, copy=False)
        payloads = np.concatenate(payload_parts, axis=0).astype(np.uint8, copy=False)
        index = len(shard_records)
        record = _write_fingerprint_shard(
            output / f"fingerprint_{index:05d}.safetensors",
            start=next_start,
            event_ids=events,
            payload_ids=payloads,
            vectors=vectors,
            source_archive_identity_sha256=source.archive_identity_sha256,
            source_cache_identity_sha256=source.cache_identity_sha256,
            normalizer_metadata=normalizer_metadata,
        )
        shard_records.append(record)
        next_start += buffered
        vector_parts, event_parts, payload_parts = [], [], []
        buffered = 0

    for indices, event_ids, payload_ids, _raw, fixed, _task_ids in iter_source_batches(
        source, batch_rows=batch_rows, verify_cache_parity=True
    ):
        normalized = normalizer.transform_fixed_log1p_batch(fixed).astype(np.float32)
        if not np.isfinite(normalized).all():
            raise StrictDataError("normalized fingerprint contains non-finite values")
        position = 0
        while position < len(indices):
            take = min(int(shard_rows) - buffered, len(indices) - position)
            vector_parts.append(normalized[position : position + take])
            event_parts.append(event_ids[position : position + take])
            payload_parts.append(payload_ids[position : position + take])
            buffered += take
            position += take
            if buffered == int(shard_rows):
                flush()
    flush()
    if next_start != source.event_count:
        raise StrictDataError("fingerprint archive coverage is incomplete")
    identity = {
        "format": FINGERPRINT_ARCHIVE_FORMAT,
        "fingerprint_version": FINGERPRINT_VERSION,
        "descriptor_schema_sha256": VECTOR397_SCHEMA_SHA256,
        "source_archive_identity_sha256": source.archive_identity_sha256,
        "source_cache_identity_sha256": source.cache_identity_sha256,
        "normalizer_artifact_sha256": normalizer_metadata.get("artifact_sha256"),
        "normalizer_sha256": normalizer_metadata.get("normalizer_sha256"),
        "scope": normalizer_metadata.get("scope"),
        "heldout_task": normalizer_metadata.get("heldout_task"),
        "events": int(source.event_count),
        "vector_dim": VECTOR397_DIM,
        "vector_dtype": "float32",
        "shards": [
            {
                key: record[key]
                for key in (
                    "records",
                    "bytes",
                    "global_start",
                    "global_end",
                    "tensor_sha256",
                    "sidecar_sha256",
                    "event_keys_sha256",
                    "payload_artifact_ids_sha256",
                )
            }
            for record in shard_records
        ],
    }
    manifest = {
        **identity,
        "status": "complete",
        "manifest_identity_sha256": semantic_sha256(identity),
        "shards": shard_records,
    }
    candidate = output / f".manifest.complete.{os.getpid()}.json"
    _atomic_json(candidate, manifest)
    read_fingerprint_archive(candidate, full_tensor_verify=True)
    os.replace(candidate, manifest_path)
    return manifest


def read_fingerprint_archive(
    path: str | Path,
    *,
    full_tensor_verify: bool = False,
    expected_normalizer_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Reopen a derived archive and fail closed on tensor/identity drift."""

    source = Path(path).resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictDataError("fingerprint archive manifest is unreadable") from error
    if not isinstance(manifest, dict) or manifest.get("format") != FINGERPRINT_ARCHIVE_FORMAT or manifest.get("status") != "complete":
        raise StrictDataError("fingerprint archive is incomplete or unsupported")
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"status", "manifest_identity_sha256", "shards"}
    }
    identity["shards"] = [
        {
            key: record[key]
            for key in (
                "records",
                "bytes",
                "global_start",
                "global_end",
                "tensor_sha256",
                "sidecar_sha256",
                "event_keys_sha256",
                "payload_artifact_ids_sha256",
            )
        }
        for record in manifest.get("shards", [])
    ]
    if semantic_sha256(identity) != manifest.get("manifest_identity_sha256"):
        raise StrictDataError("fingerprint archive manifest identity changed")
    if manifest.get("descriptor_schema_sha256") != VECTOR397_SCHEMA_SHA256 or int(manifest.get("vector_dim", -1)) != VECTOR397_DIM:
        raise StrictDataError("fingerprint archive schema changed")
    if expected_normalizer_artifact_sha256 is not None and manifest.get("normalizer_artifact_sha256") != expected_normalizer_artifact_sha256:
        raise StrictDataError("fingerprint archive normalizer identity changed")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise StrictDataError("fingerprint archive has no shards")
    cursor = 0
    for shard_number, record in enumerate(shards):
        if not isinstance(record, Mapping):
            raise StrictDataError(f"fingerprint shard {shard_number} is malformed")
        start = int(record.get("global_start", -1))
        end = int(record.get("global_end", -1))
        count = int(record.get("records", -1))
        if start != cursor or end - start != count or count <= 0:
            raise StrictDataError("fingerprint shard coverage is not contiguous")
        tensor_path = _resolve(source.parent, str(record.get("path", "")))
        sidecar_path = _resolve(source.parent, str(record.get("sidecar", "")))
        if _file_sha256(tensor_path) != record.get("tensor_sha256") or _file_sha256(sidecar_path) != record.get("sidecar_sha256"):
            raise StrictDataError("fingerprint shard checksum changed")
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            metadata.get("format") != FINGERPRINT_ARCHIVE_FORMAT
            or metadata.get("fingerprint_version") != FINGERPRINT_VERSION
            or metadata.get("descriptor_schema_sha256") != VECTOR397_SCHEMA_SHA256
            or metadata.get("source_archive_identity_sha256")
            != manifest.get("source_archive_identity_sha256")
            or metadata.get("source_cache_identity_sha256")
            != manifest.get("source_cache_identity_sha256")
            or metadata.get("tensor_sha256") != record.get("tensor_sha256")
            or int(metadata.get("global_start", -1)) != start
            or int(metadata.get("global_end", -1)) != end
            or metadata.get("normalizer_artifact_sha256") != manifest.get("normalizer_artifact_sha256")
            or metadata.get("scope") != manifest.get("scope")
            or metadata.get("heldout_task") != manifest.get("heldout_task")
        ):
            raise StrictDataError("fingerprint shard provenance changed")
        if full_tensor_verify:
            tensors = load_file(str(tensor_path), device="cpu")
            if set(tensors) != _FINGERPRINT_TENSOR_NAMES:
                raise StrictDataError("fingerprint shard tensor names changed")
            if (
                tensors["global_index"].dtype != torch.int64
                or tensors["global_index"].tolist() != list(range(start, end))
                or tensors["event_key"].dtype != torch.uint8
                or tuple(tensors["event_key"].shape) != (count, 32)
                or tensors["payload_artifact_id"].dtype != torch.uint8
                or tuple(tensors["payload_artifact_id"].shape) != (count, 32)
                or tensors[FINGERPRINT_VECTOR_NAME].dtype != torch.float32
                or tuple(tensors[FINGERPRINT_VECTOR_NAME].shape) != (count, VECTOR397_DIM)
                or not bool(torch.isfinite(tensors[FINGERPRINT_VECTOR_NAME]).all())
            ):
                raise StrictDataError("fingerprint shard tensor schema/values changed")
            if semantic_sha256(_hex_tensor(tensors["event_key"].numpy(), "event_key")) != record.get("event_keys_sha256"):
                raise StrictDataError("fingerprint shard event identity changed")
            if semantic_sha256(_hex_tensor(tensors["payload_artifact_id"].numpy(), "payload_artifact_id")) != record.get("payload_artifact_ids_sha256"):
                raise StrictDataError("fingerprint shard Payload identity changed")
        cursor = end
    if cursor != int(manifest.get("events", -1)):
        raise StrictDataError("fingerprint archive total coverage changed")
    return manifest


def iter_fingerprint_batches(
    manifest_path: str | Path,
    *,
    batch_rows: int = 8192,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield normalized archive batches as ``(indices,event,payload,vector)``."""

    manifest_file = Path(manifest_path).resolve()
    if int(batch_rows) <= 0:
        raise ValueError("batch_rows must be positive")
    manifest = read_fingerprint_archive(manifest_file, full_tensor_verify=False)
    for record in manifest["shards"]:
        tensor_path = _resolve(manifest_file.parent, str(record["path"]))
        with safe_open(str(tensor_path), framework="np", device="cpu") as handle:
            count = int(record["records"])
            for start in range(0, count, int(batch_rows)):
                end = min(count, start + int(batch_rows))
                yield (
                    np.asarray(handle.get_slice("global_index")[start:end], dtype=np.int64),
                    np.asarray(handle.get_slice("event_key")[start:end], dtype=np.uint8),
                    np.asarray(handle.get_slice("payload_artifact_id")[start:end], dtype=np.uint8),
                    np.asarray(handle.get_slice(FINGERPRINT_VECTOR_NAME)[start:end], dtype=np.float32),
                )


__all__ = [
    "ATOMIC_TASKS",
    "COMPOSITION_TASKS",
    "DEFAULT_ARCHIVE_IDENTITY_SHA256",
    "DEFAULT_CACHE_IDENTITY_SHA256",
    "DEFAULT_EVENT_COUNT",
    "DEFAULT_ATOMIC_ARCHIVE_IDENTITY_SHA256",
    "DEFAULT_ATOMIC_CACHE_IDENTITY_SHA256",
    "DEFAULT_ATOMIC_EVENT_COUNT",
    "FORMAL_SOURCE_CACHE_FORMAT",
    "FINGERPRINT_ARCHIVE_FORMAT",
    "FINGERPRINT_MANIFEST_FORMAT",
    "FINGERPRINT_VECTOR_NAME",
    "FingerprintSource",
    "fit_fingerprint_normalizers",
    "iter_fingerprint_batches",
    "iter_source_batches",
    "load_fingerprint_normalizer",
    "load_fingerprint_source",
    "materialize_fingerprint_archive",
    "read_fingerprint_archive",
    "read_fingerprint_manifest",
    "semantic_sha256",
    "write_fingerprint_manifest",
]

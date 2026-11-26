"""Validated Vector397 condition sources for HyperNet runs.

The formal token cache remains the authoritative source for query tokens,
labels, task ids, and Frozen-Base CE.  A normalized fingerprint bank is an
independent condition-only view.  Keeping the two views separate lets a new
Vector397 treatment be introduced without rebuilding (or silently changing)
the cache used by the Base-CE control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors import safe_open
from torch import Tensor

from .circuit_normalizer import Vector397Normalizer, read_vector397_normalizer
from .circuit_projection import retained_indices_from_dead_mask
from .data import VECTOR397_DIM, StrictDataError
from .vector397_fingerprint import (
    FINGERPRINT_ARCHIVE_FORMAT,
    FINGERPRINT_VECTOR_NAME,
    read_fingerprint_archive,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(parent: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _canonical_index_digest(prefix: int) -> str:
    """Digest used by the published fingerprint normalizer provenance."""

    values = np.arange(int(prefix), dtype=np.int64)
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


@dataclass(frozen=True)
class Vector397FingerprintSource:
    """A validated CPU tensor containing the canonical training prefix.

    The on-disk archive is always Vector397.  A compact route may expose the
    same validated rows as a 381-D in-memory view after the dead-coordinate
    projection; provenance and the retained raw-coordinate list remain tied
    to the original 397-D archive.
    """

    manifest_path: Path
    manifest: Mapping[str, Any]
    normalizer: Vector397Normalizer
    normalizer_metadata: Mapping[str, Any]
    vectors: Tensor
    identity: Mapping[str, Any]

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def condition_dim(self) -> int:
        """Width of the tensor returned by :meth:`index_select`."""

        if self.vectors.ndim != 2:
            raise ValueError("fingerprint source vectors must be a rank-2 tensor")
        return int(self.vectors.shape[1])

    @property
    def retained_indices(self) -> tuple[int, ...]:
        """Raw Vector397 coordinates retained by the compact condition view."""

        return retained_indices_from_dead_mask(self.normalizer.dead_mask)

    def index_select(self, indices: Tensor | list[int]) -> Tensor:
        """Return condition vectors in CPU memory for canonical row indices."""

        index = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if index.numel() == 0:
            raise IndexError("Vector397 fingerprint batch is empty")
        if bool(((index < 0) | (index >= self.size)).any()):
            raise IndexError("Vector397 fingerprint batch indices are invalid")
        return self.vectors.index_select(0, index)


def _read_shard_prefix(
    manifest_path: Path,
    record: Mapping[str, Any],
    *,
    take: int,
    expected_start: int,
) -> Tensor:
    """Read only the needed prefix from one safetensors shard."""

    start = int(record.get("global_start", -1))
    end = int(record.get("global_end", -1))
    count = int(record.get("records", -1))
    if start != int(expected_start) or end - start != count or count <= 0:
        raise StrictDataError("fingerprint source shard coverage changed")
    if not 0 < int(take) <= count:
        raise StrictDataError("fingerprint source shard prefix is invalid")
    tensor_path = _resolve(manifest_path.parent, str(record.get("path", "")))
    if not tensor_path.is_file():
        raise StrictDataError("fingerprint source shard is missing")
    try:
        with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
            expected_names = {
                "global_index",
                "event_key",
                "payload_artifact_id",
                FINGERPRINT_VECTOR_NAME,
            }
            if set(handle.keys()) != expected_names:
                raise StrictDataError("fingerprint source tensor names changed")
            index_slice = handle.get_slice("global_index")
            event_slice = handle.get_slice("event_key")
            payload_slice = handle.get_slice("payload_artifact_id")
            vector_slice = handle.get_slice(FINGERPRINT_VECTOR_NAME)
            if (
                index_slice.get_dtype() != "I64"
                or index_slice.get_shape() != [count]
                or event_slice.get_dtype() != "U8"
                or event_slice.get_shape() != [count, 32]
                or payload_slice.get_dtype() != "U8"
                or payload_slice.get_shape() != [count, 32]
                or vector_slice.get_dtype() != "F32"
                or vector_slice.get_shape() != [count, VECTOR397_DIM]
            ):
                raise StrictDataError("fingerprint source shard tensor schema changed")
            indices = index_slice[: int(take)]
            expected = torch.arange(start, start + int(take), dtype=torch.int64)
            if not torch.equal(indices, expected):
                raise StrictDataError("fingerprint source global index ordering changed")
            vectors = vector_slice[: int(take)]
    except StrictDataError:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        raise StrictDataError("fingerprint source shard is unreadable") from error
    if not isinstance(vectors, Tensor) or vectors.dtype != torch.float32:
        raise StrictDataError("fingerprint source vectors must be FP32")
    vectors = vectors.contiguous()
    if tuple(vectors.shape) != (int(take), VECTOR397_DIM) or not bool(
        torch.isfinite(vectors).all()
    ):
        raise StrictDataError("fingerprint source vectors are non-finite or malformed")
    return vectors


def load_vector397_fingerprint_source(
    *,
    manifest_path: str | Path,
    normalizer_path: str | Path,
    expected_prefix: int,
    expected_source_archive_identity_sha256: str,
    expected_source_cache_identity_sha256: str | None = None,
    compact381: bool = False,
) -> Vector397FingerprintSource:
    """Load a normalized fingerprint prefix with fail-closed provenance.

    ``expected_source_archive_identity_sha256`` must be the identity of the
    raw Vector397 archive referenced by the formal token cache.  This prevents
    an otherwise valid bank from being paired with a different query stream.
    The normalizer must have been fitted on exactly the canonical training
    prefix; a full composition bank normalizer is therefore rejected for the
    atomic SFT stage.  ``compact381=True`` projects the validated FP32 rows
    after all 397-D provenance checks; it never changes the on-disk archive.
    """

    manifest_file = Path(manifest_path).resolve()
    normalizer_file = Path(normalizer_path).resolve()
    if int(expected_prefix) <= 0:
        raise ValueError("expected_prefix must be positive")
    archive = read_fingerprint_archive(manifest_file, full_tensor_verify=False)
    if archive.get("format") != FINGERPRINT_ARCHIVE_FORMAT:
        raise StrictDataError("fingerprint source format changed")
    events = int(archive.get("events", -1))
    if events < int(expected_prefix):
        raise StrictDataError("fingerprint source is shorter than the formal prefix")
    source_archive_identity = str(archive.get("source_archive_identity_sha256", ""))
    if source_archive_identity != str(expected_source_archive_identity_sha256):
        raise StrictDataError("fingerprint source is bound to a different raw archive")
    source_cache_identity = str(archive.get("source_cache_identity_sha256", ""))
    if len(source_cache_identity) != 64 or any(
        character not in "0123456789abcdef" for character in source_cache_identity
    ):
        raise StrictDataError("fingerprint source cache identity is missing")
    if expected_source_cache_identity_sha256 is not None and source_cache_identity != str(
        expected_source_cache_identity_sha256
    ):
        raise StrictDataError("fingerprint source is bound to a different packed cache")

    normalizer, normalizer_metadata = read_vector397_normalizer(
        normalizer_file,
        expected_archive_identity_sha256=source_archive_identity,
    )
    expected_normalizer_identity = str(archive.get("normalizer_artifact_sha256", ""))
    if not expected_normalizer_identity or normalizer.artifact_sha256 != expected_normalizer_identity:
        raise StrictDataError("fingerprint source normalizer identity changed")
    if int(normalizer.training_event_count) != int(expected_prefix):
        raise StrictDataError("fingerprint normalizer was not fitted on the formal prefix")
    prefix_digest = str(normalizer.training_prefix_global_index_sha256 or "")
    if prefix_digest != _canonical_index_digest(int(expected_prefix)):
        raise StrictDataError("fingerprint normalizer training prefix is not canonical")
    if normalizer_metadata.get("source_cache_identity_sha256") != source_cache_identity:
        raise StrictDataError("fingerprint source cache provenance changed")

    parts: list[Tensor] = []
    cursor = 0
    for record in archive.get("shards", ()):
        if not isinstance(record, Mapping):
            raise StrictDataError("fingerprint source shard record is malformed")
        start = int(record.get("global_start", -1))
        end = int(record.get("global_end", -1))
        if start != cursor:
            raise StrictDataError("fingerprint source shard order changed")
        if start >= int(expected_prefix):
            break
        take = min(end, int(expected_prefix)) - start
        parts.append(
            _read_shard_prefix(
                manifest_file,
                record,
                take=take,
                expected_start=start,
            )
        )
        cursor += int(take)
        if cursor == int(expected_prefix):
            break
    if cursor != int(expected_prefix):
        raise StrictDataError("fingerprint source prefix coverage is incomplete")
    vectors = torch.cat(parts, dim=0).contiguous()
    if tuple(vectors.shape) != (int(expected_prefix), VECTOR397_DIM):
        raise StrictDataError("fingerprint source prefix shape changed")

    if compact381:
        retained_indices = retained_indices_from_dead_mask(normalizer.dead_mask)
        index = torch.as_tensor(retained_indices, dtype=torch.long)
        vectors = vectors.index_select(1, index).contiguous()

    manifest_identity = str(archive.get("manifest_identity_sha256", ""))
    identity = {
        "format": FINGERPRINT_ARCHIVE_FORMAT,
        "fingerprint_manifest_sha256": _sha256_file(manifest_file),
        "fingerprint_manifest_identity_sha256": manifest_identity,
        "fingerprint_normalizer_artifact_sha256": normalizer.artifact_sha256,
        "fingerprint_normalizer_sha256": normalizer.normalizer_sha256,
        "source_archive_identity_sha256": source_archive_identity,
        "source_cache_identity_sha256": source_cache_identity,
        "fingerprint_events": events,
        "fingerprint_prefix": int(expected_prefix),
        "fingerprint_scope": archive.get("scope"),
        "fingerprint_heldout_task": archive.get("heldout_task"),
        "condition_dim": int(vectors.shape[1]),
        "compact381": bool(compact381),
    }
    return Vector397FingerprintSource(
        manifest_path=manifest_file,
        manifest=archive,
        normalizer=normalizer,
        normalizer_metadata=normalizer_metadata,
        vectors=vectors,
        identity=identity,
    )


__all__ = [
    "Vector397FingerprintSource",
    "load_vector397_fingerprint_source",
]

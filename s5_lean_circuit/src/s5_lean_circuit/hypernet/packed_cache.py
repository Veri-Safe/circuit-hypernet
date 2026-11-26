"""Memory-mapped formal 1.2M token/Vector397 cache for S5 HyperNet."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ..protocol import ATOMIC_TASKS
from .circuit_normalizer import FixedLog1pVector397
from .config import FORMAL_PREFIX, PROTOCOL_ID
from .config import base_content_identity_matches
from .vector397_source import Vector397FingerprintSource
from .vector397_fingerprint import load_fingerprint_normalizer
from .data import (
    StrictDataError,
    _metadata_paths,
    _response_start_from_token_proto,
    _tokenizer_encode,
    iter_vector397_rows,
)


PACKED_CACHE_FORMAT = "s5_hypernet_formal_packed_cache_v1"
TASK_TO_ID = {task: index for index, task in enumerate(ATOMIC_TASKS)}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def packed_cache_identity(
    *,
    archive_manifest: str | Path,
    tokenizer_path: str | Path,
    prefix: int,
    n_ctx: int,
    metadata_root: str | Path | None = None,
) -> dict[str, Any]:
    archive = Path(archive_manifest).resolve()
    tokenizer = Path(tokenizer_path).resolve()
    manifest = json.loads(archive.read_text(encoding="utf-8"))
    transform = FixedLog1pVector397()
    metadata = _metadata_paths(
        archive,
        metadata_root=metadata_root,
        samples_path=None,
        supervision_path=None,
        event_join_path=None,
    )
    corpus_manifest = None
    if metadata.corpus_manifest is not None:
        corpus_manifest = json.loads(
            metadata.corpus_manifest.read_text(encoding="utf-8")
        )
    return {
        "format": PACKED_CACHE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "archive_manifest_sha256": _sha256_file(archive),
        "archive_identity_sha256": manifest.get("archive_identity_sha256"),
        "descriptor_schema_sha256": manifest.get("descriptor_schema_sha256"),
        "tokenizer_sha256": _sha256_file(tokenizer),
        "samples_sha256": _sha256_file(metadata.samples),
        "supervision_sha256": _sha256_file(metadata.supervision),
        "event_join_sha256": _sha256_file(metadata.event_join),
        "corpus_manifest_sha256": (
            None
            if metadata.corpus_manifest is None
            else _sha256_file(metadata.corpus_manifest)
        ),
        "corpus_manifest_identity_sha256": (
            None
            if corpus_manifest is None
            else corpus_manifest.get("manifest_identity_sha256")
        ),
        "renderer_version": (
            None if corpus_manifest is None else corpus_manifest.get("renderer_version")
        ),
        "input_treatment": transform.metadata(),
        "input_treatment_sha256": transform.artifact_sha256,
        "prefix": int(prefix),
        "n_ctx": int(n_ctx),
        "task_to_id": TASK_TO_ID,
        "full_sequence_labels": True,
    }


def _validate_payload(
    payload: Mapping[str, Any], *, prefix: int, n_ctx: int, check_values: bool = False
) -> None:
    expected = {
        "query_ids": (torch.int16, (prefix, n_ctx)),
        "query_lengths": (torch.int16, (prefix,)),
        "input_ids": (torch.int16, (prefix, n_ctx)),
        "input_lengths": (torch.int16, (prefix,)),
        "response_start": (torch.int16, (prefix,)),
        "vectors": (torch.float32, (prefix, 397)),
        "task_ids": (torch.uint8, (prefix,)),
    }
    if set(payload) != set(expected):
        raise StrictDataError("packed cache tensor schema changed")
    for name, (dtype, shape) in expected.items():
        value = payload[name]
        if not isinstance(value, Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
            raise StrictDataError(
                f"packed cache {name} must be {dtype}{list(shape)}"
            )
    if check_values and not bool(torch.isfinite(payload["vectors"]).all()):
        raise StrictDataError("packed cache vectors contain non-finite values")
    query_lengths = payload["query_lengths"].to(dtype=torch.long)
    input_lengths = payload["input_lengths"].to(dtype=torch.long)
    response_start = payload["response_start"].to(dtype=torch.long)
    if bool(((query_lengths <= 0) | (query_lengths > n_ctx)).any()):
        raise StrictDataError("packed cache query lengths are invalid")
    if bool(((input_lengths <= 1) | (input_lengths > n_ctx)).any()):
        raise StrictDataError("packed cache input lengths are invalid")
    if bool(((response_start <= 0) | (response_start >= input_lengths)).any()):
        raise StrictDataError("packed cache response boundaries are invalid")
    if bool((payload["task_ids"] >= len(ATOMIC_TASKS)).any()):
        raise StrictDataError("packed cache task ids are invalid")


def build_packed_cache(
    *,
    archive_manifest: str | Path,
    tokenizer_path: str | Path,
    cache_path: str | Path,
    prefix: int = FORMAL_PREFIX,
    n_ctx: int = 128,
    metadata_root: str | Path | None = None,
) -> dict[str, Any]:
    """Stream the canonical archive once and atomically publish a tensor cache."""

    if int(prefix) != FORMAL_PREFIX:
        raise ValueError("formal packed cache requires exactly 1.2M events")
    import sentencepiece as spm

    started = time.perf_counter()
    destination = Path(cache_path).resolve()
    sidecar = _sidecar(destination)
    identity = packed_cache_identity(
        archive_manifest=archive_manifest,
        tokenizer_path=tokenizer_path,
        prefix=prefix,
        n_ctx=n_ctx,
        metadata_root=metadata_root,
    )
    if destination.is_file() or sidecar.is_file():
        if not destination.is_file() or not sidecar.is_file():
            raise StrictDataError("packed cache payload/sidecar are incomplete")
        return validate_packed_cache(
            destination,
            expected_identity=identity,
        )

    tokenizer = spm.SentencePieceProcessor(model_file=str(Path(tokenizer_path).resolve()))
    transform = FixedLog1pVector397()
    query_ids = torch.zeros((prefix, n_ctx), dtype=torch.int16)
    query_lengths = torch.empty(prefix, dtype=torch.int16)
    input_ids = torch.zeros((prefix, n_ctx), dtype=torch.int16)
    input_lengths = torch.empty(prefix, dtype=torch.int16)
    response_start = torch.empty(prefix, dtype=torch.int16)
    vectors = torch.empty((prefix, 397), dtype=torch.float32)
    task_ids = torch.empty(prefix, dtype=torch.uint8)

    from ..circuit.hypernet_sft_corpus import regenerate_and_verify

    count = 0
    for row in iter_vector397_rows(
        archive_manifest,
        prefix=int(prefix),
        metadata_root=metadata_root,
    ):
        if row.canonical_global_index != count:
            raise StrictDataError("packed cache source order changed")
        if row.task_type not in TASK_TO_ID or row.seed is None or row.renderer_version is None:
            raise StrictDataError("packed cache row lacks formal task/renderer metadata")
        record = dict(row.identity)
        record.update(
            {
                "task_type": row.task_type,
                "seed": row.seed,
                "renderer_version": row.renderer_version,
            }
        )
        input_text, supervision_text = regenerate_and_verify(record)
        full_text = input_text + supervision_text
        query = _tokenizer_encode(tokenizer, input_text)
        full = _tokenizer_encode(tokenizer, full_text)
        boundary = _response_start_from_token_proto(
            tokenizer, full_text, input_text
        )
        if len(query) > n_ctx or len(full) > n_ctx:
            raise StrictDataError(
                f"packed cache row {count} exceeds n_ctx={n_ctx}"
            )
        query_ids[count, : len(query)] = torch.tensor(query, dtype=torch.int16)
        input_ids[count, : len(full)] = torch.tensor(full, dtype=torch.int16)
        query_lengths[count] = len(query)
        input_lengths[count] = len(full)
        response_start[count] = boundary
        vectors[count].copy_(torch.from_numpy(transform.transform_row(row)))
        task_ids[count] = TASK_TO_ID[str(row.task_type)]
        count += 1
    if count != prefix:
        raise StrictDataError(f"packed cache coverage ended at {count}")
    payload = {
        "query_ids": query_ids,
        "query_lengths": query_lengths,
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "response_start": response_start,
        "vectors": vectors,
        "task_ids": task_ids,
    }
    _validate_payload(payload, prefix=prefix, n_ctx=n_ctx, check_values=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    record = {
        **identity,
        "payload_sha256": _sha256_file(destination),
        "payload_bytes": destination.stat().st_size,
        "identity_sha256": _semantic_sha256(identity),
        "status": "complete",
        "build_wall_seconds": time.perf_counter() - started,
    }
    temporary_sidecar = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    temporary_sidecar.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_sidecar, sidecar)
    return validate_packed_cache(destination, expected_identity=identity)


def validate_packed_cache(
    cache_path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    verify_payload_checksum: bool = True,
) -> dict[str, Any]:
    source = Path(cache_path).resolve()
    sidecar = _sidecar(source)
    if not source.is_file() or not sidecar.is_file():
        raise StrictDataError("packed cache payload/sidecar are missing")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    identity = {
        key: value
        for key, value in record.items()
        if key
        not in {
            "payload_sha256",
            "payload_bytes",
            "identity_sha256",
            "status",
            "build_wall_seconds",
        }
    }
    if record.get("status") != "complete":
        raise StrictDataError("packed cache is not complete")
    if record.get("identity_sha256") != _semantic_sha256(identity):
        raise StrictDataError("packed cache identity checksum changed")
    if expected_identity is not None and dict(expected_identity) != identity:
        raise StrictDataError("packed cache identity differs from the run config")
    if verify_payload_checksum and record.get("payload_sha256") != _sha256_file(source):
        raise StrictDataError("packed cache payload checksum changed")
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    _validate_payload(payload, prefix=int(identity["prefix"]), n_ctx=int(identity["n_ctx"]))
    return record


class PackedTokenizedVector397Dataset:
    """Read-only mmap view shared by participating ranks on the same host."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        expected_identity: Mapping[str, Any] | None = None,
        verify_payload_checksum: bool = True,
        include_vector: bool = True,
        vector_source: Vector397FingerprintSource | None = None,
        fingerprint_manifest: str | Path | None = None,
        fingerprint_normalizer: str | Path | None = None,
        fingerprint_scope: str | None = None,
        fingerprint_normalizer_artifact_sha256: str | None = None,
    ) -> None:
        self.path = Path(cache_path).resolve()
        self.metadata = validate_packed_cache(
            self.path,
            expected_identity=expected_identity,
            verify_payload_checksum=verify_payload_checksum,
        )
        self.payload: Mapping[str, Tensor] = torch.load(
            self.path, map_location="cpu", weights_only=True, mmap=True
        )
        self.n_ctx = int(self.metadata["n_ctx"])
        self.size = int(self.metadata["prefix"])
        self.include_vector = bool(include_vector)
        if vector_source is not None and fingerprint_manifest is not None:
            raise StrictDataError(
                "a Vector397 source and direct normalizer cannot both be attached"
            )
        if vector_source is not None and not self.include_vector:
            raise StrictDataError(
                "a Vector397 fingerprint source cannot be attached to a query-only route"
            )
        if vector_source is not None and vector_source.size != self.size:
            raise StrictDataError(
                "Vector397 fingerprint source prefix differs from the packed cache"
            )
        self.vector_source = vector_source
        self.fingerprint_manifest = None
        self.fingerprint_scope = None
        self.fingerprint_normalizer = None
        self.fingerprint_normalizer_metadata: Mapping[str, Any] | None = None
        if (fingerprint_manifest is None) != (fingerprint_normalizer is None):
            raise StrictDataError(
                "fingerprint_manifest and fingerprint_normalizer must be supplied together"
            )
        if fingerprint_manifest is not None:
            if not self.include_vector:
                raise StrictDataError(
                    "a fingerprint normalizer cannot be attached to a query-only route"
                )
            scope = str(fingerprint_scope or "all_atomic_global_train")
            normalizer, metadata, manifest = load_fingerprint_normalizer(
                fingerprint_manifest,
                scope,
                expected_source_cache_identity_sha256=str(self.metadata["identity_sha256"]),
                expected_event_count=self.size,
                expected_heldout_task=None,
            )
            manifest_path = Path(fingerprint_manifest).resolve()
            record_path = (
                manifest_path.parent / str(manifest["normalizers"][scope]["path"])
            ).resolve()
            normalizer_path = Path(fingerprint_normalizer).resolve()
            if normalizer_path != record_path:
                raise StrictDataError(
                    "fingerprint_normalizer does not match the manifest scope"
                )
            if (
                fingerprint_normalizer_artifact_sha256 is not None
                and metadata.get("artifact_sha256")
                != fingerprint_normalizer_artifact_sha256
            ):
                raise StrictDataError("fingerprint normalizer artifact identity changed")
            self.fingerprint_manifest = manifest_path
            self.fingerprint_scope = scope
            self.fingerprint_normalizer = normalizer
            self.fingerprint_normalizer_metadata = dict(metadata)

    def __len__(self) -> int:
        return self.size

    def attach_condition_source(
        self, vector_source: Vector397FingerprintSource
    ) -> None:
        """Attach a validated condition bank after cache identity is known."""

        if not self.include_vector:
            raise StrictDataError(
                "a Vector397 fingerprint source cannot be attached to a query-only route"
            )
        if vector_source.size != self.size:
            raise StrictDataError(
                "Vector397 fingerprint source prefix differs from the packed cache"
            )
        if self.fingerprint_normalizer is not None:
            raise StrictDataError(
                "a Vector397 source cannot replace an attached direct normalizer"
            )
        self.vector_source = vector_source

    def condition_vectors(self, indices: Tensor | Sequence[int]) -> Tensor:
        """Return the route's condition vectors without changing cache data."""

        index = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if index.numel() == 0 or bool(((index < 0) | (index >= self.size)).any()):
            raise IndexError("packed cache vector indices are invalid")
        if self.vector_source is not None:
            return self.vector_source.index_select(index)
        vector = self.payload["vectors"].index_select(0, index)
        if self.fingerprint_normalizer is not None:
            normalized = self.fingerprint_normalizer.transform_fixed_log1p_batch(
                vector.numpy()
            ).astype(np.float32, copy=False)
            vector = torch.from_numpy(np.ascontiguousarray(normalized))
        return vector

    def collate_indices(
        self,
        indices: Tensor | Sequence[int],
        *,
        device: torch.device | str,
        frozen_base_ce: Tensor,
    ) -> dict[str, Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if index.numel() == 0 or bool(((index < 0) | (index >= self.size)).any()):
            raise IndexError("packed cache batch indices are invalid")
        query_lengths = self.payload["query_lengths"].index_select(0, index).long()
        input_lengths = self.payload["input_lengths"].index_select(0, index).long()
        query_width = int(query_lengths.max().item())
        input_width = int(input_lengths.max().item())
        query = self.payload["query_ids"].index_select(0, index)[:, :query_width].long()
        full = self.payload["input_ids"].index_select(0, index)[:, :input_width].long()
        query_positions = torch.arange(query_width)[None, :]
        full_positions = torch.arange(input_width)[None, :]
        query_mask = query_positions < query_lengths[:, None]
        input_mask = full_positions < input_lengths[:, None]
        labels = full.clone()
        labels.masked_fill_(~input_mask, -100)
        base_ce = frozen_base_ce.index_select(0, index).float()
        if not bool(torch.isfinite(base_ce).all()):
            raise StrictDataError("Frozen Base CE lookup is non-finite")
        vector = (
            self.condition_vectors(index)
            if self.include_vector
            else torch.zeros((index.numel(), 397), dtype=torch.float32)
        )
        return {
            "global_indices": index.to(device),
            "query_ids": query.to(device, non_blocking=True),
            "query_mask": query_mask.to(device, non_blocking=True),
            "input_ids": full.to(device, non_blocking=True),
            "input_mask": input_mask.to(device, non_blocking=True),
            "labels": labels.to(device, non_blocking=True),
            "vector397": vector.to(device, non_blocking=True),
            "task_ids": self.payload["task_ids"].index_select(0, index).to(
                device, non_blocking=True
            ),
            "frozen_base_ce": base_ce.to(device, non_blocking=True),
        }

    def collate_base_inputs(
        self,
        indices: Tensor | Sequence[int],
        *,
        device: torch.device | str,
    ) -> dict[str, Tensor]:
        index = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if index.numel() == 0 or bool(((index < 0) | (index >= self.size)).any()):
            raise IndexError("packed cache Base-CE indices are invalid")
        lengths = self.payload["input_lengths"].index_select(0, index).long()
        width = int(lengths.max().item())
        values = self.payload["input_ids"].index_select(0, index)[:, :width].long()
        mask = torch.arange(width)[None, :] < lengths[:, None]
        labels = values.clone()
        labels.masked_fill_(~mask, -100)
        return {
            "global_indices": index.to(device),
            "input_ids": values.to(device, non_blocking=True),
            "labels": labels.to(device, non_blocking=True),
        }


def load_base_ce_artifact(
    path: str | Path,
    *,
    expected_count: int,
    expected_cache_identity_sha256: str,
    expected_cache_payload_sha256: str,
    expected_base_identity: Mapping[str, Any] | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    source = Path(path).resolve()
    sidecar = _sidecar(source)
    if not source.is_file() or not sidecar.is_file():
        raise StrictDataError("Frozen Base CE artifact is missing")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("format") != "s5_hypernet_frozen_base_ce_v1":
        raise StrictDataError("Frozen Base CE artifact format changed")
    if metadata.get("status") != "complete":
        raise StrictDataError("Frozen Base CE artifact is not complete")
    if int(metadata.get("event_count", -1)) != int(expected_count):
        raise StrictDataError("Frozen Base CE event count changed")
    if metadata.get("packed_cache_identity_sha256") != expected_cache_identity_sha256:
        raise StrictDataError("Frozen Base CE cache identity changed")
    if metadata.get("packed_cache_payload_sha256") != expected_cache_payload_sha256:
        raise StrictDataError("Frozen Base CE packed payload changed")
    if expected_base_identity is not None and not base_content_identity_matches(
        dict(metadata.get("base_identity", {})), expected_base_identity
    ):
        raise StrictDataError("Frozen Base CE Base identity changed")
    if metadata.get("payload_sha256") != _sha256_file(source):
        raise StrictDataError("Frozen Base CE payload checksum changed")
    value = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(value, Tensor) or value.dtype != torch.float32 or value.shape != (expected_count,):
        raise StrictDataError("Frozen Base CE payload must be float32[N]")
    if not bool(torch.isfinite(value).all()):
        raise StrictDataError("Frozen Base CE payload contains non-finite values")
    return value, metadata


__all__ = [
    "PACKED_CACHE_FORMAT",
    "PackedTokenizedVector397Dataset",
    "build_packed_cache",
    "load_base_ce_artifact",
    "packed_cache_identity",
    "validate_packed_cache",
]

"""Fail-closed, lazy data access for the HyperNet Vector397 condition set.

The metadata join is deliberately separate from the Vector397 tensor archive.
The seven identity fields below are required in every metadata row and are
checked against the identity tensors in each safetensors shard.  A shard is
opened only while its rows are being consumed; no complete Vector397 matrix is
retained by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from safetensors import safe_open


VECTOR397_DIM = 397
VECTOR397_SCHEMA_SHA256 = (
    "e6c5528d1a06d2979babe776ff22e28ef08e40cf9a2fd9a5d69751418b337bda"
)
IDENTITY_FIELDS = (
    "canonical_global_index",
    "prompt_key",
    "event_key",
    "input_text_sha256",
    "supervision_text_sha256",
    "payload_artifact_id",
    "descriptor_schema_sha256",
)
SHA256_FIELDS = frozenset(IDENTITY_FIELDS[1:])
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TENSOR_NAMES = frozenset(
    {"global_index", "event_key", "payload_artifact_id", "descriptor_raw"}
)
_VECTOR397_BLOCK_OFFSETS = {
    "global": [0, 30],
    "layer_nodes": [30, 142],
    "layer_flow": [142, 304],
    "topology": [304, 355],
    "position": [355, 397],
}


class StrictDataError(ValueError):
    """Raised when a dataset identity or canonical-order invariant is broken."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def semantic_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )


class _SemanticListDigest:
    """Streaming equivalent of ``semantic_sha256(list_of_strings)``."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self._digest.update(b"[")
        self._first = True

    def add(self, value: str) -> None:
        if not self._first:
            self._digest.update(b",")
        self._digest.update(json.dumps(str(value), ensure_ascii=False).encode("utf-8"))
        self._first = False

    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(b"]")
        return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _resolve_metadata_root(manifest_path: Path, value: str | Path) -> Path:
    """Resolve an explicit metadata root from the caller's workspace first."""

    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    workspace_path = path.resolve()
    if workspace_path.exists():
        return workspace_path
    return (manifest_path.parent / path).resolve()


def _metadata_path(root: Path, shard: Mapping[str, Any]) -> Path:
    """Resolve metadata only for legacy archives that published per-shard rows."""

    value = shard.get("metadata_path", shard.get("rows_path"))
    if value is None:
        raise StrictDataError("legacy Vector397 shard is missing metadata_path")
    return _resolve(root, str(value))


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Read legacy per-shard metadata without relaxing JSONL validation."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise StrictDataError(f"blank metadata line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise StrictDataError(
                    f"invalid metadata JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise StrictDataError(
                    f"metadata row is not an object at {path}:{line_number}"
                )
            yield value


def _as_sha256(value: Any, field: str) -> str:
    text = str(value)
    if not _SHA256_RE.fullmatch(text):
        raise StrictDataError(f"{field} is not a lowercase SHA256")
    return text


def _text_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise StrictDataError(f"{field} text is missing or not a string")
    return _sha256_bytes(value.encode("utf-8"))


def validate_identity_row(
    row: Mapping[str, Any], *, expected_index: int | None = None
) -> dict[str, Any]:
    """Validate and return the seven-field canonical identity of one row.

    Optional ``input_text`` and ``supervision_text`` values are verified when
    present.  They are not copied into Vector397 artifacts.
    """

    missing = [field for field in IDENTITY_FIELDS if field not in row]
    if missing:
        raise StrictDataError(f"row is missing required identity fields: {missing}")
    try:
        index = int(row["canonical_global_index"])
    except (TypeError, ValueError) as error:
        raise StrictDataError("canonical_global_index is not an integer") from error
    if (
        isinstance(row["canonical_global_index"], bool)
        or index < 0
        or index != row["canonical_global_index"]
    ):
        raise StrictDataError("canonical_global_index is not a canonical nonnegative integer")
    if expected_index is not None and index != int(expected_index):
        raise StrictDataError(
            f"canonical order changed: expected {expected_index}, got {index}"
        )
    identity = {field: row[field] for field in IDENTITY_FIELDS}
    identity["canonical_global_index"] = index
    for field in SHA256_FIELDS:
        identity[field] = _as_sha256(identity[field], field)
    if "input_text" in row and _text_sha256(row["input_text"], "input_text") != identity["input_text_sha256"]:
        raise StrictDataError("input_text_sha256 mismatch")
    if "supervision_text" in row and _text_sha256(row["supervision_text"], "supervision_text") != identity["supervision_text_sha256"]:
        raise StrictDataError("supervision_text_sha256 mismatch")
    if "split" in row and not isinstance(row["split"], str):
        raise StrictDataError("split must be a string when present")
    return identity


def _bytes_to_hex(array: np.ndarray) -> list[str]:
    value = np.asarray(array)
    if value.dtype != np.uint8 or value.ndim != 2 or value.shape[1] != 32:
        raise StrictDataError("identity tensor must be uint8[N,32]")
    return [bytes(row.tolist()).hex() for row in value]


@dataclass(frozen=True)
class Vector397Row:
    """One verified metadata/vector join row."""

    identity: Mapping[str, Any]
    vector: np.ndarray
    split: str | None = None
    task_type: str | None = None
    seed: int | None = None
    renderer_version: str | None = None

    def __post_init__(self) -> None:
        identity = validate_identity_row(self.identity)
        vector = np.asarray(self.vector)
        if vector.dtype != np.float32 or vector.shape != (VECTOR397_DIM,):
            raise StrictDataError("Vector397 row vector must be float32[397]")
        if not np.isfinite(vector).all():
            raise StrictDataError("Vector397 row vector is non-finite")
        if self.split is not None and not isinstance(self.split, str):
            raise StrictDataError("Vector397 row split must be a string")
        if self.task_type is not None and not isinstance(self.task_type, str):
            raise StrictDataError("Vector397 row task_type must be a string")
        if self.seed is not None and isinstance(self.seed, bool):
            raise StrictDataError("Vector397 row seed must be an integer")
        if self.renderer_version is not None and not isinstance(self.renderer_version, str):
            raise StrictDataError("Vector397 row renderer_version must be a string")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "vector", np.ascontiguousarray(vector))

    @property
    def canonical_global_index(self) -> int:
        return int(self.identity["canonical_global_index"])

    def as_dict(self) -> dict[str, Any]:
        result = dict(self.identity)
        if self.split is not None:
            result["split"] = self.split
        if self.task_type is not None:
            result["task_type"] = self.task_type
        if self.seed is not None:
            result["seed"] = self.seed
        if self.renderer_version is not None:
            result["renderer_version"] = self.renderer_version
        return result


@dataclass(frozen=True)
class _MetadataPaths:
    samples: Path
    supervision: Path
    event_join: Path
    corpus_manifest: Path | None = None


def _metadata_paths(
    manifest_path: Path,
    *,
    metadata_root: str | Path | None,
    samples_path: str | Path | None,
    supervision_path: str | Path | None,
    event_join_path: str | Path | None,
) -> _MetadataPaths:
    """Resolve the canonical three-file metadata join for a Vector397 archive.

    The published Vector397 archive intentionally contains tensors and shard
    sidecars only.  Its query/supervision/event metadata lives beside the
    archive in the SFT run directory.  Explicit paths are supported for
    callers that stage these assets elsewhere; otherwise the nearest ancestor
    containing ``corpus/samples.jsonl`` is used.
    """

    explicit = (samples_path, supervision_path, event_join_path)
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit):
            raise StrictDataError(
                "samples_path, supervision_path, and event_join_path must be "
                "provided together"
            )
        base = (
            _resolve_metadata_root(manifest_path, metadata_root)
            if metadata_root is not None
            else manifest_path.parent
        )
        paths = _MetadataPaths(
            samples=_resolve(base, str(samples_path)),
            supervision=_resolve(base, str(supervision_path)),
            event_join=_resolve(base, str(event_join_path)),
            corpus_manifest=(
                _resolve(base, str(samples_path)).parent / "manifest.json"
                if (_resolve(base, str(samples_path)).parent / "manifest.json").is_file()
                and _resolve(base, str(supervision_path)).parent
                == _resolve(base, str(samples_path)).parent
                else None
            ),
        )
    else:
        candidates: list[Path] = []
        if metadata_root is not None:
            candidates.append(_resolve_metadata_root(manifest_path, metadata_root))
        else:
            candidates.extend((manifest_path.parent, *manifest_path.parents))
        paths = None
        for candidate in candidates:
            corpus = candidate / "corpus"
            if (corpus / "samples.jsonl").is_file() and (
                corpus / "supervision.jsonl"
            ).is_file() and (candidate / "analysis_intermediate" / "event_join.jsonl").is_file():
                paths = _MetadataPaths(
                    samples=corpus / "samples.jsonl",
                    supervision=corpus / "supervision.jsonl",
                    event_join=candidate / "analysis_intermediate" / "event_join.jsonl",
                    corpus_manifest=(corpus / "manifest.json")
                    if (corpus / "manifest.json").is_file()
                    else None,
                )
                break
            # Also accept a metadata root that is itself the corpus directory.
            if (
                (candidate / "samples.jsonl").is_file()
                and (candidate / "supervision.jsonl").is_file()
                and (candidate.parent / "analysis_intermediate" / "event_join.jsonl").is_file()
            ):
                paths = _MetadataPaths(
                    samples=candidate / "samples.jsonl",
                    supervision=candidate / "supervision.jsonl",
                    event_join=candidate.parent / "analysis_intermediate" / "event_join.jsonl",
                    corpus_manifest=(candidate / "manifest.json")
                    if (candidate / "manifest.json").is_file()
                    else None,
                )
                break
        if paths is None:
            raise StrictDataError(
                "cannot discover Vector397 metadata: expected corpus/samples.jsonl, "
                "corpus/supervision.jsonl, and analysis_intermediate/event_join.jsonl"
            )

    for name, path in (
        ("samples", paths.samples),
        ("supervision", paths.supervision),
        ("event_join", paths.event_join),
    ):
        if not path.is_file():
            raise StrictDataError(f"Vector397 {name} metadata is missing: {path}")
    return paths


def _metadata_rows(
    path: Path,
    *,
    name: str,
    index_field: str,
) -> Iterator[dict[str, Any]]:
    """Yield metadata rows while enforcing their canonical index sequence."""

    with path.open("r", encoding="utf-8") as handle:
        expected_index = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise StrictDataError(f"blank {name} metadata line at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise StrictDataError(
                    f"invalid {name} metadata JSON at {path}:{line_number}"
                ) from error
            if not isinstance(value, dict):
                raise StrictDataError(
                    f"{name} metadata row is not an object at {path}:{line_number}"
                )
            try:
                raw_index = value[index_field]
                index = int(raw_index)
            except (KeyError, TypeError, ValueError) as error:
                raise StrictDataError(
                    f"{name} metadata row is missing integer {index_field}"
                ) from error
            if (
                isinstance(raw_index, bool)
                or index != raw_index
                or index < 0
                or index != expected_index
            ):
                raise StrictDataError(
                    f"{name} metadata canonical order changed at row {expected_index}"
                )
            expected_index += 1
            yield value


def _next_metadata_row(
    iterator: Iterator[dict[str, Any]], *, name: str, index: int
) -> dict[str, Any]:
    try:
        return next(iterator)
    except StopIteration as error:
        raise StrictDataError(f"{name} metadata is missing canonical index {index}") from error


def _verify_corpus_manifest(paths: _MetadataPaths, *, expected: int) -> None:
    if paths.corpus_manifest is None:
        return
    try:
        manifest = json.loads(paths.corpus_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictDataError(
            f"cannot read HyperNet corpus manifest: {paths.corpus_manifest}"
        ) from error
    if manifest.get("status") != "complete":
        raise StrictDataError("HyperNet corpus manifest is not complete")
    identity = manifest.get("manifest_identity_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_identity_sha256"}
    if identity is not None and semantic_sha256(body) != _as_sha256(
        identity, "manifest_identity_sha256"
    ):
        raise StrictDataError("HyperNet corpus manifest identity changed")
    event_count = manifest.get("event_count")
    if isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < expected:
        raise StrictDataError("HyperNet corpus is shorter than the requested Vector397 prefix")
    for field, path in (
        ("samples_sha256", paths.samples),
        ("supervision_sha256", paths.supervision),
    ):
        expected_sha = manifest.get(field)
        if expected_sha is not None and _file_sha256(path) != _as_sha256(expected_sha, field):
            raise StrictDataError(f"HyperNet corpus {field} mismatch")


def _iter_metadata_join(
    manifest_path: Path,
    *,
    limit: int,
    require_eof: bool,
    metadata_root: str | Path | None,
    samples_path: str | Path | None,
    supervision_path: str | Path | None,
    event_join_path: str | Path | None,
) -> Iterator[dict[str, Any]]:
    paths = _metadata_paths(
        manifest_path,
        metadata_root=metadata_root,
        samples_path=samples_path,
        supervision_path=supervision_path,
        event_join_path=event_join_path,
    )
    _verify_corpus_manifest(paths, expected=limit)
    samples = _metadata_rows(
        paths.samples, name="samples", index_field="canonical_global_index"
    )
    supervision = _metadata_rows(
        paths.supervision, name="supervision", index_field="canonical_global_index"
    )
    events = _metadata_rows(
        paths.event_join, name="event_join", index_field="global_phase_index"
    )
    seen_prompts: set[str] = set()
    seen_events: set[str] = set()
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise StrictDataError(
            "frozen HyperNet corpus regeneration requires PYTHONHASHSEED=42"
        )
    try:
        from ..circuit.hypernet_sft_corpus import prompt_key, regenerate_and_verify
    except ImportError as error:
        raise StrictDataError("cannot import the frozen HyperNet corpus renderer") from error
    for index in range(limit):
        sample = _next_metadata_row(samples, name="samples", index=index)
        target = _next_metadata_row(supervision, name="supervision", index=index)
        event = _next_metadata_row(events, name="event_join", index=index)
        if sample.get("canonical_global_index") != index or target.get(
            "canonical_global_index"
        ) != index or event.get("global_phase_index") != index:
            raise StrictDataError("metadata canonical index join changed")
        prompt = sample.get("prompt_key")
        if not isinstance(prompt, str) or not _SHA256_RE.fullmatch(prompt):
            raise StrictDataError("sample prompt_key is not a lowercase SHA256")
        if prompt in seen_prompts:
            raise StrictDataError(f"duplicate prompt_key {prompt}")
        seen_prompts.add(prompt)
        if target.get("prompt_key") != prompt or event.get("prompt_key") != prompt:
            raise StrictDataError(f"prompt_key join mismatch at canonical index {index}")
        try:
            sample_seed = int(sample["seed"])
            event_seed = int(event["seed"])
        except (KeyError, TypeError, ValueError) as error:
            raise StrictDataError(f"seed is invalid at canonical index {index}") from error
        if sample.get("task_type") != event.get("task_type") or sample_seed != event_seed:
            raise StrictDataError(f"task/seed join mismatch at canonical index {index}")
        if target.get("supervision_text_sha256") != sample.get("supervision_text_sha256"):
            raise StrictDataError(f"supervision hash join mismatch at canonical index {index}")
        supervision_text = target.get("supervision_text")
        if _text_sha256(supervision_text, "supervision_text") != sample.get(
            "supervision_text_sha256"
        ):
            raise StrictDataError(f"supervision_text_sha256 mismatch at canonical index {index}")
        event_key = event.get("event_key")
        if not isinstance(event_key, str) or not _SHA256_RE.fullmatch(event_key):
            raise StrictDataError(f"event_key is invalid at canonical index {index}")
        if event_key in seen_events:
            raise StrictDataError(f"duplicate event_key {event_key}")
        seen_events.add(event_key)
        # The renderer is the only source of input text.  This verifies the
        # input hash and the prompt identity without making query text part of
        # the condition artifact.
        try:
            regenerate_and_verify(sample)
            expected_prompt = prompt_key(
                task_type=str(sample["task_type"]),
                seed=int(sample["seed"]),
                renderer_version=str(sample["renderer_version"]),
                input_text_sha256=str(sample["input_text_sha256"]),
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            raise StrictDataError(
                f"frozen renderer identity mismatch at canonical index {index}"
            ) from error
        if expected_prompt != prompt:
            raise StrictDataError(f"prompt_key identity mismatch at canonical index {index}")
        # Do not retain canonical supervision text in the condition row.  Its
        # hash has been checked above; the text is consumed only by response
        # loss in the training pipeline.
        row = dict(sample)
        row.update({"canonical_global_index": index, "event_key": event_key})
        yield row
    if require_eof:
        for iterator, name in ((samples, "samples"), (supervision, "supervision"), (events, "event_join")):
            try:
                next(iterator)
            except StopIteration:
                continue
            raise StrictDataError(f"{name} metadata has rows beyond canonical index {limit - 1}")


def _validate_shard_sidecar(
    *,
    sidecar_path: Path,
    shard: Mapping[str, Any],
    manifest: Mapping[str, Any],
    shard_number: int,
) -> None:
    if not sidecar_path.is_file():
        raise StrictDataError(f"shard {shard_number} sidecar is missing")
    if shard.get("sidecar_sha256") is not None and _file_sha256(sidecar_path) != _as_sha256(
        shard["sidecar_sha256"], "sidecar_sha256"
    ):
        raise StrictDataError(f"shard {shard_number} sidecar checksum mismatch")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictDataError(f"cannot read shard {shard_number} sidecar") from error
    if sidecar.get("format") != manifest.get("format"):
        raise StrictDataError(f"shard {shard_number} sidecar format mismatch")
    if sidecar.get("source_identity_sha256") != manifest.get(
        "source_identity_sha256"
    ):
        raise StrictDataError(f"shard {shard_number} source identity mismatch")
    if sidecar.get("descriptor_schema_sha256") != manifest["descriptor_schema_sha256"]:
        raise StrictDataError(f"shard {shard_number} schema identity mismatch")
    if int(sidecar.get("records", -1)) != int(shard.get("records", -2)):
        raise StrictDataError(f"shard {shard_number} sidecar records mismatch")
    if sidecar.get("tensor_sha256") != shard.get("tensor_sha256"):
        raise StrictDataError(f"shard {shard_number} sidecar tensor_sha256 mismatch")
    if "global_start" in shard and int(sidecar.get("global_start", -1)) != int(shard["global_start"]):
        raise StrictDataError(f"shard {shard_number} sidecar global_start mismatch")
    if "global_end" in shard and int(sidecar.get("global_end", -1)) != int(shard["global_end"]):
        raise StrictDataError(f"shard {shard_number} sidecar global_end mismatch")
    for field in ("event_keys_sha256", "payload_artifact_ids_sha256"):
        if field in shard and sidecar.get(field) != shard.get(field):
            raise StrictDataError(f"shard {shard_number} sidecar {field} mismatch")


def _validate_descriptor_schema(manifest: Mapping[str, Any]) -> None:
    schema = manifest.get("descriptor_schema")
    if not isinstance(schema, dict):
        raise StrictDataError("Vector397 manifest has no descriptor_schema object")
    if semantic_sha256(schema) != manifest["descriptor_schema_sha256"]:
        raise StrictDataError("Vector397 descriptor_schema identity mismatch")
    if schema.get("dimension") != VECTOR397_DIM or schema.get("dtype") != "float32":
        raise StrictDataError("Vector397 descriptor schema dimension/dtype changed")
    if schema.get("block_offsets") != _VECTOR397_BLOCK_OFFSETS:
        raise StrictDataError("Vector397 descriptor block offsets changed")
    if schema.get("block_order") != list(_VECTOR397_BLOCK_OFFSETS):
        raise StrictDataError("Vector397 descriptor block order changed")


def _validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], Path, int]:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StrictDataError(f"cannot read Vector397 manifest: {manifest_path}") from error
    if not isinstance(manifest, dict):
        raise StrictDataError("Vector397 manifest must be an object")
    if manifest.get("status") != "complete":
        raise StrictDataError("Vector397 manifest is not complete")
    _as_sha256(manifest.get("descriptor_schema_sha256"), "descriptor_schema_sha256")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise StrictDataError("Vector397 manifest has no shards")
    events = manifest.get("events", manifest.get("event_count"))
    if isinstance(events, bool) or not isinstance(events, int) or events <= 0:
        raise StrictDataError("Vector397 manifest event count is invalid")
    if "manifest_identity_sha256" in manifest:
        body = {key: value for key, value in manifest.items() if key != "manifest_identity_sha256"}
        if semantic_sha256(body) != _as_sha256(manifest["manifest_identity_sha256"], "manifest_identity_sha256"):
            raise StrictDataError("Vector397 manifest identity hash mismatch")
    for field in ("event_keys_sha256", "payload_artifact_ids_sha256"):
        if field in manifest:
            _as_sha256(manifest[field], field)
    cursor = 0
    for shard_number, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise StrictDataError(f"shard {shard_number} is not an object")
        if "global_start" in shard or "global_end" in shard or "records" in shard:
            try:
                start = int(shard["global_start"])
                end = int(shard["global_end"])
                records = int(shard["records"])
            except (KeyError, TypeError, ValueError) as error:
                raise StrictDataError(f"shard {shard_number} coverage is invalid") from error
            if start != cursor or end <= start or end - start != records:
                raise StrictDataError(f"shard {shard_number} coverage is not contiguous")
            cursor = end
    if cursor and cursor != events:
        raise StrictDataError("Vector397 shard coverage does not match events")
    if "archive_identity_sha256" in manifest:
        required_source = (
            "format",
            "descriptor_schema_sha256",
            "payload_collection_identity_sha256",
            "events",
            "event_keys_sha256",
            "payload_artifact_ids_sha256",
            "worker_artifact_inventory_sha256",
        )
        try:
            source_identity = {key: manifest[key] for key in required_source}
            worker_inventory = manifest["worker_artifacts"]
            shard_identity = [
                {
                    "records": int(row["records"]),
                    "bytes": int(row["bytes"]),
                    "global_start": int(row["global_start"]),
                    "global_end": int(row["global_end"]),
                    "tensor_sha256": str(row["tensor_sha256"]),
                    "sidecar_sha256": str(row["sidecar_sha256"]),
                    "event_keys_sha256": str(row["event_keys_sha256"]),
                    "payload_artifact_ids_sha256": str(
                        row["payload_artifact_ids_sha256"]
                    ),
                }
                for row in shards
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise StrictDataError("Vector397 archive identity fields are incomplete") from error
        if semantic_sha256(source_identity) != manifest.get("source_identity_sha256"):
            raise StrictDataError("Vector397 source identity changed")
        if semantic_sha256(worker_inventory) != manifest.get(
            "worker_artifact_inventory_sha256"
        ):
            raise StrictDataError("Vector397 worker inventory identity changed")
        publication_identity = {
            **source_identity,
            "worker_artifacts": worker_inventory,
            "shards": shard_identity,
        }
        if semantic_sha256(publication_identity) != _as_sha256(
            manifest["archive_identity_sha256"], "archive_identity_sha256"
        ):
            raise StrictDataError("Vector397 archive identity changed")
    return manifest, manifest_path.parent, events


def _check_digest(manifest: Mapping[str, Any], field: str, actual: str) -> None:
    expected = manifest.get(field)
    if expected is not None and actual != _as_sha256(expected, field):
        raise StrictDataError(f"{field} mismatch")


def iter_vector397_rows(
    manifest_path: str | Path,
    *,
    split: str | None = None,
    prefix: int | None = None,
    metadata_root: str | Path | None = None,
    samples_path: str | Path | None = None,
    supervision_path: str | Path | None = None,
    event_join_path: str | Path | None = None,
) -> Iterator[Vector397Row]:
    """Yield verified Vector397 rows in canonical order.

    Published archives use corpus-level metadata.  ``metadata_root`` or the
    three explicit paths must be supplied when the archive is detached from
    its originating run.  The old per-shard ``metadata_path``/``rows_path``
    format remains readable for API compatibility.
    """

    path = Path(manifest_path).resolve()
    manifest, root, events = _validate_manifest(path)
    if prefix is not None and (
        isinstance(prefix, bool) or not isinstance(prefix, int) or not 0 < prefix <= events
    ):
        raise StrictDataError("prefix must be within the complete Vector397 archive")
    limit = events if prefix is None else prefix
    if manifest["descriptor_schema_sha256"] != VECTOR397_SCHEMA_SHA256:
        raise StrictDataError("descriptor_schema_sha256 is not the frozen Vector397 schema")
    real_archive = any(
        isinstance(shard, dict)
        and shard.get("metadata_path", shard.get("rows_path")) is None
        for shard in manifest["shards"]
    )
    if "descriptor_schema" in manifest:
        _validate_descriptor_schema(manifest)
    elif real_archive:
        raise StrictDataError("Vector397 manifest omits descriptor_schema")
    if real_archive:
        metadata_iterator = _iter_metadata_join(
            path,
            limit=limit,
            require_eof=prefix is None,
            metadata_root=metadata_root,
            samples_path=samples_path,
            supervision_path=supervision_path,
            event_join_path=event_join_path,
        )
    else:
        metadata_iterator = None
    expected_index = 0
    seen_prompt: set[str] = set()
    seen_event: set[str] = set()
    seen_artifact: set[str] = set()
    all_digests = {
        field: _SemanticListDigest()
        for field in IDENTITY_FIELDS[1:]
        if field != "descriptor_schema_sha256"
    }
    for shard_number, raw_shard in enumerate(manifest["shards"]):
        if not isinstance(raw_shard, dict):
            raise StrictDataError(f"shard {shard_number} is not an object")
        tensor_path = _resolve(root, str(raw_shard.get("path", "")))
        if not tensor_path.is_file():
            raise StrictDataError(f"shard {shard_number} is incomplete")
        if raw_shard.get("tensor_sha256") is None:
            raise StrictDataError(f"shard {shard_number} is missing tensor_sha256")
        if _file_sha256(tensor_path) != _as_sha256(raw_shard["tensor_sha256"], "tensor_sha256"):
            raise StrictDataError(f"shard {shard_number} tensor checksum mismatch")
        if real_archive:
            sidecar_value = raw_shard.get("sidecar") or f"{tensor_path.name}.json"
            _validate_shard_sidecar(
                sidecar_path=_resolve(root, str(sidecar_value)),
                shard=raw_shard,
                manifest=manifest,
                shard_number=shard_number,
            )
            shard_start = int(raw_shard.get("global_start", -1))
            shard_end = int(raw_shard.get("global_end", -1))
            if shard_start != expected_index or shard_end - shard_start != int(raw_shard.get("records", -1)):
                raise StrictDataError(f"shard {shard_number} global coverage changed")
        else:
            metadata_path = _metadata_path(root, raw_shard)
            if not metadata_path.is_file():
                raise StrictDataError(f"shard {shard_number} is incomplete")
            if raw_shard.get("metadata_sha256") is not None and _file_sha256(metadata_path) != _as_sha256(raw_shard["metadata_sha256"], "metadata_sha256"):
                raise StrictDataError(f"shard {shard_number} metadata checksum mismatch")
        with safe_open(str(tensor_path), framework="np", device="cpu") as handle:
            if set(handle.keys()) != _TENSOR_NAMES:
                raise StrictDataError(f"shard {shard_number} tensor schema changed")
            index_shape = handle.get_slice("global_index").get_shape()
            event_shape = handle.get_slice("event_key").get_shape()
            artifact_shape = handle.get_slice("payload_artifact_id").get_shape()
            vector_shape = handle.get_slice("descriptor_raw").get_shape()
            if (
                handle.get_slice("global_index").get_dtype() != "I64"
                or len(index_shape) != 1
            ):
                raise StrictDataError("global_index tensor must be int64[N]")
            count = int(index_shape[0])
            if real_archive and count != int(raw_shard.get("records", -1)):
                raise StrictDataError(f"shard {shard_number} record count changed")
            if event_shape != [count, 32] or artifact_shape != [count, 32]:
                raise StrictDataError("Vector397 identity tensor shape mismatch")
            if (
                handle.get_slice("event_key").get_dtype() != "U8"
                or handle.get_slice("payload_artifact_id").get_dtype() != "U8"
            ):
                raise StrictDataError("Vector397 identity tensors must be uint8[N,32]")
            if (
                handle.get_slice("descriptor_raw").get_dtype() != "F32"
                or vector_shape != [count, VECTOR397_DIM]
            ):
                raise StrictDataError("descriptor_raw must be float32[N,397]")
            take = min(count, limit - expected_index)
            indices = handle.get_slice("global_index")[:take]
            event_ids = handle.get_slice("event_key")[:take]
            artifact_ids = handle.get_slice("payload_artifact_id")[:take]
            raw_vectors = handle.get_slice("descriptor_raw")[:take]
            if not np.isfinite(raw_vectors).all():
                raise StrictDataError("descriptor_raw contains non-finite values")
            tensor_events = _bytes_to_hex(event_ids)
            tensor_artifacts = _bytes_to_hex(artifact_ids)
            if real_archive:
                rows = [
                    _next_metadata_row(metadata_iterator, name="metadata join", index=expected_index + offset)
                    for offset in range(take)
                ]
            else:
                rows = []
                metadata_count = 0
                for row in _read_jsonl(metadata_path):
                    if metadata_count < take:
                        rows.append(row)
                    metadata_count += 1
                if metadata_count != count:
                    raise StrictDataError(f"shard {shard_number} metadata/tensor count mismatch")
            for local_index, row in enumerate(rows[:take]):
                if expected_index >= limit:
                    break
                if real_archive:
                    identity = dict(row)
                    identity.update(
                        {
                            "payload_artifact_id": tensor_artifacts[local_index],
                            "descriptor_schema_sha256": manifest["descriptor_schema_sha256"],
                        }
                    )
                else:
                    identity = validate_identity_row(row, expected_index=expected_index)
                if int(indices[local_index]) != expected_index:
                    raise StrictDataError("safetensors global_index is missing, duplicated, or out of order")
                if identity["descriptor_schema_sha256"] != manifest["descriptor_schema_sha256"]:
                    raise StrictDataError("descriptor_schema_sha256 mismatch")
                if identity["event_key"] != tensor_events[local_index] or identity["payload_artifact_id"] != tensor_artifacts[local_index]:
                    raise StrictDataError("Vector397 identity tensor does not match metadata")
                identity = validate_identity_row(identity, expected_index=expected_index)
                for field, seen in (("prompt_key", seen_prompt), ("event_key", seen_event), ("payload_artifact_id", seen_artifact)):
                    if identity[field] in seen:
                        raise StrictDataError(f"duplicate {field}")
                    seen.add(identity[field])
                for field, digest in all_digests.items():
                    digest.add(identity[field])
                yield Vector397Row(
                    identity=identity,
                    vector=np.array(raw_vectors[local_index], copy=True),
                    split=row.get("split"),
                    task_type=row.get("task_type"),
                    seed=(int(row["seed"]) if row.get("seed") is not None else None),
                    renderer_version=row.get("renderer_version"),
                )
                expected_index += 1
        if expected_index >= limit:
            break
    if metadata_iterator is not None and prefix is None:
        # _iter_metadata_join performs the EOF check when advanced once past
        # the expected prefix.  This catches an extra metadata row as well as
        # a tensor archive that ends at the declared event count.
        try:
            next(metadata_iterator)
        except StopIteration:
            pass
        else:
            raise StrictDataError("metadata join has rows beyond archive coverage")
    if expected_index != limit:
        raise StrictDataError(f"Vector397 coverage ended at {expected_index}, expected {limit}")
    if prefix is None:
        if expected_index != events:
            raise StrictDataError("Vector397 event count mismatch")
        for field, digest in all_digests.items():
            digest_names = {
                "prompt_key": ("prompt_keys_sha256", "prompt_key_sha256"),
                "event_key": ("event_keys_sha256", "event_key_sha256"),
                "input_text_sha256": ("input_text_sha256s_sha256", "input_text_sha256"),
                "supervision_text_sha256": (
                    "supervision_text_sha256s_sha256",
                    "supervision_text_sha256",
                ),
                "payload_artifact_id": (
                    "payload_artifact_ids_sha256",
                    "payload_artifact_id_sha256",
                ),
            }
            for digest_name in digest_names[field]:
                if digest_name in manifest:
                    _check_digest(manifest, digest_name, digest.hexdigest())
                    break


def audit_vector397_prefix(
    manifest_path: str | Path,
    prefix: int,
    *,
    metadata_root: str | Path | None = None,
    samples_path: str | Path | None = None,
    supervision_path: str | Path | None = None,
    event_join_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit exactly rows ``0..prefix-1`` without materializing later shards."""

    count = 0
    schema_sha: str | None = None
    prompt_digest = _SemanticListDigest()
    event_digest = _SemanticListDigest()
    payload_digest = _SemanticListDigest()
    for row in iter_vector397_rows(
        manifest_path,
        prefix=int(prefix),
        metadata_root=metadata_root,
        samples_path=samples_path,
        supervision_path=supervision_path,
        event_join_path=event_join_path,
    ):
        if row.canonical_global_index != count:
            raise StrictDataError("Vector397 prefix is not canonical")
        schema_sha = str(row.identity["descriptor_schema_sha256"])
        prompt_digest.add(str(row.identity["prompt_key"]))
        event_digest.add(str(row.identity["event_key"]))
        payload_digest.add(str(row.identity["payload_artifact_id"]))
        count += 1
    if count != int(prefix):
        raise StrictDataError(f"Vector397 prefix coverage ended at {count}")
    return {
        "status": "complete",
        "prefix": int(prefix),
        "descriptor_schema_sha256": schema_sha,
        "prompt_keys_sha256": prompt_digest.hexdigest(),
        "event_keys_sha256": event_digest.hexdigest(),
        "payload_artifact_ids_sha256": payload_digest.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Tokenized training view
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TokenizedS5Row:
    """One strict query/supervision/circuit training example.

    The condition side contains only ``query_ids`` and the transformed circuit
    vector.  ``labels`` supervise every non-padding token in ``input_ids``; the
    renderer text itself is deliberately not retained in this object.
    """

    identity: Mapping[str, Any]
    query_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    vector: np.ndarray
    response_start: int
    task_type: str

    def __post_init__(self) -> None:
        identity = validate_identity_row(self.identity)
        query_ids = tuple(int(value) for value in self.query_ids)
        input_ids = tuple(int(value) for value in self.input_ids)
        labels = tuple(int(value) for value in self.labels)
        if not query_ids or not input_ids:
            raise StrictDataError("tokenized S5 rows must contain query and input tokens")
        if len(input_ids) != len(labels):
            raise StrictDataError("tokenized input_ids and labels are misaligned")
        if not 0 < int(self.response_start) < len(input_ids):
            raise StrictDataError("response_start must leave prompt and response tokens")
        if any(value < 0 or value >= 2048 for value in query_ids + input_ids):
            raise StrictDataError("tokenized S5 vocabulary id is outside [0,2048)")
        if labels != input_ids:
            raise StrictDataError("full-sequence labels must equal every input token id")
        vector = np.asarray(self.vector)
        if vector.dtype != np.float32 or vector.shape != (VECTOR397_DIM,):
            raise StrictDataError("transformed Vector397 must be float32[397]")
        if not np.isfinite(vector).all():
            raise StrictDataError("transformed Vector397 contains non-finite values")
        if not isinstance(self.task_type, str) or not self.task_type:
            raise StrictDataError("tokenized S5 task_type is missing")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "query_ids", query_ids)
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "vector", np.ascontiguousarray(vector))
        object.__setattr__(self, "response_start", int(self.response_start))

    @property
    def canonical_global_index(self) -> int:
        """Return the canonical archive index carried by ``identity``."""

        return int(self.identity["canonical_global_index"])


def _tokenizer_encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        values = tokenizer.encode(text)
    except TypeError:
        values = tokenizer.encode(text, add_bos=False, add_eos=False)
    values = tuple(int(value) for value in values)
    if not values:
        raise StrictDataError("S5 tokenizer returned an empty sequence")
    return values


def _response_start_from_token_proto(tokenizer: Any, full_text: str, input_text: str) -> int:
    """Find the first token touching the response character boundary.

    SentencePiece may merge the final prompt characters with the first
    response piece, so ``len(encode(input_text))`` is not a valid boundary.
    The immutable proto carries byte offsets and gives a deterministic,
    renderer-aligned response-only mask.
    """

    encoder = getattr(tokenizer, "encode_as_immutable_proto", None)
    if encoder is None:
        encoder = getattr(tokenizer, "EncodeAsImmutableProto", None)
    if encoder is None:
        # Small test tokenizers often do not expose offsets.  Their tokenizers
        # are expected to concatenate cleanly; fail closed if they do not.
        prompt = _tokenizer_encode(tokenizer, input_text)
        full = _tokenizer_encode(tokenizer, full_text)
        if tuple(full[: len(prompt)]) != prompt:
            raise StrictDataError(
                "tokenizer lacks offsets and does not preserve the query prefix"
            )
        return len(prompt)
    proto = encoder(full_text)
    boundary = len(input_text.encode("utf-8"))
    for index, piece in enumerate(proto.pieces):
        if int(piece.end) > boundary:
            return index
    raise StrictDataError("tokenizer produced no response-side token")


def tokenize_vector397_row(
    row: Vector397Row,
    *,
    tokenizer: Any,
    normalizer: Any,
    n_ctx: int = 128,
) -> TokenizedS5Row:
    """Regenerate and tokenize one verified row without leaking supervision."""

    if n_ctx < 2:
        raise ValueError("n_ctx must be at least two")
    from ..circuit.hypernet_sft_corpus import regenerate_and_verify

    render_record = dict(row.identity)
    if row.task_type is None or row.seed is None or row.renderer_version is None:
        raise StrictDataError(
            "Vector397 row is missing task/seed/renderer metadata required for tokenization"
        )
    render_record.update(
        {
            "task_type": row.task_type,
            "seed": row.seed,
            "renderer_version": row.renderer_version,
        }
    )
    input_text, supervision_text = regenerate_and_verify(render_record)
    full_text = input_text + supervision_text
    query_ids = _tokenizer_encode(tokenizer, input_text)
    input_ids = _tokenizer_encode(tokenizer, full_text)
    response_start = _response_start_from_token_proto(tokenizer, full_text, input_text)
    if len(query_ids) > n_ctx or len(input_ids) > n_ctx:
        raise StrictDataError(
            f"S5 tokenized row exceeds context {n_ctx}: "
            f"query={len(query_ids)} full={len(input_ids)} "
            f"index={row.canonical_global_index}"
        )
    if response_start <= 0 or response_start >= len(input_ids):
        raise StrictDataError("S5 tokenized row has an empty prompt or response")
    # Recheck the two renderer hashes at this boundary even though the archive
    # iterator already checked them.  This keeps detached/cache builders
    # fail-closed if a caller supplies a hand-built Vector397Row.
    if hashlib.sha256(input_text.encode("utf-8")).hexdigest() != row.identity["input_text_sha256"]:
        raise StrictDataError("tokenized query renderer hash changed")
    if hashlib.sha256(supervision_text.encode("utf-8")).hexdigest() != row.identity["supervision_text_sha256"]:
        raise StrictDataError("tokenized supervision renderer hash changed")
    transformed = np.asarray(normalizer.transform_row(row), dtype=np.float32)
    labels = input_ids
    return TokenizedS5Row(
        identity=row.identity,
        query_ids=query_ids,
        input_ids=input_ids,
        labels=labels,
        vector=transformed,
        response_start=response_start,
        task_type=str(row.task_type),
    )


class TokenizedVector397Dataset(Sequence[TokenizedS5Row]):
    """Materialized, canonical-order tokenized view for one train prefix."""

    def __init__(self, rows: Sequence[TokenizedS5Row], *, prefix: int | None = None) -> None:
        if not rows:
            raise StrictDataError("tokenized Vector397 dataset is empty")
        self.rows = tuple(rows)
        expected = list(range(len(self.rows)))
        actual = [row.canonical_global_index for row in self.rows]
        if actual != expected:
            raise StrictDataError("tokenized dataset lost canonical global order")
        if prefix is not None and len(self.rows) != int(prefix):
            raise StrictDataError("tokenized dataset prefix cardinality changed")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> TokenizedS5Row:
        return self.rows[index]


def build_tokenized_vector397_dataset(
    manifest_path: str | Path,
    *,
    tokenizer: Any,
    normalizer: Any,
    prefix: int,
    metadata_root: str | Path | None = None,
    samples_path: str | Path | None = None,
    supervision_path: str | Path | None = None,
    event_join_path: str | Path | None = None,
    n_ctx: int = 128,
) -> TokenizedVector397Dataset:
    """Build the strict tokenized screen/formal view from the archive join."""

    rows = [
        tokenize_vector397_row(
            row,
            tokenizer=tokenizer,
            normalizer=normalizer,
            n_ctx=n_ctx,
        )
        for row in iter_vector397_rows(
            manifest_path,
            prefix=int(prefix),
            metadata_root=metadata_root,
            samples_path=samples_path,
            supervision_path=supervision_path,
            event_join_path=event_join_path,
        )
    ]
    return TokenizedVector397Dataset(rows, prefix=int(prefix))


def collate_tokenized_s5_rows(
    rows: Sequence[TokenizedS5Row],
    *,
    device: Any | None = None,
) -> dict[str, Any]:
    """Right-pad a batch while keeping query and supervision views separate."""

    if not rows:
        raise ValueError("cannot collate an empty S5 batch")
    import torch
    from torch.nn.utils.rnn import pad_sequence

    target_device = device if device is not None else "cpu"
    query = pad_sequence(
        [torch.tensor(row.query_ids, dtype=torch.long) for row in rows],
        batch_first=True,
        padding_value=0,
    ).to(target_device)
    full = pad_sequence(
        [torch.tensor(row.input_ids, dtype=torch.long) for row in rows],
        batch_first=True,
        padding_value=0,
    ).to(target_device)
    labels = pad_sequence(
        [torch.tensor(row.labels, dtype=torch.long) for row in rows],
        batch_first=True,
        padding_value=-100,
    ).to(target_device)
    query_lengths = torch.tensor([len(row.query_ids) for row in rows], device=target_device)
    full_lengths = torch.tensor([len(row.input_ids) for row in rows], device=target_device)
    query_positions = torch.arange(query.shape[1], device=target_device)[None, :]
    full_positions = torch.arange(full.shape[1], device=target_device)[None, :]
    return {
        "query_ids": query,
        "query_mask": query_positions < query_lengths[:, None],
        "input_ids": full,
        "input_mask": full_positions < full_lengths[:, None],
        "labels": labels,
        "vector397": torch.from_numpy(np.stack([row.vector for row in rows], axis=0)).to(target_device),
        "rows": tuple(rows),
        "identities": tuple(row.identity for row in rows),
        "task_types": tuple(row.task_type for row in rows),
    }

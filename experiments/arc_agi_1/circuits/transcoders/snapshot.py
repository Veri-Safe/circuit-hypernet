from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import yaml

from .identity import (
    package_code_identity,
    semantic_sha256,
    sha256_file,
    trace_resume_code_compatibility,
    trace_resume_compatibility_is_audited,
)
from .traces import TraceRecord, read_trace_file, trace_files, write_json_atomic


PROFILE_FORMAT = "nvarc_frozen_trace_training_profile_v1"
SNAPSHOT_FORMAT = "nvarc_frozen_trace_snapshot_v1"


def _expand_path(value: Any, *, profile_root: Path) -> str:
    expanded = os.path.expandvars(str(value))
    if "${" in expanded:
        raise ValueError(
            f"Unresolved environment variable in snapshot profile: {expanded}"
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = profile_root / path
    return str(path.resolve())


@dataclass(frozen=True)
class FrozenSnapshotProfile:
    format_version: str
    base_config: str
    source_trace_root: str
    snapshot_root: str
    output_root: str
    run_name: str

    @property
    def semantic_sha256(self) -> str:
        value = asdict(self)
        for key in ("base_config", "source_trace_root", "snapshot_root", "output_root"):
            value[key] = _portable_hc_path(value[key])
        return semantic_sha256(value)


def load_snapshot_profile(path: str | Path) -> FrozenSnapshotProfile:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Frozen snapshot profile must contain a mapping")
    required = {
        "format_version",
        "base_config",
        "source_trace_root",
        "snapshot_root",
        "output_root",
        "run_name",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(
            f"Frozen snapshot profile is missing fields: {sorted(missing)}"
        )
    result = FrozenSnapshotProfile(
        format_version=str(value["format_version"]),
        base_config=_expand_path(value["base_config"], profile_root=source.parent),
        source_trace_root=_expand_path(
            value["source_trace_root"], profile_root=source.parent
        ),
        snapshot_root=_expand_path(value["snapshot_root"], profile_root=source.parent),
        output_root=_expand_path(value["output_root"], profile_root=source.parent),
        run_name=str(value["run_name"]),
    )
    if result.format_version != PROFILE_FORMAT:
        raise ValueError(
            f"Unsupported frozen snapshot profile: {result.format_version}"
        )
    if not result.run_name or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in result.run_name
    ):
        raise ValueError(
            "Snapshot run_name may contain only letters, digits, dot, underscore, and dash"
        )
    if Path(result.snapshot_root) == Path(result.source_trace_root):
        raise ValueError("Snapshot metadata must not overwrite the live trace root")
    return result


def _metadata_path(tensor_path: Path) -> Path:
    return tensor_path.with_suffix(".json")


def _completion_path(tensor_path: Path) -> Path:
    return Path(str(tensor_path) + ".done")


def _portable_hc_path(value: str | Path) -> str:
    path = Path(value).resolve()
    root_value = os.environ.get("HC_ROOT")
    if not root_value:
        return str(path)
    root = Path(root_value).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return str(path)
    return "${HC_ROOT}/" + relative.as_posix()


def _resolve_portable_path(value: str | Path) -> Path:
    expanded = os.path.expandvars(str(value))
    if "${" in expanded:
        raise ValueError(f"Unresolved environment variable in snapshot path: {value}")
    return Path(expanded).resolve()


def resolve_snapshot_source_root(manifest: Mapping[str, Any]) -> Path:
    return _resolve_portable_path(manifest["source_trace_root"])


def _dataset_identities_match(
    current: Mapping[str, Any],
    persisted: Mapping[str, Any],
) -> bool:
    current_value = dict(current)
    persisted_value = dict(persisted)
    current_path = current_value.pop("path", None)
    persisted_path = persisted_value.pop("path", None)
    if current_value != persisted_value:
        return False
    if current_path == persisted_path:
        return True
    root_value = os.environ.get("HC_ROOT")
    if not root_value or current_path is None or persisted_path is None:
        return False
    try:
        relative = Path(str(current_path)).resolve().relative_to(
            Path(root_value).resolve()
        )
    except ValueError:
        return False
    persisted_parts = Path(str(persisted_path)).parts
    return tuple(persisted_parts[-len(relative.parts) :]) == tuple(relative.parts)


def _load_resume_identity(shard_dir: Path) -> dict[str, Any]:
    path = shard_dir / "resume_identity.json"
    if not path.is_file():
        raise FileNotFoundError(f"Trace shard has no resume identity: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != "nvarc_generated_trace_resume_identity_v1":
        raise ValueError(f"Unsupported trace resume identity: {path}")
    return value


def _canonical_resume_identity(
    shard_dir: Path,
    resume_identity: Mapping[str, Any],
    *,
    current_code_identity_sha256: str,
) -> dict[str, Any]:
    """Normalize an audited code-only refresh before freezing a snapshot."""

    expected = dict(resume_identity)
    expected["code_identity_sha256"] = current_code_identity_sha256
    compatibility = trace_resume_code_compatibility(resume_identity, expected)
    if compatibility is not None:
        compatibility_path = shard_dir / "resume_compatibility.json"
        if not compatibility_path.is_file():
            raise ValueError(
                "Trace shard has a code-only identity change without an "
                f"audited compatibility record: {compatibility_path}"
            )
        persisted = json.loads(compatibility_path.read_text(encoding="utf-8"))
        if not isinstance(persisted, Mapping) or not trace_resume_compatibility_is_audited(
            persisted, compatibility
        ):
            raise ValueError(
                "Trace shard resume compatibility is not an audited migration: "
                f"{compatibility_path}"
            )
    return expected


def _snapshot_hash_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in manifest.items() if key not in {"snapshot_sha256"}
    }


def load_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    if source.is_dir():
        manifest_path = source / "manifest.json"
        completion_path = source / "COMPLETE"
    else:
        manifest_path = source
        completion_path = source.parent / "COMPLETE"
    if not completion_path.is_file() or completion_path.read_bytes() != b"ok\n":
        raise ValueError(f"Frozen trace snapshot is incomplete: {source}")
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("format") != SNAPSHOT_FORMAT or value.get("status") != "complete":
        raise ValueError(
            f"Unsupported or incomplete frozen trace snapshot: {manifest_path}"
        )
    expected_sha = semantic_sha256(_snapshot_hash_payload(value))
    if value.get("snapshot_sha256") != expected_sha:
        raise ValueError(
            f"Frozen trace snapshot manifest hash mismatch: {manifest_path}"
        )
    return value


def validate_materialized_snapshot_lineage(
    materialized_manifest: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
) -> None:
    """Verify that an LR-audit snapshot is an exact child of its source snapshot."""

    selection = materialized_manifest.get("row_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("Materialized snapshot is missing row_selection identity")
    expected_manifest = {
        "profile_sha256": source_manifest.get("profile_sha256"),
        "source_config_sha256": source_manifest.get("source_config_sha256"),
        "base_identity_sha256": source_manifest.get("base_identity_sha256"),
        "dataset": source_manifest.get("dataset"),
    }
    expected_selection = {
        "format": "nvarc_lr_audit_row_selection_v1",
        "sample_format": "nvarc_snapshot_lr_audit_row_sample_v1",
        "source_snapshot_sha256": source_manifest.get("snapshot_sha256"),
        "fraction": 0.05,
        "rows": round(int(source_manifest["rows"]) * 0.05),
        "puzzles": int(source_manifest["puzzles"]),
        "dataset_indices_sha256": materialized_manifest.get(
            "dataset_indices_sha256"
        ),
    }
    mismatches = {
        key: {"expected": expected, "actual": materialized_manifest.get(key)}
        for key, expected in expected_manifest.items()
        if materialized_manifest.get(key) != expected
    }
    mismatches.update(
        {
            f"row_selection.{key}": {
                "expected": expected,
                "actual": selection.get(key),
            }
            for key, expected in expected_selection.items()
            if selection.get(key) != expected
        }
    )
    source_generation = source_manifest.get("source_generation_identity")
    materialized_generation = materialized_manifest.get("source_generation_identity")
    if not isinstance(source_generation, Mapping) or not isinstance(
        materialized_generation, Mapping
    ):
        mismatches["source_generation_identity"] = {
            "expected": "source and materialized generation mappings",
            "actual": materialized_generation,
        }
    else:
        for key in (
            "config_sha256",
            "base_identity_sha256",
            "code_identity_sha256",
            "dataset",
            "generation",
        ):
            expected = source_generation.get(key)
            if materialized_generation.get(key) != expected:
                mismatches[f"source_generation_identity.{key}"] = {
                    "expected": expected,
                    "actual": materialized_generation.get(key),
                }
    for key in ("sample_sha256", "records_sha256"):
        value = selection.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            mismatches[f"row_selection.{key}"] = {
                "expected": "64-character SHA256",
                "actual": value,
            }
    if mismatches:
        raise ValueError(f"Materialized snapshot lineage mismatch: {mismatches}")


def _validate_file_entry(source_root: Path, entry: Mapping[str, Any]) -> Path:
    relative = Path(str(entry["relative_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe snapshot relative path: {relative}")
    tensor_path = (source_root / relative).resolve()
    if not tensor_path.is_relative_to(source_root):
        raise ValueError(f"Snapshot path escapes its source root: {tensor_path}")
    metadata_path = _metadata_path(tensor_path)
    completion_path = _completion_path(tensor_path)
    if not completion_path.is_file() or completion_path.read_bytes() != b"ok\n":
        raise ValueError(f"Snapshot trace chunk is no longer committed: {tensor_path}")
    if not tensor_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"Snapshot trace chunk payload is missing: {tensor_path}"
        )
    if tensor_path.stat().st_size != int(entry["tensor_bytes"]):
        raise ValueError(f"Snapshot tensor size changed: {tensor_path}")
    if metadata_path.stat().st_size != int(entry["metadata_bytes"]):
        raise ValueError(f"Snapshot metadata size changed: {metadata_path}")
    if sha256_file(metadata_path) != entry["metadata_sha256"]:
        raise ValueError(f"Snapshot metadata hash changed: {metadata_path}")
    return tensor_path


def verify_snapshot_files(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    verify_tensor_hashes: bool = True,
) -> dict[str, Any]:
    manifest = (
        dict(manifest_or_path)
        if isinstance(manifest_or_path, Mapping)
        else load_snapshot_manifest(manifest_or_path)
    )
    source_root = resolve_snapshot_source_root(manifest)
    checked_bytes = 0
    for entry in manifest["files"]:
        tensor_path = _validate_file_entry(source_root, entry)
        if verify_tensor_hashes and sha256_file(tensor_path) != entry["tensor_sha256"]:
            raise ValueError(f"Snapshot tensor hash changed: {tensor_path}")
        checked_bytes += int(entry["tensor_bytes"]) + int(entry["metadata_bytes"])
    return {
        "files": len(manifest["files"]),
        "bytes": checked_bytes,
        "tensor_hashes_verified": bool(verify_tensor_hashes),
        "snapshot_sha256": manifest["snapshot_sha256"],
    }


def iter_snapshot_records(
    manifest_or_path: Mapping[str, Any] | str | Path,
) -> Iterator[TraceRecord]:
    manifest = (
        dict(manifest_or_path)
        if isinstance(manifest_or_path, Mapping)
        else load_snapshot_manifest(manifest_or_path)
    )
    source_root = resolve_snapshot_source_root(manifest)
    previous = -1
    indices: list[int] = []
    rows = events = tokens = eligible_tokens = 0
    for entry in manifest["files"]:
        tensor_path = _validate_file_entry(source_root, entry)
        records = read_trace_file(tensor_path)
        if len(records) != int(entry["records"]):
            raise ValueError(f"Snapshot record count changed: {tensor_path}")
        for record in records:
            if record.dataset_index <= previous:
                raise ValueError(
                    "Frozen snapshot is not in strictly increasing dataset order"
                )
            previous = record.dataset_index
            indices.append(record.dataset_index)
            rows += 1
            events += len(record.events)
            tokens += len(record.input_ids)
            eligible_tokens += sum(record.eligible_mask)
            yield record
    actual = {
        "rows": rows,
        "events": events,
        "tokens": tokens,
        "eligible_tokens": eligible_tokens,
        "dataset_indices_sha256": semantic_sha256(indices),
    }
    mismatches = {
        key: {"expected": manifest.get(key), "actual": value}
        for key, value in actual.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen snapshot aggregate mismatch: {mismatches}")


def freeze_trace_snapshot(
    *,
    source_root: str | Path,
    destination: str | Path,
    source_config_sha256: str,
    base_identity_sha256: str,
    dataset_identity: Mapping[str, Any],
    dataset_puzzle_names: Sequence[str],
    profile_sha256: str,
    row_selection_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(source_root).resolve()
    portable_source = _portable_hc_path(source)
    target = Path(destination).resolve()
    if (target / "COMPLETE").is_file():
        existing = load_snapshot_manifest(target)
        expected = {
            "source_trace_root": portable_source,
            "source_config_sha256": source_config_sha256,
            "base_identity_sha256": base_identity_sha256,
            "profile_sha256": profile_sha256,
        }
        mismatches = {
            key: {"expected": value, "actual": existing.get(key)}
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Existing frozen snapshot identity mismatch: {mismatches}"
            )
        return existing
    if target.exists():
        raise FileExistsError(
            f"Incomplete frozen snapshot destination exists: {target}"
        )

    files = trace_files(source)
    if not files:
        raise FileNotFoundError(f"No committed trace chunks under {source}")
    common_identity: dict[str, Any] | None = None
    shard_identity_cache: dict[Path, dict[str, Any]] = {}
    current_code_identity = package_code_identity()
    current_code_identity_sha256 = current_code_identity.get("sha256")
    if not isinstance(current_code_identity_sha256, str) or len(
        current_code_identity_sha256
    ) != 64:
        raise ValueError("Current trace code identity is not a SHA256")
    entries: list[dict[str, Any]] = []
    indices: list[int] = []
    selected_puzzles: Counter[str] = Counter()
    rows = events = tokens = eligible_tokens = 0
    previous = -1
    for tensor_path in files:
        shard_dir = tensor_path.parent.resolve()
        if shard_dir not in shard_identity_cache:
            loaded_identity = _load_resume_identity(shard_dir)
            shard_identity_cache[shard_dir] = _canonical_resume_identity(
                shard_dir,
                loaded_identity,
                current_code_identity_sha256=current_code_identity_sha256,
            )
        resume_identity = shard_identity_cache[shard_dir]
        expected_identity = {
            "split": "train",
            "config_sha256": source_config_sha256,
            "base_identity_sha256": base_identity_sha256,
        }
        mismatches = {
            key: {"expected": value, "actual": resume_identity.get(key)}
            for key, value in expected_identity.items()
            if resume_identity.get(key) != value
        }
        persisted_dataset = resume_identity.get("dataset")
        if not isinstance(persisted_dataset, Mapping) or not _dataset_identities_match(
            dataset_identity, persisted_dataset
        ):
            mismatches["dataset"] = {
                "expected": dict(dataset_identity),
                "actual": persisted_dataset,
            }
        if mismatches:
            raise ValueError(f"Trace chunk resume identity mismatch: {mismatches}")
        canonical_dataset = dict(dataset_identity)
        if "path" in canonical_dataset:
            canonical_dataset["path"] = _portable_hc_path(canonical_dataset["path"])
        shared = {
            key: resume_identity.get(key)
            for key in (
                "num_shards",
                "config_sha256",
                "base_identity_sha256",
                "code_identity_sha256",
                "dataset",
                "generation",
            )
        }
        shared["dataset"] = canonical_dataset
        if common_identity is None:
            common_identity = shared
        elif shared != common_identity:
            raise ValueError(
                "Committed trace chunks contain mixed generation identities"
            )

        records = read_trace_file(tensor_path)
        chunk_events = chunk_tokens = chunk_eligible = 0
        for record in records:
            if record.dataset_index <= previous:
                raise ValueError(
                    "Committed trace chunks are not in strictly increasing order"
                )
            if not 0 <= record.dataset_index < len(dataset_puzzle_names):
                raise ValueError(
                    f"Trace dataset index is out of range: {record.dataset_index}"
                )
            expected_puzzle = str(dataset_puzzle_names[record.dataset_index])
            if record.puzzle_name != expected_puzzle:
                raise ValueError(
                    f"Trace puzzle mismatch at {record.dataset_index}: "
                    f"{record.puzzle_name!r} != {expected_puzzle!r}"
                )
            previous = record.dataset_index
            indices.append(record.dataset_index)
            selected_puzzles[expected_puzzle] += 1
            chunk_events += len(record.events)
            chunk_tokens += len(record.input_ids)
            chunk_eligible += sum(record.eligible_mask)
        metadata_path = _metadata_path(tensor_path)
        entries.append(
            {
                "relative_path": tensor_path.relative_to(source).as_posix(),
                "records": len(records),
                "events": chunk_events,
                "tokens": chunk_tokens,
                "eligible_tokens": chunk_eligible,
                "first_dataset_index": records[0].dataset_index,
                "last_dataset_index": records[-1].dataset_index,
                "tensor_bytes": tensor_path.stat().st_size,
                "tensor_sha256": sha256_file(tensor_path),
                "metadata_bytes": metadata_path.stat().st_size,
                "metadata_sha256": sha256_file(metadata_path),
            }
        )
        rows += len(records)
        events += chunk_events
        tokens += chunk_tokens
        eligible_tokens += chunk_eligible

    all_puzzles = Counter(str(value) for value in dataset_puzzle_names)
    full = sorted(
        name for name, count in all_puzzles.items() if selected_puzzles[name] == count
    )
    partial = sorted(
        name for name in selected_puzzles if selected_puzzles[name] < all_puzzles[name]
    )
    zero = sorted(name for name in all_puzzles if selected_puzzles[name] == 0)
    coverage = {
        "source_rows": len(dataset_puzzle_names),
        "source_puzzles": len(all_puzzles),
        "puzzles_any": len(selected_puzzles),
        "puzzles_full": len(full),
        "puzzles_partial": len(partial),
        "puzzles_zero": len(zero),
        "full_puzzle_ids": full,
        "partial_puzzle_ids": partial,
        "zero_puzzle_ids": zero,
        "selected_rows_by_puzzle": dict(sorted(selected_puzzles.items())),
    }
    manifest: dict[str, Any] = {
        "format": SNAPSHOT_FORMAT,
        "status": "complete",
        "profile_sha256": profile_sha256,
        "source_trace_root": portable_source,
        "source_config_sha256": source_config_sha256,
        "base_identity_sha256": base_identity_sha256,
        "source_generation_identity": common_identity,
        "dataset": common_identity["dataset"],
        "files": entries,
        "rows": rows,
        "events": events,
        "tokens": tokens,
        "eligible_tokens": eligible_tokens,
        "first_dataset_index": indices[0],
        "last_dataset_index": indices[-1],
        "cursor_end": indices[-1] + 1,
        "dataset_indices_sha256": semantic_sha256(indices),
        "puzzles": len(selected_puzzles),
        "coverage": coverage,
        "row_selection": (
            dict(row_selection_identity) if row_selection_identity is not None else None
        ),
    }
    manifest["snapshot_sha256"] = semantic_sha256(_snapshot_hash_payload(manifest))

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Stale snapshot temporary directory exists: {temporary}")
    temporary.mkdir()
    try:
        write_json_atomic(temporary / "manifest.json", manifest)
        write_json_atomic(temporary / "coverage.json", coverage)
        (temporary / "COMPLETE").write_bytes(b"ok\n")
        os.replace(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest

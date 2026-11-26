#!/usr/bin/env python3
"""Build the ARC-AGI-1-safe validation split for clean-core FullFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator

from datasets import Dataset, load_from_disk


EXPECTED_SOURCE_ROWS = 720
EXPECTED_SOURCE_PUZZLES = 120
EXPECTED_OVERLAP_PUZZLES = 6
EXPECTED_OUTPUT_ROWS = 684
EXPECTED_OUTPUT_PUZZLES = 114
EXPECTED_OUTPUT_ROW_SEMANTIC_SHA256 = (
    "641ccf93a9114a4b675b240d024d41a06b8d44abe326ce11b68bc7d37bf021d6"
)
EXPECTED_TRAINING_SPLITS = {
    "rearc": {"rows": 102392, "puzzles": 400, "fingerprint": "332b5af0e60bee1d"},
    "rearc128": {
        "rows": 51200,
        "puzzles": 400,
        "fingerprint": "1351eb1f7ee044e3",
    },
    "rearc256": {
        "rows": 102400,
        "puzzles": 400,
        "fingerprint": "693a23157c19c0ee",
    },
    "mini": {"rows": 37632, "puzzles": 147, "fingerprint": "66acf25a0463ab59"},
    "concept": {"rows": 40960, "puzzles": 160, "fingerprint": "2e9f5fdec5842bb6"},
}
EXPECTED_TRAINING_PROFILES = {
    ("rearc",): {"rows": 102392, "puzzles": 400},
    ("rearc128",): {"rows": 51200, "puzzles": 400},
    ("rearc256",): {"rows": 102400, "puzzles": 400},
    ("rearc", "mini", "concept"): {"rows": 180984, "puzzles": 707},
}


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize deterministic validation writes across concurrent jobs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    handle: BinaryIO = path.open("r+b")
    locked = False
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked and os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        elif locked:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_rows(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_bytes(row))
        digest.update(b"\n")
    return digest.hexdigest()


def load_target_ids(challenges_path: Path) -> set[str]:
    payload = json.loads(challenges_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or len(payload) != 400:
        raise ValueError(
            f"Expected 400 ARC-AGI-1 evaluation challenges, found "
            f"{len(payload) if isinstance(payload, dict) else type(payload).__name__}"
        )
    return set(payload)


def load_source(path: Path) -> Dataset:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Hugging Face Dataset at {path}")
    if dataset.column_names != ["puzzle_name", "messages"]:
        raise ValueError(f"Unexpected source columns: {dataset.column_names}")
    return dataset


def validate_training_datasets(
    paths: list[Path], target_ids: set[str]
) -> list[dict[str, Any]]:
    observed_names = [path.name for path in paths]
    combined_expected = EXPECTED_TRAINING_PROFILES.get(tuple(observed_names))
    if combined_expected is None:
        raise ValueError(
            "Expected ordered training splits ['rearc'], ['rearc128'], "
            "['rearc256'], or ['rearc', 'mini', 'concept']; found "
            f"{observed_names}"
        )

    records: list[dict[str, Any]] = []
    all_names: set[str] = set()
    total_rows = 0
    for path in paths:
        dataset = load_source(path)
        puzzle_names = set(dataset["puzzle_name"])
        expected = EXPECTED_TRAINING_SPLITS[path.name]
        observed = {
            "rows": len(dataset),
            "puzzles": len(puzzle_names),
            "fingerprint": dataset._fingerprint,
        }
        if observed != expected:
            raise ValueError(
                f"Training split {path.name} identity mismatch: "
                f"observed={observed} expected={expected}"
            )
        leaked = sorted(puzzle_names & target_ids)
        if leaked:
            raise ValueError(
                f"Training split {path.name} contains ARC-AGI-1 evaluation tasks: "
                f"{leaked}"
            )
        duplicates = sorted(puzzle_names & all_names)
        if duplicates:
            raise ValueError(
                f"Training split {path.name} repeats task IDs from an earlier split: "
                f"{duplicates}"
            )
        all_names.update(puzzle_names)
        total_rows += len(dataset)
        records.append({"name": path.name, **observed})

    if (
        total_rows != combined_expected["rows"]
        or len(all_names) != combined_expected["puzzles"]
    ):
        raise ValueError(
            "Combined training identity mismatch: "
            f"rows={total_rows}/{combined_expected['rows']}, "
            f"puzzles={len(all_names)}/{combined_expected['puzzles']}"
        )
    return records


def validate_counts(
    source: Dataset,
    clean: Dataset,
    target_ids: set[str],
) -> tuple[list[str], list[str]]:
    source_names = set(source["puzzle_name"])
    clean_names = set(clean["puzzle_name"])
    overlap = sorted(source_names & target_ids)
    errors = []
    observed = {
        "source_rows": len(source),
        "source_puzzles": len(source_names),
        "overlap_puzzles": len(overlap),
        "output_rows": len(clean),
        "output_puzzles": len(clean_names),
    }
    expected = {
        "source_rows": EXPECTED_SOURCE_ROWS,
        "source_puzzles": EXPECTED_SOURCE_PUZZLES,
        "overlap_puzzles": EXPECTED_OVERLAP_PUZZLES,
        "output_rows": EXPECTED_OUTPUT_ROWS,
        "output_puzzles": EXPECTED_OUTPUT_PUZZLES,
    }
    for key, value in expected.items():
        if observed[key] != value:
            errors.append(f"{key}: observed={observed[key]} expected={value}")
    leaked = sorted(clean_names & target_ids)
    if leaked:
        errors.append(f"clean validation still contains ARC-AGI-1 targets: {leaked}")
    if errors:
        raise ValueError("ARC-AGI-1 validation contract failed:\n- " + "\n- ".join(errors))
    return overlap, sorted(clean_names)


def build(
    source_path: Path,
    challenges_path: Path,
    output_path: Path,
    manifest_path: Path,
    training_paths: list[Path] | None = None,
) -> dict[str, Any]:
    source = load_source(source_path)
    target_ids = load_target_ids(challenges_path)
    training_splits = (
        validate_training_datasets(training_paths, target_ids)
        if training_paths
        else []
    )
    keep_indices = [
        index
        for index, name in enumerate(source["puzzle_name"])
        if name not in target_ids
    ]
    clean = source.select(keep_indices)
    overlap, clean_names = validate_counts(source, clean, target_ids)
    manifest = {
        "format_version": "nvarc_arc1_clean_validation_v1",
        "source_dataset": str(source_path.resolve()),
        "source_fingerprint": source._fingerprint,
        "source_rows": len(source),
        "source_puzzles": len(set(source["puzzle_name"])),
        "arc1_challenges": str(challenges_path.resolve()),
        "arc1_challenge_file_sha256": hashlib.sha256(
            challenges_path.read_bytes()
        ).hexdigest(),
        "training_splits": training_splits,
        "training_rows": sum(split["rows"] for split in training_splits),
        "training_puzzles": sum(split["puzzles"] for split in training_splits),
        "training_arc1_evaluation_overlap": 0 if training_splits else None,
        "removed_puzzle_ids": overlap,
        "removed_puzzles": len(overlap),
        "removed_rows": len(source) - len(clean),
        "output_dataset": str(output_path.resolve()),
        "output_fingerprint": clean._fingerprint,
        "output_rows": len(clean),
        "output_puzzles": len(clean_names),
        "output_row_semantic_sha256": hash_rows(clean),
    }
    if (
        manifest["output_row_semantic_sha256"]
        != EXPECTED_OUTPUT_ROW_SEMANTIC_SHA256
    ):
        raise ValueError(
            "ARC-AGI-1-safe validation semantic hash changed: "
            f"{manifest['output_row_semantic_sha256']} != "
            f"{EXPECTED_OUTPUT_ROW_SEMANTIC_SHA256}"
        )

    lock_path = output_path.parent / f".{output_path.name}.lock"
    with exclusive_file_lock(lock_path):
        if output_path.exists():
            if not output_path.is_dir():
                raise FileExistsError(
                    f"Output exists and is not a directory: {output_path}"
                )
            existing = load_source(output_path)
            validate_counts(source, existing, target_ids)
            existing_hash = hash_rows(existing)
            if existing_hash != manifest["output_row_semantic_sha256"]:
                raise ValueError(
                    "Existing clean validation content differs from the "
                    "deterministic rebuild"
                )
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(
                    prefix=f".{output_path.name}.tmp-", dir=output_path.parent
                )
            )
            try:
                shutil.rmtree(temporary)
                clean.save_to_disk(str(temporary))
                os.replace(temporary, output_path)
            except BaseException:
                shutil.rmtree(temporary, ignore_errors=True)
                raise

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        encoded_manifest = canonical_json_bytes(manifest) + b"\n"
        if manifest_path.exists():
            if manifest_path.read_bytes() != encoded_manifest:
                raise FileExistsError(
                    f"Existing validation manifest does not match: {manifest_path}"
                )
        else:
            temporary_manifest = manifest_path.with_suffix(
                manifest_path.suffix + f".tmp-{os.getpid()}"
            )
            temporary_manifest.write_bytes(encoded_manifest)
            os.replace(temporary_manifest, manifest_path)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset", type=Path, required=True)
    parser.add_argument("--arc1-challenges", type=Path, required=True)
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--training-dataset",
        type=Path,
        action="append",
        default=[],
        help=(
            "Ordered training split to validate; pass rearc alone, or repeat for "
            "rearc, mini, concept."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(
        args.source_dataset,
        args.arc1_challenges,
        args.output_dataset,
        args.manifest,
        args.training_dataset,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

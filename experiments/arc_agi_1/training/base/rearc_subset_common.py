"""Shared deterministic contracts for puzzle-stratified RE-ARC subsets."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class DatasetContractError(RuntimeError):
    """Raised when a source or derived dataset violates its pinned contract."""


@dataclass(frozen=True)
class SourceContract:
    fingerprint: str
    rows: int
    puzzles: int
    columns: tuple[str, ...] = ("puzzle_name", "messages")


@dataclass(frozen=True)
class SubsetSpec:
    algorithm_version: str
    seed: int
    rows_per_puzzle: int
    output_split: str
    expected_assistant_events: int | None = None


def assistant_event_count(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(message.get("role") == "assistant" for message in messages)


def selection_score(
    *,
    spec: SubsetSpec,
    source_fingerprint: str,
    puzzle_name: str,
    source_index: int,
) -> bytes:
    payload = "\0".join(
        (
            spec.algorithm_version,
            source_fingerprint,
            str(spec.seed),
            puzzle_name,
            str(source_index),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).digest()


def select_subset_indices(
    puzzle_names: Sequence[str],
    *,
    source_fingerprint: str,
    spec: SubsetSpec,
) -> np.ndarray:
    grouped: dict[str, list[int]] = defaultdict(list)
    for source_index, puzzle_name in enumerate(puzzle_names):
        grouped[str(puzzle_name)].append(source_index)

    selected: list[int] = []
    undersized: list[tuple[str, int]] = []
    for puzzle_name in sorted(grouped):
        source_indices = grouped[puzzle_name]
        if len(source_indices) < spec.rows_per_puzzle:
            undersized.append((puzzle_name, len(source_indices)))
            continue
        ranked = sorted(
            source_indices,
            key=lambda source_index: (
                selection_score(
                    spec=spec,
                    source_fingerprint=source_fingerprint,
                    puzzle_name=puzzle_name,
                    source_index=source_index,
                ),
                source_index,
            ),
        )
        selected.extend(ranked[: spec.rows_per_puzzle])
    if undersized:
        details = ", ".join(f"{name}:{rows}" for name, rows in undersized[:10])
        raise DatasetContractError(
            f"Puzzles with fewer than {spec.rows_per_puzzle} rows: {details}"
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def inspect_source(dataset: Any, contract: SourceContract) -> list[str]:
    errors: list[str] = []
    fingerprint = getattr(dataset, "_fingerprint", None)
    if fingerprint != contract.fingerprint:
        errors.append(
            f"fingerprint expected {contract.fingerprint!r}, got {fingerprint!r}"
        )
    if len(dataset) != contract.rows:
        errors.append(f"rows expected {contract.rows}, got {len(dataset)}")
    if tuple(dataset.column_names) != contract.columns:
        errors.append(
            f"columns expected {contract.columns!r}, got {tuple(dataset.column_names)!r}"
        )
    puzzle_names = [str(name) for name in dataset["puzzle_name"]]
    if len(set(puzzle_names)) != contract.puzzles:
        errors.append(
            f"puzzles expected {contract.puzzles}, got {len(set(puzzle_names))}"
        )
    if errors:
        raise DatasetContractError(
            "Source RE-ARC contract mismatch: " + "; ".join(errors)
        )
    return puzzle_names


def dataset_stats(dataset: Any) -> dict[str, Any]:
    puzzle_counts = Counter(str(name) for name in dataset["puzzle_name"])
    event_counts = Counter(
        assistant_event_count(messages) for messages in dataset["messages"]
    )
    return {
        "rows": len(dataset),
        "puzzles": len(puzzle_counts),
        "rows_per_puzzle": {
            str(rows): puzzles
            for rows, puzzles in sorted(Counter(puzzle_counts.values()).items())
        },
        "assistant_events": sum(events * rows for events, rows in event_counts.items()),
        "rows_by_assistant_event_count": {
            str(events): rows for events, rows in sorted(event_counts.items())
        },
    }


def dataset_semantic_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    """Hash dataset rows independently of Hugging Face runtime fingerprints."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                row,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def validate_derived(
    dataset: Any,
    source: Any,
    *,
    source_contract: SourceContract,
    spec: SubsetSpec,
) -> dict[str, Any]:
    expected_rows = source_contract.puzzles * spec.rows_per_puzzle
    if len(dataset) != expected_rows:
        raise DatasetContractError(
            f"Derived rows expected {expected_rows}, got {len(dataset)}"
        )
    if dataset.features != source.features:
        raise DatasetContractError("Derived feature schema differs from source")
    stats = dataset_stats(dataset)
    expected_rows_per_puzzle = {str(spec.rows_per_puzzle): source_contract.puzzles}
    if stats["puzzles"] != source_contract.puzzles:
        raise DatasetContractError(
            f"Derived puzzles expected {source_contract.puzzles}, "
            f"got {stats['puzzles']}"
        )
    if stats["rows_per_puzzle"] != expected_rows_per_puzzle:
        raise DatasetContractError(
            f"Derived rows-per-puzzle contract failed: {stats['rows_per_puzzle']}"
        )
    if (
        spec.expected_assistant_events is not None
        and stats["assistant_events"] != spec.expected_assistant_events
    ):
        raise DatasetContractError(
            "Derived assistant event count expected "
            f"{spec.expected_assistant_events}, got {stats['assistant_events']}"
        )
    return stats

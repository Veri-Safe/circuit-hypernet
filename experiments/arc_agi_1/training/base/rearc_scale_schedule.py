"""Deterministic nested RE-ARC scale selection and 50-epoch schedules."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nvarc_qwen3.nvarc_sft_data import reference_global_permutations


SCHEDULE_VERSION = "rearc_scale_logical_epoch_schedule_v1"
PADDING_VERSION = "rearc8_balanced_distinct_puzzle_padding_v1"


def indices_sha256(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _rank(seed: int, puzzle_name: str, row_index: int) -> bytes:
    value = f"{PADDING_VERSION}\0{seed}\0{puzzle_name}\0{row_index}"
    return hashlib.sha256(value.encode("utf-8")).digest()


def balanced_padding_cycle(puzzle_names: Sequence[str], seed: int) -> np.ndarray:
    """Return all rows once, arranged so every 128-row window has unique puzzles."""

    grouped: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(puzzle_names):
        grouped[str(name)].append(index)
    row_counts = set(map(len, grouped.values()))
    if len(grouped) != 400 or len(row_counts) != 1:
        raise ValueError(
            "Balanced padding requires 400 puzzles with equal rows per puzzle"
        )
    rows_per_puzzle = next(iter(row_counts))

    puzzle_order = sorted(
        grouped,
        key=lambda name: (hashlib.sha256(f"{PADDING_VERSION}\0{seed}\0{name}".encode()).digest(), name),
    )
    per_puzzle = {
        name: sorted(grouped[name], key=lambda index: (_rank(seed, name, index), index))
        for name in puzzle_order
    }
    cycle = np.asarray(
        [
            per_puzzle[name][round_index]
            for round_index in range(rows_per_puzzle)
            for name in puzzle_order
        ],
        dtype=np.int64,
    )
    if len(set(map(int, cycle))) != len(puzzle_names):
        raise AssertionError("Padding cycle must contain every row exactly once")
    return cycle


def _remove_final_batch_overlap(
    permutation: np.ndarray, padding: np.ndarray
) -> tuple[np.ndarray, int]:
    """Keep padding duplicates out of the same final global batch."""

    result = permutation.copy()
    pad_set = set(map(int, padding))
    tail_start = len(result) - len(padding)
    overlap_positions = [
        position
        for position in range(tail_start, len(result))
        if int(result[position]) in pad_set
    ]
    replacement_positions = [
        position
        for position in range(tail_start)
        if int(result[position]) not in pad_set
    ]
    if len(replacement_positions) < len(overlap_positions):
        raise AssertionError("Insufficient non-padding rows for deterministic swaps")
    for tail_position, replacement_position in zip(
        overlap_positions, replacement_positions, strict=False
    ):
        result[tail_position], result[replacement_position] = (
            result[replacement_position],
            result[tail_position],
        )
    final_base_half = set(map(int, result[tail_start:]))
    if final_base_half & pad_set:
        raise AssertionError("Final global batch contains an immediate duplicate")
    return result, len(overlap_positions)


@dataclass(frozen=True)
class RearcSchedule:
    indices: np.ndarray
    steps_per_epoch: int
    rows_per_epoch: int
    padding_rows_per_epoch: int
    epoch_hashes: tuple[str, ...]
    padding_hashes: tuple[str, ...]
    padding_swap_counts: tuple[int, ...]

    @property
    def schedule_sha256(self) -> str:
        return indices_sha256(self.indices)


def build_rearc_schedule(
    puzzle_names: Sequence[str],
    *,
    rows_per_puzzle: int,
    seed: int,
    logical_epochs: int,
    global_batch_size: int = 256,
) -> RearcSchedule:
    """Build an auditable, full-global-batch schedule for one scale and seed."""

    if rows_per_puzzle < 8 or rows_per_puzzle > 128 or rows_per_puzzle % 8:
        raise ValueError("rows_per_puzzle must be a multiple of 8 in [8, 128]")
    if logical_epochs <= 0 or global_batch_size <= 0:
        raise ValueError("epochs and global batch size must be positive")
    dataset_size = len(puzzle_names)
    expected_size = 400 * rows_per_puzzle
    if dataset_size != expected_size:
        raise ValueError(f"Expected {expected_size} rows, got {dataset_size}")

    remainder = dataset_size % global_batch_size
    padding_count = 0 if remainder == 0 else global_batch_size - remainder
    if padding_count not in {0, 128}:
        raise ValueError(
            "Supported RE-ARC scales must require either zero or 128 padding rows"
        )

    base_epochs = reference_global_permutations(dataset_size, seed, logical_epochs)
    padding_cycle = (
        balanced_padding_cycle(puzzle_names, seed) if padding_count else np.empty(0, dtype=np.int64)
    )
    scheduled_epochs: list[np.ndarray] = []
    padding_hashes: list[str] = []
    swap_counts: list[int] = []
    for epoch_index, permutation in enumerate(base_epochs):
        if padding_count:
            start = (epoch_index * padding_count) % dataset_size
            positions = (np.arange(padding_count, dtype=np.int64) + start) % dataset_size
            padding = padding_cycle[positions]
            puzzle_slice = [str(puzzle_names[int(index)]) for index in padding]
            if len(set(puzzle_slice)) != padding_count:
                raise AssertionError("Each epoch must pad with 128 distinct puzzles")
            permutation, swaps = _remove_final_batch_overlap(permutation, padding)
            epoch = np.concatenate((permutation, padding))
            padding_hashes.append(indices_sha256(padding))
            swap_counts.append(swaps)
        else:
            epoch = permutation
            padding_hashes.append(indices_sha256(np.empty(0, dtype=np.int64)))
            swap_counts.append(0)
        if len(epoch) % global_batch_size:
            raise AssertionError("Logical epoch is not divisible by global batch size")
        scheduled_epochs.append(epoch.astype(np.int64, copy=False))

    schedule = np.concatenate(scheduled_epochs)
    rows_per_epoch = len(scheduled_epochs[0])
    return RearcSchedule(
        indices=schedule,
        steps_per_epoch=rows_per_epoch // global_batch_size,
        rows_per_epoch=rows_per_epoch,
        padding_rows_per_epoch=padding_count,
        epoch_hashes=tuple(indices_sha256(epoch) for epoch in scheduled_epochs),
        padding_hashes=tuple(padding_hashes),
        padding_swap_counts=tuple(swap_counts),
    )


def schedule_manifest(
    puzzle_names: Sequence[str],
    *,
    rows_per_puzzle: int,
    seed: int,
    logical_epochs: int,
    global_batch_size: int = 256,
) -> tuple[RearcSchedule, dict[str, object]]:
    schedule = build_rearc_schedule(
        puzzle_names,
        rows_per_puzzle=rows_per_puzzle,
        seed=seed,
        logical_epochs=logical_epochs,
        global_batch_size=global_batch_size,
    )
    dataset_size = len(puzzle_names)
    exposure_counts = np.bincount(schedule.indices, minlength=dataset_size)
    manifest: dict[str, object] = {
        "format_version": SCHEDULE_VERSION,
        "padding_version": PADDING_VERSION if schedule.padding_rows_per_epoch else None,
        "rows_per_puzzle": rows_per_puzzle,
        "training_seed": seed,
        "logical_epochs": logical_epochs,
        "loader_epochs": 1,
        "global_batch_size": global_batch_size,
        "unique_dataset_rows": dataset_size,
        "rows_per_logical_epoch": schedule.rows_per_epoch,
        "padding_rows_per_logical_epoch": schedule.padding_rows_per_epoch,
        "steps_per_logical_epoch": schedule.steps_per_epoch,
        "max_steps": schedule.steps_per_epoch * logical_epochs,
        "scheduled_row_exposures": len(schedule.indices),
        "schedule_sha256": schedule.schedule_sha256,
        "epoch_sha256": list(schedule.epoch_hashes),
        "padding_sha256": list(schedule.padding_hashes),
        "padding_swap_counts": list(schedule.padding_swap_counts),
        "min_row_exposures": int(exposure_counts.min()),
        "max_row_exposures": int(exposure_counts.max()),
    }
    return schedule, manifest

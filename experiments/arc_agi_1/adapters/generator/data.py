from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from nvarc_qwen3.rearc_scale_schedule import build_rearc_schedule
from torch.utils.data import Dataset, Sampler

from .events import TokenizedRow, build_row_events


@dataclass(frozen=True)
class LoadedTrainingSources:
    sources: tuple[Any, ...]
    combined: Any


def load_training_sources(
    paths: Sequence[str],
    *,
    loader: Any = None,
    concatenate: Any = None,
) -> LoadedTrainingSources:
    """Load fixed-order profile splits and expose both source and combined views."""

    if not paths:
        raise ValueError("At least one training dataset path is required")
    if loader is None or concatenate is None:
        from datasets import concatenate_datasets, load_from_disk

        loader = load_from_disk if loader is None else loader
        concatenate = concatenate_datasets if concatenate is None else concatenate
    sources = tuple(loader(path) for path in paths)
    combined = sources[0] if len(sources) == 1 else concatenate(list(sources))
    return LoadedTrainingSources(sources=sources, combined=combined)


def reference_global_permutation(dataset_size: int, seed: int) -> np.ndarray:
    """Match the NVARC Dataset.shuffle + stateful sampler order."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    dataset_permutation = np.random.default_rng(seed).permutation(dataset_size)
    seed_generator = torch.Generator()
    seed_generator.manual_seed(seed)
    sampler_seed = int(
        torch.empty((), dtype=torch.int64).random_(generator=seed_generator).item()
    )
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(sampler_seed)
    sampler_permutation = torch.randperm(
        dataset_size,
        generator=sampler_generator,
        dtype=torch.int64,
    ).numpy()
    return dataset_permutation[sampler_permutation]


class RankStridedRowSampler(Sampler[int]):
    """Shard deterministic global row batches without changing their membership."""

    def __init__(
        self,
        *,
        dataset_size: int,
        seed: int,
        rank: int,
        world_size: int,
        global_batch_size: int,
        micro_batch_size: int,
        max_steps: int,
        start_step: int = 0,
        allow_repeats: bool = False,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if min(dataset_size, world_size, global_batch_size, micro_batch_size, max_steps) <= 0:
            raise ValueError("sampler sizes must be positive")
        local_width = world_size * micro_batch_size
        if global_batch_size % local_width:
            raise ValueError("global batch is not divisible by world size * micro batch")
        requested_rows = max_steps * global_batch_size
        if requested_rows > dataset_size and not allow_repeats:
            raise ValueError("requested optimizer steps exceed unique dataset rows")
        if not 0 <= start_step <= max_steps:
            raise ValueError("start_step is outside the optimizer-step range")

        self.gradient_accumulation_steps = global_batch_size // local_width
        if requested_rows <= dataset_size:
            global_indices = reference_global_permutation(dataset_size, seed)[
                :requested_rows
            ]
        else:
            epochs = (requested_rows + dataset_size - 1) // dataset_size
            global_indices = np.concatenate(
                [
                    reference_global_permutation(dataset_size, seed + epoch)
                    for epoch in range(epochs)
                ]
            )[:requested_rows]
        shaped = global_indices.reshape(
            max_steps,
            self.gradient_accumulation_steps,
            world_size,
            micro_batch_size,
        )
        self._indices = shaped[start_step:, :, rank, :].reshape(-1).astype(np.int64)

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class RearcLogicalEpochRowSampler(Sampler[int]):
    """Shard the pinned RE-ARC-8 balanced-padding logical-epoch schedule."""

    def __init__(
        self,
        *,
        puzzle_names: Sequence[str],
        rows_per_puzzle: int,
        seed: int,
        logical_epochs: int,
        expected_schedule_sha256: str,
        rank: int,
        world_size: int,
        global_batch_size: int,
        micro_batch_size: int,
        max_steps: int,
        start_step: int = 0,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if min(world_size, global_batch_size, micro_batch_size, max_steps) <= 0:
            raise ValueError("sampler sizes must be positive")
        local_width = world_size * micro_batch_size
        if global_batch_size % local_width:
            raise ValueError("global batch is not divisible by world size * micro batch")
        gradient_accumulation_steps = global_batch_size // local_width
        schedule = build_rearc_schedule(
            puzzle_names,
            rows_per_puzzle=rows_per_puzzle,
            seed=seed,
            logical_epochs=logical_epochs,
            global_batch_size=global_batch_size,
        )
        if schedule.schedule_sha256 != expected_schedule_sha256:
            raise ValueError(
                "RE-ARC logical-epoch schedule identity changed: "
                f"{schedule.schedule_sha256} != {expected_schedule_sha256}"
            )
        total_steps = schedule.indices.size // global_batch_size
        if not 0 <= start_step <= max_steps <= total_steps:
            raise ValueError(
                f"requested step range [{start_step}, {max_steps}) exceeds "
                f"the {total_steps}-step RE-ARC schedule"
            )
        shaped = schedule.indices.reshape(
            total_steps,
            gradient_accumulation_steps,
            world_size,
            micro_batch_size,
        )
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._indices = (
            shaped[start_step:max_steps, :, rank, :].reshape(-1).astype(np.int64)
        )

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class NvarcRowDataset(Dataset[TokenizedRow]):
    """Lazily expose one all-position TokenizedRow per source row."""

    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        *,
        max_sequence_length: int,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TokenizedRow:
        row = self.dataset[int(index)]
        return build_row_events(
            row["messages"],
            self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            dataset_index=int(index),
            puzzle_name=row.get("puzzle_name"),
        )


def collate_micro_rows(rows: Sequence[TokenizedRow]) -> tuple[TokenizedRow, ...]:
    """Keep CPU rows unpadded; the training system pads each GPU microbatch."""

    if not rows:
        raise ValueError("Cannot collate an empty row batch")
    return tuple(rows)

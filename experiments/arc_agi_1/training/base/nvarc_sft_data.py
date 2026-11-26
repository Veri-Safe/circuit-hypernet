#!/usr/bin/env python3
"""Deterministic NVARC SFT tokenization, masking, collation, and sampling."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


IGNORE_INDEX = -100
REFERENCE_SHUFFLE_ALGORITHM = "hf_default_rng_then_stateful_random_sampler"


def reference_global_permutations(
    dataset_size: int, seed: int, epochs: int
) -> list[np.ndarray]:
    """Reproduce Dataset.shuffle plus successive StatefulDataLoader epochs."""

    if dataset_size <= 0:
        raise ValueError("dataset_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    dataset_permutation = np.random.default_rng(seed).permutation(dataset_size)
    seed_generator = torch.Generator()
    seed_generator.manual_seed(seed)
    sampler_seed = int(
        torch.empty((), dtype=torch.int64).random_(generator=seed_generator).item()
    )
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(sampler_seed)
    permutations = []
    for _ in range(epochs):
        sampler_permutation = torch.randperm(
            dataset_size, generator=sampler_generator, dtype=torch.int64
        ).numpy()
        permutations.append(dataset_permutation[sampler_permutation])
    return permutations


def reference_global_permutation(dataset_size: int, seed: int) -> np.ndarray:
    """Reproduce the first shuffled epoch used by the reference trainer."""

    return reference_global_permutations(dataset_size, seed, epochs=1)[0]


def first_differing_index(left: str, right: str) -> int:
    for index, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return index
    return min(len(left), len(right))


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text=text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise ValueError("Tokenizer unexpectedly returned a batched result")
        input_ids = input_ids[0]
    return [int(token_id) for token_id in input_ids]


def tokenize_chat_for_sft(
    messages: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    max_sequence_length: int,
    skip_first_assistant_loss: bool = False,
) -> dict[str, Any]:
    """Tokenize cumulative chat chunks and select assistant chunks for supervision."""

    if not messages:
        raise ValueError("messages must not be empty")

    normalized = [
        {"role": str(message["role"]), "content": message["content"]}
        for message in messages
    ]
    previous_formatted = ""
    input_ids: list[int] = []
    labels: list[int] = []
    assistant_tokens = 0
    assistant_events = 0
    masked_assistant_events = 0
    masked_assistant_tokens = 0

    for index, message in enumerate(normalized):
        role = message["role"].lower()
        formatted = tokenizer.apply_chat_template(
            normalized[: index + 1],
            tokenize=False,
            add_generation_prompt=role in {"user", "tool"},
            add_special_tokens=False,
        )
        chunk_start = first_differing_index(previous_formatted, formatted)
        chunk = formatted[chunk_start:]
        chunk_ids = _token_ids(tokenizer, chunk)
        input_ids.extend(chunk_ids)
        is_masked_assistant = (
            role == "assistant"
            and skip_first_assistant_loss
            and assistant_events == 0
        )
        if role == "assistant" and not is_masked_assistant:
            labels.extend(chunk_ids)
            assistant_tokens += len(chunk_ids)
        else:
            labels.extend([IGNORE_INDEX] * len(chunk_ids))
        if role == "assistant":
            assistant_events += 1
            if is_masked_assistant:
                masked_assistant_events += 1
                masked_assistant_tokens += len(chunk_ids)
        previous_formatted = formatted

    if len(input_ids) != len(labels):
        raise AssertionError("input_ids and labels diverged")
    if not input_ids:
        raise ValueError("Conversation tokenized to zero tokens")
    if assistant_tokens == 0:
        raise ValueError(
            "Conversation contains no trainable assistant tokens after applying "
            "the assistant loss mask"
        )
    if len(input_ids) > max_sequence_length:
        raise ValueError(
            f"Conversation has {len(input_ids)} tokens, exceeding "
            f"max_sequence_length={max_sequence_length}"
        )
    return {
        "input_ids": input_ids,
        "labels": labels,
        "length": len(input_ids),
        "assistant_tokens": assistant_tokens,
        "assistant_events": assistant_events,
        "supervised_assistant_events": assistant_events - masked_assistant_events,
        "masked_assistant_events": masked_assistant_events,
        "masked_assistant_tokens": masked_assistant_tokens,
    }


class TokenizedNvarcDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        dataset: Any,
        tokenizer: Any,
        max_sequence_length: int,
        *,
        skip_first_assistant_loss: bool = False,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.skip_first_assistant_loss = bool(skip_first_assistant_loss)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[int(index)]
        encoded = tokenize_chat_for_sft(
            row["messages"],
            self.tokenizer,
            max_sequence_length=self.max_sequence_length,
            skip_first_assistant_loss=self.skip_first_assistant_loss,
        )
        encoded["dataset_index"] = int(index)
        encoded["puzzle_name"] = row.get("puzzle_name")
        return encoded


@dataclass(frozen=True)
class SFTCollator:
    pad_token_id: int
    pad_to_multiple_of: int = 64

    def __call__(self, rows: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not rows:
            raise ValueError("Cannot collate an empty batch")
        max_length = max(len(row["input_ids"]) for row in rows)
        if self.pad_to_multiple_of > 1:
            max_length = (
                math.ceil(max_length / self.pad_to_multiple_of)
                * self.pad_to_multiple_of
            )

        input_ids = torch.full(
            (len(rows), max_length), self.pad_token_id, dtype=torch.long
        )
        labels = torch.full((len(rows), max_length), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
        dataset_indices = torch.empty(len(rows), dtype=torch.long)
        for batch_index, row in enumerate(rows):
            length = len(row["input_ids"])
            input_ids[batch_index, :length] = torch.tensor(
                row["input_ids"], dtype=torch.long
            )
            labels[batch_index, :length] = torch.tensor(row["labels"], dtype=torch.long)
            attention_mask[batch_index, :length] = 1
            dataset_indices[batch_index] = int(row["dataset_index"])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "dataset_indices": dataset_indices,
        }


class RankStridedBatchSampler(Sampler[int]):
    """Shard deterministic epoch permutations without changing global batches."""

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
        epochs: int = 1,
    ) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        if (
            min(
                dataset_size,
                world_size,
                global_batch_size,
                micro_batch_size,
                max_steps,
                epochs,
            )
            <= 0
        ):
            raise ValueError("sampler sizes must be positive")
        local_batch = world_size * micro_batch_size
        if global_batch_size % local_batch:
            raise ValueError(
                f"global batch {global_batch_size} is not divisible by "
                f"world_size * micro_batch_size = {local_batch}"
            )
        steps_per_epoch = dataset_size // global_batch_size
        if steps_per_epoch <= 0:
            raise ValueError(
                f"dataset_size={dataset_size} is smaller than "
                f"global_batch_size={global_batch_size}"
            )
        available_steps = steps_per_epoch * epochs
        if max_steps > available_steps:
            raise ValueError(
                f"Training requests {max_steps} steps, but {epochs} epoch(s) over "
                f"{dataset_size} rows provide only {available_steps} complete global batches"
            )

        self.gradient_accumulation_steps = global_batch_size // local_batch
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        usable_rows = steps_per_epoch * global_batch_size
        epoch_indices = reference_global_permutations(dataset_size, seed, epochs)
        global_indices = np.concatenate(
            [permutation[:usable_rows] for permutation in epoch_indices]
        )[: max_steps * global_batch_size]
        shaped = global_indices.reshape(
            max_steps, self.gradient_accumulation_steps, world_size, micro_batch_size
        )
        if not 0 <= start_step <= max_steps:
            raise ValueError(
                f"start_step must be in [0, {max_steps}], got {start_step}"
            )
        self._indices = (
            shaped[start_step:, :, rank, :].reshape(-1).astype(np.int64, copy=False)
        )

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self._indices)

    def __len__(self) -> int:
        return len(self._indices)


class RankStridedSequentialSampler(Sampler[int]):
    def __init__(self, *, sample_count: int, rank: int, world_size: int) -> None:
        if sample_count < 0:
            raise ValueError("sample_count must be nonnegative")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is outside world size {world_size}")
        self._indices = range(rank, sample_count, world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def causal_loss_sum(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid_tokens = shift_labels.ne(IGNORE_INDEX).sum()
    if int(valid_tokens) == 0:
        raise ValueError("Batch has no trainable next-token targets")
    loss_sum = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum, valid_tokens

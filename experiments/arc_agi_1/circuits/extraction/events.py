from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .identity import event_key as build_event_key
from .storage import load_safetensors_bundle, write_safetensors_bundle


CATALOG_FORMAT = "nvarc_rearc_event_catalog_chunk_v1"


@dataclass(frozen=True)
class CircuitEvent:
    event_key: str
    dataset_index: int
    pair_index: int
    message_index: int
    puzzle_name: str | None
    condition_input_ids: tuple[int, ...]
    gold_response_ids: tuple[int, ...]
    row_token_count: int
    work_unit: int
    trace_generation_mode: str = "cumulative_generated"
    source_name: str | None = None
    source_index: int | None = None
    view_index: int | None = None
    test_index: int | None = None

    @property
    def condition_length(self) -> int:
        return len(self.condition_input_ids)

    def validate(self, *, max_sequence_length: int = 16_384) -> None:
        if len(self.event_key) != 64:
            raise ValueError("Circuit event key must be a SHA256")
        if (
            min(self.dataset_index, self.pair_index, self.message_index, self.work_unit)
            < 0
        ):
            raise ValueError(f"Circuit event has a negative index: {self}")
        if not self.condition_input_ids:
            raise ValueError("Circuit event condition is empty")
        if not self.gold_response_ids:
            raise ValueError("Circuit event gold response is empty")
        if not self.condition_length < self.row_token_count <= max_sequence_length:
            raise ValueError("Circuit event boundaries are invalid")
        if self.trace_generation_mode not in {
            "cumulative_generated",
            "final_only_gold_prefix",
        }:
            raise ValueError(
                f"Unsupported circuit event mode: {self.trace_generation_mode}"
            )
        for label, value in (
            ("source_index", self.source_index),
            ("view_index", self.view_index),
            ("test_index", self.test_index),
        ):
            if value is not None and value < 0:
                raise ValueError(f"Circuit event {label} is negative")


def _first_differing_index(left: str, right: str) -> int:
    for index, (left_character, right_character) in enumerate(zip(left, right)):
        if left_character != right_character:
            return index
    return min(len(left), len(right))


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text=text, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Tokenizer unexpectedly returned a batched result")
        encoded = encoded[0]
    return [int(item) for item in encoded]


def _normalize_messages(
    messages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("messages must not be empty")
    normalized = [
        {"role": str(message["role"]).lower(), "content": message["content"]}
        for message in messages
    ]
    for index, message in enumerate(normalized):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError(
                f"NVARC row message {index} is {message['role']!r}, expected {expected!r}"
            )
    if normalized[-1]["role"] != "assistant":
        raise ValueError("NVARC rows must end with an assistant response")
    return normalized


def build_circuit_events(
    messages: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    dataset_index: int,
    puzzle_name: str | None,
    dataset_snapshot_sha256: str,
    max_sequence_length: int,
    num_work_units: int,
    trace_generation_mode: str = "cumulative_generated",
    source_name: str | None = None,
    source_index: int | None = None,
    view_index: int | None = None,
    test_index: int | None = None,
) -> tuple[CircuitEvent, ...]:
    """Build the exact gold-prefix SFT/S2 event conditions for one source row."""

    if max_sequence_length <= 0 or num_work_units <= 0:
        raise ValueError("Sequence and work-unit limits must be positive")
    normalized = _normalize_messages(messages)
    if trace_generation_mode not in {
        "cumulative_generated",
        "final_only_gold_prefix",
    }:
        raise ValueError(f"Unsupported trace_generation_mode: {trace_generation_mode}")
    if (
        trace_generation_mode == "final_only_gold_prefix"
        and normalized[-1]["content"] != ""
    ):
        raise ValueError(
            "final_only_gold_prefix requires an empty terminal assistant slot so "
            "the current gold answer cannot enter the prompt"
        )
    previous_formatted = ""
    all_input_ids: list[int] = []
    raw_events: list[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = []

    for message_index, message in enumerate(normalized):
        role = message["role"]
        formatted = tokenizer.apply_chat_template(
            normalized[: message_index + 1],
            tokenize=False,
            add_generation_prompt=role == "user",
            add_special_tokens=False,
        )
        chunk_start = _first_differing_index(previous_formatted, formatted)
        chunk_ids = _token_ids(tokenizer, formatted[chunk_start:])
        token_start = len(all_input_ids)
        all_input_ids.extend(chunk_ids)
        if role == "assistant":
            if not chunk_ids:
                raise ValueError(
                    f"Assistant message {message_index} produced no tokens"
                )
            raw_events.append(
                (
                    len(raw_events),
                    message_index,
                    tuple(all_input_ids[:token_start]),
                    tuple(chunk_ids),
                )
            )
        previous_formatted = formatted

    row_token_count = len(all_input_ids)
    if not 0 < row_token_count <= max_sequence_length:
        raise ValueError(
            f"Conversation has {row_token_count} tokens; truncation is forbidden at "
            f"max_sequence_length={max_sequence_length}"
        )

    result = []
    if trace_generation_mode == "final_only_gold_prefix":
        raw_events = raw_events[-1:]
    for pair_index, message_index, condition_ids, gold_ids in raw_events:
        key = build_event_key(dataset_snapshot_sha256, dataset_index, pair_index)
        work_unit = int(key[:16], 16) % num_work_units
        event = CircuitEvent(
            event_key=key,
            dataset_index=int(dataset_index),
            pair_index=pair_index,
            message_index=message_index,
            puzzle_name=puzzle_name,
            condition_input_ids=condition_ids,
            gold_response_ids=gold_ids,
            row_token_count=row_token_count,
            work_unit=work_unit,
            trace_generation_mode=trace_generation_mode,
            source_name=source_name,
            source_index=source_index,
            view_index=view_index,
            test_index=test_index,
        )
        event.validate(max_sequence_length=max_sequence_length)
        result.append(event)
    return tuple(result)


def write_catalog_chunk(
    path: str | Path,
    events: Sequence[CircuitEvent],
    *,
    dataset_snapshot_sha256: str,
    config_sha256: str,
) -> dict[str, Any]:
    if not events:
        raise ValueError("Cannot write an empty event catalog chunk")
    condition_offsets = [0]
    gold_offsets = [0]
    condition_ids: list[int] = []
    gold_ids: list[int] = []
    records = []
    seen: set[str] = set()
    for event in events:
        event.validate()
        if event.event_key in seen:
            raise ValueError(f"Duplicate event key in catalog chunk: {event.event_key}")
        seen.add(event.event_key)
        condition_ids.extend(event.condition_input_ids)
        gold_ids.extend(event.gold_response_ids)
        condition_offsets.append(len(condition_ids))
        gold_offsets.append(len(gold_ids))
        record = asdict(event)
        record.pop("condition_input_ids")
        record.pop("gold_response_ids")
        if (
            event.trace_generation_mode == "cumulative_generated"
            and event.source_name is None
            and event.source_index is None
            and event.view_index is None
            and event.test_index is None
        ):
            for key in (
                "trace_generation_mode",
                "source_name",
                "source_index",
                "view_index",
                "test_index",
            ):
                record.pop(key)
        records.append(record)
    tensors = {
        "condition_input_ids": torch.tensor(condition_ids, dtype=torch.int16),
        "condition_offsets": torch.tensor(condition_offsets, dtype=torch.int64),
        "gold_response_ids": torch.tensor(gold_ids, dtype=torch.int16),
        "gold_offsets": torch.tensor(gold_offsets, dtype=torch.int64),
    }
    return write_safetensors_bundle(
        path,
        tensors=tensors,
        metadata={
            "format": CATALOG_FORMAT,
            "dataset_snapshot_sha256": dataset_snapshot_sha256,
            "config_sha256": config_sha256,
            "events": len(events),
            "records": records,
        },
    )


def read_catalog_chunk(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[CircuitEvent, ...]:
    tensors, metadata = load_safetensors_bundle(path, expected_format=CATALOG_FORMAT)
    if expected_identity is not None:
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in expected_identity.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Catalog chunk identity changed: {mismatches}")
    records = metadata.get("records")
    if not isinstance(records, list):
        raise ValueError("Catalog metadata has no records list")
    condition_offsets = tensors["condition_offsets"].tolist()
    gold_offsets = tensors["gold_offsets"].tolist()
    if (
        len(condition_offsets) != len(records) + 1
        or len(gold_offsets) != len(records) + 1
    ):
        raise ValueError("Catalog offsets do not align with records")
    result = []
    for index, record in enumerate(records):
        c_start, c_end = condition_offsets[index : index + 2]
        g_start, g_end = gold_offsets[index : index + 2]
        event = CircuitEvent(
            **record,
            condition_input_ids=tuple(
                int(item)
                for item in tensors["condition_input_ids"][c_start:c_end].tolist()
            ),
            gold_response_ids=tuple(
                int(item)
                for item in tensors["gold_response_ids"][g_start:g_end].tolist()
            ),
        )
        event.validate()
        result.append(event)
    return tuple(result)

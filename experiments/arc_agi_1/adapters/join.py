from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from s2_hypernet.nvarc_pg.events import TokenizedRow, build_row_events
from s3circuit.nvarc_circuit.events import CircuitEvent
from s3circuit.nvarc_circuit.storage import load_safetensors_bundle, write_safetensors_bundle

from .artifacts import CircuitVectorBank
from .contracts import (
    EVAL_JOIN_FORMAT,
    JOIN_FORMAT,
    VECTOR_SCHEMA,
    semantic_sha256,
    token_ids_sha256,
)


@dataclass(frozen=True)
class JoinRecord:
    event_key: str
    dataset_index: int
    pair_index: int
    condition_input_ids_sha256: str
    gold_response_ids_sha256: str
    artifact_id: str
    final_shard_relative_path: str
    row_in_shard: int


@dataclass(frozen=True)
class EvalJoinRecord:
    query_index: int
    event_key: str
    condition_input_ids_sha256: str
    artifact_id: str
    final_shard_relative_path: str
    row_in_shard: int


def _ascii_hashes(values: Sequence[str]) -> torch.Tensor:
    return torch.tensor(
        [list(value.encode("ascii")) for value in values],
        dtype=torch.uint8,
    )


def _event_tokens(row: TokenizedRow, pair_index: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    event = row.events[pair_index]
    return (
        tuple(row.input_ids[: event.condition_end]),
        tuple(row.input_ids[event.response_start : event.response_end]),
    )


def _catalog_indexes(
    events: Mapping[str, CircuitEvent],
) -> tuple[
    dict[tuple[int, int], CircuitEvent],
    dict[tuple[str, int, int | None, int | None, int], CircuitEvent],
    dict[tuple[str, str, int], list[CircuitEvent]],
]:
    direct: dict[tuple[int, int], CircuitEvent] = {}
    source: dict[
        tuple[str, int, int | None, int | None, int], CircuitEvent
    ] = {}
    content: dict[tuple[str, str, int], list[CircuitEvent]] = {}
    for event in events.values():
        direct_key = (event.dataset_index, event.pair_index)
        if direct_key in direct:
            raise ValueError(f"Catalog repeats dataset/pair coordinates: {direct_key}")
        direct[direct_key] = event
        if event.source_name is not None and event.source_index is not None:
            source_key = (
                event.source_name,
                event.source_index,
                event.view_index,
                event.test_index,
                event.pair_index,
            )
            if source_key in source:
                raise ValueError(f"Catalog repeats source coordinates: {source_key}")
            source[source_key] = event
        key = (
            token_ids_sha256(event.condition_input_ids),
            token_ids_sha256(event.gold_response_ids),
            event.pair_index,
        )
        content.setdefault(key, []).append(event)
    return direct, source, content


def build_join_records(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
    max_sequence_length: int,
    catalog_events: Mapping[str, CircuitEvent],
    vector_bank: CircuitVectorBank,
) -> tuple[JoinRecord, ...]:
    direct, source, content = _catalog_indexes(catalog_events)
    records: list[JoinRecord] = []
    seen_events: set[str] = set()
    for local_index, raw_row in enumerate(raw_rows):
        row = build_row_events(
            raw_row["messages"],
            tokenizer,
            max_sequence_length=max_sequence_length,
            dataset_index=local_index,
            puzzle_name=raw_row.get("puzzle_name"),
        )
        for pair_index in range(len(row.events)):
            condition_ids, gold_ids = _event_tokens(row, pair_index)
            condition_sha = token_ids_sha256(condition_ids)
            gold_sha = token_ids_sha256(gold_ids)
            match = None
            source_name = raw_row.get("source_name")
            source_index = raw_row.get("source_index")
            if source_name is not None and source_index is not None:
                match = source.get(
                    (
                        str(source_name),
                        int(source_index),
                        (
                            int(raw_row["view_index"])
                            if raw_row.get("view_index") is not None
                            else None
                        ),
                        (
                            int(raw_row["test_index"])
                            if raw_row.get("test_index") is not None
                            else None
                        ),
                        pair_index,
                    )
                )
            candidates = content.get((condition_sha, gold_sha, pair_index), [])
            if match is None:
                match = direct.get((local_index, pair_index))
            if match is not None and (
                condition_ids != match.condition_input_ids
                or gold_ids != match.gold_response_ids
            ):
                if len(candidates) == 1:
                    match = candidates[0]
                elif len(candidates) > 1:
                    raise ValueError(
                        f"Ambiguous token-identical circuit events for row={local_index}, "
                        f"pair={pair_index}"
                    )
            if match is None:
                if len(candidates) == 1:
                    match = candidates[0]
                elif len(candidates) > 1:
                    raise ValueError(
                        f"Ambiguous token-identical circuit events for row={local_index}, "
                        f"pair={pair_index}"
                    )
            if match is None:
                raise ValueError(
                    f"No circuit event matches row={local_index}, pair={pair_index}"
                )
            if condition_ids != match.condition_input_ids or gold_ids != match.gold_response_ids:
                raise ValueError(
                    f"Circuit token identity mismatch for row={local_index}, pair={pair_index}"
                )
            if match.pair_index != pair_index:
                raise ValueError("Circuit pair index changed")
            if match.event_key in seen_events:
                raise ValueError(f"Circuit event matched twice: {match.event_key}")
            vector = vector_bank.record(match.event_key)
            seen_events.add(match.event_key)
            records.append(
                JoinRecord(
                    event_key=match.event_key,
                    dataset_index=local_index,
                    pair_index=pair_index,
                    condition_input_ids_sha256=condition_sha,
                    gold_response_ids_sha256=gold_sha,
                    artifact_id=vector.artifact_id,
                    final_shard_relative_path=vector.shard_relative_path,
                    row_in_shard=vector.row_in_shard,
                )
            )
    if len(records) != len(catalog_events) or seen_events != set(catalog_events):
        missing = sorted(set(catalog_events) - seen_events)[:5]
        raise ValueError(
            f"Strict circuit join coverage failed: joined={len(records)}, "
            f"catalog={len(catalog_events)}, missing={missing}"
        )
    return tuple(records)


def write_join_bundle(
    path: str | Path,
    records: Sequence[JoinRecord],
    *,
    artifact_contract_sha256: str,
    catalog_manifest_sha256: str,
    final_root_manifest_sha256: str,
) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda item: (item.dataset_index, item.pair_index)))
    if not ordered:
        raise ValueError("Cannot write an empty circuit join")
    shard_paths = sorted({item.final_shard_relative_path for item in ordered})
    shard_index = {path: index for index, path in enumerate(shard_paths)}
    identity_rows = [
        {
            "event_key": item.event_key,
            "dataset_index": item.dataset_index,
            "pair_index": item.pair_index,
            "condition_input_ids_sha256": item.condition_input_ids_sha256,
            "gold_response_ids_sha256": item.gold_response_ids_sha256,
            "artifact_id": item.artifact_id,
            "final_shard_relative_path": item.final_shard_relative_path,
            "row_in_shard": item.row_in_shard,
        }
        for item in ordered
    ]
    join_sha = semantic_sha256(
        {
            "format": JOIN_FORMAT,
            "artifact_contract_sha256": artifact_contract_sha256,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "final_root_manifest_sha256": final_root_manifest_sha256,
            "records": identity_rows,
        }
    )
    return write_safetensors_bundle(
        path,
        tensors={
            "event_key_ascii": _ascii_hashes([item.event_key for item in ordered]),
            "artifact_id_ascii": _ascii_hashes([item.artifact_id for item in ordered]),
            "dataset_index": torch.tensor(
                [item.dataset_index for item in ordered], dtype=torch.int32
            ),
            "pair_index": torch.tensor(
                [item.pair_index for item in ordered], dtype=torch.int16
            ),
            "shard_index": torch.tensor(
                [shard_index[item.final_shard_relative_path] for item in ordered],
                dtype=torch.int32,
            ),
            "row_in_shard": torch.tensor(
                [item.row_in_shard for item in ordered], dtype=torch.int32
            ),
        },
        metadata={
            "format": JOIN_FORMAT,
            "status": "complete",
            "artifact_contract_sha256": artifact_contract_sha256,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "final_root_manifest_sha256": final_root_manifest_sha256,
            "vector_schema": VECTOR_SCHEMA,
            "events": len(ordered),
            "shard_paths": shard_paths,
            "join_sha256": join_sha,
            "records": identity_rows,
        },
    )


def load_join_bundle(path: str | Path) -> tuple[dict[str, JoinRecord], dict[str, Any]]:
    tensors, metadata = load_safetensors_bundle(path, expected_format=JOIN_FORMAT)
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != metadata.get("events"):
        raise ValueError("Circuit join records are missing or incomplete")
    result = {item["event_key"]: JoinRecord(**item) for item in records}
    if len(result) != len(records):
        raise ValueError("Circuit join repeats event identities")
    expected_keys = tuple(item["event_key"] for item in records)
    actual_keys = tuple(
        bytes(row.tolist()).decode("ascii") for row in tensors["event_key_ascii"]
    )
    if actual_keys != expected_keys:
        raise ValueError("Circuit join tensor and metadata orders differ")
    return result, metadata


def _prompt_token_ids(tokenizer: Any, prompt: str) -> tuple[int, ...]:
    encoded = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("Evaluation tokenizer unexpectedly returned a batch")
        encoded = encoded[0]
    return tuple(int(value) for value in encoded)


def _eval_target_pair_index(view: Any) -> int:
    """Return the final prediction's pair index in the frozen ARC prompt."""

    spec = getattr(view, "spec", None)
    train_order = getattr(spec, "train_order", None)
    if train_order is None:
        raise ValueError("Evaluation view has no frozen training-example order")
    train_count = len(train_order)
    removed = int(getattr(view, "removed_train_examples", -1))
    if not 0 <= removed <= train_count:
        raise ValueError(
            "Evaluation view has invalid removed_train_examples: "
            f"removed={removed}, train_count={train_count}"
        )
    return train_count - removed


def _eval_catalog_view_index(view: Any) -> int:
    """Translate S2 author-view indices to the S3 catalog coordinate space."""

    view_index = int(view.view_index)
    if bool(view.is_canonical):
        if view_index != 0:
            raise ValueError(
                f"Canonical evaluation view has view_index={view_index}, expected 0"
            )
        return 0
    if view_index < 0:
        raise ValueError(f"Author evaluation view has negative view_index={view_index}")
    # S2 numbers author views from zero. S3 reserves zero for the canonical
    # view and therefore stores author views at one-based ordinals.
    return view_index + 1


def build_eval_join_records(
    query_views: Sequence[Any],
    *,
    tokenizer: Any,
    catalog_events: Mapping[str, CircuitEvent],
    vector_bank: CircuitVectorBank,
) -> tuple[EvalJoinRecord, ...]:
    """Strictly bind each ARC public query view to its finalized S3 vector."""

    by_source_index: dict[int, CircuitEvent] = {}
    for event in catalog_events.values():
        if event.source_index is None:
            raise ValueError("Evaluation catalog event has no source_index")
        source_index = int(event.source_index)
        if source_index in by_source_index:
            raise ValueError(f"Evaluation catalog repeats source_index={source_index}")
        by_source_index[source_index] = event

    records: list[EvalJoinRecord] = []
    seen: set[str] = set()
    for view in query_views:
        query_index = int(view.query_index)
        event = by_source_index.get(query_index)
        if event is None:
            raise ValueError(f"No evaluation circuit for query_index={query_index}")
        expected_source = (
            "arc1_public_canonical"
            if bool(view.is_canonical)
            else "arc1_public_author_view"
        )
        expected_puzzle = f"arc1_public::{view.task_id}"
        expected_pair_index = _eval_target_pair_index(view)
        expected_view_index = _eval_catalog_view_index(view)
        coordinate_mismatches = {
            "source_name": (expected_source, event.source_name),
            "view_index": (expected_view_index, event.view_index),
            "test_index": (int(view.test_index), event.test_index),
            "puzzle_name": (expected_puzzle, event.puzzle_name),
            "pair_index": (expected_pair_index, event.pair_index),
            "trace_generation_mode": (
                "final_only_gold_prefix",
                event.trace_generation_mode,
            ),
        }
        changed = {
            key: {"expected": values[0], "actual": values[1]}
            for key, values in coordinate_mismatches.items()
            if values[0] != values[1]
        }
        if changed:
            raise ValueError(
                f"Evaluation circuit coordinates changed for query={query_index}: {changed}"
            )
        condition_ids = _prompt_token_ids(tokenizer, str(view.prompt))
        if condition_ids != event.condition_input_ids:
            raise ValueError(
                f"Evaluation circuit prompt tokens changed for query={query_index}"
            )
        if event.event_key in seen:
            raise ValueError(f"Evaluation circuit matched twice: {event.event_key}")
        vector = vector_bank.record(event.event_key)
        seen.add(event.event_key)
        records.append(
            EvalJoinRecord(
                query_index=query_index,
                event_key=event.event_key,
                condition_input_ids_sha256=token_ids_sha256(condition_ids),
                artifact_id=vector.artifact_id,
                final_shard_relative_path=vector.shard_relative_path,
                row_in_shard=vector.row_in_shard,
            )
        )
    if len(records) != len(catalog_events) or seen != set(catalog_events):
        raise ValueError(
            "Evaluation circuit join is not bijective: "
            f"queries={len(records)} catalog={len(catalog_events)}"
        )
    return tuple(records)


def write_eval_join_bundle(
    path: str | Path,
    records: Sequence[EvalJoinRecord],
    *,
    eval_artifact_contract_sha256: str,
    query_manifest_sha256: str,
    catalog_manifest_sha256: str,
    final_root_manifest_sha256: str,
) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda item: item.query_index))
    if not ordered or tuple(item.query_index for item in ordered) != tuple(range(len(ordered))):
        raise ValueError("Evaluation join must cover contiguous query indices")
    shard_paths = sorted({item.final_shard_relative_path for item in ordered})
    shard_index = {value: index for index, value in enumerate(shard_paths)}
    identity_rows = [item.__dict__ for item in ordered]
    join_sha = semantic_sha256(
        {
            "format": EVAL_JOIN_FORMAT,
            "eval_artifact_contract_sha256": eval_artifact_contract_sha256,
            "query_manifest_sha256": query_manifest_sha256,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "final_root_manifest_sha256": final_root_manifest_sha256,
            "records": identity_rows,
        }
    )
    return write_safetensors_bundle(
        path,
        tensors={
            "event_key_ascii": _ascii_hashes([item.event_key for item in ordered]),
            "artifact_id_ascii": _ascii_hashes([item.artifact_id for item in ordered]),
            "query_index": torch.tensor(
                [item.query_index for item in ordered], dtype=torch.int32
            ),
            "shard_index": torch.tensor(
                [shard_index[item.final_shard_relative_path] for item in ordered],
                dtype=torch.int32,
            ),
            "row_in_shard": torch.tensor(
                [item.row_in_shard for item in ordered], dtype=torch.int32
            ),
        },
        metadata={
            "format": EVAL_JOIN_FORMAT,
            "status": "complete",
            "eval_artifact_contract_sha256": eval_artifact_contract_sha256,
            "query_manifest_sha256": query_manifest_sha256,
            "catalog_manifest_sha256": catalog_manifest_sha256,
            "final_root_manifest_sha256": final_root_manifest_sha256,
            "vector_schema": VECTOR_SCHEMA,
            "queries": len(ordered),
            "shard_paths": shard_paths,
            "join_sha256": join_sha,
            "records": identity_rows,
        },
    )


def load_eval_join_bundle(
    path: str | Path,
) -> tuple[dict[int, EvalJoinRecord], dict[str, Any]]:
    tensors, metadata = load_safetensors_bundle(path, expected_format=EVAL_JOIN_FORMAT)
    records = metadata.get("records")
    if not isinstance(records, list) or len(records) != metadata.get("queries"):
        raise ValueError("Evaluation join records are missing or incomplete")
    result = {int(item["query_index"]): EvalJoinRecord(**item) for item in records}
    if tuple(sorted(result)) != tuple(range(len(records))):
        raise ValueError("Evaluation join query indices are incomplete")
    tensor_indices = tuple(int(value) for value in tensors["query_index"].tolist())
    if tensor_indices != tuple(range(len(records))):
        raise ValueError("Evaluation join tensor order changed")
    tensor_keys = tuple(
        bytes(row.tolist()).decode("ascii") for row in tensors["event_key_ascii"]
    )
    if tensor_keys != tuple(result[index].event_key for index in range(len(result))):
        raise ValueError("Evaluation join event identities changed")
    return result, metadata


def validate_matching_eval_join_bundle(
    path: str | Path,
    records: Sequence[EvalJoinRecord],
    *,
    expected_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Load an immutable eval join only when it exactly matches this build."""

    restored, metadata = load_eval_join_bundle(path)
    expected_records = {record.query_index: record for record in records}
    if restored != expected_records:
        raise ValueError("Existing evaluation join records differ from this build")
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected_metadata.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing evaluation join identity changed: {mismatches}")
    return metadata

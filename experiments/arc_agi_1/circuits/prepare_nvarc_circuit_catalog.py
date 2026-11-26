from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

S3_ROOT = Path(__file__).resolve().parents[1]
HC_ROOT = S3_ROOT.parent
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
sys.path.insert(0, str(S3_ROOT / "src"))

from s3circuit.nvarc_circuit.config import (
    MANIFEST_DATA_CONTRACTS,
    OUTSIDE_B0_DATA_CONTRACT,
    OUTSIDE_B1_DATA_CONTRACT,
    OUTSIDE_SUBSET_DATA_CONTRACTS,
    OUTSIDE_UNION_DATA_CONTRACT,
    REARC8_DATA_CONTRACT,
    REARC8_PUBLIC_EVAL_DATA_CONTRACT,
    REARC8_SEED24_DATA_CONTRACT,
    load_config,
)
from s3circuit.nvarc_circuit.events import (
    build_circuit_events,
    read_catalog_chunk,
    write_catalog_chunk,
)
from s3circuit.nvarc_circuit.identity import (
    dataset_snapshot_identity,
    semantic_sha256,
    sha256_file,
)
from s3circuit.nvarc_circuit.storage import write_json_atomic
from s3circuit.nvarc_circuit.rearc8_contract import (
    EXPECTED_STATS,
    PUBLIC_EVAL_EVENTS,
    PUBLIC_EVAL_PUZZLES,
    PUBLIC_EVAL_ROWS,
    SEED24_EXCLUSIVE_EVENTS,
    SEED24_EXCLUSIVE_PUZZLES,
    SEED24_EXCLUSIVE_ROWS,
    load_and_validate_dataset_manifest,
    normalize_rearc8_event_coordinates,
    validate_catalog_source_counts,
    validate_public_eval_catalog_source_counts,
    validate_seed24_exclusive_catalog_source_counts,
)
from s3circuit.nvarc_circuit.outside_contract import (
    EXPECTED_STATS as OUTSIDE_EXPECTED_STATS,
    load_and_validate_dataset_manifest as load_and_validate_outside_manifest,
    normalize_outside_event_coordinates,
    validate_catalog_source_counts as validate_outside_catalog_source_counts,
)
from s3circuit.nvarc_transcoder.identity import inspect_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Stage-B event catalog for the configured data contract."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rows_per_chunk", type=int, default=256)
    return parser.parse_args()


def _tokenizer(path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        path, use_fast=True, local_files_only=True
    )
    for value in range(10):
        if tokenizer.encode(str(value), add_special_tokens=False) != [value]:
            raise ValueError(f"NVARC digit tokenization changed for {value}")
    expected = {"\n": [10]}
    for text, token_ids in expected.items():
        if tokenizer.encode(text, add_special_tokens=False) != token_ids:
            raise ValueError(f"NVARC tokenizer contract changed for {text!r}")
    return tokenizer


def main() -> None:
    args = parse_args()
    if args.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    config = load_config(args.config)
    output = (
        args.output.resolve()
        if args.output
        else Path(config.artifacts.output_root).resolve() / "catalog"
    )
    output.mkdir(parents=True, exist_ok=True)
    is_rearc8 = config.data.contract in {
        REARC8_DATA_CONTRACT,
        REARC8_PUBLIC_EVAL_DATA_CONTRACT,
        REARC8_SEED24_DATA_CONTRACT,
    }
    is_rearc8_subset = config.data.contract in {
        REARC8_PUBLIC_EVAL_DATA_CONTRACT,
        REARC8_SEED24_DATA_CONTRACT,
    }
    is_outside = config.data.contract in {
        OUTSIDE_UNION_DATA_CONTRACT,
        OUTSIDE_B0_DATA_CONTRACT,
        OUTSIDE_B1_DATA_CONTRACT,
    }
    is_outside_subset = config.data.contract in OUTSIDE_SUBSET_DATA_CONTRACTS
    is_manifest_contract = config.data.contract in MANIFEST_DATA_CONTRACTS
    is_direct_outside_subset = is_outside_subset
    is_source_subset = is_rearc8_subset
    inspect_rows = (
        EXPECTED_STATS["rows"]
        if is_source_subset
        else config.data.expected_train_rows
    )
    inspect_puzzles = (
        EXPECTED_STATS["puzzles"]
        if is_source_subset
        else config.data.expected_train_puzzles
    )
    inspect_events = (
        EXPECTED_STATS["events"]
        if is_source_subset
        else config.data.expected_train_events
    )
    dataset, dataset_value = inspect_dataset(
        config.data.train_path,
        expected_rows=inspect_rows,
        expected_puzzles=inspect_puzzles,
        expected_events=inspect_events,
        expected_generation_modes=(
            EXPECTED_STATS["generation_mode_rows"]
            if is_rearc8
            else (
                (
                    {"cumulative_generated": config.data.expected_train_rows}
                    if is_direct_outside_subset
                    else OUTSIDE_EXPECTED_STATS["generation_mode_rows"]
                )
                if is_outside
                else None
            )
        ),
    )
    if not dataset_value.get("fingerprint"):
        raise ValueError("RE-ARC dataset has no stable fingerprint")
    dataset_manifest = None
    snapshot_schema: dict[str, object] = {
        "columns": sorted(dataset.column_names)
    }
    if is_manifest_contract:
        dataset_manifest = (
            load_and_validate_dataset_manifest(config.data.manifest_path)
            if is_rearc8
            else load_and_validate_outside_manifest(
                config.data.manifest_path,
                selected_sources=config.data.source_names,
            )
        )
        # Direct outside manifests preserve both identities: the generated
        # Dataset fingerprint and the fingerprint written by save_to_disk.
        # load_from_disk exposes the latter, whereas union manifests record
        # that same value as their dataset_fingerprint.
        expected_loaded_fingerprint = dataset_manifest.get(
            "storage_fingerprint", dataset_manifest["dataset_fingerprint"]
        )
        if expected_loaded_fingerprint != dataset_value["fingerprint"]:
            raise ValueError(
                "Arrow storage fingerprint differs from the frozen dataset manifest"
            )
        snapshot_schema.update(
            data_contract=config.data.contract,
            dataset_manifest_sha256=dataset_manifest["sha256"],
        )
    source_dataset_indices = tuple(range(len(dataset)))
    direct_parent_indices: tuple[int, ...] | None = None
    if is_direct_outside_subset:
        direct_parent_indices = tuple(
            int(value)
            for value in np.load(dataset_manifest["source_indices_path"]).tolist()
        )
        if len(direct_parent_indices) != len(dataset):
            raise ValueError("Direct outside parent-index coverage changed")
        snapshot_schema.update(
            source_names=list(config.data.source_names),
            source_indices_sha256=dataset_manifest["source_indices_sha256"],
            row_identity="split_local_row_index_v1",
        )
    if is_source_subset:
        selected_sources = set(config.data.source_names)
        source_dataset_indices = tuple(
            index
            for index, source_name in enumerate(dataset["source_name"])
            if str(source_name) in selected_sources
        )
        dataset = dataset.select(source_dataset_indices)
        puzzle_names = {
            str(value) for value in dataset["puzzle_name"] if value is not None
        }
        event_count = sum(
            1
            if str(mode) == "final_only_gold_prefix"
            else sum(
                1
                for message in messages
                if str(message["role"]).lower() == "assistant"
            )
            for mode, messages in zip(
                dataset["trace_generation_mode"], dataset["messages"]
            )
        )
        if is_outside_subset:
            source = config.data.source_names[0]
            expected_subset = (
                OUTSIDE_EXPECTED_STATS["source_rows"][source],
                400,
                OUTSIDE_EXPECTED_STATS["source_events"][source],
            )
        else:
            expected_subset = (
                (
                    SEED24_EXCLUSIVE_ROWS,
                    SEED24_EXCLUSIVE_PUZZLES,
                    SEED24_EXCLUSIVE_EVENTS,
                )
                if config.data.contract == REARC8_SEED24_DATA_CONTRACT
                else (PUBLIC_EVAL_ROWS, PUBLIC_EVAL_PUZZLES, PUBLIC_EVAL_EVENTS)
            )
        if (
            len(dataset),
            len(puzzle_names),
            event_count,
        ) != expected_subset:
            raise ValueError(
                f"{config.data.contract} source selection changed"
            )
        selection_format = (
            "nvarc_rearc128_outside_source_selection_v1"
            if is_outside_subset
            else (
                "nvarc_rearc8_seed24_exclusive_selection_v1"
                if config.data.contract == REARC8_SEED24_DATA_CONTRACT
                else "nvarc_rearc8_public_eval_selection_v1"
            )
        )
        selection_sha256 = semantic_sha256(
            {
                "format": selection_format,
                "union_fingerprint": dataset_value["fingerprint"],
                "source_names": config.data.source_names,
                "source_dataset_indices": source_dataset_indices,
            }
        )
        dataset_value = {
            **dataset_value,
            "rows": len(dataset),
            "puzzles": len(puzzle_names),
            "events": event_count,
            "fingerprint": selection_sha256,
            "union_fingerprint": dataset_manifest["dataset_fingerprint"],
            "source_names": list(config.data.source_names),
            "source_selection_sha256": selection_sha256,
        }
        snapshot_schema.update(
            source_names=list(config.data.source_names),
            source_selection_sha256=selection_sha256,
        )
    dataset_sha = dataset_snapshot_identity(
        fingerprint=str(dataset_value["fingerprint"]),
        rows=int(dataset_value["rows"]),
        puzzles=int(dataset_value["puzzles"]),
        events=int(dataset_value["events"]),
        schema=snapshot_schema,
    )
    tokenizer = _tokenizer(config.stage_a_config.base.tokenizer)
    chunks = []
    event_keys: set[str] = set()
    work_units: Counter[int] = Counter()
    total_events = 0
    source_rows: Counter[str] = Counter()
    source_events: Counter[str] = Counter()
    generation_mode_rows: Counter[str] = Counter()
    for start in range(0, len(dataset), args.rows_per_chunk):
        end = min(start + args.rows_per_chunk, len(dataset))
        events = []
        for dataset_index in range(start, end):
            row = dataset[dataset_index]
            mode = str(row.get("trace_generation_mode", "cumulative_generated"))
            source_name = (
                str(row["source_name"])
                if row.get("source_name") is not None
                else (
                    config.data.source_names[0]
                    if is_direct_outside_subset
                    else None
                )
            )
            view_index = (
                int(row["view_index"])
                if row.get("view_index") is not None
                else (-1 if is_direct_outside_subset else None)
            )
            test_index = (
                int(row["test_index"])
                if row.get("test_index") is not None
                else (-1 if is_direct_outside_subset else None)
            )
            if is_manifest_contract:
                if source_name is None or view_index is None or test_index is None:
                    raise ValueError(
                        f"Manifest-backed row {dataset_index} lacks source/view/test metadata"
                    )
                if is_outside:
                    view_index, test_index = normalize_outside_event_coordinates(
                        source_name=source_name,
                        trace_generation_mode=mode,
                        view_index=view_index,
                        test_index=test_index,
                    )
                else:
                    view_index, test_index = normalize_rearc8_event_coordinates(
                        source_name=source_name,
                        trace_generation_mode=mode,
                        view_index=view_index,
                        test_index=test_index,
                    )
            row_events = build_circuit_events(
                row["messages"],
                tokenizer,
                dataset_index=source_dataset_indices[dataset_index],
                puzzle_name=(
                    str(row["puzzle_name"])
                    if row.get("puzzle_name") is not None
                    else None
                ),
                dataset_snapshot_sha256=dataset_sha,
                max_sequence_length=config.data.max_sequence_length,
                num_work_units=config.artifacts.num_work_units,
                trace_generation_mode=mode,
                source_name=source_name,
                source_index=(
                    int(row["source_index"])
                    if row.get("source_index") is not None
                    else (
                        direct_parent_indices[dataset_index]
                        if direct_parent_indices is not None
                        else None
                    )
                ),
                view_index=view_index,
                test_index=test_index,
            )
            events.extend(row_events)
            if is_manifest_contract:
                if source_name is None:
                    raise ValueError(
                        f"Manifest-backed row {dataset_index} has no source_name"
                    )
                source_rows[source_name] += 1
                source_events[source_name] += len(row_events)
                generation_mode_rows[mode] += 1
        path = output / f"rows_{start:06d}_{end - 1:06d}.safetensors"
        if path.is_file():
            restored = read_catalog_chunk(
                path,
                expected_identity={
                    "dataset_snapshot_sha256": dataset_sha,
                    "config_sha256": config.semantic_sha256,
                },
            )
            if tuple(events) != restored:
                raise ValueError(f"Existing catalog chunk changed content: {path}")
        else:
            write_catalog_chunk(
                path,
                events,
                dataset_snapshot_sha256=dataset_sha,
                config_sha256=config.semantic_sha256,
            )
        for event in events:
            if event.event_key in event_keys:
                raise ValueError(f"Duplicate event key: {event.event_key}")
            event_keys.add(event.event_key)
            work_units[event.work_unit] += 1
        total_events += len(events)
        chunks.append(
            {
                "path": path.name,
                "row_start": start,
                "row_end": end,
                "events": len(events),
                "tensor_sha256": sha256_file(path),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "catalog",
                    "rows_complete": end,
                    "rows_total": len(dataset),
                    "events_complete": total_events,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if total_events != config.data.expected_train_events:
        raise ValueError(
            f"Catalog has {total_events} events, expected {config.data.expected_train_events}"
        )
    if len(work_units) != config.artifacts.num_work_units:
        raise ValueError("At least one deterministic work unit received no events")
    if config.data.contract == REARC8_DATA_CONTRACT:
        validate_catalog_source_counts(
            source_rows=source_rows,
            source_events=source_events,
            generation_mode_rows=generation_mode_rows,
        )
    elif config.data.contract == REARC8_SEED24_DATA_CONTRACT:
        validate_seed24_exclusive_catalog_source_counts(
            source_rows=source_rows,
            source_events=source_events,
            generation_mode_rows=generation_mode_rows,
        )
    elif config.data.contract == REARC8_PUBLIC_EVAL_DATA_CONTRACT:
        validate_public_eval_catalog_source_counts(
            source_rows=source_rows,
            source_events=source_events,
            generation_mode_rows=generation_mode_rows,
        )
    elif is_outside:
        validate_outside_catalog_source_counts(
            selected_sources=config.data.source_names,
            source_rows=source_rows,
            source_events=source_events,
            generation_mode_rows=generation_mode_rows,
        )
    manifest = {
        "format": "nvarc_rearc_event_catalog_v1",
        "config_sha256": config.semantic_sha256,
        "dataset": dataset_value,
        "dataset_snapshot_sha256": dataset_sha,
        "rows": len(dataset),
        "events": total_events,
        "unique_event_keys": len(event_keys),
        "num_work_units": config.artifacts.num_work_units,
        "data_contract": config.data.contract,
        "source_names": list(config.data.source_names),
        "dataset_manifest_path": (
            dataset_manifest["path"] if dataset_manifest is not None else None
        ),
        "dataset_manifest_sha256": (
            dataset_manifest["sha256"] if dataset_manifest is not None else None
        ),
        "source_rows": dict(sorted(source_rows.items())),
        "source_events": dict(sorted(source_events.items())),
        "generation_mode_rows": dict(sorted(generation_mode_rows.items())),
        "work_unit_counts": {
            str(key): value for key, value in sorted(work_units.items())
        },
        "chunks": chunks,
        "status": "complete",
    }
    write_json_atomic(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(output / "manifest.json"),
                "rows": len(dataset),
                "events": total_events,
                "chunks": len(chunks),
                "work_units": len(work_units),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

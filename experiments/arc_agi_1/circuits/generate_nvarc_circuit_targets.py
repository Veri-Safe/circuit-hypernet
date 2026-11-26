from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path

import torch


S3_ROOT = Path(__file__).resolve().parents[1]
HC_ROOT = S3_ROOT.parent
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
sys.path.insert(0, str(S3_ROOT / "src"))
sys.path.insert(0, str(HC_ROOT / "s1_lora_sft" / "src"))

from s3circuit.nvarc_circuit.config import load_config
from s3circuit.nvarc_circuit.events import read_catalog_chunk
from s3circuit.nvarc_circuit.generation import (
    generate_events,
    load_native_hf_model_and_tokenizer,
    read_generation_chunk,
    write_generation_chunk,
)
from s3circuit.nvarc_circuit.identity import (
    model_identity_from_base_identity,
    sha256_file,
    tokenizer_identity_from_base_identity,
)
from s3circuit.nvarc_circuit.storage import write_json_atomic
from s3circuit.nvarc_circuit.production_schedule import (
    scheduled_work_units,
    validate_production_schedule,
)
from s3circuit.nvarc_circuit.work_units import resolve_work_units
from s3circuit.nvarc_transcoder.identity import load_or_create_base_identity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen targets for one unit or one persistent GPU worker."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--work_unit", type=int)
    parser.add_argument("--worker_index", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument(
        "--work_unit_schedule",
        type=Path,
        help="Optional immutable RE-ARC128 cost-balanced production schedule.",
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--events_per_chunk", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_selected_events(catalog_root, catalog_manifest, selected_work_units):
    selected = set(selected_work_units)
    events = {work_unit: [] for work_unit in selected_work_units}
    for chunk in catalog_manifest["chunks"]:
        path = catalog_root / chunk["path"]
        if sha256_file(path) != chunk["tensor_sha256"]:
            raise ValueError(f"Catalog chunk checksum changed: {path}")
        for event in read_catalog_chunk(path):
            if event.work_unit in selected:
                events[event.work_unit].append(event)
    for work_unit, values in events.items():
        expected = int(catalog_manifest["work_unit_counts"][str(work_unit)])
        if len(values) != expected:
            raise ValueError(
                f"Work unit {work_unit} has {len(values)} events, expected {expected}"
            )
    return events


def _work_unit_output(args, config, work_units, work_unit):
    if args.output:
        root = args.output.resolve()
        return root if len(work_units) == 1 else root / f"work_unit_{work_unit:04d}"
    return (
        Path(config.artifacts.output_root).resolve()
        / "generations"
        / f"work_unit_{work_unit:04d}"
    )


def _generate_work_unit(
    *,
    args,
    config,
    catalog_manifest,
    work_units,
    work_unit,
    events,
    model,
    token_contract,
    device,
    model_identity_sha,
    tokenizer_identity_sha,
):
    output = _work_unit_output(args, config, work_units, work_unit)
    output.mkdir(parents=True, exist_ok=True)
    chunks = []
    complete = 0
    for chunk_index, start in enumerate(range(0, len(events), args.events_per_chunk)):
        batch_events = events[start : start + args.events_per_chunk]
        path = output / f"part_{chunk_index:05d}.safetensors"
        expected_keys = tuple(event.event_key for event in batch_events)
        if path.is_file():
            records = read_generation_chunk(
                path,
                expected_identity={
                    "config_sha256": config.semantic_sha256,
                    "model_identity_sha256": model_identity_sha,
                    "tokenizer_identity_sha256": tokenizer_identity_sha,
                    "decoder": "constrained_greedy_full_softmax_v1",
                    "batch_size": args.batch_size,
                },
            )
            if tuple(record.event_key for record in records) != expected_keys:
                raise ValueError(f"Existing generation chunk changed coverage: {path}")
        else:
            records = []
            for batch_start in range(0, len(batch_events), args.batch_size):
                records.extend(
                    generate_events(
                        model,
                        batch_events[batch_start : batch_start + args.batch_size],
                        token_contract=token_contract,
                        max_new_tokens=config.generation.max_new_tokens,
                        max_sequence_length=config.data.max_sequence_length,
                        device=device,
                    )
                )
            write_generation_chunk(
                path,
                records,
                config_sha256=config.semantic_sha256,
                model_identity_sha256=model_identity_sha,
                tokenizer_identity_sha256=tokenizer_identity_sha,
                batch_size=args.batch_size,
            )
        complete += len(batch_events)
        chunks.append(
            {
                "path": path.name,
                "events": len(batch_events),
                "tensor_sha256": sha256_file(path),
                "first_event_key": expected_keys[0],
                "last_event_key": expected_keys[-1],
            }
        )
        progress = {
            "phase": "generation",
            "worker_index": args.worker_index,
            "work_unit": work_unit,
            "events_complete": complete,
            "events_total": len(events),
        }
        write_json_atomic(output / "progress.json", progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
    manifest = {
        "format": "nvarc_rearc_frozen_generation_work_unit_v1",
        "status": "complete",
        "config_sha256": config.semantic_sha256,
        "dataset_snapshot_sha256": catalog_manifest["dataset_snapshot_sha256"],
        "model_identity_sha256": model_identity_sha,
        "tokenizer_identity_sha256": tokenizer_identity_sha,
        "work_unit": work_unit,
        "events": len(events),
        "batch_size": args.batch_size,
        "chunks": chunks,
    }
    write_json_atomic(output / "manifest.json", manifest)
    write_json_atomic(
        output / "progress.json",
        {
            "phase": "generation",
            "status": "complete",
            "worker_index": args.worker_index,
            "work_unit": work_unit,
            "events_complete": len(events),
            "events_total": len(events),
        },
    )
    return manifest


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.work_unit_schedule is not None:
        if args.work_unit is not None or args.worker_index is None or args.num_workers is None:
            raise ValueError(
                "A production schedule requires --worker_index/--num_workers only"
            )
        schedule = json.loads(
            args.work_unit_schedule.resolve().read_text(encoding="utf-8")
        )
        validate_production_schedule(schedule, config_sha256=config.semantic_sha256)
        work_units = scheduled_work_units(
            schedule,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
        )
    else:
        work_units = resolve_work_units(
            total_work_units=config.artifacts.num_work_units,
            work_unit=args.work_unit,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
        )
    if args.batch_size not in config.generation.batch_candidates:
        raise ValueError("batch_size is not a declared audit candidate")
    if args.events_per_chunk <= 0:
        raise ValueError("events_per_chunk must be positive")
    catalog_root = args.catalog.resolve()
    catalog_manifest = json.loads(
        (catalog_root / "manifest.json").read_text(encoding="utf-8")
    )
    expected_catalog = {
        "format": "nvarc_rearc_event_catalog_v1",
        "config_sha256": config.semantic_sha256,
        "status": "complete",
    }
    mismatches = {
        key: {"expected": value, "actual": catalog_manifest.get(key)}
        for key, value in expected_catalog.items()
        if catalog_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Catalog identity changed: {mismatches}")
    events_by_work_unit = _load_selected_events(
        catalog_root, catalog_manifest, work_units
    )
    identity_root = Path(config.artifacts.output_root).resolve() / "identities"
    base_identity = load_or_create_base_identity(
        config.stage_a_config,
        destination=identity_root / "base_identity.json",
        allow_hc_root_mount_alias=True,
    )
    model_identity_sha = model_identity_from_base_identity(base_identity)
    tokenizer_identity_sha = tokenizer_identity_from_base_identity(base_identity)
    device = torch.device(args.device)
    model, _tokenizer, token_contract = load_native_hf_model_and_tokenizer(
        checkpoint=config.stage_a_config.base.hf_checkpoint,
        tokenizer_path=config.stage_a_config.base.tokenizer,
        device=device,
        attention_backend=config.generation.attention_backend,
    )
    manifests = []
    for work_unit in work_units:
        manifests.append(
            _generate_work_unit(
                args=args,
                config=config,
                catalog_manifest=catalog_manifest,
                work_units=work_units,
                work_unit=work_unit,
                events=events_by_work_unit[work_unit],
                model=model,
                token_contract=token_contract,
                device=device,
                model_identity_sha=model_identity_sha,
                tokenizer_identity_sha=tokenizer_identity_sha,
            )
        )
    print(
        json.dumps(
            {
                "status": "complete",
                "worker_index": args.worker_index,
                "num_workers": args.num_workers,
                "work_units": len(manifests),
                "events": sum(int(manifest["events"]) for manifest in manifests),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

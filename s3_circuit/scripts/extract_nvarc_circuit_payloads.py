from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections import Counter
import json
import os
import socket
import sys
import time
import traceback
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
    LEGACY_REPLAY_VALIDATION,
    S2_KV_FROZEN_TRAJECTORY_VALIDATION,
    TokenContract,
    merge_replay_validation_mode,
    read_generation_chunk,
)
from s3circuit.nvarc_circuit.identity import sha256_file
from s3circuit.nvarc_circuit.payload import (
    CompactPayload,
    read_payload_chunk,
    write_payload_chunk,
)
from s3circuit.nvarc_circuit.policy import (
    add_compute_microbatch_argument,
    load_production_policy,
)
from s3circuit.nvarc_circuit.storage import write_json_atomic
from s3circuit.nvarc_circuit.transcoder_adapter import (
    build_codecircuit_transcoder_set,
    load_canonical_bank,
)
from s3circuit.nvarc_circuit.worker import (
    build_event_artifact_id,
    extract_event_payload,
)
from nvarc_qwen3.nvarc_grid_grammar import GridGrammarState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract compact Stage-B payloads for one deterministic work unit."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    add_compute_microbatch_argument(parser)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--work_unit", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument(
        "--max_exact_row_cache_bytes", type=int, default=214_748_364_800
    )
    parser.add_argument(
        "--cpu_threads", type=int, default=int(os.environ.get("S3_CPU_THREADS", "16"))
    )
    parser.add_argument(
        "--replay_validation_mode",
        choices=(LEGACY_REPLAY_VALIDATION, S2_KV_FROZEN_TRAJECTORY_VALIDATION),
        default=LEGACY_REPLAY_VALIDATION,
    )
    return parser.parse_args()


def _payload_bytes(payload: CompactPayload) -> int:
    total = 0
    for value in payload.__dict__.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def _load_events(catalog_root: Path, work_unit: int):
    manifest = json.loads((catalog_root / "manifest.json").read_text(encoding="utf-8"))
    events = []
    for chunk in manifest["chunks"]:
        events.extend(
            event
            for event in read_catalog_chunk(catalog_root / chunk["path"])
            if event.work_unit == work_unit
        )
    return manifest, events


def _load_generations(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = {}
    for chunk in manifest["chunks"]:
        for record in read_generation_chunk(root / chunk["path"]):
            if record.event_key in records:
                raise ValueError(f"Duplicate frozen generation: {record.event_key}")
            records[record.event_key] = record
    return manifest, records


def main() -> None:
    args = parse_args()
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    config = load_config(args.config)
    source_policy = load_production_policy(args.policy, config=config)
    policy = source_policy.with_compute_microbatch_cap(args.compute_microbatch_cap)
    if not 0 <= args.work_unit < config.artifacts.num_work_units:
        raise ValueError("work_unit is outside the frozen range")
    catalog_manifest, events = _load_events(args.catalog.resolve(), args.work_unit)
    generation_root = args.generations.resolve()
    generation_manifest, generations = _load_generations(generation_root)
    generation_manifest_sha256 = sha256_file(generation_root / "manifest.json")
    expected = {
        "config_sha256": config.semantic_sha256,
        "dataset_snapshot_sha256": catalog_manifest["dataset_snapshot_sha256"],
        "model_identity_sha256": policy.model_identity_sha256,
        "tokenizer_identity_sha256": policy.tokenizer_identity_sha256,
        "work_unit": args.work_unit,
        "status": "complete",
    }
    mismatches = {
        key: {"expected": value, "actual": generation_manifest.get(key)}
        for key, value in expected.items()
        if generation_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen generation identity changed: {mismatches}")
    if set(generations) != {event.event_key for event in events}:
        raise ValueError("Catalog and generation work-unit coverage differ")
    output = (
        args.output.resolve()
        if args.output
        else Path(config.artifacts.output_root).resolve()
        / "payloads"
        / f"work_unit_{args.work_unit:04d}"
    )
    output.mkdir(parents=True, exist_ok=True)
    policy.save_for_extraction(output)
    manifest_path = output / "manifest.json"
    manifest = {
        "format": "nvarc_rearc_payload_work_unit_v1",
        "status": "running",
        "config_sha256": config.semantic_sha256,
        "production_policy_sha256": policy.semantic_sha256,
        "source_generation_manifest_sha256": generation_manifest_sha256,
        "work_unit": args.work_unit,
        "expected_events": len(events),
        "replay_validation_modes": [args.replay_validation_mode],
        "chunks": [],
    }
    committed: set[str] = set()
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "format",
            "config_sha256",
            "production_policy_sha256",
            "source_generation_manifest_sha256",
            "work_unit",
            "expected_events",
        ):
            if existing.get(key) != manifest[key]:
                raise ValueError(f"Payload resume identity changed at {key}")
        manifest = existing
        manifest["replay_validation_modes"] = merge_replay_validation_mode(
            manifest.get("replay_validation_modes"),
            args.replay_validation_mode,
        )
        for chunk in manifest["chunks"]:
            path = output / chunk["path"]
            if sha256_file(path) != chunk["tensor_sha256"]:
                raise ValueError(f"Committed payload checksum changed: {path}")
            for payload in read_payload_chunk(
                path,
                expected_identity={
                    "config_sha256": config.semantic_sha256,
                    "production_policy_sha256": policy.semantic_sha256,
                    "root_view": "joint_signed_raw_sum",
                },
            ):
                if payload.event_key in committed:
                    raise ValueError(
                        f"Duplicate committed payload: {payload.event_key}"
                    )
                committed.add(payload.event_key)
        write_json_atomic(manifest_path, manifest)
    pending = [event for event in events if event.event_key not in committed]
    device = torch.device(args.device)
    bank = load_canonical_bank(
        config.stage_a.canonical_export_manifest,
        stage_a_config=config.stage_a_config,
    )
    if bank.identity_sha256 != policy.transcoder_identity_sha256:
        raise ValueError("Canonical transcoder identity differs from production policy")
    transcoders = build_codecircuit_transcoder_set(bank, device=device)
    from s3circuit.nvarc_circuit.tl_bridge import build_local_qwen3_replacement_model

    bridge = build_local_qwen3_replacement_model(
        checkpoint=config.stage_a_config.base.hf_checkpoint,
        tokenizer_path=config.stage_a_config.base.tokenizer,
        transcoders=transcoders,
        device=device,
        manifest_path=output / "tl_conversion_manifest.json",
        numerics_backend=getattr(policy, "numerics_backend", "tl_sdpa_direct_effect_numerics_v1"),
    )
    token_contract = TokenContract(
        digit_token_ids=tuple(range(10)),
        newline_token_id=10,
        pad_token_id=int(bridge.tokenizer.pad_token_id),
        eos_token_id=int(bridge.tokenizer.eos_token_id),
        vocab_size=int(bridge.model.cfg.d_vocab),
    )
    token_contract.validate()
    buffered: list[CompactPayload] = []
    buffered_bytes = 0
    failure_history_path = output / "failure_attempts.json"
    if failure_history_path.is_file():
        failure_history = json.loads(failure_history_path.read_text(encoding="utf-8"))
        if not isinstance(failure_history, list):
            raise ValueError("Payload failure-attempt history is not a list")
    else:
        failure_history = []
    attempt_counts = Counter(str(record.get("event_key")) for record in failure_history)
    failures = []
    worker_identity = f"{socket.gethostname()}:{os.getpid()}:{args.device}"
    started = time.monotonic()
    last_runtime_metrics: dict[str, int] = {}
    last_event_seconds = 0.0

    def flush() -> None:
        nonlocal buffered, buffered_bytes, manifest
        if not buffered:
            return
        chunk_index = len(manifest["chunks"])
        path = output / f"part_{chunk_index:05d}.safetensors"
        if path.is_file():
            restored = read_payload_chunk(
                path,
                expected_identity={
                    "config_sha256": config.semantic_sha256,
                    "production_policy_sha256": policy.semantic_sha256,
                    "root_view": "joint_signed_raw_sum",
                },
            )
            expected_identity = tuple(
                (payload.event_key, payload.artifact_id) for payload in buffered
            )
            actual_identity = tuple(
                (payload.event_key, payload.artifact_id) for payload in restored
            )
            if actual_identity != expected_identity:
                raise ValueError(f"Orphan payload chunk changed coverage: {path}")
        else:
            write_payload_chunk(
                path,
                buffered,
                config_sha256=config.semantic_sha256,
                production_policy_sha256=policy.semantic_sha256,
            )
        manifest["chunks"].append(
            {
                "path": path.name,
                "events": len(buffered),
                "bytes": path.stat().st_size,
                "tensor_sha256": sha256_file(path),
                "first_event_key": buffered[0].event_key,
                "last_event_key": buffered[-1].event_key,
            }
        )
        manifest["completed_events"] = sum(
            int(chunk["events"]) for chunk in manifest["chunks"]
        )
        write_json_atomic(manifest_path, manifest)
        buffered = []
        buffered_bytes = 0

    for index, event in enumerate(pending, start=1):
        generation = generations[event.event_key]
        identifier = None
        try:
            event_started = time.monotonic()
            runtime_metrics: dict[str, int] = {}
            identifier = build_event_artifact_id(
                event=event,
                generation=generation,
                config=config,
                policy=policy,
            )
            payload = extract_event_payload(
                model=bridge.model,
                event=event,
                generation=generation,
                config=config,
                policy=policy,
                token_contract=token_contract,
                grammar_factory=GridGrammarState,
                runtime_metrics=runtime_metrics,
                max_exact_row_cache_bytes=args.max_exact_row_cache_bytes,
                require_replay_argmax=(
                    args.replay_validation_mode
                    != S2_KV_FROZEN_TRAJECTORY_VALIDATION
                ),
                align_target_diagnostics_to_span_replay=(
                    args.replay_validation_mode
                    == S2_KV_FROZEN_TRAJECTORY_VALIDATION
                ),
            )
            last_event_seconds = time.monotonic() - event_started
            last_runtime_metrics = runtime_metrics
            size = _payload_bytes(payload)
            if (
                buffered
                and buffered_bytes + size > config.artifacts.payload_target_bytes
            ):
                flush()
            buffered.append(payload)
            buffered_bytes += size
        except Exception as error:
            attempt_counts[event.event_key] += 1
            failure = {
                "event_key": event.event_key,
                "artifact_id": identifier,
                "stage": "payload_extraction",
                "exception": type(error).__name__,
                "message": str(error)[:2000],
                "traceback": traceback.format_exc(limit=8),
                "retry_count": attempt_counts[event.event_key],
                "worker_identity": worker_identity,
                "generation_manifest_sha256": generation_manifest_sha256,
            }
            failures.append(failure)
            failure_history.append(failure)
            write_json_atomic(output / "failures.json", failures)
            write_json_atomic(failure_history_path, failure_history)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        if index % args.progress_every == 0 or index == len(pending):
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = index / elapsed * 3600
            remaining = len(pending) - index
            progress = {
                "phase": "payload_extraction",
                "work_unit": args.work_unit,
                "attempted": index,
                "pending_total": len(pending),
                "committed_before_resume": len(committed),
                "failures": len(failures),
                "events_per_hour": rate,
                "eta_seconds": remaining / max(rate / 3600, 1e-9),
                "buffered_bytes": buffered_bytes,
                "last_event_seconds": last_event_seconds,
                "cpu_threads": args.cpu_threads,
                "replay_validation_mode": args.replay_validation_mode,
                **last_runtime_metrics,
                "cuda_allocated_bytes": (
                    torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
                ),
            }
            write_json_atomic(output / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
    flush()
    completed = sum(int(chunk["events"]) for chunk in manifest["chunks"])
    manifest["completed_events"] = completed
    manifest["failures"] = len(failures)
    manifest["status"] = (
        "complete" if completed == len(events) and not failures else "incomplete"
    )
    write_json_atomic(output / "failures.json", failures)
    write_json_atomic(manifest_path, manifest)
    if manifest["status"] != "complete":
        raise RuntimeError(
            f"Payload work unit is incomplete: completed={completed}/{len(events)}, "
            f"failures={len(failures)}"
        )
    write_json_atomic(
        output / "progress.json",
        {
            "phase": "payload_extraction",
            "status": "complete",
            "work_unit": args.work_unit,
            "events_complete": completed,
            "events_total": len(events),
            "failures": 0,
        },
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

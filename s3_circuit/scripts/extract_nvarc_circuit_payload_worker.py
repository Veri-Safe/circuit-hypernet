from __future__ import annotations

# ruff: noqa: E402

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import traceback
import threading
from typing import TextIO

import torch


S3_ROOT = Path(__file__).resolve().parents[1]
HC_ROOT = S3_ROOT.parent
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
sys.path.insert(0, str(S3_ROOT / "src"))
sys.path.insert(0, str(HC_ROOT / "s1_lora_sft" / "src"))

from nvarc_qwen3.nvarc_grid_grammar import GridGrammarState
from s3circuit.nvarc_circuit.config import load_config
from s3circuit.nvarc_circuit.artifact_lifecycle import (
    validate_finalized_work_unit_for_payload_skip,
)
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
from s3circuit.nvarc_circuit.repair_schedule import (
    scheduled_work_units,
    validate_repair_schedule,
)
from s3circuit.nvarc_circuit.production_schedule import (
    scheduled_work_units as scheduled_production_work_units,
    validate_production_schedule,
)
from s3circuit.nvarc_circuit.storage import write_json_atomic
from s3circuit.nvarc_circuit.transcoder_adapter import (
    build_codecircuit_transcoder_set,
    load_canonical_bank,
)
from s3circuit.nvarc_circuit.work_units import resolve_work_units
from s3circuit.nvarc_circuit.worker import (
    build_event_artifact_id,
    extract_event_payload,
    finalize_event_payload_pipeline,
    prepare_event_payload_pipeline,
)


class StorageBackpressureError(RuntimeError):
    """Stop a resumable producer before an idle GPU allocation can time out."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract several deterministic Stage-B work units with one persistent "
            "model and transcoder load."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    add_compute_microbatch_argument(parser)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--generations_root", type=Path, required=True)
    parser.add_argument("--worker_index", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument(
        "--work_unit_schedule",
        type=Path,
        help=(
            "Optional immutable repair schedule. When provided, its explicit "
            "work-unit assignment replaces the modulo worker partition."
        ),
    )
    parser.add_argument(
        "--production_schedule",
        type=Path,
        help="Optional immutable cost-balanced production schedule.",
    )
    parser.add_argument(
        "--worker_state_root",
        type=Path,
        help=(
            "Optional worker metadata root, used by repair jobs to avoid "
            "overwriting original production worker reports."
        ),
    )
    parser.add_argument(
        "--work_unit_lock_root",
        type=Path,
        help=(
            "Optional shared lock directory used to coordinate payload producers "
            "with independent CPU materializers."
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument(
        "--final_root",
        type=Path,
        help=(
            "Validated CPU artifact root. Completed work units whose payloads were "
            "safely deleted are skipped during GPU resume."
        ),
    )
    parser.add_argument("--progress_every", type=int, default=1)
    parser.add_argument(
        "--max_exact_row_cache_bytes", type=int, default=214_748_364_800
    )
    parser.add_argument(
        "--min_free_storage_bytes", type=int, default=5_000_000_000_000
    )
    parser.add_argument("--storage_wait_timeout_seconds", type=float, default=0.0)
    parser.add_argument("--storage_poll_seconds", type=float, default=60.0)
    parser.add_argument(
        "--numerics_backend",
        help="Defaults to the backend in the input policy when specified.",
    )
    parser.add_argument(
        "--materialize_final_artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--materializer_parallelism", type=int, default=1)
    parser.add_argument("--materializer_cpu_threads", type=int, default=2)
    parser.add_argument(
        "--materializer",
        type=Path,
        default=S3_ROOT / "scripts" / "materialize_nvarc_circuit_artifacts.py",
        help="Supply the compatible materialization script when enabling automatic final artifacts.",
    )
    parser.add_argument("--materializer_log_root", type=Path)
    parser.add_argument(
        "--cpu_threads", type=int, default=int(os.environ.get("S3_CPU_THREADS", "16"))
    )
    parser.add_argument(
        "--replay_validation_mode",
        choices=(LEGACY_REPLAY_VALIDATION, S2_KV_FROZEN_TRAJECTORY_VALIDATION),
        default=LEGACY_REPLAY_VALIDATION,
        help=(
            "The S2 KV mode keeps frozen generation tokens and aligns diagnostics "
            "to the attribution span; the legacy mode requires replay greedy parity."
        ),
    )
    parser.add_argument(
        "--event_lanes",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Use two exact event lanes per physical GPU to overlap CPU ranking "
            "with serialized CUDA work. OOM retries use one lane."
        ),
    )
    return parser.parse_args()


def _payload_bytes(payload: CompactPayload) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in payload.__dict__.values()
        if isinstance(value, torch.Tensor)
    )


@contextmanager
def _work_unit_lock(root: Path | None, work_unit: int):
    if root is None:
        yield
        return
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    path = root / f"work_unit_{work_unit:04d}.lock"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _ensure_storage(
    root: Path,
    minimum_free_bytes: int,
    *,
    wait_timeout_seconds: float = 0.0,
    poll_seconds: float = 60.0,
) -> int:
    started = time.monotonic()
    while True:
        free_bytes = int(shutil.disk_usage(root).free)
        if free_bytes >= minimum_free_bytes:
            return free_bytes
        elapsed = time.monotonic() - started
        if wait_timeout_seconds <= 0 or elapsed >= wait_timeout_seconds:
            raise StorageBackpressureError(
                "Shared storage reserve breached after backpressure wait: "
                f"{free_bytes} < {minimum_free_bytes}; waited={elapsed:.1f}s"
            )
        status = {
            "phase": "storage_backpressure",
            "free_storage_bytes": free_bytes,
            "minimum_free_storage_bytes": minimum_free_bytes,
            "waited_seconds": elapsed,
            "timeout_seconds": wait_timeout_seconds,
        }
        print(json.dumps(status, sort_keys=True), flush=True)
        time.sleep(min(poll_seconds, max(wait_timeout_seconds - elapsed, 0.1)))


def _load_selected_events(
    catalog_root: Path, work_units: tuple[int, ...], *, config_sha256: str
):
    manifest_path = catalog_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format": "nvarc_rearc_event_catalog_v1",
        "status": "complete",
        "config_sha256": config_sha256,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Catalog identity changed: {mismatches}")
    selected = set(work_units)
    events = {work_unit: [] for work_unit in work_units}
    for chunk in manifest["chunks"]:
        path = catalog_root / chunk["path"]
        if sha256_file(path) != chunk["tensor_sha256"]:
            raise ValueError(f"Catalog chunk checksum changed: {path}")
        for event in read_catalog_chunk(path):
            if event.work_unit in selected:
                events[event.work_unit].append(event)
    for work_unit, values in events.items():
        expected_count = int(manifest["work_unit_counts"][str(work_unit)])
        if len(values) != expected_count:
            raise ValueError(
                f"Work unit {work_unit} has {len(values)} events, "
                f"expected {expected_count}"
            )
    return manifest, events


def _load_generations(root: Path):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    records = {}
    for chunk in manifest["chunks"]:
        path = root / chunk["path"]
        if sha256_file(path) != chunk["tensor_sha256"]:
            raise ValueError(f"Generation chunk checksum changed: {path}")
        for record in read_generation_chunk(path):
            if record.event_key in records:
                raise ValueError(f"Duplicate frozen generation: {record.event_key}")
            records[record.event_key] = record
    return manifest, records


def _extract_work_unit(
    *,
    args: argparse.Namespace,
    config,
    policy,
    catalog_manifest: dict,
    events: list,
    work_unit: int,
    bridge,
    token_contract: TokenContract,
    event_lanes: int,
    dual_lane_replay_threshold: int | None,
) -> dict:
    generation_root = args.generations_root.resolve() / f"work_unit_{work_unit:04d}"
    generation_manifest, generations = _load_generations(generation_root)
    generation_manifest_sha256 = sha256_file(generation_root / "manifest.json")
    expected_generation = {
        "config_sha256": config.semantic_sha256,
        "dataset_snapshot_sha256": catalog_manifest["dataset_snapshot_sha256"],
        "model_identity_sha256": policy.model_identity_sha256,
        "tokenizer_identity_sha256": policy.tokenizer_identity_sha256,
        "work_unit": work_unit,
        "status": "complete",
    }
    mismatches = {
        key: {"expected": value, "actual": generation_manifest.get(key)}
        for key, value in expected_generation.items()
        if generation_manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Frozen generation identity changed: {mismatches}")
    if set(generations) != {event.event_key for event in events}:
        raise ValueError("Catalog and generation work-unit coverage differ")

    output = args.output_root.resolve() / f"work_unit_{work_unit:04d}"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = {
        "format": "nvarc_rearc_payload_work_unit_v1",
        "status": "running",
        "config_sha256": config.semantic_sha256,
        "production_policy_sha256": policy.semantic_sha256,
        "source_generation_manifest_sha256": generation_manifest_sha256,
        "work_unit": work_unit,
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
                    raise ValueError(f"Duplicate committed payload: {payload.event_key}")
                committed.add(payload.event_key)
        write_json_atomic(manifest_path, manifest)
    pending = [event for event in events if event.event_key not in committed]
    if not pending and manifest.get("status") == "complete":
        return manifest

    buffered: list[CompactPayload] = []
    buffered_bytes = 0
    failure_history_path = output / "failure_attempts.json"
    failure_history = (
        json.loads(failure_history_path.read_text(encoding="utf-8"))
        if failure_history_path.is_file()
        else []
    )
    if not isinstance(failure_history, list):
        raise ValueError("Payload failure-attempt history is not a list")
    attempt_counts = Counter(str(record.get("event_key")) for record in failure_history)
    failures = []
    worker_identity = f"{socket.gethostname()}:{os.getpid()}:{args.device}"
    started = time.monotonic()
    last_runtime_metrics: dict[str, int | float | bool | str] = {}
    last_event_seconds = 0.0
    dual_lane_fallbacks = 0
    long_event_single_lane_fallbacks = 0
    gpu_lock = threading.Lock() if event_lanes == 2 else None

    def flush() -> None:
        nonlocal buffered, buffered_bytes, manifest
        if not buffered:
            return
        free_bytes = _ensure_storage(
            output,
            args.min_free_storage_bytes,
            wait_timeout_seconds=args.storage_wait_timeout_seconds,
            poll_seconds=args.storage_poll_seconds,
        )
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
                "free_storage_bytes_after": free_bytes,
            }
        )
        manifest["completed_events"] = sum(
            int(chunk["events"]) for chunk in manifest["chunks"]
        )
        write_json_atomic(manifest_path, manifest)
        buffered = []
        buffered_bytes = 0

    def extract_one(event, *, fallback_to_single_lane: bool = False):
        generation = generations[event.event_key]
        event_started = time.monotonic()
        runtime_metrics: dict[str, int | float | bool | str] = {
            "event_lanes": event_lanes,
            "single_lane_fallback": fallback_to_single_lane,
        }
        identifier = build_event_artifact_id(
            event=event,
            generation=generation,
            config=config,
            policy=policy,
        )
        if event_lanes == 2 and not fallback_to_single_lane:
            prepared = prepare_event_payload_pipeline(
                model=bridge.model,
                event=event,
                generation=generation,
                config=config,
                policy=policy,
                token_contract=token_contract,
                grammar_factory=GridGrammarState,
                max_exact_row_cache_bytes=args.max_exact_row_cache_bytes,
                gpu_lock=gpu_lock,
                require_replay_argmax=(
                    args.replay_validation_mode
                    != S2_KV_FROZEN_TRAJECTORY_VALIDATION
                ),
                align_target_diagnostics_to_span_replay=(
                    args.replay_validation_mode
                    == S2_KV_FROZEN_TRAJECTORY_VALIDATION
                ),
            )
            runtime_metrics.update(prepared.replay_metrics)
            payload = finalize_event_payload_pipeline(
                prepared, runtime_metrics=runtime_metrics
            )
        else:
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
        return identifier, payload, runtime_metrics, time.monotonic() - event_started

    attempted = 0
    batches: list[list] = []
    pending_pair: list = []
    for event in pending:
        generation = generations[event.event_key]
        replay_length = event.condition_length + generation.target_count - 1
        dual_eligible = (
            event_lanes == 2
            and dual_lane_replay_threshold is not None
            and replay_length <= dual_lane_replay_threshold
        )
        if dual_eligible:
            pending_pair.append(event)
            if len(pending_pair) == 2:
                batches.append(pending_pair)
                pending_pair = []
        else:
            if pending_pair:
                batches.append(pending_pair)
                pending_pair = []
            batches.append([event])
    if pending_pair:
        batches.append(pending_pair)
    for batch in batches:
        if event_lanes == 2 and len(batch) == 2:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-prod-event") as executor:
                outcomes = []
                futures = [(event, executor.submit(extract_one, event)) for event in batch]
                for event, future in futures:
                    try:
                        outcomes.append((event, future.result(), None))
                    except Exception as error:  # noqa: BLE001
                        outcomes.append((event, None, error))
        else:
            outcomes = []
            for event in batch:
                try:
                    generation = generations[event.event_key]
                    replay_length = event.condition_length + generation.target_count - 1
                    long_single_fallback = (
                        event_lanes == 2
                        and dual_lane_replay_threshold is not None
                        and replay_length > dual_lane_replay_threshold
                    )
                    if long_single_fallback:
                        long_event_single_lane_fallbacks += 1
                    outcomes.append(
                        (
                            event,
                            extract_one(
                                event,
                                fallback_to_single_lane=long_single_fallback,
                            ),
                            None,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    outcomes.append((event, None, error))

        for event, outcome, error in outcomes:
            attempted += 1
            identifier = None
            if isinstance(error, torch.cuda.OutOfMemoryError) and event_lanes == 2:
                dual_lane_fallbacks += 1
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    outcome = extract_one(event, fallback_to_single_lane=True)
                    error = None
                except Exception as retry_error:  # noqa: BLE001
                    error = retry_error
            try:
                if error is not None or outcome is None:
                    if error is not None:
                        raise error
                    raise RuntimeError("Missing event outcome")
                identifier, payload, runtime_metrics, last_event_seconds = outcome
                last_runtime_metrics = runtime_metrics
                size = _payload_bytes(payload)
                if (
                    buffered
                    and buffered_bytes + size > config.artifacts.payload_target_bytes
                ):
                    flush()
                buffered.append(payload)
                buffered_bytes += size
            except StorageBackpressureError:
                raise
            except Exception as error:  # noqa: BLE001 - persist every failed event
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
            if attempted % args.progress_every == 0 or attempted == len(pending):
                elapsed = max(time.monotonic() - started, 1e-9)
                rate = attempted / elapsed * 3600
                remaining = len(pending) - attempted
                progress = {
                    "phase": "payload_extraction",
                    "worker_index": args.worker_index,
                    "num_workers": args.num_workers,
                    "work_unit": work_unit,
                    "attempted": attempted,
                    "pending_total": len(pending),
                    "committed_before_resume": len(committed),
                    "failures": len(failures),
                    "events_per_hour": rate,
                    "eta_seconds": remaining / max(rate / 3600, 1e-9),
                    "buffered_bytes": buffered_bytes,
                    "last_event_seconds": last_event_seconds,
                    "cpu_threads": args.cpu_threads,
                    "replay_validation_mode": args.replay_validation_mode,
                    "event_lanes": event_lanes,
                    "dual_lane_oom_fallbacks": dual_lane_fallbacks,
                    "long_event_single_lane_fallbacks": (
                        long_event_single_lane_fallbacks
                    ),
                    **last_runtime_metrics,
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(args.device),
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
            "worker_index": args.worker_index,
            "num_workers": args.num_workers,
            "work_unit": work_unit,
            "events_complete": completed,
            "events_total": len(events),
            "failures": 0,
            "event_lanes": event_lanes,
            "dual_lane_oom_fallbacks": dual_lane_fallbacks,
            "long_event_single_lane_fallbacks": long_event_single_lane_fallbacks,
        },
    )
    return manifest


def _schedule_policy_sha256(schedule, source_policy, effective_policy) -> str:
    recorded = schedule.get("production_policy_sha256")
    if recorded not in (source_policy.semantic_sha256, effective_policy.semantic_sha256):
        raise ValueError("Schedule was issued for a different production policy")
    return recorded


def main() -> None:
    args = parse_args()
    if (
        args.cpu_threads <= 0
        or args.progress_every <= 0
        or args.storage_wait_timeout_seconds < 0
        or args.storage_poll_seconds <= 0
        or args.materializer_parallelism <= 0
        or args.materializer_cpu_threads <= 0
    ):
        raise ValueError("CPU threads and progress interval must be positive")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    config = load_config(args.config)
    source_policy = load_production_policy(args.policy, config=config)
    policy = source_policy.with_compute_microbatch_cap(args.compute_microbatch_cap)
    policy_numerics_backend = getattr(policy, "numerics_backend", None)
    if args.numerics_backend is None:
        args.numerics_backend = policy_numerics_backend or "tl_sdpa_direct_effect_numerics_v1"
    if (
        policy_numerics_backend is not None
        and args.numerics_backend != policy_numerics_backend
    ):
        raise ValueError("Payload worker numerics backend differs from its policy")
    if args.materialize_final_artifacts and not args.materializer.is_file():
        raise FileNotFoundError(args.materializer)
    repair_schedule = None
    production_schedule = None
    effective_event_lanes = args.event_lanes
    dual_lane_replay_threshold = None
    if args.work_unit_schedule and args.production_schedule:
        raise ValueError("Repair and production schedules are mutually exclusive")
    if args.production_schedule:
        production_schedule = json.loads(
            args.production_schedule.resolve().read_text(encoding="utf-8")
        )
        validate_production_schedule(
            production_schedule,
            config_sha256=config.semantic_sha256,
            production_policy_sha256=_schedule_policy_sha256(
                production_schedule, source_policy, policy
            ),
            catalog_manifest_sha256=sha256_file(
                args.catalog.resolve() / "manifest.json"
            ),
        )
        work_units = scheduled_production_work_units(
            production_schedule,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
        )
        dual_lane_replay_threshold = int(production_schedule["long_replay_threshold"])
    elif args.event_lanes != 1:
        raise ValueError("Dual event lanes require an immutable production schedule")
    elif args.work_unit_schedule:
        repair_schedule = json.loads(
            args.work_unit_schedule.resolve().read_text(encoding="utf-8")
        )
        validate_repair_schedule(
            repair_schedule,
            config_sha256=config.semantic_sha256,
            production_policy_sha256=_schedule_policy_sha256(
                repair_schedule, source_policy, policy
            ),
            catalog_manifest_sha256=sha256_file(
                args.catalog.resolve() / "manifest.json"
            ),
        )
        work_units = scheduled_work_units(
            repair_schedule,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
        )
        if not work_units:
            raise ValueError(
                f"Repair schedule assigned no work units to worker {args.worker_index}"
            )
    else:
        work_units = resolve_work_units(
            total_work_units=config.artifacts.num_work_units,
            work_unit=None,
            worker_index=args.worker_index,
            num_workers=args.num_workers,
        )
    catalog_manifest, events_by_work_unit = _load_selected_events(
        args.catalog.resolve(), work_units, config_sha256=config.semantic_sha256
    )
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    effective_policy_path = policy.save_for_extraction(args.output_root.resolve())
    final_root = (
        args.final_root.resolve()
        if args.final_root
        else Path(config.artifacts.output_root).resolve() / "final"
    )
    def final_artifact_replaces_payload(work_unit: int) -> bool:
        return validate_finalized_work_unit_for_payload_skip(
            payload_root=args.output_root.resolve() / f"work_unit_{work_unit:04d}",
            final_root=final_root / f"work_unit_{work_unit:04d}",
            work_unit=work_unit,
            expected_event_keys=(
                event.event_key for event in events_by_work_unit[work_unit]
            ),
            config_sha256=config.semantic_sha256,
            production_policy_sha256=policy.semantic_sha256,
        )
    finalized_work_units = {
        work_unit for work_unit in work_units if final_artifact_replaces_payload(work_unit)
    }
    pending_work_units = tuple(
        work_unit for work_unit in work_units if work_unit not in finalized_work_units
    )
    if pending_work_units:
        _ensure_storage(
            args.output_root.resolve(),
            args.min_free_storage_bytes,
            wait_timeout_seconds=args.storage_wait_timeout_seconds,
            poll_seconds=args.storage_poll_seconds,
        )
    worker_root = (
        args.worker_state_root.resolve()
        if args.worker_state_root
        else args.output_root.resolve() / "workers"
    )
    worker_root.mkdir(parents=True, exist_ok=True)

    bridge = None
    token_contract = None
    if pending_work_units:
        from s3circuit.nvarc_circuit.tl_bridge import build_local_qwen3_replacement_model

        device = torch.device(args.device)
        bank = load_canonical_bank(
            config.stage_a.canonical_export_manifest,
            stage_a_config=config.stage_a_config,
        )
        if bank.identity_sha256 != policy.transcoder_identity_sha256:
            raise ValueError(
                "Canonical transcoder identity differs from production policy"
            )
        transcoders = build_codecircuit_transcoder_set(bank, device=device)
        bridge = build_local_qwen3_replacement_model(
            checkpoint=config.stage_a_config.base.hf_checkpoint,
            tokenizer_path=config.stage_a_config.base.tokenizer,
            transcoders=transcoders,
            device=device,
            manifest_path=(
                worker_root / f"worker_{args.worker_index:03d}_tl_conversion.json"
            ),
            numerics_backend=args.numerics_backend,
        )
        if bridge.manifest.get("numerics_backend") != args.numerics_backend:
            raise ValueError("Loaded bridge numerics backend changed")
        token_contract = TokenContract(
            digit_token_ids=tuple(range(10)),
            newline_token_id=10,
            pad_token_id=int(bridge.tokenizer.pad_token_id),
            eos_token_id=int(bridge.tokenizer.eos_token_id),
            vocab_size=int(bridge.model.cfg.d_vocab),
        )
        token_contract.validate()

    completed = []
    failures = []
    finalized = list(sorted(finalized_work_units))
    finalizer_failures = []
    active_finalizers: dict[int, tuple[subprocess.Popen, TextIO, float]] = {}
    materializer_log_root = (
        args.materializer_log_root.resolve()
        if args.materializer_log_root
        else worker_root / "materializer_logs"
    )
    if args.materialize_final_artifacts:
        materializer_log_root.mkdir(parents=True, exist_ok=True)

    def collect_finalizer(work_unit: int, *, wait: bool) -> bool:
        process, handle, launched = active_finalizers[work_unit]
        returncode = process.wait() if wait else process.poll()
        if returncode is None:
            return False
        handle.close()
        del active_finalizers[work_unit]
        if returncode == 0:
            finalized.append(work_unit)
        else:
            finalizer_failures.append(
                {
                    "work_unit": work_unit,
                    "exception": "MaterializerProcessError",
                    "message": f"Materializer exited with status {returncode}",
                    "elapsed_seconds": time.monotonic() - launched,
                }
            )
        return True

    def reap_finalizers(*, wait_for_slot: bool = False) -> None:
        for work_unit in list(active_finalizers):
            collect_finalizer(work_unit, wait=False)
        if wait_for_slot and len(active_finalizers) >= args.materializer_parallelism:
            collect_finalizer(next(iter(active_finalizers)), wait=True)

    def launch_finalizer(work_unit: int) -> None:
        reap_finalizers(wait_for_slot=True)
        log_path = materializer_log_root / f"work_unit_{work_unit:04d}.log"
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        command = [
            sys.executable,
            str(args.materializer.resolve()),
            "--config",
            str(args.config.resolve()),
            "--policy",
            str(effective_policy_path),
            "--payload_work_unit",
            str(args.output_root.resolve() / f"work_unit_{work_unit:04d}"),
            "--output",
            str(final_root / f"work_unit_{work_unit:04d}"),
            "--delete_source_payload",
        ]
        environment = dict(os.environ)
        threads = str(args.materializer_cpu_threads)
        environment.update(
            {
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        handle.write(
            f"\n=== worker={args.worker_index} work_unit={work_unit} "
            f"started={time.time()} ===\n"
        )
        process = subprocess.Popen(
            command,
            cwd=str(HC_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        active_finalizers[work_unit] = (process, handle, time.monotonic())

    started = time.monotonic()
    for work_unit in work_units:
        try:
            with _work_unit_lock(args.work_unit_lock_root, work_unit):
                # A separate CPU consumer may have finalized and deleted this
                # payload after the worker's startup inventory.
                finalized_now = final_artifact_replaces_payload(work_unit)
                if finalized_now:
                    finalized.append(work_unit)
                else:
                    if bridge is None or token_contract is None:
                        raise AssertionError("Pending work unit has no loaded GPU bridge")
                    _extract_work_unit(
                        args=args,
                        config=config,
                        policy=policy,
                        catalog_manifest=catalog_manifest,
                        events=events_by_work_unit[work_unit],
                        work_unit=work_unit,
                        bridge=bridge,
                        token_contract=token_contract,
                        event_lanes=effective_event_lanes,
                        dual_lane_replay_threshold=dual_lane_replay_threshold,
                    )
                    if args.materialize_final_artifacts:
                        launch_finalizer(work_unit)
            completed.append(work_unit)
            status = (
                "final_artifact_complete"
                if work_unit in set(finalized)
                else "complete"
            )
            message = None
        except StorageBackpressureError:
            raise
        except Exception as error:  # noqa: BLE001 - continue independent work units
            failures.append(
                {
                    "work_unit": work_unit,
                    "exception": type(error).__name__,
                    "message": str(error)[:2000],
                    "traceback": traceback.format_exc(limit=8),
                }
            )
            status = "failed"
            message = str(error)[:500]
        progress = {
            "phase": "persistent_payload_worker",
            "worker_index": args.worker_index,
            "num_workers": args.num_workers,
            "work_unit": work_unit,
            "work_unit_status": status,
            "message": message,
            "work_units_complete": len(completed),
            "work_units_failed": len(failures),
            "work_units_total": len(work_units),
            "elapsed_seconds": time.monotonic() - started,
            "numerics_backend": args.numerics_backend,
            "materialization_enabled": args.materialize_final_artifacts,
            "work_units_finalized": len(set(finalized)),
            "materializers_active": len(active_finalizers),
            "materializer_failures": len(finalizer_failures),
            "repair_schedule_sha256": (
                repair_schedule.get("schedule_sha256") if repair_schedule else None
            ),
            "production_schedule_sha256": (
                production_schedule.get("schedule_sha256")
                if production_schedule else None
            ),
            "event_lanes": effective_event_lanes,
        }
        write_json_atomic(
            worker_root / f"worker_{args.worker_index:03d}_progress.json", progress
        )
        print(json.dumps(progress, sort_keys=True), flush=True)

    while active_finalizers:
        collect_finalizer(next(iter(active_finalizers)), wait=True)
    failures.extend(finalizer_failures)

    result = {
        "format": "nvarc_rearc_persistent_payload_worker_v1",
        "status": "complete" if not failures else "incomplete",
        "config_sha256": config.semantic_sha256,
        "production_policy_sha256": policy.semantic_sha256,
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "work_units": list(work_units),
        "completed_work_units": completed,
        "resumed_from_final_work_units": sorted(finalized_work_units),
        "finalized_work_units": sorted(set(finalized)),
        "materialization_enabled": args.materialize_final_artifacts,
        "numerics_backend": args.numerics_backend,
        "materializer_parallelism": args.materializer_parallelism,
        "materializer_cpu_threads": args.materializer_cpu_threads,
        "replay_validation_mode": args.replay_validation_mode,
        "repair_schedule_sha256": (
            repair_schedule.get("schedule_sha256") if repair_schedule else None
        ),
        "production_schedule_sha256": (
            production_schedule.get("schedule_sha256")
            if production_schedule else None
        ),
        "event_lanes": effective_event_lanes,
        "failures": failures,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(worker_root / f"worker_{args.worker_index:03d}.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if failures:
        raise RuntimeError(
            f"Persistent payload worker {args.worker_index} failed "
            f"{len(failures)} work units"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate frozen self-output trace shards for the S5 circuit pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


S5_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = S5_ROOT.parent
sys.path.insert(0, str(S5_ROOT / "src"))

import torch  # noqa: E402

from s5_lean_circuit.circuit.model_io import (  # noqa: E402
    load_frozen_source_model,
    load_tokenizer,
    sha256_file,
)
from s5_lean_circuit.circuit.settings import load_circuit_config  # noqa: E402
from s5_lean_circuit.circuit.traces import (  # noqa: E402
    analysis_join_record,
    balanced_trace_stream_batched,
    deterministic_logic_sample,
    greedy_full_output_traces_batched,
    read_trace_chunk,
    trace_generation_code_sha256,
    trace_generation_contract,
    write_trace_chunk,
)
from s5_lean_circuit.config import load_config  # noqa: E402
from s5_lean_circuit.protocol import (  # noqa: E402
    ALL_TASKS,
    ATOMIC_TASKS,
    expected_first_tactic,
    extract_first_tactic,
)


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "validation", "audit64", "audit256"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", choices=ALL_TASKS, default=None)
    args = parser.parse_args()

    config = load_circuit_config(repo_path(args.config))
    source = config["source_model"]
    source_config = load_config(repo_path(source["architecture_config"]))
    device = torch.device(args.device)
    model, model_identity = load_frozen_source_model(
        source_config,
        repo_path(source["checkpoint"]),
        device=device,
        expected_checkpoint_config_sha256=source["checkpoint_config_sha256"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_architecture_config_sha256=source["architecture_config_sha256"],
    )
    tokenizer_path = repo_path(source["tokenizer"])
    tokenizer = load_tokenizer(
        tokenizer_path, expected_sha256=source["tokenizer_sha256"]
    )
    tokenizer_sha256 = sha256_file(tokenizer_path)
    generation_code_sha256 = trace_generation_code_sha256()
    output = repo_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows_per_chunk = int(config["traces"]["rows_per_chunk"])
    generation_contract = trace_generation_contract(config["traces"])
    manifest_path = output / "manifest.json"
    existing_paths = sorted(output.glob("trace_chunk_*.safetensors"))
    if existing_paths and not manifest_path.is_file():
        raise ValueError("partial trace chunks exist without their identity manifest")
    existing_traces = []
    chunks = []
    for index, path in enumerate(existing_paths):
        if path.name != f"trace_chunk_{index:05d}.safetensors":
            raise ValueError("existing trace chunk sequence is not contiguous")
        loaded = read_trace_chunk(path)
        existing_traces.extend(loaded)
        sidecar = path.with_suffix(path.suffix + ".json")
        chunks.append(
            {
                "path": str(path.resolve()),
                "sidecar": str(sidecar.resolve()),
                "records": len(loaded),
                "eligible_tokens": sum(trace.eligible_tokens for trace in loaded),
                "bytes": path.stat().st_size + sidecar.stat().st_size,
                "tensor_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            }
        )
    existing_eligible = sum(trace.eligible_tokens for trace in existing_traces)
    existing_records = len(existing_traces)
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous.get("format") != "s5_circuit_trace_manifest_v1"
            or previous.get("split") != args.split
            or previous.get("task_filter") != args.task
            or previous.get("config_sha256") != config["_config_sha256"]
            or previous.get("generation_contract") != generation_contract
            or previous.get("tokenizer_sha256") != tokenizer_sha256
            or previous.get("generation_code_sha256") != generation_code_sha256
            or previous.get("model_identity", {}).get("checkpoint_sha256")
            != model_identity["checkpoint_sha256"]
            or previous.get("model_identity", {}).get("checkpoint_config_sha256")
            != model_identity["checkpoint_config_sha256"]
            or previous.get("model_identity", {}).get(
                "current_architecture_config_sha256"
            )
            != model_identity["current_architecture_config_sha256"]
            or previous.get("model_identity", {}).get("dtype")
            != model_identity["dtype"]
        ):
            raise ValueError("existing trace manifest identity changed")
        previous_records = int(previous.get("records", -1))
        interval = int(generation_contract["canonical_regeneration_interval"])
        expected_regenerated = (
            previous_records
            if args.split in ("audit64", "audit256")
            else (
                0
                if previous_records == 0
                else (previous_records - 1) // interval + 1
            )
        )
        verification = previous.get("verification", {})
        if (
            int(verification.get("teacher_forced_verified_records", -1))
            != previous_records
            or int(verification.get("canonical_regenerated_records", -1))
            != expected_regenerated
        ):
            raise ValueError("existing trace verification coverage changed")
        if previous.get("status") == "complete":
            if previous_records != existing_records or previous.get("chunks") != chunks:
                raise ValueError("complete trace manifest no longer matches its chunks")
            print(json.dumps(previous, sort_keys=True))
            return
        if previous.get("status") != "running":
            raise ValueError("partial trace manifest has an unsupported status")
        previous_chunks = previous.get("chunks", [])
        if (
            previous_records
            != sum(int(item["records"]) for item in previous_chunks)
            or previous_chunks != chunks[: len(previous_chunks)]
        ):
            raise ValueError("partial trace manifest is not a verified chunk prefix")

    teacher_verified_records = existing_records
    regeneration_interval = int(
        generation_contract["canonical_regeneration_interval"]
    )
    if args.split in ("audit64", "audit256"):
        canonical_regenerated_records = existing_records
    else:
        canonical_regenerated_records = (
            0
            if existing_records == 0
            else (existing_records - 1) // regeneration_interval + 1
        )
    record_ordinal = existing_records

    def write_manifest(status: str) -> None:
        manifest = {
            "format": "s5_circuit_trace_manifest_v1",
            "split": args.split,
            "task_filter": args.task,
            "config_sha256": config["_config_sha256"],
            "model_identity": model_identity,
            "tokenizer_sha256": tokenizer_sha256,
            "generation_code_sha256": generation_code_sha256,
            "generation_contract": generation_contract,
            "chunks": chunks,
            "records": sum(int(item["records"]) for item in chunks),
            "eligible_tokens": sum(int(item["eligible_tokens"]) for item in chunks),
            "verification": {
                "teacher_forced_verified_records": teacher_verified_records,
                "canonical_regenerated_records": canonical_regenerated_records,
            },
            "status": status,
        }
        write_json(manifest_path, manifest)

    # Publish identity before the first chunk. Rewriting here also adopts a
    # fully reopened chunk that survived a crash immediately before manifest update.
    write_manifest("running")

    if args.split in ("train", "validation"):
        if args.task is None:
            raise ValueError("resumable train/validation trace jobs require --task")
        seed_base = int(config["traces"][f"{args.split}_seed_base"])
        tasks = (args.task,)
        seed_base += ALL_TASKS.index(args.task) * 100_000_000
        tokens_per_task = int(config["traces"][f"{args.split}_tokens_per_task"])
        if existing_eligible > tokens_per_task:
            raise ValueError("partial trace root exceeds the task token quota")
        expected_existing = [
            (args.task, seed_base + offset) for offset in range(existing_records)
        ]
        actual_existing = [
            (trace.task_type, trace.seed) for trace in existing_traces
        ]
        if actual_existing != expected_existing:
            raise ValueError("partial trace root is not the deterministic task prefix")
        stream = balanced_trace_stream_batched(
            model,
            tokenizer,
            tasks=tasks,
            seed_base=seed_base,
            tokens_per_task=tokens_per_task,
            device=device,
            generation_batch_size=int(generation_contract["batch_size"]),
            generation_pool_size=int(generation_contract["pool_size"]),
            teacher_forced_verify_all=bool(
                generation_contract["teacher_forced_verify_all"]
            ),
            canonical_regeneration_interval=regeneration_interval,
            n_distractors=int(config["traces"]["n_distractors"]),
            context_length=int(config["generation"]["context_length"]),
            max_new_tokens=int(config["generation"]["max_new_tokens"]),
            stop_markers=tuple(config["generation"]["stop_markers"]),
            start_offsets={args.task: existing_records},
            initial_totals={args.task: existing_eligible},
        )
    else:
        if args.task is not None:
            raise ValueError("--task is valid only for train/validation trace splits")
        examples = 64 if args.split == "audit64" else 256
        if existing_records > examples:
            raise ValueError("partial audit trace root exceeds the expected examples")
        per_task = examples // len(ATOMIC_TASKS)
        expected_prefix = [
            (task, 777_777_000 + task_index * 100_000_000 + offset)
            for offset in range(per_task)
            for task_index, task in enumerate(ATOMIC_TASKS)
        ][:existing_records]
        if [(trace.task_type, trace.seed) for trace in existing_traces] != expected_prefix:
            raise ValueError("partial audit trace root is not the deterministic prefix")

        requests = tuple(
            (
                task,
                777_777_000 + task_index * 100_000_000 + offset,
            )
            for offset in range(per_task)
            for task_index, task in enumerate(ATOMIC_TASKS)
        )[existing_records:]
        stream = iter(
            greedy_full_output_traces_batched(
                model,
                tokenizer,
                requests=requests,
                device=device,
                batch_size=int(generation_contract["batch_size"]),
                n_distractors=int(config["traces"]["n_distractors"]),
                context_length=int(config["generation"]["context_length"]),
                max_new_tokens=int(config["generation"]["max_new_tokens"]),
                stop_markers=tuple(config["generation"]["stop_markers"]),
                teacher_forced_verify_all=True,
                canonical_regeneration_mask=(True,) * len(requests),
            )
        )

    pending = []
    join_path = repo_path(args.analysis_output)
    try:
        join_path.relative_to(output)
    except ValueError:
        pass
    else:
        raise ValueError("analysis join must be rooted outside the trace shard directory")
    try:
        output.relative_to(join_path.parent)
    except ValueError:
        pass
    else:
        raise ValueError("trace shard directory must be outside the analysis root")
    join_path.parent.mkdir(parents=True, exist_ok=True)
    with join_path.open("w", encoding="utf-8") as join:
        for trace in existing_traces:
            _input_text, gold_output = deterministic_logic_sample(
                trace.task_type,
                trace.seed,
                n_distractors=int(config["traces"]["n_distractors"]),
            )
            predicted = extract_first_tactic(tokenizer.decode(trace.generated_token_ids))
            expected = expected_first_tactic(gold_output)
            join.write(
                json.dumps(
                    analysis_join_record(
                        trace, gold_tactic=expected, correct=predicted == expected
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
        for trace in stream:
            teacher_verified_records += 1
            if args.split in ("audit64", "audit256") or (
                record_ordinal % regeneration_interval == 0
            ):
                canonical_regenerated_records += 1
            record_ordinal += 1
            pending.append(trace)
            _input_text, gold_output = deterministic_logic_sample(
                trace.task_type,
                trace.seed,
                n_distractors=int(config["traces"]["n_distractors"]),
            )
            predicted = extract_first_tactic(tokenizer.decode(trace.generated_token_ids))
            expected = expected_first_tactic(gold_output)
            join.write(
                json.dumps(
                    analysis_join_record(
                        trace, gold_tactic=expected, correct=predicted == expected
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            if len(pending) >= rows_per_chunk:
                chunks.append(
                    write_trace_chunk(
                        output / f"trace_chunk_{len(chunks):05d}.safetensors",
                        pending,
                        config_sha256=config["_config_sha256"],
                    )
                )
                pending = []
                write_manifest("running")
                print(
                    json.dumps(
                        {
                            "trace_progress": {
                                "split": args.split,
                                "task": args.task,
                                "records": sum(int(item["records"]) for item in chunks),
                                "eligible_tokens": sum(
                                    int(item["eligible_tokens"]) for item in chunks
                                ),
                                "teacher_verified_records": teacher_verified_records,
                                "canonical_regenerated_records": canonical_regenerated_records,
                            }
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if pending:
            chunks.append(
                write_trace_chunk(
                    output / f"trace_chunk_{len(chunks):05d}.safetensors",
                    pending,
                    config_sha256=config["_config_sha256"],
                )
            )
            write_manifest("running")
    write_manifest("complete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

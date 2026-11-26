#!/usr/bin/env python3
"""Evaluate every scheduled S5 composition-LOCO checkpoint on atomic + visible tasks.

This is a diagnostic/backfill evaluator.  It never edits the training
selection.json and never reads held-out OOD or final-test rows.  The selection
record in its own output is a counterfactual view using four atomic tasks plus
the four seen composition tasks, with earlier positive scheduled steps winning
ties.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist


REPO_ROOT = Path(__file__).resolve().parents[2]
for source in (
    REPO_ROOT / "s5_lean_circuit" / "src",
    REPO_ROOT / "s2_hypernet" / "src",
    REPO_ROOT / "s3_circuit" / "src",
):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from s5_lean_circuit.circuit.model_io import load_tokenizer  # noqa: E402
from s5_lean_circuit.composition_loco import (  # noqa: E402
    ATOMIC_TASKS,
    COMPOSITION_TASKS,
    PROTOCOL_ID,
    STAGE_A_BASE_IDENTITY,
    VALIDATION_STEPS,
    load_eval_bank,
    sha256_file,
)
from s5_lean_circuit.composition_loco_config import (  # noqa: E402
    CIRCUIT_RETENTION20_PROTOCOL_ID,
    QUERY_RETENTION20_PROTOCOL_ID,
    RAW_RETENTION20_PROTOCOL_ID,
    load_config,
)
from s5_lean_circuit.composition_loco_eval import (  # noqa: E402
    DEFAULT_EVAL_BATCH_SIZE,
    evaluate_bank,
)
from s5_lean_circuit.composition_loco_vector422 import load_vector422_eval_bank  # noqa: E402
from s5_lean_circuit.composition_loco_vector422_spec import VECTOR422_PROTOCOL_ID  # noqa: E402
from s5_lean_circuit.composition_loco_train import (  # noqa: E402
    _architecture,
    _load_hypernet,
    _rank_world,
)
from s5_lean_circuit.modeling import build_model  # noqa: E402


FORMAT = "s5_composition_loco_checkpoint_validation_v1"
RUN_FORMAT = "s5_composition_loco_checkpoint_validation_run_v1"
OUTPUT_NAME = "checkpoint_validation_all_v1"
CHECKPOINT_FORMAT = "s5_composition_loco_checkpoint_v1"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != payload:
            raise ValueError(f"refusing to overwrite a different result: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _parse_steps(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return tuple(int(step) for step in VALIDATION_STEPS)
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(step < 0 for step in values):
        raise ValueError("checkpoint steps must be unique non-negative integers")
    return values


def _verify_bank(
    bank: Any,
    *,
    name: str,
    tasks: Sequence[str],
    seed_base: int,
    samples_per_task: int,
) -> None:
    metadata = bank.metadata
    expected_counts = Counter({str(task): int(samples_per_task) for task in tasks})
    counts = Counter(str(record.task_type) for record in bank.records)
    if counts != expected_counts or len(bank.records) != sum(expected_counts.values()):
        raise ValueError(f"{name} coverage changed: expected {dict(expected_counts)}, got {dict(counts)}")
    if (
        metadata.get("corpus_name") != name
        or metadata.get("tasks") != list(tasks)
        or int(metadata.get("seed_base", -1)) != int(seed_base)
        or int(metadata.get("samples_per_task", -1)) != int(samples_per_task)
        or int(metadata.get("event_count", -1)) != sum(expected_counts.values())
        or metadata.get("expected_source_base_identity") != STAGE_A_BASE_IDENTITY
    ):
        raise ValueError(f"{name} bank identity changed")


def _checkpoint(path: Path, config: Any, step: int) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    expected = {
        "format": CHECKPOINT_FORMAT,
        "protocol_id": config.protocol_id,
        "config_sha256": config.computed_sha256,
        "route": config.route,
        "fold_id": config.fold_id,
        "kind": "model_only",
    }
    mismatched = [key for key, value in expected.items() if payload.get(key) != value]
    if int(payload.get("global_step", -1)) != int(step):
        mismatched.append("global_step")
    state = payload.get("model_state_dict")
    if not isinstance(state, Mapping):
        mismatched.append("model_state_dict")
    if mismatched:
        raise ValueError(f"checkpoint identity changed at step {step}: {', '.join(mismatched)}")
    reference = {
        "path": str(path.relative_to(REPO_ROOT).as_posix()),
        "filename": path.name,
        "global_step": int(step),
        "payload_bytes": path.stat().st_size,
        "payload_sha256": sha256_file(path),
        "config_sha256": config.computed_sha256,
    }
    return reference, state


def _load_step_state(model: Any, config: Any, path: Path, step: int) -> dict[str, Any]:
    reference, state = _checkpoint(path, config, step)
    if config.route == "direct_sft":
        model.load_state_dict(state, strict=True)
    else:
        model.load_trainable_state_dict(state)
    del state
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return reference


def _load_model_for_eval(config: Any, device: torch.device) -> Any:
    if config.route == "direct_sft":
        # Direct SFT's historical init checkpoint is not part of the current
        # artifact family. Positive-step model-only checkpoints are complete
        # state dicts, so construct the architecture directly and load them.
        model = build_model(_architecture(config)).to(device)
        model.config.grad_checkpointing = False
        return model
    return _load_hypernet(config, device)


def _counts(result: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    raw = result.get("task_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("evaluation result has no task counts")
    return {
        str(task): {"correct": int(row["correct"]), "total": int(row["total"])}
        for task, row in raw.items()
    }


def _summary(result: Mapping[str, Any], tasks: Sequence[str]) -> dict[str, Any]:
    counts = _counts(result)
    if set(counts) != set(tasks):
        raise ValueError(f"evaluation task coverage changed: expected {list(tasks)}, got {sorted(counts)}")
    if any(row["total"] != 500 or not 0 <= row["correct"] <= row["total"] for row in counts.values()):
        raise ValueError("evaluation task counts are outside the frozen 500-row contract")
    correct = sum(row["correct"] for row in counts.values())
    total = sum(row["total"] for row in counts.values())
    return {
        "task_counts": counts,
        "overall_correct": correct,
        "overall_total": total,
        "overall_accuracy": 100.0 * correct / total if total else 0.0,
        "macro_accuracy": sum(100.0 * row["correct"] / row["total"] for row in counts.values()) / len(counts),
    }


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _evaluate(
    model: Any,
    tokenizer: Any,
    visible_bank: Any,
    atomic_bank: Any,
    config: Any,
    *,
    device: torch.device,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    gate = 1.0 if config.route == "direct_sft" else float(config.model.get("gate_max", 1.0))
    # The frozen training configs intentionally do not encode a diagnostic
    # batch-size choice. Allow the server-side all-checkpoint runner to use a
    # larger batch without changing the config hash bound into every model
    # checkpoint. The default remains the historical evaluator default.
    eval_batch_size = int(
        os.environ.get(
            "S5_LOCO_CHECKPOINT_EVAL_BATCH_SIZE",
            config.evaluation.get("eval_batch_size", DEFAULT_EVAL_BATCH_SIZE),
        )
    )
    if eval_batch_size <= 1:
        raise ValueError("checkpoint validation must use a batched eval_batch_size > 1")
    visible = evaluate_bank(
        model,
        tokenizer,
        visible_bank,
        device=device,
        task_types=config.train_tasks,
        route=config.route,
        condition="matched",
        gate_multiplier=gate,
        batch_size=eval_batch_size,
    )
    atomic = evaluate_bank(
        model,
        tokenizer,
        atomic_bank,
        device=device,
        task_types=ATOMIC_TASKS,
        route=config.route,
        condition="matched",
        gate_multiplier=gate,
        batch_size=eval_batch_size,
    )
    if dist.is_available() and dist.is_initialized():
        _barrier()
    return visible, atomic


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if config.protocol_id not in {
        PROTOCOL_ID,
        CIRCUIT_RETENTION20_PROTOCOL_ID,
        VECTOR422_PROTOCOL_ID,
        QUERY_RETENTION20_PROTOCOL_ID,
        RAW_RETENTION20_PROTOCOL_ID,
    }:
        raise ValueError(
            "this evaluator is restricted to the frozen composition protocols, "
            f"got {config.protocol_id}"
        )
    steps = (
        tuple(int(step) for step in config.training["validation_steps"])
        if config.protocol_id == VECTOR422_PROTOCOL_ID and args.checkpoint_steps is None
        else _parse_steps(args.checkpoint_steps)
    )
    rank, world, _local_rank, device = _rank_world()

    try:
        # Keep paths lexical under the snapshot: outputs/datasets are shared-volume
        # symlinks in Siflow pods, and resolving them would break portable
        # provenance paths even though the target files are valid.
        visible_path = REPO_ROOT / str(config.data["validation_bank"])
        atomic_path = REPO_ROOT / str(config.data["atomic_bank"])
        tokenizer_path = REPO_ROOT / str(config.data["tokenizer_path"])
        visible_bank = (
            load_vector422_eval_bank(config, "validation")
            if config.protocol_id == VECTOR422_PROTOCOL_ID else load_eval_bank(visible_path)
        )
        atomic_bank = (
            load_vector422_eval_bank(config, "atomic_bank")
            if config.protocol_id == VECTOR422_PROTOCOL_ID else load_eval_bank(atomic_path)
        )
        _verify_bank(
            visible_bank,
            name="visible_validation",
            tasks=COMPOSITION_TASKS,
            seed_base=2_000_000_000,
            samples_per_task=500,
        )
        _verify_bank(
            atomic_bank,
            name="atomic_retention",
            tasks=ATOMIC_TASKS,
            seed_base=4_000_000_000,
            samples_per_task=500,
        )
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"tokenizer is missing: {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
        tokenizer_sha = sha256_file(tokenizer_path)
        if visible_bank.metadata.get("tokenizer_sha256") != tokenizer_sha or atomic_bank.metadata.get("tokenizer_sha256") != tokenizer_sha:
            raise ValueError("validation-bank tokenizer identity differs from the run tokenizer")

        output_root = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else (REPO_ROOT / config.output_dir / OUTPUT_NAME)
        )
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_root = REPO_ROOT / config.output_dir / "checkpoints"
        model = _load_model_for_eval(config, device)
        raw_cache_repair = getattr(model, "_s5_raw_cache_repair", None)
        model.eval()

        records: list[dict[str, Any]] = []
        for step in steps:
            checkpoint_ref: dict[str, Any] | None = None
            if step > 0:
                checkpoint_ref = _load_step_state(
                    model,
                    config,
                    checkpoint_root / f"model_step{step:06d}.pt",
                    step,
                )
            visible_result, atomic_result = _evaluate(
                model,
                tokenizer,
                visible_bank,
                atomic_bank,
                config,
                device=device,
            )
            if rank == 0:
                if visible_result is None or atomic_result is None:
                    raise RuntimeError("rank zero did not receive evaluation results")
                visible_summary = _summary(visible_result, config.train_tasks)
                atomic_summary = _summary(atomic_result, ATOMIC_TASKS)
                combined_correct = visible_summary["overall_correct"] + atomic_summary["overall_correct"]
                combined_total = visible_summary["overall_total"] + atomic_summary["overall_total"]
                row = {
                    "format": FORMAT,
                    "protocol_id": config.protocol_id,
                    "config_sha256": config.computed_sha256,
                    "route": config.route,
                    "fold_id": config.fold_id,
                    "heldout_task": config.heldout_task,
                    "step": int(step),
                    "checkpoint": checkpoint_ref,
                    "visible_validation": {
                        "bank_identity": dict(visible_bank.metadata),
                        **visible_summary,
                    },
                    "atomic_retention": {
                        "bank_identity": dict(atomic_bank.metadata),
                        **atomic_summary,
                    },
                    "combined_atomic_plus_visible": {
                        "tasks": list(ATOMIC_TASKS) + list(config.train_tasks),
                        "overall_correct": combined_correct,
                        "overall_total": combined_total,
                        "overall_accuracy": 100.0 * combined_correct / combined_total,
                        "selection_eligible": int(step) > 0,
                    },
                }
                _atomic_json(output_root / f"step{step:06d}.json", row)
                records.append(row)
            _barrier()

        if rank == 0:
            candidates = [
                row for row in records
                if row["combined_atomic_plus_visible"]["selection_eligible"]
            ]
            if not candidates:
                raise ValueError("no positive checkpoint is eligible for counterfactual selection")
            selected = max(
                candidates,
                key=lambda row: (
                    int(row["combined_atomic_plus_visible"]["overall_correct"]),
                    -int(row["step"]),
                ),
            )
            run_payload = {
                "format": RUN_FORMAT,
                "status": "complete",
                "final_test_run": False,
                "ood_run": False,
                "selection_view": "four_atomic_plus_four_seen_composition_tasks",
                "selection_rule": "max_combined_correct_count_ties_earlier_positive_scheduled_step",
                "deployed_commit": os.environ.get("S5_DEPLOY_COMMIT"),
                "config_path": str(config_path.relative_to(REPO_ROOT).as_posix()),
                "config_sha256": config.computed_sha256,
                "route": config.route,
                "fold_id": config.fold_id,
                "heldout_task": config.heldout_task,
                "validation_steps": list(steps),
                "visible_bank": {
                    "path": str(visible_path.relative_to(REPO_ROOT).as_posix()),
                    "sha256": sha256_file(visible_path),
                    "identity_sha256": visible_bank.metadata.get("identity_sha256"),
                },
                "atomic_bank": {
                    "path": str(atomic_path.relative_to(REPO_ROOT).as_posix()),
                    "sha256": sha256_file(atomic_path),
                    "identity_sha256": atomic_bank.metadata.get("identity_sha256"),
                },
                "raw_cache_repair": raw_cache_repair,
                "selected_step": int(selected["step"]),
                "selected_combined_correct": int(selected["combined_atomic_plus_visible"]["overall_correct"]),
                "selected_combined_total": int(selected["combined_atomic_plus_visible"]["overall_total"]),
                "selected_combined_accuracy": float(selected["combined_atomic_plus_visible"]["overall_accuracy"]),
                "evaluations": [
                    {
                        "step": int(row["step"]),
                        "checkpoint": row["checkpoint"],
                        "visible_correct": int(row["visible_validation"]["overall_correct"]),
                        "atomic_correct": int(row["atomic_retention"]["overall_correct"]),
                        "combined_correct": int(row["combined_atomic_plus_visible"]["overall_correct"]),
                        "combined_total": int(row["combined_atomic_plus_visible"]["overall_total"]),
                    }
                    for row in records
                ],
                "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            }
            _atomic_json(output_root / "run_result.json", run_payload)
            print(json.dumps({
                "route": config.route,
                "fold_id": config.fold_id,
                "selected_step": run_payload["selected_step"],
                "selected_combined_accuracy": run_payload["selected_combined_accuracy"],
                "output": str(output_root),
            }, sort_keys=True), flush=True)
        _barrier()
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-steps", help="comma-separated scheduled steps; defaults to VALIDATION_STEPS")
    parser.add_argument("--output-dir", help="override the per-run checkpoint-validation output directory")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())

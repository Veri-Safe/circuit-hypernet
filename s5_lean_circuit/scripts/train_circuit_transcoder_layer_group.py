#!/usr/bin/env python3
"""Train one S5 layer's audited or selected candidates with exact sharded replay."""

from __future__ import annotations

import argparse
from datetime import timedelta
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import sys
S5_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = S5_ROOT.parent
sys.path.insert(0, str(S5_ROOT / "src"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from s5_lean_circuit.circuit.distributed_transcoder_training import (  # noqa: E402
    train_transcoder_layer_group_distributed,
)
from s5_lean_circuit.circuit.model_io import (  # noqa: E402
    load_frozen_source_model,
    sha256_file,
)
from s5_lean_circuit.circuit.settings import (  # noqa: E402
    load_circuit_config,
    trace_source_config_sha256,
)
from s5_lean_circuit.circuit.traces import (  # noqa: E402
    iter_trace_chunks_round_robin,
    trace_generation_contract,
    validate_trace_inventory,
)
from s5_lean_circuit.config import load_config  # noqa: E402


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _lr_label(value: float) -> str:
    return f"{float(value):.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _candidate_dirs(
    root: Path, *, layer: int, width_multiplier: int, seed: int, labels: list[str]
) -> dict[str, Path]:
    return {
        label: root
        / f"layer_{layer}"
        / f"w{width_multiplier}"
        / f"seed{seed}"
        / f"lr{label}"
        for label in labels
    }


def _checkpoint_update(path: Path) -> int:
    value = torch.load(path, map_location="cpu", weights_only=False)
    return int(value.get("update", -1))


def _common_resume_paths(
    candidates: dict[str, Path],
) -> dict[str, Path | None]:
    latest = {label: root / "latest.pt" for label, root in candidates.items()}
    if all(path.is_file() for path in latest.values()):
        updates = {_checkpoint_update(path) for path in latest.values()}
        if len(updates) != 1:
            raise ValueError("shared-replay latest checkpoints have mixed cursors")
        return latest
    for percent in range(100, 0, -10):
        paths = {
            label: root / "candidates" / f"p{percent:03d}.pt"
            for label, root in candidates.items()
        }
        if all(path.is_file() for path in paths.values()):
            updates = {_checkpoint_update(path) for path in paths.values()}
            if len(updates) != 1:
                raise ValueError("shared-replay progress checkpoints have mixed cursors")
            return paths
    return {label: None for label in candidates}


def _truncate_metrics_to_resume(
    candidates: dict[str, Path], resume_paths: dict[str, Path | None]
) -> None:
    for label, root in candidates.items():
        resume = resume_paths[label]
        resume_update = 0 if resume is None else _checkpoint_update(resume)
        metrics = root / "train_metrics.jsonl"
        if not metrics.is_file():
            continue
        rows = []
        for line in metrics.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if int(value["update"]) <= resume_update:
                rows.append(json.dumps(value, sort_keys=True))
        temporary = metrics.with_name(f".{metrics.name}.truncate.{os.getpid()}")
        temporary.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
        os.replace(temporary, metrics)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trace-glob", required=True)
    parser.add_argument("--expected-generation-code-sha256", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--width-multiplier", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--checkpoint-percent-override", type=int, default=None)
    parser.add_argument(
        "--run-kind",
        choices=("lr-audit", "full-bank"),
        default="lr-audit",
    )
    parser.add_argument("--selection-contract", default=None)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_generation_code_sha256):
        raise ValueError("expected Trace generation identity is not a SHA256")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl", timeout=timedelta(hours=2), device_id=device
    )
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    config = load_circuit_config(repo_path(args.config))
    if int(args.width_multiplier) not in {
        int(value) for value in config["transcoder"]["width_multipliers"]
    }:
        raise ValueError("requested width is outside the frozen Transcoder config")
    if int(args.seed) != int(config["transcoder"]["production_seed"]):
        raise ValueError("exact-sharded production replay is locked to seed 42")

    optimization = config["optimization"]
    configured_learning_rates = [float(value) for value in optimization["learning_rates"]]
    manifest_metadata: dict[str, object] | None = None
    if args.run_kind == "lr-audit":
        if int(args.layer) not in {
            int(value) for value in config["transcoder"]["audit_layers"]
        }:
            raise ValueError("LR audit layer is outside the frozen audit layers")
        if args.selection_contract is not None:
            raise ValueError("LR audit must not consume a post-selection contract")
        learning_rates = [
            (_lr_label(value), float(value)) for value in configured_learning_rates
        ]
    else:
        if args.selection_contract is None:
            raise ValueError("full-bank replay requires --selection-contract")
        if args.max_tokens is not None:
            raise ValueError("initial full-bank replay must consume the frozen 90M tokens")
        if args.checkpoint_percent_override is not None:
            raise ValueError("initial full-bank replay must preserve the 10-percent grid")
        selection_path = repo_path(args.selection_contract).resolve()
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("format") != "s5_width8_relative_selection_review_v1":
            raise ValueError("unsupported width8 selection contract")
        if selection.get("status") != "READY_FOR_INITIAL_FULL_BANK":
            raise ValueError("width8 selection contract is not ready for initial bank")
        if selection.get("circuit_config_sha256") != config["_config_sha256"]:
            raise ValueError("width8 selection contract config identity mismatch")
        if int(selection.get("width_multiplier", -1)) != int(args.width_multiplier):
            raise ValueError("width8 selection contract width mismatch")
        if int(selection.get("production_seed", -1)) != int(args.seed):
            raise ValueError("width8 selection contract seed mismatch")
        missing_layers = {int(value) for value in selection["full_bank_missing_layers"]}
        if int(args.layer) not in missing_layers:
            raise ValueError("full-bank replay only trains the five missing layers")
        if not 0 <= int(args.layer) < int(config["transcoder"]["layers"]):
            raise ValueError("full-bank layer is outside the source model")
        selected_learning_rate = float(selection["full_bank_learning_rate"])
        if selected_learning_rate not in configured_learning_rates:
            raise ValueError("selected full-bank learning rate is outside the audit sweep")
        if int(selection["initial_training_tokens"]) != (
            int(config["traces"]["train_tokens_per_task"])
            * len(config["traces"]["tasks"])
        ):
            raise ValueError("selection contract initial token budget mismatch")
        learning_rates = [
            (_lr_label(selected_learning_rate), selected_learning_rate)
        ]
        manifest_metadata = {
            "run_kind": "full-bank",
            "selection_contract": selection_path.relative_to(
                REPO_ROOT.resolve()
            ).as_posix(),
            "selection_contract_sha256": sha256_file(selection_path),
            "selection_validation_archive_sha256": selection[
                "validation_archive_sha256"
            ],
        }

    source = config["source_model"]
    source_config = load_config(repo_path(source["architecture_config"]))
    model, model_identity = load_frozen_source_model(
        source_config,
        repo_path(source["checkpoint"]),
        device=device,
        expected_checkpoint_config_sha256=source["checkpoint_config_sha256"],
        expected_checkpoint_sha256=source["checkpoint_sha256"],
        expected_architecture_config_sha256=source["architecture_config_sha256"],
    )
    trace_paths = sorted(glob.glob(str(repo_path(args.trace_glob))))
    if not trace_paths:
        raise FileNotFoundError(f"Trace glob matched no files: {args.trace_glob}")

    if rank == 0:
        tokenizer_sha256 = sha256_file(repo_path(source["tokenizer"]))
        if tokenizer_sha256 != source["tokenizer_sha256"]:
            raise ValueError("circuit tokenizer file SHA256 mismatch")
        validate_trace_inventory(
            trace_paths,
            expected_split="train",
            expected_config_sha256=trace_source_config_sha256(config),
            expected_tasks=config["traces"]["tasks"],
            expected_tokens_per_task=int(config["traces"]["train_tokens_per_task"]),
            expected_checkpoint_config_sha256=source["checkpoint_config_sha256"],
            expected_architecture_config_sha256=model_identity[
                "current_architecture_config_sha256"
            ],
            expected_checkpoint_sha256=model_identity["checkpoint_sha256"],
            expected_model_dtype=model_identity["dtype"],
            expected_tokenizer_sha256=tokenizer_sha256,
            expected_generation_code_sha256=args.expected_generation_code_sha256,
            expected_seed_base=int(config["traces"]["train_seed_base"]),
            expected_generation_contract=trace_generation_contract(config["traces"]),
        )
    dist.barrier()

    effective_tokens = int(optimization["effective_tokens"])
    total_tokens = int(config["traces"]["train_tokens_per_task"]) * len(
        config["traces"]["tasks"]
    )
    if args.max_tokens is not None:
        if not 0 < int(args.max_tokens) <= total_tokens:
            raise ValueError("max tokens must be within the frozen Trace quota")
        total_tokens = int(args.max_tokens)
    labels = [label for label, _value in learning_rates]
    output_root = repo_path(args.output_root).resolve()
    candidate_dirs = _candidate_dirs(
        output_root,
        layer=int(args.layer),
        width_multiplier=int(args.width_multiplier),
        seed=int(args.seed),
        labels=labels,
    )
    resume_paths: dict[str, Path | None] = {label: None for label in labels}
    if rank == 0:
        resume_paths = _common_resume_paths(candidate_dirs)
        _truncate_metrics_to_resume(candidate_dirs, resume_paths)
        execution_identity = {
            "format": "s5_sharded_transcoder_execution_identity_v1",
            "config_sha256": config["_config_sha256"],
            "trace_source_config_sha256": trace_source_config_sha256(config),
            "trace_generation_code_sha256": args.expected_generation_code_sha256,
            "checkpoint_sha256": source["checkpoint_sha256"],
            "layer": int(args.layer),
            "width_multiplier": int(args.width_multiplier),
            "seed": int(args.seed),
            "learning_rates": dict(learning_rates),
            "run_kind": str(args.run_kind),
            "selection_contract_sha256": (
                manifest_metadata.get("selection_contract_sha256")
                if manifest_metadata is not None
                else None
            ),
            "world_size": world_size,
            "total_tokens": total_tokens,
        }
        execution_identity["identity_sha256"] = hashlib.sha256(
            json.dumps(execution_identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / f"layer_{int(args.layer)}_execution_identity.json").write_text(
            json.dumps(execution_identity, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    result = train_transcoder_layer_group_distributed(
        model,
        iter_trace_chunks_round_robin(trace_paths) if rank == 0 else None,
        layer=int(args.layer),
        d_feature=int(args.width_multiplier) * int(model.config.d_model),
        seed=int(args.seed),
        learning_rates=learning_rates,
        output_dirs=candidate_dirs,
        resume_paths=resume_paths,
        device=device,
        config_sha256=config["_config_sha256"],
        effective_tokens=effective_tokens,
        microbatch_tokens=effective_tokens // world_size,
        beta1=float(optimization["beta1"]),
        beta2=float(optimization["beta2"]),
        eps=float(optimization["eps"]),
        weight_decay=float(optimization["weight_decay"]),
        max_grad_norm=float(optimization["max_grad_norm"]),
        warmup_fraction=float(optimization["warmup_fraction"]),
        cooldown_fraction=float(optimization["cooldown_fraction"]),
        checkpoint_percent=(
            int(args.checkpoint_percent_override)
            if args.checkpoint_percent_override is not None
            else int(optimization["checkpoint_percent"])
        ),
        sparsity_coeff=float(config["transcoder"]["sparsity_coeff"]),
        c_coeff=float(config["transcoder"]["c_coeff"]),
        total_tokens=total_tokens,
        progress_interval=int(args.progress_interval),
        manifest_metadata=manifest_metadata,
    )
    if rank == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

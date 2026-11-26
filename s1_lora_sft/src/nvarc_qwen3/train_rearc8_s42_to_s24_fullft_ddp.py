#!/usr/bin/env python3
"""Continue the frozen RE-ARC8 seed42 checkpoint on seed24-exclusive rows."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
import yaml
from datasets import Dataset, load_from_disk
from torch.utils.data import DataLoader, Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer

from nvarc_qwen3.fullft_ddp.checkpoint import checkpoint_complete, save_hf_checkpoint
from nvarc_qwen3.fullft_ddp.config import configure_process_batch, learning_rate_for_step
from nvarc_qwen3.fullft_ddp.runtime import (
    CachedTokenizedNvarcDataset,
    TrainingError,
    append_jsonl,
    assert_same_fingerprint,
    atomic_write_json,
    distributed_totals,
    evaluate,
    model_fingerprint,
    normalize_ddp_gradients,
    optimizer_parameter_groups,
    set_seed,
    setup_distributed,
    wrap_ddp,
)
from nvarc_qwen3.nvarc_sft_data import (
    RankStridedSequentialSampler,
    SFTCollator,
    causal_loss_sum,
)
from nvarc_qwen3.rearc_scale_schedule import schedule_manifest


FORMAT = "nvarc_rearc8_s42_to_s24_hf_ddp_v1"
SKIP_B1_FORMAT = "nvarc_rearc8_s42_to_s24_skip_b1_hf_ddp_v2"
FULL_SCHEDULE_EPOCHS = 50
RECOVERY_FORMAT = "nvarc_arc1_direct_retrain_resume_v1"
REPLICATION_PROTOCOL = "arc1_paired_multiseed_v1"


def replication_contract(config: dict[str, Any]) -> dict[str, Any] | None:
    """Validate the paired replication."""
    if "replication" not in config:
        return None
    value = config["replication"]
    expected_keys = {"protocol", "seed", "data_seed", "world_size", "full_schedule_sha256"}
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise TrainingError("Replication must declare the complete paired-multiseed contract")
    if value["protocol"] != REPLICATION_PROTOCOL:
        raise TrainingError("Unsupported ARC1 replication protocol")
    if (
        type(value["seed"]) is not int
        or value["seed"] not in range(42, 47)
        or type(value["data_seed"]) is not int
        or value["data_seed"] != value["seed"]
        or type(value["world_size"]) is not int
        or value["world_size"] <= 0
    ):
        raise TrainingError("Paired replication requires matching seeds 42--46 and positive process count")
    digest = value["full_schedule_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise TrainingError("Replication requires a lowercase full schedule SHA256")
    if config.get("format_version") != SKIP_B1_FORMAT:
        raise TrainingError("Paired replication requires the frozen skip-B1 recipe")
    return dict(value)


def resolve_stop_step(config: dict[str, Any], stop_after_step: int | None) -> int:
    planned = int(config["training"]["max_steps"])
    target = planned if stop_after_step is None else int(stop_after_step)
    if not 1 <= target <= planned:
        raise TrainingError("stop-after-step is outside the configured training range")
    return target


def recovery_identity(
    args: argparse.Namespace, config: dict[str, Any], schedule: dict[str, Any]
) -> dict[str, Any]:
    return {
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "base_model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "dataset_semantic_sha256": args.expected_dataset_semantic_sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "world_size": int(config["training"]["world_size"]),
        "global_batch_size": int(config["training"]["global_batch_size"]),
        "planned_total_steps": int(config["training"]["max_steps"]),
    }


def gather_rank_rng_states(context: Any) -> list[dict[str, Any]]:
    state = {
        "rank": context.rank,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(context.device) if context.device.type == "cuda" else None,
    }
    if context.world_size == 1:
        return [state]
    states: list[Any] = [None] * context.world_size
    dist.all_gather_object(states, state)
    if any(not isinstance(item, dict) or item.get("rank") != rank for rank, item in enumerate(states)):
        raise TrainingError("Recovery RNG collection has incomplete rank coverage")
    return states


def cpu_state(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: cpu_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(item) for item in value)
    return value


def build_recovery_state(
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    identity: dict[str, Any],
    rng_by_rank: list[dict[str, Any]],
) -> dict[str, Any]:
    raw_model = model.module if hasattr(model, "module") else model
    return {
        "format": RECOVERY_FORMAT,
        "identity": identity,
        "completed_steps": int(step),
        "schedule_cursor": int(step) * identity["global_batch_size"],
        # The inference HF export is BF16. Resume must retain the original
        # FP32 training parameters and Adam states without that rounding.
        "model_state_dict": cpu_state(raw_model.state_dict()),
        "optimizer_state_dict": cpu_state(optimizer.state_dict()),
        "rng_by_rank": rng_by_rank,
    }


def write_recovery_metadata(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    record = {
        "format": RECOVERY_FORMAT,
        "status": "complete",
        "completed_steps": payload["completed_steps"],
        "planned_total_steps": payload["identity"]["planned_total_steps"],
        "schedule_cursor": payload["schedule_cursor"],
        "world_size": payload["identity"]["world_size"],
        "identity": payload["identity"],
        "recovery_file": path.name,
        "recovery_bytes": path.stat().st_size,
    }
    atomic_write_json(path.with_suffix(".json"), record)
    return record


def load_recovery_state(
    checkpoint: Path,
    model: Any,
    optimizer: torch.optim.Optimizer,
    *,
    identity: dict[str, Any],
    context: Any,
    stop_step: int,
) -> tuple[int, dict[str, Any]]:
    path = checkpoint / "training_recovery.pt" if checkpoint.is_dir() else checkpoint
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (
        metadata.get("format") != RECOVERY_FORMAT
        or metadata.get("status") != "complete"
        or metadata.get("identity") != identity
        or metadata.get("planned_total_steps") != identity["planned_total_steps"]
        or metadata.get("world_size") != context.world_size
        or identity["world_size"] != context.world_size
        or metadata.get("recovery_file") != path.name
        or metadata.get("recovery_bytes") != path.stat().st_size
    ):
        raise TrainingError("Recovery checkpoint metadata or identity mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format") != RECOVERY_FORMAT or payload.get("identity") != identity:
        raise TrainingError("Recovery checkpoint configuration or data identity mismatch")
    step = int(payload.get("completed_steps", -1))
    if not 0 <= step < stop_step:
        raise TrainingError("Recovery checkpoint has already reached the requested stop step")
    if payload.get("schedule_cursor") != step * identity["global_batch_size"]:
        raise TrainingError("Recovery checkpoint schedule cursor mismatch")
    if metadata.get("completed_steps") != step or metadata.get("schedule_cursor") != payload["schedule_cursor"]:
        raise TrainingError("Recovery checkpoint metadata step or cursor mismatch")
    states = payload.get("rng_by_rank")
    if not isinstance(states, list) or len(states) != context.world_size or any(
        not isinstance(state, dict)
        or state.get("rank") != rank
        or not {"python", "numpy", "torch", "cuda"} <= state.keys()
        for rank, state in enumerate(states)
    ):
        raise TrainingError("Recovery checkpoint is missing per-rank RNG states")
    state = states[context.rank]
    if context.device.type == "cuda" and state["cuda"] is None:
        raise TrainingError("Recovery checkpoint is missing CUDA RNG state")
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return step, state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def loss_protocol(config: dict[str, Any]) -> dict[str, Any]:
    if config["format_version"] == FORMAT:
        return {
            "name": "all_assistant_events_v1",
            "skip_first_assistant_loss": False,
            "supervised_assistant_start_index": 0,
        }
    protocol = config.get("loss_protocol")
    expected = {
        "name": "first_pair_demonstration_only_v1",
        "skip_first_assistant_loss": True,
        "supervised_assistant_start_index": 1,
    }
    if protocol != expected:
        raise TrainingError(
            f"Unexpected skip-B1 loss protocol: observed={protocol!r} expected={expected!r}"
        )
    return dict(expected)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument("--expected-full-schedule-sha256", required=True)
    parser.add_argument("--expected-dataset-semantic-sha256", required=True)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--resume", type=Path, help="Target-mode step directory or training_recovery.pt")
    return parser.parse_args(argv)


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format_version") not in {
        FORMAT,
        SKIP_B1_FORMAT,
    }:
        raise TrainingError(f"Unexpected continuation config: {path}")
    loss_protocol(payload)
    training = payload["training"]
    configure_process_batch(training)
    replication = replication_contract(payload)
    if replication is not None and training["world_size"] != replication["world_size"]:
        raise TrainingError("Replication topology differs from the launch")
    locked = {
        "seed": 42,
        "global_batch_size": 256,
        "micro_batch_size": 1,
        "steps_per_epoch": 13,
    }
    if replication is not None:
        locked["seed"] = replication["seed"]
    changed = [
        f"{key}={training.get(key)!r}, expected {expected!r}"
        for key, expected in locked.items()
        if training.get(key) != expected
    ]
    if changed:
        raise TrainingError("Continuation training contract changed: " + "; ".join(changed))
    if (
        training["world_size"]
        * training["micro_batch_size"]
        * training["gradient_accumulation_steps"]
        != training["global_batch_size"]
    ):
        raise TrainingError("world_size * micro_batch * accumulation must equal global batch")
    observed_schedule = (
        int(training["logical_epochs"]),
        int(training["max_steps"]),
        float(payload["optimizer"]["learning_rate"]),
        int(payload["checkpointing"]["save_period"]),
    )
    allowed_schedules = {
        (10, 130, 2.0e-5, 26),
        (6, 78, 7.5e-6, 13),
        (6, 78, 1.0e-5, 13),
        (6, 78, 1.2e-5, 13),
        (6, 78, 1.5e-5, 13),
    }
    if observed_schedule not in allowed_schedules:
        raise TrainingError(
            "Unsupported continuation schedule: "
            f"observed={observed_schedule!r}, expected one of "
            f"{sorted(allowed_schedules)!r}"
        )
    if replication is not None and observed_schedule != (6, 78, 1.0e-5, 13):
        raise TrainingError("Paired replication fixes the original LR1e-5/78-step recipe")
    if int(payload["scheduler"]["decay_steps"]) != int(training["max_steps"]):
        raise TrainingError("Scheduler decay_steps must equal training max_steps")
    if payload["optimizer"]["weight_decay"] != 0.1:
        raise TrainingError("This baseline locks weight decay to 0.1")
    expected_normalization = (
        "global_supervised_assistant_tokens"
        if payload["format_version"] == SKIP_B1_FORMAT
        else "global_assistant_tokens"
    )
    if training.get("loss_normalization") != expected_normalization:
        raise TrainingError(
            "Unexpected continuation loss normalization: "
            f"{training.get('loss_normalization')!r} != {expected_normalization!r}"
        )
    return payload


def load_nvarc_dataset(path: Path) -> Dataset:
    dataset = load_from_disk(str(path))
    if not isinstance(dataset, Dataset):
        raise TrainingError(f"Expected one Dataset at {path}")
    if dataset.column_names != ["puzzle_name", "messages"]:
        raise TrainingError(f"Unexpected dataset schema at {path}: {dataset.column_names}")
    return dataset


class RankScheduleSampler(Sampler[int]):
    """Shard an already-expanded global schedule without changing row order."""

    def __init__(
        self,
        indices: np.ndarray,
        *,
        rank: int,
        world_size: int,
        global_batch_size: int,
        micro_batch_size: int,
        max_steps: int,
        completed_steps: int = 0,
    ) -> None:
        local_width = world_size * micro_batch_size
        if global_batch_size % local_width:
            raise TrainingError("Global batch is not divisible by local width")
        self.gradient_accumulation_steps = global_batch_size // local_width
        needed = max_steps * global_batch_size
        if len(indices) < needed:
            raise TrainingError("Expanded schedule is shorter than requested training")
        if not 0 <= completed_steps <= max_steps:
            raise TrainingError("Completed schedule steps are outside the requested training range")
        shaped = np.asarray(indices[:needed], dtype=np.int64).reshape(
            max_steps,
            self.gradient_accumulation_steps,
            world_size,
            micro_batch_size,
        )
        self._indices = shaped[completed_steps:, :, rank, :].reshape(-1)

    def __iter__(self) -> Iterator[int]:
        return (int(index) for index in self._indices)

    def __len__(self) -> int:
        return len(self._indices)


def build_schedule(
    args: argparse.Namespace,
    config: dict[str, Any],
    dataset: Dataset,
) -> tuple[np.ndarray, dict[str, Any]]:
    training = config["training"]
    replication = replication_contract(config)
    if replication is not None and args.expected_full_schedule_sha256 != replication["full_schedule_sha256"]:
        raise TrainingError("CLI full schedule SHA256 disagrees with the replication contract")
    puzzle_names = [str(name) for name in dataset["puzzle_name"]]
    _, full_manifest = schedule_manifest(
        puzzle_names,
        rows_per_puzzle=8,
        seed=int(training["seed"]),
        logical_epochs=FULL_SCHEDULE_EPOCHS,
        global_batch_size=int(training["global_batch_size"]),
    )
    if full_manifest["schedule_sha256"] != args.expected_full_schedule_sha256:
        raise TrainingError(
            "Pinned 50-epoch seed24-exclusive schedule changed: "
            f"{full_manifest['schedule_sha256']} != {args.expected_full_schedule_sha256}"
        )
    schedule, manifest = schedule_manifest(
        puzzle_names,
        rows_per_puzzle=8,
        seed=int(training["seed"]),
        logical_epochs=int(training["logical_epochs"]),
        global_batch_size=int(training["global_batch_size"]),
    )
    manifest.update(
        {
            "backend": "huggingface_pytorch_ddp",
            "base_checkpoint_role": "rearc8_s42_e10_step130",
            "training_data_role": "seed24_exclusive_of_seed42",
            "base_dataset_path": str(args.train_dataset),
            "base_dataset_storage_fingerprint": getattr(dataset, "_fingerprint", None),
            "base_dataset_semantic_sha256": args.expected_dataset_semantic_sha256,
            "full_50_epoch_schedule_sha256": full_manifest["schedule_sha256"],
            "micro_batch_size": int(training["micro_batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "peak_lr": float(config["optimizer"]["learning_rate"]),
            "weight_decay": float(config["optimizer"]["weight_decay"]),
            "loss_protocol": loss_protocol(config),
        }
    )
    if replication is not None:
        manifest["replication"] = replication
    if manifest["steps_per_logical_epoch"] != int(training["steps_per_epoch"]):
        raise TrainingError("Derived steps per logical epoch changed")
    if manifest["max_steps"] != int(training["max_steps"]):
        raise TrainingError("Derived maximum step count changed")
    return schedule.indices, manifest


def validate_artifacts(args: argparse.Namespace) -> None:
    for path, marker in (
        (args.model, "config.json"),
        (args.tokenizer, "tokenizer_config.json"),
        (args.train_dataset, "dataset_info.json"),
        (args.validation_dataset, "dataset_info.json"),
    ):
        if not (path / marker).is_file():
            raise FileNotFoundError(path / marker)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    replication = replication_contract(config)
    effective_steps = resolve_stop_step(config, args.stop_after_step)
    target_mode = args.stop_after_step is not None
    if args.resume is not None and not target_mode:
        raise TrainingError("--resume requires an explicit --stop-after-step target")
    if target_mode and int(config["data"]["num_workers"]) != 0:
        raise TrainingError("Target-mode recovery requires num_workers=0 for an exact data cursor")
    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.is_relative_to(args.output_root.resolve() / "checkpoints"):
            raise TrainingError("Resume checkpoint must belong to this output root")
    validate_artifacts(args)
    training = config["training"]
    context = setup_distributed(
        expected_world_size=int(training["world_size"]),
        expected_gpus_per_node=int(training["gpus_per_node"]),
    )
    try:
        set_seed(int(training["seed"]))
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        args.log_root.mkdir(parents=True, exist_ok=True)
        if context.is_rank_zero:
            if args.resume is None:
                args.output_root.mkdir(parents=True, exist_ok=False)
                (args.output_root / "checkpoints").mkdir()
                (args.log_root / "metrics.jsonl").unlink(missing_ok=True)
            elif not (args.output_root / "checkpoints").is_dir():
                raise TrainingError("Resume output root is missing its checkpoint directory")
        dist.barrier()

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
        if tokenizer.pad_token_id is None:
            raise TrainingError("Tokenizer has no pad token")
        train_dataset = load_nvarc_dataset(args.train_dataset)
        validation_dataset = load_nvarc_dataset(args.validation_dataset)
        indices, schedule = build_schedule(args, config, train_dataset)
        if context.is_rank_zero:
            atomic_write_json(args.schedule_manifest, schedule)

        model_cfg = config["model"]
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation=model_cfg["attention_implementation"],
        )
        model.config.use_cache = False
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != int(model_cfg["expected_parameters"]):
            raise TrainingError(
                f"Expected {model_cfg['expected_parameters']} parameters, got {parameter_count}"
            )
        fingerprint = model_fingerprint(model)
        assert_same_fingerprint(fingerprint, context.world_size)
        model.to(context.device)
        model.train()
        ddp_model = wrap_ddp(model, context)
        parameters = [p for p in ddp_model.parameters() if p.requires_grad]

        optimizer_cfg = config["optimizer"]
        optimizer = torch.optim.AdamW(
            optimizer_parameter_groups(
                ddp_model, weight_decay=float(optimizer_cfg["weight_decay"])
            ),
            lr=float(optimizer_cfg["learning_rate"]),
            betas=(float(optimizer_cfg["beta1"]), float(optimizer_cfg["beta2"])),
            eps=float(optimizer_cfg["epsilon"]),
            fused=bool(optimizer_cfg["fused"]),
        )
        identity = recovery_identity(args, config, schedule)
        completed_steps = 0
        restored_rng = None
        if args.resume is not None:
            completed_steps, restored_rng = load_recovery_state(
                args.resume,
                ddp_model,
                optimizer,
                identity=identity,
                context=context,
                stop_step=effective_steps,
            )
            fingerprint = model_fingerprint(model)
            assert_same_fingerprint(fingerprint, context.world_size)

        max_length = int(model_cfg["max_sequence_length"])
        protocol = loss_protocol(config)
        skip_first_assistant_loss = bool(protocol["skip_first_assistant_loss"])
        train_wrapped = CachedTokenizedNvarcDataset(
            train_dataset,
            tokenizer,
            max_length,
            skip_first_assistant_loss=skip_first_assistant_loss,
        )
        validation_wrapped = CachedTokenizedNvarcDataset(
            validation_dataset,
            tokenizer,
            max_length,
            skip_first_assistant_loss=skip_first_assistant_loss,
        )
        collator = SFTCollator(
            pad_token_id=int(tokenizer.pad_token_id),
            pad_to_multiple_of=int(config["data"]["pad_to_multiple_of"]),
        )
        sampler = RankScheduleSampler(
            indices,
            rank=context.rank,
            world_size=context.world_size,
            global_batch_size=int(training["global_batch_size"]),
            micro_batch_size=int(training["micro_batch_size"]),
            max_steps=effective_steps,
            completed_steps=completed_steps,
        )
        train_loader = DataLoader(
            train_wrapped,
            batch_size=int(training["micro_batch_size"]),
            sampler=sampler,
            collate_fn=collator,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=True,
            drop_last=True,
        )
        validation_loader = DataLoader(
            validation_wrapped,
            batch_size=int(config["validation"]["micro_batch_size"]),
            sampler=RankStridedSequentialSampler(
                sample_count=len(validation_dataset),
                rank=context.rank,
                world_size=context.world_size,
            ),
            collate_fn=collator,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=True,
            drop_last=False,
        )

        if context.is_rank_zero:
            trainer_manifest = {
                "format_version": (
                    "nvarc_rearc8_s42_to_s24_skip_b1_trainer_v2"
                    if skip_first_assistant_loss
                    else "nvarc_rearc8_s42_to_s24_trainer_v1"
                ),
                "status": "running",
                "backend": "huggingface_pytorch_ddp",
                "base_model": str(args.model),
                "train_dataset": str(args.train_dataset),
                "world_size": context.world_size,
                "global_batch_size": training["global_batch_size"],
                "micro_batch_size": training["micro_batch_size"],
                "gradient_accumulation_steps": sampler.gradient_accumulation_steps,
                "parameter_count": parameter_count,
                "model_fingerprint": fingerprint,
                "peak_lr": optimizer_cfg["learning_rate"],
                "weight_decay": optimizer_cfg["weight_decay"],
                "max_steps": effective_steps,
                "schedule_sha256": schedule["schedule_sha256"],
                "loss_protocol": protocol,
                **({"replication": replication} if replication is not None else {}),
                **(
                    {
                        "planned_total_steps": int(training["max_steps"]),
                        "stop_after_step": int(args.stop_after_step),
                        "actual_steps": completed_steps,
                        "resume_checkpoint": str(args.resume) if args.resume else None,
                    }
                    if target_mode
                    else {}
                ),
            }
            atomic_write_json(args.log_root / "trainer_manifest.json", trainer_manifest)
            print(json.dumps({"event": "trainer_ready", **trainer_manifest}), flush=True)

        if config["validation"]["at_start"] and args.resume is None:
            val_loss, val_tokens = evaluate(ddp_model, validation_loader, device=context.device)
            if context.is_rank_zero:
                append_jsonl(
                    args.log_root / "metrics.jsonl",
                    {
                        "event": "validation",
                        "step": 0,
                        "validation_loss": val_loss,
                        "tokens": val_tokens,
                    },
                )

        iterator = iter(train_loader)
        if restored_rng is not None:
            # DataLoader iterator construction consumes a Torch base seed even
            # with zero workers. Restore the checkpoint RNG only afterwards.
            restore_rng_state(restored_rng, context.device)
        if target_mode and context.is_rank_zero:
            (args.output_root / "target_reached.ready").unlink(missing_ok=True)
        scheduler = config["scheduler"]
        started = time.perf_counter()
        checkpoint_period = int(config["checkpointing"]["save_period"])
        validation_period = int(config["validation"]["period"])
        target_checkpoint = None
        for step in range(completed_steps + 1, effective_steps + 1):
            learning_rate = learning_rate_for_step(
                step,
                peak_lr=float(optimizer_cfg["learning_rate"]),
                warmup_steps=int(scheduler["warmup_steps"]),
                warmup_initial_lr=float(scheduler["warmup_initial_lr"]),
                decay_steps=int(scheduler["decay_steps"]),
                min_lr=float(scheduler["min_lr"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=context.device, dtype=torch.float64)
            local_tokens = torch.zeros((), device=context.device, dtype=torch.long)
            step_started = time.perf_counter()
            for micro_step in range(sampler.gradient_accumulation_steps):
                batch = next(iterator)
                synchronize = micro_step == sampler.gradient_accumulation_steps - 1
                sync_context = contextlib.nullcontext() if synchronize else ddp_model.no_sync()
                with sync_context:
                    input_ids = batch["input_ids"].to(context.device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(
                        context.device, non_blocking=True
                    )
                    labels = batch["labels"].to(context.device, non_blocking=True)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = ddp_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        ).logits
                        loss_sum, valid_tokens = causal_loss_sum(logits, labels)
                    loss_sum.backward()
                    local_loss += loss_sum.detach().double()
                    local_tokens += valid_tokens.detach()
            global_loss, global_tokens = distributed_totals(local_loss, local_tokens)
            normalize_ddp_gradients(
                parameters, world_size=context.world_size, global_tokens=global_tokens
            )
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(optimizer_cfg["clip_grad_norm"]),
                error_if_nonfinite=True,
            )
            optimizer.step()
            torch.cuda.synchronize(context.device)
            if context.is_rank_zero:
                record = {
                    "event": "train",
                    "step": step,
                    "logical_epoch": math.ceil(step / training["steps_per_epoch"]),
                    "loss": global_loss / global_tokens,
                    "assistant_tokens": global_tokens,
                    "learning_rate": learning_rate,
                    "grad_norm": float(grad_norm.item()),
                    "step_seconds": time.perf_counter() - step_started,
                    "max_memory_allocated_gib": torch.cuda.max_memory_allocated(
                        context.device
                    )
                    / 2**30,
                }
                append_jsonl(args.log_root / "metrics.jsonl", record)
                print(json.dumps(record), flush=True)

            if step % validation_period == 0:
                val_loss, val_tokens = evaluate(
                    ddp_model, validation_loader, device=context.device
                )
                if context.is_rank_zero:
                    append_jsonl(
                        args.log_root / "metrics.jsonl",
                        {
                            "event": "validation",
                            "step": step,
                            "validation_loss": val_loss,
                            "tokens": val_tokens,
                        },
                    )

            is_target_checkpoint = target_mode and step == effective_steps
            if step % checkpoint_period == 0 or is_target_checkpoint:
                dist.barrier()
                rng_by_rank = gather_rank_rng_states(context) if target_mode else None
                if context.is_rank_zero:
                    recovery = (
                        build_recovery_state(
                            ddp_model, optimizer, step=step, identity=identity, rng_by_rank=rng_by_rank
                        )
                        if rng_by_rank is not None
                        else None
                    )
                    checkpoint = save_hf_checkpoint(
                        ddp_model,
                        tokenizer,
                        checkpoint_root=args.output_root / "checkpoints",
                        step=step,
                        trainer_state={
                            "format_version": (
                                "nvarc_rearc8_s42_to_s24_skip_b1_checkpoint_v2"
                                if skip_first_assistant_loss
                                else "nvarc_rearc8_s42_to_s24_checkpoint_v1"
                            ),
                            "completed_steps": step,
                            "logical_epoch": step / training["steps_per_epoch"],
                            "backend": "huggingface_pytorch_ddp",
                            "base_model": str(args.model),
                            "train_dataset": str(args.train_dataset),
                            "peak_lr": optimizer_cfg["learning_rate"],
                            "weight_decay": optimizer_cfg["weight_decay"],
                            "schedule_sha256": schedule["schedule_sha256"],
                            "loss_protocol": protocol,
                            **({"replication": replication} if replication is not None else {}),
                            "saved_weight_dtype": config["checkpointing"]["save_dtype"],
                            **(
                                {
                                    "planned_total_steps": int(training["max_steps"]),
                                    "stop_after_step": int(args.stop_after_step),
                                    "recovery_state_required": True,
                                    "resume_supported": True,
                                }
                                if target_mode
                                else {}
                            ),
                        },
                        max_shard_size=str(config["checkpointing"]["max_shard_size"]),
                        **({"recovery_state": recovery} if target_mode else {}),
                    )
                    if recovery is not None:
                        write_recovery_metadata(checkpoint / "training_recovery.pt", recovery)
                    if is_target_checkpoint:
                        target_checkpoint = checkpoint
                    del recovery
                    print(f"hf_checkpoint_saved step={step} path={checkpoint}", flush=True)
                dist.barrier()

        if context.is_rank_zero:
            complete = {
                **json.loads(
                    (args.log_root / "trainer_manifest.json").read_text(encoding="utf-8")
                ),
                "status": "target_reached" if target_mode else "complete",
                "wall_seconds": time.perf_counter() - started,
                "completed_steps": effective_steps,
                **(
                    {
                        "planned_total_steps": int(training["max_steps"]),
                        "stop_after_step": int(args.stop_after_step),
                        "actual_steps": effective_steps,
                        "global_step": effective_steps,
                        "resume_supported": True,
                    }
                    if target_mode
                    else {}
                ),
            }
            if target_mode:
                if target_checkpoint is None or not checkpoint_complete(target_checkpoint):
                    raise TrainingError("Target reached without a complete HF/recovery checkpoint")
                recovery_path = target_checkpoint / "training_recovery.pt"
                if not recovery_path.is_file():
                    raise TrainingError("Target reached without full training recovery state")
                if not recovery_path.with_suffix(".json").is_file():
                    raise TrainingError("Target reached without training recovery metadata")
                complete.update(
                    model_checkpoint={"path": str(target_checkpoint.resolve()), "step": effective_steps},
                    resume_checkpoint={"path": str(recovery_path.resolve()), "step": effective_steps},
                )
            atomic_write_json(args.log_root / "trainer_manifest.json", complete)
            if target_mode:
                atomic_write_json(args.output_root / "target_reached_summary.json", complete)
                atomic_write_json(args.output_root / "target_reached.ready", complete)
            print(f"rearc8_s42_to_s24_fullft={complete['status']} steps={effective_steps}", flush=True)
        return 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run FullFT on a pre-expanded deterministic RE-ARC logical-epoch schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import pprint
from pathlib import Path
from typing import Any, Mapping


PRINTED_SAMPLE = False


class _RetrainTargetReached(Exception):
    """Driver-local control flow after the native checkpoint has finalized."""


def validate_stop_after_step(
    config: Mapping[str, Any], stop_after_step: int | None
) -> int | None:
    """Validate a save-boundary stop without changing the training horizon."""

    if stop_after_step is None:
        return None
    planned = int(config["sft"]["max_num_steps"])
    target = int(stop_after_step)
    period = int(config["checkpointing"]["save_period"])
    if not 0 < target <= planned:
        raise ValueError("stop-after-step must lie within the original training horizon")
    if not config["checkpointing"]["enabled"] or period <= 0 or target % period:
        raise ValueError("stop-after-step must be an enabled original checkpoint boundary")
    megatron = config["policy"].get("megatron_cfg", {})
    if not megatron.get("enabled"):
        raise ValueError("Checkpoint-boundary stopping requires the original Megatron backend")
    if int(megatron["scheduler"]["lr_decay_iters"]) != planned:
        raise ValueError("Megatron scheduler must retain the original full training horizon")
    if int(megatron.get("train_iters", planned)) != planned:
        raise ValueError("Megatron train_iters must retain the original full training horizon")
    return target


def _small_file_record(path: Path, root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def _distributed_checkpoint_files(root: Path) -> list[Path]:
    """Check every native DCP storage reference without reading weight shards."""

    # This is metadata just written by our own Megatron workers, not an
    # arbitrary downloaded pickle. Its PyTorch classes exist in that runtime.
    with (root / ".metadata").open("rb") as handle:
        metadata = pickle.load(handle)
    storage = metadata.get("storage_data") if isinstance(metadata, Mapping) else getattr(metadata, "storage_data", None)
    if not isinstance(storage, Mapping) or not storage:
        raise RuntimeError("Target Megatron metadata has no distributed storage references")
    files = set()
    for record in storage.values():
        def field(name: str) -> Any:
            return record[name] if isinstance(record, Mapping) else getattr(record, name)
        path = (root / str(field("relative_path"))).resolve()
        offset, length = int(field("offset")), int(field("length"))
        if not path.is_relative_to(root.resolve()) or offset < 0 or length <= 0:
            raise RuntimeError("Target Megatron metadata has an invalid shard reference")
        if not path.is_file() or path.stat().st_size < offset + length:
            raise RuntimeError("Target Megatron checkpoint has a missing or truncated referenced shard")
        files.add(path)
    return sorted(files)


class TargetStopCheckpointer:
    """Stop pinned NeMo-RL only after its ordinary save/finalize completes.

    NeMo-RL 779f775 has a Ray training loop rather than a Lightning Trainer or
    callback parameter. Passing this per-call adapter preserves its loop and
    CheckpointManager; no upstream module, worker, or scheduler is patched.
    """

    def __init__(
        self, delegate: Any, config: Mapping[str, Any], *, stop_after_step: int,
        logical_epochs: int, resume_state: Mapping[str, Any] | None = None,
    ) -> None:
        self._delegate = delegate
        self._config = config
        self.target = validate_stop_after_step(config, stop_after_step)
        self.planned = int(config["sft"]["max_num_steps"])
        self.logical_epochs = int(logical_epochs)
        self.checkpoint_dir = Path(config["checkpointing"]["checkpoint_dir"]).resolve()
        if Path(delegate.checkpoint_dir).resolve() != self.checkpoint_dir:
            raise ValueError("NeMo checkpoint directory differs from the configured directory")
        self.output_dir = self.checkpoint_dir.parent
        self._pending: tuple[Path, int] | None = None
        self.record: dict[str, Any] | None = None
        if (self.output_dir / "target_reached.ready").exists():
            raise FileExistsError("Retraining target already has a completion marker")
        if resume_state and int(resume_state["total_steps"]) >= self.target:
            raise ValueError("Resume checkpoint has already reached or passed the stop target")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def init_tmp_checkpoint(
        self, step: int, training_info: Mapping[str, Any],
        run_config: Mapping[str, Any] | None = None,
    ) -> Any:
        if int(step) > self.target or int(training_info["total_steps"]) != int(step):
            raise RuntimeError("NeMo checkpoint step passed or disagrees with the stop target")
        if run_config is None or validate_stop_after_step(run_config, self.target) != self.target:
            raise RuntimeError("NeMo checkpoint omitted its original training configuration")
        if int(run_config["sft"]["max_num_steps"]) != self.planned:
            raise RuntimeError("NeMo changed the planned training horizon before saving")
        temporary = self._delegate.init_tmp_checkpoint(step, training_info, run_config)
        temporary_path = Path(temporary).resolve()
        if temporary_path != self.checkpoint_dir / f"tmp_step_{int(step)}":
            raise RuntimeError("Unexpected pinned NeMo temporary checkpoint layout")
        self._pending = (temporary_path, int(step))
        return temporary

    def finalize_checkpoint(self, checkpoint_path: str | Path) -> Any:
        if self._pending is None or Path(checkpoint_path).resolve() != self._pending[0]:
            raise RuntimeError("Finalization did not match the pending native checkpoint")
        step = self._pending[1]
        result = self._delegate.finalize_checkpoint(checkpoint_path)
        self._pending = None
        if step != self.target:
            return result

        # Original finalize renames tmp_step_N after all Ray workers finish.
        # Megatron stores weights, optimizer and scheduler together under this
        # weights directory; there is no separate optimizer/.metadata contract.
        final = self.checkpoint_dir / f"step_{self.target}"
        required = [final / "training_info.json", final / "config.yaml",
                    final / "train_dataloader.pt",
                    final / "policy/weights/iter_0000000/.metadata",
                    final / "policy/weights/iter_0000000/common.pt",
                    final / "policy/weights/iter_0000000/metadata.json"]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("Target native checkpoint has not completely landed on disk")
        shard_root = final / "policy/weights/iter_0000000"
        shards = _distributed_checkpoint_files(shard_root)
        info = json.loads(required[0].read_text(encoding="utf-8"))
        if int(info["total_steps"]) != self.target:
            raise RuntimeError("Saved native training_info does not match the stop target")
        import yaml

        saved_config = yaml.safe_load(required[1].read_text(encoding="utf-8"))
        validate_stop_after_step(saved_config, self.target)
        if int(saved_config["sft"]["max_num_steps"]) != self.planned:
            raise RuntimeError("Saved checkpoint changed the original scheduler horizon")
        self.record = {
            "format": "nvarc_rearc_scale_target_reached_v1", "status": "target_reached",
            "global_step": self.target, "actual_steps": self.target,
            "stop_after_step": self.target, "planned_total_steps": self.planned,
            "logical_epochs": self.logical_epochs,
            "world_size": int(saved_config["cluster"]["num_nodes"])
            * int(saved_config["cluster"]["gpus_per_node"]),
            "checkpoint_path": str(final), "checkpoint_format": "nemo_rl_megatron_merged_state",
            "checkpoint_metadata": [_small_file_record(path, final) for path in required],
            "distributed_shards": [
                {"path": path.relative_to(final).as_posix(), "bytes": path.stat().st_size}
                for path in shards
            ],
            "weight_shard_hashes_computed": False,
        }
        write_manifest(self.output_dir / "target_reached_summary.json", self.record)
        write_manifest(self.output_dir / "target_reached.ready", self.record)
        raise _RetrainTargetReached()


def train_to_checkpoint_target(
    train: Any, arguments: tuple[Any, ...], *, stop_after_step: int | None,
    logical_epochs: int,
) -> dict[str, Any] | None:
    """Run the original NeMo function with an optional finalized-save hook."""

    if stop_after_step is None:
        train(*arguments)
        return None
    # The public NeMo 779f775 sft_train interface has configuration at index 5,
    # CheckpointManager at index 8, and SFTSaveState at index 9.
    adapter = TargetStopCheckpointer(
        arguments[8], arguments[5], stop_after_step=stop_after_step,
        logical_epochs=logical_epochs, resume_state=arguments[9],
    )
    updated = (*arguments[:8], adapter, arguments[9])
    try:
        train(*updated)
    except _RetrainTargetReached:
        if adapter.record is None:
            raise RuntimeError("Target stop fired without a verified checkpoint record")
        print(json.dumps(adapter.record, sort_keys=True), flush=True)
        return adapter.record
    raise RuntimeError("NeMo returned before the requested target checkpoint finalized")


def sft_preprocessor(
    sample: dict[str, Any], task_data_spec: Any, tokenizer: Any, max_seq_length: int, idx: int
) -> dict[str, Any]:
    """Top-level copy of the pinned NVARC formatter, safe for Ray serialization."""

    global PRINTED_SAMPLE
    from nemo_rl.data.llm_message_utils import get_formatted_message_log

    tokenized = get_formatted_message_log(
        sample["messages"],
        tokenizer,
        task_data_spec,
        add_bos_token=False,
        add_eos_token=False,
        add_generation_prompt=True,
    )
    length = sum(len(message["token_ids"]) for message in tokenized)
    if not PRINTED_SAMPLE:
        PRINTED_SAMPLE = True
        print(
            f"First scheduled sample idx={idx} messages={len(tokenized)} length={length}"
        )
    return {
        "message_log": tokenized,
        "length": length,
        "extra_env_info": None,
        "loss_multiplier": 1.0,
        "idx": idx,
    }


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rows-per-puzzle", type=int, choices=(8, 16, 32, 64), required=True)
    parser.add_argument("--training-seed", type=int, choices=(24, 42, 123), required=True)
    parser.add_argument("--logical-epochs", type=int, default=50)
    parser.add_argument("--schedule-manifest", type=Path, required=True)
    parser.add_argument(
        "--stop-after-step", type=int,
        help="Stop after this original native checkpoint boundary; preserve the full scheduler.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Construct the distributed model and dataloaders, then exit before training.",
    )
    return parser.parse_known_args()


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = json.loads(path.read_text(encoding="utf-8"))
        if observed != payload:
            raise ValueError(f"Existing schedule manifest differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args, overrides = parse_args()
    from datasets import load_from_disk
    from omegaconf import OmegaConf
    from nvarc_qwen3.rearc_scale_schedule import schedule_manifest

    # NeMo imports are intentionally delayed so CPU preflight can audit every
    # dataset and schedule without requiring the GPU training environment.
    from nemo_rl.algorithms.sft import setup, sft_train
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.datasets import AllTaskProcessedDataset
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.distributed.virtual_cluster import init_ray
    from nemo_rl.utils.config import load_config, parse_hydra_overrides
    from nemo_rl.utils.logger import get_next_experiment_dir

    config = load_config(str(args.config))
    if overrides:
        config = parse_hydra_overrides(config, overrides)
    config = OmegaConf.to_container(config, resolve=True)
    assert isinstance(config, dict)
    if config["data"].get("shuffle") is not False:
        raise ValueError("Expanded schedule requires data.shuffle=false")
    if int(config["sft"]["max_num_epochs"]) != 1:
        raise ValueError("Expanded schedule requires exactly one loader epoch")
    if int(config["sft"]["seed"]) != args.training_seed:
        raise ValueError("Config training seed disagrees with schedule seed")

    train_path = config["data"]["train_dataset_path"]
    if isinstance(train_path, list):
        if len(train_path) != 1:
            raise ValueError("Scale training accepts exactly one base dataset")
        train_path = train_path[0]
    base_dataset = load_from_disk(train_path)
    schedule, manifest = schedule_manifest(
        [str(name) for name in base_dataset["puzzle_name"]],
        rows_per_puzzle=args.rows_per_puzzle,
        seed=args.training_seed,
        logical_epochs=args.logical_epochs,
        global_batch_size=int(config["policy"]["train_global_batch_size"]),
    )
    expected_steps = int(manifest["max_steps"])
    if int(config["sft"]["max_num_steps"]) != expected_steps:
        raise ValueError("Config max steps disagrees with expanded schedule")
    if int(config["sft"]["val_period"]) != int(manifest["steps_per_logical_epoch"]):
        raise ValueError("Validation must run once per logical epoch")
    expected_save_period = 2 * int(manifest["steps_per_logical_epoch"])
    if int(config["checkpointing"]["save_period"]) != expected_save_period:
        raise ValueError("Checkpoint must save once per two logical epochs")
    validate_stop_after_step(config, args.stop_after_step)
    manifest.update(
        {
            "base_dataset_path": str(train_path),
            "base_dataset_fingerprint": getattr(base_dataset, "_fingerprint", None),
        }
    )
    write_manifest(args.schedule_manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return 0

    scheduled_dataset = base_dataset.select(schedule.indices.tolist())
    config["logger"]["log_dir"] = get_next_experiment_dir(config["logger"]["log_dir"])
    pprint.pprint(config)
    init_ray()
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    task_spec = TaskDataSpec(task_name="AGI")

    train_dataset = AllTaskProcessedDataset(
        scheduled_dataset, tokenizer, task_spec, sft_preprocessor
    )
    val_dataset = AllTaskProcessedDataset(
        load_from_disk(config["data"]["val_dataset_path"]),
        tokenizer,
        task_spec,
        sft_preprocessor,
    )
    components = setup(config, tokenizer, train_dataset, val_dataset)
    (
        policy, _cluster, train_dataloader, val_dataloader, loss_fn,
        logger, checkpointer, sft_save_state, master_config,
    ) = components
    if master_config["data"].get("shuffle") is not False:
        raise ValueError("NeMo setup changed the no-shuffle schedule contract")
    if args.setup_only:
        print(
            "rearc_scale_setup=complete "
            f"rows_per_puzzle={args.rows_per_puzzle} seed={args.training_seed}",
            flush=True,
        )
        return 0
    train_to_checkpoint_target(
        sft_train,
        (policy, train_dataloader, val_dataloader, tokenizer, loss_fn,
         master_config, logger, task_spec, checkpointer, sft_save_state),
        stop_after_step=args.stop_after_step, logical_epochs=args.logical_epochs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

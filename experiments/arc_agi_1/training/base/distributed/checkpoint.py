"""Atomic directly-loadable Hugging Face checkpoint writer."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from nvarc_qwen3.fullft_ddp.runtime import (
    TrainingError,
    atomic_write_json,
    sampled_tensor_positions,
)
from safetensors import safe_open
from torch.nn.parallel import DistributedDataParallel


def checkpoint_complete(path: Path) -> bool:
    weights = (
        (path / "model.safetensors.index.json").is_file()
        or any(path.glob("model*.safetensors"))
        or any(path.glob("pytorch_model*.bin"))
    )
    basic = (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "trainer_state.json").is_file()
        and weights
    )
    if not basic:
        return False
    trainer_state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    return not trainer_state.get("recovery_state_required", False) or (
        path / "training_recovery.pt"
    ).is_file()


def _bf16_cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            value = tensor.detach()
            if value.is_floating_point():
                value = value.to(device="cpu", dtype=torch.bfloat16)
            else:
                value = value.to(device="cpu")
            converted[name] = value
    return converted


def _flat_position_coordinates(position: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    coordinates = [0] * len(shape)
    for dimension in range(len(shape) - 1, -1, -1):
        coordinates[dimension] = position % shape[dimension]
        position //= shape[dimension]
    return tuple(coordinates)


def _safetensors_dtype(tensor: torch.Tensor, floating_sample_dtype: torch.dtype | None) -> str:
    dtype = floating_sample_dtype if floating_sample_dtype is not None and tensor.is_floating_point() else tensor.dtype
    names = {
        torch.bool: "BOOL",
        torch.int8: "I8",
        torch.uint8: "U8",
        torch.int16: "I16",
        torch.int32: "I32",
        torch.int64: "I64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float64: "F64",
    }
    try:
        return names[dtype]
    except KeyError as exc:
        raise TrainingError(f"Unsupported checkpoint dtype for fingerprint: {dtype}") from exc


def safetensors_model_fingerprint(
    checkpoint: Path,
    model: torch.nn.Module,
    *,
    floating_sample_dtype: torch.dtype | None = None,
) -> str:
    """Fingerprint a saved checkpoint without reconstructing the full model.

    Every model state-dict key is checked against the safetensors index and
    shard metadata.  Values use the same deterministic sample and dtype
    normalization as :func:`model_fingerprint`, so a BF16 checkpoint is
    compared against the in-memory model after the expected BF16 rounding.
    """

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise TrainingError(f"Invalid safetensors index: {index_path}")
        weight_map = {str(name): str(shard) for name, shard in weight_map.items()}
    else:
        shards = sorted(checkpoint.glob("model*.safetensors"))
        if len(shards) != 1:
            raise TrainingError(
                "A multi-shard checkpoint must have model.safetensors.index.json: "
                f"{checkpoint}"
            )
        with safe_open(str(shards[0]), framework="pt", device="cpu") as handle:
            weight_map = {name: shards[0].name for name in handle.keys()}  # noqa: SIM118

    state_dict = model.state_dict()
    expected_keys = set(state_dict)
    indexed_keys = set(weight_map)
    if indexed_keys != expected_keys:
        missing = sorted(expected_keys - indexed_keys)
        extra = sorted(indexed_keys - expected_keys)
        raise TrainingError(
            "Checkpoint state-dict keys differ from the in-memory model: "
            f"missing={missing[:4]} extra={extra[:4]}"
        )

    digest = hashlib.sha256()
    with ExitStack() as stack:
        handles = {}
        actual_keys: set[str] = set()
        for shard_name in sorted(set(weight_map.values())):
            shard_path = checkpoint / shard_name
            if not shard_path.is_file():
                raise TrainingError(f"Missing safetensors shard: {shard_path}")
            handle = stack.enter_context(
                safe_open(str(shard_path), framework="pt", device="cpu")
            )
            handles[shard_name] = handle
            actual_keys.update(handle.keys())
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise TrainingError(
                "Checkpoint shard keys differ from the in-memory model: "
                f"missing={missing[:4]} extra={extra[:4]}"
            )

        with torch.no_grad():
            for name, tensor in state_dict.items():
                tensor_slice = handles[weight_map[name]].get_slice(name)
                shape = tuple(tensor_slice.get_shape())
                expected_shape = tuple(tensor.shape)
                if shape != expected_shape:
                    raise TrainingError(
                        f"Checkpoint shape mismatch for {name}: {shape} != {expected_shape}"
                    )
                actual_dtype = tensor_slice.get_dtype()
                expected_dtype = _safetensors_dtype(tensor, floating_sample_dtype)
                if actual_dtype != expected_dtype:
                    raise TrainingError(
                        f"Checkpoint dtype mismatch for {name}: "
                        f"{actual_dtype} != {expected_dtype}"
                    )

                digest.update(name.encode("utf-8"))
                digest.update(str(expected_shape).encode("ascii"))
                if tensor.numel():
                    positions = sampled_tensor_positions(tensor.numel())
                    samples = [
                        tensor_slice[_flat_position_coordinates(position, expected_shape)]
                        for position in positions
                    ]
                    sample = torch.stack(samples)
                    if floating_sample_dtype is not None and sample.is_floating_point():
                        sample = sample.to(floating_sample_dtype)
                    digest.update(sample.float().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_hf_checkpoint(
    ddp_model: DistributedDataParallel,
    tokenizer: Any,
    *,
    checkpoint_root: Path,
    step: int,
    trainer_state: dict[str, Any],
    max_shard_size: str,
    recovery_state: dict[str, Any] | None = None,
) -> Path:
    destination = checkpoint_root / f"step_{step}"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_root / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model = ddp_model.module
    state_dict: dict[str, torch.Tensor] | None = None
    try:
        state_dict = _bf16_cpu_state_dict(model)
        original_use_cache = bool(model.config.use_cache)
        model.config.use_cache = True
        model.save_pretrained(
            temporary,
            state_dict=state_dict,
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        model.config.use_cache = original_use_cache
        tokenizer.save_pretrained(temporary)
        if recovery_state is not None:
            torch.save(recovery_state, temporary / "training_recovery.pt")
        atomic_write_json(temporary / "trainer_state.json", trainer_state)
        if not checkpoint_complete(temporary):
            raise TrainingError(f"Incomplete HF checkpoint produced: {temporary}")
        temporary.replace(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        if state_dict is not None:
            state_dict.clear()
        gc.collect()
    return destination

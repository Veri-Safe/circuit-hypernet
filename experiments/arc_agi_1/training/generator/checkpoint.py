from __future__ import annotations

import json
import os
import random
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .contracts import CHECKPOINT_FORMAT


@dataclass(frozen=True)
class S4TrainerState:
    format: str
    global_step: int
    rows_consumed: int
    variant_id: str
    training_config_sha256: str
    base_manifest_sha256: str
    layout_sha256: str
    train_root_manifest_sha256: str
    train_join_sha256: str
    batch_manifest_sha256: str
    fresh_init_sha256: str
    attention_backend: str = "sdpa"
    missing_circuits: int = 0
    corrupt_circuits: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "S4TrainerState":
        return cls(
            **{
                key: value[key]
                for key in cls.__dataclass_fields__
                if key in value
            }
        )


def attention_backend_identity() -> str:
    backend = os.environ.get("NVARC_ATTENTION_IMPLEMENTATION", "sdpa")
    if backend not in {"sdpa", "flash_attention_2"}:
        raise ValueError(f"Unsupported S4 attention backend: {backend}")
    return backend


def unwrap_module(module: nn.Module) -> nn.Module:
    return module.module if hasattr(module, "module") else module


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["cuda_current_device"] = torch.cuda.get_rng_state()
    return value


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if "cuda_current_device" in value and torch.cuda.is_available():
        torch.cuda.set_rng_state(value["cuda_current_device"])


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_identity(actual: S4TrainerState, expected: S4TrainerState) -> None:
    ignored = {"global_step", "rows_consumed"}
    mismatches = {
        key: {"expected": getattr(expected, key), "actual": getattr(actual, key)}
        for key in S4TrainerState.__dataclass_fields__
        if key not in ignored and getattr(actual, key) != getattr(expected, key)
    }
    if mismatches:
        raise ValueError(f"S4 checkpoint identity mismatch: {mismatches}")


def save_checkpoint_core(
    *,
    output_root: str | Path,
    parameter_generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    trainer_state: S4TrainerState,
) -> Path:
    root = Path(output_root) / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{trainer_state.global_step:08d}"
    if destination.exists():
        if (destination / "COMPLETE").is_file():
            raise FileExistsError(f"Refusing to overwrite complete checkpoint: {destination}")
        preserved = root / f"{destination.name}.incomplete-{int(time.time())}"
        os.replace(destination, preserved)
    temporary = root / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        torch.save(
            unwrap_module(parameter_generator).state_dict(),
            temporary / "parameter_generator.pt",
        )
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        (temporary / "trainer_state.json").write_text(
            json.dumps(asdict(trainer_state), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def save_rank_rng(checkpoint: str | Path, *, rank: int) -> Path:
    destination = Path(checkpoint) / f"rng_state_rank{rank}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(_rng_state(), temporary)
    os.replace(temporary, destination)
    return destination


def finalize_checkpoint(
    checkpoint: str | Path,
    *,
    output_root: str | Path,
    world_size: int,
) -> None:
    path = Path(checkpoint)
    root = Path(output_root)
    if path.resolve().parent != (root / "checkpoints").resolve():
        raise ValueError(f"Checkpoint is outside the run's checkpoints directory: {path}")
    missing = [
        rank
        for rank in range(world_size)
        if not (path / f"rng_state_rank{rank}.pt").is_file()
    ]
    if missing:
        raise ValueError(f"Cannot finalize checkpoint; rank RNG states missing: {missing}")
    _write_json_atomic(
        path / "COMPLETE",
        {"status": "complete", "world_size": world_size},
    )
    _write_json_atomic(
        root / "latest.json",
        {"status": "complete", "checkpoint": f"checkpoints/{path.name}"},
    )


def load_checkpoint(
    checkpoint: str | Path,
    *,
    parameter_generator: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    expected_identity: S4TrainerState,
    rank: int | None,
) -> S4TrainerState:
    source = Path(checkpoint).resolve()
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"S4 checkpoint is not complete: {source}")
    value = json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    state = S4TrainerState.from_mapping(value)
    if state.format != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported S4 checkpoint format: {state.format}")
    _validate_identity(state, expected_identity)
    unwrap_module(parameter_generator).load_state_dict(
        torch.load(source / "parameter_generator.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    if optimizer is not None:
        optimizer.load_state_dict(
            torch.load(source / "optimizer.pt", map_location="cpu", weights_only=False)
        )
    if scheduler is not None:
        scheduler.load_state_dict(
            torch.load(source / "scheduler.pt", map_location="cpu", weights_only=False)
        )
    if rank is not None:
        rng_path = source / f"rng_state_rank{rank}.pt"
        if not rng_path.is_file():
            raise FileNotFoundError(f"Checkpoint rank RNG state is missing: {rng_path}")
        _restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return state


def load_optimizer_fork_checkpoint(
    checkpoint: str | Path,
    *,
    parameter_generator: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: S4TrainerState,
    expected_parent_training_config_sha256: str,
    expected_parent_step: int,
    rank: int,
) -> S4TrainerState:
    """Fork model, Adam moments, and RNG while allowing a declared new schedule."""

    source = Path(checkpoint).resolve()
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"S4 optimizer-fork checkpoint is not complete: {source}")
    state = S4TrainerState.from_mapping(
        json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    )
    if state.format != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported S4 checkpoint format: {state.format}")
    if state.global_step != expected_parent_step:
        raise ValueError(
            "S4 optimizer-fork source step changed: "
            f"{state.global_step} != {expected_parent_step}"
        )
    if state.training_config_sha256 != expected_parent_training_config_sha256:
        raise ValueError("S4 optimizer-fork parent training config identity changed")
    ignored = {"global_step", "rows_consumed", "training_config_sha256"}
    mismatches = {
        key: {
            "expected": getattr(expected_identity, key),
            "actual": getattr(state, key),
        }
        for key in S4TrainerState.__dataclass_fields__
        if key not in ignored and getattr(state, key) != getattr(expected_identity, key)
    }
    if mismatches:
        raise ValueError(f"S4 optimizer-fork checkpoint mismatch: {mismatches}")

    unwrap_module(parameter_generator).load_state_dict(
        torch.load(
            source / "parameter_generator.pt", map_location="cpu", weights_only=True
        ),
        strict=True,
    )
    parent_optimizer = torch.load(
        source / "optimizer.pt", map_location="cpu", weights_only=False
    )
    parent_groups = parent_optimizer.get("param_groups", [])
    parent_layout = [
        (str(group.get("name", "")), len(group.get("params", ())))
        for group in parent_groups
    ]
    target_layout = [
        (str(group.get("name", "")), len(group.get("params", ())))
        for group in optimizer.state_dict().get("param_groups", [])
    ]
    if parent_layout != target_layout:
        raise ValueError(
            "S4 optimizer-fork parameter-group layout changed: "
            f"parent={parent_layout} target={target_layout}"
        )
    optimizer.load_state_dict(parent_optimizer)
    rng_path = source / f"rng_state_rank{rank}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"Checkpoint rank RNG state is missing: {rng_path}")
    _restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    return state


def load_model_initialization(
    checkpoint: str | Path,
    *,
    parameter_generator: nn.Module,
    expected_identity: S4TrainerState,
) -> S4TrainerState:
    """Load model tensors from a compatible parent while resetting optimizer/RNG."""
    source = Path(checkpoint).resolve()
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"S4 initialization checkpoint is not complete: {source}")
    state = S4TrainerState.from_mapping(
        json.loads((source / "trainer_state.json").read_text(encoding="utf-8"))
    )
    for field in (
        "format",
        "variant_id",
        "base_manifest_sha256",
        "layout_sha256",
        "train_root_manifest_sha256",
        "train_join_sha256",
        "batch_manifest_sha256",
        "fresh_init_sha256",
        "attention_backend",
    ):
        if getattr(state, field) != getattr(expected_identity, field):
            raise ValueError(f"Warm-start checkpoint mismatch: {field}")
    unwrap_module(parameter_generator).load_state_dict(
        torch.load(source / "parameter_generator.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    return state


def resolve_latest(output_root: str | Path) -> Path | None:
    root = Path(output_root).resolve()
    latest = root / "latest.json"
    checkpoint_root = (root / "checkpoints").resolve()

    def complete_step(path: Path) -> int | None:
        if not path.is_dir() or not (path / "COMPLETE").is_file():
            return None
        state_path = path / "trainer_state.json"
        if not state_path.is_file():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            step = int(state["global_step"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if path.name != f"step_{step:08d}":
            return None
        return step

    discovered: list[tuple[int, Path]] = []
    if checkpoint_root.is_dir():
        for candidate in checkpoint_root.glob("step_*"):
            step = complete_step(candidate)
            if step is not None:
                discovered.append((step, candidate.resolve()))

    pointed: tuple[int, Path] | None = None
    pointer_error: Exception | None = None
    if latest.is_file():
        try:
            value = json.loads(latest.read_text(encoding="utf-8"))
            checkpoint = (root / str(value["checkpoint"])).resolve()
            if checkpoint.parent != checkpoint_root:
                raise ValueError(
                    f"Latest checkpoint pointer escaped its output root: {checkpoint}"
                )
            step = complete_step(checkpoint)
            if step is None:
                raise ValueError(f"Latest checkpoint pointer is incomplete: {checkpoint}")
            pointed = (step, checkpoint)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            pointer_error = error

    candidates = discovered + ([pointed] if pointed is not None else [])
    if not candidates:
        if pointer_error is not None:
            raise ValueError(f"Latest checkpoint pointer is invalid: {latest}") from pointer_error
        return None

    step, checkpoint = max(candidates, key=lambda value: value[0])
    if pointed != (step, checkpoint):
        _write_json_atomic(
            latest,
            {
                "status": "complete",
                "checkpoint": f"checkpoints/{checkpoint.name}",
                "recovered_by_scan": True,
                "global_step": step,
            },
        )
    return checkpoint

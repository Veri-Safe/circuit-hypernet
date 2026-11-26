from __future__ import annotations

import gc
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def emit(phase: str, **fields: Any) -> None:
    if phase == "support_complete" and torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    print(json.dumps({"phase": phase, "time": time.time(), **fields}, sort_keys=True), flush=True)


@contextmanager
def exclusive_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another TTFT worker owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def clone_anchor(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def reset_model(model: torch.nn.Module, anchor: dict) -> None:
    model.zero_grad(set_to_none=True)
    model.load_state_dict(anchor, strict=True)


def save_progress(path: Path, *, model, optimizer, scheduler, identity: str, step: int,
                  min_free_gib: float = 0) -> None:
    """One atomic rolling checkpoint per worker, not one multi-GB PG per puzzle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    value = {
        "format": "s4_ttft_resume_v1", "identity": identity, "step": step,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

    def tensor_bytes(item):
        if isinstance(item, torch.Tensor):
            return item.numel() * item.element_size()
        if isinstance(item, dict):
            return sum(tensor_bytes(child) for child in item.values())
        if isinstance(item, (tuple, list)):
            return sum(tensor_bytes(child) for child in item)
        return 0

    needed = int(tensor_bytes(value) * 1.05) + 1024**2 + int(min_free_gib * 1024**3)
    if shutil.disk_usage(path.parent).free < needed:
        raise OSError("Insufficient space for atomic TTFT snapshot; previous committed snapshot retained")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def restore_progress(path: Path, *, model, optimizer, scheduler, identity: str) -> int:
    if not path.is_file():
        return 0
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("format") != "s4_ttft_resume_v1":
        raise ValueError("Invalid TTFT resume format")
    if value["identity"] != identity:
        return 0  # A completed previous puzzle occupies the rolling slot.
    model.load_state_dict(value["model"], strict=True)
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    torch.set_rng_state(value["torch_rng"])
    if value["cuda_rng"]:
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    return int(value["step"])

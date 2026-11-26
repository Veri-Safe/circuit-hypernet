from __future__ import annotations

import json
import os
import random
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .model import ReLUTranscoder


# The first atomic-directory implementation used this package identity.  The
# migration below changes checkpoint transaction paths and the resume-manifest
# routing that recognizes that change; both are safe for already-complete
# checkpoints. Any other code change must still invalidate a resume.
CHECKPOINT_RESUME_LEGACY_CODE_IDENTITY_SHA256 = (
    "a13b3c07db9a16aa4ae7b6063b3d197adfefcca39966029b63f8de795b1c3c80"
)
_CHECKPOINT_RESUME_COMPATIBLE_CODE_PATHS = frozenset(
    {
        "src/s3circuit/nvarc_transcoder/checkpoint.py",
        "src/s3circuit/nvarc_transcoder/trainer.py",
    }
)


@dataclass(frozen=True)
class TrainingCursor:
    dataset_index: int
    eligible_offset: int
    block_index: int
    update_index: int

    def validate(self) -> None:
        if min(
            self.dataset_index,
            self.eligible_offset,
            self.block_index,
            self.update_index,
        ) < 0:
            raise ValueError(f"Negative training cursor: {self}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _rng_state() -> dict[str, Any]:
    value: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        value["torch_cuda"] = torch.cuda.get_rng_state_all()
    return value


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in value:
        torch.cuda.set_rng_state_all(value["torch_cuda"])


def checkpoint_code_identity_compatibility(
    actual: Mapping[str, Any] | None,
    expected: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Allow the audited atomic-checkpoint storage migration on resume."""

    if actual is not None and dict(actual) == dict(expected):
        return None
    if not isinstance(actual, Mapping) or not isinstance(expected, Mapping):
        raise ValueError("Checkpoint code identity is not a mapping")
    if actual.get("format") != expected.get("format"):
        raise ValueError("Checkpoint code identity format changed")
    actual_artifacts = actual.get("artifacts")
    expected_artifacts = expected.get("artifacts")
    if not isinstance(actual_artifacts, Mapping) or not isinstance(
        expected_artifacts, Mapping
    ):
        raise ValueError("Checkpoint code identity artifacts are missing")
    changed_paths = sorted(
        path
        for path in set(actual_artifacts) | set(expected_artifacts)
        if actual_artifacts.get(path) != expected_artifacts.get(path)
    )
    if actual.get("sha256") != CHECKPOINT_RESUME_LEGACY_CODE_IDENTITY_SHA256 or set(
        changed_paths
    ) != _CHECKPOINT_RESUME_COMPATIBLE_CODE_PATHS:
        raise ValueError("Checkpoint code identity changed outside the audited migration")
    return {
        "format": "nvarc_checkpoint_resume_code_compatibility_v1",
        "legacy_code_identity_sha256": actual["sha256"],
        "current_code_identity_sha256": expected.get("sha256"),
        "changed_artifacts": changed_paths,
        "reason": "atomic_checkpoint_transaction_path_migration",
    }


def _atomic_directory(destination: Path, writer: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The writer can be interrupted by SIGTERM after creating its temporary
    # directory.  A PID-only name then collides with a later retry (and may
    # even collide with a reused PID), making an otherwise resumable run fail
    # before it writes anything.  A unique directory also permits independent
    # destinations to be saved by workers that happen to share a process ID.
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    backup = destination.parent / f".{destination.name}.old-{uuid.uuid4().hex}"
    try:
        writer(temporary)
        if destination.exists():
            os.replace(destination, backup)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    else:
        # The new destination is already complete at this point.  Failure to
        # remove a superseded backup must not turn a successful checkpoint
        # commit into a failed training update; a later retry uses a new
        # transaction path and is unaffected by the leftover backup.
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


def save_resumable_checkpoint(
    destination: str | Path,
    *,
    transcoder: ReLUTranscoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cursor: TrainingCursor,
    manifest: Mapping[str, Any],
) -> None:
    from safetensors.torch import save_file

    cursor.validate()
    path = Path(destination).resolve()

    def write(root: Path) -> None:
        save_file(
            {key: value.detach().float().cpu().contiguous() for key, value in transcoder.state_dict().items()},
            str(root / "transcoder.safetensors"),
        )
        torch.save(optimizer.state_dict(), root / "optimizer.pt")
        torch.save(scheduler.state_dict(), root / "scheduler.pt")
        torch.save(_rng_state(), root / "rng.pt")
        _write_json(
            root / "state.json",
            {"format": "nvarc_transcoder_resume_v1", "cursor": asdict(cursor), **dict(manifest)},
        )
        (root / "COMPLETE").write_text("ok\n", encoding="ascii")

    _atomic_directory(path, write)


def load_resumable_checkpoint(
    source: str | Path,
    *,
    transcoder: ReLUTranscoder,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_manifest: Mapping[str, Any],
) -> TrainingCursor:
    from safetensors.torch import load_file

    path = Path(source).resolve()
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Checkpoint is incomplete: {path}")
    state = json.loads((path / "state.json").read_text(encoding="utf-8"))
    if state.get("format") != "nvarc_transcoder_resume_v1":
        raise ValueError(f"Unsupported resumable checkpoint: {path}")
    mismatches: dict[str, Any] = {}
    for key, expected in expected_manifest.items():
        actual = state.get(key)
        if key == "code_identity":
            try:
                checkpoint_code_identity_compatibility(actual, expected)
            except ValueError:
                mismatches[key] = {"expected": expected, "actual": actual}
            continue
        if actual != expected:
            mismatches[key] = {"expected": expected, "actual": actual}
    if mismatches:
        raise ValueError(f"Resume identity mismatch: {mismatches}")
    transcoder.load_state_dict(load_file(str(path / "transcoder.safetensors"), device="cpu"))
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=False))
    _restore_rng_state(torch.load(path / "rng.pt", map_location="cpu", weights_only=False))
    cursor = TrainingCursor(**{key: int(value) for key, value in state["cursor"].items()})
    cursor.validate()
    return cursor


def save_weight_snapshot(
    destination: str | Path,
    *,
    transcoder: ReLUTranscoder,
    manifest: Mapping[str, Any],
) -> None:
    from safetensors.torch import save_file

    path = Path(destination).resolve()

    def write(root: Path) -> None:
        save_file(
            {key: value.detach().float().cpu().contiguous() for key, value in transcoder.state_dict().items()},
            str(root / "transcoder.safetensors"),
        )
        _write_json(root / "manifest.json", dict(manifest))
        (root / "COMPLETE").write_text("ok\n", encoding="ascii")

    _atomic_directory(path, write)


def load_weight_snapshot(source: str | Path) -> tuple[ReLUTranscoder, dict[str, Any]]:
    from safetensors.torch import load_file

    path = Path(source).resolve()
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"Weight snapshot is incomplete: {path}")
    tensors = load_file(str(path / "transcoder.safetensors"), device="cpu")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    return ReLUTranscoder.from_state_dict(tensors), manifest

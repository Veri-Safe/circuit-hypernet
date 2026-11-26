"""Frozen S5 checkpoint and tokenizer loading for circuit work."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

from ..modeling import build_model


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_tokenizer(
    path: str | Path, *, expected_sha256: str | None = None
) -> spm.SentencePieceProcessor:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"tokenizer is missing: {source}")
    if expected_sha256 is not None and sha256_file(source) != str(expected_sha256):
        raise ValueError("S5 circuit tokenizer SHA256 mismatch")
    tokenizer = spm.SentencePieceProcessor(model_file=str(source))
    if int(tokenizer.vocab_size()) != 2048:
        raise ValueError("S5 circuit tokenizer must have vocabulary size 2048")
    return tokenizer


def load_frozen_source_model(
    source_config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    expected_checkpoint_config_sha256: str | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_architecture_config_sha256: str | None = None,
    verified_checkpoint_sha256: str | None = None,
    canonical_checkpoint_path: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_source = Path(checkpoint_path).resolve()
    if not checkpoint_source.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint_source}")
    canonical_checkpoint_source = Path(
        canonical_checkpoint_path or checkpoint_source
    ).resolve()
    # Prefer the canonical checkpoint when a launcher supplies a staged envelope.
    if canonical_checkpoint_path is not None and canonical_checkpoint_source != checkpoint_source:
        checkpoint_source = canonical_checkpoint_source
    verified_sha256 = (
        str(verified_checkpoint_sha256)
        if verified_checkpoint_sha256 is not None
        else None
    )
    measured_checkpoint_sha256 = (
        None if verified_sha256 is not None else sha256_file(checkpoint_source)
    )
    if (
        expected_architecture_config_sha256 is not None
        and source_config.get("_config_sha256")
        != str(expected_architecture_config_sha256)
    ):
        raise ValueError("source architecture-config file SHA256 mismatch")
    model = build_model(source_config["model"])
    # mmap keeps the serialized checkpoint backed by the shared page cache
    # while each worker copies only the tensors it needs onto its GPU.
    checkpoint = torch.load(
        checkpoint_source,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    actual_checkpoint_sha256 = verified_sha256 or str(measured_checkpoint_sha256)
    if (
        expected_checkpoint_sha256 is not None
        and actual_checkpoint_sha256 != str(expected_checkpoint_sha256)
    ):
        raise ValueError("source checkpoint file SHA256 mismatch")
    if int(checkpoint.get("step", -1)) != 38_000:
        raise ValueError("circuit source checkpoint must be step 38000")
    expected_config_sha = (
        str(expected_checkpoint_config_sha256)
        if expected_checkpoint_config_sha256 is not None
        else str(source_config.get("_config_sha256"))
    )
    if checkpoint.get("config_sha256") != expected_config_sha:
        raise ValueError("source checkpoint's frozen training-config SHA256 mismatch")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.config.grad_checkpointing = False
    model.to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    identity = {
        "checkpoint_path": str(canonical_checkpoint_source),
        "checkpoint_sha256": actual_checkpoint_sha256,
        "checkpoint_step": 38_000,
        "checkpoint_config_sha256": expected_config_sha,
        "current_architecture_config_sha256": source_config["_config_sha256"],
        "dtype": str(dtype),
        "device_type": device.type,
    }
    return model, identity

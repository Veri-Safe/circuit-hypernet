"""Resolve a direct-SFT TTFT base without an S2 training-profile identity."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvarc_qwen3.arc_eval_protocol import (
    ARC_AGI1_PUBLIC_VALIDATION_CONFIG_SHA256,
    ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION,
    canonical_json_bytes,
    sha256_file,
)
from s2_hypernet.nvarc_pg.target_layout import (
    QwenTargetLayout,
    build_qwen3_4b_target_layout,
)


@dataclass(frozen=True)
class DirectBase:
    hf_checkpoint: str
    tokenizer: str
    source_step: int
    source_kind: str = "direct_sft_hf"


@dataclass(frozen=True)
class DirectEvaluation:
    protocol_config: str
    challenges: str
    solutions: str
    protocol_kind: str = "arc_agi1_public_validation"


@dataclass(frozen=True)
class DirectRuntime:
    """The attributes used by tokenizer loading and frozen-base loading only."""

    base: DirectBase
    evaluation: DirectEvaluation
    target_layout: QwenTargetLayout
    checkpoint_identity: dict[str, Any]


def _workspace_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.anchor and not path.is_relative_to(root) and "HyperCombine" in path.parts:
        # Older external manifests include the checkout name after the mount.
        path = root.joinpath(*path.parts[path.parts.index("HyperCombine") + 1:])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Direct TTFT path is outside HC_ROOT: {path}")
    return path


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {label} SHA256")
    return value


def _hash_hf_model_artifacts(model_dir: Path) -> dict[str, Any]:
    # This existing evaluator still uses sibling absolute imports. Restrict the
    # temporary import path to its package, then restore it before hashing.
    import nvarc_qwen3

    package_dir = str(Path(next(iter(nvarc_qwen3.__path__))).resolve())
    sys.path.insert(0, package_dir)
    try:
        from nvarc_qwen3.eval_nvarc_internal import hash_hf_model_artifacts
    finally:
        sys.path.remove(package_dir)
    return hash_hf_model_artifacts(model_dir)


def load_direct_runtime(
    root: Path, cfg: dict, *, verify_weights: bool = False
) -> DirectRuntime:
    """Bind step65 to its pinned historical evaluation, optionally hashing weights.

    The returned identity is identical for cheap reads and verified reads. A
    caller must request weight verification before creating or resuming a run.
    This narrow config is intended for ``load_frozen_qwen(for_training=False)``;
    it deliberately has no S2 profile, parameter generator, or circuit config.
    """

    root = root.resolve()
    if cfg.get("format") != "s4_nvarc_sft_ttft_v3":
        raise ValueError("Direct TTFT runtime requires s4_nvarc_sft_ttft_v3")
    if type(cfg.get("source_step")) is not int or cfg["source_step"] != 65:
        raise ValueError("Direct TTFT runtime requires source_step=65")
    manifest_path = _workspace_path(root, cfg["base_identity_manifest"])
    manifest_sha = _sha(cfg["base_identity_manifest_sha256"], "identity manifest")
    if sha256_file(manifest_path) != manifest_sha:
        raise ValueError("Direct TTFT identity manifest SHA256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_source") != "hf" or manifest.get("adapter_path") is not None:
        raise ValueError("Direct TTFT identity must describe an HF base without an adapter")
    label = cfg["model_label"]
    if not isinstance(label, str) or not label or label != manifest.get("model_label"):
        raise ValueError("Direct TTFT model_label differs from identity manifest")

    model_dir = _workspace_path(root, cfg["base_checkpoint"])
    tokenizer_dir = _workspace_path(root, cfg["tokenizer"])
    if model_dir != _workspace_path(root, manifest["model_path"]):
        raise ValueError("Direct TTFT checkpoint path differs from identity manifest")
    if model_dir.name != "step_65":
        raise ValueError("Direct TTFT checkpoint directory must be step_65")
    if tokenizer_dir != _workspace_path(root, manifest["tokenizer_path"]):
        raise ValueError("Direct TTFT tokenizer path differs from identity manifest")
    if manifest.get("protocol_version") != ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION:
        raise ValueError("Direct TTFT identity requires the ARC-AGI-1 public validation protocol")

    evaluation_paths = {}
    for name, manifest_key in (
        ("protocol_config", "config_path"),
        ("challenges", "challenge_path"),
        ("solutions", "solution_path"),
    ):
        path = _workspace_path(root, cfg["evaluation"][name])
        if path != _workspace_path(root, manifest[manifest_key]):
            raise ValueError(f"Direct TTFT {name} path differs from identity manifest")
        if not path.is_file():
            raise FileNotFoundError(path)
        evaluation_paths[name] = str(path)
    protocol_sha = sha256_file(evaluation_paths["protocol_config"])
    if protocol_sha != manifest["config_sha256"] or protocol_sha != ARC_AGI1_PUBLIC_VALIDATION_CONFIG_SHA256:
        raise ValueError("Direct TTFT evaluation protocol SHA256 mismatch")
    tokenizer_sha = sha256_file(tokenizer_dir / "tokenizer_config.json")
    if tokenizer_sha != manifest["tokenizer_config_sha256"]:
        raise ValueError("Direct TTFT tokenizer config SHA256 mismatch")

    artifacts = manifest["model_artifacts"]
    if _workspace_path(root, artifacts["root"]) != model_dir:
        raise ValueError("Direct TTFT model artifact root differs from checkpoint path")
    aggregate = _sha(cfg["expected_model_aggregate_sha256"], "model aggregate")
    files = artifacts["files"]
    if not isinstance(files, dict) or not files or "config.json" not in files:
        raise ValueError("Direct TTFT identity has no model artifact files")
    for name, metadata in files.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Direct TTFT model artifacts must use plain filenames")
        _sha(metadata["sha256"], f"model artifact {name}")
        if type(metadata["size_bytes"]) is not int or metadata["size_bytes"] < 0:
            raise ValueError(f"Invalid model artifact size: {name}")
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != metadata["size_bytes"]:
            raise ValueError(f"Direct TTFT model artifact missing or size changed: {name}")
    if (
        artifacts["aggregate_sha256"] != aggregate
        or hashlib.sha256(canonical_json_bytes({"files": files})).hexdigest() != aggregate
    ):
        raise ValueError("Direct TTFT model aggregate differs from identity manifest")
    if verify_weights:
        actual = _hash_hf_model_artifacts(model_dir)
        if actual["files"] != files or actual["aggregate_sha256"] != aggregate:
            raise ValueError("Direct TTFT model weights differ from identity manifest")

    identity = {
        "format": "sft_ttft_direct_checkpoint_identity_v1",
        "base_identity_manifest": str(manifest_path.relative_to(root)),
        "base_identity_manifest_sha256": manifest_sha,
        "base_checkpoint": str(model_dir.relative_to(root)),
        "tokenizer": str(tokenizer_dir.relative_to(root)),
        "model_label": label,
        "source_step": 65,
        "source_kind": "direct_sft_hf",
        "model_aggregate_sha256": aggregate,
        "model_artifact_files": files,
        "tokenizer_config_sha256": tokenizer_sha,
        "protocol_version": ARC_AGI1_PUBLIC_VALIDATION_PROTOCOL_VERSION,
        "protocol_config_sha256": protocol_sha,
    }
    return DirectRuntime(
        base=DirectBase(str(model_dir), str(tokenizer_dir), 65),
        evaluation=DirectEvaluation(**evaluation_paths),
        target_layout=build_qwen3_4b_target_layout(
            rank=int(cfg["lora_rank"]), alpha=int(cfg["lora_alpha"]),
            token_dim=2560, dim_accumulation=1,
        ),
        checkpoint_identity=identity,
    )

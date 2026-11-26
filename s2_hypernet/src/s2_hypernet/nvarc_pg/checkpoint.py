from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .profiles import ExperimentProfile, K8_PROFILE
from .target_layout import QwenTargetLayout


MODEL_CONTRACT = {
    "model_type": "qwen3",
    "hidden_size": 2560,
    "intermediate_size": 9728,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 16,
    "tie_word_embeddings": True,
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_manifest(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.name)
    }


def _semantic_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_pinned_hf_artifacts(
    model_dir: Path,
    *,
    profile: ExperimentProfile,
) -> str | None:
    if not profile.hf_artifact_sha256:
        return None
    artifacts: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in sorted(profile.hf_artifact_sha256):
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Pinned HF artifact is missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Pinned HF artifact hash mismatch for {name}: "
                f"{actual_sha256} != {expected_sha256}"
            )
        artifacts[name] = {
            "sha256": actual_sha256,
            "size_bytes": path.stat().st_size,
        }
    aggregate = _semantic_sha256({"files": artifacts})
    if aggregate != profile.hf_aggregate_sha256:
        raise ValueError(
            "Pinned HF aggregate hash mismatch: "
            f"{aggregate} != {profile.hf_aggregate_sha256}"
        )
    return aggregate


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_source_run_manifest(
    path: str | Path,
    *,
    profile: ExperimentProfile,
) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Source FullFT run manifest is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    expected = {
        "run_name": profile.source_run_name,
        "run_kind": profile.source_run_kind,
        "status": "complete",
        "exit_status": 0,
        "max_steps": profile.expected_source_run_max_steps,
        "max_epochs": profile.expected_source_run_max_epochs,
        "train_splits": [split.name for split in profile.train_splits],
        "expected_train_rows": profile.train_rows,
        "expected_data_batches_sha256": profile.source_data_batches_sha256,
        "expected_data_permutation_sha256": profile.source_data_permutation_sha256,
    }
    observed = dict(manifest)
    if profile.source_run_name and profile.source_run_name.startswith("sha256:"):
        observed["run_name"] = "sha256:" + hashlib.sha256(str(manifest.get("run_name", "")).encode()).hexdigest()
    mismatches = {
        key: {"expected": expected_value, "actual": observed.get(key)}
        for key, expected_value in expected.items()
        if observed.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Source FullFT run is not the completed profile base: {mismatches}")
    return manifest, sha256_file(manifest_path)


def inspect_frozen_hf_checkpoint(
    *,
    hf_checkpoint: str | Path,
    tokenizer: str | Path,
    source_run_manifest: str | Path | None,
    profile: ExperimentProfile,
    layout: QwenTargetLayout,
) -> dict[str, Any]:
    """Validate a frozen 16-token Qwen checkpoint directly from HF artifacts."""

    model_dir = Path(hf_checkpoint).resolve()
    tokenizer_dir = Path(tokenizer).resolve()
    expected_step_name = f"step_{profile.source_step}"
    if (
        profile.source_run_name is not None
        and model_dir.name != f"{expected_step_name}-hf"
    ):
        raise ValueError(f"HF checkpoint directory must be {expected_step_name}-hf")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"HF checkpoint directory does not exist: {model_dir}")
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_dir}")

    model_config_path = model_dir / "config.json"
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    token_mapping_path = tokenizer_dir / "nvarc_16t_token_mapping.json"
    for required in (model_config_path, tokenizer_config_path, token_mapping_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required checkpoint artifact is missing: {required}")

    model_config = _read_json(model_config_path)
    if "Qwen3ForCausalLM" not in model_config.get("architectures", []):
        raise ValueError("HF checkpoint is not Qwen3ForCausalLM")
    mismatches = {
        key: {"expected": expected, "actual": model_config.get(key)}
        for key, expected in MODEL_CONTRACT.items()
        if model_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"HF checkpoint model contract mismatch: {mismatches}")

    weight_paths = sorted(model_dir.glob("*.safetensors")) + sorted(
        model_dir.glob("pytorch_model*.bin")
    )
    index_paths = [
        path
        for path in (
            model_dir / "model.safetensors.index.json",
            model_dir / "pytorch_model.bin.index.json",
        )
        if path.is_file()
    ]
    if not weight_paths:
        raise FileNotFoundError(f"HF checkpoint has no model weights: {model_dir}")

    pinned_hf_aggregate_sha256 = _validate_pinned_hf_artifacts(
        model_dir,
        profile=profile,
    )
    tokenizer_config_sha256 = sha256_file(tokenizer_config_path)
    token_mapping_sha256 = sha256_file(token_mapping_path)
    if (
        profile.tokenizer_config_sha256 is not None
        and tokenizer_config_sha256 != profile.tokenizer_config_sha256
    ):
        raise ValueError("Pinned tokenizer_config.json hash mismatch")
    if (
        profile.token_mapping_sha256 is not None
        and token_mapping_sha256 != profile.token_mapping_sha256
    ):
        raise ValueError("Pinned 16-token mapping hash mismatch")

    source_manifest_sha256 = None
    source_manifest = None
    if profile.source_run_name is not None:
        if source_run_manifest is None or not str(source_run_manifest):
            raise ValueError(f"{profile.profile_id} requires a source FullFT run manifest")
        source_manifest, source_manifest_sha256 = _validate_source_run_manifest(
            source_run_manifest,
            profile=profile,
        )

    manifest = {
        "format": "nvarc_pg_base_checkpoint_v3",
        "profile_id": profile.profile_id,
        "runtime": "hf_only",
        "source_kind": profile.source_kind,
        "source_step": profile.source_step,
        "source_run_max_steps": profile.expected_source_run_max_steps,
        "source_run_manifest": (
            str(Path(source_run_manifest).resolve()) if source_run_manifest else None
        ),
        "source_run_manifest_sha256": source_manifest_sha256,
        "source_run_name": source_manifest.get("run_name") if source_manifest else None,
        "hf_checkpoint": str(model_dir),
        "tokenizer": str(tokenizer_dir),
        "model_contract": dict(MODEL_CONTRACT),
        "model_config_sha256": sha256_file(model_config_path),
        "tokenizer_config_sha256": tokenizer_config_sha256,
        "token_mapping_sha256": token_mapping_sha256,
        "hf_artifacts": _artifact_manifest(weight_paths + index_paths),
        "target_layout_sha256": layout.layout_sha256,
    }
    if pinned_hf_aggregate_sha256 is not None:
        manifest["hf_aggregate_sha256"] = pinned_hf_aggregate_sha256
    return manifest


def inspect_step1400_hf_checkpoint(
    *,
    hf_checkpoint: str | Path,
    tokenizer: str | Path,
    layout: QwenTargetLayout,
) -> dict[str, Any]:
    """Backward-compatible K8 checkpoint inspection surface."""

    manifest = inspect_frozen_hf_checkpoint(
        hf_checkpoint=hf_checkpoint,
        tokenizer=tokenizer,
        source_run_manifest=None,
        profile=K8_PROFILE,
        layout=layout,
    )
    manifest["format"] = "nvarc_pg_k8_base_checkpoint_v1"
    return manifest


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)

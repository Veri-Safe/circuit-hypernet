from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .config import S4R1Config
from .contracts import FRESH_INIT_FORMAT, FUSION_VARIANTS, semantic_sha256, sha256_file
from .pg_core import S4CircuitParameterGenerator


def tensor_sha256(value: Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def state_dict_identity(state: Mapping[str, Tensor]) -> dict[str, Any]:
    tensors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": tensor_sha256(value),
        }
        for key, value in sorted(state.items())
    }
    return {
        "tensors": tensors,
        "aggregate_sha256": semantic_sha256(tensors),
    }


def _same_pg_config_identity(recorded: Any, current: Any) -> bool:
    """Compare PG configs using the JSON identity used by the manifest."""

    return semantic_sha256(recorded) == semantic_sha256(current)


def prepare_fresh_init(
    config: S4R1Config,
    *,
    source_commit: str,
) -> dict[str, Any]:
    root = Path(config.fresh_init_root)
    snapshot_path = root / "common_pg_trunk.pt"
    manifest_path = root / "manifest.json"
    if snapshot_path.exists() or manifest_path.exists():
        if not snapshot_path.is_file() or not manifest_path.is_file():
            raise ValueError(f"Fresh-init artifact is incomplete: {root}")
        return validate_fresh_init(root, config=config)

    from s2_hypernet.nvarc_pg.config import load_experiment_config

    s2_config = load_experiment_config(config.s2_config)
    torch.manual_seed(config.seed)
    model = S4CircuitParameterGenerator(
        s2_config.build_parameter_generator_config(),
        variant_id="f0_single_token",
        encoder_hidden_dim=config.circuit_encoder_hidden_dim,
    )
    state = model.common_trunk_state_dict()
    identity = state_dict_identity(state)
    root.mkdir(parents=True, exist_ok=True)
    temporary = snapshot_path.with_suffix(".pt.tmp")
    torch.save(OrderedDict(state), temporary)
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, snapshot_path)
    manifest = {
        "format": FRESH_INIT_FORMAT,
        "status": "complete",
        "seed": config.seed,
        "source_commit": source_commit,
        "s2_config": config.s2_config,
        "s2_profile": s2_config.profile.profile_id,
        "base_aggregate_sha256": config.base_aggregate_sha256,
        "pg_config": s2_config.build_parameter_generator_config().to_dict(),
        "snapshot_path": snapshot_path.name,
        "snapshot_sha256": sha256_file(snapshot_path),
        **identity,
        **(
            {"replication": config.semantic_dict["replication"]}
            if config.replication else {}
        ),
    }
    manifest["manifest_semantic_sha256"] = semantic_sha256(manifest)
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return validate_fresh_init(root, config=config)


def validate_fresh_init(
    root: str | Path, *, config: S4R1Config | None = None
) -> dict[str, Any]:
    base = Path(root)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FRESH_INIT_FORMAT or manifest.get("status") != "complete":
        raise ValueError(f"Fresh-init manifest is not complete: {manifest_path}")
    recorded_semantic = manifest.pop("manifest_semantic_sha256", None)
    actual_semantic = semantic_sha256(manifest)
    manifest["manifest_semantic_sha256"] = recorded_semantic
    if recorded_semantic != actual_semantic:
        raise ValueError("Fresh-init manifest identity changed")
    snapshot_path = base / str(manifest["snapshot_path"])
    if sha256_file(snapshot_path) != manifest.get("snapshot_sha256"):
        raise ValueError("Fresh-init snapshot checksum changed")
    state = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    actual_identity = state_dict_identity(state)
    if actual_identity["aggregate_sha256"] != manifest.get("aggregate_sha256"):
        raise ValueError("Fresh-init common trunk identity changed")
    if config is not None:
        expected = {
            "seed": config.seed,
            "base_aggregate_sha256": config.base_aggregate_sha256,
            "s2_config": config.s2_config,
            "replication": config.semantic_dict.get("replication"),
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Fresh-init experiment identity changed: {mismatches}")
    return manifest


def load_fresh_model(
    config: S4R1Config,
    *,
    device: torch.device | None = None,
    pg_dtype: torch.dtype = torch.bfloat16,
) -> tuple[S4CircuitParameterGenerator, dict[str, Any]]:
    from s2_hypernet.nvarc_pg.config import load_experiment_config

    manifest = validate_fresh_init(config.fresh_init_root, config=config)
    expected_identity = {
        "seed": config.seed,
        "base_aggregate_sha256": config.base_aggregate_sha256,
        "s2_profile": "arc1_rearc8_s42_step130",
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected_identity.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Fresh-init experiment identity changed: {mismatches}")
    variant_seed = config.seed + 1_000 + list(FUSION_VARIANTS).index(config.variant_id)
    torch.manual_seed(config.seed)
    s2_config = load_experiment_config(config.s2_config)
    current_pg_config = s2_config.build_parameter_generator_config()
    if not _same_pg_config_identity(manifest.get("pg_config"), current_pg_config.to_dict()):
        raise ValueError("Fresh-init Parameter Generator configuration changed")
    model = S4CircuitParameterGenerator(
        current_pg_config,
        variant_id=config.variant_id,
        encoder_hidden_dim=config.circuit_encoder_hidden_dim,
        fusion_seed=variant_seed,
        trainable_global_lora_scale=(
            config.optimization.trainable_global_lora_scale
        ),
        global_lora_scale_initial=config.optimization.global_lora_scale_initial,
        global_lora_scale_max=config.optimization.global_lora_scale_max,
        circuit_input_mode=config.circuit_input_mode,
    )
    snapshot_path = Path(config.fresh_init_root) / str(manifest["snapshot_path"])
    state = torch.load(snapshot_path, map_location="cpu", weights_only=True)
    model.load_common_trunk_state_dict(state)
    if state_dict_identity(model.common_trunk_state_dict())["aggregate_sha256"] != manifest[
        "aggregate_sha256"
    ]:
        raise ValueError("Variant did not load the exact common PG trunk")
    if device is not None:
        model.prepare_runtime(device=device, pg_dtype=pg_dtype)
    return model, manifest

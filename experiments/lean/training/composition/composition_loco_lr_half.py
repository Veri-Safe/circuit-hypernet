"""Immutable preflight and optimizer contracts for the approved LR-half wave."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .composition_loco import COMPOSITION_TASKS, semantic_sha256, sha256_file
from .composition_loco_config import (
    LR_HALF_PROTOCOL_ID,
    LR_HALF_OUTPUT_ROOT,
    RECREATED_INPUTS_PROTOCOL_ID,
    RECREATED_INPUTS_OUTPUT_ROOT,
    ORIGINAL_LR_RECREATED_PROTOCOL_ID,
    load_config,
)
from .composition_loco_ood_audit import require_ood_audit

CONFIG_ROOT = "s5_lean_circuit/configs/composition_loco_lr_half_v1"
AUDIT_PATH = "logs/data/2026-09-07/s5-composition-loco-ood-validation/audit.json"
PEAK_LRS = {
    "direct_sft": {"0": 7.5e-6},
    "query_only": {"pg": 3.75e-5, "query": 1.875e-5},
    "raw_weight": {"pg": 1.875e-5, "query": 9.375e-6, "raw_encoder": 1e-4, "raw_output": 2.5e-4},
    "circuit_a": {"pg": 1.25e-5, "query": 6.25e-6, "circuit": 2.5e-5},
}
ORIGINAL_LR_PEAK_LRS = {
    "raw_weight": {"pg": 3.75e-5, "query": 1.875e-5, "raw_encoder": 2e-4, "raw_output": 5e-4},
}
ORIGINAL_LR_RECREATED_RUNNER = "4server/s5_lean_circuit/run_composition_loco_original_lr_recreated.sh"
CHECKPOINT_BYTES = {
    "direct_sft": (1659643336, 4978973151),
    "query_only": (1123450625, 3370418451),
    "raw_weight": (1146568801, 3439779923),
    "circuit_a": (1135105085, 3405391126),
}


def validate_optimizer_peak_lrs(config: Any, groups: Sequence[Mapping[str, Any]]) -> None:
    if config.protocol_id == ORIGINAL_LR_RECREATED_PROTOCOL_ID:
        expected = ORIGINAL_LR_PEAK_LRS.get(config.route)
        label = "original-LR"
    elif config.protocol_id in {LR_HALF_PROTOCOL_ID, RECREATED_INPUTS_PROTOCOL_ID}:
        expected = PEAK_LRS.get(config.route)
        label = "all-LR x0.5"
    else:
        raise ValueError("optimizer recovery-wave check used for another protocol")
    if expected is None:
        raise ValueError(f"optimizer recovery-wave route is unsupported: {config.route}")
    actual = {str(group.get("name", i)): float(group["lr"]) for i, group in enumerate(groups)}
    if len(actual) != len(groups) or actual != expected:
        raise ValueError(f"optimizer groups do not implement {label}: {actual}")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def _path(root: Path, value: str | Path) -> Path:
    value = Path(value)
    result = (value if value.is_absolute() else root / value).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"LR-half source/output must stay inside repository: {result}")
    return result


def _file(root: Path, path: Path, *, expected_sha: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing/empty prerequisite: {path}")
    digest = sha256_file(path)
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"prerequisite SHA changed: {path}")
    return {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest}


def training_preflight(root: str | Path, config_path: str | Path, *,
                       audit_path: str | Path = AUDIT_PATH, require_fresh: bool = True) -> dict[str, Any]:
    """No model inference; the trainer still verifies tensor contents on load."""
    root = Path(root).resolve()
    config_file = _path(root, config_path)
    config = load_config(config_file)
    allowed = {LR_HALF_PROTOCOL_ID, RECREATED_INPUTS_PROTOCOL_ID, ORIGINAL_LR_RECREATED_PROTOCOL_ID}
    if config.protocol_id not in allowed:
        raise ValueError("recovery-wave entrypoint refuses original v1 or another experiment")
    if os.environ.get("S5_LOCO_RESUME") or os.environ.get("S5_LOCO_RAW_CACHE_REPAIR_MANIFEST"):
        raise ValueError("fresh recovery wave must not use an old training resume/cache-repair exception")
    output = _path(root, config.output_dir)
    if require_fresh and output.exists() and any(output.iterdir()):
        raise ValueError(f"refusing nonempty recovery-wave training output; no automatic resume: {output}")
    bank = _path(root, config.data["validation_bank"])
    audit = _path(root, audit_path)
    require_ood_audit(audit, config, bank)
    files = {"config": _file(root, config_file), "audit": _file(root, audit)}
    for label, path, expected in (
        ("initial_checkpoint", config.initialization["checkpoint_path"], config.initialization["checkpoint_sha256"]),
        ("base_checkpoint", config.base["checkpoint_path"], config.base["checkpoint_sha256"]),
        ("base_config", config.base["architecture_config_path"], None),
        ("tokenizer", config.data["tokenizer_path"], None),
    ):
        files[label] = _file(root, _path(root, path), expected_sha=expected)
    if _json(_path(root, config.base["architecture_config_path"])).get("config_sha256") != config.base["config_sha256"]:
        raise ValueError("Base architecture semantic config identity changed")
    cache_metadata = _json(_path(root, config.data["cache_manifest"]))
    large_cache = _path(root, config.data["cache_path"])
    if not large_cache.is_file() or large_cache.stat().st_size != cache_metadata["payload_bytes"]:
        raise ValueError("shared 1.5M tensor cache is missing or its size changed")
    # Large payload is hashed by the trainer's existing strict cache loader.
    files["training_cache"] = {
        "path": large_cache.relative_to(root).as_posix(), "bytes": large_cache.stat().st_size,
        "mtime_ns": large_cache.stat().st_mtime_ns, "expected_sha256": cache_metadata["payload_sha256"],
    }
    for key in ("cache_manifest", "fold_index", "validation_bank"):
        path = _path(root, config.data[key])
        files[key] = _file(root, path)
        if key != "cache_manifest":
            files[key + "_metadata"] = _file(root, Path(str(path) + ".json"))
    bank_metadata = _json(Path(str(bank) + ".json"))
    if bank_metadata.get("payload_sha256") != files["validation_bank"]["sha256"]:
        raise ValueError("validation tensor bank differs from its manifest")
    files["validation_records"] = _file(root, Path(str(bank) + ".records.jsonl"))
    fold = _json(_path(root, config.data["fold_index"] + ".json"))
    if (fold.get("heldout_task") != config.heldout_task or fold.get("train_tasks") != list(config.train_tasks)
            or fold.get("event_count") != 1200000 or fold.get("cache_identity_sha256") != cache_metadata["identity_sha256"]):
        raise ValueError("training fold/data identity changed")
    if config.route != "direct_sft":
        files["base_ce"] = _file(root, _path(root, config.data["base_ce_path"]))
        files["base_ce_metadata"] = _file(root, _path(root, config.data["base_ce_path"] + ".json"))
        ce = _json(_path(root, config.data["base_ce_path"] + ".json"))
        if (ce.get("status") != "complete" or ce.get("event_count") != 1500000
                or ce.get("cache_identity_sha256") != cache_metadata["identity_sha256"]
                or ce.get("cache_payload_sha256") != cache_metadata["payload_sha256"]
                or ce.get("payload_sha256") != files["base_ce"]["sha256"]):
            raise ValueError("Base CE cache is not bound to the frozen composition corpus")
    raw = None
    if config.route == "raw_weight":
        if config.protocol_id in {RECREATED_INPUTS_PROTOCOL_ID, ORIGINAL_LR_RECREATED_PROTOCOL_ID}:
            from .composition_loco_ood_raw_cache import check_recreated_raw_cache
            raw = check_recreated_raw_cache(config, root)
        else:
            from .composition_loco_ood_raw_cache import check_raw_cache_for_ood
            raw = check_raw_cache_for_ood(config, root)
    sources = []
    for folder in ("s5_lean_circuit/src", "s5_lean_circuit/scripts", "s2_hypernet/src"):
        sources.extend((root / folder).rglob("*.py"))
    runner = (
        ORIGINAL_LR_RECREATED_RUNNER
        if config.protocol_id == ORIGINAL_LR_RECREATED_PROTOCOL_ID
        else "4server/s5_lean_circuit/run_composition_loco_lr_half.sh"
    )
    sources += [root / runner]
    expected_lrs = (
        ORIGINAL_LR_PEAK_LRS[config.route]
        if config.protocol_id == ORIGINAL_LR_RECREATED_PROTOCOL_ID
        else PEAK_LRS[config.route]
    )
    result = {
        "format": "s5_composition_loco_lr_half_preflight_v1", "protocol_id": config.protocol_id,
        "config_sha256": config.computed_sha256, "route": config.route, "fold_id": config.fold_id,
        "output": config.output_dir, "optimizer_peak_lrs": expected_lrs,
        "files": files, "raw_cache_identity": raw,
        "source_files_sha256": {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(set(sources))},
        "fresh_atomic_initialization": True, "selection_uses_ood": False,
    }
    result["identity_sha256"] = semantic_sha256(result)
    return result


def require_training_receipt(config: Any, *, root: Path, audit_path: str | Path,
                             expected_identity: str, require_fresh: bool = True,
                             config_path: str | Path | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = f"{CONFIG_ROOT}/{config.heldout_task}/{config.route}.json"
    if len(expected_identity) != 64:
        raise ValueError("missing sealed training identity; use the official recovery-wave entrypoint")
    checked = training_preflight(root, config_path, audit_path=audit_path, require_fresh=require_fresh)
    if checked["config_sha256"] != config.computed_sha256 or checked["identity_sha256"] != expected_identity:
        raise ValueError("recovery-wave inputs/code/config changed after submission")
    return checked


def required_checkpoint_headroom(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    output_root: str = LR_HALF_OUTPUT_ROOT,
) -> int:
    """Full planned remaining saves plus the approved proportional safety margin."""
    remaining = 0
    full_wave = sum(5 * (38 * model + 8 * resume) for model, resume in CHECKPOINT_BYTES.values())
    for row in rows:
        model, resume = CHECKPOINT_BYTES[row["route"]]
        directory = root / output_root / "runs" / row["heldout_task"] / row["route"] / "checkpoints"
        # Count only named completed save files, never temporary/unknown files.
        stored = sum(p.stat().st_size for pattern in ("model_step*.pt", "resume_step*.pt")
                     for p in directory.glob(pattern) if p.is_file())
        remaining += max(0, 38 * model + 8 * resume - stored)
    return math.ceil(remaining * (1.8 * 1024 ** 4) / full_wave)

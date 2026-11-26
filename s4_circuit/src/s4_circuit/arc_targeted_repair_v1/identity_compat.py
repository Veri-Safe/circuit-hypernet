"""Strict, S0-only reconciliation of two historical mount-path identities.

This does not inspect/re-hash large weights. The caller must first run the native
pinned Base audit. No native validator is bypassed: the returned two hashes may
be used in dataclasses.replace(expected, **validated_checkpoint_hashes), followed
by the unchanged checkpoint._validate_identity(actual, expected).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

ARCHIVE_PATH = "logs/eval/2026-11-05/section4-results-completion-v1/arc"
# Release file digests. Checkpoint, parameter and training-identity digests inside the records remain pinned.
ARCHIVE_SHA256 = {
    "checkpoint_identity.json": "dabe269d23bf53b4a7cab44f0b654e75f29d5e61b2c674bb6e3ccb8267f073e8",
    "base_checkpoint_manifest.json": "5dd0671ed5eb703a347a11c012249dd51ae0584d5ecc670d1c69585a26e9f481",
    "training_run_manifest.json": "55b50dc883655df5834e7a3ad8de556af5dab2ee614f2c77faeb70a88e88e6dd",
    "source_config.yaml": "19bc6b34f60f2cb7f47d2bf94cbb83fcf14db559a79403f7f569183b21882751",
    "common_config.yaml": "42ea630371b112a5fa28de0bce00688ea14b115b03b80fa3cfae5ce1d1e334a1",
    "s2_config.yaml": "e1a858ef45d12a9b88d337b625d40925d9fe26daa9626080c0f6657297728411",
}
HISTORICAL_ROOT = "/mnt/kbei/HyperCombine"
CHECKPOINT_RELATIVE = (
    "outputs/nvarc/circuit_stage_c/r1_rearc8_step130/selected_generator/"
    "train/checkpoints/step_00000300"
)
TRAINING_PATH_FIELDS = (
    ("s2_config",),
    ("train_artifacts", "catalog_root"),
    ("train_artifacts", "final_root"),
    ("train_artifacts", "root_manifest"),
    ("output_root",),
    ("fresh_init_root",),
    ("batch_manifest",),
)
BASE_PATH_FIELDS = (("hf_checkpoint",), ("tokenizer",))


def semantic_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_archive_bytes(repo_root: Path, name: str) -> bytes:
    """Read one of the six bundled files and verify its release digest."""
    expected = ARCHIVE_SHA256.get(name)
    if expected is None:
        raise ValueError(f"Unknown S0 archive file: {name}")
    data = (repo_root / ARCHIVE_PATH / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"S0 archive SHA-256 mismatch: {name}")
    return data


def rebind_mount_fields(
    value: Mapping[str, Any], *, fields: tuple[tuple[str, ...], ...],
    current_root: str | Path, historical_root: str | Path = HISTORICAL_ROOT,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Only rebind listed path fields, proving both names are the same resource.

    All fields must exist and be paths under one of the two exact roots. Neither
    arbitrary recursive string replacement nor new-default stripping is allowed.
    """
    current = Path(current_root).resolve(strict=True)
    historical = Path(historical_root)
    if not historical.exists() or historical.resolve(strict=True) != current:
        raise ValueError("Historical/current mount roots do not resolve to the same directory")
    result = deepcopy(dict(value))
    changes = []
    for field in fields:
        parent = result
        for component in field[:-1]:
            parent = parent[component]
        original = parent[field[-1]]
        if not isinstance(original, str):
            raise ValueError(f"Identity path is not a string: {field}")
        original_path = Path(original)
        try:
            relative = original_path.relative_to(current)
        except ValueError:
            try:
                relative = original_path.relative_to(historical)
            except ValueError as exc:
                raise ValueError(f"Identity path outside approved roots: {field}") from exc
        if ".." in relative.parts:
            raise ValueError(f"Identity path contains parent traversal: {field}")
        rebound = historical / relative
        if (not original_path.exists() or not rebound.exists()
                or original_path.resolve(strict=True) != rebound.resolve(strict=True)
                or not original_path.samefile(rebound)):
            raise ValueError(f"Identity path aliases differ or are missing: {field}")
        parent[field[-1]] = str(rebound)
        changes.append({"field": "/" + "/".join(field), "current": original,
                        "historical": str(rebound), "samefile": True,
                        "resolved": str(original_path.resolve(strict=True)),
                        "changed": original != str(rebound)})
    return result, changes


def validate_archive_mount_compatibility(
    *, repo_root: str | Path, training_semantic: Mapping[str, Any],
    base_manifest: Mapping[str, Any], actual_identity: Any,
) -> dict[str, Any]:
    """Prove S0's two identity mismatches are solely the approved mount names.

    The source YAMLs are byte-checked against the pinned release copies, and
    trainer_state is verified against the original checkpoint digest. Every Base field
    outside the two paths must equal the archived Base manifest. The seven
    training paths are the only permitted substitutions; the resulting complete
    semantic object must reproduce both archived training hashes exactly.
    """
    root = Path(repo_root).resolve(strict=True)
    if is_dataclass(actual_identity):
        actual = asdict(actual_identity)
    else:
        actual = dict(actual_identity)
    archive = {}
    archive_receipts = []
    for name in ("checkpoint_identity.json", "base_checkpoint_manifest.json",
                 "training_run_manifest.json"):
        data = read_archive_bytes(root, name)
        archive[name] = json.loads(data)
        archive_receipts.append({"path": f"{ARCHIVE_PATH}/{name}",
                                 "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    sources = []
    for archived, local in (
        ("source_config.yaml", "s4_circuit/configs/selected/s0.yaml"),
        ("common_config.yaml", "s4_circuit/configs/r1_rearc8_step130_common_v1.yaml"),
        ("s2_config.yaml", "s2_hypernet/configs/nvarc_pg/arc1_rearc8_s42_step130_v1.yaml"),
    ):
        saved = read_archive_bytes(root, archived)
        current = (root / local).read_bytes()
        if current != saved:
            raise ValueError(f"Published configuration differs: {local}")
        sources.append({"path": local, "archive_name": archived, "bytes": len(current),
                        "sha256": hashlib.sha256(current).hexdigest(), "byte_equal": True})
    checkpoint_archive = archive["checkpoint_identity.json"]
    if checkpoint_archive.get("accessible_path") != CHECKPOINT_RELATIVE:
        raise ValueError("Archive does not name the requested S0 step300 checkpoint")
    state_path = root / CHECKPOINT_RELATIVE / "trainer_state.json"
    state_bytes = state_path.read_bytes()
    receipts = [x for x in checkpoint_archive["files"]
                if x["path"] == f"{CHECKPOINT_RELATIVE}/trainer_state.json"]
    if len(receipts) != 1 or receipts[0]["bytes"] != len(state_bytes) or (
        receipts[0]["sha256"] != hashlib.sha256(state_bytes).hexdigest()
    ):
        raise ValueError("Physical S0 trainer_state does not match immutable archive receipt")
    state = json.loads(state_bytes)
    if any(state.get(key) != val for key, val in actual.items()):
        raise ValueError("Supplied identity differs from physical archived S0 trainer_state")
    if state.get("global_step") != 300 or state.get("variant_id") != "f3_independent_attention":
        raise ValueError("Compatibility is restricted to S0 step300")

    legacy_base, base_changes = rebind_mount_fields(
        base_manifest, fields=BASE_PATH_FIELDS, current_root=root)
    archived_base = archive["base_checkpoint_manifest.json"]
    if legacy_base != archived_base:
        raise ValueError("Base identity differs beyond the two approved mount paths")
    base_hash = semantic_sha256(legacy_base)
    train_run = archive["training_run_manifest.json"]
    if base_hash != state["base_manifest_sha256"] or base_hash != train_run["base_manifest_sha256"]:
        raise ValueError("Archived Base manifest hash does not match checkpoint and run identities")
    legacy_training, training_changes = rebind_mount_fields(
        training_semantic, fields=TRAINING_PATH_FIELDS, current_root=root)
    training_hash = semantic_sha256(legacy_training)
    if not all(training_hash == ref for ref in (
        state["training_config_sha256"], train_run["training_config_sha256"],
        checkpoint_archive["training_config_sha256"],
    )):
        raise ValueError("Training semantics differ beyond the seven approved mount paths")
    return {
        "format": "arc_s0_mount_only_identity_compatibility_v1", "passed": True,
        "scope": "S0 step300 only; no semantic defaults ignored; no large weights rehashed",
        "validated_checkpoint_hashes": {"training_config_sha256": training_hash,
                                         "base_manifest_sha256": base_hash},
        "current_hashes": {"training_config_sha256": semantic_sha256(training_semantic),
                           "base_manifest_sha256": semantic_sha256(base_manifest)},
        "base_path_evidence": base_changes, "training_path_evidence": training_changes,
        "configuration_source_evidence": sources, "archive_receipts": archive_receipts,
        "trainer_state_receipt": receipts[0],
        "base_other_fields_equal": True, "new_default_fields_ignored": [],
        "required_next_check": "Unchanged native _validate_identity for every checkpoint field",
    }

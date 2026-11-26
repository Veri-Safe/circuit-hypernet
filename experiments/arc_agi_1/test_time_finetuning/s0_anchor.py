"""Pinned S0 step300 adapters from existing first-test canonical circuits.

Preparation hashes the frozen assets once on CPU. Workers reuse those receipts,
check small-file hashes and large-file stat identities, and retain the unchanged
native checkpoint identity validator. No circuit extraction or gold loading lives
on this path. Returned generated factors are raw: the caller applies gate=0.25.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from s4_circuit.arc_targeted_repair_v1.identity_compat import (
    ARCHIVE_PATH, archived_mount, read_archive_bytes, validate_archive_mount_compatibility,
)
from s4_circuit.contracts import semantic_sha256, sha256_file, token_ids_sha256


SOURCE_CONFIG = "s4_circuit/configs/selected/s0.yaml"
SOURCE_CHECKPOINT = "outputs/nvarc/circuit_stage_c/r1_rearc8_step130/selected_generator/train/checkpoints/step_00000300"
MANIFESTS = "outputs/nvarc/circuit_stage_c/r1_rearc8_step130/manifests"
LARGE_FILE_BYTES = 16 * 1024 * 1024


def _path(root: Path, value: str | Path) -> Path:
    from .direct_runtime import _workspace_path
    return _workspace_path(root, value)


def _warmstart(root: Path, cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    warm = cfg["warmstart"]
    if (_path(root, warm["source_config"]) != root / SOURCE_CONFIG
            or _path(root, warm["source_checkpoint"]) != (root / SOURCE_CHECKPOINT).resolve()
            or warm.get("anchor_rule") != "first_test_canonical_v1"
            or float(warm.get("gate", -1)) != 0.25):
        raise ValueError("Warmstart must use pinned S0 step300, first-test canonical, gate 0.25")
    return warm


@contextmanager
def _historical_environment(root: Path):
    """Retain the physical source's original path-bearing native identities."""
    archive_root = archived_mount(root)
    values = {"HC_ROOT": str(archive_root), "NVARC_ATTENTION_IMPLEMENTATION": "sdpa"}
    for key, rel in (
        ("S4_TRAIN_ROOT_SHA256", "outputs/nvarc/circuit_stage_b/rearc8_s42_step130_seed24_exclusive/training_circuits/final/root_manifest.json"),
        ("S4_EVAL_ROOT_SHA256", "outputs/nvarc/circuit_stage_b/rearc8_s42_step130_public_eval/eval_circuits/final/root_manifest.json"),
    ):
        values[key] = sha256_file(root / rel)
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield archive_root
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_source_config(root: Path, cfg: Mapping[str, Any]):
    from s4_circuit.config import load_config

    _warmstart(root, cfg)
    if os.environ.get("NVARC_ATTENTION_IMPLEMENTATION", "sdpa") != "sdpa":
        raise ValueError("S0 step300 source requires its sdpa checkpoint backend")
    with _historical_environment(root) as archive_root:
        config = load_config(root / SOURCE_CONFIG)
        # An archived trainer hashes absolute artifact paths. Public paths may
        # be links to the restored offline archive. Canonicalize only those
        # aliases and retain the native identities of all files and settings.
        def archived_path(path):
            return str(archive_root / _path(root, path).relative_to(root))

        def archived_bank(bank):
            return replace(bank, final_root=archived_path(bank.final_root),
                           root_manifest=archived_path(bank.root_manifest))

        config = replace(config, output_root=archived_path(config.output_root),
                         train_artifacts=archived_bank(config.train_artifacts),
                         eval_artifacts=archived_bank(config.eval_artifacts))
        s2 = config.load_s2_config()
    return config, s2


def _receipt(path: Path, *, digest: str | None = None) -> dict[str, Any]:
    stat = path.stat()
    result = {"sha256": digest or sha256_file(path), "size_bytes": stat.st_size}
    if stat.st_size > LARGE_FILE_BYTES:
        result["mtime_ns"] = stat.st_mtime_ns
    return result


def _verify_receipts(root: Path, receipts: Mapping[str, Mapping[str, Any]]) -> None:
    for rel, receipt in receipts.items():
        path = _path(root, rel)
        stat = path.stat()
        if stat.st_size != receipt["size_bytes"]:
            raise ValueError(f"S0 asset size changed: {rel}")
        if "mtime_ns" in receipt:
            if stat.st_mtime_ns != receipt["mtime_ns"]:
                raise ValueError(f"S0 verified large asset changed; repeat CPU preparation: {rel}")
        elif sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"S0 asset checksum changed: {rel}")


def _anchor_record(view, record, bank, tokenizer) -> dict[str, Any]:
    if not view.is_canonical or int(view.test_index) != 0:
        raise ValueError("S0 anchor must be the first test's canonical view")
    tokens = tuple(tokenizer.encode(view.prompt, add_special_tokens=False))
    condition_sha = token_ids_sha256(tokens)
    if condition_sha != record.condition_input_ids_sha256:
        raise ValueError("S0 anchor prompt differs from the existing circuit join")
    vector_record = bank.record(record.event_key)
    if (vector_record.artifact_id != record.artifact_id
            or vector_record.shard_relative_path != record.final_shard_relative_path
            or vector_record.row_in_shard != record.row_in_shard):
        raise ValueError("S0 anchor circuit artifact differs from the existing join")
    vector = vector_record.vector.detach().float().cpu()
    if vector.shape != (59,) or not torch.isfinite(vector).all():
        raise ValueError("S0 anchor must contain a finite 59D circuit vector")
    values = vector.tolist()
    result = {
        **view.manifest_record(), "event_key": record.event_key,
        "artifact_id": record.artifact_id,
        "condition_input_ids_sha256": condition_sha,
        "final_shard_relative_path": record.final_shard_relative_path,
        "row_in_shard": record.row_in_shard,
        "vector": values, "vector_sha256": semantic_sha256(values),
    }
    result["anchor_sha256"] = semantic_sha256(result)
    return result


def prepare_source(root: Path, cfg: dict, views: Sequence[Any]):
    """Audit the frozen source and materialize 400 circuit anchors on CPU."""
    from s2_hypernet.nvarc_pg.checkpoint import inspect_frozen_hf_checkpoint
    from s2_hypernet.nvarc_pg.eval_manifest import query_manifest_sha256
    from s2_hypernet.nvarc_pg.runtime import load_tokenizer
    from s4_circuit.artifacts import load_vector_bank
    from s4_circuit.checkpoint import S4TrainerState, _validate_identity
    from s4_circuit.evaluation import _checkpoint_identity
    from s4_circuit.join import load_eval_join_bundle, load_join_bundle

    root = Path(root).resolve()
    config, s2 = _load_source_config(root, cfg)
    checkpoint = root / SOURCE_CHECKPOINT
    archived = json.loads(read_archive_bytes(root, "checkpoint_identity.json"))
    receipts: dict[str, dict[str, Any]] = {}
    for item in archived["files"]:
        # Optimizer/RNG state is not consumed when loading an evaluation model.
        if Path(item["path"]).name not in {"parameter_generator.pt", "trainer_state.json", "COMPLETE"}:
            continue
        path = _path(root, item["path"])
        receipt = _receipt(path)
        if receipt["sha256"] != item["sha256"] or receipt["size_bytes"] != item["bytes"]:
            raise ValueError("S0 checkpoint differs from its immutable archive")
        receipts[item["path"]] = receipt
    state = S4TrainerState.from_mapping(json.loads((checkpoint / "trainer_state.json").read_text()))
    base = inspect_frozen_hf_checkpoint(
        hf_checkpoint=s2.base.hf_checkpoint, tokenizer=s2.base.tokenizer,
        source_run_manifest=s2.base.source_run_manifest, profile=s2.profile, layout=s2.target_layout)
    train_join, train_meta = load_join_bundle(root / MANIFESTS / "train_join_v1.safetensors")
    eval_join, eval_meta = load_eval_join_bundle(root / MANIFESTS / "eval_join_v1.safetensors")
    if len(train_join) != 20826 or len(eval_join) != 7123:
        raise ValueError("S0 train/eval circuit join coverage changed")
    if (eval_meta["eval_artifact_contract_sha256"] != config.eval_artifact_contract_sha256
            or eval_meta["final_root_manifest_sha256"] != config.eval_artifacts.root_manifest_sha256):
        raise ValueError("S0 evaluation circuit contract changed")
    query_sha = query_manifest_sha256(tuple(views))
    if (len(views) != 7123 or query_sha != eval_meta["query_manifest_sha256"]
            or {int(v.query_index) for v in views} != set(range(7123))):
        raise ValueError("S0 source requires the complete frozen query manifest")
    fresh_path = _path(root, config.fresh_init_root) / "manifest.json"
    fresh = json.loads(fresh_path.read_text())
    if semantic_sha256({k: v for k, v in fresh.items() if k != "manifest_semantic_sha256"}) != fresh["manifest_semantic_sha256"]:
        raise ValueError("S0 fresh-init manifest changed")
    expected = _checkpoint_identity(
        config=config, s2=s2, base_manifest_sha256=semantic_sha256(base),
        train_join_metadata=train_meta, fresh_init_sha256=fresh["aggregate_sha256"])
    compatibility = validate_archive_mount_compatibility(
        repo_root=root, training_semantic=config.training_semantic_dict,
        base_manifest=base, actual_identity=state)
    expected = replace(expected, **compatibility["validated_checkpoint_hashes"])
    _validate_identity(state, expected)
    if state.global_step != 300 or config.optimization.lora_gate_at_step(300) != 0.25:
        raise ValueError("S0 source step/gate changed")
    asset_paths = [
        root / SOURCE_CONFIG, root / "s4_circuit/configs/r1_rearc8_step130_common_v1.yaml",
        _path(root, config.s2_config), _path(root, config.batch_manifest),
        _path(root, config.train_artifacts.root_manifest), _path(root, config.eval_artifacts.root_manifest),
        root / MANIFESTS / "train_join_v1.safetensors", root / MANIFESTS / "eval_join_v1.safetensors",
        fresh_path, fresh_path.parent / fresh["snapshot_path"],
        _path(root, s2.evaluation.protocol_config), _path(root, s2.evaluation.challenges),
        _path(root, s2.base.tokenizer) / "tokenizer_config.json",
        _path(root, s2.base.tokenizer) / "tokenizer.json",
        _path(root, s2.base.tokenizer) / "nvarc_16t_token_mapping.json",
    ]
    for path in asset_paths:
        receipts[str(path.relative_to(root))] = _receipt(path)
    snapshot_rel = str((fresh_path.parent / fresh["snapshot_path"]).relative_to(root))
    if receipts[snapshot_rel]["sha256"] != fresh["snapshot_sha256"]:
        raise ValueError("S0 fresh-init snapshot checksum changed")
    for name, digest in s2.profile.hf_artifact_sha256:
        path = _path(root, s2.base.hf_checkpoint) / name
        receipts[str(path.relative_to(root))] = _receipt(path, digest=digest)
    bank = load_vector_bank(config.eval_artifacts)
    tokenizer = load_tokenizer(s2)
    anchors = {}
    for view in views:
        if int(view.test_index) == 0 and view.is_canonical:
            if view.task_id in anchors:
                raise ValueError("Repeated S0 task anchor")
            anchors[view.task_id] = _anchor_record(view, eval_join[int(view.query_index)], bank, tokenizer)
    task_ids = {view.task_id for view in views}
    if len(anchors) != 400 or set(anchors) != task_ids:
        raise ValueError("S0 warmstart requires exactly one anchor for each of 400 tasks")
    identity = {
        "format": "s0_step300_anchor_source_v1", "source_config": SOURCE_CONFIG,
        "source_checkpoint": SOURCE_CHECKPOINT, "source_step": 300,
        "anchor_rule": "first_test_canonical_v1", "anchor_count": 400, "gate": 0.25,
        "base_checkpoint": str(_path(root, s2.base.hf_checkpoint).relative_to(root)),
        "tokenizer": str(_path(root, s2.base.tokenizer).relative_to(root)),
        "base_aggregate_sha256": base["hf_aggregate_sha256"],
        "expected_checkpoint_identity": asdict(expected),
        "parameter_generator_sha256": receipts[SOURCE_CHECKPOINT + "/parameter_generator.pt"]["sha256"],
        "checkpoint_state_sha256": receipts[SOURCE_CHECKPOINT + "/trainer_state.json"]["sha256"],
        "query_manifest_sha256": query_sha, "eval_join_sha256": eval_meta["join_sha256"],
        "archive_path": ARCHIVE_PATH,
        "mount_compatibility": {"passed": True, "validated_checkpoint_hashes": compatibility["validated_checkpoint_hashes"]},
        "files": receipts,
    }
    identity["sha256"] = semantic_sha256(identity)
    return identity, anchors


def build_source(root: Path, cfg: dict, source_identity: dict, device):
    """Load the pinned step130 encoder and S0 PG; never load the step65 model."""
    from s2_hypernet.nvarc_pg.runtime import DistributedContext
    from s4_circuit.checkpoint import S4TrainerState, _validate_identity
    from s4_circuit.runtime import build_evaluation_system

    root = Path(root).resolve()
    config, s2 = _load_source_config(root, cfg)
    body = {k: v for k, v in source_identity.items() if k != "sha256"}
    if semantic_sha256(body) != source_identity.get("sha256"):
        raise ValueError("S0 source identity checksum changed")
    if (source_identity["source_config"] != SOURCE_CONFIG
            or source_identity["source_checkpoint"] != SOURCE_CHECKPOINT
            or source_identity["anchor_rule"] != "first_test_canonical_v1"
            or source_identity["source_step"] != 300 or source_identity["gate"] != 0.25):
        raise ValueError("S0 source identity does not match the warmstart configuration")
    _verify_receipts(root, source_identity["files"])
    checkpoint = root / SOURCE_CHECKPOINT
    actual = S4TrainerState.from_mapping(json.loads((checkpoint / "trainer_state.json").read_text()))
    expected = S4TrainerState.from_mapping(source_identity["expected_checkpoint_identity"])
    if (config.training_semantic_sha256 != expected.training_config_sha256
            or s2.profile.hf_aggregate_sha256 != source_identity["base_aggregate_sha256"]):
        raise ValueError("S0 prepared source differs from native configuration identity")
    _validate_identity(actual, expected)
    if actual.global_step != 300:
        raise ValueError("S0 checkpoint is not step300")
    device = torch.device(device)
    context = DistributedContext(rank=0, local_rank=device.index or 0, world_size=1, device=device)
    with _historical_environment(root):
        system, tokenizer, _, model = build_evaluation_system(
            config, context=context, checkpoint=checkpoint, expected_checkpoint_identity=expected)
    model.eval()
    model.requires_grad_(False)
    return system, tokenizer, model


def generate_anchor(system, tokenizer, model, view, anchor: dict, device) -> dict[str, torch.Tensor]:
    """Generate raw factors once; return normal CPU tensors without an autograd graph."""
    from s4_circuit.evaluation import _generate_lora

    if not view.is_canonical or int(view.test_index) != 0:
        raise ValueError("S0 anchor must be first-test canonical")
    if semantic_sha256({k: v for k, v in anchor.items() if k != "anchor_sha256"}) != anchor.get("anchor_sha256"):
        raise ValueError("S0 anchor checksum changed")
    if any(anchor.get(k) != v for k, v in view.manifest_record().items()):
        raise ValueError("S0 query view differs from its pinned anchor")
    if token_ids_sha256(tuple(tokenizer.encode(view.prompt, add_special_tokens=False))) != anchor["condition_input_ids_sha256"]:
        raise ValueError("S0 anchor prompt token identity changed")
    vector = torch.tensor(anchor["vector"], dtype=torch.float32)
    if (vector.shape != (59,) or not torch.isfinite(vector).all()
            or semantic_sha256(anchor["vector"]) != anchor["vector_sha256"]):
        raise ValueError("S0 anchor circuit vector changed")
    if system.parameter_generator is not model or system.controller._state is not None:
        raise ValueError("S0 generation must use its unadapted frozen encoder and pinned PG")
    with torch.no_grad():
        raw = _generate_lora(
            system, tokenizer, [view.prompt], circuit_vectors=vector.unsqueeze(0),
            circuit_present=torch.ones(1, dtype=torch.bool), device=torch.device(device))
        system.controller._validate_state(raw)
    # _generate_lora uses inference_mode internally. Cloning outside it prevents
    # inference tensors from later being saved by adapter-training autograd.
    with torch.inference_mode(False), torch.no_grad():
        result = {key: value.detach().to(device="cpu").clone() for key, value in raw.items()}
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("S0 generated adapter contains nonfinite values")
    return result

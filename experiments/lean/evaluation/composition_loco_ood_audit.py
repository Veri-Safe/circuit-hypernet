"""Prompt-content independence gate for the existing LOCO OOD development bank.

This audits text, not model scores.  The final-test model bank is never opened.
The large frozen training JSONL is streamed once, with O(dev-size) retained state.
No rows are filtered or regenerated when a duplicate is found.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .circuit.composition_circuit_corpus import load_frozen_metadata, validate_sample_row
from .composition_loco import (
    CACHE_FORMAT, COMPOSITION_EVENT_COUNT, COMPOSITION_TASKS, EVAL_BANK_FORMAT,
    EVAL_CORPUS_FORMAT, FINAL_SAMPLES_PER_TASK, PROTOCOL_ID, STAGE_A_BASE_IDENTITY,
    STAGE_B_LEGACY_RENDERER_VERSION, VALIDATION_SAMPLES_PER_TASK, semantic_sha256,
    sha256_file,
)
from .composition_loco_config import CompositionLOCOConfig, load_config


AUDIT_FORMAT = "s5_composition_loco_ood_prompt_audit_v1"
DEFAULT_CONFIG = "s5_lean_circuit/configs/composition_loco_sft_v1/T_and_or/direct_sft.json"
DEFAULT_OUTPUT = "logs/data/2026-09-07/s5-composition-loco-ood-validation/audit.json"
_SIDECAR_EXCLUDED = {"status", "payload_path", "payload_sha256", "payload_bytes", "identity_sha256"}


def _audit_identity(report: Mapping[str, Any]) -> str:
    return semantic_sha256({key: value for key, value in report.items()
                           if key not in {"audit_identity_sha256", "created_at"}})


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _metadata(path: Path, expected_format: str) -> dict[str, Any]:
    value = _json(path)
    identity = {key: item for key, item in value.items() if key not in _SIDECAR_EXCLUDED}
    if (
        value.get("format") != expected_format
        or value.get("status") != "complete"
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("identity_sha256") != semantic_sha256(identity)
        or value.get("expected_source_base_identity") != STAGE_A_BASE_IDENTITY
    ):
        raise ValueError(f"invalid LOCO source metadata: {path}")
    return value


def _eval_manifest(path: Path, name: str, per_task: int, seed_base: int) -> dict[str, Any]:
    value = _json(path)
    body = {key: item for key, item in value.items() if key != "manifest_identity_sha256"}
    if (
        value.get("format") != EVAL_CORPUS_FORMAT
        or value.get("status") != "complete"
        or value.get("protocol_id") != PROTOCOL_ID
        or value.get("manifest_identity_sha256") != semantic_sha256(body)
        or value.get("name") != name
        or value.get("seed_base") != seed_base
        or value.get("samples_per_task") != per_task
        or value.get("event_count") != len(COMPOSITION_TASKS) * per_task
        or value.get("tasks") != list(COMPOSITION_TASKS)
        or value.get("task_counts") != {task: per_task for task in COMPOSITION_TASKS}
        or value.get("renderer_version") != STAGE_B_LEGACY_RENDERER_VERSION
        or value.get("n_distractors") != 2
    ):
        raise ValueError(f"evaluation text corpus protocol changed: {path}")
    return value


def _scan(
    manifest_path: Path, manifest: Mapping[str, Any], *, training: bool,
    keep_hashes: bool = False, needles: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, list[int]], dict[str, list[int]]]:
    """Compute both file SHA and actual query SHA during one JSONL pass."""
    samples = manifest_path.parent / str(manifest["samples_path"])
    before = _fingerprint(samples)
    digest = hashlib.sha256()
    hashes: dict[str, list[int]] = defaultdict(list)
    matched: dict[str, list[int]] = defaultdict(list)
    counts: Counter[str] = Counter()
    query_sequence = hashlib.sha256()
    record_sequence = hashlib.sha256()
    count = 0
    with samples.open("rb") as handle:
        for index, line in enumerate(handle):
            digest.update(line)
            if not line.strip():
                raise ValueError(f"blank sample line: {samples}:{index + 1}")
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("input_text"), str):
                raise ValueError(f"invalid sample query: {samples}:{index + 1}")
            if training:
                validate_sample_row(row, expected_index=index)
            elif (
                row.get("format") != EVAL_CORPUS_FORMAT
                or row.get("canonical_global_index") != index
                or row.get("task_type") not in COMPOSITION_TASKS
            ):
                raise ValueError(f"evaluation text row order/schema changed: {samples}:{index + 1}")
            query_hash = hashlib.sha256(row["input_text"].encode("utf-8")).hexdigest()
            if row.get("input_text_sha256") != query_hash:
                raise ValueError(f"sample query hash changed: {samples}:{index + 1}")
            # No output_text, proof_group, correct, or model score is consulted.
            if not training:
                key = semantic_sha256({
                    "format": EVAL_CORPUS_FORMAT, "canonical_global_index": index,
                    "task_type": row["task_type"], "seed": row["seed"],
                    "renderer_version": manifest["renderer_version"],
                    "input_text_sha256": query_hash,
                })
                if row.get("prompt_key") != key:
                    raise ValueError("evaluation prompt provenance changed")
            query_sequence.update(bytes.fromhex(query_hash))
            record_sequence.update(bytes.fromhex(semantic_sha256({
                "index": index, "task_type": row["task_type"], "seed": row["seed"],
                "input_text_sha256": query_hash,
            })))
            counts[str(row["task_type"])] += 1
            count += 1
            if keep_hashes:
                hashes[query_hash].append(index)
            if needles is not None and query_hash in needles:
                # Counts can be large; retained witness indices are bounded.
                if len(matched[query_hash]) < 8:
                    matched[query_hash].append(index)
    after = _fingerprint(samples)
    if before != after:
        raise ValueError(f"corpus changed during audit: {samples}")
    if (
        digest.hexdigest() != manifest.get("samples_sha256")
        or before["bytes"] != manifest.get("samples_bytes")
        or count != manifest.get("event_count")
        or dict(counts) != manifest.get("task_counts")
    ):
        raise ValueError(f"corpus bytes/counts differ from frozen manifest: {samples}")
    report = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "samples": {**before, "sha256": digest.hexdigest()},
        "event_count": count, "task_counts": dict(counts),
        "input_sha256_sequence_sha256": query_sequence.hexdigest(),
        "record_provenance_sequence_sha256": record_sequence.hexdigest(),
    }
    return report, dict(hashes), dict(matched)


def audit_prompt_corpora(
    training_manifest: str | Path, validation_manifest: str | Path, final_manifest: str | Path,
) -> dict[str, Any]:
    """Audit immutable source corpora; missing final text is a blocking error."""
    train_path, dev_path, final_path = map(Path, (training_manifest, validation_manifest, final_manifest))
    train, _lock = load_frozen_metadata(train_path)
    dev = _eval_manifest(dev_path, "visible_validation", VALIDATION_SAMPLES_PER_TASK, 2_000_000_000)
    final = _eval_manifest(final_path, "final_test", FINAL_SAMPLES_PER_TASK, 3_000_000_000)
    dev_report, dev_hashes, _ = _scan(dev_path, dev, training=False, keep_hashes=True)
    train_report, _, train_matches = _scan(train_path, train, training=True, needles=set(dev_hashes))
    final_report, _, final_matches = _scan(final_path, final, training=False, needles=set(dev_hashes))
    duplicates = {key: value for key, value in dev_hashes.items() if len(value) > 1}

    def witnesses(matches: Mapping[str, list[int]]) -> list[dict[str, Any]]:
        return [{"input_text_sha256": key, "dev_indices": dev_hashes[key], "source_indices": matches[key]}
                for key in sorted(matches)[:12]]

    failures = []
    if duplicates:
        failures.append("duplicate_input_text_within_ood_validation")
    if train_matches:
        failures.append("ood_validation_overlaps_training_corpus")
    if final_matches:
        failures.append("ood_validation_overlaps_final_text_corpus")
    return {
        "status": "blocked" if failures else "ready", "blocking_reasons": failures,
        "hash_basis": "sha256(input_text.encode('utf-8')); seed/index/prompt_key excluded",
        "training": train_report, "validation": dev_report, "final_text": final_report,
        "checks": {
            "within_dev_duplicate_query_count": len(duplicates),
            "within_dev_duplicate_extra_rows": sum(len(rows) - 1 for rows in duplicates.values()),
            "dev_queries_overlapping_training": len(train_matches),
            "dev_queries_overlapping_final_text": len(final_matches),
            "duplicate_witnesses": witnesses(duplicates),
            "training_overlap_witnesses": witnesses(train_matches),
            "final_overlap_witnesses": witnesses(final_matches),
        },
    }


def _resolve_training_manifest(root: Path, cache: Mapping[str, Any]) -> Path:
    outputs = root / "outputs" / "s5_lean_circuit"
    # Match the official materialize/extract launcher first. Never recursively
    # enumerate the millions of Graph Payload artifact directories.
    candidates = [outputs / "composition_circuit_1p5m_v1/corpus/manifest.json"]
    if cache.get("corpus_manifest_path"):
        candidates.insert(0, _path(root, cache["corpus_manifest_path"]))
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == cache["corpus_manifest_sha256"]:
            value = _json(candidate)
            if (candidate.parent / str(value.get("samples_path", ""))).is_file():
                return candidate.resolve()
    matches = []
    for pattern in ("*/manifest.json", "*/*/manifest.json"):
        for candidate in outputs.glob(pattern):
            if sha256_file(candidate) == cache["corpus_manifest_sha256"]:
                value = _json(candidate)
                if (candidate.parent / str(value.get("samples_path", ""))).is_file():
                    matches.append(candidate.resolve())
    if not matches:
        raise ValueError("cannot find the frozen training corpus matching the cache metadata")
    # Byte-identical immutable copies are interchangeable, not distinct datasets.
    return sorted(matches)[0]


def run_audit(root: str | Path, *, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(root).resolve()
    report: dict[str, Any] = {
        "format": AUDIT_FORMAT, "protocol_id": PROTOCOL_ID, "root": str(root),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "blocked",
        "scope": "all five existing validation task pools; no checkpoint selection or final scoring",
        "final_model_bank_opened": False, "rows_filtered_or_regenerated": False,
    }
    try:
        config_file = _path(root, config_path or DEFAULT_CONFIG)
        config = load_config(config_file)
        cache_path = _path(root, config.data["cache_manifest"])
        cache = _metadata(cache_path, CACHE_FORMAT)
        if cache.get("event_count") != COMPOSITION_EVENT_COUNT:
            raise ValueError("training cache does not cover the frozen 1.5M source")
        bank = _path(root, config.data["validation_bank"])
        bank_sidecar = bank.with_suffix(bank.suffix + ".json")
        bank_meta = _metadata(bank_sidecar, EVAL_BANK_FORMAT)
        if bank_meta.get("corpus_name") != "visible_validation":
            raise ValueError("expected the existing visible_validation bank")
        training_manifest = _resolve_training_manifest(root, cache)
        dev_manifest = bank.parent / "corpora" / "visible_validation" / "manifest.json"
        final_manifest = bank.parent / "corpora" / "final_test" / "manifest.json"
        report.update(audit_prompt_corpora(training_manifest, dev_manifest, final_manifest))
        if (
            report["validation"]["manifest_sha256"] != bank_meta.get("corpus_manifest_sha256")
            or report["validation"]["manifest_identity_sha256"] != bank_meta.get("corpus_manifest_identity_sha256")
            or report["training"]["manifest_identity_sha256"] != cache.get("corpus_manifest_identity_sha256")
        ):
            raise ValueError("audited text corpora differ from bank/cache sources")
        # Tie actual bank query records to the audited corpus, not just metadata assertions.
        records = bank.with_suffix(bank.suffix + ".records.jsonl")
        record_digest = hashlib.sha256()
        sequence = hashlib.sha256()
        provenance_sequence = hashlib.sha256()
        count = 0
        with records.open("rb") as handle:
            for index, line in enumerate(handle):
                record_digest.update(line)
                row = json.loads(line)
                if row.get("canonical_global_index") != index:
                    raise ValueError("validation bank record order changed")
                query_digest = hashlib.sha256(row["input_text"].encode("utf-8")).digest()
                sequence.update(query_digest)
                provenance_sequence.update(bytes.fromhex(semantic_sha256({
                    "index": index, "task_type": row["task_type"], "seed": row["seed"],
                    "input_text_sha256": query_digest.hex(),
                })))
                count += 1
        if (
            record_digest.hexdigest() != bank_meta.get("records_sha256")
            or sequence.hexdigest() != report["validation"]["input_sha256_sequence_sha256"]
            or provenance_sequence.hexdigest() != report["validation"]["record_provenance_sequence_sha256"]
            or count != report["validation"]["event_count"]
        ):
            raise ValueError("validation bank queries do not match the audited text corpus")
        report["bindings"] = {
            "cache_manifest_path": str(cache_path), "cache_manifest_sha256": sha256_file(cache_path),
            "cache_identity_sha256": cache["identity_sha256"],
            "bank_path": str(bank), "bank_sidecar_sha256": sha256_file(bank_sidecar),
            "bank_identity_sha256": bank_meta["identity_sha256"],
            "bank_payload_sha256": bank_meta["payload_sha256"],
            "bank_records_sha256": record_digest.hexdigest(),
        }
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["status"] = "blocked"
        report.setdefault("blocking_reasons", []).append(f"{type(error).__name__}: {error}")
    report["audit_identity_sha256"] = _audit_identity(report)
    return report


def require_ood_audit(
    audit_path: str | Path, config: CompositionLOCOConfig, bank_path: str | Path,
    *, bank_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Require a ready source audit without re-streaming 1.5M rows per GPU job.

Source file size/mtime guards invalidate any ordinary replacement since the scan;
the report records its actual byte SHA.  A changed source requires a new audit.
The evaluator separately performs the full validation-bank payload SHA check.
"""
    report = _json(Path(audit_path))
    if (
        report.get("format") != AUDIT_FORMAT or report.get("protocol_id") != PROTOCOL_ID
        or report.get("status") != "ready" or report.get("blocking_reasons")
        or report.get("audit_identity_sha256") != _audit_identity(report)
        or report.get("rows_filtered_or_regenerated") is not False
        or report.get("final_model_bank_opened") is not False
    ):
        raise ValueError("OOD validation prompt-independence audit is missing, blocked, or changed")
    saved_root = Path(report["root"])
    bank = Path(bank_path).resolve()
    declared_bank = Path(config.data["validation_bank"])
    if declared_bank.is_absolute():
        declared_bank = declared_bank.relative_to(saved_root)
    if tuple(bank.parts[-len(declared_bank.parts):]) != tuple(declared_bank.parts):
        raise ValueError("OOD validation bank is outside its declared repository-relative path")
    root = bank.parents[len(declared_bank.parts) - 1]

    def rebased(value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute() and path.is_relative_to(saved_root):
            return root / path.relative_to(saved_root)
        return _path(root, path)

    binding = report["bindings"]
    cache_path = rebased(config.data["cache_manifest"])
    sidecar = bank.with_suffix(bank.suffix + ".json")
    metadata = bank_metadata if bank_metadata is not None else _metadata(sidecar, EVAL_BANK_FORMAT)
    if (
        bank != rebased(config.data["validation_bank"])
        or bank != rebased(binding["bank_path"])
        or cache_path != rebased(binding["cache_manifest_path"])
        or sha256_file(cache_path) != binding["cache_manifest_sha256"]
        or sha256_file(sidecar) != binding["bank_sidecar_sha256"]
        or metadata.get("identity_sha256") != binding["bank_identity_sha256"]
        or metadata.get("payload_sha256") != binding["bank_payload_sha256"]
        or metadata.get("corpus_manifest_sha256") != report["validation"]["manifest_sha256"]
        or sha256_file(bank.with_suffix(bank.suffix + ".records.jsonl")) != binding["bank_records_sha256"]
    ):
        raise ValueError("OOD validation audit does not bind these model data sources")
    for name in ("training", "validation", "final_text"):
        source = report[name]
        samples = source["samples"]
        sample_path = rebased(samples["path"])
        fingerprint = _fingerprint(sample_path)
        if (
            sha256_file(rebased(source["manifest_path"])) != source["manifest_sha256"]
            or any(fingerprint[key] != samples[key] for key in ("bytes", "mtime_ns"))
            or (name == "validation" and sha256_file(sample_path) != samples["sha256"])
        ):
            raise ValueError(f"{name} corpus changed since OOD independence audit; rerun audit")
    return report

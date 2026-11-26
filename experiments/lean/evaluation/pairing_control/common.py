"""CPU-only contracts for evaluation of five immutable shuffle checkpoints."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
LOG = ROOT / "logs/eval/2026-11-23/lean-stage2-permuted-formal-eval-v1"
TASKS = ("T_and_or", "T_or_and", "T_and_or_and", "T_nested_and", "T_nested_or")
ATOMS = ("S_and", "S_or_l", "S_or_r", "S_id")
SUITES = ("suite1", "suite2")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path, value):
    """Never replace an existing artifact, including failed attempts."""
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_manifest():
    return read(PACKAGE / "eval_manifest.json")


def check_file(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"immutable file changed: {path}: {actual} != {expected}")
    return actual


def verify_sources(manifest):
    for path, digest in manifest["runtime_sources"].items():
        check_file(ROOT / path, digest)


def config_for(task, manifest):
    spec = manifest["tasks"][task]
    path = ROOT / spec["config_path"]
    check_file(path, spec["config_file_sha256"])
    config = read(path)
    body = {k: v for k, v in config.items() if k != "config_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    if digest != spec["config_sha256"] or config["config_sha256"] != digest:
        raise ValueError("release training configuration identity changed")
    seen = [t for t in TASKS if t != task]
    if (config["heldout_task"] != task or config["train_tasks"] != seen
            or config["evaluation"]["selection_tasks"] != seen
            or config["route"] != "circuit_a" or config["model"]["condition_source"] != "circuit"
            or config["data"]["validation_bank"] != manifest["validation"]["path"]
            or config["data"]["atomic_bank"] != manifest["atomic_validation"]["path"]
            or config["training"]["validation_steps"] != [0] + manifest["validation"]["eligible_steps"]):
        raise ValueError("training selection contract differs from main table")
    return config


def validate_counts(counts, tasks):
    if set(counts) != set(tasks):
        raise ValueError("task coverage mismatch")
    for row in counts.values():
        if (type(row.get("total")) is not int or row["total"] != 500
                or type(row.get("correct")) is not int or not 0 <= row["correct"] <= 500):
            raise ValueError("each task must have exactly 500 scored questions")
    return sum(row["correct"] for row in counts.values())


def validate_selection(selection, task, spec, validation, history):
    """Audit original dev-only selection; never rank on formal-test results.

    ``history`` is loaded from the original per-step visible evaluation files,
    which retain the bank identity omitted from some normalized selection rows.
    """
    seen = [t for t in TASKS if t != task]
    steps = [0] + validation["eligible_steps"]
    if (selection.get("status") != "complete"
            or selection.get("config_sha256") != spec["config_sha256"]
            or selection.get("heldout_task") != task
            or selection.get("visible_tasks") != seen
            or selection.get("selection_metric") != "visible_composition_macro_accuracy"
            or selection.get("selection_denominator") != 2000
            or selection.get("step_zero_reference_only") is not True
            or selection.get("validation_steps") != steps
            or selection.get("atomic_validation", {}).get("selection_uses_atomic") is not False):
        raise ValueError("selection receipt differs from main-table dev-only rule")
    rows = selection.get("evaluations", [])
    if [row["step"] for row in rows] != steps or set(history) != set(steps):
        raise ValueError("incomplete or duplicated validation history")
    candidates = []
    for row in rows:
        step = row["step"]
        original = history[step]
        correct = validate_counts(row["task_counts"], seen)
        if (row.get("overall_correct") != correct or row.get("overall_total") != 2000
                or abs(row.get("score", -1) - correct / 20) > 1e-9
                or original.get("format") != "s5_composition_loco_visible_validation_v1"
                or original.get("step") != step or original.get("heldout_task") != task
                or original.get("visible_tasks") != seen
                or original.get("task_counts") != row["task_counts"]
                or original.get("validation_bank_identity", {}).get("identity_sha256") != validation["identity_sha256"]
                or original.get("validation_bank_identity", {}).get("payload_sha256") != validation["sha256"]):
            raise ValueError(f"validation row identity/count/bank mismatch at step {step}")
        if step > 0:
            checkpoint = row.get("checkpoint", {})
            original_checkpoint = original.get("checkpoint", {})
            if (checkpoint.get("global_step") != step
                    or checkpoint.get("payload_sha256") != original_checkpoint.get("payload_sha256")
                    or not re.fullmatch(r"[0-9a-f]{64}", checkpoint.get("payload_sha256", ""))
                    or Path(checkpoint.get("path", "")).name != f"model_step{step:06d}.pt"):
                raise ValueError(f"validation checkpoint binding changed at step {step}")
            candidates.append((correct, -step, row))
    _, _, chosen = max(candidates, key=lambda item: item[:2])
    if (chosen["step"] != spec["selected_step"] or selection.get("selected_step") != chosen["step"]
            or selection.get("selected_correct") != chosen["overall_correct"]
            or abs(selection.get("selected_score", -1) - chosen["score"]) > 1e-9
            or chosen["checkpoint"]["payload_sha256"] != spec["checkpoint_sha256"]):
        raise ValueError("fixed checkpoint is not the original earliest dev winner; do not reselect")
    return {"status": "passed", "task_counts": chosen["task_counts"],
            "selected_correct": chosen["overall_correct"], "selected_step": chosen["step"],
            "bank_sha256": validation["sha256"], "bank_identity_sha256": validation["identity_sha256"],
            "selection_denominator": 2000, "eligible_steps": validation["eligible_steps"],
            "tie_break": "earliest_positive_step", "selection_uses_atomic": False}


def selection_audit(task, config, manifest, checkpoint_config_sha256):
    output = ROOT / config["output_dir"]
    path = output / "selection.json"
    steps = [0] + manifest["validation"]["eligible_steps"]
    paths = {step: output / "evaluations" / f"visible_step{step:06d}.json" for step in steps}
    archived_spec = {**manifest["tasks"][task], "config_sha256": checkpoint_config_sha256}
    audit = validate_selection(read(path), task, archived_spec, manifest["validation"],
                               {step: read(p) for step, p in paths.items()})
    audit["selection_path"] = str(path.relative_to(ROOT))
    audit["selection_sha256"] = sha(path)
    audit["history_sha256"] = {str(step): sha(p) for step, p in paths.items()}
    atomic = read(path)["atomic_validation"].get("bank_identity", {})
    if (atomic.get("identity_sha256") != manifest["atomic_validation"]["identity_sha256"]
            or atomic.get("payload_sha256") != manifest["atomic_validation"]["sha256"]):
        raise ValueError("atomic diagnostic bank differs from main table")
    return audit


def check_bank_metadata(metadata, spec, tasks, manifest):
    if (metadata.get("identity_sha256") != spec["identity_sha256"]
            or metadata.get("payload_sha256") != spec["sha256"]
            or metadata.get("tokenizer_sha256") != manifest["tokenizer"]["sha256"]
            or metadata.get("n_ctx") != 128
            or metadata.get("event_count") != 500 * len(tasks)
            or metadata.get("task_counts") != {task: 500 for task in tasks}
            or ("seed_base" in spec and metadata.get("seed_base") != spec["seed_base"])):
        raise ValueError("bank metadata differs from frozen formal/development contract")


def verify_predictions(result, tasks):
    validate_counts(result["task_counts"], tasks)
    rows = result["predictions"]
    if len(rows) != 500 * len(tasks) or result["event_count"] != len(rows):
        raise ValueError("incomplete prediction coverage")
    if len({p["target_event_key"] for p in rows}) != len(rows):
        raise ValueError("duplicate prediction event keys")
    if [p["canonical_global_index"] for p in rows] != list(range(len(rows))):
        raise ValueError("prediction order/indices changed")
    counts = {t: {"correct": 0, "total": 0} for t in tasks}
    for row in rows:
        counts[row["task_type"]]["total"] += 1
        counts[row["task_type"]]["correct"] += int(row["correct"])
    if counts != result["task_counts"]:
        raise ValueError("predictions and reported counts disagree")


def merge_family_results(results, tasks):
    """Join separately decoded task families without changing their batches."""
    if len(results) != len(tasks):
        raise ValueError("one formal evaluation result is required per task")
    counts, groups, predictions = {}, {}, []
    mapping = results[0]["condition_mapping"]
    for task, result in zip(tasks, results, strict=True):
        validate_counts(result["task_counts"], (task,))
        if (result.get("condition") != "matched" or result.get("route") != "circuit_a"
                or result.get("condition_mapping") != mapping or result.get("event_count") != 500
                or set(result["proof_groups"]) != {task}
                or len(result["predictions"]) != 500
                or any(row["task_type"] != task for row in result["predictions"])):
            raise ValueError("per-task formal evaluation identity/coverage differs")
        counts.update(result["task_counts"])
        groups.update(result["proof_groups"])
        predictions.extend(result["predictions"])
    predictions.sort(key=lambda row: row["canonical_global_index"])
    combined = {"task_counts": counts, "proof_groups": groups, "predictions": predictions,
                "event_count": 500 * len(tasks), "condition_mapping": mapping}
    verify_predictions(combined, tasks)
    return combined


def write_prediction_chunks(directory, rows, limit=4_000_000):
    chunks, data, count = [], bytearray(), 0
    def flush():
        path = directory / f"predictions-{len(chunks):03d}.jsonl"
        with path.open("xb") as stream:
            stream.write(data)
        chunks.append({"path": path.name, "sha256": sha(path), "rows": count, "bytes": len(data)})
    for row in rows:
        encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if len(encoded) > limit:
            raise ValueError("one prediction exceeds chunk limit")
        if data and len(data) + len(encoded) > limit:
            flush()
            data, count = bytearray(), 0
        data.extend(encoded)
        count += 1
    if data:
        flush()
    return chunks

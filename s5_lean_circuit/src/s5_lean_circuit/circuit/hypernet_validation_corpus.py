"""Frozen all-nine public-validation corpus for HyperNet Circuit extraction."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from ..protocol import (
    ALL_TASKS,
    EvaluationSample,
    STAGE_B_LEGACY_RENDERER_VERSION,
    TASK_TO_ID,
    create_evaluation_dataset,
)


CORPUS_FORMAT = "s5_hypernet_allnine_validation_circuit_corpus_v1"
SAMPLE_FORMAT = "s5_hypernet_allnine_validation_circuit_sample_v1"
PROTOCOL_ID = "s5_hypernet_allnine_matched_validation_seed888888000_v1"
EVENT_COUNT = 450
SAMPLES_PER_TASK = 50
SEED_BASE = 888_888_000


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def _jsonl_bytes(values: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        for value in values
    )


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("validation Circuit row is not an object")
                yield value


def build_corpus() -> dict[str, Any]:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("validation Circuit corpus requires PYTHONHASHSEED=42")
    state = random.getstate()
    random.seed(42)
    try:
        records = create_evaluation_dataset(
            task_types=ALL_TASKS,
            samples_per_task=SAMPLES_PER_TASK,
            seed_base=SEED_BASE,
            n_distractors=2,
            surface_family="narrow",
        )
    finally:
        random.setstate(state)
    if len(records) != EVENT_COUNT:
        raise RuntimeError("all-nine validation suite cardinality changed")
    samples = []
    supervision = []
    suite_rows = []
    for index, record in enumerate(records):
        input_sha = hashlib.sha256(record.input_text.encode("utf-8")).hexdigest()
        output_sha = hashlib.sha256(record.output_text.encode("utf-8")).hexdigest()
        prompt_key = semantic_sha256(
            {
                "format": SAMPLE_FORMAT,
                "canonical_global_index": index,
                "task_type": record.task_type,
                "seed": int(record.seed),
                "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
                "input_text_sha256": input_sha,
            }
        )
        samples.append(
            {
                "format": SAMPLE_FORMAT,
                "canonical_global_index": index,
                "prompt_key": prompt_key,
                "task_type": record.task_type,
                "task_id": TASK_TO_ID[record.task_type],
                "seed": int(record.seed),
                "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
                "surface_family": "narrow",
                "input_text": record.input_text,
                "input_text_sha256": input_sha,
                "supervision_text_sha256": output_sha,
            }
        )
        supervision.append(
            {
                "canonical_global_index": index,
                "prompt_key": prompt_key,
                "supervision_text": record.output_text,
                "supervision_text_sha256": output_sha,
            }
        )
        suite_rows.append(
            {
                "task_type": record.task_type,
                "seed": int(record.seed),
                "input_text": record.input_text,
                "output_text": record.output_text,
            }
        )
    counts = Counter(value["task_type"] for value in samples)
    if counts != Counter({task: SAMPLES_PER_TASK for task in ALL_TASKS}):
        raise RuntimeError("all-nine validation task balance changed")
    sample_bytes = _jsonl_bytes(samples)
    supervision_bytes = _jsonl_bytes(supervision)
    return {
        "format": CORPUS_FORMAT,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "role": "matched_validation_circuit_bank",
        "event_count": EVENT_COUNT,
        "samples_per_task": SAMPLES_PER_TASK,
        "seed_base": SEED_BASE,
        "surface_family": "narrow",
        "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
        "n_distractors": 2,
        "tasks": list(ALL_TASKS),
        "task_counts": dict(counts),
        "suite_sha256": semantic_sha256(suite_rows),
        "samples": samples,
        "supervision": supervision,
        "samples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "supervision_sha256": hashlib.sha256(supervision_bytes).hexdigest(),
    }


def write_corpus(output_dir: str | Path, corpus: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    samples = list(corpus["samples"])
    supervision = list(corpus["supervision"])
    sample_bytes = _jsonl_bytes(samples)
    supervision_bytes = _jsonl_bytes(supervision)
    if hashlib.sha256(sample_bytes).hexdigest() != corpus["samples_sha256"]:
        raise ValueError("validation samples changed before publication")
    if hashlib.sha256(supervision_bytes).hexdigest() != corpus["supervision_sha256"]:
        raise ValueError("validation supervision changed before publication")
    _atomic_bytes(root / "samples.jsonl", sample_bytes)
    _atomic_bytes(root / "supervision.jsonl", supervision_bytes)
    manifest = {
        key: value
        for key, value in corpus.items()
        if key not in {"samples", "supervision"}
    }
    manifest.update(
        {"samples_path": "samples.jsonl", "supervision_path": "supervision.jsonl"}
    )
    manifest["manifest_identity_sha256"] = semantic_sha256(manifest)
    _atomic_bytes(
        root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def verify_corpus(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CORPUS_FORMAT or manifest.get("status") != "complete":
        raise ValueError("validation Circuit corpus is incomplete")
    identity = str(manifest.get("manifest_identity_sha256", ""))
    body = {
        key: value for key, value in manifest.items() if key != "manifest_identity_sha256"
    }
    if semantic_sha256(body) != identity:
        raise ValueError("validation Circuit manifest identity changed")
    samples_path = manifest_path.parent / str(manifest["samples_path"])
    supervision_path = manifest_path.parent / str(manifest["supervision_path"])
    if hashlib.sha256(samples_path.read_bytes()).hexdigest() != manifest["samples_sha256"]:
        raise ValueError("validation Circuit samples checksum changed")
    if (
        hashlib.sha256(supervision_path.read_bytes()).hexdigest()
        != manifest["supervision_sha256"]
    ):
        raise ValueError("validation Circuit supervision checksum changed")
    samples = list(iter_jsonl(samples_path))
    supervision = list(iter_jsonl(supervision_path))
    if len(samples) != EVENT_COUNT or len(supervision) != EVENT_COUNT:
        raise ValueError("validation Circuit coverage changed")
    if [value["prompt_key"] for value in samples] != [
        value["prompt_key"] for value in supervision
    ]:
        raise ValueError("validation Circuit supervision join changed")
    if Counter(str(value["task_type"]) for value in samples) != Counter(
        {task: SAMPLES_PER_TASK for task in ALL_TASKS}
    ):
        raise ValueError("validation Circuit task balance changed")
    return manifest


def load_samples(path: str | Path) -> tuple[dict[str, Any], ...]:
    manifest_path = Path(path).resolve()
    manifest = verify_corpus(manifest_path)
    return tuple(iter_jsonl(manifest_path.parent / str(manifest["samples_path"])))


def load_stored_prompt(record: Mapping[str, Any]) -> tuple[str, str]:
    input_text = str(record["input_text"])
    if hashlib.sha256(input_text.encode("utf-8")).hexdigest() != record[
        "input_text_sha256"
    ]:
        raise ValueError("validation stored prompt changed")
    return input_text, ""


def load_evaluation_records(path: str | Path) -> tuple[EvaluationSample, ...]:
    manifest_path = Path(path).resolve()
    manifest = verify_corpus(manifest_path)
    samples = list(iter_jsonl(manifest_path.parent / str(manifest["samples_path"])))
    supervision = list(
        iter_jsonl(manifest_path.parent / str(manifest["supervision_path"]))
    )
    records = tuple(
        EvaluationSample(
            task_type=str(sample["task_type"]),
            seed=int(sample["seed"]),
            input_text=str(sample["input_text"]),
            output_text=str(target["supervision_text"]),
        )
        for sample, target in zip(samples, supervision, strict=True)
    )
    suite_rows = [
        {
            "task_type": value.task_type,
            "seed": int(value.seed),
            "input_text": value.input_text,
            "output_text": value.output_text,
        }
        for value in records
    ]
    if semantic_sha256(suite_rows) != manifest["suite_sha256"]:
        raise ValueError("validation suite semantic identity changed")
    return records


__all__ = [
    "CORPUS_FORMAT",
    "EVENT_COUNT",
    "PROTOCOL_ID",
    "SAMPLES_PER_TASK",
    "SEED_BASE",
    "build_corpus",
    "load_evaluation_records",
    "load_samples",
    "load_stored_prompt",
    "verify_corpus",
    "write_corpus",
]

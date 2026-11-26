"""Frozen sample-count corpus manifests for HyperNet Atomic-SFT Circuit assets."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..protocol import (
    ATOMIC_TASKS,
    STAGE_B_RENDERER_VERSION,
    STAGE_B_SUPERVISED_SEED_BASE,
    STAGE_B_TASK_SEED_STRIDE,
    TASK_TO_ID,
    generate_sample,
    stage_b_training_samples,
)


CORPUS_FORMAT = "s5_hypernet_atomic_sft_circuit_corpus_v1"
SAMPLE_FORMAT = "s5_hypernet_atomic_sft_circuit_sample_v1"
PROTOCOL_ID = "s5_hypernet_atomic_sft_120k_v1"
PROTOCOL_ID_1P2M = "s5_hypernet_atomic_sft_1p2m_v1"
SUPPORTED_PREFIXES = (400, 1_200, 4_000, 12_000, 40_000, 120_000, 1_200_000)
COUNTS_PER_100 = {task: 25 for task in ATOMIC_TASKS}


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def prompt_key(
    *, task_type: str, seed: int, renderer_version: str, input_text_sha256: str
) -> str:
    return semantic_sha256(
        {
            "format": SAMPLE_FORMAT,
            "task_type": str(task_type),
            "seed": int(seed),
            "renderer_version": str(renderer_version),
            "input_text_sha256": str(input_text_sha256),
        }
    )


def build_corpus(*, event_count: int) -> dict[str, Any]:
    if int(event_count) not in SUPPORTED_PREFIXES:
        raise ValueError(f"event_count must be one of {SUPPORTED_PREFIXES}")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("HyperNet SFT corpus materialization requires PYTHONHASHSEED=42")
    assignments = stage_b_training_samples(
        global_step=1,
        global_batch_size=int(event_count),
        rank=0,
        world_size=1,
        train_tasks=ATOMIC_TASKS,
        counts_per_cycle=COUNTS_PER_100,
        sampling_seed=42,
        seed_base=STAGE_B_SUPERVISED_SEED_BASE,
        task_seed_stride=STAGE_B_TASK_SEED_STRIDE,
    )
    samples = []
    supervision = []
    for global_index, assignment in enumerate(assignments):
        input_text, output_text = generate_sample(
            assignment.task_type,
            assignment.seed,
            n_distractors=2,
            renderer_version=STAGE_B_RENDERER_VERSION,
        )
        input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
        output_sha = hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        key = prompt_key(
            task_type=assignment.task_type,
            seed=assignment.seed,
            renderer_version=STAGE_B_RENDERER_VERSION,
            input_text_sha256=input_sha,
        )
        samples.append(
            {
                "format": SAMPLE_FORMAT,
                "canonical_global_index": global_index,
                "prompt_key": key,
                "renderer_version": STAGE_B_RENDERER_VERSION,
                "task_type": assignment.task_type,
                "task_id": TASK_TO_ID[assignment.task_type],
                "seed": int(assignment.seed),
                "input_text_sha256": input_sha,
                "supervision_text_sha256": output_sha,
            }
        )
        supervision.append(
            {
                "canonical_global_index": global_index,
                "prompt_key": key,
                "supervision_text_sha256": output_sha,
                "supervision_text": output_text,
            }
        )
    task_counts = Counter(str(record["task_type"]) for record in samples)
    expected_per_task = int(event_count) // len(ATOMIC_TASKS)
    if task_counts != Counter({task: expected_per_task for task in ATOMIC_TASKS}):
        raise ValueError("HyperNet SFT corpus task balance changed")
    if [record["canonical_global_index"] for record in samples] != list(
        range(int(event_count))
    ):
        raise ValueError("HyperNet SFT canonical global order changed")
    keys = [str(record["prompt_key"]) for record in samples]
    if len(set(keys)) != len(keys):
        raise ValueError("HyperNet SFT corpus contains duplicate prompt identities")
    sample_bytes = _jsonl_bytes(samples)
    supervision_bytes = _jsonl_bytes(supervision)
    return {
        "format": CORPUS_FORMAT,
        "status": "complete",
        "protocol_id": (
            PROTOCOL_ID_1P2M if int(event_count) == 1_200_000 else PROTOCOL_ID
        ),
        "event_count": int(event_count),
        "full_cardinality": (
            1_200_000 if int(event_count) == 1_200_000 else 120_000
        ),
        "prefixes": list(
            SUPPORTED_PREFIXES
            if int(event_count) == 1_200_000
            else tuple(value for value in SUPPORTED_PREFIXES if value <= 120_000)
        ),
        "surface_family": "narrow",
        "renderer_version": STAGE_B_RENDERER_VERSION,
        "n_distractors": 2,
        "sampler": {
            "version": "s5_stage_b_weighted_cycle_v1",
            "sampling_seed": 42,
            "seed_base": STAGE_B_SUPERVISED_SEED_BASE,
            "task_seed_stride": STAGE_B_TASK_SEED_STRIDE,
            "counts_per_100": COUNTS_PER_100,
        },
        "task_counts": dict(task_counts),
        "seed_ranges": {
            task: {
                "first": STAGE_B_SUPERVISED_SEED_BASE
                + TASK_TO_ID[task] * STAGE_B_TASK_SEED_STRIDE,
                "last": STAGE_B_SUPERVISED_SEED_BASE
                + TASK_TO_ID[task] * STAGE_B_TASK_SEED_STRIDE
                + expected_per_task
                - 1,
            }
            for task in ATOMIC_TASKS
        },
        "samples": samples,
        "supervision": supervision,
        "samples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "supervision_sha256": hashlib.sha256(supervision_bytes).hexdigest(),
    }


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                dict(record),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def write_corpus(output_dir: str | Path, corpus: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    samples = list(corpus["samples"])
    supervision = list(corpus["supervision"])
    sample_bytes = _jsonl_bytes(samples)
    supervision_bytes = _jsonl_bytes(supervision)
    if hashlib.sha256(sample_bytes).hexdigest() != corpus["samples_sha256"]:
        raise ValueError("HyperNet SFT samples changed before publication")
    if hashlib.sha256(supervision_bytes).hexdigest() != corpus["supervision_sha256"]:
        raise ValueError("HyperNet SFT supervision changed before publication")
    _atomic_write(root / "samples.jsonl", sample_bytes)
    _atomic_write(root / "supervision.jsonl", supervision_bytes)
    manifest = {
        key: value
        for key, value in corpus.items()
        if key not in {"samples", "supervision"}
    }
    manifest.update(
        {
            "samples_path": "samples.jsonl",
            "samples_bytes": len(sample_bytes),
            "supervision_path": "supervision.jsonl",
            "supervision_bytes": len(supervision_bytes),
        }
    )
    manifest["manifest_identity_sha256"] = semantic_sha256(manifest)
    _atomic_write(
        root / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return manifest


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("HyperNet SFT JSONL record is not an object")
                yield value


def verify_corpus(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != CORPUS_FORMAT or manifest.get("status") != "complete":
        raise ValueError("HyperNet SFT corpus manifest is incomplete")
    identity = str(manifest.get("manifest_identity_sha256", ""))
    body = {key: value for key, value in manifest.items() if key != "manifest_identity_sha256"}
    if semantic_sha256(body) != identity:
        raise ValueError("HyperNet SFT manifest identity changed")
    samples_path = manifest_path.parent / str(manifest["samples_path"])
    supervision_path = manifest_path.parent / str(manifest["supervision_path"])
    if hashlib.sha256(samples_path.read_bytes()).hexdigest() != manifest["samples_sha256"]:
        raise ValueError("HyperNet SFT samples checksum changed")
    if hashlib.sha256(supervision_path.read_bytes()).hexdigest() != manifest[
        "supervision_sha256"
    ]:
        raise ValueError("HyperNet SFT supervision checksum changed")
    samples = list(iter_jsonl(samples_path))
    supervision = list(iter_jsonl(supervision_path))
    if len(samples) != int(manifest["event_count"]) or len(supervision) != len(samples):
        raise ValueError("HyperNet SFT corpus coverage changed")
    if [record["prompt_key"] for record in samples] != [
        record["prompt_key"] for record in supervision
    ]:
        raise ValueError("HyperNet SFT query/supervision join changed")
    return manifest


def load_samples(path: str | Path) -> tuple[dict[str, Any], ...]:
    manifest_path = Path(path).resolve()
    manifest = verify_corpus(manifest_path)
    return tuple(iter_jsonl(manifest_path.parent / str(manifest["samples_path"])))


def regenerate_and_verify(record: Mapping[str, Any]) -> tuple[str, str]:
    input_text, output_text = generate_sample(
        str(record["task_type"]),
        int(record["seed"]),
        n_distractors=2,
        renderer_version=str(record["renderer_version"]),
    )
    if hashlib.sha256(input_text.encode("utf-8")).hexdigest() != record[
        "input_text_sha256"
    ]:
        raise ValueError("HyperNet SFT query renderer changed")
    if hashlib.sha256(output_text.encode("utf-8")).hexdigest() != record[
        "supervision_text_sha256"
    ]:
        raise ValueError("HyperNet SFT supervision renderer changed")
    return input_text, output_text

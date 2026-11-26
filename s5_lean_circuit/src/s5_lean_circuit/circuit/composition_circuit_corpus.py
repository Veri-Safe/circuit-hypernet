"""Immutable 1.5M composition-only prompt corpus for S5 Circuit extraction."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from ..protocol import (
    COMPOSITION_TASKS,
    STAGE_B_LEGACY_RENDERER_VERSION,
    generate_sample,
)


CORPUS_FORMAT = "s5_composition_circuit_corpus_v1"
ROW_FORMAT = "s5_unified_rl_prompt_union_row_v1"
LOCK_FORMAT = "s5_composition_circuit_corpus_lock_v1"
PROTOCOL_ID = "s5_composition_only_circuit_1p5m_v1"
EVENT_COUNT = 1_500_000
SAMPLES_PER_TASK = 300_000
SEED_BASE = 42_000
PREFIX_EVENT_COUNT = 3_125
EXPECTED_PREFIX_SAMPLES_SHA256 = (
    "faf1e1bb384c1138dc221be0832f3aba316bc1679cee2c90127267c584ce126e"
)
EXPECTED_SAMPLES_BYTES = 743_686_810
EXPECTED_SAMPLES_SHA256 = (
    "1c78ae780d882f18739130b863678212328bdec0656767debf2b9f26f31432f9"
)
EXPECTED_PROMPT_KEYS_SHA256 = (
    "ab97480bc28a6c88948bd0569bc45753e34f82bf32648e973908e21a4aefc631"
)


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _SemanticStringListDigest:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.digest.update(b"[")
        self.first = True

    def add(self, value: str) -> None:
        if not self.first:
            self.digest.update(b",")
        self.first = False
        self.digest.update(json.dumps(str(value), ensure_ascii=False).encode("utf-8"))

    def hexdigest(self) -> str:
        value = self.digest.copy()
        value.update(b"]")
        return value.hexdigest()


def _manifest_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    return path if path.name == "manifest.json" else path / "manifest.json"


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_sample_row(
    row: Mapping[str, Any], *, expected_index: int | None = None
) -> dict[str, Any]:
    index = int(row.get("canonical_global_index", -1))
    if expected_index is not None and index != int(expected_index):
        raise ValueError("composition corpus canonical order changed")
    if not 0 <= index < EVENT_COUNT:
        raise ValueError("composition corpus index is outside formal coverage")
    task = str(row.get("task_type", ""))
    if task != COMPOSITION_TASKS[index % len(COMPOSITION_TASKS)]:
        raise ValueError("composition corpus repeat-major task order changed")
    seed = int(row.get("seed", -1))
    if seed != SEED_BASE + index:
        raise ValueError("composition corpus seed order changed")
    input_text = str(row.get("input_text", ""))
    input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
    expected_key = semantic_sha256(
        {
            "format": "s5_paper_rl_prompt_circuit_sample_v1",
            "pool_index": index,
            "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
            "task_type": task,
            "seed": seed,
            "input_text_sha256": input_sha,
        }
    )
    if (
        row.get("format") != ROW_FORMAT
        or row.get("input_text_sha256") != input_sha
        or row.get("prompt_key") != expected_key
        or not isinstance(row.get("supervision_text_sha256"), str)
        or len(str(row["supervision_text_sha256"])) != 64
        or any(
            field in row
            for field in ("output_text", "expected_tactic", "reward", "correct")
        )
    ):
        raise ValueError("composition corpus row identity changed")
    return dict(row)


def iter_jsonl_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank composition row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("composition corpus row is not an object")
            yield value


def load_frozen_metadata(value: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate manifest/lock semantics without reopening the large samples file."""

    path = _manifest_path(value)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    body = {
        key: item
        for key, item in manifest.items()
        if key != "manifest_identity_sha256"
    }
    if (
        manifest.get("format") != CORPUS_FORMAT
        or manifest.get("status") != "complete"
        or manifest.get("protocol_id") != PROTOCOL_ID
        or int(manifest.get("event_count", -1)) != EVENT_COUNT
        or int(manifest.get("samples_per_task", -1)) != SAMPLES_PER_TASK
        or tuple(manifest.get("tasks", ())) != COMPOSITION_TASKS
        or manifest.get("task_counts")
        != {task: SAMPLES_PER_TASK for task in COMPOSITION_TASKS}
        or manifest.get("renderer_version") != STAGE_B_LEGACY_RENDERER_VERSION
        or manifest.get("ordering") != "repeat_then_composition_task"
        or manifest.get("seed_first") != SEED_BASE
        or manifest.get("seed_last") != SEED_BASE + EVENT_COUNT - 1
        or manifest.get("samples_bytes") != EXPECTED_SAMPLES_BYTES
        or manifest.get("samples_sha256") != EXPECTED_SAMPLES_SHA256
        or manifest.get("prompt_keys_sha256") != EXPECTED_PROMPT_KEYS_SHA256
        or semantic_sha256(body) != manifest.get("manifest_identity_sha256")
    ):
        raise ValueError("composition corpus manifest identity changed")
    lock_path = path.with_name("FROZEN.lock.json")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    body = {key: item for key, item in lock.items() if key != "lock_identity_sha256"}
    if (
        lock.get("format") != LOCK_FORMAT
        or lock.get("status") != "frozen"
        or lock.get("protocol_id") != PROTOCOL_ID
        or int(lock.get("event_count", -1)) != EVENT_COUNT
        or lock.get("manifest_identity_sha256")
        != manifest["manifest_identity_sha256"]
        or lock.get("manifest_file_sha256") != sha256_file(path)
        or lock.get("samples_sha256") != manifest["samples_sha256"]
        or semantic_sha256(body) != lock.get("lock_identity_sha256")
    ):
        raise ValueError("composition corpus frozen lock changed")
    return manifest, lock


def verify_corpus(value: str | Path) -> dict[str, Any]:
    path = _manifest_path(value)
    manifest, _lock = load_frozen_metadata(path)
    samples_path = path.parent / str(manifest.get("samples_path", ""))
    if (
        not samples_path.is_file()
        or samples_path.stat().st_size != int(manifest.get("samples_bytes", -1))
        or sha256_file(samples_path) != manifest.get("samples_sha256")
    ):
        raise ValueError("composition corpus samples artifact changed")
    return manifest


def verify_lock(value: str | Path) -> dict[str, Any]:
    _manifest, lock = load_frozen_metadata(value)
    return lock


def load_worker_shard(
    *,
    corpus_manifest_path: str | Path,
    shard_path: str | Path,
    shard_inventory_path: str | Path,
    shard_kind: str,
    worker_id: int,
    worker_count: int,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    """Open one checksum-bound modulo shard after the corpus was verified once."""

    if shard_kind not in {"trace", "persistent"}:
        raise ValueError("composition shard kind must be trace or persistent")
    manifest, lock = load_frozen_metadata(corpus_manifest_path)
    inventory_path = Path(shard_inventory_path).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    identity = {
        key: item
        for key, item in inventory.items()
        if key not in {"status", "shard_inventory_identity_sha256"}
    }
    if (
        inventory.get("format") != "s5_composition_circuit_worker_shards_v1"
        or inventory.get("status") != "complete"
        or inventory.get("corpus_manifest_identity_sha256")
        != manifest["manifest_identity_sha256"]
        or inventory.get("corpus_lock_identity_sha256")
        != lock["lock_identity_sha256"]
        or semantic_sha256(identity)
        != inventory.get("shard_inventory_identity_sha256")
    ):
        raise ValueError("composition shard inventory identity changed")
    records = inventory[f"{shard_kind}_shards"]
    if len(records) != int(worker_count):
        raise ValueError("composition shard worker coverage changed")
    record = records[int(worker_id)]
    source = Path(shard_path).resolve()
    expected_path = (inventory_path.parent / str(record["path"])).resolve()
    if (
        int(record.get("worker_id", -1)) != int(worker_id)
        or int(record.get("worker_count", -1)) != int(worker_count)
        or source != expected_path
        or not source.is_file()
        or source.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(source) != record.get("sha256")
    ):
        raise ValueError("composition worker shard artifact changed")
    rows = tuple(iter_jsonl_rows(source))
    if len(rows) != int(record["records"]):
        raise ValueError("composition worker shard row count changed")
    for row in rows:
        validated = validate_sample_row(row)
        if int(validated["canonical_global_index"]) % int(worker_count) != int(
            worker_id
        ):
            raise ValueError("composition worker shard modulo routing changed")
    return manifest, rows


def load_samples(
    value: str | Path,
    *,
    _verified_manifest: Mapping[str, Any] | None = None,
) -> Iterable[dict[str, Any]]:
    path = _manifest_path(value)
    manifest = (
        dict(_verified_manifest)
        if _verified_manifest is not None
        else verify_corpus(path)
    )
    samples_path = path.parent / str(manifest["samples_path"])
    for index, row in enumerate(iter_jsonl_rows(samples_path)):
        yield validate_sample_row(row, expected_index=index)


def load_stored_prompt(record: Mapping[str, Any]) -> tuple[str, str]:
    row = validate_sample_row(record)
    return str(row["input_text"]), ""


def materialize_corpus(output_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("composition corpus materialization requires PYTHONHASHSEED=42")
    root = Path(output_dir).resolve()
    manifest_path = root / "manifest.json"
    lock_path = root / "FROZEN.lock.json"
    samples_path = root / "samples.jsonl"
    if manifest_path.is_file() or lock_path.is_file() or samples_path.is_file():
        if not (manifest_path.is_file() and lock_path.is_file() and samples_path.is_file()):
            raise FileExistsError("composition corpus has incomplete published artifacts")
        return verify_corpus(manifest_path), verify_lock(manifest_path)

    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f".samples.jsonl.tmp.{os.getpid()}"
    samples_digest = hashlib.sha256()
    prefix_digest = hashlib.sha256()
    prompt_digest = _SemanticStringListDigest()
    task_counts: Counter[str] = Counter()
    seen_input_sha256: set[str] = set()
    state = random.getstate()
    random.seed(42)
    try:
        with temporary.open("wb") as handle:
            for repeat in range(SAMPLES_PER_TASK):
                for task in COMPOSITION_TASKS:
                    index = repeat * len(COMPOSITION_TASKS) + COMPOSITION_TASKS.index(task)
                    seed = SEED_BASE + index
                    input_text, output_text = generate_sample(
                        task,
                        seed,
                        n_distractors=2,
                        renderer_version=STAGE_B_LEGACY_RENDERER_VERSION,
                        surface_family="narrow",
                    )
                    input_sha = hashlib.sha256(input_text.encode("utf-8")).hexdigest()
                    if input_sha in seen_input_sha256:
                        raise ValueError(
                            f"composition corpus prompt text duplicated at index {index}"
                        )
                    seen_input_sha256.add(input_sha)
                    prompt_key = semantic_sha256(
                        {
                            "format": "s5_paper_rl_prompt_circuit_sample_v1",
                            "pool_index": index,
                            "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
                            "task_type": task,
                            "seed": seed,
                            "input_text_sha256": input_sha,
                        }
                    )
                    row = {
                        "format": ROW_FORMAT,
                        "canonical_global_index": index,
                        "prompt_key": prompt_key,
                        "task_type": task,
                        "seed": seed,
                        "input_text": input_text,
                        "input_text_sha256": input_sha,
                        "supervision_text_sha256": hashlib.sha256(
                            output_text.encode("utf-8")
                        ).hexdigest(),
                    }
                    encoded = (
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        )
                        + "\n"
                    ).encode("utf-8")
                    handle.write(encoded)
                    samples_digest.update(encoded)
                    if index < PREFIX_EVENT_COUNT:
                        prefix_digest.update(encoded)
                    prompt_digest.add(prompt_key)
                    task_counts[task] += 1
                    if (index + 1) % 100_000 == 0:
                        print(
                            json.dumps(
                                {
                                    "status": "generating",
                                    "events": index + 1,
                                    "target": EVENT_COUNT,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        random.setstate(state)

    if (
        len(seen_input_sha256) != EVENT_COUNT
        or task_counts != Counter({task: SAMPLES_PER_TASK for task in COMPOSITION_TASKS})
        or prefix_digest.hexdigest() != EXPECTED_PREFIX_SAMPLES_SHA256
        or temporary.stat().st_size != EXPECTED_SAMPLES_BYTES
        or samples_digest.hexdigest() != EXPECTED_SAMPLES_SHA256
        or prompt_digest.hexdigest() != EXPECTED_PROMPT_KEYS_SHA256
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("composition corpus final coverage/prefix gate failed")
    os.replace(temporary, samples_path)
    manifest = {
        "format": CORPUS_FORMAT,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "role": "composition_only_circuit_source",
        "event_count": EVENT_COUNT,
        "samples_per_task": SAMPLES_PER_TASK,
        "tasks": list(COMPOSITION_TASKS),
        "task_counts": {task: SAMPLES_PER_TASK for task in COMPOSITION_TASKS},
        "seed_first": SEED_BASE,
        "seed_last": SEED_BASE + EVENT_COUNT - 1,
        "ordering": "repeat_then_composition_task",
        "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
        "surface_family": "narrow",
        "n_distractors": 2,
        "unique_prompt_texts": EVENT_COUNT,
        "prompt_keys_sha256": EXPECTED_PROMPT_KEYS_SHA256,
        "samples_path": samples_path.name,
        "samples_bytes": EXPECTED_SAMPLES_BYTES,
        "samples_sha256": EXPECTED_SAMPLES_SHA256,
        "prefix_event_count": PREFIX_EVENT_COUNT,
        "prefix_samples_sha256": prefix_digest.hexdigest(),
        "prefix_compatibility": "paper_rl_and_paired_loco_stream_v1",
        "forbidden_fields": ["output_text", "expected_tactic", "reward", "correct"],
    }
    manifest["manifest_identity_sha256"] = semantic_sha256(manifest)
    _atomic_json(manifest_path, manifest)
    lock = {
        "format": LOCK_FORMAT,
        "status": "frozen",
        "protocol_id": PROTOCOL_ID,
        "event_count": EVENT_COUNT,
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "samples_sha256": manifest["samples_sha256"],
        "generator_code_sha256": sha256_file(Path(__file__).resolve()),
        "mutation_policy": "immutable_refuse_overwrite",
    }
    lock["lock_identity_sha256"] = semantic_sha256(lock)
    _atomic_json(lock_path, lock)
    return verify_corpus(manifest_path), verify_lock(manifest_path)


__all__ = [
    "CORPUS_FORMAT",
    "EVENT_COUNT",
    "LOCK_FORMAT",
    "PROTOCOL_ID",
    "SAMPLES_PER_TASK",
    "iter_jsonl_rows",
    "load_frozen_metadata",
    "load_samples",
    "load_stored_prompt",
    "load_worker_shard",
    "materialize_corpus",
    "semantic_sha256",
    "validate_sample_row",
    "verify_corpus",
    "verify_lock",
]

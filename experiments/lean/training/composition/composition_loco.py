"""Data, fold, and evaluation contracts for composition leave-one-out SFT.

This module is intentionally independent of the historical formal 1.2M cache.
The frozen composition Circuit archive contains 1.5M rows (300k per task),
while every leave-one-out run consumes a deterministic 1.2M view of that
archive.  Keeping the full archive cache and fold index manifests separate
prevents five copies of the 2.4GB Vector397 tensor from being materialized.

The module is also the single source of truth for the data split.  Trainers and
evaluators must consume the manifests produced here instead of filtering task
names ad hoc.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file
from torch import Tensor

from .circuit.composition_circuit_corpus import (
    EVENT_COUNT as COMPOSITION_EVENT_COUNT,
    SAMPLES_PER_TASK as COMPOSITION_SAMPLES_PER_TASK,
    iter_jsonl_rows,
    validate_sample_row,
    verify_corpus,
)
from .circuit.descriptor_v2_397_artifact import read_descriptor397_archive
from .hypernet.circuit_normalizer import FixedLog1pVector397, read_vector397_normalizer
from .hypernet.vector397_fingerprint import load_fingerprint_normalizer
from .hypernet.data import StrictDataError, _response_start_from_token_proto
from .protocol import (
    ALL_TASKS,
    ATOMIC_TASKS as _ATOMIC_TASKS,
    COMPOSITION_TASKS,
    EvaluationSample,
    STAGE_B_LEGACY_RENDERER_VERSION,
    create_evaluation_dataset,
    expected_first_tactic,
    generate_sample,
)


PROTOCOL_ID = "s5_composition_loco_sft_v1"
STAGE_A_BASE_IDENTITY = {
    "checkpoint_sha256": "ee2a7efdf77262ee1fbdf7d2080f930c2bedc27e45b7712537fba964ee05931c",
    "model_state_sha256": "bd0277b72e7f17f864bd9e8064e1dff989a81254c5353550c6f912944ecf1afd",
    "config_sha256": "d87e874cdee838e7d07b99631d06ee64ec3466d24012a08af75d032eddf7169e",
    "step": 38_000,
}
CACHE_FORMAT = "s5_composition_loco_vector397_cache_v1"
FOLD_FORMAT = "s5_composition_loco_fold_index_v1"
EVAL_CORPUS_FORMAT = "s5_composition_loco_eval_corpus_v1"
EVAL_BANK_FORMAT = "s5_composition_loco_eval_bank_v1"
N_CTX = 128
GLOBAL_BATCH_SIZE = 512
TOTAL_STEPS = 38_000
TOTAL_PADDED_TOKENS = TOTAL_STEPS * GLOBAL_BATCH_SIZE * N_CTX
VALIDATION_SAMPLES_PER_TASK = 500
FINAL_SAMPLES_PER_TASK = 2_000
ATOMIC_SAMPLES_PER_TASK = 500
VALIDATION_STEPS = (0, 1_000, 2_000, 4_000, 8_000, 16_000, 24_000, 32_000, 38_000)
TASK_TO_ID = {task: index for index, task in enumerate(ALL_TASKS)}
ATOMIC_TASKS = _ATOMIC_TASKS
COMPOSITION_TASK_TO_ID = {task: index for index, task in enumerate(COMPOSITION_TASKS)}
ROUTES = ("direct_sft", "query_only", "raw_weight", "circuit_a")
FOLD_TASKS = {
    heldout: tuple(task for task in COMPOSITION_TASKS if task != heldout)
    for heldout in COMPOSITION_TASKS
}


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


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Mapping[str, Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(dict(value), temporary)
    os.replace(temporary, path)


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _tokenizer_encode(tokenizer: Any, text: str) -> tuple[int, ...]:
    try:
        values = tokenizer.encode(text)
    except TypeError:
        values = tokenizer.encode(text, add_bos=False, add_eos=False)
    values = tuple(int(value) for value in values)
    if not values:
        raise StrictDataError("composition tokenizer returned an empty sequence")
    if any(value < 0 or value >= 2048 for value in values):
        raise StrictDataError("composition tokenizer id is outside the frozen vocabulary")
    return values


def proof_group(output_text: str) -> str:
    """Classify the *gold generator structure* for the diagnostic report.

    The grouping is deliberately based on the canonical tactic, never on a
    model prediction.  A constructor with an inner constructor is ``nested``;
    the two ``Or.inr h`` branches are the simple alternatives in the public
    composition families.
    """

    tactic = expected_first_tactic(str(output_text)).strip()
    if not tactic:
        raise ValueError("gold output has no first tactic")
    return "nested" if re.search(r"\((?:And|Or)\.", tactic) else "simple"


def task_fold(heldout_task: str) -> tuple[str, ...]:
    heldout = str(heldout_task)
    if heldout not in COMPOSITION_TASK_TO_ID:
        raise ValueError(f"unknown composition held-out task: {heldout}")
    return FOLD_TASKS[heldout]


def _hex_rows(value: Tensor) -> list[str]:
    if value.dtype != torch.uint8 or value.ndim != 2 or value.shape[1] != 32:
        raise StrictDataError("identity tensor must be uint8[N,32]")
    return [bytes(row.tolist()).hex() for row in value]


def _iter_vector_archive_rows(
    manifest_path: str | Path,
) -> Iterator[tuple[int, str, str, np.ndarray]]:
    """Yield canonical ``(index,event_key,payload_id,raw_vector)`` rows."""

    source = Path(manifest_path).resolve()
    archive = read_descriptor397_archive(source, full_tensor_verify=False)
    if int(archive.get("events", -1)) != COMPOSITION_EVENT_COUNT:
        raise StrictDataError("composition Vector397 archive must contain 1.5M events")
    cursor = 0
    for shard in archive["shards"]:
        path = Path(str(shard["path"]))
        if not path.is_absolute():
            path = source.parent / path
        tensors = load_file(str(path.resolve()), device="cpu")
        count = int(shard["records"])
        if set(tensors) != {
            "global_index",
            "event_key",
            "payload_artifact_id",
            "descriptor_raw",
        }:
            raise StrictDataError("composition Vector397 tensor schema changed")
        if tensors["global_index"].tolist() != list(range(cursor, cursor + count)):
            raise StrictDataError("composition Vector397 canonical order changed")
        raw = tensors["descriptor_raw"]
        if raw.dtype != torch.float32 or raw.shape != (count, 397):
            raise StrictDataError("composition Vector397 raw tensor shape changed")
        if not bool(torch.isfinite(raw).all()):
            raise StrictDataError("composition Vector397 raw tensor is non-finite")
        event_keys = _hex_rows(tensors["event_key"])
        payload_ids = _hex_rows(tensors["payload_artifact_id"])
        for offset in range(count):
            index = cursor + offset
            yield index, event_keys[offset], payload_ids[offset], np.asarray(
                raw[offset].numpy(), dtype=np.float32
            )
        cursor += count
    if cursor != COMPOSITION_EVENT_COUNT:
        raise StrictDataError("composition Vector397 coverage is incomplete")


def _read_event_join_rows(path: str | Path, *, expected_count: int) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    rows: list[dict[str, Any] | None] = [None] * int(expected_count)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise StrictDataError(f"blank event-join row at line {line_number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise StrictDataError("event-join row is not an object")
            raw_index = value.get("global_phase_index", -1)
            index = int(raw_index)
            key = str(value.get("event_key", ""))
            if (
                isinstance(raw_index, bool)
                or index != raw_index
                or not 0 <= index < int(expected_count)
                or rows[index] is not None
            ):
                raise StrictDataError("event-join canonical coverage changed")
            if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
                raise StrictDataError("event-join event_key is not a SHA256")
            row = dict(value)
            row["event_key"] = key
            rows[index] = row
    if any(value is None for value in rows):
        raise StrictDataError("event-join does not cover all composition rows")
    return [dict(value) for value in rows if value is not None]


def _read_event_keys(path: str | Path, *, expected_count: int) -> list[str]:
    return [str(row["event_key"]) for row in _read_event_join_rows(path, expected_count=expected_count)]


def _validate_payload(payload: Mapping[str, Tensor], *, count: int, n_ctx: int) -> None:
    expected = {
        "query_ids": (torch.int16, (count, n_ctx)),
        "query_lengths": (torch.int16, (count,)),
        "input_ids": (torch.int16, (count, n_ctx)),
        "input_lengths": (torch.int16, (count,)),
        "response_start": (torch.int16, (count,)),
        "vectors": (torch.float32, (count, 397)),
        "task_ids": (torch.uint8, (count,)),
        "event_key": (torch.uint8, (count, 32)),
        "payload_artifact_id": (torch.uint8, (count, 32)),
    }
    if set(payload) != set(expected):
        raise StrictDataError("composition cache tensor schema changed")
    for name, (dtype, shape) in expected.items():
        value = payload[name]
        if value.dtype != dtype or tuple(value.shape) != shape:
            raise StrictDataError(f"composition cache {name} must be {dtype}{list(shape)}")
    query_lengths = payload["query_lengths"].long()
    input_lengths = payload["input_lengths"].long()
    response_start = payload["response_start"].long()
    if bool(((query_lengths <= 0) | (query_lengths > n_ctx)).any()):
        raise StrictDataError("composition cache query lengths are invalid")
    if bool(((input_lengths <= 1) | (input_lengths > n_ctx)).any()):
        raise StrictDataError("composition cache input lengths are invalid")
    if bool(((response_start <= 0) | (response_start >= input_lengths)).any()):
        raise StrictDataError("composition cache response boundaries are invalid")
    if bool((payload["task_ids"] >= len(COMPOSITION_TASKS)).any()):
        raise StrictDataError("composition cache task ids are invalid")
    if not bool(torch.isfinite(payload["vectors"]).all()):
        raise StrictDataError("composition cache vectors are non-finite")


def _cache_source_identity(archive_base: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the source identities shared by cache writers and validators.

    ``expected_source_base_identity`` describes the Base checkpoint required by
    the S5 protocol.  ``archive_source_base_identity`` records what the
    Vector397 archive actually declares; it may be ``None`` only for an older
    archive that carries no Base identity.  Keeping these fields in one helper
    prevents a newly written cache from being rejected by its own validator.
    """

    if archive_base is not None and not isinstance(archive_base, Mapping):
        raise StrictDataError("composition Vector397 archive Base identity is malformed")
    return {
        "expected_source_base_identity": dict(STAGE_A_BASE_IDENTITY),
        "archive_source_base_identity": (
            None if archive_base is None else dict(archive_base)
        ),
    }


def build_composition_cache(
    *,
    corpus_manifest: str | Path,
    vector_manifest: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    event_join: str | Path | None = None,
    n_ctx: int = N_CTX,
) -> dict[str, Any]:
    """Materialize one shared, checksum-bound 1.5M token/vector cache."""

    if int(n_ctx) != N_CTX:
        raise ValueError("composition LOCO fixes n_ctx=128")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("composition LOCO cache requires PYTHONHASHSEED=42")
    corpus_path = Path(corpus_manifest).resolve()
    vector_path = Path(vector_manifest).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    destination = Path(output_path).resolve()
    if not tokenizer_file.is_file():
        raise FileNotFoundError(f"composition tokenizer is missing: {tokenizer_file}")
    corpus = verify_corpus(corpus_path)
    archive = read_descriptor397_archive(vector_path, full_tensor_verify=False)
    if int(archive.get("events", -1)) != COMPOSITION_EVENT_COUNT:
        raise StrictDataError("composition vector archive coverage changed")
    archive_base = archive.get("base_identity") or archive.get("source_base_identity")
    if archive_base is not None and not isinstance(archive_base, Mapping):
        raise StrictDataError("composition Vector397 archive Base identity is malformed")
    if archive_base is not None and any(
        archive_base.get(key) != value for key, value in STAGE_A_BASE_IDENTITY.items()
    ):
        raise StrictDataError("composition Vector397 archive was extracted from a different Base")
    samples_path = corpus_path.parent / str(corpus["samples_path"])
    if event_join is None:
        raise ValueError("composition cache publication requires --event-join")
    event_join_rows = _read_event_join_rows(event_join, expected_count=COMPOSITION_EVENT_COUNT)
    event_keys = [str(row["event_key"]) for row in event_join_rows]
    if destination.is_file() or _sidecar(destination).is_file():
        if not destination.is_file() or not _sidecar(destination).is_file():
            raise StrictDataError("composition cache payload/sidecar are incomplete")
        existing = validate_composition_cache(destination)
        expected_identity = {
            "corpus_manifest_sha256": sha256_file(corpus_path),
            "corpus_manifest_identity_sha256": corpus["manifest_identity_sha256"],
            "vector_manifest_sha256": sha256_file(vector_path),
            "vector_archive_identity_sha256": archive["archive_identity_sha256"],
            "tokenizer_sha256": sha256_file(tokenizer_file),
            "event_join_sha256": sha256_file(event_join),
            **_cache_source_identity(archive_base),
        }
        if any(existing.get(key) != value for key, value in expected_identity.items()):
            raise StrictDataError("existing composition cache is bound to different source artifacts")
        return existing

    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    query_ids = torch.zeros((COMPOSITION_EVENT_COUNT, n_ctx), dtype=torch.int16)
    query_lengths = torch.empty(COMPOSITION_EVENT_COUNT, dtype=torch.int16)
    input_ids = torch.zeros((COMPOSITION_EVENT_COUNT, n_ctx), dtype=torch.int16)
    input_lengths = torch.empty(COMPOSITION_EVENT_COUNT, dtype=torch.int16)
    response_start = torch.empty(COMPOSITION_EVENT_COUNT, dtype=torch.int16)
    vectors = torch.empty((COMPOSITION_EVENT_COUNT, 397), dtype=torch.float32)
    task_ids = torch.empty(COMPOSITION_EVENT_COUNT, dtype=torch.uint8)
    event_tensor = torch.empty((COMPOSITION_EVENT_COUNT, 32), dtype=torch.uint8)
    payload_tensor = torch.empty((COMPOSITION_EVENT_COUNT, 32), dtype=torch.uint8)
    transform = FixedLog1pVector397()
    sample_iterator = iter(iter_jsonl_rows(samples_path))
    count = 0
    task_counts: Counter[str] = Counter()
    # The legacy renderer uses Python's global RNG for premise order.  Replay
    # the exact corpus publication stream rather than calling it with an
    # arbitrary process-global state.
    rng_state = random.getstate()
    random.seed(42)
    for index, event_key, payload_id, raw_vector in _iter_vector_archive_rows(vector_path):
        if index != count:
            raise StrictDataError("composition cache source index changed")
        try:
            sample = validate_sample_row(next(sample_iterator), expected_index=index)
        except StopIteration as error:
            raise StrictDataError("composition corpus ended before Vector397") from error
        task = str(sample["task_type"])
        seed = int(sample["seed"])
        generated_input, generated_output = generate_sample(
            task,
            seed,
            n_distractors=2,
            renderer_version=STAGE_B_LEGACY_RENDERER_VERSION,
            surface_family="narrow",
        )
        if generated_input != str(sample["input_text"]):
            raise StrictDataError(f"composition renderer text changed at row {index}")
        if hashlib.sha256(generated_input.encode("utf-8")).hexdigest() != sample[
            "input_text_sha256"
        ]:
            raise StrictDataError(f"composition input hash changed at row {index}")
        if hashlib.sha256(generated_output.encode("utf-8")).hexdigest() != sample[
            "supervision_text_sha256"
        ]:
            raise StrictDataError(f"composition supervision hash changed at row {index}")
        if event_keys is not None and event_key != event_keys[index]:
            raise StrictDataError(f"composition Vector397/event join changed at row {index}")
        event_row = event_join_rows[index]
        if (
            event_row.get("prompt_key") is not None
            and event_row.get("prompt_key") != sample["prompt_key"]
        ) or (
            event_row.get("task_type") is not None
            and event_row.get("task_type") != task
        ) or (
            event_row.get("seed") is not None
            and int(event_row["seed"]) != seed
        ):
            raise StrictDataError(f"composition event-join prompt identity changed at row {index}")
        full_text = generated_input + generated_output
        query = _tokenizer_encode(tokenizer, generated_input)
        full = _tokenizer_encode(tokenizer, full_text)
        boundary = _response_start_from_token_proto(tokenizer, full_text, generated_input)
        if len(query) > n_ctx or len(full) > n_ctx:
            raise StrictDataError(
                f"composition row {index} exceeds n_ctx={n_ctx}: "
                f"query={len(query)} full={len(full)}"
            )
        query_ids[index, : len(query)] = torch.tensor(query, dtype=torch.int16)
        input_ids[index, : len(full)] = torch.tensor(full, dtype=torch.int16)
        query_lengths[index] = len(query)
        input_lengths[index] = len(full)
        response_start[index] = boundary
        vectors[index] = torch.from_numpy(transform.transform(raw_vector))
        task_ids[index] = COMPOSITION_TASK_TO_ID[task]
        event_tensor[index] = torch.tensor(list(bytes.fromhex(event_key)), dtype=torch.uint8)
        payload_tensor[index] = torch.tensor(list(bytes.fromhex(payload_id)), dtype=torch.uint8)
        task_counts[task] += 1
        count += 1
    try:
        next(sample_iterator)
    except StopIteration:
        pass
    else:
        raise StrictDataError("composition corpus contains rows beyond Vector397 coverage")
    random.setstate(rng_state)
    if count != COMPOSITION_EVENT_COUNT or task_counts != Counter(
        {task: COMPOSITION_SAMPLES_PER_TASK for task in COMPOSITION_TASKS}
    ):
        raise StrictDataError("composition cache task balance changed")
    payload = {
        "query_ids": query_ids,
        "query_lengths": query_lengths,
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "response_start": response_start,
        "vectors": vectors,
        "task_ids": task_ids,
        "event_key": event_tensor,
        "payload_artifact_id": payload_tensor,
    }
    _validate_payload(payload, count=COMPOSITION_EVENT_COUNT, n_ctx=n_ctx)
    _atomic_torch(destination, payload)
    identity = {
        "format": CACHE_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "event_count": COMPOSITION_EVENT_COUNT,
        "tasks": list(COMPOSITION_TASKS),
        "task_counts": dict(task_counts),
        "n_ctx": int(n_ctx),
        "corpus_manifest_sha256": sha256_file(corpus_path),
        "corpus_manifest_identity_sha256": corpus["manifest_identity_sha256"],
        "vector_manifest_sha256": sha256_file(vector_path),
        "vector_archive_identity_sha256": archive["archive_identity_sha256"],
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "event_join_sha256": None if event_join is None else sha256_file(event_join),
        **_cache_source_identity(archive_base),
        "input_treatment": transform.metadata(),
        "input_treatment_sha256": transform.artifact_sha256,
        "payload_schema": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape)}
            for name, value in payload.items()
        },
    }
    record = {
        **identity,
        "status": "complete",
        "payload_path": destination.name,
        "payload_sha256": sha256_file(destination),
        "payload_bytes": destination.stat().st_size,
        "identity_sha256": semantic_sha256(identity),
    }
    _atomic_json(_sidecar(destination), record)
    return validate_composition_cache(destination)


def validate_composition_cache(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    sidecar = _sidecar(source)
    if not source.is_file() or not sidecar.is_file():
        raise StrictDataError("composition cache payload/sidecar are missing")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    identity = {
        key: value
        for key, value in record.items()
        if key
        not in {"status", "payload_path", "payload_sha256", "payload_bytes", "identity_sha256"}
    }
    if (
        record.get("format") != CACHE_FORMAT
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("status") != "complete"
        or record.get("payload_path") != source.name
        or int(record.get("payload_bytes", -1)) != source.stat().st_size
        or record.get("payload_sha256") != sha256_file(source)
        or record.get("identity_sha256") != semantic_sha256(identity)
        or int(record.get("event_count", -1)) != COMPOSITION_EVENT_COUNT
        or record.get("tasks") != list(COMPOSITION_TASKS)
        or record.get("expected_source_base_identity") != STAGE_A_BASE_IDENTITY
        or "archive_source_base_identity" not in record
        or (
            record.get("archive_source_base_identity") is not None
            and not isinstance(record.get("archive_source_base_identity"), Mapping)
        )
        or not isinstance(record.get("event_join_sha256"), str)
        or len(record.get("event_join_sha256", "")) != 64
    ):
        raise StrictDataError("composition cache identity changed")
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    _validate_payload(payload, count=COMPOSITION_EVENT_COUNT, n_ctx=int(record["n_ctx"]))
    return record


def write_fold_indices(
    *,
    cache_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write the five canonical 1.2M leave-one-out index manifests."""

    cache_file = Path(cache_path).resolve()
    cache_record = validate_composition_cache(cache_file)
    payload = torch.load(cache_file, map_location="cpu", weights_only=True, mmap=True)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    task_ids = payload["task_ids"].long()
    for heldout in COMPOSITION_TASKS:
        heldout_id = COMPOSITION_TASK_TO_ID[heldout]
        indices = torch.nonzero(task_ids != heldout_id, as_tuple=False).flatten().long()
        expected_tasks = task_fold(heldout)
        counts = Counter(COMPOSITION_TASKS[int(task_ids[index])] for index in indices.tolist())
        if len(indices) != 1_200_000 or counts != Counter({task: 300_000 for task in expected_tasks}):
            raise StrictDataError(f"composition fold coverage changed for {heldout}")
        fold_name = f"holdout_{heldout}"
        index_path = output / f"{fold_name}_indices.pt"
        _atomic_torch(index_path, {"indices": indices})
        digest = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
        identity = {
            "format": FOLD_FORMAT,
            "protocol_id": PROTOCOL_ID,
            "fold_id": fold_name,
            "heldout_task": heldout,
            "train_tasks": list(expected_tasks),
            "event_count": int(indices.numel()),
            "task_counts": dict(counts),
            "cache_identity_sha256": cache_record["identity_sha256"],
            "index_sha256": digest,
            "index_path": index_path.name,
        }
        record = {**identity, "status": "complete", "identity_sha256": semantic_sha256(identity)}
        _atomic_json(_sidecar(index_path), record)
        result[heldout] = record
    return result


class CompositionFoldDataset:
    """Read-only fold view over the shared full composition cache."""

    def __init__(
        self,
        cache_path: str | Path,
        fold_index_path: str | Path,
        *,
        expected_heldout_task: str | None = None,
        expected_train_tasks: Sequence[str] | None = None,
        fingerprint_manifest: str | Path | None = None,
        fingerprint_normalizer: str | Path | None = None,
        fingerprint_scope: str | None = None,
        fingerprint_normalizer_artifact_sha256: str | None = None,
    ) -> None:
        self.cache_path = Path(cache_path).resolve()
        self.fold_index_path = Path(fold_index_path).resolve()
        self.cache_metadata = validate_composition_cache(self.cache_path)
        sidecar = _sidecar(self.fold_index_path)
        if not self.fold_index_path.is_file() or not sidecar.is_file():
            raise StrictDataError("composition fold index/sidecar is missing")
        self.fold_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        identity = {
            key: value
            for key, value in self.fold_metadata.items()
            if key not in {"status", "identity_sha256"}
        }
        if (
            self.fold_metadata.get("format") != FOLD_FORMAT
            or self.fold_metadata.get("status") != "complete"
            or self.fold_metadata.get("identity_sha256") != semantic_sha256(identity)
            or self.fold_metadata.get("cache_identity_sha256")
            != self.cache_metadata["identity_sha256"]
        ):
            raise StrictDataError("composition fold identity changed")
        if expected_heldout_task is not None and self.fold_metadata.get("heldout_task") != str(expected_heldout_task):
            raise StrictDataError("composition fold held-out task differs from the run config")
        if expected_train_tasks is not None and self.fold_metadata.get("train_tasks") != [str(task) for task in expected_train_tasks]:
            raise StrictDataError("composition fold train task order differs from the run config")
        indices_payload = torch.load(
            self.fold_index_path, map_location="cpu", weights_only=True, mmap=True
        )
        if set(indices_payload) != {"indices"}:
            raise StrictDataError("composition fold index tensor schema changed")
        self.indices = indices_payload["indices"].long()
        if self.indices.numel() != int(self.fold_metadata["event_count"]):
            raise StrictDataError("composition fold index cardinality changed")
        if bool((self.indices < 0).any()) or bool((self.indices >= COMPOSITION_EVENT_COUNT).any()):
            raise StrictDataError("composition fold index range changed")
        if self.indices.unique().numel() != self.indices.numel():
            raise StrictDataError("composition fold index contains duplicates")
        if hashlib.sha256(self.indices.numpy().tobytes()).hexdigest() != self.fold_metadata[
            "index_sha256"
        ]:
            raise StrictDataError("composition fold index checksum changed")
        self.payload = torch.load(
            self.cache_path, map_location="cpu", weights_only=True, mmap=True
        )
        heldout_name = str(self.fold_metadata["heldout_task"])
        train_names = tuple(str(task) for task in self.fold_metadata["train_tasks"])
        actual_counts = Counter(
            COMPOSITION_TASKS[int(task_id)]
            for task_id in self.payload["task_ids"].index_select(0, self.indices).tolist()
        )
        expected_counts = Counter({task: 300_000 for task in train_names})
        if actual_counts != expected_counts or actual_counts.get(heldout_name, 0) != 0:
            raise StrictDataError("composition fold task membership changed")
        self.n_ctx = int(self.cache_metadata["n_ctx"])
        self.fingerprint_manifest = None
        self.fingerprint_scope = None
        self.fingerprint_normalizer = None
        self.fingerprint_normalizer_metadata: Mapping[str, Any] | None = None
        if (fingerprint_manifest is None) != (fingerprint_normalizer is None):
            raise StrictDataError(
                "fingerprint_manifest and fingerprint_normalizer must be supplied together"
            )
        if fingerprint_manifest is not None:
            scope = str(fingerprint_scope or f"loco_holdout_{heldout_name}")
            expected_scope = f"loco_holdout_{heldout_name}"
            if scope != expected_scope:
                raise StrictDataError(
                    "composition z-score training must use the matching LOCO normalizer"
                )
            normalizer, metadata, manifest = load_fingerprint_normalizer(
                fingerprint_manifest,
                scope,
                expected_source_cache_identity_sha256=str(
                    self.cache_metadata["identity_sha256"]
                ),
                expected_event_count=COMPOSITION_EVENT_COUNT,
                expected_heldout_task=heldout_name,
            )
            manifest_path = Path(fingerprint_manifest).resolve()
            record_path = (
                manifest_path.parent / str(manifest["normalizers"][scope]["path"])
            ).resolve()
            normalizer_path = Path(fingerprint_normalizer).resolve()
            if normalizer_path != record_path:
                raise StrictDataError(
                    "fingerprint_normalizer does not match the manifest scope"
                )
            if (
                fingerprint_normalizer_artifact_sha256 is not None
                and metadata.get("artifact_sha256")
                != fingerprint_normalizer_artifact_sha256
            ):
                raise StrictDataError("fingerprint normalizer artifact identity changed")
            self.fingerprint_manifest = manifest_path
            self.fingerprint_scope = scope
            self.fingerprint_normalizer = normalizer
            self.fingerprint_normalizer_metadata = dict(metadata)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def collate_indices(
        self,
        indices: Tensor | Sequence[int],
        *,
        device: torch.device | str,
        frozen_base_ce: Tensor | None = None,
    ) -> dict[str, Tensor]:
        local = torch.as_tensor(indices, dtype=torch.long, device="cpu").reshape(-1)
        if local.numel() == 0 or bool(((local < 0) | (local >= len(self))).any()):
            raise IndexError("composition fold batch indices are invalid")
        canonical = self.indices.index_select(0, local)
        query_lengths = self.payload["query_lengths"].index_select(0, canonical).long()
        input_lengths = self.payload["input_lengths"].index_select(0, canonical).long()
        query_width = int(query_lengths.max().item())
        input_width = int(input_lengths.max().item())
        query = self.payload["query_ids"].index_select(0, canonical)[:, :query_width].long()
        full = self.payload["input_ids"].index_select(0, canonical)[:, :input_width].long()
        query_mask = torch.arange(query_width)[None, :] < query_lengths[:, None]
        input_mask = torch.arange(input_width)[None, :] < input_lengths[:, None]
        labels = full.clone()
        labels.masked_fill_(~input_mask, -100)
        vector = self.payload["vectors"].index_select(0, canonical)
        if self.fingerprint_normalizer is not None:
            normalized = self.fingerprint_normalizer.transform_fixed_log1p_batch(
                vector.numpy()
            ).astype(np.float32, copy=False)
            vector = torch.from_numpy(np.ascontiguousarray(normalized))
        if frozen_base_ce is None:
            base_ce = torch.zeros(local.numel(), dtype=torch.float32)
        else:
            base_ce = frozen_base_ce.detach().cpu().float().reshape(-1)
            if base_ce.numel() != local.numel():
                raise ValueError("composition frozen_base_ce batch shape changed")
        return {
            "local_indices": local.to(device),
            "global_indices": canonical.to(device),
            "query_ids": query.to(device, non_blocking=True),
            "query_mask": query_mask.to(device, non_blocking=True),
            "input_ids": full.to(device, non_blocking=True),
            "input_mask": input_mask.to(device, non_blocking=True),
            "labels": labels.to(device, non_blocking=True),
            "vector397": vector.to(device, non_blocking=True),
            "task_ids": self.payload["task_ids"].index_select(0, canonical).to(
                device, non_blocking=True
            ),
            "frozen_base_ce": base_ce.to(device, non_blocking=True),
        }


def _eval_corpus_rows(path: Path) -> tuple[dict[str, Any], tuple[EvaluationSample, ...]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != EVAL_CORPUS_FORMAT or manifest.get("status") != "complete":
        raise StrictDataError("composition evaluation corpus is incomplete")
    body = {key: value for key, value in manifest.items() if key != "manifest_identity_sha256"}
    if manifest.get("manifest_identity_sha256") != semantic_sha256(body):
        raise StrictDataError("composition evaluation corpus identity changed")
    samples_path = path.parent / str(manifest["samples_path"])
    if (
        not samples_path.is_file()
        or manifest.get("samples_sha256") != sha256_file(samples_path)
        or int(manifest.get("samples_bytes", -1)) != samples_path.stat().st_size
    ):
        raise StrictDataError("composition evaluation corpus samples checksum changed")
    rows = tuple(iter_jsonl_rows(samples_path))
    if len(rows) != int(manifest["event_count"]):
        raise StrictDataError("composition evaluation corpus count changed")
    records: list[EvaluationSample] = []
    for index, row in enumerate(rows):
        if (
            row.get("format") != EVAL_CORPUS_FORMAT
            or int(row.get("canonical_global_index", -1)) != index
            or str(row.get("task_type", "")) not in tuple(manifest["tasks"])
        ):
            raise StrictDataError("composition evaluation corpus order changed")
        input_text = str(row.get("input_text", ""))
        output_text = str(row.get("output_text", ""))
        expected_prompt_key = semantic_sha256(
            {
                "format": EVAL_CORPUS_FORMAT,
                "canonical_global_index": index,
                "task_type": str(row["task_type"]),
                "seed": int(row["seed"]),
                "renderer_version": manifest["renderer_version"],
                "input_text_sha256": row["input_text_sha256"],
            }
        )
        if row.get("prompt_key") != expected_prompt_key:
            raise StrictDataError("composition evaluation prompt identity changed")
        if hashlib.sha256(input_text.encode()).hexdigest() != row.get("input_text_sha256"):
            raise StrictDataError("composition evaluation input hash changed")
        if hashlib.sha256(output_text.encode()).hexdigest() != row.get("output_text_sha256"):
            raise StrictDataError("composition evaluation output hash changed")
        if row.get("proof_group") != proof_group(output_text):
            raise StrictDataError("composition evaluation proof grouping changed")
        records.append(
            EvaluationSample(
                task_type=str(row["task_type"]),
                seed=int(row["seed"]),
                input_text=input_text,
                output_text=output_text,
            )
        )
    counts = Counter(record.task_type for record in records)
    if counts != Counter({str(task): int(manifest["samples_per_task"]) for task in manifest["tasks"]}):
        raise StrictDataError("composition evaluation task balance changed")
    return manifest, tuple(records)


def materialize_eval_corpus(
    *,
    output_dir: str | Path,
    task_types: Sequence[str],
    samples_per_task: int,
    seed_base: int,
    name: str,
) -> dict[str, Any]:
    """Create a text+gold bank before the corresponding Circuit extraction."""

    tasks = tuple(str(task) for task in task_types)
    if not tasks or any(task not in ALL_TASKS for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("evaluation task list is invalid")
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("composition evaluation corpus requires PYTHONHASHSEED=42")
    if int(samples_per_task) <= 0:
        raise ValueError("samples_per_task must be positive")
    output = Path(output_dir).resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        _eval_corpus_rows(manifest_path)
        if (
            existing.get("name") != str(name)
            or existing.get("tasks") != list(tasks)
            or int(existing.get("samples_per_task", -1)) != int(samples_per_task)
            or int(existing.get("seed_base", -1)) != int(seed_base)
        ):
            raise StrictDataError("existing composition evaluation corpus parameters changed")
        return existing
    state = random.getstate()
    random.seed(42)
    try:
        records = create_evaluation_dataset(
            task_types=tasks,
            samples_per_task=int(samples_per_task),
            seed_base=int(seed_base),
            n_distractors=2,
            surface_family="narrow",
        )
    finally:
        random.setstate(state)
    rows = []
    for index, record in enumerate(records):
        rows.append(
            {
                "format": EVAL_CORPUS_FORMAT,
                "canonical_global_index": index,
                "task_type": record.task_type,
                "seed": int(record.seed),
                "prompt_key": semantic_sha256(
                    {
                        "format": EVAL_CORPUS_FORMAT,
                        "canonical_global_index": index,
                        "task_type": record.task_type,
                        "seed": int(record.seed),
                        "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
                        "input_text_sha256": hashlib.sha256(record.input_text.encode()).hexdigest(),
                    }
                ),
                "input_text": record.input_text,
                "output_text": record.output_text,
                "input_text_sha256": hashlib.sha256(record.input_text.encode()).hexdigest(),
                "output_text_sha256": hashlib.sha256(record.output_text.encode()).hexdigest(),
                "proof_group": proof_group(record.output_text),
            }
        )
    sample_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        for row in rows
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "samples.jsonl").write_bytes(sample_bytes)
    counts = Counter(str(row["task_type"]) for row in rows)
    identity = {
        "format": EVAL_CORPUS_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "name": str(name),
        "event_count": len(rows),
        "samples_per_task": int(samples_per_task),
        "seed_base": int(seed_base),
        "tasks": list(tasks),
        "task_counts": dict(counts),
        "prompt_keys_sha256": semantic_sha256([str(row["prompt_key"]) for row in rows]),
        "renderer_version": STAGE_B_LEGACY_RENDERER_VERSION,
        "n_distractors": 2,
        "samples_path": "samples.jsonl",
        "samples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "samples_bytes": len(sample_bytes),
    }
    manifest = {**identity, "status": "complete"}
    manifest["manifest_identity_sha256"] = semantic_sha256(manifest)
    _atomic_json(manifest_path, manifest)
    return manifest


def build_eval_bank(
    *,
    corpus_manifest: str | Path,
    vector_manifest: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    event_join: str | Path | None = None,
    n_ctx: int = N_CTX,
) -> dict[str, Any]:
    """Bind an extracted Vector397 archive to a frozen evaluation corpus."""

    if int(n_ctx) != N_CTX:
        raise ValueError("composition evaluation banks fix n_ctx=128")
    corpus_path = Path(corpus_manifest).resolve()
    if os.environ.get("PYTHONHASHSEED") != "42":
        raise RuntimeError("composition evaluation bank requires PYTHONHASHSEED=42")
    vector_path = Path(vector_manifest).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    destination = Path(output_path).resolve()
    manifest, records = _eval_corpus_rows(corpus_path)
    archive = read_descriptor397_archive(vector_path, full_tensor_verify=False)
    if int(archive.get("events", -1)) != len(records):
        raise StrictDataError("evaluation vector/corpus event count differs")
    archive_base = archive.get("base_identity") or archive.get("source_base_identity")
    if archive_base is not None and not isinstance(archive_base, Mapping):
        raise StrictDataError("evaluation Vector397 archive Base identity is malformed")
    if archive_base is not None and any(
        archive_base.get(key) != value for key, value in STAGE_A_BASE_IDENTITY.items()
    ):
        raise StrictDataError("evaluation Vector397 archive was extracted from a different Base")
    if event_join is None:
        raise ValueError("evaluation bank publication requires --event-join")
    event_join_rows = _read_event_join_rows(event_join, expected_count=len(records))
    event_keys = [str(row["event_key"]) for row in event_join_rows]
    if destination.is_file() or _sidecar(destination).is_file():
        if not destination.is_file() or not _sidecar(destination).is_file():
            raise StrictDataError("evaluation bank payload/sidecar are incomplete")
        existing = validate_eval_bank(destination)
        expected_identity = {
            "corpus_manifest_sha256": sha256_file(corpus_path),
            "corpus_manifest_identity_sha256": manifest["manifest_identity_sha256"],
            "vector_manifest_sha256": sha256_file(vector_path),
            "vector_archive_identity_sha256": archive["archive_identity_sha256"],
            "tokenizer_sha256": sha256_file(tokenizer_file),
            "event_join_sha256": sha256_file(event_join),
        }
        if any(existing.get(key) != value for key, value in expected_identity.items()):
            raise StrictDataError("existing evaluation bank is bound to different source artifacts")
        return existing
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_file))
    count = len(records)
    query_ids = torch.zeros((count, n_ctx), dtype=torch.int16)
    query_lengths = torch.empty(count, dtype=torch.int16)
    input_ids = torch.zeros((count, n_ctx), dtype=torch.int16)
    input_lengths = torch.empty(count, dtype=torch.int16)
    response_start = torch.empty(count, dtype=torch.int16)
    vectors = torch.empty((count, 397), dtype=torch.float32)
    task_ids = torch.empty(count, dtype=torch.uint8)
    event_tensor = torch.empty((count, 32), dtype=torch.uint8)
    payload_tensor = torch.empty((count, 32), dtype=torch.uint8)
    transform = FixedLog1pVector397()
    rng_state = random.getstate()
    random.seed(42)
    for index, event_key, payload_id, raw_vector in _iter_eval_vector_rows(vector_path, count):
        record = records[index]
        generated_input, generated_output = generate_sample(
            record.task_type,
            int(record.seed),
            n_distractors=2,
            renderer_version=STAGE_B_LEGACY_RENDERER_VERSION,
            surface_family="narrow",
        )
        if generated_input != record.input_text or generated_output != record.output_text:
            raise StrictDataError(f"evaluation renderer changed at row {index}")
        if event_key != event_keys[index]:
            raise StrictDataError(f"evaluation Vector397/event join changed at row {index}")
        event_row = event_join_rows[index]
        expected_prompt_key = semantic_sha256(
            {
                "format": EVAL_CORPUS_FORMAT,
                "canonical_global_index": index,
                "task_type": record.task_type,
                "seed": int(record.seed),
                "renderer_version": manifest["renderer_version"],
                "input_text_sha256": hashlib.sha256(record.input_text.encode()).hexdigest(),
            }
        )
        if (
            event_row.get("prompt_key") is not None
            and event_row.get("prompt_key") != expected_prompt_key
        ) or (
            event_row.get("task_type") is not None
            and event_row.get("task_type") != record.task_type
        ) or (
            event_row.get("seed") is not None
            and int(event_row["seed"]) != int(record.seed)
        ):
            raise StrictDataError(f"evaluation event-join prompt identity changed at row {index}")
        query = _tokenizer_encode(tokenizer, record.input_text)
        full = _tokenizer_encode(tokenizer, record.input_text + record.output_text)
        boundary = _response_start_from_token_proto(
            tokenizer, record.input_text + record.output_text, record.input_text
        )
        if len(query) > n_ctx or len(full) > n_ctx:
            raise StrictDataError(f"evaluation row {index} exceeds n_ctx={n_ctx}")
        query_ids[index, : len(query)] = torch.tensor(query, dtype=torch.int16)
        input_ids[index, : len(full)] = torch.tensor(full, dtype=torch.int16)
        query_lengths[index] = len(query)
        input_lengths[index] = len(full)
        response_start[index] = boundary
        vectors[index] = torch.from_numpy(transform.transform(raw_vector))
        task_ids[index] = TASK_TO_ID[record.task_type]
        event_tensor[index] = torch.tensor(list(bytes.fromhex(event_key)), dtype=torch.uint8)
        payload_tensor[index] = torch.tensor(list(bytes.fromhex(payload_id)), dtype=torch.uint8)
    random.setstate(rng_state)
    payload = {
        "query_ids": query_ids,
        "query_lengths": query_lengths,
        "input_ids": input_ids,
        "input_lengths": input_lengths,
        "response_start": response_start,
        "vectors": vectors,
        "task_ids": task_ids,
        "event_key": event_tensor,
        "payload_artifact_id": payload_tensor,
    }
    _validate_eval_payload(payload, count=count, n_ctx=n_ctx)
    _atomic_torch(destination, payload)
    record_rows = [
        {
            "canonical_global_index": index,
            "task_type": row.task_type,
            "seed": int(row.seed),
            "proof_group": proof_group(row.output_text),
            "input_text": row.input_text,
            "output_text": row.output_text,
        }
        for index, row in enumerate(records)
    ]
    records_path = destination.with_suffix(destination.suffix + ".records.jsonl")
    records_bytes = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        for row in record_rows
    )
    records_path.write_bytes(records_bytes)
    identity = {
        "format": EVAL_BANK_FORMAT,
        "protocol_id": PROTOCOL_ID,
        "event_count": count,
        "corpus_name": manifest["name"],
        "seed_base": int(manifest["seed_base"]),
        "samples_per_task": int(manifest["samples_per_task"]),
        "tasks": list(manifest["tasks"]),
        "task_counts": dict(manifest["task_counts"]),
        "n_ctx": int(n_ctx),
        "corpus_manifest_sha256": sha256_file(corpus_path),
        "corpus_manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "vector_manifest_sha256": sha256_file(vector_path),
        "vector_archive_identity_sha256": archive["archive_identity_sha256"],
        "tokenizer_sha256": sha256_file(tokenizer_file),
        "event_join_sha256": None if event_join is None else sha256_file(event_join),
        "expected_source_base_identity": dict(STAGE_A_BASE_IDENTITY),
        "archive_source_base_identity": None if archive_base is None else dict(archive_base),
        "input_treatment": transform.metadata(),
        "input_treatment_sha256": transform.artifact_sha256,
        "records_path": records_path.name,
        "records_sha256": hashlib.sha256(records_bytes).hexdigest(),
    }
    record = {
        **identity,
        "status": "complete",
        "payload_path": destination.name,
        "payload_sha256": sha256_file(destination),
        "payload_bytes": destination.stat().st_size,
        "identity_sha256": semantic_sha256(identity),
    }
    _atomic_json(_sidecar(destination), record)
    return validate_eval_bank(destination)


def _iter_eval_vector_rows(
    manifest_path: str | Path, count: int
) -> Iterator[tuple[int, str, str, np.ndarray]]:
    cursor = 0
    for index, event_key, payload_id, raw in _iter_vector_archive_rows_generic(manifest_path, count):
        yield index, event_key, payload_id, raw
        cursor += 1
    if cursor != count:
        raise StrictDataError("evaluation Vector397 coverage is incomplete")


def _iter_vector_archive_rows_generic(
    manifest_path: str | Path, count: int
) -> Iterator[tuple[int, str, str, np.ndarray]]:
    source = Path(manifest_path).resolve()
    archive = read_descriptor397_archive(source, full_tensor_verify=False)
    if int(archive.get("events", -1)) != int(count):
        raise StrictDataError("evaluation Vector397 event count changed")
    cursor = 0
    for shard in archive["shards"]:
        path = Path(str(shard["path"]))
        if not path.is_absolute():
            path = source.parent / path
        tensors = load_file(str(path.resolve()), device="cpu")
        rows = int(shard["records"])
        if tensors["global_index"].tolist() != list(range(cursor, cursor + rows)):
            raise StrictDataError("evaluation Vector397 canonical order changed")
        raw = tensors["descriptor_raw"]
        if raw.dtype != torch.float32 or raw.shape != (rows, 397) or not bool(torch.isfinite(raw).all()):
            raise StrictDataError("evaluation Vector397 raw tensor changed")
        event_keys = _hex_rows(tensors["event_key"])
        payload_ids = _hex_rows(tensors["payload_artifact_id"])
        for offset in range(rows):
            yield cursor + offset, event_keys[offset], payload_ids[offset], np.asarray(raw[offset].numpy(), dtype=np.float32)
        cursor += rows


def _validate_eval_payload(payload: Mapping[str, Tensor], *, count: int, n_ctx: int) -> None:
    expected = {
        "query_ids": (torch.int16, (count, n_ctx)),
        "query_lengths": (torch.int16, (count,)),
        "input_ids": (torch.int16, (count, n_ctx)),
        "input_lengths": (torch.int16, (count,)),
        "response_start": (torch.int16, (count,)),
        "vectors": (torch.float32, (count, 397)),
        "task_ids": (torch.uint8, (count,)),
        "event_key": (torch.uint8, (count, 32)),
        "payload_artifact_id": (torch.uint8, (count, 32)),
    }
    if set(payload) != set(expected):
        raise StrictDataError("evaluation bank tensor schema changed")
    for name, (dtype, shape) in expected.items():
        if payload[name].dtype != dtype or tuple(payload[name].shape) != shape:
            raise StrictDataError(f"evaluation bank {name} shape/dtype changed")
    if bool((payload["task_ids"] >= len(ALL_TASKS)).any()) or not bool(torch.isfinite(payload["vectors"]).all()):
        raise StrictDataError("evaluation bank values changed")


def validate_eval_bank(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    sidecar = _sidecar(source)
    if not source.is_file() or not sidecar.is_file():
        raise StrictDataError("evaluation bank payload/sidecar are missing")
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    identity = {
        key: value
        for key, value in record.items()
        if key not in {"status", "payload_path", "payload_sha256", "payload_bytes", "identity_sha256"}
    }
    if (
        record.get("format") != EVAL_BANK_FORMAT
        or record.get("protocol_id") != PROTOCOL_ID
        or record.get("status") != "complete"
        or record.get("payload_path") != source.name
        or record.get("payload_sha256") != sha256_file(source)
        or int(record.get("payload_bytes", -1)) != source.stat().st_size
        or record.get("identity_sha256") != semantic_sha256(identity)
        or int(record.get("n_ctx", -1)) != N_CTX
        or record.get("expected_source_base_identity") != STAGE_A_BASE_IDENTITY
        or not isinstance(record.get("event_join_sha256"), str)
        or len(record.get("event_join_sha256", "")) != 64
    ):
        raise StrictDataError("evaluation bank identity changed")
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    _validate_eval_payload(payload, count=int(record["event_count"]), n_ctx=int(record["n_ctx"]))
    records_path = source.with_suffix(source.suffix + ".records.jsonl")
    if record.get("records_path") != records_path.name or record.get("records_sha256") != sha256_file(records_path):
        raise StrictDataError("evaluation bank records identity changed")
    return record


@dataclass(frozen=True)
class EvalBank:
    metadata: Mapping[str, Any]
    payload: Mapping[str, Tensor]
    records: tuple[EvaluationSample, ...]


def load_eval_bank(
    path: str | Path,
    *,
    fingerprint_normalizer_path: str | Path | None = None,
    expected_fingerprint_normalizer_artifact_sha256: str | None = None,
) -> EvalBank:
    source = Path(path).resolve()
    metadata = dict(validate_eval_bank(source))
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if fingerprint_normalizer_path is not None:
        if metadata.get("input_treatment") != FixedLog1pVector397().metadata():
            raise StrictDataError(
                "z-score evaluation expects the canonical fixed-log1p bank"
            )
        normalizer, normalizer_metadata = read_vector397_normalizer(
            fingerprint_normalizer_path
        )
        if (
            expected_fingerprint_normalizer_artifact_sha256 is not None
            and normalizer_metadata.get("artifact_sha256")
            != expected_fingerprint_normalizer_artifact_sha256
        ):
            raise StrictDataError("evaluation fingerprint normalizer artifact identity changed")
        fixed = payload["vectors"].numpy()
        normalized = normalizer.transform_fixed_log1p_batch(fixed).astype(
            np.float32, copy=False
        )
        payload = dict(payload)
        payload["vectors"] = torch.from_numpy(np.ascontiguousarray(normalized))
        metadata["input_treatment"] = "train_zscore_v1"
        metadata["input_treatment_identity"] = normalizer_metadata.get("artifact_sha256")
        metadata["fingerprint_normalizer_sha256"] = normalizer_metadata.get(
            "normalizer_sha256"
        )
    records_path = source.with_suffix(source.suffix + ".records.jsonl")
    records: list[EvaluationSample] = []
    for index, row in enumerate(iter_jsonl_rows(records_path)):
        if int(row.get("canonical_global_index", -1)) != index:
            raise StrictDataError("evaluation bank record order changed")
        records.append(
            EvaluationSample(
                task_type=str(row["task_type"]),
                seed=int(row["seed"]),
                input_text=str(row["input_text"]),
                output_text=str(row["output_text"]),
            )
        )
    if len(records) != int(metadata["event_count"]):
        raise StrictDataError("evaluation bank record coverage changed")
    return EvalBank(metadata=metadata, payload=payload, records=tuple(records))


__all__ = [
    "ATOMIC_SAMPLES_PER_TASK",
    "ATOMIC_TASKS",
    "CACHE_FORMAT",
    "COMPOSITION_EVENT_COUNT",
    "COMPOSITION_TASK_TO_ID",
    "COMPOSITION_TASKS",
    "EvalBank",
    "EVAL_BANK_FORMAT",
    "EVAL_CORPUS_FORMAT",
    "FINAL_SAMPLES_PER_TASK",
    "FOLD_FORMAT",
    "FOLD_TASKS",
    "GLOBAL_BATCH_SIZE",
    "N_CTX",
    "PROTOCOL_ID",
    "STAGE_A_BASE_IDENTITY",
    "ROUTES",
    "TASK_TO_ID",
    "TOTAL_PADDED_TOKENS",
    "TOTAL_STEPS",
    "VALIDATION_SAMPLES_PER_TASK",
    "VALIDATION_STEPS",
    "CompositionFoldDataset",
    "build_composition_cache",
    "build_eval_bank",
    "load_eval_bank",
    "materialize_eval_corpus",
    "proof_group",
    "semantic_sha256",
    "sha256_file",
    "task_fold",
    "validate_composition_cache",
    "validate_eval_bank",
    "write_fold_indices",
]

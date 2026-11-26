"""Matched all-nine greedy validation for formal S5 HyperNet training."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import time
from typing import Any, Mapping

from safetensors.torch import load_file
import torch
from torch import Tensor

from ..circuit.descriptor_v2_397_artifact import read_descriptor397_archive
from ..circuit.hypernet_validation_corpus import (
    EVENT_COUNT,
    load_evaluation_records,
    load_samples,
    semantic_sha256,
    verify_corpus,
)
from ..protocol import (
    ALL_TASKS,
    KNOWN_TASKS,
    accuracy_from_counts,
    expected_first_tactic,
    extract_first_tactic,
)
from .circuit_normalizer import (
    FixedLog1pVector397,
    Vector397Normalizer,
    read_vector397_normalizer,
)
from .circuit_projection import retained_indices_from_dead_mask
from .data import _response_start_from_token_proto, _tokenizer_encode
from .evaluation import effective_delta_weight_norms, factor_l2_per_example
from .losses import causal_loss_stats_per_example
from .runtime import S5HyperNet


@dataclass(frozen=True)
class MatchedValidationBank:
    records: tuple[Any, ...]
    vectors: Tensor
    identity: Mapping[str, Any]


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def load_matched_validation_bank(
    *,
    corpus_manifest: str | Path,
    vector_manifest: str | Path,
    binding_path: str | Path,
    expected_suite_sha256: str,
    load_vectors: bool = True,
    vector397_treatment: FixedLog1pVector397 | Vector397Normalizer | None = None,
    compact381: bool = False,
    fingerprint_normalizer_path: str | Path | None = None,
    expected_fingerprint_normalizer_artifact_sha256: str | None = None,
) -> MatchedValidationBank:
    """Load the matched validation bank with either source API spelling.

    The remote compact381 path supplies an already-loaded Vector397Normalizer;
    the original train-only z-score path supplies its persisted normalizer path.
    Both are applied exactly once to raw validation descriptors.
    """

    if vector397_treatment is not None and fingerprint_normalizer_path is not None:
        raise ValueError("provide either vector397_treatment or fingerprint_normalizer_path")
    corpus_path = Path(corpus_manifest).resolve()
    vector_path = Path(vector_manifest).resolve()
    binding_file = Path(binding_path).resolve()
    artifact_root = binding_file.parent
    corpus = verify_corpus(corpus_path)
    archive = read_descriptor397_archive(
        vector_path, full_tensor_verify=bool(load_vectors)
    )
    trace = json.loads(
        (artifact_root / "traces" / "manifest.json").read_text(encoding="utf-8")
    )
    collection = json.loads(
        (artifact_root / "payload_collection" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    binding = json.loads(binding_file.read_text(encoding="utf-8"))
    body = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if (
        binding.get("format") != "s5_hypernet_validation_circuit_binding_v1"
        or binding.get("status") != "complete"
        or semantic_sha256(body) != binding.get("binding_sha256")
        or binding.get("corpus_manifest_identity_sha256")
        != corpus["manifest_identity_sha256"]
        or binding.get("validation_suite_sha256") != expected_suite_sha256
        or corpus.get("suite_sha256") != expected_suite_sha256
        or binding.get("vector_archive_identity_sha256")
        != archive["archive_identity_sha256"]
        or binding.get("payload_collection_identity_sha256")
        != archive["payload_collection_identity_sha256"]
        or binding.get("trace_event_keys_sha256") != archive["event_keys_sha256"]
        or binding.get("payload_canonical_event_keys_sha256")
        != archive["event_keys_sha256"]
        or int(binding.get("event_count", -1)) != EVENT_COUNT
        or int(archive.get("events", -1)) != EVENT_COUNT
        or binding.get("trace_collection_identity_sha256")
        != trace.get("collection_identity_sha256")
        or trace.get("prompt_keys")
        != [value["prompt_key"] for value in load_samples(corpus_path)]
        or trace.get("event_keys_sha256") != archive["event_keys_sha256"]
        or collection.get("canonical_event_keys_sha256")
        != archive["event_keys_sha256"]
        or collection.get("index_identity_sha256")
        != archive["payload_collection_identity_sha256"]
    ):
        raise ValueError("matched validation Circuit binding changed")

    fingerprint_metadata: Mapping[str, Any] | None = None
    if fingerprint_normalizer_path is not None:
        loaded_treatment, fingerprint_metadata = read_vector397_normalizer(
            fingerprint_normalizer_path
        )
        if (
            expected_fingerprint_normalizer_artifact_sha256 is not None
            and fingerprint_metadata.get("artifact_sha256")
            != expected_fingerprint_normalizer_artifact_sha256
        ):
            raise ValueError("matched validation fingerprint normalizer identity changed")
        treatment: FixedLog1pVector397 | Vector397Normalizer = loaded_treatment
    else:
        treatment = vector397_treatment or FixedLog1pVector397()
    if compact381 and not isinstance(treatment, Vector397Normalizer):
        raise ValueError("compact381 validation requires a train z-score normalizer")

    if load_vectors:
        raw_parts = []
        for record in archive["shards"]:
            tensors = load_file(
                str(_resolve(vector_path.parent, str(record["path"]))), device="cpu"
            )
            raw_parts.append(tensors["descriptor_raw"].float())
        raw = torch.cat(raw_parts, dim=0).contiguous()
        if raw.shape != (EVENT_COUNT, 397) or not bool(torch.isfinite(raw).all()):
            raise ValueError("matched validation Vector397 tensor changed")
        transformed = torch.stack(
            [torch.from_numpy(treatment.transform(row.numpy())) for row in raw], dim=0
        ).to(dtype=torch.float32).contiguous()
        if compact381:
            retained = torch.as_tensor(
                retained_indices_from_dead_mask(treatment.dead_mask), dtype=torch.long
            )
            transformed = transformed.index_select(1, retained).contiguous()
    else:
        transformed = torch.zeros(
            (EVENT_COUNT, 381 if compact381 else 397), dtype=torch.float32
        )
    records = load_evaluation_records(corpus_path)
    if len(records) != EVENT_COUNT:
        raise ValueError("matched validation record coverage changed")
    treatment_artifact = getattr(treatment, "artifact_sha256", None)
    identity = {
        "binding_sha256": binding["binding_sha256"],
        "corpus_manifest_identity_sha256": corpus["manifest_identity_sha256"],
        "vector_archive_identity_sha256": archive["archive_identity_sha256"],
        "validation_suite_sha256": expected_suite_sha256,
        "event_count": EVENT_COUNT,
        "input_treatment_identity": treatment_artifact,
        "condition_dim": int(transformed.shape[1]),
        "compact381": bool(compact381),
    }
    if isinstance(treatment, Vector397Normalizer):
        identity.update(
            {
                "input_treatment": (
                    "train_zscore_compact381_v1"
                    if compact381
                    else "train_zscore_v1"
                ),
                "fingerprint_normalizer_sha256": treatment.normalizer_sha256,
            }
        )
    if fingerprint_metadata is not None:
        identity["fingerprint_normalizer_artifact_sha256"] = fingerprint_metadata.get(
            "artifact_sha256"
        )
    return MatchedValidationBank(records=records, vectors=transformed, identity=identity)


def _decode_greedy(
    core: S5HyperNet,
    tokenizer: Any,
    *,
    prompt_ids: list[int],
    max_new_tokens: int,
    n_ctx: int,
) -> str:
    device = next(core.parameter_generator.parameters()).device
    values = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    prompt_length = len(prompt_ids)
    for _ in range(max_new_tokens):
        if values.shape[1] >= n_ctx:
            break
        logits = core.base_logits(values)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        values = torch.cat((values, next_id), dim=1)
        decoded = tokenizer.decode(values[0].tolist())
        if any(marker in decoded for marker in ("proof complete", "no goals")):
            break
    return tokenizer.decode(values[0, prompt_length:].tolist())


@torch.inference_mode()
def evaluate_matched_allnine(
    core: S5HyperNet,
    tokenizer: Any,
    bank: MatchedValidationBank,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
    gate_multiplier: float,
    n_ctx: int = 128,
    max_new_tokens: int = 50,
    raw_weight_control: str = "matched",
    raw_weight_layer_permutation: tuple[int, ...] | None = None,
    raw_weight_module_permutation: tuple[int, ...] | None = None,
    task_types: tuple[str, ...] | None = None,
    collect_predictions: bool = False,
) -> dict[str, Any] | None:
    if len(bank.records) != bank.vectors.shape[0] or world_size <= 0:
        raise ValueError("matched validation bank/runtime coverage changed")
    active_tasks = tuple(ALL_TASKS if task_types is None else task_types)
    if not active_tasks or len(set(active_tasks)) != len(active_tasks) or any(
        task not in KNOWN_TASKS for task in active_tasks
    ):
        raise ValueError("matched validation task mask is invalid")
    task_index = {task: index for index, task in enumerate(active_tasks)}
    counts = torch.zeros((len(active_tasks), 2), device=device, dtype=torch.float64)
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    local_predictions: list[dict[str, Any]] = []
    was_training = core.training
    core.eval()
    started = time.perf_counter()
    try:
        for index in range(rank, len(bank.records), world_size):
            record = bank.records[index]
            if record.task_type not in task_index:
                continue
            prompt_ids = _tokenizer_encode(tokenizer, record.input_text)
            if not prompt_ids or len(prompt_ids) >= n_ctx:
                raise ValueError("matched validation prompt does not fit context")
            query_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
            query_mask = torch.ones_like(query_ids, dtype=torch.bool)
            vector = bank.vectors[index : index + 1].to(device=device)
            with (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            ):
                state = core.generate_lora(
                    query_ids,
                    query_mask,
                    vector,
                    gate_multiplier=gate_multiplier,
                    raw_weight_control=raw_weight_control,
                    raw_weight_layer_permutation=raw_weight_layer_permutation,
                    raw_weight_module_permutation=raw_weight_module_permutation,
                )
            with core.controller.activate(state):
                generated = _decode_greedy(
                    core,
                    tokenizer,
                    prompt_ids=prompt_ids,
                    max_new_tokens=max_new_tokens,
                    n_ctx=n_ctx,
                )
                full_text = record.input_text + record.output_text
                full_ids = _tokenizer_encode(tokenizer, full_text)
                if len(full_ids) > n_ctx:
                    raise ValueError("matched validation supervision exceeds context")
                boundary = _response_start_from_token_proto(
                    tokenizer, full_text, record.input_text
                )
                input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
                labels = input_ids.clone()
                labels[:, :boundary] = -100
                logits = core.base_logits(input_ids)
                loss_sum, _mean, token_count = causal_loss_stats_per_example(
                    logits, labels
                )
            predicted = extract_first_tactic(generated)
            expected = expected_first_tactic(record.output_text)
            task = task_index[record.task_type]
            counts[task, 0] += float(predicted == expected)
            counts[task, 1] += 1.0
            totals[0] += loss_sum.double().sum()
            totals[1] += token_count.double().sum()
            totals[2] += factor_l2_per_example(state).double().sum()
            totals[3] += effective_delta_weight_norms(state).double().sum()
            totals[4] += 1.0
            if collect_predictions:
                local_predictions.append(
                    {
                        "canonical_global_index": int(index),
                        "task_type": record.task_type,
                        "seed": int(record.seed),
                        "predicted_tactic": predicted,
                        "expected_tactic": expected,
                        "correct": bool(predicted == expected),
                    }
                )
    finally:
        core.train(was_training)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(counts)
        torch.distributed.all_reduce(totals)
    gathered_predictions: list[Any] | None = None
    if collect_predictions:
        gathered_predictions = [None] * int(world_size)
        torch.distributed.all_gather_object(gathered_predictions, local_predictions)
    elapsed = time.perf_counter() - started
    if rank != 0:
        return None
    task_counts = {
        task: {
            "correct": int(counts[index, 0].item()),
            "total": int(counts[index, 1].item()),
        }
        for index, task in enumerate(active_tasks)
    }
    result = accuracy_from_counts(task_counts)
    samples = float(totals[4].item())
    result.update(
        {
            "condition": "matched",
            "raw_weight_control": raw_weight_control,
            "raw_weight_layer_permutation": (
                None
                if raw_weight_layer_permutation is None
                else list(raw_weight_layer_permutation)
            ),
            "raw_weight_module_permutation": (
                None
                if raw_weight_module_permutation is None
                else list(raw_weight_module_permutation)
            ),
            "response_nll": float(totals[0].item() / totals[1].item()),
            "factor_l2_example_mean": float(totals[2].item() / samples),
            "effective_delta_norm_example_mean": float(totals[3].item() / samples),
            "gate_multiplier": float(gate_multiplier),
            "world_size": int(world_size),
            "wall_seconds_rank0": elapsed,
            "validation_bank": dict(bank.identity),
        }
    )
    if collect_predictions:
        result["predictions"] = sorted(
            [row for shard in (gathered_predictions or []) for row in (shard or [])],
            key=lambda row: int(row["canonical_global_index"]),
        )
    return result


__all__ = [
    "MatchedValidationBank",
    "evaluate_matched_allnine",
    "load_matched_validation_bank",
]

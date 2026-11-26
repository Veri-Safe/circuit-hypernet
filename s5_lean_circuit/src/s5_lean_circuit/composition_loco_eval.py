"""Unified exact-match evaluator for composition LOCO checkpoints."""

from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import random
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch import Tensor

from .composition_loco import EvalBank, proof_group
from .protocol import expected_first_tactic, extract_first_tactic


GROUPS = ("simple", "nested")
DEFAULT_EVAL_BATCH_SIZE = 16


def _derange_values(values: list[int], rng: random.Random) -> list[int]:
    if len(values) < 2:
        raise ValueError("shuffled Circuit control requires at least two events")
    target_values = list(values)
    permutation = list(values)
    rng.shuffle(permutation)
    fixed = [offset for offset, source in enumerate(permutation) if target_values[offset] == source]
    if len(fixed) == 1:
        index = fixed[0]
        other = (index + 1) % len(values)
        permutation[index], permutation[other] = permutation[other], permutation[index]
    elif fixed:
        fixed_values = [permutation[index] for index in fixed]
        for offset, index in enumerate(fixed):
            permutation[index] = fixed_values[(offset + 1) % len(fixed_values)]
    if any(target_values[index] == source for index, source in enumerate(permutation)):
        raise RuntimeError("seed-42 Circuit shuffle produced a fixed point")
    if sorted(permutation) != sorted(target_values):
        raise RuntimeError("seed-42 Circuit shuffle is not a permutation")
    return permutation


def _shuffle_permutation(records: Sequence[Any]) -> list[int]:
    """Return a seed-42 derangement within each task family."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(str(record.task_type), []).append(index)
    permutation = list(range(len(records)))
    rng = random.Random(42)
    for task in sorted(groups):
        indices = groups[task]
        if len(indices) < 2:
            raise ValueError(f"shuffled Circuit control needs at least two rows for {task}")
        sources = _derange_values(indices, rng)
        for target, source in zip(indices, sources, strict=True):
            permutation[target] = source
    return permutation


def cross_task_source_indices(
    records: Sequence[Any], *, target_task: str, donor_task: str, pairing_seed: int = 42,
) -> list[int]:
    """Pair one target family with another family in the same bank.

    The target-to-donor map is bijective. Other rows are unused identity entries;
    this is deliberately NOT a permutation of the entire nine-task bank.
    """
    if target_task == donor_task:
        raise ValueError("cross_task requires a different donor task")
    targets = [i for i, row in enumerate(records) if row.task_type == target_task]
    donors = [i for i, row in enumerate(records) if row.task_type == donor_task]
    if not targets or len(targets) != len(donors):
        raise ValueError("cross_task needs nonempty target/donor families of equal size")
    random.Random(int(pairing_seed)).shuffle(donors)
    sources = list(range(len(records)))
    for target, donor in zip(targets, donors, strict=True):
        sources[target] = donor
    return sources


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


@torch.inference_mode()
def _decode_model(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    *,
    device: torch.device,
    n_ctx: int,
    max_new_tokens: int,
) -> str:
    values = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    start = values.shape[1]
    for _ in range(int(max_new_tokens)):
        if values.shape[1] >= int(n_ctx):
            break
        with _autocast(device):
            output = model(values)
            logits = output[0] if isinstance(output, (tuple, list)) else getattr(output, "logits", output)
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        values = torch.cat((values, next_id), dim=1)
        decoded = tokenizer.decode(values[0].tolist())
        if "proof complete" in decoded or "no goals" in decoded:
            break
    return tokenizer.decode(values[0, start:].tolist())


@torch.inference_mode()
def _decode_hypernet(
    core: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    vector: Tensor,
    *,
    device: torch.device,
    n_ctx: int,
    max_new_tokens: int,
    circuit_present: bool = True,
) -> str:
    values = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    query_ids = values.clone()
    query_mask = torch.ones_like(query_ids, dtype=torch.bool)
    with _autocast(device):
        state = core.generate_lora(
            query_ids,
            query_mask,
            vector.to(device=device),
            circuit_present=torch.tensor([circuit_present], device=device),
            gate_multiplier=float(getattr(core, "_loco_gate_multiplier", 1.0)),
        )
    start = values.shape[1]
    with core.controller.activate(state):
        for _ in range(int(max_new_tokens)):
            if values.shape[1] >= int(n_ctx):
                break
            with _autocast(device):
                logits = core.base_logits(values)
                next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            values = torch.cat((values, next_id), dim=1)
            decoded = tokenizer.decode(values[0].tolist())
            if "proof complete" in decoded or "no goals" in decoded:
                break
    return tokenizer.decode(values[0, start:].tolist())


@torch.inference_mode()
def _decode_model_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_ids_batch: Sequence[Sequence[int]],
    *,
    device: torch.device,
    n_ctx: int,
    max_new_tokens: int,
) -> list[str]:
    """Greedily decode equal-length prompts as one model batch.

    The S5 GPT forward path has no padding attention mask.  Callers therefore
    group prompts by their exact canonical length before entering this helper;
    every row remains aligned at every generation step.
    """

    if not prompt_ids_batch:
        return []
    lengths = {len(prompt) for prompt in prompt_ids_batch}
    if len(lengths) != 1:
        raise ValueError("batched composition decoding requires equal prompt lengths")
    values = torch.tensor(list(prompt_ids_batch), dtype=torch.long, device=device)
    start = int(values.shape[1])
    generated: list[list[int]] = [[] for _ in prompt_ids_batch]
    finished = torch.zeros((values.shape[0],), dtype=torch.bool, device=device)
    max_steps = min(int(max_new_tokens), max(0, int(n_ctx) - start))
    for _ in range(max_steps):
        with _autocast(device):
            output = model(values)
            logits = output[0] if isinstance(output, (tuple, list)) else getattr(output, "logits", output)
            predicted = logits[:, -1, :].argmax(dim=-1)
        active = ~finished
        for row in range(values.shape[0]):
            if bool(active[row].item()):
                token = int(predicted[row].item())
                generated[row].append(token)
            else:
                token = int(values[row, -1].item())
            predicted[row] = token
        values = torch.cat((values, predicted.unsqueeze(1)), dim=1)
        for row in range(values.shape[0]):
            if bool(active[row].item()):
                decoded = tokenizer.decode(values[row].tolist())
                if "proof complete" in decoded or "no goals" in decoded:
                    finished[row] = True
        if bool(finished.all().item()):
            break
    return [tokenizer.decode(tokens) for tokens in generated]


@torch.inference_mode()
def _decode_hypernet_batch(
    core: Any,
    tokenizer: Any,
    prompt_ids_batch: Sequence[Sequence[int]],
    vectors: Tensor,
    *,
    device: torch.device,
    n_ctx: int,
    max_new_tokens: int,
    circuit_present: bool = True,
) -> list[str]:
    """Generate one LoRA per row and decode the rows in one forward batch."""

    if not prompt_ids_batch:
        return []
    lengths = {len(prompt) for prompt in prompt_ids_batch}
    if len(lengths) != 1:
        raise ValueError("batched HyperNet decoding requires equal prompt lengths")
    if vectors.ndim != 2 or vectors.shape[0] != len(prompt_ids_batch):
        raise ValueError("batched HyperNet vectors must align with prompt rows")
    values = torch.tensor(list(prompt_ids_batch), dtype=torch.long, device=device)
    query_ids = values.clone()
    query_mask = torch.ones_like(query_ids, dtype=torch.bool)
    present = torch.full(
        (values.shape[0],), bool(circuit_present), dtype=torch.bool, device=device
    )
    with _autocast(device):
        state = core.generate_lora(
            query_ids,
            query_mask,
            vectors.to(device=device),
            circuit_present=present,
            gate_multiplier=float(getattr(core, "_loco_gate_multiplier", 1.0)),
        )
    start = int(values.shape[1])
    generated: list[list[int]] = [[] for _ in prompt_ids_batch]
    finished = torch.zeros((values.shape[0],), dtype=torch.bool, device=device)
    max_steps = min(int(max_new_tokens), max(0, int(n_ctx) - start))
    with core.controller.activate(state):
        for _ in range(max_steps):
            with _autocast(device):
                logits = core.base_logits(values)
                predicted = logits[:, -1, :].argmax(dim=-1)
            active = ~finished
            for row in range(values.shape[0]):
                if bool(active[row].item()):
                    token = int(predicted[row].item())
                    generated[row].append(token)
                else:
                    token = int(values[row, -1].item())
                predicted[row] = token
            values = torch.cat((values, predicted.unsqueeze(1)), dim=1)
            for row in range(values.shape[0]):
                if bool(active[row].item()):
                    decoded = tokenizer.decode(values[row].tolist())
                    if "proof complete" in decoded or "no goals" in decoded:
                        finished[row] = True
            if bool(finished.all().item()):
                break
    return [tokenizer.decode(tokens) for tokens in generated]


def _empty(task_types: Sequence[str]) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    task_counts = {task: {"correct": 0, "total": 0} for task in task_types}
    groups = {
        task: {group: {"correct": 0, "total": 0} for group in GROUPS}
        for task in task_types
    }
    return task_counts, groups


def _merge_counts(
    local: Tensor,
    groups: Tensor,
    *,
    device: torch.device,
    task_types: Sequence[str],
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, dict[str, int]]]]:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
        dist.all_reduce(groups, op=dist.ReduceOp.SUM)
    task_counts, proof_groups = _empty(task_types)
    for task_index, task in enumerate(task_types):
        task_counts[task] = {
            "correct": int(local[task_index, 0].item()),
            "total": int(local[task_index, 1].item()),
        }
        for group_index, group in enumerate(GROUPS):
            proof_groups[task][group] = {
                "correct": int(groups[task_index, group_index, 0].item()),
                "total": int(groups[task_index, group_index, 1].item()),
            }
    return task_counts, proof_groups


def _result(
    task_counts: Mapping[str, Mapping[str, int]],
    proof_groups: Mapping[str, Mapping[str, Mapping[str, int]]],
    *,
    condition: str,
    gate_multiplier: float,
) -> dict[str, Any]:
    task_accuracy = {
        task: (100.0 * int(row["correct"]) / int(row["total"]) if int(row["total"]) else 0.0)
        for task, row in task_counts.items()
    }
    total_correct = sum(int(row["correct"]) for row in task_counts.values())
    total = sum(int(row["total"]) for row in task_counts.values())
    macro = sum(task_accuracy.values()) / len(task_accuracy) if task_accuracy else 0.0
    group_totals = {
        group: {
            "correct": sum(int(proof_groups[task][group]["correct"]) for task in proof_groups),
            "total": sum(int(proof_groups[task][group]["total"]) for task in proof_groups),
        }
        for group in GROUPS
    }
    for group in GROUPS:
        group_totals[group]["accuracy"] = (
            100.0 * group_totals[group]["correct"] / group_totals[group]["total"]
            if group_totals[group]["total"]
            else 0.0
        )
    return {
        "metric": "strict_first_tactic_exact_match",
        "condition": str(condition),
        "gate_multiplier": float(gate_multiplier),
        "task_counts": {task: dict(row) for task, row in task_counts.items()},
        "task_accuracy": task_accuracy,
        "composition_macro_accuracy": macro,
        "overall_correct": total_correct,
        "overall_total": total,
        "overall_accuracy": 100.0 * total_correct / total if total else 0.0,
        "proof_groups": {
            task: {group: dict(row) for group, row in groups.items()}
            for task, groups in proof_groups.items()
        },
        "proof_group_totals": group_totals,
    }


@torch.inference_mode()
def evaluate_bank(
    model_or_core: Any,
    tokenizer: Any,
    bank: EvalBank,
    *,
    device: torch.device,
    task_types: Sequence[str],
    n_ctx: int = 128,
    max_new_tokens: int = 50,
    route: str,
    condition: str = "matched",
    gate_multiplier: float = 1.0,
    indices: Sequence[int] | None = None,
    collect_predictions: bool = False,
    collect_generated_text: bool = False,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    donor_task: str | None = None,
    pairing_seed: int = 42,
) -> dict[str, Any] | None:
    """Evaluate one bank; all ranks call this function when distributed."""

    if int(batch_size) <= 0:
        raise ValueError("composition evaluation batch_size must be positive")
    if collect_generated_text and not collect_predictions:
        raise ValueError("collect_generated_text requires collect_predictions")

    tasks = tuple(str(task) for task in task_types)
    task_index = {task: index for index, task in enumerate(tasks)}
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
    local = torch.zeros((len(tasks), 2), dtype=torch.int64, device=device)
    groups = torch.zeros((len(tasks), len(GROUPS), 2), dtype=torch.int64, device=device)
    selected = list(range(len(bank.records))) if indices is None else [int(value) for value in indices]
    if condition not in {"matched", "shuffled", "absent", "zero_as_absent", "cross_task"}:
        raise ValueError("unknown Circuit condition")
    if condition != "cross_task" and donor_task is not None:
        raise ValueError("donor_task is only valid for the cross_task condition")
    permutation = list(range(len(bank.records)))
    if condition == "cross_task":
        if route != "circuit_a" or len(tasks) != 1 or donor_task is None:
            raise ValueError("cross_task requires Circuit route, one target task and donor_task")
        permutation = cross_task_source_indices(
            bank.records, target_task=tasks[0], donor_task=donor_task, pairing_seed=pairing_seed,
        )
        event_keys = bank.payload["event_key"]
        if any(
            torch.equal(event_keys[index], event_keys[source])
            for index, source in enumerate(permutation)
            if bank.records[index].task_type == tasks[0]
        ):
            raise ValueError("cross_task target and donor share an event identity")
    if condition == "shuffled":
        permutation = _shuffle_permutation(bank.records)
        event_keys = bank.payload["event_key"]
        if any(
            torch.equal(event_keys[index], event_keys[source])
            for index, source in enumerate(permutation)
        ):
            raise RuntimeError("seed-42 Circuit shuffle retained an event identity")
        if any(
            bank.records[index].task_type != bank.records[source].task_type
            for index, source in enumerate(permutation)
        ):
            raise RuntimeError("seed-42 Circuit shuffle crossed task families")
    local_predictions: list[dict[str, Any]] = []
    was_training = bool(getattr(model_or_core, "training", False))
    model_or_core.eval()
    local_entries = [
        (position, index)
        for position, index in enumerate(selected)
        if position % world == rank and bank.records[index].task_type in task_index
    ]
    # The GPT forward does not accept an attention mask, so batch only rows
    # with identical canonical prompt lengths.  This still gives substantial
    # batches on the fixed S5 banks without changing any token positions.
    by_length: dict[int, list[tuple[int, int]]] = {}
    for position, index in local_entries:
        query_length = int(bank.payload["query_lengths"][index].item())
        by_length.setdefault(query_length, []).append((position, index))
    for query_length in sorted(by_length):
        entries = by_length[query_length]
        for offset in range(0, len(entries), int(batch_size)):
            batch_entries = entries[offset : offset + int(batch_size)]
            batch_indices = [index for _, index in batch_entries]
            prompt_ids_batch = [
                bank.payload["query_ids"][index, :query_length].tolist()
                for index in batch_indices
            ]
            source_indices: list[int | None]
            if route == "direct_sft":
                source_indices = [None] * len(batch_indices)
                generated_batch = _decode_model_batch(
                    model_or_core,
                    tokenizer,
                    prompt_ids_batch,
                    device=device,
                    n_ctx=n_ctx,
                    max_new_tokens=max_new_tokens,
                )
            else:
                source_indices = [
                    permutation[index] if condition in {"shuffled", "cross_task"} else index
                    for index in batch_indices
                ]
                present = condition not in {"absent", "zero_as_absent"}
                vectors = bank.payload["vectors"][source_indices]
                if condition == "zero_as_absent":
                    vectors = torch.zeros_like(vectors)
                model_or_core._loco_gate_multiplier = float(gate_multiplier)
                generated_batch = _decode_hypernet_batch(
                    model_or_core,
                    tokenizer,
                    prompt_ids_batch,
                    vectors,
                    device=device,
                    n_ctx=n_ctx,
                    max_new_tokens=max_new_tokens,
                    circuit_present=present,
                )
            for index, source_index, generated in zip(
                batch_indices, source_indices, generated_batch, strict=True
            ):
                record = bank.records[index]
                task_slot = task_index[record.task_type]
                predicted = extract_first_tactic(generated)
                expected = expected_first_tactic(record.output_text)
                correct = int(predicted == expected)
                group = proof_group(record.output_text)
                local[task_slot, 0] += correct
                local[task_slot, 1] += 1
                groups[task_slot, GROUPS.index(group), 0] += correct
                groups[task_slot, GROUPS.index(group), 1] += 1
                if collect_predictions:
                    local_predictions.append(
                        {
                            "canonical_global_index": int(index),
                            "task_type": record.task_type,
                            "seed": int(record.seed),
                            "condition": condition,
                            "source_index": source_index,
                            "source_event_key": (
                                None
                                if source_index is None
                                else bytes(bank.payload["event_key"][source_index].tolist()).hex()
                            ),
                            "target_event_key": bytes(bank.payload["event_key"][index].tolist()).hex(),
                            "predicted_tactic": predicted,
                            "expected_tactic": expected,
                            "correct": bool(correct),
                            "proof_group": group,
                        }
                    )
                    if collect_generated_text:
                        prompt_ids = bank.payload["query_ids"][index, :query_length].tolist()
                        local_predictions[-1].update(
                            generated_text=generated,
                            prompt_text=tokenizer.decode(prompt_ids),
                            prompt_token_ids=prompt_ids,
                            prompt_token_count=query_length,
                            effective_max_new_tokens=min(max_new_tokens, max(0, n_ctx - query_length)),
                        )
                    if condition == "cross_task":
                        local_predictions[-1].update(
                            source_task_type=bank.records[source_index].task_type,
                            pairing_seed=int(pairing_seed), circuit_present=True,
                        )
    if was_training:
        model_or_core.train()
    gathered_predictions: list[dict[str, Any]] | None = None
    if collect_predictions:
        if dist.is_available() and dist.is_initialized():
            gathered: list[list[dict[str, Any]] | None] = [None] * world
            dist.all_gather_object(gathered, local_predictions)
            gathered_predictions = [
                item for shard in gathered if shard is not None for item in shard
            ]
        else:
            gathered_predictions = local_predictions
        gathered_predictions.sort(key=lambda item: int(item["canonical_global_index"]))
    task_counts, proof_groups = _merge_counts(local, groups, device=device, task_types=tasks)
    if rank != 0:
        return None
    result = _result(task_counts, proof_groups, condition=condition, gate_multiplier=gate_multiplier)
    result["route"] = str(route)
    result["event_count"] = int(sum(row["total"] for row in task_counts.values()))
    result["condition_mapping"] = {
        "kind": "identity" if condition == "matched" else condition,
        "source_index_for_target": (
            None
            if condition not in {"shuffled", "cross_task"}
            else permutation
        ),
        "sha256": hashlib.sha256(
            json.dumps(
                None if condition not in {"shuffled", "cross_task"} else permutation,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }
    if condition == "shuffled":
        result["condition_mapping"]["seed"] = 42
    if condition == "cross_task":
        result["condition_mapping"].update(
            seed=int(pairing_seed), target_task=tasks[0], donor_task=donor_task,
            mapping_scope="target_task_only_other_rows_unused", circuit_present=True,
            vector_unit="entire_397d_row",
        )
    if gathered_predictions is not None:
        result["predictions"] = gathered_predictions
    return result


__all__ = ["DEFAULT_EVAL_BATCH_SIZE", "GROUPS", "evaluate_bank", "cross_task_source_indices"]

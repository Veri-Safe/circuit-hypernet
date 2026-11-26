"""Evaluation and counterfactual assignment utilities for S5 HyperNet.

The evaluator deliberately lives outside :mod:`runtime`.  Runtime owns the
query/circuit -> Generated-LoRA forward path; this module owns the frozen
evaluation assignments, response-only metrics, and paired comparisons.  In
particular, an ``absent`` assignment removes circuit memory through an
explicit mask.  It never substitutes a zero Vector397, which would be a
different condition.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import TokenizedS5Row, collate_tokenized_s5_rows


AssignmentMode = Literal[
    "matched",
    "shuffled",
    "global_unstratified_shuffled",
    "cross_task_shuffled",
    "absent",
    "zero_present",
    "mean_present",
    "block_permuted",
]
_ASSIGNMENT_MODES = frozenset(
    {
        "matched",
        "shuffled",
        "global_unstratified_shuffled",
        "cross_task_shuffled",
        "absent",
        "zero_present",
        "mean_present",
        "block_permuted",
    }
)
_DEFAULT_BLOCK_PERMUTATION = (1, 2, 3, 4, 0)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _as_cpu_long(value: Tensor | Sequence[int], *, name: str) -> Tensor:
    result = torch.as_tensor(value, dtype=torch.long, device="cpu").reshape(-1)
    if result.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    return result


def _tuple_strings(value: Iterable[Any] | None, *, size: int, name: str) -> tuple[str, ...]:
    if value is None or (isinstance(value, (tuple, list)) and len(value) == 0):
        return tuple("" for _ in range(size))
    result = tuple(str(item) for item in value)
    if len(result) != size:
        raise ValueError(f"{name} must have one entry per batch row")
    return result


@dataclass(frozen=True)
class EvaluationBatch:
    """One immutable, condition-safe batch used by the evaluator.

    ``task_type`` and ``event_key`` are metadata for stratification and
    reporting only.  They are never passed to the model.  ``input_mask`` is
    retained for audits because the S5 Base currently accepts right-padded
    input IDs without an attention-mask argument.
    """

    canonical_global_index: Tensor
    query_ids: Tensor
    query_mask: Tensor
    input_ids: Tensor
    labels: Tensor
    vector397: Tensor
    input_mask: Tensor | None = None
    task_type: tuple[str, ...] = ()
    event_key: tuple[str, ...] = ()
    prompt_key: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = _as_cpu_long(self.canonical_global_index, name="canonical_global_index")
        if ids.unique().numel() != ids.numel():
            raise ValueError("canonical_global_index contains duplicates")
        tensors = {
            "query_ids": self.query_ids,
            "query_mask": self.query_mask,
            "input_ids": self.input_ids,
            "labels": self.labels,
        }
        for name, value in tensors.items():
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch Tensor")
            if value.ndim != 2 or value.shape[0] != ids.numel():
                raise ValueError(f"{name} must have shape [B,T]")
        if self.query_mask.shape != self.query_ids.shape:
            raise ValueError("query_mask/query_ids shape mismatch")
        if self.labels.shape != self.input_ids.shape:
            raise ValueError("labels/input_ids shape mismatch")
        if self.query_mask.dtype != torch.bool:
            raise ValueError("query_mask must be bool")
        if self.input_mask is not None:
            if self.input_mask.shape != self.input_ids.shape or self.input_mask.dtype != torch.bool:
                raise ValueError("input_mask must be bool and match input_ids")
        vector = self.vector397
        if not isinstance(vector, Tensor) or vector.ndim != 2 or tuple(vector.shape) != (ids.numel(), 397):
            raise ValueError("vector397 must have shape [B,397]")
        if vector.dtype != torch.float32:
            raise ValueError("vector397 must be FP32")
        if not bool(torch.isfinite(vector).all()):
            raise ValueError("vector397 contains non-finite values")
        object.__setattr__(self, "canonical_global_index", ids)
        object.__setattr__(self, "task_type", _tuple_strings(self.task_type, size=ids.numel(), name="task_type"))
        object.__setattr__(self, "event_key", _tuple_strings(self.event_key, size=ids.numel(), name="event_key"))
        object.__setattr__(self, "prompt_key", _tuple_strings(self.prompt_key, size=ids.numel(), name="prompt_key"))

    @property
    def batch_size(self) -> int:
        return int(self.canonical_global_index.numel())

    @classmethod
    def from_collated(cls, batch: Mapping[str, Any]) -> "EvaluationBatch":
        """Build an evaluation batch from ``collate_tokenized_s5_rows`` output."""

        rows = tuple(batch.get("rows", ()))
        if rows:
            ids = tuple(int(row.identity["canonical_global_index"]) for row in rows)
            tasks = tuple(str(row.task_type) for row in rows)
            events = tuple(str(row.identity["event_key"]) for row in rows)
            prompts = tuple(str(row.identity["prompt_key"]) for row in rows)
        else:
            identities = tuple(batch.get("identities", ()))
            ids = tuple(int(identity["canonical_global_index"]) for identity in identities)
            tasks = tuple(str(value) for value in batch.get("task_types", ()))
            events = tuple(str(identity.get("event_key", "")) for identity in identities)
            prompts = tuple(str(identity.get("prompt_key", "")) for identity in identities)
        if not ids:
            raise ValueError("collated batch has no canonical identities")
        return cls(
            canonical_global_index=torch.tensor(ids, dtype=torch.long),
            query_ids=batch["query_ids"],
            query_mask=batch["query_mask"].to(dtype=torch.bool),
            input_ids=batch["input_ids"],
            input_mask=batch.get("input_mask", None),
            labels=batch["labels"],
            vector397=batch["vector397"].to(dtype=torch.float32),
            task_type=tasks,
            event_key=events,
            prompt_key=prompts,
        )

    @classmethod
    def from_rows(
        cls, rows: Sequence[TokenizedS5Row], *, device: torch.device | str | None = None
    ) -> "EvaluationBatch":
        return cls.from_collated(collate_tokenized_s5_rows(rows, device=device))


@dataclass(frozen=True)
class AssignmentPlan:
    """Frozen target -> source condition assignment for one evaluation batch."""

    mode: AssignmentMode
    seed: int
    target_global_index: tuple[int, ...]
    source_global_index: tuple[int, ...]
    permutation_sha256: str | None = None
    block_permutation: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).lower()
        if mode not in _ASSIGNMENT_MODES:
            raise ValueError(f"unknown assignment mode: {self.mode!r}")
        target = tuple(int(value) for value in self.target_global_index)
        source = tuple(int(value) for value in self.source_global_index)
        if not target or len(target) != len(source):
            raise ValueError("assignment target/source cardinality is invalid")
        if len(set(target)) != len(target):
            raise ValueError("assignment target contains duplicates")
        if mode in {
            "matched",
            "absent",
            "zero_present",
            "mean_present",
            "block_permuted",
        } and target != source:
            raise ValueError(f"{mode} assignment must use matched source indices")
        block = self.block_permutation
        if block is not None:
            block = tuple(int(value) for value in block)
            if len(block) != 5 or sorted(block) != list(range(5)):
                raise ValueError("block_permutation must be a permutation of [0,1,2,3,4]")
        if mode == "block_permuted" and block is None:
            block = _DEFAULT_BLOCK_PERMUTATION
        if mode != "block_permuted" and block is not None:
            raise ValueError("block_permutation is only valid for block_permuted mode")
        if mode in {"shuffled", "global_unstratified_shuffled", "cross_task_shuffled"}:
            if len(set(source)) != len(source):
                raise ValueError("shuffled source assignment contains duplicates")
            if any(a == b for a, b in zip(target, source)):
                raise ValueError("shuffled assignment must be a derangement")
        payload = {
            "mode": mode,
            "seed": int(self.seed),
            "target_global_index": list(target),
            "source_global_index": list(source),
            "block_permutation": list(block) if block is not None else None,
        }
        digest = self.permutation_sha256
        if mode in {"shuffled", "global_unstratified_shuffled", "cross_task_shuffled"}:
            expected = _sha256_json(payload)
            if digest is not None and str(digest) != expected:
                raise ValueError("shuffled permutation SHA256 mismatch")
            digest = expected
        else:
            digest = None
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "target_global_index", target)
        object.__setattr__(self, "source_global_index", source)
        object.__setattr__(self, "block_permutation", block)
        object.__setattr__(self, "permutation_sha256", digest)

    @property
    def assignment_sha256(self) -> str | None:
        return self.permutation_sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": int(self.seed),
            "target_global_index": list(self.target_global_index),
            "source_global_index": list(self.source_global_index),
            "permutation_sha256": self.permutation_sha256,
            "block_permutation": list(self.block_permutation)
            if self.block_permutation is not None
            else None,
        }


@dataclass
class BatchEvaluation:
    mode: AssignmentMode
    canonical_global_index: Tensor
    loss_sum: Tensor
    token_count: Tensor
    response_nll: Tensor
    per_example_nll: Tensor
    per_example_correct: Tensor | None = None
    factor_l2_per_example: Tensor | None = None
    effective_delta_l2_per_example: Tensor | None = None
    task_type: tuple[str, ...] = ()
    logits: Tensor | None = None
    memory_mask: Tensor | None = None
    activation_summary: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        ids = _as_cpu_long(self.canonical_global_index, name="canonical_global_index")
        size = ids.numel()
        for name in ("loss_sum", "token_count", "per_example_nll"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.numel() != size:
                raise ValueError(f"{name} must have one value per example")
        if bool((self.token_count <= 0).any()):
            raise ValueError("every example must contain response targets")
        if self.per_example_correct is not None and self.per_example_correct.numel() != size:
            raise ValueError("per_example_correct must have one value per example")
        for value in (self.factor_l2_per_example, self.effective_delta_l2_per_example):
            if value is not None and value.numel() != size:
                raise ValueError("diagnostic norm must have one value per example")
        object.__setattr__(self, "canonical_global_index", ids)
        object.__setattr__(self, "task_type", _tuple_strings(self.task_type, size=size, name="task_type"))

    @property
    def total_loss_sum(self) -> Tensor:
        return self.loss_sum.sum()

    @property
    def total_token_count(self) -> Tensor:
        return self.token_count.sum()


@dataclass
class EvaluationResult:
    mode: AssignmentMode
    rows: dict[int, BatchEvaluation] = field(default_factory=dict)
    permutation_sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, result: BatchEvaluation) -> None:
        for offset, index in enumerate(result.canonical_global_index.tolist()):
            if int(index) in self.rows:
                raise ValueError(f"duplicate evaluation row {index}")
            self.rows[int(index)] = _slice_batch_evaluation(result, offset)

    @property
    def size(self) -> int:
        return len(self.rows)


def _slice_batch_evaluation(result: BatchEvaluation, index: int) -> BatchEvaluation:
    def one(value: Tensor | None) -> Tensor | None:
        return None if value is None else value[index : index + 1].detach().cpu()

    return BatchEvaluation(
        mode=result.mode,
        canonical_global_index=result.canonical_global_index[index : index + 1],
        loss_sum=result.loss_sum[index : index + 1].detach().cpu(),
        token_count=result.token_count[index : index + 1].detach().cpu(),
        response_nll=result.per_example_nll[index : index + 1].detach().cpu(),
        per_example_nll=result.per_example_nll[index : index + 1].detach().cpu(),
        per_example_correct=one(result.per_example_correct),
        factor_l2_per_example=one(result.factor_l2_per_example),
        effective_delta_l2_per_example=one(result.effective_delta_l2_per_example),
        task_type=(result.task_type[index],),
        logits=None,
        memory_mask=None,
        activation_summary=result.activation_summary,
    )


def response_loss_stats_per_example(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-example response loss sum, mean NLL, and token count.

    The shift and ``-100`` contract exactly match the S2 objective used by
    training.  Reduction is always in FP32, even when the Base runs in BF16.
    """

    if logits.ndim != 3 or labels.ndim != 2 or tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError("expected logits [B,T,V] and labels [B,T] with matching sequence shape")
    shift_logits = logits[:, :-1, :].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    token_count = valid.sum(dim=1)
    if bool((token_count <= 0).any()):
        raise ValueError("some example has no response targets")
    token_losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    loss_sum = (token_losses * valid).sum(dim=1)
    mean_nll = loss_sum / token_count.to(dtype=loss_sum.dtype)
    if not bool(torch.isfinite(mean_nll).all()):
        raise FloatingPointError("response NLL is non-finite")
    return loss_sum, mean_nll, token_count


def response_nll_per_example(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(mean response NLL, response-token count)`` per example.

    This compact two-value API is convenient for callers that only need the
    protocol metric.  ``response_loss_stats_per_example`` exposes the loss
    numerator as well for weighted aggregation.
    """

    _loss_sum, mean_nll, token_count = response_loss_stats_per_example(logits, labels)
    return mean_nll, token_count


def effective_delta_weight_norms(state: Mapping[str, Tensor]) -> Tensor:
    """Return the per-example sum of Frobenius norms of effective Delta-W.

    ``||B A||_F`` is evaluated through rank-sized Gram matrices instead of
    materializing the very large ``out_features x in_features`` products.
    The fixed Generated-LoRA scale is included.
    """

    pairs: list[tuple[Tensor, Tensor]] = []
    for key, value in state.items():
        if key.endswith(".lora_A"):
            base = key[: -len(".lora_A")]
            other = state.get(f"{base}.lora_B")
            if other is None:
                raise ValueError(f"missing B factor for {key}")
            pairs.append((value, other))
    if not pairs:
        raise ValueError("Generated LoRA state has no factor pairs")
    batch = int(pairs[0][0].shape[0])
    result = torch.zeros(batch, device=pairs[0][0].device, dtype=torch.float32)
    scale = state.get("scale")
    if scale is None:
        raise ValueError("Generated LoRA state has no scale")
    scale_value = torch.as_tensor(scale, device=result.device, dtype=torch.float32).abs()
    if scale_value.numel() != 1:
        raise ValueError("S5 Generated-LoRA scale must be scalar")
    for lora_a, lora_b in pairs:
        if lora_a.ndim != 3 or lora_b.ndim != 3:
            raise ValueError("LoRA factors must be batched rank-3 tensors")
        if lora_a.shape[0] != batch or lora_b.shape[0] != batch or lora_a.shape[1] != lora_b.shape[2]:
            raise ValueError("LoRA factor batch/rank shapes are inconsistent")
        a = lora_a.float()
        b = lora_b.float()
        # ||BA||_F^2 = tr((B^T B)(A A^T)).
        b_gram = torch.bmm(b.transpose(1, 2), b)
        a_gram = torch.bmm(a, a.transpose(1, 2))
        squared = (b_gram * a_gram.transpose(1, 2)).sum(dim=(1, 2)).clamp_min(0.0)
        result = result + torch.sqrt(squared)
    return result * scale_value


def factor_l2_per_example(state: Mapping[str, Tensor]) -> Tensor:
    """Return the mean squared generated A/B factor magnitude per example."""

    factors = [value for key, value in state.items() if key.endswith((".lora_A", ".lora_B"))]
    if not factors:
        raise ValueError("Generated LoRA state has no A/B factors")
    batch = int(factors[0].shape[0])
    total = torch.zeros(batch, device=factors[0].device, dtype=torch.float32)
    count = 0
    for value in factors:
        if value.ndim < 2 or value.shape[0] != batch:
            raise ValueError("Generated LoRA factors have inconsistent batch shape")
        flat = value.float().reshape(batch, -1)
        total = total + flat.square().sum(dim=1)
        count += flat.shape[1]
    return total / float(count)


def _as_vector(value: Any, *, expected_id: int) -> Tensor:
    if isinstance(value, TokenizedS5Row):
        actual = value.canonical_global_index
        vector = value.vector
    elif isinstance(value, Mapping):
        actual = value.get("canonical_global_index", expected_id)
        vector = value.get("vector397", value.get("vector"))
    else:
        actual = expected_id
        vector = value
    if int(actual) != int(expected_id):
        raise ValueError(f"vector bank identity mismatch for global index {expected_id}")
    result = torch.as_tensor(vector, dtype=torch.float32)
    if tuple(result.shape) != (397,) or not bool(torch.isfinite(result).all()):
        raise ValueError(f"vector bank entry {expected_id} is not finite FP32[397]")
    return result


def _metadata_for_global_rows(global_rows: Any) -> dict[int, tuple[str, int, Tensor]]:
    """Normalize supported full-bank inputs for deterministic shuffling."""

    result: dict[int, tuple[str, int, Tensor]] = {}

    def add(index: int, value: Any, task: str | None = None) -> None:
        if isinstance(value, EvaluationBatch):
            positions = [int(item) for item in value.canonical_global_index.tolist()]
            for offset, item in enumerate(positions):
                add(item, {
                    "canonical_global_index": item,
                    "task_type": value.task_type[offset],
                    "vector397": value.vector397[offset].detach().cpu(),
                })
            return
        if isinstance(value, TokenizedS5Row):
            item = value.canonical_global_index
            task_value = str(value.task_type)
            vector = _as_vector(value, expected_id=item)
        elif isinstance(value, Mapping):
            item = int(value.get("canonical_global_index", index))
            task_value = str(value.get("task_type", task or ""))
            vector = _as_vector(value, expected_id=item)
        else:
            item = int(index)
            task_value = str(task or "")
            vector = _as_vector(value, expected_id=item)
        if item in result:
            raise ValueError(f"duplicate global evaluation row {item}")
        result[item] = (task_value, 0, vector)

    if isinstance(global_rows, Mapping):
        for key, value in global_rows.items():
            if isinstance(value, EvaluationBatch):
                add(int(key), value)
            else:
                add(int(key), value)
    else:
        for value in global_rows:
            if isinstance(value, EvaluationBatch):
                add(0, value)
            elif isinstance(value, TokenizedS5Row):
                add(value.canonical_global_index, value)
            elif isinstance(value, Mapping):
                add(int(value["canonical_global_index"]), value)
            else:
                raise TypeError("global_rows entries must be EvaluationBatch, TokenizedS5Row, or mappings")
    # Query length is optional in metadata mappings; use zero as a stable
    # fallback.  EvaluationBatch contributes its actual query length below.
    if isinstance(global_rows, Mapping):
        for value in global_rows.values():
            if isinstance(value, EvaluationBatch):
                for offset, item in enumerate(value.canonical_global_index.tolist()):
                    task, _old, vector = result[int(item)]
                    result[int(item)] = (task, int(value.query_mask[offset].sum().item()), vector)
    return result


def _derangement(ids: Sequence[int], *, rng: np.random.Generator) -> dict[int, int]:
    values = list(int(item) for item in ids)
    if len(values) < 2:
        raise ValueError("a derangement requires at least two rows")
    for _ in range(256):
        order = rng.permutation(len(values))
        if np.all(order != np.arange(len(values))):
            return {target: values[int(order[offset])] for offset, target in enumerate(values)}
    # A cyclic shift is a deterministic fail-safe and is a derangement for
    # every group with at least two members.
    shift = int(rng.integers(1, len(values)))
    return {target: values[(offset + shift) % len(values)] for offset, target in enumerate(values)}


def build_global_derangement(
    global_rows: Mapping[int, Any] | Sequence[Any], *, seed: int
) -> dict[int, int]:
    """Build one deterministic, no-self-match assignment for the whole bank."""

    metadata = _metadata_for_global_rows(global_rows)
    if len(metadata) < 2:
        raise ValueError("shuffled evaluation requires at least two global rows")
    rng = np.random.default_rng(int(seed))
    remaining = set(metadata)
    assignment: dict[int, int] = {}

    # First preserve the protocol's preferred same-task/near-length strata.
    exact: dict[tuple[str, int], list[int]] = {}
    for item, (task, qlen, _vector) in metadata.items():
        exact.setdefault((task, qlen), []).append(item)
    singleton: list[int] = []
    for group in sorted(exact.values(), key=lambda values: min(values)):
        if len(group) >= 2:
            assignment.update(_derangement(sorted(group), rng=rng))
            remaining.difference_update(group)
        else:
            singleton.extend(group)

    # Singleton length strata are still kept within task where possible.
    by_task: dict[str, list[int]] = {}
    for item in singleton:
        by_task.setdefault(metadata[item][0], []).append(item)
    leftovers: list[int] = []
    for group in sorted(by_task.values(), key=lambda values: min(values)):
        if len(group) >= 2:
            assignment.update(_derangement(sorted(group), rng=rng))
            remaining.difference_update(group)
        else:
            leftovers.extend(group)

    if leftovers:
        # If every task/length cell is a singleton, a global derangement is
        # the only valid non-leaking control.  Never silently use identity.
        assignment.update(_derangement(sorted(leftovers), rng=rng))
        remaining.difference_update(leftovers)
    if remaining:
        raise RuntimeError("internal shuffled assignment coverage error")
    if any(target == source for target, source in assignment.items()):
        raise RuntimeError("internal shuffled assignment produced a self-match")
    return assignment


def build_unstratified_global_derangement(
    global_rows: Mapping[int, Any] | Sequence[Any], *, seed: int
) -> dict[int, int]:
    """Build one no-self-match permutation without task/length stratification."""

    metadata = _metadata_for_global_rows(global_rows)
    if len(metadata) < 2:
        raise ValueError("global unstratified shuffle requires at least two rows")
    return _derangement(
        sorted(metadata),
        rng=np.random.default_rng(int(seed)),
    )


def build_cross_task_derangement(
    global_rows: Mapping[int, Any] | Sequence[Any], *, seed: int
) -> dict[int, int]:
    """Build a bijection whose source task differs from every target task."""

    metadata = _metadata_for_global_rows(global_rows)
    if len(metadata) < 2:
        raise ValueError("cross-task shuffle requires at least two rows")
    task_counts: dict[str, int] = {}
    for task, _length, _vector in metadata.values():
        task_counts[task] = task_counts.get(task, 0) + 1
    if max(task_counts.values()) * 2 > len(metadata):
        raise ValueError("cross-task derangement is impossible for this task bank")
    targets = np.asarray(sorted(metadata), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    for _ in range(256):
        sources = rng.permutation(targets)
        if all(
            metadata[int(target)][0] != metadata[int(source)][0]
            for target, source in zip(targets, sources, strict=True)
        ):
            return {
                int(target): int(source)
                for target, source in zip(targets, sources, strict=True)
            }
    ordered = sorted(targets.tolist(), key=lambda index: (metadata[index][0], index))
    for shift in range(1, len(ordered)):
        sources = ordered[shift:] + ordered[:shift]
        if all(
            metadata[target][0] != metadata[source][0]
            for target, source in zip(ordered, sources, strict=True)
        ):
            return dict(zip(ordered, sources, strict=True))
    raise RuntimeError("failed to construct cross-task derangement")


def build_assignment_plan(
    batch: EvaluationBatch,
    *,
    mode: AssignmentMode,
    seed: int = 42,
    global_rows: Mapping[int, Any] | Sequence[Any] | None = None,
    block_permutation: Sequence[int] | None = None,
) -> AssignmentPlan:
    """Create a frozen assignment plan for ``batch``.

    For a shuffled plan, ``global_rows`` should describe the complete frozen
    validation bank.  Omitting it is allowed for small synthetic tests, but
    then the batch itself is the assignment population and the resulting
    hash must not be used as a full-bank scientific result.
    """

    normalized = str(mode).lower()
    targets = tuple(int(item) for item in batch.canonical_global_index.tolist())
    if normalized in {
        "shuffled",
        "global_unstratified_shuffled",
        "cross_task_shuffled",
    }:
        population: Any = global_rows if global_rows is not None else {
            item: {
                "canonical_global_index": item,
                "task_type": batch.task_type[offset],
                "vector397": batch.vector397[offset].detach().cpu(),
            }
            for offset, item in enumerate(targets)
        }
        if normalized == "shuffled":
            mapping = build_global_derangement(population, seed=seed)
        elif normalized == "global_unstratified_shuffled":
            mapping = build_unstratified_global_derangement(population, seed=seed)
        else:
            mapping = build_cross_task_derangement(population, seed=seed)
        try:
            sources = tuple(mapping[item] for item in targets)
        except KeyError as error:
            raise ValueError("shuffled global bank does not cover the evaluation batch") from error
    else:
        sources = targets
    block = None if block_permutation is None else tuple(int(item) for item in block_permutation)
    return AssignmentPlan(
        mode=normalized,  # type: ignore[arg-type]
        seed=int(seed),
        target_global_index=targets,
        source_global_index=sources,
        block_permutation=block,
    )


def _vectors_for_plan(
    batch: EvaluationBatch,
    plan: AssignmentPlan,
    vector_bank: Mapping[int, Any] | None,
    global_rows: Mapping[int, Any] | Sequence[Any] | None,
) -> Tensor:
    if plan.mode == "absent":
        # Keep the matched vector in the batch for provenance; runtime removes
        # it with circuit_present=False and therefore never sees a zero input.
        return batch.vector397
    if plan.mode == "zero_present":
        return torch.zeros_like(batch.vector397)
    if plan.mode == "mean_present":
        bank = vector_bank
        if bank is None and global_rows is not None:
            metadata = _metadata_for_global_rows(global_rows)
            bank = {
                index: vector
                for index, (_task, _length, vector) in metadata.items()
            }
        if bank is None:
            mean = batch.vector397.mean(dim=0, keepdim=True)
        else:
            mean = torch.stack(
                [
                    _as_vector(bank[index], expected_id=int(index))
                    for index in sorted(bank)
                ],
                dim=0,
            ).mean(dim=0, keepdim=True)
        return mean.to(
            device=batch.vector397.device,
            dtype=torch.float32,
        ).expand(batch.batch_size, -1)
    if plan.mode in {"matched", "block_permuted"}:
        # Candidate B's block-permuted control changes only the five token
        # contents/addresses inside the encoder.  It is deliberately not a
        # second event assignment, so it must use the current matched vector
        # and must not require an external bank.
        return batch.vector397
    bank = vector_bank
    if bank is None and global_rows is not None:
        metadata = _metadata_for_global_rows(global_rows)
        bank = {index: vector for index, (_task, _length, vector) in metadata.items()}
    if bank is None:
        raise ValueError("a shuffled assignment requires a vector bank")
    values = [_as_vector(bank[int(source)], expected_id=int(source)) for source in plan.source_global_index]
    return torch.stack(values, dim=0).to(device=batch.vector397.device, dtype=torch.float32)


def evaluate_assignment(
    model: Any,
    batch: EvaluationBatch,
    plan: AssignmentPlan,
    *,
    vector_bank: Mapping[int, Any] | None = None,
    global_rows: Mapping[int, Any] | Sequence[Any] | None = None,
    score_fn: Callable[[Tensor, EvaluationBatch], Tensor] | None = None,
    no_grad: bool = True,
    return_logits: bool = False,
) -> BatchEvaluation:
    """Run one frozen assignment with batched Generated-LoRA application."""

    target = tuple(int(item) for item in batch.canonical_global_index.tolist())
    if target != plan.target_global_index:
        raise ValueError("assignment plan target indices do not match batch")
    if plan.mode == "block_permuted" and getattr(model, "candidate", None) not in {"B", "AB"}:
        raise ValueError("block_permuted evaluation is valid only for Candidate B or AB")
    vectors = _vectors_for_plan(batch, plan, vector_bank, global_rows)
    device = batch.query_ids.device
    present = torch.zeros(batch.batch_size, dtype=torch.bool, device=device)
    if plan.mode != "absent":
        present.fill_(True)
    context = torch.no_grad() if no_grad else nullcontext()
    with context:
        state = model.generate_lora(
            batch.query_ids,
            batch.query_mask,
            vectors.to(device=device, dtype=torch.float32),
            circuit_present=present,
            block_permutation=plan.block_permutation,
        )
        with model.controller.activate(state):
            logits = model.base_logits(batch.input_ids)
        loss_sum, mean_nll, token_count = response_loss_stats_per_example(logits, batch.labels)
        factors = factor_l2_per_example(state).detach().cpu()
        effective = effective_delta_weight_norms(state).detach().cpu()
        correct = None
        if score_fn is not None:
            correct = torch.as_tensor(score_fn(logits, batch), device="cpu")
            if correct.numel() != batch.batch_size:
                raise ValueError("score_fn must return one correctness value per example")
            correct = correct.reshape(-1).to(dtype=torch.bool)
        memory_mask = None
        activation_summary = None
        # The state already came from this exact mask.  Avoid a second PG
        # pass; record only the expected compact memory lengths here.
        query_lengths = batch.query_mask.to(dtype=torch.long).sum(dim=1).cpu()
        extra = 0
        if plan.mode != "absent":
            fusion_mode = str(getattr(model, "fusion_mode", "append"))
            if fusion_mode == "append":
                extra = int(getattr(model, "circuit_token_count", 0))
            elif fusion_mode == "split_f3i":
                extra = 1
        memory_mask = torch.arange(int(query_lengths.max().item()) + extra)[None, :] < (
            query_lengths[:, None] + extra
        )
        if hasattr(model, "controller"):
            activation_summary = dict(model.controller.activation_summary())
        return BatchEvaluation(
            mode=plan.mode,
            canonical_global_index=batch.canonical_global_index,
            loss_sum=loss_sum.detach().cpu(),
            token_count=token_count.detach().cpu(),
            response_nll=mean_nll.detach().cpu().mean(),
            per_example_nll=mean_nll.detach().cpu(),
            per_example_correct=correct,
            factor_l2_per_example=factors,
            effective_delta_l2_per_example=effective,
            task_type=batch.task_type,
            logits=logits.detach().cpu() if return_logits else None,
            memory_mask=memory_mask,
            activation_summary=activation_summary,
        )


def _coerce_result(value: EvaluationResult | BatchEvaluation | Sequence[BatchEvaluation], mode: str) -> EvaluationResult:
    if isinstance(value, EvaluationResult):
        return value
    result = EvaluationResult(mode=mode)  # type: ignore[arg-type]
    if isinstance(value, BatchEvaluation):
        result.add(value)
    else:
        for batch in value:
            result.add(batch)
    return result


def _task_accuracy(result: EvaluationResult) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for row in result.rows.values():
        if row.per_example_correct is None:
            continue
        task = row.task_type[0] if row.task_type else ""
        counts.setdefault(task, [0, 0])
        counts[task][0] += int(bool(row.per_example_correct.reshape(-1)[0].item()))
        counts[task][1] += 1
    return {task: correct / total for task, (correct, total) in sorted(counts.items()) if total}


def _paired_values(left: EvaluationResult, right: EvaluationResult, field: str) -> Tensor:
    if set(left.rows) != set(right.rows):
        raise ValueError("paired condition row sets differ")
    values: list[float] = []
    for index in sorted(left.rows):
        a = getattr(left.rows[index], field)
        b = getattr(right.rows[index], field)
        if a is None or b is None:
            raise ValueError(f"paired field {field} is unavailable for row {index}")
        values.append(
            float(
                (
                    a.reshape(-1)[0].to(dtype=torch.float64)
                    - b.reshape(-1)[0].to(dtype=torch.float64)
                ).item()
            )
        )
    return torch.tensor(values, dtype=torch.float64)


def compare_assignments(
    results: Mapping[AssignmentMode, EvaluationResult | BatchEvaluation | Sequence[BatchEvaluation]],
    *,
    expected_task_types: Sequence[str] = (),
) -> dict[str, Any]:
    """Aggregate condition metrics and paired deltas without opening final test."""

    normalized = {str(mode): _coerce_result(value, str(mode)) for mode, value in results.items()}
    summary: dict[str, Any] = {"conditions": {}, "paired": {}}
    for mode, result in normalized.items():
        total_loss = sum(float(row.total_loss_sum.item()) for row in result.rows.values())
        total_tokens = sum(int(row.total_token_count.item()) for row in result.rows.values())
        nlls = [float(row.per_example_nll.reshape(-1)[0].item()) for row in result.rows.values()]
        task_accuracy = _task_accuracy(result)
        task_order = tuple(expected_task_types) or tuple(task_accuracy)
        available = [task_accuracy[item] for item in task_order if item in task_accuracy]
        condition = {
            "count": result.size,
            "response_nll_token_weighted": total_loss / total_tokens if total_tokens else None,
            "response_nll_example_macro": float(np.mean(nlls)) if nlls else None,
            "task_accuracy": task_accuracy,
            "atomic_macro_accuracy": float(np.mean(available)) if available else None,
            "assignment_sha256": result.permutation_sha256,
        }
        summary["conditions"][mode] = condition
    if "matched" in normalized:
        for mode, label in (
            ("shuffled", "shuffled_minus_matched"),
            ("absent", "absent_minus_matched"),
            ("block_permuted", "block_permuted_minus_matched"),
        ):
            if mode not in normalized:
                continue
            nll_delta = _paired_values(normalized[mode], normalized["matched"], "per_example_nll")
            entry: dict[str, Any] = {
                "nll_" + label: float(nll_delta.mean().item()),
                "nll_delta_per_example": nll_delta.tolist(),
            }
            if normalized[mode].rows and all(row.per_example_correct is not None for row in normalized["matched"].rows.values()):
                accuracy_delta = _paired_values(normalized["matched"], normalized[mode], "per_example_correct")
                entry["accuracy_matched_minus_" + mode] = float(accuracy_delta.mean().item())
                entry["accuracy_delta_per_example"] = accuracy_delta.tolist()
            summary["paired"][mode] = entry
    return summary


def gradient_norms_by_group(model: Any) -> dict[str, float]:
    """Return finite gradient norms for the three protocol optimizer groups."""

    if hasattr(model, "group_gradient_norms"):
        return dict(model.group_gradient_norms())
    groups = {
        "pg": getattr(model, "parameter_generator", None),
        "query": getattr(model, "query_encoder", None),
        "circuit": getattr(model, "circuit_encoder", None),
    }
    result: dict[str, float] = {}
    for name, module in groups.items():
        if module is None:
            result[name] = 0.0
            continue
        values = [parameter.grad.detach().float().square().sum() for parameter in module.parameters() if parameter.grad is not None]
        result[name] = float(torch.sqrt(torch.stack(values).sum()).item()) if values else 0.0
    return result


def save_assignment_plan(path: str, plan: AssignmentPlan) -> None:
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_assignment_plan(path: str) -> AssignmentPlan:
    from pathlib import Path

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return AssignmentPlan(
        mode=value["mode"],
        seed=int(value["seed"]),
        target_global_index=tuple(value["target_global_index"]),
        source_global_index=tuple(value["source_global_index"]),
        permutation_sha256=value.get("permutation_sha256"),
        block_permutation=value.get("block_permutation"),
    )


__all__ = [
    "AssignmentMode",
    "AssignmentPlan",
    "BatchEvaluation",
    "EvaluationBatch",
    "EvaluationResult",
    "build_assignment_plan",
    "build_cross_task_derangement",
    "build_global_derangement",
    "build_unstratified_global_derangement",
    "compare_assignments",
    "effective_delta_weight_norms",
    "factor_l2_per_example",
    "gradient_norms_by_group",
    "evaluate_assignment",
    "load_assignment_plan",
    "response_nll_per_example",
    "response_loss_stats_per_example",
    "save_assignment_plan",
]

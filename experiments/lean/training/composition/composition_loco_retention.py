"""Atomic replay and response-only distillation for Direct LOCO SFT.

The module is intentionally independent of the historical composition trainer's
sampling code.  It supplies the small, auditable pieces used by the
``direct_retention4`` protocol: fixed per-task quotas, a restartable sampler,
mixed-cache collation, and a causal response-token KL objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from .composition_loco import ATOMIC_TASKS, COMPOSITION_TASKS
from .composition_loco_circuit_retention20_spec import (
    CIRCUIT_RETENTION20_PROTOCOL_ID,
    circuit_retention20_group_values,
)
from .composition_loco_vector422_spec import VECTOR422_PROTOCOL_ID
from .composition_loco_query_retention20_spec import (
    QUERY_RETENTION20_PROTOCOL_ID,
    query_retention20_group_values,
)


from .composition_loco_raw_retention20_spec import (
    RAW_RETENTION20_PROTOCOL_ID, raw_retention20_group_values,
)

DIRECT_RETENTION4_PROTOCOL_ID = "s5_composition_loco_direct_retention4_v1"
DIRECT_RETENTION32_PROTOCOL_ID = "s5_composition_loco_direct_retention32_12k_v1"
DIRECT_RETENTION4_OUTPUT_ROOT = (
    "outputs/s5_lean_circuit/composition_loco_direct_retention4_v1"
)
DIRECT_RETENTION4_CONFIG_ROOT = (
    "s5_lean_circuit/configs/composition_loco_direct_retention4_v1"
)
DIRECT_RETENTION4_YAML_ROOT = "4server/0913/s5_composition_loco_direct_retention4_v1"
DIRECT_RETENTION4_GROUPS: dict[str, dict[str, Any]] = {
    "D1_replay125_kd050": {
        "composition_per_batch": 448,
        "atomic_per_batch": 64,
        "composition_fraction": 0.875,
        "atomic_fraction": 0.125,
        "kd_lambda": 0.50,
        "composition_per_task": 112,
        "atomic_per_task": 16,
    },
    "D2_replay125_kd025": {
        "composition_per_batch": 448,
        "atomic_per_batch": 64,
        "composition_fraction": 0.875,
        "atomic_fraction": 0.125,
        "kd_lambda": 0.25,
        "composition_per_task": 112,
        "atomic_per_task": 16,
    },
    "D3_replay250_kd050": {
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "kd_lambda": 0.50,
        "composition_per_task": 96,
        "atomic_per_task": 32,
    },
    "D4_replay250_kd025": {
        "composition_per_batch": 384,
        "atomic_per_batch": 128,
        "composition_fraction": 0.75,
        "atomic_fraction": 0.25,
        "kd_lambda": 0.25,
        "composition_per_task": 96,
        "atomic_per_task": 32,
    },
}
DIRECT_RETENTION32_GROUPS: dict[str, dict[str, Any]] = {
    "D1_lr15_replay125_kd050": {"learning_rate": 1.5e-5, "composition_per_batch": 448, "atomic_per_batch": 64, "composition_fraction": 0.875, "atomic_fraction": 0.125, "kd_lambda": 0.50, "composition_per_task": 112, "atomic_per_task": 16},
    "D2_lr15_replay125_kd025": {"learning_rate": 1.5e-5, "composition_per_batch": 448, "atomic_per_batch": 64, "composition_fraction": 0.875, "atomic_fraction": 0.125, "kd_lambda": 0.25, "composition_per_task": 112, "atomic_per_task": 16},
    "D3_lr15_replay250_kd050": {"learning_rate": 1.5e-5, "composition_per_batch": 384, "atomic_per_batch": 128, "composition_fraction": 0.75, "atomic_fraction": 0.25, "kd_lambda": 0.50, "composition_per_task": 96, "atomic_per_task": 32},
    "D4_lr15_replay250_kd025": {"learning_rate": 1.5e-5, "composition_per_batch": 384, "atomic_per_batch": 128, "composition_fraction": 0.75, "atomic_fraction": 0.25, "kd_lambda": 0.25, "composition_per_task": 96, "atomic_per_task": 32},
    "D5_lr075_replay125_kd050": {"learning_rate": 7.5e-6, "composition_per_batch": 448, "atomic_per_batch": 64, "composition_fraction": 0.875, "atomic_fraction": 0.125, "kd_lambda": 0.50, "composition_per_task": 112, "atomic_per_task": 16},
    "D6_lr075_replay125_kd025": {"learning_rate": 7.5e-6, "composition_per_batch": 448, "atomic_per_batch": 64, "composition_fraction": 0.875, "atomic_fraction": 0.125, "kd_lambda": 0.25, "composition_per_task": 112, "atomic_per_task": 16},
    "D7_lr075_replay250_kd050": {"learning_rate": 7.5e-6, "composition_per_batch": 384, "atomic_per_batch": 128, "composition_fraction": 0.75, "atomic_fraction": 0.25, "kd_lambda": 0.50, "composition_per_task": 96, "atomic_per_task": 32},
    "D8_lr075_replay250_kd025": {"learning_rate": 7.5e-6, "composition_per_batch": 384, "atomic_per_batch": 128, "composition_fraction": 0.75, "atomic_fraction": 0.25, "kd_lambda": 0.25, "composition_per_task": 96, "atomic_per_task": 32},
}

RETENTION_GLOBAL_BATCH_SIZE = 512
RETENTION_TOTAL_STEPS = 38_000
RETENTION_KD_TEMPERATURE = 2.0
RETENTION_HELDOUT_TASK = "T_and_or"
RETENTION_COMPOSITION_TASKS = tuple(
    task for task in COMPOSITION_TASKS if task != RETENTION_HELDOUT_TASK
)
RETENTION_ATOMIC_TASK_IDS = {task: index for index, task in enumerate(ATOMIC_TASKS)}
RETENTION_COMPOSITION_TASK_IDS = {
    task: len(ATOMIC_TASKS) + index for index, task in enumerate(COMPOSITION_TASKS)
}


@dataclass(frozen=True)
class RetentionExampleRef:
    """A source-qualified row reference used by the mixed sampler."""

    source: str
    task: str
    index: int

    def __post_init__(self) -> None:
        if self.source not in {"atomic", "composition"}:
            raise ValueError(f"unsupported retention source: {self.source}")
        if int(self.index) < 0:
            raise ValueError("retention row index must be non-negative")


def retention_group(
    group_id: str, *, protocol_id: str = DIRECT_RETENTION4_PROTOCOL_ID
) -> dict[str, Any]:
    """Return a defensive copy of a sealed replay/KD group row."""

    if str(protocol_id) in {CIRCUIT_RETENTION20_PROTOCOL_ID, VECTOR422_PROTOCOL_ID}:
        return circuit_retention20_group_values(str(group_id))
    if str(protocol_id) == RAW_RETENTION20_PROTOCOL_ID:
        return raw_retention20_group_values(str(group_id))
    if str(protocol_id) == QUERY_RETENTION20_PROTOCOL_ID:
        return query_retention20_group_values(str(group_id))
    groups = (
        DIRECT_RETENTION32_GROUPS
        if str(protocol_id) == DIRECT_RETENTION32_PROTOCOL_ID
        else DIRECT_RETENTION4_GROUPS
    )
    try:
        value = groups[str(group_id)]
    except KeyError as exc:
        raise ValueError(
            f"unknown Direct retention group for {protocol_id}: {group_id}"
        ) from exc
    return dict(value)


def validate_retention_group(
    group_id: str,
    *,
    world_size: int = 1,
    protocol_id: str = DIRECT_RETENTION4_PROTOCOL_ID,
    composition_tasks: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate exact global and rank-local quotas for one run."""

    values = retention_group(group_id, protocol_id=protocol_id)
    visible_tasks = tuple(
        RETENTION_COMPOSITION_TASKS if composition_tasks is None else composition_tasks
    )
    if not visible_tasks:
        raise ValueError("retention composition task list cannot be empty")
    if int(values["composition_per_batch"]) + int(values["atomic_per_batch"]) != RETENTION_GLOBAL_BATCH_SIZE:
        raise ValueError("retention group does not fill the global batch")
    if len(visible_tasks) * int(values["composition_per_task"]) != int(values["composition_per_batch"]):
        raise ValueError("composition task quotas do not fill the group quota")
    if len(ATOMIC_TASKS) * int(values["atomic_per_task"]) != int(values["atomic_per_batch"]):
        raise ValueError("atomic task quotas do not fill the group quota")
    if int(world_size) <= 0 or RETENTION_GLOBAL_BATCH_SIZE % int(world_size):
        raise ValueError("retention world size must divide the global batch")
    for key in ("composition_per_task", "atomic_per_task"):
        if int(values[key]) % int(world_size):
            raise ValueError(f"{key} must divide evenly across ranks")
    if not 0.0 < float(values["kd_lambda"]):
        raise ValueError("retention KD lambda must be positive")
    if float(values["composition_fraction"]) + float(values["atomic_fraction"]) != 1.0:
        raise ValueError("retention fractions must sum to one")
    return values


def _check_task_pools(
    pools: Mapping[str, Tensor], expected_tasks: Sequence[str], name: str
) -> dict[str, Tensor]:
    expected = tuple(str(task) for task in expected_tasks)
    if tuple(pools) != expected:
        raise ValueError(f"{name} task order must be {expected}")
    result: dict[str, Tensor] = {}
    for task in expected:
        values = torch.as_tensor(pools[task], dtype=torch.long, device="cpu").reshape(-1)
        if values.numel() == 0:
            raise ValueError(f"{name} task pool is empty: {task}")
        if bool((values < 0).any()) or values.unique().numel() != values.numel():
            raise ValueError(f"{name} task pool contains invalid/duplicate indices: {task}")
        result[task] = values
    return result


class RetentionBatchSampler:
    """Restartable deterministic sampler with exact global task quotas.

    Each task owns a seeded permutation stream.  The row at a given global
    step is a pure function of ``(seed, task, step, quota)``; no mutable cursor
    or dataloader state is required for resume.  Per-task slices are split
    evenly between ranks, so every rank receives both source types even when
    the atomic replay quota is small.
    """

    def __init__(
        self,
        composition_pools: Mapping[str, Tensor],
        atomic_pools: Mapping[str, Tensor],
        *,
        group_id: str,
        seed: int = 42,
        world_size: int = 1,
        composition_tasks: Sequence[str] | None = None,
        heldout_task: str | None = None,
        protocol_id: str = DIRECT_RETENTION4_PROTOCOL_ID,
    ) -> None:
        self.group_id = str(group_id)
        self.protocol_id = str(protocol_id)
        self.heldout_task = (
            RETENTION_HELDOUT_TASK if heldout_task is None else str(heldout_task)
        )
        self.composition_tasks = tuple(
            RETENTION_COMPOSITION_TASKS
            if composition_tasks is None
            else (str(task) for task in composition_tasks)
        )
        self.values = validate_retention_group(
            self.group_id,
            world_size=world_size,
            protocol_id=self.protocol_id,
            composition_tasks=self.composition_tasks,
        )
        self.seed = int(seed)
        self.world_size = int(world_size)
        self.composition_pools = _check_task_pools(
            composition_pools, self.composition_tasks, "composition"
        )
        self.atomic_pools = _check_task_pools(atomic_pools, ATOMIC_TASKS, "atomic")
        self._permutations: dict[tuple[str, str, int], Tensor] = {}

    def _permutation(self, source: str, task: str, epoch: int) -> Tensor:
        key = (source, task, int(epoch))
        if key not in self._permutations:
            pool = (
                self.atomic_pools[task]
                if source == "atomic"
                else self.composition_pools[task]
            )
            generator = torch.Generator(device="cpu")
            task_index = (
                ATOMIC_TASKS.index(task)
                if source == "atomic"
                else self.composition_tasks.index(task)
            )
            source_offset = 0 if source == "atomic" else 10_000_000
            generator.manual_seed(
                self.seed + source_offset + task_index * 1_000_003 + int(epoch) * 97_531
            )
            self._permutations[key] = pool[torch.randperm(pool.numel(), generator=generator)]
        return self._permutations[key]

    def _stream_slice(self, source: str, task: str, *, start: int, count: int) -> Tensor:
        pool = self.atomic_pools[task] if source == "atomic" else self.composition_pools[task]
        if int(count) <= 0:
            return torch.empty(0, dtype=torch.long)
        pieces: list[Tensor] = []
        cursor = int(start)
        remaining = int(count)
        while remaining:
            epoch, offset = divmod(cursor, int(pool.numel()))
            permutation = self._permutation(source, task, epoch)
            take = min(remaining, int(pool.numel()) - offset)
            pieces.append(permutation[offset : offset + take])
            cursor += take
            remaining -= take
        return torch.cat(pieces, dim=0)

    def _rank_refs(self, step: int, rank: int) -> list[RetentionExampleRef]:
        if int(step) < 0:
            raise ValueError("retention step must be non-negative")
        if not 0 <= int(rank) < self.world_size:
            raise ValueError("retention rank is outside world size")
        comp_each = int(self.values["composition_per_task"]) // self.world_size
        atomic_each = int(self.values["atomic_per_task"]) // self.world_size
        refs: list[RetentionExampleRef] = []
        for task in self.composition_tasks:
            total = int(self.values["composition_per_task"])
            start = int(step) * total + int(rank) * comp_each
            refs.extend(
                RetentionExampleRef("composition", task, int(index))
                for index in self._stream_slice(
                    "composition", task, start=start, count=comp_each
                ).tolist()
            )
        for task in ATOMIC_TASKS:
            total = int(self.values["atomic_per_task"])
            start = int(step) * total + int(rank) * atomic_each
            refs.extend(
                RetentionExampleRef("atomic", task, int(index))
                for index in self._stream_slice(
                    "atomic", task, start=start, count=atomic_each
                ).tolist()
            )
        return refs

    def local_batch(self, step: int, *, rank: int) -> tuple[RetentionExampleRef, ...]:
        refs = self._rank_refs(step, int(rank))
        expected = RETENTION_GLOBAL_BATCH_SIZE // self.world_size
        if len(refs) != expected:
            raise RuntimeError("retention rank-local batch cardinality changed")
        return tuple(refs)

    def global_batch(self, step: int) -> tuple[RetentionExampleRef, ...]:
        refs: list[RetentionExampleRef] = []
        for rank in range(self.world_size):
            refs.extend(self._rank_refs(step, rank))
        if len(refs) != RETENTION_GLOBAL_BATCH_SIZE:
            raise RuntimeError("retention global batch cardinality changed")
        return tuple(refs)

    def metadata(self) -> dict[str, Any]:
        values = retention_group(self.group_id, protocol_id=self.protocol_id)
        metadata = {
            "format": "s5_composition_loco_retention_sampler_v1",
            "group_id": self.group_id,
            "seed": self.seed,
            "world_size": self.world_size,
            "global_batch_size": RETENTION_GLOBAL_BATCH_SIZE,
            "composition_tasks": list(self.composition_tasks),
            "atomic_tasks": list(ATOMIC_TASKS),
            "quotas": values,
            "stream": "per_task_epoch_permutation_v1",
            "heldout_task": self.heldout_task,
        }
        # Keep the historical retention4 sampler identity byte-for-byte
        # compatible with checkpoints created before retention32 existed.
        if self.protocol_id != DIRECT_RETENTION4_PROTOCOL_ID:
            metadata["protocol_id"] = self.protocol_id
        return metadata

    def cursor_metadata(self, next_global_step: int) -> dict[str, Any]:
        """Serialize the stateless stream position used by a resume checkpoint."""

        step = int(next_global_step)
        if step < 0:
            raise ValueError("retention next_global_step must be non-negative")
        values = retention_group(self.group_id, protocol_id=self.protocol_id)
        return {
            "format": "s5_composition_loco_retention_cursor_v1",
            "next_global_step": step,
            "per_task_next_offset": {
                **{
                    task: step * int(values["composition_per_task"])
                    for task in self.composition_tasks
                },
                **{
                    task: step * int(values["atomic_per_task"])
                    for task in ATOMIC_TASKS
                },
            },
        }


def build_task_index_pools(
    composition_dataset: Any,
    atomic_dataset: Any,
    *,
    heldout_task: str = RETENTION_HELDOUT_TASK,
    composition_tasks: Sequence[str] | None = None,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Build source-qualified task pools from the two canonical packed views."""

    composition_ids = composition_dataset.payload["task_ids"].index_select(
        0, torch.as_tensor(composition_dataset.indices, dtype=torch.long)
    ).long()
    visible_tasks = tuple(
        RETENTION_COMPOSITION_TASKS if composition_tasks is None else composition_tasks
    )
    if tuple(visible_tasks) != tuple(
        task for task in COMPOSITION_TASKS if task != str(heldout_task)
    ):
        raise ValueError("retention composition tasks must be the four non-heldout tasks")
    heldout_id = COMPOSITION_TASKS.index(str(heldout_task))
    if bool((composition_ids == int(heldout_id)).any()):
        raise ValueError("heldout composition task leaked into the replay fold")
    composition: dict[str, Tensor] = {}
    for task_index, task in enumerate(COMPOSITION_TASKS):
        if task not in visible_tasks:
            continue
        local = torch.nonzero(composition_ids == int(task_index), as_tuple=False).flatten()
        # The sampler indexes the composition dataset's local fold view.
        composition[task] = local
    atomic_ids = atomic_dataset.payload["task_ids"].long()
    atomic: dict[str, Tensor] = {}
    for task_index, task in enumerate(ATOMIC_TASKS):
        atomic[task] = torch.nonzero(atomic_ids == int(task_index), as_tuple=False).flatten()
    # Keep the visible task order, and fail closed if a held-out row leaked
    # into the replay view.
    composition = {task: composition[task] for task in visible_tasks}
    return composition, atomic


def validate_retention_pool_cardinality(
    composition_pools: Mapping[str, Tensor],
    atomic_pools: Mapping[str, Tensor],
    *,
    expected_per_task: int = 300_000,
    composition_tasks: Sequence[str] | None = None,
) -> None:
    """Fail closed unless every replay task has its full canonical train pool.

    The sampler itself deliberately accepts small synthetic pools for unit
    tests.  The formal trainer calls this gate after opening the real caches,
    where silently training on a truncated or evaluation-derived pool would
    invalidate the retention comparison.
    """

    expected = int(expected_per_task)
    if expected <= 0:
        raise ValueError("retention pool cardinality must be positive")
    visible_tasks = tuple(
        RETENTION_COMPOSITION_TASKS if composition_tasks is None else composition_tasks
    )
    for name, pools, tasks in (
        ("composition", composition_pools, visible_tasks),
        ("atomic", atomic_pools, ATOMIC_TASKS),
    ):
        for task in tasks:
            if task not in pools:
                raise ValueError(f"{name} replay pool is missing task: {task}")
            values = torch.as_tensor(pools[task], dtype=torch.long, device="cpu").reshape(-1)
            if values.numel() != expected:
                raise ValueError(
                    f"{name} replay pool cardinality changed for {task}: "
                    f"expected {expected}, got {values.numel()}"
                )


def atomic_response_token_mask(
    labels: Tensor,
    *,
    response_start: Tensor,
    input_lengths: Tensor,
    is_atomic: Tensor,
    special_token_ids: Iterable[int] = (),
) -> Tensor:
    """Return a mask over shifted logits for atomic response tokens only.

    ``labels[:, 1:]`` is the target for ``logits[:, :-1]``.  Therefore the
    absolute target position is ``1..sequence_length-1`` and the first valid
    response position is the stored, tokenizer-derived ``response_start``.
    """

    if labels.ndim != 2:
        raise ValueError("labels must be [batch, sequence]")
    batch, sequence = labels.shape
    response_start = torch.as_tensor(response_start, device=labels.device, dtype=torch.long).reshape(-1)
    input_lengths = torch.as_tensor(input_lengths, device=labels.device, dtype=torch.long).reshape(-1)
    is_atomic = torch.as_tensor(is_atomic, device=labels.device, dtype=torch.bool).reshape(-1)
    if response_start.numel() != batch or input_lengths.numel() != batch or is_atomic.numel() != batch:
        raise ValueError("response metadata does not align with labels")
    if bool(((response_start <= 0) | (response_start >= input_lengths)).any()):
        raise ValueError("response boundaries must leave prompt and response tokens")
    positions = torch.arange(1, sequence, device=labels.device, dtype=torch.long)[None, :]
    mask = (
        is_atomic[:, None]
        & (positions >= response_start[:, None])
        & (positions < input_lengths[:, None])
        & (labels[:, 1:] != -100)
    )
    special = tuple(int(value) for value in special_token_ids)
    if special:
        special_tensor = torch.tensor(special, device=labels.device, dtype=labels.dtype)
        mask &= ~torch.isin(labels[:, 1:], special_tensor)
    return mask


def atomic_kd_loss_sum(
    student_logits: Tensor,
    teacher_logits: Tensor,
    response_mask: Tensor,
    *,
    temperature: float = RETENTION_KD_TEMPERATURE,
) -> tuple[Tensor, int]:
    """Return ``(sum, count)`` for ``KL(teacher || student)`` at temperature T."""

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher/student logits must have identical shapes")
    if student_logits.ndim != 3:
        raise ValueError("teacher/student logits must be [batch, sequence, vocab]")
    expected = (student_logits.shape[0], student_logits.shape[1] - 1)
    if tuple(response_mask.shape) != expected:
        raise ValueError(f"response_mask must have shape {expected}")
    if not float(temperature) > 0:
        raise ValueError("KD temperature must be positive")
    shifted_student = student_logits[:, :-1].float() / float(temperature)
    shifted_teacher = teacher_logits[:, :-1].float() / float(temperature)
    token_kl = F.kl_div(
        F.log_softmax(shifted_student, dim=-1),
        F.softmax(shifted_teacher, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    mask = response_mask.to(device=token_kl.device, dtype=torch.bool)
    count = int(mask.sum().item())
    if count == 0:
        return token_kl.sum() * 0.0, 0
    return token_kl.masked_select(mask).sum() * (float(temperature) ** 2), count


def collate_retention_batch(
    composition_dataset: Any,
    atomic_dataset: Any,
    refs: Sequence[RetentionExampleRef],
    *,
    device: torch.device | str,
    include_vector: bool = True,
    expected_vector_dim: int = 397,
) -> dict[str, Tensor]:
    """Collate composition and atomic packed rows into one retention batch.

    Query-only and Direct routes do not consume Vector397.  They can request a
    vector-free batch so replay remains text/event based; Circuit-A keeps the
    default real-vector path.
    """

    if not refs:
        raise ValueError("retention batch cannot be empty")
    rows: list[tuple[Any, int, bool, str]] = []
    for ref in refs:
        if ref.source == "composition":
            dataset, index = composition_dataset, int(ref.index)
        else:
            dataset, index = atomic_dataset, int(ref.index)
        if index < 0 or index >= len(dataset):
            raise IndexError(f"{ref.source} retention index is outside its dataset")
        rows.append((dataset, index, ref.source == "atomic", ref.task))
    def payload_index(dataset: Any, index: int) -> int:
        return int(dataset.indices[index]) if hasattr(dataset, "indices") else int(index)

    query_lengths = torch.tensor(
        [
            int(dataset.payload["query_lengths"][payload_index(dataset, index)])
            for dataset, index, _, _ in rows
        ],
        dtype=torch.long,
    )
    input_lengths = torch.tensor(
        [
            int(dataset.payload["input_lengths"][payload_index(dataset, index)])
            for dataset, index, _, _ in rows
        ],
        dtype=torch.long,
    )
    response_start = torch.tensor(
        [
            int(dataset.payload["response_start"][payload_index(dataset, index)])
            for dataset, index, _, _ in rows
        ],
        dtype=torch.long,
    )
    query_width = int(query_lengths.max().item())
    input_width = int(input_lengths.max().item())
    query = torch.zeros((len(rows), query_width), dtype=torch.long)
    full = torch.zeros((len(rows), input_width), dtype=torch.long)
    for row, (dataset, index, _, _) in enumerate(rows):
        # Composition refs index the fold-local view; atomic refs index their
        # packed cache directly.  Never let the two integer namespaces alias.
        payload_index = (
            int(dataset.indices[index])
            if hasattr(dataset, "indices")
            else int(index)
        )
        q_len = int(query_lengths[row].item())
        i_len = int(input_lengths[row].item())
        query[row, :q_len] = dataset.payload["query_ids"][payload_index, :q_len].long()
        full[row, :i_len] = dataset.payload["input_ids"][payload_index, :i_len].long()
    query_mask = torch.arange(query_width)[None, :] < query_lengths[:, None]
    input_mask = torch.arange(input_width)[None, :] < input_lengths[:, None]
    labels = full.clone()
    labels.masked_fill_(~input_mask, -100)
    is_atomic = torch.tensor([atomic for _, _, atomic, _ in rows], dtype=torch.bool)
    # Keep the two index namespaces explicit.  Composition Base-CE is keyed
    # by the canonical 1.5M cache index, whereas atomic replay is keyed by its
    # independent 1.2M cache.  ``global_indices`` is retained for historical
    # callers, but is -1 for atomic rows and must never be used to index the
    # composition CE artifact without the mask below.
    composition_global_indices = torch.full(
        (len(rows),), -1, dtype=torch.long
    )
    for row, (dataset, index, atomic, _task) in enumerate(rows):
        if not atomic:
            payload_index = int(dataset.indices[index]) if hasattr(dataset, "indices") else int(index)
            composition_global_indices[row] = payload_index
    task_ids = torch.tensor(
        [
            (RETENTION_ATOMIC_TASK_IDS[task] if atomic else RETENTION_COMPOSITION_TASK_IDS[task])
            for _, _, atomic, task in rows
        ],
        dtype=torch.long,
    )
    source_indices = torch.tensor([index for _, index, _, _ in rows], dtype=torch.long)
    vector397: Tensor | None = None
    if include_vector:
        vectors: list[Tensor] = []
        for dataset, index, _atomic, _task in rows:
            payload_index = int(dataset.indices[index]) if hasattr(dataset, "indices") else int(index)
            if hasattr(dataset, "condition_vectors"):
                value = dataset.condition_vectors(torch.tensor([payload_index], dtype=torch.long))[0]
            else:
                value = dataset.payload["vectors"][payload_index]
                normalizer = getattr(dataset, "fingerprint_normalizer", None)
                if normalizer is not None:
                    value = torch.from_numpy(
                        normalizer.transform_fixed_log1p_batch(
                            value.reshape(1, -1).numpy()
                        ).astype("float32", copy=False)[0]
                    )
            value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            if value.numel() != int(expected_vector_dim) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"retention circuit Vector{expected_vector_dim} row is missing or non-finite")
            vectors.append(value)
        vector397 = torch.stack(vectors, dim=0)
    payload = {
        "query_ids": query,
        "query_mask": query_mask,
        "input_ids": full,
        "input_mask": input_mask,
        "labels": labels,
        "query_lengths": query_lengths,
        "input_lengths": input_lengths,
        "response_start": response_start,
        "is_atomic": is_atomic,
        "task_ids": task_ids,
        "source_indices": source_indices,
        "global_indices": composition_global_indices,
        "composition_global_indices": composition_global_indices,
    }
    if vector397 is not None:
        payload["vector397"] = vector397
    return {key: value.to(device=device, non_blocking=True) for key, value in payload.items()}


def load_atomic_replay_dataset(
    replay_config: Mapping[str, Any],
    *,
    expected_tokenizer_sha256: str,
    expected_base_identity: Mapping[str, Any],
    include_vector: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Open the verified 1.2M atomic *training* cache, never an eval bank."""

    from .hypernet.packed_cache import (
        PackedTokenizedVector397Dataset,
        validate_packed_cache,
    )

    def required_path(key: str) -> Path:
        value = replay_config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"atomic replay provenance field is missing: {key}")
        path = Path(value)
        return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()

    cache_path = required_path("cache_path")
    manifest_path = required_path("cache_manifest")
    source_config_path = required_path("source_config_path")
    if "atomic_retention" in cache_path.name.lower() or "atomic_retention" in str(cache_path.parent).lower():
        raise ValueError("atomic_retention evaluation bank cannot be used as replay data")
    if not manifest_path.is_file() or manifest_path != cache_path.with_suffix(cache_path.suffix + ".json"):
        raise ValueError("atomic replay cache manifest does not accompany the payload")
    if not source_config_path.is_file():
        raise FileNotFoundError(f"atomic replay source config is missing: {source_config_path}")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if not isinstance(source_config, dict):
        raise ValueError("atomic replay source config must be a JSON object")
    source_config_body = dict(source_config)
    declared_source_config_sha = source_config_body.pop("config_sha256", None)
    source_config_sha = hashlib.sha256(
        json.dumps(source_config_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    if (
        declared_source_config_sha != source_config_sha
        or source_config_sha != str(replay_config.get("expected_source_config_sha256"))
    ):
        raise ValueError("atomic replay source-config identity changed")
    metadata = validate_packed_cache(cache_path, verify_payload_checksum=True)
    expected = {
        "prefix": int(replay_config.get("expected_prefix", -1)),
        "n_ctx": int(replay_config.get("expected_n_ctx", -1)),
    }
    if expected != {"prefix": 1_200_000, "n_ctx": 128}:
        raise ValueError("atomic replay dimensions are not the formal 1.2M/128 contract")
    if int(metadata.get("prefix", -1)) != expected["prefix"] or int(metadata.get("n_ctx", -1)) != expected["n_ctx"]:
        raise ValueError("atomic replay cache dimensions changed")
    if metadata.get("full_sequence_labels") is not True or metadata.get("task_to_id") != RETENTION_ATOMIC_TASK_IDS:
        raise ValueError("atomic replay cache task/label contract changed")
    if metadata.get("tokenizer_sha256") != str(expected_tokenizer_sha256):
        raise ValueError("atomic replay tokenizer differs from the composition cache")
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    source_base = source_config.get("base", {})
    for key, value in expected_base_identity.items():
        if source_base.get(key) != value:
            raise ValueError(f"atomic replay source config is not bound to Stage-A Base: {key}")
    dataset = PackedTokenizedVector397Dataset(
        cache_path,
        verify_payload_checksum=False,
        include_vector=bool(include_vector),
    )
    return dataset, {
        "format": "s5_composition_loco_atomic_replay_source_v1",
        "cache_path": str(cache_path),
        "cache_manifest": str(manifest_path),
        "cache_identity_sha256": metadata["identity_sha256"],
        "cache_payload_sha256": metadata["payload_sha256"],
        "cache_payload_bytes": int(metadata["payload_bytes"]),
        "source_config_path": str(source_config_path),
        "source_config_sha256": source_config_sha,
        "prefix": int(metadata["prefix"]),
        "n_ctx": int(metadata["n_ctx"]),
        "task_to_id": dict(metadata["task_to_id"]),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "include_vector": bool(include_vector),
    }


__all__ = [
    "DIRECT_RETENTION4_PROTOCOL_ID",
    "DIRECT_RETENTION32_PROTOCOL_ID",
    "CIRCUIT_RETENTION20_PROTOCOL_ID",
    "QUERY_RETENTION20_PROTOCOL_ID",
    "DIRECT_RETENTION4_OUTPUT_ROOT",
    "DIRECT_RETENTION4_CONFIG_ROOT",
    "DIRECT_RETENTION4_YAML_ROOT",
    "DIRECT_RETENTION4_GROUPS",
    "DIRECT_RETENTION32_GROUPS",
    "RETENTION_GLOBAL_BATCH_SIZE",
    "RETENTION_TOTAL_STEPS",
    "RETENTION_KD_TEMPERATURE",
    "RETENTION_HELDOUT_TASK",
    "RETENTION_COMPOSITION_TASKS",
    "RetentionExampleRef",
    "RetentionBatchSampler",
    "retention_group",
    "validate_retention_group",
    "build_task_index_pools",
    "validate_retention_pool_cardinality",
    "atomic_response_token_mask",
    "atomic_kd_loss_sum",
    "collate_retention_batch",
    "load_atomic_replay_dataset",
]

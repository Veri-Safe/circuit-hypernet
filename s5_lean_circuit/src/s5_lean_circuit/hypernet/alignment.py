"""Parameter-free query/circuit correspondence objectives for S5 HyperNet."""

from __future__ import annotations

from typing import Literal

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import Tensor
from torch.nn import functional as F


AlignmentMode = Literal[
    "none",
    "global_inbatch_infonce_v1",
    "same_task_inbatch_infonce_v1",
    "hybrid_inbatch_infonce_v1",
]
ALIGNMENT_MODES = {
    "none",
    "global_inbatch_infonce_v1",
    "same_task_inbatch_infonce_v1",
    "hybrid_inbatch_infonce_v1",
}


def _pooled_condition_parts(
    memory: Tensor,
    query_mask: Tensor,
    *,
    circuit_token_count: int,
) -> tuple[Tensor, Tensor]:
    if memory.ndim != 3 or query_mask.ndim != 2:
        raise ValueError("alignment memory/query mask rank changed")
    if memory.shape[0] != query_mask.shape[0] or circuit_token_count <= 0:
        raise ValueError("alignment batch/token dimensions changed")
    lengths = query_mask.to(device=memory.device, dtype=torch.long).sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("alignment query contains no valid token")
    positions = torch.arange(memory.shape[1], device=memory.device)[None, :]
    query_positions = positions < lengths[:, None]
    circuit_positions = (
        (positions >= lengths[:, None])
        & (positions < lengths[:, None] + int(circuit_token_count))
    )
    if bool((circuit_positions.sum(dim=1) != int(circuit_token_count)).any()):
        raise ValueError("alignment circuit tokens are absent from compact memory")
    query = (
        memory.float() * query_positions[:, :, None]
    ).sum(dim=1) / lengths[:, None].float()
    circuit = (
        memory.float() * circuit_positions[:, :, None]
    ).sum(dim=1) / float(circuit_token_count)
    return query, circuit


def _fixed_feature_projection(values: Tensor, width: int) -> Tensor:
    input_width = values.shape[-1]
    if int(width) == input_width:
        return values
    if int(width) <= 0 or input_width % int(width):
        raise ValueError("alignment projection width must divide the token width")
    indices = torch.arange(input_width, device=values.device)
    permutation = (indices * 193 + 17) % input_width
    signs = torch.where(indices % 2 == 0, 1.0, -1.0).to(
        device=values.device, dtype=values.dtype
    )
    grouped = (values.index_select(-1, permutation) * signs).reshape(
        values.shape[0], int(width), input_width // int(width)
    )
    return grouped.mean(dim=-1) * (input_width // int(width)) ** 0.5


def _symmetric_cross_entropy_sum(logits: Tensor, allowed: Tensor | None) -> Tensor:
    batch = logits.shape[0]
    labels = torch.arange(batch, device=logits.device)
    values = logits
    if allowed is not None:
        if allowed.shape != logits.shape or not bool(torch.diagonal(allowed).all()):
            raise ValueError("alignment allowed-pair mask changed")
        values = values.masked_fill(~allowed, torch.finfo(values.dtype).min)
    row = F.cross_entropy(values, labels, reduction="sum")
    column = F.cross_entropy(values.transpose(0, 1), labels, reduction="sum")
    return 0.5 * (row + column)


def query_circuit_alignment_stats(
    memory: Tensor,
    query_mask: Tensor,
    *,
    circuit_token_count: int,
    mode: AlignmentMode | str,
    temperature: float,
    projection_width: int = 128,
    task_ids: Tensor | None = None,
) -> dict[str, Tensor]:
    normalized_mode = str(mode)
    if normalized_mode not in ALIGNMENT_MODES:
        raise ValueError(f"unsupported query/circuit alignment mode: {mode}")
    if not 0.0 < float(temperature) <= 1.0:
        raise ValueError("alignment temperature must be in (0,1]")
    query, circuit = _pooled_condition_parts(
        memory,
        query_mask,
        circuit_token_count=circuit_token_count,
    )
    query = F.normalize(_fixed_feature_projection(query, projection_width), dim=-1)
    circuit = F.normalize(_fixed_feature_projection(circuit, projection_width), dim=-1)
    if dist.is_available() and dist.is_initialized():
        query = torch.cat(tuple(dist_nn.all_gather(query)), dim=0)
        circuit = torch.cat(tuple(dist_nn.all_gather(circuit)), dim=0)
        if task_ids is not None:
            local_tasks = task_ids.to(
                device=query.device, dtype=torch.long
            ).contiguous()
            gathered_tasks = [
                torch.empty_like(local_tasks) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_tasks, local_tasks)
            task_ids = torch.cat(gathered_tasks, dim=0)
    batch = query.shape[0]
    if batch < 2 and normalized_mode != "none":
        raise ValueError("in-batch alignment requires at least two events")
    similarity = query @ circuit.transpose(0, 1)
    diagonal = torch.diagonal(similarity)
    off_diagonal = ~torch.eye(batch, device=similarity.device, dtype=torch.bool)
    negative_sum = similarity.masked_select(off_diagonal).sum()
    negative_count = off_diagonal.sum()
    if task_ids is None:
        same_task_negative_sum = similarity.sum() * 0.0
        same_task_negative_count = torch.zeros(
            (), device=similarity.device, dtype=torch.long
        )
    else:
        tasks = task_ids.to(device=similarity.device)
        same_task_negative_mask = off_diagonal & (tasks[:, None] == tasks[None, :])
        same_task_negative_sum = similarity.masked_select(
            same_task_negative_mask
        ).sum()
        same_task_negative_count = same_task_negative_mask.sum()
    labels = torch.arange(batch, device=similarity.device)
    retrieval_correct = 0.5 * (
        (similarity.argmax(dim=1) == labels).float().sum()
        + (similarity.argmax(dim=0) == labels).float().sum()
    )
    zero = similarity.sum() * 0.0
    if normalized_mode == "none":
        loss_sum = zero
    else:
        logits = similarity / float(temperature)
        global_sum = _symmetric_cross_entropy_sum(logits, None)
        same_task_sum = zero
        if normalized_mode in {
            "same_task_inbatch_infonce_v1",
            "hybrid_inbatch_infonce_v1",
        }:
            if task_ids is None or task_ids.shape != (batch,):
                raise ValueError("same-task alignment requires task_ids [B]")
            tasks = task_ids.to(device=similarity.device)
            allowed = tasks[:, None] == tasks[None, :]
            same_task_sum = _symmetric_cross_entropy_sum(logits, allowed)
        if normalized_mode == "global_inbatch_infonce_v1":
            loss_sum = global_sum
        elif normalized_mode == "same_task_inbatch_infonce_v1":
            loss_sum = same_task_sum
        else:
            loss_sum = 0.5 * (global_sum + same_task_sum)
    return {
        "loss_sum": loss_sum,
        "event_count": torch.tensor(batch, device=memory.device, dtype=torch.long),
        "positive_similarity_sum": diagonal.sum(),
        "negative_similarity_sum": negative_sum,
        "negative_similarity_count": negative_count,
        "same_task_negative_similarity_sum": same_task_negative_sum,
        "same_task_negative_similarity_count": same_task_negative_count,
        "retrieval_correct_sum": retrieval_correct,
    }


__all__ = [
    "ALIGNMENT_MODES",
    "AlignmentMode",
    "query_circuit_alignment_stats",
]

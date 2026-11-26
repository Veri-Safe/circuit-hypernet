from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PrunedGraph:
    node_mask: Tensor
    edge_mask: Tensor
    node_influence: Tensor
    node_cumulative_score: Tensor
    edge_influence: Tensor


def normalize_direct_effect_rows(matrix: Tensor, *, epsilon: float = 1e-10) -> Tensor:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Direct-effect matrix must be square")
    magnitude = matrix.abs()
    return magnitude / magnitude.sum(dim=1, keepdim=True).clamp_min(epsilon)


def propagate_influence(
    normalized_matrix: Tensor,
    root_weights: Tensor,
    *,
    max_iter: int | None = None,
) -> Tensor:
    if (
        normalized_matrix.ndim != 2
        or normalized_matrix.shape[0] != normalized_matrix.shape[1]
    ):
        raise ValueError("Influence matrix must be square")
    if root_weights.shape != (normalized_matrix.shape[0],):
        raise ValueError("Root weights do not align with influence matrix")
    limit = int(max_iter or normalized_matrix.shape[0] + 1)
    current = root_weights @ normalized_matrix
    influence = current.clone()
    for _ in range(limit):
        if not torch.any(current != 0):
            return influence
        current = current @ normalized_matrix
        influence = influence + current
    raise RuntimeError(
        "Influence propagation did not converge; graph may contain a cycle"
    )


def threshold_for_cumulative_fraction(scores: Tensor, fraction: float) -> Tensor:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("Influence threshold fraction must lie in [0, 1]")
    flat = scores.flatten()
    positive = flat[flat > 0]
    if positive.numel() == 0:
        return torch.tensor(float("inf"), device=scores.device, dtype=scores.dtype)
    sorted_scores = torch.sort(positive, descending=True).values
    cumulative = torch.cumsum(sorted_scores, dim=0) / sorted_scores.sum()
    index = min(
        int(torch.searchsorted(cumulative, fraction).item()), len(cumulative) - 1
    )
    return sorted_scores[index]


def cumulative_rank_scores(scores: Tensor) -> Tensor:
    if scores.ndim != 1:
        raise ValueError("Node scores must be one-dimensional")
    result = torch.zeros_like(scores)
    total = scores.sum()
    if not torch.isfinite(total) or total <= 0:
        return result
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    result[sorted_indices] = torch.cumsum(sorted_scores, dim=0) / total
    return result


def prune_direct_effect_graph(
    signed_direct_effect: Tensor,
    *,
    token_indices: Tensor,
    root_index: int,
    feature_count: int,
    node_threshold: float,
    edge_threshold: float,
) -> PrunedGraph:
    """Apply CodeCircuit influence propagation and one global pruning policy."""

    matrix = signed_direct_effect.float()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("signed_direct_effect must be square")
    node_count = matrix.shape[0]
    if not 0 <= root_index < node_count or not 0 <= feature_count <= node_count:
        raise ValueError("Root or feature boundary is outside the graph")
    token_indices = token_indices.to(device=matrix.device, dtype=torch.long)
    if token_indices.numel() and (
        int(token_indices.min()) < 0 or int(token_indices.max()) >= node_count
    ):
        raise ValueError("Token index is outside the graph")

    normalized = normalize_direct_effect_rows(matrix)
    root_weights = torch.zeros(node_count, device=matrix.device, dtype=matrix.dtype)
    root_weights[root_index] = 1.0
    node_influence = propagate_influence(normalized, root_weights)
    node_cutoff = threshold_for_cumulative_fraction(node_influence, node_threshold)
    node_mask = node_influence >= node_cutoff
    node_mask[token_indices] = True
    node_mask[root_index] = True

    pruned_matrix = matrix.clone()
    pruned_matrix[~node_mask] = 0
    pruned_matrix[:, ~node_mask] = 0
    normalized_pruned = normalize_direct_effect_rows(pruned_matrix)
    pruned_influence = propagate_influence(normalized_pruned, root_weights)
    output_influence = pruned_influence + root_weights
    edge_influence = normalized_pruned * output_influence[:, None]
    edge_cutoff = threshold_for_cumulative_fraction(edge_influence, edge_threshold)
    edge_mask = (edge_influence >= edge_cutoff) & (normalized_pruned > 0)

    protected = torch.zeros(node_count, dtype=torch.bool, device=matrix.device)
    protected[token_indices] = True
    protected[root_index] = True
    previous = ~node_mask
    while not torch.equal(previous, node_mask):
        previous = node_mask.clone()
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        nonprotected = ~protected
        node_mask[nonprotected] &= edge_mask[:, nonprotected].any(dim=0)
        if feature_count:
            feature_slice = slice(0, feature_count)
            node_mask[feature_slice] &= edge_mask[feature_slice].any(dim=1)
        node_mask[protected] = True

    edge_mask[~node_mask] = False
    edge_mask[:, ~node_mask] = False
    return PrunedGraph(
        node_mask=node_mask,
        edge_mask=edge_mask,
        node_influence=node_influence,
        node_cumulative_score=cumulative_rank_scores(node_influence),
        edge_influence=edge_influence,
    )


def prune_edges_after_node_selection(
    signed_direct_effect: Tensor,
    *,
    token_indices: Tensor,
    root_index: int,
    feature_count: int,
    edge_threshold: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run CodeCircuit edge pruning after the node threshold is already applied."""

    matrix = signed_direct_effect.float()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("signed_direct_effect must be square")
    node_count = int(matrix.shape[0])
    if not 0 <= root_index < node_count or not 0 <= feature_count <= node_count:
        raise ValueError("Root or feature boundary is outside the selected graph")
    token_indices = token_indices.to(device=matrix.device, dtype=torch.long)
    normalized = normalize_direct_effect_rows(matrix)
    root_weights = torch.zeros(node_count, device=matrix.device, dtype=matrix.dtype)
    root_weights[root_index] = 1.0
    pruned_influence = propagate_influence(normalized, root_weights)
    output_influence = pruned_influence + root_weights
    edge_influence = normalized * output_influence[:, None]
    edge_cutoff = threshold_for_cumulative_fraction(edge_influence, edge_threshold)
    edge_mask = (edge_influence >= edge_cutoff) & (normalized > 0)

    protected = torch.zeros(node_count, dtype=torch.bool, device=matrix.device)
    protected[token_indices] = True
    protected[root_index] = True
    node_mask = torch.ones(node_count, dtype=torch.bool, device=matrix.device)
    previous = ~node_mask
    while not torch.equal(previous, node_mask):
        previous = node_mask.clone()
        edge_mask[~node_mask] = False
        edge_mask[:, ~node_mask] = False
        node_mask[~protected] &= edge_mask[:, ~protected].any(dim=0)
        if feature_count:
            node_mask[:feature_count] &= edge_mask[:feature_count].any(dim=1)
        node_mask[protected] = True
    edge_mask[~node_mask] = False
    edge_mask[:, ~node_mask] = False
    return node_mask, edge_mask, edge_influence

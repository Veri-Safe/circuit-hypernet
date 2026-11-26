from __future__ import annotations

from dataclasses import dataclass
import networkx as nx
import numpy as np

from .payload import CompactPayload, NodeType
from .sparse_topology import (
    build_sparse_dag,
    k_best_paths_to_sink,
    numba_available,
)


TRAIL_SCHEMA = "nvarc_qwen3_joint_span_trails_v1"


@dataclass(frozen=True)
class TrailNode:
    logical_index: int
    node_type: int
    layer: int
    position: int
    feature_or_token_id: int
    activation: float
    influence: float


@dataclass(frozen=True)
class AttributionTrail:
    node_indices: tuple[int, ...]
    signed_direct_effects: tuple[float, ...]
    nonnegative_influences: tuple[float, ...]
    normalized_strengths: tuple[float, ...]
    edge_costs: tuple[float, ...]
    total_cost: float
    score: float


@dataclass(frozen=True)
class TrailResult:
    event_key: str
    artifact_id: str
    nodes: tuple[TrailNode, ...]
    trails: tuple[AttributionTrail, ...]
    no_valid_trail: bool
    schema: str = TRAIL_SCHEMA


def _node(payload: CompactPayload, index: int) -> TrailNode:
    if index < payload.feature_count:
        return TrailNode(
            index,
            int(NodeType.FEATURE),
            int(payload.feature_layer[index]),
            int(payload.feature_position[index]),
            int(payload.feature_id[index]),
            float(payload.feature_activation[index]),
            float(payload.feature_influence[index]),
        )
    error_offset = index - payload.feature_count
    if error_offset < payload.error_count:
        return TrailNode(
            index,
            int(NodeType.ERROR),
            int(payload.error_layer[error_offset]),
            int(payload.error_position[error_offset]),
            -1,
            0.0,
            float(payload.error_influence[error_offset]),
        )
    token_offset = index - payload.token_start
    if token_offset < payload.token_count:
        node_type = (
            NodeType.PROMPT_TOKEN
            if token_offset < payload.condition_length
            else NodeType.GENERATED_PREFIX_TOKEN
        )
        return TrailNode(
            index,
            int(node_type),
            -1,
            token_offset,
            int(payload.replay_token_ids[token_offset]),
            0.0,
            float(payload.token_influence[token_offset]),
        )
    if index == payload.root_index:
        return TrailNode(
            index,
            int(NodeType.SPAN_ROOT),
            -1,
            -1,
            -1,
            0.0,
            float(payload.root_influence),
        )
    raise IndexError(f"Node index {index} is outside the payload")


def _trail_graph(payload: CompactPayload, epsilon: float) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(payload.node_count))
    sources = payload.edge_source.tolist()
    targets = payload.edge_target.tolist()
    influences = payload.edge_nonnegative_influence.float().tolist()
    row_totals: dict[int, float] = {}
    for target, influence in zip(targets, influences, strict=True):
        row_totals[int(target)] = row_totals.get(int(target), 0.0) + float(influence)
    for edge_index, (source, target, influence) in enumerate(
        zip(sources, targets, influences, strict=True)
    ):
        total = row_totals[int(target)]
        if influence <= 0.0 or total <= 0.0:
            continue
        q = float(influence) / total
        cost = -float(np.log(np.clip(q, epsilon, 1.0)))
        graph.add_edge(
            int(source),
            int(target),
            edge_index=edge_index,
            normalized_strength=q,
            cost=cost,
        )
    return graph


def extract_top_k_trails(
    payload: CompactPayload,
    *,
    max_trails: int = 8,
    epsilon: float = 1e-12,
) -> TrailResult:
    payload.validate()
    if max_trails <= 0 or not 0.0 < epsilon < 1.0:
        raise ValueError("Trail limits or epsilon are invalid")
    if numba_available():
        return _extract_top_k_trails_sparse(
            payload, max_trails=max_trails, epsilon=epsilon
        )
    graph = _trail_graph(payload, epsilon)
    topological = list(nx.topological_sort(graph))
    paths: dict[int, list[tuple[float, tuple[int, ...]]]] = {
        payload.root_index: [(0.0, (payload.root_index,))]
    }
    for node in reversed(topological):
        if node == payload.root_index:
            continue
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for successor in sorted(graph.successors(node)):
            edge_cost = float(graph[node][successor]["cost"])
            for tail_cost, tail_nodes in paths.get(successor, []):
                candidates.append((edge_cost + tail_cost, (node, *tail_nodes)))
        candidates.sort(key=lambda item: (item[0], item[1]))
        deduplicated = []
        seen = set()
        for candidate in candidates:
            if candidate[1] in seen:
                continue
            seen.add(candidate[1])
            deduplicated.append(candidate)
            if len(deduplicated) == max_trails:
                break
        paths[node] = deduplicated

    candidates = []
    for source in range(payload.token_start, payload.root_index):
        candidates.extend(paths.get(source, []))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[tuple[float, tuple[int, ...]]] = []
    seen_paths = set()
    for candidate in candidates:
        if candidate[1] in seen_paths:
            continue
        seen_paths.add(candidate[1])
        selected.append(candidate)
        if len(selected) == max_trails:
            break

    trails = []
    used_nodes = set()
    for total_cost, node_indices in selected:
        signed = []
        influence = []
        strengths = []
        costs = []
        for source, target in zip(node_indices[:-1], node_indices[1:], strict=True):
            edge = graph[source][target]
            edge_index = int(edge["edge_index"])
            signed.append(float(payload.edge_signed_direct_effect[edge_index]))
            influence.append(float(payload.edge_nonnegative_influence[edge_index]))
            strengths.append(float(edge["normalized_strength"]))
            costs.append(float(edge["cost"]))
        used_nodes.update(node_indices)
        trails.append(
            AttributionTrail(
                node_indices=node_indices,
                signed_direct_effects=tuple(signed),
                nonnegative_influences=tuple(influence),
                normalized_strengths=tuple(strengths),
                edge_costs=tuple(costs),
                total_cost=float(total_cost),
                score=float(np.exp(-total_cost)),
            )
        )
    nodes = tuple(_node(payload, index) for index in sorted(used_nodes))
    return TrailResult(
        event_key=payload.event_key,
        artifact_id=payload.artifact_id,
        nodes=nodes,
        trails=tuple(trails),
        no_valid_trail=not trails,
    )


def _extract_top_k_trails_sparse(
    payload: CompactPayload, *, max_trails: int, epsilon: float
) -> TrailResult:
    sources = payload.edge_source.numpy()
    targets = payload.edge_target.numpy()
    influences = payload.edge_nonnegative_influence.float().numpy().astype(
        np.float64, copy=False
    )
    row_totals = np.bincount(
        targets, weights=influences, minlength=payload.node_count
    ).astype(np.float64, copy=False)
    valid = (influences > 0.0) & (row_totals[targets] > 0.0)
    edge_indices = np.flatnonzero(valid).astype(np.int32, copy=False)
    graph = build_sparse_dag(
        sources[valid],
        targets[valid],
        influences[valid],
        node_count=payload.node_count,
        epsilon=epsilon,
        edge_indices=edge_indices,
    )
    path_costs, path_lengths, path_nodes, path_edges = k_best_paths_to_sink(
        graph, sink=payload.root_index, keep=max_trails
    )

    candidates: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    for source in range(payload.token_start, payload.root_index):
        for rank in range(max_trails):
            length = int(path_lengths[source, rank])
            if length <= 0:
                continue
            nodes = tuple(int(value) for value in path_nodes[source, rank, :length])
            edges = tuple(
                int(value)
                for value in path_edges[source, rank, : max(length - 1, 0)]
            )
            candidates.append((float(path_costs[source, rank]), nodes, edges))
    candidates.sort(key=lambda item: (item[0], item[1]))
    selected: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    seen_paths: set[tuple[int, ...]] = set()
    for candidate in candidates:
        if candidate[1] in seen_paths:
            continue
        seen_paths.add(candidate[1])
        selected.append(candidate)
        if len(selected) == max_trails:
            break

    trails: list[AttributionTrail] = []
    used_nodes: set[int] = set()
    for total_cost, node_indices, edge_indices_for_path in selected:
        signed = tuple(
            float(payload.edge_signed_direct_effect[index])
            for index in edge_indices_for_path
        )
        influence = tuple(
            float(payload.edge_nonnegative_influence[index])
            for index in edge_indices_for_path
        )
        strengths = tuple(
            float(influence_value / row_totals[int(targets[index])])
            for influence_value, index in zip(
                influence, edge_indices_for_path, strict=True
            )
        )
        costs = tuple(
            -float(np.log(np.clip(value, epsilon, 1.0))) for value in strengths
        )
        used_nodes.update(node_indices)
        trails.append(
            AttributionTrail(
                node_indices=node_indices,
                signed_direct_effects=signed,
                nonnegative_influences=influence,
                normalized_strengths=strengths,
                edge_costs=costs,
                total_cost=float(total_cost),
                score=float(np.exp(-total_cost)),
            )
        )
    nodes = tuple(_node(payload, index) for index in sorted(used_nodes))
    return TrailResult(
        event_key=payload.event_key,
        artifact_id=payload.artifact_id,
        nodes=nodes,
        trails=tuple(trails),
        no_valid_trail=not trails,
    )

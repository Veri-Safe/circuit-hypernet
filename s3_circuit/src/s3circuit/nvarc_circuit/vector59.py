from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np

from .payload import CompactPayload
from .sparse_topology import (
    build_sparse_dag,
    exact_betweenness_and_path_mean,
    numba_available,
    shortest_distances_to_sink,
    weak_component_count,
)


VECTOR_SCHEMA = "nvarc_qwen3_36l_joint_signed_span_mean_59d_v1"
VECTOR_DIM = 59


@dataclass(frozen=True)
class VectorResult:
    event_key: str
    artifact_id: str
    values: np.ndarray
    schema: str = VECTOR_SCHEMA


def _mean(values: np.ndarray, default: float = 0.0) -> float:
    return float(values.mean()) if values.size else float(default)


def _max(values: np.ndarray, default: float = 0.0) -> float:
    return float(values.max()) if values.size else float(default)


def _std_population(values: np.ndarray, default: float = 0.0) -> float:
    return float(values.std(ddof=0)) if values.size else float(default)


def _std_sample(values: np.ndarray, default: float = 0.0) -> float:
    return float(values.std(ddof=1)) if values.size >= 2 else float(default)


def _mean_view_edge_weights(payload: CompactPayload) -> np.ndarray:
    values = (
        payload.edge_signed_direct_effect.float().numpy().astype(np.float64, copy=True)
    )
    targets = payload.edge_target.numpy()
    values[targets == payload.root_index] /= payload.target_count
    return values


def _causal_graph(
    payload: CompactPayload,
    weights: np.ndarray,
    *,
    epsilon: float,
) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(payload.node_count))
    sources = payload.edge_source.numpy().astype(np.int64, copy=False)
    targets = payload.edge_target.numpy().astype(np.int64, copy=False)
    row_totals: dict[int, float] = {}
    for target, weight in zip(targets, weights, strict=True):
        row_totals[int(target)] = row_totals.get(int(target), 0.0) + abs(float(weight))
    for source, target, weight in zip(sources, targets, weights, strict=True):
        magnitude = abs(float(weight))
        total = row_totals[int(target)]
        q = magnitude / total if total > 0.0 else 0.0
        cost = -float(np.log(np.clip(q, epsilon, 1.0)))
        graph.add_edge(
            int(source),
            int(target),
            signed_direct_effect=float(weight),
            magnitude=magnitude,
            q=q,
            cost=cost,
        )
    return graph


def _finite_directed_path_mean(graph: nx.DiGraph) -> float:
    values = []
    for source, lengths in nx.all_pairs_dijkstra_path_length(graph, weight="cost"):
        values.extend(
            float(length)
            for target, length in lengths.items()
            if target != source and np.isfinite(length)
        )
    return float(np.mean(values)) if values else -1.0


def _token_to_root_min(payload: CompactPayload, graph: nx.DiGraph) -> float:
    values = []
    for source in range(payload.token_start, payload.root_index):
        try:
            values.append(
                float(
                    nx.shortest_path_length(
                        graph, source=source, target=payload.root_index, weight="cost"
                    )
                )
            )
        except nx.NetworkXNoPath:
            continue
    return min(values) if values else -1.0


def vectorize_payload(
    payload: CompactPayload, *, epsilon: float = 1e-12
) -> VectorResult:
    payload.validate()
    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must lie strictly between zero and one")
    features: list[float] = [
        float(payload.total_active_features),
        float(payload.feature_count),
        float(payload.error_count),
        _mean(payload.target_probabilities.float().numpy()),
        _mean(payload.target_entropies.float().numpy()),
        _mean(payload.node_cumulative_scores().numpy()),
        float(payload.error_cumulative_score.float().sum().item()),
        _mean(payload.error_cumulative_score.float().numpy()),
    ]
    activations = (
        payload.feature_activation.float().numpy().astype(np.float64, copy=False)
    )
    features.extend(
        (
            _mean(activations),
            _max(activations),
            _std_sample(activations),
        )
    )
    layer_counts = np.bincount(
        payload.feature_layer.long().numpy(), minlength=36
    ).astype(np.float64)
    features.extend(float(item) for item in layer_counts)

    edge_weights = _mean_view_edge_weights(payload)
    features.extend(
        (
            float(edge_weights.sum()) if edge_weights.size else 0.0,
            _mean(edge_weights),
            _std_population(edge_weights),
        )
    )
    if numba_available():
        graph = build_sparse_dag(
            payload.edge_source.numpy(),
            payload.edge_target.numpy(),
            np.abs(edge_weights),
            node_count=payload.node_count,
            epsilon=epsilon,
        )
        features.append(float(graph.edge_count))
        density = (
            graph.edge_count / (graph.node_count * (graph.node_count - 1))
            if graph.node_count > 1
            else 0.0
        )
        features.append(float(density))
        features.append(float(weak_component_count(graph)))
        degree = (
            np.diff(graph.out_offsets)
            + np.bincount(graph.out_targets, minlength=graph.node_count)
        ).astype(np.float64, copy=False)
        degree /= max(graph.node_count - 1, 1)
        features.extend((_mean(degree), _max(degree)))
        betweenness, path_mean = exact_betweenness_and_path_mean(graph)
        features.extend((_mean(betweenness), _max(betweenness)))
        features.append(path_mean)
        distances = shortest_distances_to_sink(graph, payload.root_index)
        token_distances = distances[payload.token_start : payload.root_index]
        finite = token_distances[np.isfinite(token_distances)]
        features.append(float(finite.min()) if finite.size else -1.0)
    else:
        graph = _causal_graph(payload, edge_weights, epsilon=epsilon)
        features.append(float(graph.number_of_edges()))
        features.append(float(nx.density(graph)))
        features.append(float(nx.number_weakly_connected_components(graph)))
        degree = np.asarray(list(nx.degree_centrality(graph).values()), dtype=np.float64)
        features.extend((_mean(degree), _max(degree)))
        betweenness = np.asarray(
            list(nx.betweenness_centrality(graph, weight="cost", normalized=True).values()),
            dtype=np.float64,
        )
        features.extend((_mean(betweenness), _max(betweenness)))
        features.append(_finite_directed_path_mean(graph))
        features.append(_token_to_root_min(payload, graph))

    vector = np.asarray(features, dtype=np.float32)
    if vector.shape != (VECTOR_DIM,):
        raise AssertionError(f"59D vector construction returned shape {vector.shape}")
    if not np.isfinite(vector).all():
        raise ValueError("59D vector contains non-finite values")
    return VectorResult(payload.event_key, payload.artifact_id, vector)


def vectorize_many(
    payloads: Iterable[CompactPayload], *, epsilon: float = 1e-12
) -> tuple[VectorResult, ...]:
    return tuple(vectorize_payload(payload, epsilon=epsilon) for payload in payloads)

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit, prange, set_num_threads

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _NUMBA_AVAILABLE = False


def numba_available() -> bool:
    return _NUMBA_AVAILABLE


if _NUMBA_AVAILABLE:
    _thread_value = (
        os.environ.get("NUMBA_NUM_THREADS")
        or os.environ.get("S3_CPU_THREADS")
        or os.environ.get("FINALIZER_CPU_THREADS")
    )
    if _thread_value:
        try:
            set_num_threads(max(1, int(_thread_value)))
        except (ValueError, RuntimeError):
            pass


@dataclass(frozen=True)
class SparseDag:
    node_count: int
    edge_count: int
    out_offsets: np.ndarray
    out_targets: np.ndarray
    out_costs: np.ndarray
    out_edge_indices: np.ndarray
    in_offsets: np.ndarray
    in_sources: np.ndarray
    in_costs: np.ndarray
    topological_order: np.ndarray


if _NUMBA_AVAILABLE:

    @njit
    def _topological_order_kernel(
        node_count: int, out_offsets: np.ndarray, out_targets: np.ndarray
    ) -> tuple[np.ndarray, int]:
        indegree = np.zeros(node_count, dtype=np.int64)
        for target in out_targets:
            indegree[target] += 1
        queue = np.empty(node_count, dtype=np.int32)
        order = np.empty(node_count, dtype=np.int32)
        queue_size = 0
        for node in range(node_count):
            if indegree[node] == 0:
                queue[queue_size] = node
                queue_size += 1
        head = 0
        processed = 0
        while head < queue_size:
            node = queue[head]
            head += 1
            order[processed] = node
            processed += 1
            for edge in range(out_offsets[node], out_offsets[node + 1]):
                target = out_targets[edge]
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue[queue_size] = target
                    queue_size += 1
        return order, processed

    @njit
    def _shortest_to_sink_kernel(
        topological_order: np.ndarray,
        out_offsets: np.ndarray,
        out_targets: np.ndarray,
        out_costs: np.ndarray,
        sink: int,
    ) -> np.ndarray:
        distances = np.full(topological_order.size, np.inf, dtype=np.float64)
        distances[sink] = 0.0
        for index in range(topological_order.size - 1, -1, -1):
            source = topological_order[index]
            if source == sink:
                continue
            best = np.inf
            for edge in range(out_offsets[source], out_offsets[source + 1]):
                target = out_targets[edge]
                candidate = out_costs[edge] + distances[target]
                if candidate < best:
                    best = candidate
            distances[source] = best
        return distances

    @njit
    def _longest_path_length_kernel(
        topological_order: np.ndarray,
        out_offsets: np.ndarray,
        out_targets: np.ndarray,
        sink: int,
    ) -> int:
        lengths = np.zeros(topological_order.size, dtype=np.int32)
        lengths[sink] = 1
        longest = 1
        for index in range(topological_order.size - 1, -1, -1):
            source = topological_order[index]
            if source == sink:
                continue
            best = 0
            for edge in range(out_offsets[source], out_offsets[source + 1]):
                target_length = lengths[out_targets[edge]]
                if target_length > 0 and target_length + 1 > best:
                    best = target_length + 1
            lengths[source] = best
            if best > longest:
                longest = best
        return longest

    @njit
    def _weak_component_count_kernel(
        node_count: int,
        out_offsets: np.ndarray,
        out_targets: np.ndarray,
    ) -> int:
        parent = np.arange(node_count, dtype=np.int32)
        rank = np.zeros(node_count, dtype=np.int8)
        for source in range(node_count):
            for edge in range(out_offsets[source], out_offsets[source + 1]):
                target = out_targets[edge]
                left = source
                while parent[left] != left:
                    parent[left] = parent[parent[left]]
                    left = parent[left]
                right = target
                while parent[right] != right:
                    parent[right] = parent[parent[right]]
                    right = parent[right]
                if left == right:
                    continue
                if rank[left] < rank[right]:
                    parent[left] = right
                elif rank[left] > rank[right]:
                    parent[right] = left
                else:
                    parent[right] = left
                    rank[left] += 1
        count = 0
        for node in range(node_count):
            if parent[node] == node:
                count += 1
        return count

    @njit(parallel=True)
    def _brandes_batch_kernel(
        sources: np.ndarray,
        topological_order: np.ndarray,
        out_offsets: np.ndarray,
        out_targets: np.ndarray,
        out_costs: np.ndarray,
        in_offsets: np.ndarray,
        in_sources: np.ndarray,
        in_costs: np.ndarray,
        node_count: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dependencies_by_source = np.zeros(
            (sources.size, node_count), dtype=np.float64
        )
        path_sums = np.zeros(sources.size, dtype=np.float64)
        reachable_counts = np.zeros(sources.size, dtype=np.int64)
        for source_index in prange(sources.size):
            source = sources[source_index]
            distances = np.full(node_count, np.inf, dtype=np.float64)
            sigma = np.zeros(node_count, dtype=np.float64)
            dependencies = np.zeros(node_count, dtype=np.float64)
            distances[source] = 0.0
            sigma[source] = 1.0

            for order_index in range(topological_order.size):
                node = topological_order[order_index]
                distance = distances[node]
                if not np.isfinite(distance):
                    continue
                for edge in range(out_offsets[node], out_offsets[node + 1]):
                    target = out_targets[edge]
                    candidate = distance + out_costs[edge]
                    if candidate < distances[target]:
                        distances[target] = candidate
                        sigma[target] = sigma[node]
                    elif candidate == distances[target]:
                        sigma[target] += sigma[node]

            path_sum = 0.0
            reachable = 0
            for target in range(node_count):
                if target != source and np.isfinite(distances[target]):
                    path_sum += distances[target]
                    reachable += 1
            path_sums[source_index] = path_sum
            reachable_counts[source_index] = reachable

            for order_index in range(topological_order.size - 1, -1, -1):
                node = topological_order[order_index]
                if node == source or not np.isfinite(distances[node]):
                    continue
                if sigma[node] == 0.0:
                    continue
                for edge in range(in_offsets[node], in_offsets[node + 1]):
                    predecessor = in_sources[edge]
                    if not np.isfinite(distances[predecessor]):
                        continue
                    if (
                        distances[predecessor] + in_costs[edge]
                        == distances[node]
                    ):
                        dependencies[predecessor] += (
                            sigma[predecessor]
                            / sigma[node]
                            * (1.0 + dependencies[node])
                        )
                if node != source:
                    dependencies_by_source[source_index, node] = dependencies[node]
        return dependencies_by_source, path_sums, reachable_counts

    @njit
    def _k_best_paths_kernel(
        topological_order: np.ndarray,
        out_offsets: np.ndarray,
        out_targets: np.ndarray,
        out_costs: np.ndarray,
        out_edge_indices: np.ndarray,
        root: int,
        keep: int,
        depth: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        path_costs = np.full(
            (topological_order.size, keep), np.inf, dtype=np.float64
        )
        path_lengths = np.zeros(
            (topological_order.size, keep), dtype=np.int32
        )
        path_nodes = np.full(
            (topological_order.size, keep, depth), -1, dtype=np.int32
        )
        path_edges = np.full(
            (topological_order.size, keep, max(depth - 1, 1)),
            -1,
            dtype=np.int32,
        )
        path_counts = np.zeros(topological_order.size, dtype=np.int32)
        path_counts[root] = 1
        path_costs[root, 0] = 0.0
        path_lengths[root, 0] = 1
        path_nodes[root, 0, 0] = root

        for order_index in range(topological_order.size - 1, -1, -1):
            source = topological_order[order_index]
            if source == root:
                continue
            count = 0
            for edge in range(out_offsets[source], out_offsets[source + 1]):
                target = out_targets[edge]
                target_count = path_counts[target]
                for target_rank in range(target_count):
                    tail_length = path_lengths[target, target_rank]
                    candidate_length = tail_length + 1
                    candidate_cost = (
                        out_costs[edge] + path_costs[target, target_rank]
                    )
                    insertion = count
                    for rank in range(count):
                        existing_length = path_lengths[source, rank]
                        candidate_is_better = False
                        if candidate_cost < path_costs[source, rank]:
                            candidate_is_better = True
                        elif candidate_cost == path_costs[source, rank]:
                            common = candidate_length
                            if existing_length < common:
                                common = existing_length
                            for position in range(common):
                                candidate_node = source
                                if position > 0:
                                    candidate_node = path_nodes[
                                        target, target_rank, position - 1
                                    ]
                                existing_node = path_nodes[source, rank, position]
                                if candidate_node < existing_node:
                                    candidate_is_better = True
                                    break
                                if candidate_node > existing_node:
                                    break
                            if (
                                not candidate_is_better
                                and candidate_length < existing_length
                            ):
                                candidate_is_better = True
                        if candidate_is_better:
                            insertion = rank
                            break
                    if insertion == keep:
                        continue
                    if count < keep:
                        count += 1
                    for rank in range(count - 1, insertion, -1):
                        path_costs[source, rank] = path_costs[source, rank - 1]
                        path_lengths[source, rank] = path_lengths[source, rank - 1]
                        for position in range(depth):
                            path_nodes[source, rank, position] = path_nodes[
                                source, rank - 1, position
                            ]
                        for position in range(max(depth - 1, 1)):
                            path_edges[source, rank, position] = path_edges[
                                source, rank - 1, position
                            ]
                    path_costs[source, insertion] = candidate_cost
                    path_lengths[source, insertion] = candidate_length
                    path_nodes[source, insertion, 0] = source
                    path_edges[source, insertion, 0] = out_edge_indices[edge]
                    for position in range(1, depth):
                        if position < candidate_length:
                            path_nodes[source, insertion, position] = path_nodes[
                                target, target_rank, position - 1
                            ]
                        else:
                            path_nodes[source, insertion, position] = -1
                    for position in range(1, max(depth - 1, 1)):
                        if position < candidate_length - 1:
                            path_edges[source, insertion, position] = path_edges[
                                target, target_rank, position - 1
                            ]
                        else:
                            path_edges[source, insertion, position] = -1
            path_counts[source] = count
        return path_costs, path_lengths, path_nodes, path_edges


def _offsets_for(values: np.ndarray, node_count: int) -> np.ndarray:
    counts = np.bincount(values, minlength=node_count)
    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    return offsets


def _python_topological_order(
    node_count: int, out_offsets: np.ndarray, out_targets: np.ndarray
) -> np.ndarray:
    indegree = np.bincount(out_targets, minlength=node_count).astype(
        np.int64, copy=False
    )
    queue = np.flatnonzero(indegree == 0).tolist()
    order: list[int] = []
    head = 0
    while head < len(queue):
        source = queue[head]
        head += 1
        order.append(source)
        for target in out_targets[out_offsets[source] : out_offsets[source + 1]]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(int(target))
    if len(order) != node_count:
        raise ValueError("Sparse topology must be a DAG")
    return np.asarray(order, dtype=np.int32)


def build_sparse_dag(
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    edge_weights: np.ndarray,
    *,
    node_count: int,
    epsilon: float,
    edge_indices: np.ndarray | None = None,
) -> SparseDag:
    source = np.asarray(edge_source, dtype=np.int32)
    target = np.asarray(edge_target, dtype=np.int32)
    weights = np.asarray(edge_weights, dtype=np.float64)
    if source.shape != target.shape or source.shape != weights.shape:
        raise ValueError("Sparse edge arrays are misaligned")
    if edge_indices is None:
        original_edges = np.arange(source.size, dtype=np.int32)
    else:
        original_edges = np.asarray(edge_indices, dtype=np.int32)
        if original_edges.shape != source.shape:
            raise ValueError("Sparse edge indices are misaligned")

    row_totals = np.bincount(
        target, weights=weights, minlength=node_count
    ).astype(np.float64, copy=False)
    normalized = np.divide(
        weights,
        row_totals[target],
        out=np.zeros_like(weights),
        where=row_totals[target] > 0.0,
    )
    costs = -np.log(np.clip(normalized, epsilon, 1.0))

    out_order = np.argsort(source, kind="stable")
    out_source = source[out_order]
    out_offsets = _offsets_for(out_source, node_count)
    out_targets = target[out_order]
    out_costs = costs[out_order]
    out_edge_indices = original_edges[out_order]

    in_order = np.argsort(target, kind="stable")
    in_target = target[in_order]
    in_offsets = _offsets_for(in_target, node_count)
    in_sources = source[in_order]
    in_costs = costs[in_order]

    if _NUMBA_AVAILABLE:
        topological, processed = _topological_order_kernel(
            node_count, out_offsets, out_targets
        )
        if processed != node_count:
            raise ValueError("Sparse topology must be a DAG")
        topological = topological[:processed]
    else:
        topological = _python_topological_order(
            node_count, out_offsets, out_targets
        )
    return SparseDag(
        node_count=node_count,
        edge_count=int(source.size),
        out_offsets=out_offsets,
        out_targets=out_targets,
        out_costs=out_costs,
        out_edge_indices=out_edge_indices,
        in_offsets=in_offsets,
        in_sources=in_sources,
        in_costs=in_costs,
        topological_order=topological,
    )


def exact_betweenness_and_path_mean(
    graph: SparseDag, *, batch_size: int = 64
) -> tuple[np.ndarray, float]:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for sparse exact topology metrics")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sources = np.arange(max(graph.node_count - 1, 0), dtype=np.int32)
    centrality_sum = np.zeros(graph.node_count, dtype=np.float64)
    path_sum = 0.0
    reachable_count = 0
    for start in range(0, sources.size, batch_size):
        batch = sources[start : start + batch_size]
        dependencies, sums, counts = _brandes_batch_kernel(
            batch,
            graph.topological_order,
            graph.out_offsets,
            graph.out_targets,
            graph.out_costs,
            graph.in_offsets,
            graph.in_sources,
            graph.in_costs,
            graph.node_count,
        )
        centrality_sum += dependencies.sum(axis=0)
        path_sum += float(sums.sum())
        reachable_count += int(counts.sum())
    if graph.node_count > 2:
        centrality_sum /= (graph.node_count - 1) * (graph.node_count - 2)
    else:
        centrality_sum.fill(0.0)
    mean_path = path_sum / reachable_count if reachable_count else -1.0
    return centrality_sum, float(mean_path)


def shortest_distances_to_sink(graph: SparseDag, sink: int) -> np.ndarray:
    if _NUMBA_AVAILABLE:
        return _shortest_to_sink_kernel(
            graph.topological_order,
            graph.out_offsets,
            graph.out_targets,
            graph.out_costs,
            sink,
        )
    distances = np.full(graph.node_count, np.inf, dtype=np.float64)
    distances[sink] = 0.0
    for source in graph.topological_order[::-1]:
        if int(source) == sink:
            continue
        outgoing = slice(graph.out_offsets[source], graph.out_offsets[source + 1])
        if graph.out_targets[outgoing].size:
            distances[source] = np.min(
                graph.out_costs[outgoing] + distances[graph.out_targets[outgoing]]
            )
    return distances


def weak_component_count(graph: SparseDag) -> int:
    if _NUMBA_AVAILABLE:
        return int(
            _weak_component_count_kernel(
                graph.node_count, graph.out_offsets, graph.out_targets
            )
        )
    parent = np.arange(graph.node_count, dtype=np.int32)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for source in range(graph.node_count):
        for target in graph.out_targets[
            graph.out_offsets[source] : graph.out_offsets[source + 1]
        ]:
            left, right = find(source), find(int(target))
            if left != right:
                parent[right] = left
    return int(sum(find(node) == node for node in range(graph.node_count)))


def k_best_paths_to_sink(
    graph: SparseDag, *, sink: int, keep: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not _NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for sparse exact trail extraction")
    if keep <= 0:
        raise ValueError("keep must be positive")
    depth = (
        _longest_path_length_kernel(
            graph.topological_order,
            graph.out_offsets,
            graph.out_targets,
            sink,
        )
        if graph.node_count
        else 1
    )
    return _k_best_paths_kernel(
        graph.topological_order,
        graph.out_offsets,
        graph.out_targets,
        graph.out_costs,
        graph.out_edge_indices,
        sink,
        keep,
        max(1, int(depth)),
    )

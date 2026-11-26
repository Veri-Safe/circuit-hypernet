"""CodeCircuit-style statistics for an already retained S5 graph.

This module is deliberately an adapter, not a second circuit extractor.  It
reads only :class:`S5GraphPayloadV2` and produces the 25 statistics that are
recoverable from the old CodeCircuit-style representation after accounting for
the S5 retained-graph boundary.  The signed-DAG path implementation is kept
here instead of delegating to Dijkstra because S5 payload edges may be
negative.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .payload_v2 import S5GraphPayloadV2, semantic_sha256


LEGACY25_FORMAT = "codecircuit25_s5_retained_signed_v1"
LEGACY25_VERSION = 1
LEGACY25_DIM = 25
S5_LAYER_COUNT = 8

LEGACY25_COORDINATES: tuple[str, ...] = (
    "feature_count",
    "error_count",
    "feature_activation_mean",
    "feature_activation_max",
    "feature_activation_std_sample",
    "feature_layer_0_count",
    "feature_layer_1_count",
    "feature_layer_2_count",
    "feature_layer_3_count",
    "feature_layer_4_count",
    "feature_layer_5_count",
    "feature_layer_6_count",
    "feature_layer_7_count",
    "signed_edge_weight_sum",
    "signed_edge_weight_mean",
    "signed_edge_weight_std_population",
    "edge_count",
    "directed_density",
    "weak_component_count",
    "degree_centrality_mean",
    "degree_centrality_max",
    "weighted_betweenness_mean",
    "weighted_betweenness_max",
    "largest_weak_component_avg_shortest_path",
    "token_to_joint_root_min_shortest_path",
)

# These are the coordinates that can be independently projected from raw397.
# The remaining coordinates need the actual retained graph.
RAW397_RECOVERABLE_LEGACY25_INDICES: tuple[int, ...] = (
    0,
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
)


@dataclass(frozen=True)
class Legacy25Schema:
    """Versioned names and semantics for the 25D S5 adaptation."""

    format: str = LEGACY25_FORMAT
    version: int = LEGACY25_VERSION

    def identity(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "dtype": "float32",
            "dimension": LEGACY25_DIM,
            "model_layers": S5_LAYER_COUNT,
            "coordinates": list(LEGACY25_COORDINATES),
            "graph_boundary": "validated_S5GraphPayloadV2_retained_graph",
            "feature_identity": "(layer,position,feature_id) remains payload-only",
            "edge_orientation": "edge_source_to_edge_target",
            "edge_weight": "edge_signed_direct_effect",
            "nonnegative_edge_channel": "not_used_for_signed_statistics",
            "feature_activation_std": "sample_ddof1_if_count_ge_2_else_zero",
            "edge_weight_std": "population_ddof0",
            "centrality": {
                "degree": "(in_degree+out_degree)/(N-1)",
                "betweenness": "directed_weighted_shortest_paths_endpoints_excluded",
                "normalization": "1/((N-1)*(N-2))_when_N_gt_2",
            },
            "shortest_path": {
                "weight": "signed_edge_weight_sum",
                "algorithm": "exact_DAG_dynamic_programming",
                "tie_rule": "exact_FP64_equality",
                "largest_component": "all_ordered_pairs_directed; -1_if_singleton_or_not_strongly_connected",
                "token_root": "all_token_occurrences_to_joint_root; -1_if_unreachable",
            },
            "empty_group": "zero",
            "validity_flags": "stored_as_diagnostics_not_numeric_coordinates",
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(self.identity())


SCHEMA_LEGACY25 = Legacy25Schema()
LEGACY25_SCHEMA_SHA256 = (
    "223f14e34beef55e71d2a22ced70c070d33802de23dea6a6447bb68fbaca556d"
)
if SCHEMA_LEGACY25.sha256 != LEGACY25_SCHEMA_SHA256:
    raise RuntimeError("frozen legacy25 descriptor schema identity changed")


@dataclass(frozen=True)
class Legacy25Diagnostics:
    """Non-condition metadata explaining path sentinels and computation."""

    largest_component_size: int
    largest_component_status: str
    token_root_path_status: str
    token_root_reachable_count: int
    topological_node_count: int
    betweenness_algorithm: str = "signed_dag_brandes_exact_v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "largest_component_size": int(self.largest_component_size),
            "largest_component_status": self.largest_component_status,
            "token_root_path_status": self.token_root_path_status,
            "token_root_reachable_count": int(self.token_root_reachable_count),
            "topological_node_count": int(self.topological_node_count),
            "betweenness_algorithm": self.betweenness_algorithm,
        }


@dataclass(frozen=True)
class Legacy25Computation:
    """A vector plus diagnostics kept outside the condition tensor."""

    event_key: str
    payload_artifact_id: str
    schema_sha256: str
    vector: Tensor
    diagnostics: Legacy25Diagnostics

    def validate(self) -> None:
        if (
            len(self.event_key) != 64
            or len(self.payload_artifact_id) != 64
            or any(char not in "0123456789abcdef" for char in self.event_key)
            or any(char not in "0123456789abcdef" for char in self.payload_artifact_id)
        ):
            raise ValueError("legacy25 source identities must be SHA256 strings")
        if self.schema_sha256 != SCHEMA_LEGACY25.sha256:
            raise ValueError("legacy25 schema identity changed")
        if (
            self.vector.device.type != "cpu"
            or self.vector.dtype != torch.float32
            or self.vector.shape != (LEGACY25_DIM,)
            or not bool(torch.isfinite(self.vector).all())
        ):
            raise ValueError("legacy25 vector must be finite CPU FP32[25]")


@dataclass(frozen=True)
class _GraphView:
    node_count: int
    root_index: int
    source: np.ndarray
    target: np.ndarray
    weight: np.ndarray
    adjacency: tuple[tuple[tuple[int, float], ...], ...]
    reverse: tuple[tuple[int, ...], ...]
    topological: np.ndarray
    components: tuple[tuple[int, ...], ...]


def _sample_std(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1, dtype=np.float64))


def _population_std(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.std(values, ddof=0, dtype=np.float64))


def _topological_order(node_count: int, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    indegree = np.zeros(node_count, dtype=np.int64)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for left, right in zip(source.tolist(), target.tolist(), strict=True):
        left_i = int(left)
        right_i = int(right)
        indegree[right_i] += 1
        adjacency[left_i].append(right_i)
    for row in adjacency:
        row.sort()
    ready = [index for index, value in enumerate(indegree.tolist()) if value == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        node = heapq.heappop(ready)
        order.append(int(node))
        for neighbor in adjacency[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(ready, neighbor)
    if len(order) != node_count:
        raise ValueError("legacy25 graph is not a DAG")
    return np.asarray(order, dtype=np.int64)


def _weak_components(node_count: int, source: np.ndarray, target: np.ndarray) -> tuple[tuple[int, ...], ...]:
    parent = list(range(node_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for left, right in zip(source.tolist(), target.tolist(), strict=True):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[int, list[int]] = {}
    for node in range(node_count):
        groups.setdefault(find(node), []).append(node)
    return tuple(
        tuple(sorted(nodes))
        for nodes in sorted(groups.values(), key=lambda values: (len(values), values[0]))
    )


def _build_graph_view(payload: S5GraphPayloadV2) -> _GraphView:
    node_count = payload.node_count
    source = payload.edge_source.detach().cpu().numpy().astype(np.int64, copy=False)
    target = payload.edge_target.detach().cpu().numpy().astype(np.int64, copy=False)
    weight = payload.edge_signed_direct_effect.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(node_count)]
    reverse: list[list[int]] = [[] for _ in range(node_count)]
    for left, right, value in sorted(
        zip(source.tolist(), target.tolist(), weight.tolist()),
        key=lambda item: (int(item[0]), int(item[1])),
    ):
        adjacency[int(left)].append((int(right), float(value)))
        reverse[int(right)].append(int(left))
    for row in reverse:
        row.sort()
    topological = _topological_order(node_count, source, target)
    return _GraphView(
        node_count=node_count,
        root_index=payload.root_index,
        source=source,
        target=target,
        weight=weight,
        adjacency=tuple(tuple(row) for row in adjacency),
        reverse=tuple(tuple(row) for row in reverse),
        topological=topological,
        components=_weak_components(node_count, source, target),
    )


def _reachable(start: int, adjacency: Sequence[Sequence[tuple[int, float]]], allowed: set[int]) -> set[int]:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor, _weight in adjacency[node]:
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _reverse_reachable(start: int, reverse: Sequence[Sequence[int]], allowed: set[int]) -> set[int]:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in reverse[node]:
            if neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _is_strongly_connected(view: _GraphView, component: Sequence[int]) -> bool:
    if not component:
        return False
    allowed = set(int(node) for node in component)
    start = int(component[0])
    return (
        len(_reachable(start, view.adjacency, allowed)) == len(allowed)
        and len(_reverse_reachable(start, view.reverse, allowed)) == len(allowed)
    )


def _shortest_paths_from_source(
    view: _GraphView, source: int
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, ...], ...]]:
    """Exact shortest paths and predecessor sets on a signed DAG."""

    distance = np.full(view.node_count, np.inf, dtype=np.float64)
    sigma = np.zeros(view.node_count, dtype=np.float64)
    predecessor: list[list[int]] = [[] for _ in range(view.node_count)]
    distance[int(source)] = 0.0
    sigma[int(source)] = 1.0
    for node_value in view.topological.tolist():
        node = int(node_value)
        if not np.isfinite(distance[node]):
            continue
        base = float(distance[node])
        for neighbor, edge_weight in view.adjacency[node]:
            candidate = base + float(edge_weight)
            if candidate < distance[neighbor]:
                distance[neighbor] = candidate
                sigma[neighbor] = sigma[node]
                predecessor[neighbor] = [node]
            elif candidate == distance[neighbor]:
                sigma[neighbor] += sigma[node]
                predecessor[neighbor].append(node)
    if not bool(np.isfinite(distance[sigma > 0]).all()):
        raise ValueError("legacy25 shortest-path calculation became non-finite")
    return distance, sigma, tuple(tuple(row) for row in predecessor)


def _weighted_betweenness(view: _GraphView) -> np.ndarray:
    scores = np.zeros(view.node_count, dtype=np.float64)
    if view.node_count <= 2:
        return scores
    for source in range(view.node_count):
        _distance, sigma, predecessor = _shortest_paths_from_source(view, source)
        dependency = np.zeros(view.node_count, dtype=np.float64)
        for node_value in reversed(view.topological.tolist()):
            node = int(node_value)
            if sigma[node] == 0.0:
                continue
            for parent in predecessor[node]:
                dependency[parent] += (
                    sigma[parent] / sigma[node]
                ) * (1.0 + dependency[node])
            if node != source:
                scores[node] += dependency[node]
    scores /= float((view.node_count - 1) * (view.node_count - 2))
    if not bool(np.isfinite(scores).all()):
        raise ValueError("legacy25 betweenness calculation became non-finite")
    return scores


def _largest_component_average_path(view: _GraphView) -> tuple[float, int, str]:
    if not view.components:
        return -1.0, 0, "empty"
    component = min(
        view.components,
        key=lambda values: (-len(values), values[0]),
    )
    size = len(component)
    if size < 2:
        return -1.0, size, "singleton"
    if not _is_strongly_connected(view, component):
        return -1.0, size, "not_strongly_connected"

    # A validated S5 payload is a DAG, so a non-singleton strongly connected
    # component cannot occur. Keep this exact fallback for schema-level tests
    # and future graph adapters rather than silently changing the sentinel.
    component_set = set(component)
    distances: list[float] = []
    for source in component:
        distance, _sigma, _predecessor = _shortest_paths_from_source(view, int(source))
        for target in component:
            if target != source:
                if not np.isfinite(distance[target]):
                    return -1.0, size, "not_strongly_connected"
                distances.append(float(distance[target]))
    if not distances or not component_set:
        return -1.0, size, "singleton"
    return float(np.mean(np.asarray(distances, dtype=np.float64))), size, "ok"


def _token_root_shortest(view: _GraphView, token_start: int, root_index: int) -> tuple[float, int, str]:
    distance = np.full(view.node_count, np.inf, dtype=np.float64)
    distance[int(root_index)] = 0.0
    for node_value in reversed(view.topological.tolist()):
        node = int(node_value)
        if node == int(root_index):
            continue
        candidates = [
            float(edge_weight) + float(distance[neighbor])
            for neighbor, edge_weight in view.adjacency[node]
            if np.isfinite(distance[neighbor])
        ]
        if candidates:
            distance[node] = min(candidates)
    token_indices = range(int(token_start), int(root_index))
    finite = [float(distance[index]) for index in token_indices if np.isfinite(distance[index])]
    if not finite:
        return -1.0, 0, "no_path"
    return min(finite), len(finite), "ok"


def _legacy25_values(
    payload: S5GraphPayloadV2,
    view: _GraphView,
) -> tuple[np.ndarray, Legacy25Diagnostics]:
    feature_activation = payload.feature_activation.detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    feature_layer = payload.feature_layer.detach().cpu().numpy().astype(np.int64, copy=False)
    signed = view.weight
    feature_count = payload.feature_count
    error_count = payload.error_count
    edge_count = payload.edge_count
    node_count = payload.node_count

    activation_mean = float(feature_activation.mean()) if feature_count else 0.0
    activation_max = float(feature_activation.max()) if feature_count else 0.0
    activation_std = _sample_std(feature_activation)
    layer_counts = np.bincount(feature_layer, minlength=S5_LAYER_COUNT).astype(np.float64)

    edge_sum = float(signed.sum()) if edge_count else 0.0
    edge_mean = edge_sum / edge_count if edge_count else 0.0
    edge_std = _population_std(signed)
    density = (
        float(edge_count) / float(node_count * (node_count - 1))
        if node_count >= 2
        else 0.0
    )

    indegree = np.bincount(view.target, minlength=node_count).astype(np.float64)
    outdegree = np.bincount(view.source, minlength=node_count).astype(np.float64)
    if node_count >= 2:
        degree = (indegree + outdegree) / float(node_count - 1)
        degree_mean = float(degree.mean())
        degree_max = float(degree.max())
    else:
        degree_mean = 0.0
        degree_max = 0.0

    betweenness = _weighted_betweenness(view)
    betweenness_mean = float(betweenness.mean()) if betweenness.size else 0.0
    betweenness_max = float(betweenness.max()) if betweenness.size else 0.0
    average_path, component_size, component_status = _largest_component_average_path(view)
    token_path, reachable_count, token_status = _token_root_shortest(
        view,
        payload.token_start,
        payload.root_index,
    )

    values = np.asarray(
        [
            float(feature_count),
            float(error_count),
            activation_mean,
            activation_max,
            activation_std,
            *layer_counts.tolist(),
            edge_sum,
            edge_mean,
            edge_std,
            float(edge_count),
            density,
            float(len(view.components)),
            degree_mean,
            degree_max,
            betweenness_mean,
            betweenness_max,
            average_path,
            token_path,
        ],
        dtype=np.float64,
    )
    if values.shape != (LEGACY25_DIM,) or not bool(np.isfinite(values).all()):
        raise ValueError("legacy25 statistics are not finite [25]")
    diagnostics = Legacy25Diagnostics(
        largest_component_size=component_size,
        largest_component_status=component_status,
        token_root_path_status=token_status,
        token_root_reachable_count=reachable_count,
        topological_node_count=view.node_count,
    )
    return values, diagnostics


def compute_legacy25(
    payload: S5GraphPayloadV2,
    *,
    n_layers: int = S5_LAYER_COUNT,
    d_feature: int,
    archive_identity_sha256: str | None = None,
) -> Legacy25Computation:
    """Compute one validated S5 CodeCircuit-style 25D result."""

    if int(n_layers) != S5_LAYER_COUNT:
        raise ValueError("legacy25 is frozen for the eight-layer S5 model")
    payload.validate(
        n_layers=int(n_layers),
        d_feature=int(d_feature),
        archive_identity_sha256=archive_identity_sha256,
    )
    view = _build_graph_view(payload)
    values, diagnostics = _legacy25_values(payload, view)
    vector = torch.from_numpy(np.asarray(values, dtype=np.float32)).contiguous()
    result = Legacy25Computation(
        event_key=payload.event_key,
        payload_artifact_id=payload.payload_artifact_id,
        schema_sha256=SCHEMA_LEGACY25.sha256,
        vector=vector,
        diagnostics=diagnostics,
    )
    result.validate()
    return result


def vectorize_payload_v2_legacy25(
    payload: S5GraphPayloadV2,
    *,
    n_layers: int = S5_LAYER_COUNT,
    d_feature: int,
    archive_identity_sha256: str | None = None,
) -> Tensor:
    """Return one finite CPU FP32[25] CodeCircuit-style adaptation."""

    return compute_legacy25(
        payload,
        n_layers=n_layers,
        d_feature=d_feature,
        archive_identity_sha256=archive_identity_sha256,
    ).vector


def vectorize_many_payload_v2_legacy25(
    payloads: Sequence[S5GraphPayloadV2],
    *,
    n_layers: int = S5_LAYER_COUNT,
    d_feature: int,
    archive_identity_sha256: str | None = None,
) -> Tensor:
    """Vectorize a bounded in-memory fixture batch."""

    if not payloads:
        return torch.empty((0, LEGACY25_DIM), dtype=torch.float32)
    return torch.stack(
        [
            vectorize_payload_v2_legacy25(
                payload,
                n_layers=n_layers,
                d_feature=d_feature,
                archive_identity_sha256=archive_identity_sha256,
            )
            for payload in payloads
        ],
    ).contiguous()


LEGACY25_LOG1P_INDICES: tuple[int, ...] = tuple(range(13)) + (15, 16, 18)
LEGACY25_SIGNED_LOG1P_INDICES: tuple[int, ...] = (13, 14, 23, 24)


def fixed_log1p_legacy25(
    vector: Tensor | np.ndarray,
    *,
    diagnostics: Legacy25Diagnostics | None = None,
) -> Tensor:
    """Apply the fixed transform declared for the 25D retained-graph view.

    Count/non-negative coordinates use ``log1p``.  Signed edge/path
    coordinates use a sign-preserving ``log1p``.  The two path coordinates
    reserve ``-1`` for an invalid/non-existent path and therefore keep that
    sentinel unchanged.
    """

    value = torch.as_tensor(vector, dtype=torch.float32).detach().cpu().contiguous()
    if value.shape != (LEGACY25_DIM,) or not bool(torch.isfinite(value).all()):
        raise ValueError("fixed_log1p_legacy25 expects finite FP32[25]")
    result = value.clone()
    for index in LEGACY25_LOG1P_INDICES:
        if float(result[index]) < 0.0:
            raise ValueError(f"legacy25 coordinate {index} cannot receive log1p")
        result[index] = torch.log1p(result[index])
    for index in LEGACY25_SIGNED_LOG1P_INDICES:
        raw = result[index]
        if index == 23 and diagnostics is not None and diagnostics.largest_component_status != "ok":
            continue
        if index == 23 and diagnostics is None and float(raw) == -1.0:
            continue
        if index == 24 and diagnostics is not None and diagnostics.token_root_path_status != "ok":
            continue
        if index == 24 and diagnostics is None and float(raw) == -1.0:
            continue
        result[index] = torch.sign(raw) * torch.log1p(torch.abs(raw))
    if not bool(torch.isfinite(result).all()):
        raise ValueError("fixed_log1p_legacy25 produced non-finite values")
    return result.contiguous()


def recover_legacy25_from_raw397(raw397: Tensor | np.ndarray) -> tuple[Tensor, Tensor]:
    """Recover the 18 algebraic legacy coordinates from raw397.

    Returns ``(values, mask)``.  Unrecoverable coordinates are NaN and false in
    the mask; callers must not treat this as a complete CodeCircuit25 vector.
    The helper exists for an independent cross-check of payload vectorization.
    """

    value = np.asarray(
        raw397.detach().cpu().numpy() if isinstance(raw397, Tensor) else raw397,
        dtype=np.float64,
    )
    if value.shape != (397,) or not bool(np.isfinite(value).all()):
        raise ValueError("raw397 must be finite shape [397]")
    result = np.full(LEGACY25_DIM, np.nan, dtype=np.float64)
    mask = np.zeros(LEGACY25_DIM, dtype=np.bool_)
    feature_count = float(value[0])
    error_count = float(value[9])
    token_count = float(value[14])
    edge_count = float(value[19])
    node_count = feature_count + error_count + token_count + 1.0
    result[0] = feature_count
    result[1] = error_count
    result[2] = value[1] / feature_count if feature_count else 0.0
    result[3] = value[4]
    result[5:13] = [
        value[30 + layer * 14] * feature_count for layer in range(S5_LAYER_COUNT)
    ]
    layer_counts = np.rint(result[5:13])
    if bool(np.any(np.abs(result[5:13] - layer_counts) > 1.0e-3)):
        raise ValueError("raw397 layer fractions do not recover integral counts")
    result[5:13] = layer_counts
    result[13] = value[24] - value[25]
    result[14] = result[13] / edge_count if edge_count else 0.0
    result[16] = edge_count
    result[17] = (
        edge_count / (node_count * (node_count - 1.0))
        if node_count >= 2.0
        else 0.0
    )
    result[18] = value[304 + 37]
    result[19] = 2.0 * edge_count / (node_count * (node_count - 1.0)) if node_count >= 2.0 else 0.0
    for index in RAW397_RECOVERABLE_LEGACY25_INDICES:
        mask[index] = True
    if not bool(np.isfinite(result[mask]).all()):
        raise ValueError("raw397 recoverable projection is non-finite")
    return (
        torch.from_numpy(result.astype(np.float32, copy=False)),
        torch.from_numpy(mask),
    )

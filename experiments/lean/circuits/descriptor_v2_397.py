"""Deterministic condition-safe 397D statistics for S5 Graph Payload v2."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .payload_v2 import S5GraphPayloadV2


DESCRIPTOR_397_FORMAT = "s5_circuit_statistical_descriptor_397_v1"
DESCRIPTOR_397_SCHEMA_VERSION = 1
BLOCK_ORDER = ("global", "layer_nodes", "layer_flow", "topology", "position")
BLOCK_DIMS = {
    "global": 30,
    "layer_nodes": 112,
    "layer_flow": 162,
    "topology": 51,
    "position": 42,
}
DESCRIPTOR_DIM = sum(BLOCK_DIMS.values())
NODE_KINDS = ("token", "feature", "error", "root")
FLOW_CATEGORIES = (
    "token",
    "root",
    *(f"feature_l{layer}" for layer in range(8)),
    *(f"error_l{layer}" for layer in range(8)),
)


def _semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _flow_allowed_pairs() -> tuple[tuple[int, int], ...]:
    pairs = {(0, 1), *((0, target) for target in range(2, 10))}
    for layer in range(8):
        feature_source = 2 + layer
        error_source = 10 + layer
        pairs.add((feature_source, 1))
        pairs.add((error_source, 1))
        for target_layer in range(layer + 1, 8):
            target = 2 + target_layer
            pairs.add((feature_source, target))
            pairs.add((error_source, target))
    result = tuple(sorted(pairs))
    if len(result) != 81:
        raise AssertionError("397D Layer Flow must contain exactly 81 routes")
    return result


def _kind_edge_allowed(source: int, target: int) -> bool:
    return source in (0, 1, 2) and target in (1, 3)


FLOW_ALLOWED_PAIRS = _flow_allowed_pairs()
MOTIF2_SEQUENCES = tuple(
    (first, middle, last)
    for first in range(4)
    for middle in range(4)
    for last in range(4)
    if _kind_edge_allowed(first, middle) and _kind_edge_allowed(middle, last)
)
MOTIF3_SEQUENCES = tuple(
    (first, second, third, last)
    for first in range(4)
    for second in range(4)
    for third in range(4)
    for last in range(4)
    if _kind_edge_allowed(first, second)
    and _kind_edge_allowed(second, third)
    and _kind_edge_allowed(third, last)
)
if len(MOTIF2_SEQUENCES) != 6 or len(MOTIF3_SEQUENCES) != 6:
    raise AssertionError("397D topology must contain six motifs at each path length")


GLOBAL_COORDINATES = (
    "feature.count",
    "feature.activation.sum",
    "feature.activation.median",
    "feature.activation.p90",
    "feature.activation.max",
    "feature.influence.sum",
    "feature.influence.median",
    "feature.influence.p90",
    "feature.influence.max",
    "error.count",
    "error.influence.sum",
    "error.influence.median",
    "error.influence.p90",
    "error.influence.max",
    "token.count",
    "token.influence.sum",
    "token.influence.median",
    "token.influence.p90",
    "token.influence.max",
    "edge.count",
    "edge.influence.sum",
    "edge.influence.median",
    "edge.influence.p90",
    "edge.influence.max",
    "edge.signed.positive_sum",
    "edge.signed.negative_magnitude_sum",
    "edge.signed.positive_fraction",
    "edge.signed.max_positive",
    "edge.signed.max_negative_magnitude",
    "root.influence",
)
LAYER_NODE_FIELDS = (
    "feature.count_fraction",
    "feature.activation_sum_fraction",
    "feature.influence_sum_fraction",
    "feature.activation.median",
    "feature.activation.p90",
    "feature.activation.max",
    "feature.influence.median",
    "feature.influence.p90",
    "feature.influence.max",
    "error.count_fraction",
    "error.influence_sum_fraction",
    "error.influence.median",
    "error.influence.p90",
    "error.influence.max",
)
LAYER_NODE_COORDINATES = tuple(
    f"layer_{layer}.{field}" for layer in range(8) for field in LAYER_NODE_FIELDS
)
LAYER_FLOW_COORDINATES = tuple(
    f"{FLOW_CATEGORIES[source]}->{FLOW_CATEGORIES[target]}.{field}"
    for source, target in FLOW_ALLOWED_PAIRS
    for field in ("influence_fraction", "signed_balance")
)
ROOT_DISTANCE_FIELDS = ("distance_1", "distance_2", "distance_3", "distance_4", "distance_5_plus", "unreachable")
TOPOLOGY_COORDINATES = (
    *(
        f"{kind}.root_{field}"
        for kind in ("token", "feature", "error")
        for field in ROOT_DISTANCE_FIELDS
    ),
    *(f"token.outdegree.{stat}" for stat in ("median", "p90", "max")),
    *(f"feature.indegree.{stat}" for stat in ("median", "p90", "max")),
    *(f"feature.outdegree.{stat}" for stat in ("median", "p90", "max")),
    *(f"error.outdegree.{stat}" for stat in ("median", "p90", "max")),
    "root.indegree",
    "feature.zero_indegree_fraction",
    "feature.zero_outdegree_fraction",
    "feature.branch_fraction",
    "feature.convergence_fraction",
    "token.branch_fraction",
    "error.branch_fraction",
    "weak_component_count",
    "longest_path_to_root",
    *(
        "motif2." + "->".join(NODE_KINDS[kind] for kind in sequence)
        for sequence in MOTIF2_SEQUENCES
    ),
    *(
        "motif3." + "->".join(NODE_KINDS[kind] for kind in sequence)
        for sequence in MOTIF3_SEQUENCES
    ),
)
POSITION_FIELDS = (
    "token.count_fraction",
    "token.influence_fraction",
    "feature.count_fraction",
    "feature.activation_fraction",
    "feature.influence_fraction",
    "error.count_fraction",
    "error.influence_fraction",
)
POSITION_COORDINATES = tuple(
    f"{region}_{third}.{field}"
    for region in ("prompt", "generated")
    for third in ("early", "middle", "late")
    for field in POSITION_FIELDS
)
BLOCK_COORDINATES = {
    "global": GLOBAL_COORDINATES,
    "layer_nodes": LAYER_NODE_COORDINATES,
    "layer_flow": LAYER_FLOW_COORDINATES,
    "topology": TOPOLOGY_COORDINATES,
    "position": POSITION_COORDINATES,
}
for _block_name, _coordinates in BLOCK_COORDINATES.items():
    if len(_coordinates) != BLOCK_DIMS[_block_name]:
        raise AssertionError(
            f"397D block {_block_name} has {len(_coordinates)} coordinates"
        )


def _log1p_coordinate_names() -> frozenset[str]:
    result = {name for name in GLOBAL_COORDINATES if not name.endswith("positive_fraction")}
    for name in LAYER_NODE_COORDINATES:
        if not name.endswith("fraction"):
            result.add(name)
    for name in TOPOLOGY_COORDINATES[18:31]:
        result.add(name)
    result.update(("weak_component_count", "longest_path_to_root"))
    return frozenset(result)


LOG1P_COORDINATE_NAMES = _log1p_coordinate_names()


@dataclass(frozen=True)
class Descriptor397Schema:
    format: str = DESCRIPTOR_397_FORMAT
    version: int = DESCRIPTOR_397_SCHEMA_VERSION

    @property
    def block_offsets(self) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name in BLOCK_ORDER:
            result[name] = (cursor, cursor + BLOCK_DIMS[name])
            cursor += BLOCK_DIMS[name]
        if cursor != DESCRIPTOR_DIM:
            raise AssertionError("397D block offsets do not sum to 397")
        return result

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return tuple(
            coordinate
            for block in BLOCK_ORDER
            for coordinate in BLOCK_COORDINATES[block]
        )

    @property
    def log1p_mask(self) -> Tensor:
        return torch.tensor(
            [name in LOG1P_COORDINATE_NAMES for name in self.coordinate_names],
            dtype=torch.bool,
        )

    def identity(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "dtype": "float32",
            "dimension": DESCRIPTOR_DIM,
            "n_layers": 8,
            "block_order": list(BLOCK_ORDER),
            "block_dims": dict(BLOCK_DIMS),
            "block_offsets": {
                name: list(offset) for name, offset in self.block_offsets.items()
            },
            "block_coordinates": {
                name: list(BLOCK_COORDINATES[name]) for name in BLOCK_ORDER
            },
            "flow_categories": list(FLOW_CATEGORIES),
            "flow_allowed_pairs": [list(pair) for pair in FLOW_ALLOWED_PAIRS],
            "motif2_sequences": [list(value) for value in MOTIF2_SEQUENCES],
            "motif3_sequences": [list(value) for value in MOTIF3_SEQUENCES],
            "log1p_mask": self.log1p_mask.to(torch.uint8).tolist(),
            "aggregation_rules": {
                "empty_group": "all_zero",
                "zero_denominator_fraction": 0.0,
                "quantiles": {
                    "median": 0.5,
                    "p90": 0.9,
                    "method": "linear",
                },
                "layer_flow_influence": "route_sum/global_edge_influence_sum",
                "layer_flow_signed_balance": "route_signed_sum/route_absolute_signed_sum",
                "root_distance": "shortest_directed_path_to_joint_root",
                "longest_path": "longest_directed_path_ending_at_joint_root",
                "weak_components": "undirected_connectivity_over_all_retained_nodes",
                "motifs": "directed_edge-combination_path_counts_normalized_within_path_length",
                "position_bins": "floor(relative_position*3/region_length)_clipped_0_2",
                "position_denominators": "separate_prompt_and_generated_prefix_totals",
            },
        }

    @property
    def sha256(self) -> str:
        return _semantic_sha256(self.identity())


SCHEMA_397 = Descriptor397Schema()
DESCRIPTOR_397_SCHEMA_SHA256 = (
    "e6c5528d1a06d2979babe776ff22e28ef08e40cf9a2fd9a5d69751418b337bda"
)
if SCHEMA_397.sha256 != DESCRIPTOR_397_SCHEMA_SHA256:
    raise RuntimeError("frozen 397D descriptor schema identity changed")


@dataclass(frozen=True)
class Descriptor397Record:
    event_key: str
    payload_artifact_id: str
    schema_sha256: str
    blocks: Mapping[str, Tensor]

    def raw(self) -> Tensor:
        value = torch.cat([self.blocks[name] for name in BLOCK_ORDER]).contiguous()
        if value.dtype != torch.float32 or value.shape != (DESCRIPTOR_DIM,):
            raise ValueError("397D descriptor concatenation changed")
        return value

    def validate(self) -> None:
        if len(self.event_key) != 64 or len(self.payload_artifact_id) != 64:
            raise ValueError("397D descriptor source identity is invalid")
        if self.schema_sha256 != SCHEMA_397.sha256:
            raise ValueError("397D descriptor schema identity changed")
        if tuple(self.blocks) != BLOCK_ORDER:
            raise ValueError("397D descriptor block order changed")
        for name in BLOCK_ORDER:
            value = self.blocks[name]
            if (
                value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.shape != (BLOCK_DIMS[name],)
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"397D descriptor block {name} is invalid")
        if not bool(torch.isfinite(self.raw()).all()):
            raise ValueError("397D descriptor is non-finite")


def _fraction(value: float, denominator: float) -> float:
    return float(value / denominator) if denominator > 0 else 0.0


def _shape(values: np.ndarray) -> tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(np.quantile(data, 0.5, method="linear")),
        float(np.quantile(data, 0.9, method="linear")),
        float(data.max()),
    )


def _sum_shape(values: np.ndarray) -> tuple[float, float, float, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    median, p90, maximum = _shape(data)
    return float(data.sum()), median, p90, maximum


def _node_arrays(payload: S5GraphPayloadV2) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.empty(payload.node_count, dtype=np.int8)
    typed = np.empty(payload.node_count, dtype=np.int16)
    feature_end = payload.feature_count
    error_end = feature_end + payload.error_count
    token_end = error_end + payload.token_count
    kinds[:feature_end] = 1
    typed[:feature_end] = 2 + payload.feature_layer.numpy().astype(np.int16)
    kinds[feature_end:error_end] = 2
    typed[feature_end:error_end] = 10 + payload.error_layer.numpy().astype(np.int16)
    kinds[error_end:token_end] = 0
    typed[error_end:token_end] = 0
    kinds[payload.root_index] = 3
    typed[payload.root_index] = 1
    return kinds, typed


def _global_block(payload: S5GraphPayloadV2) -> Tensor:
    activation = payload.feature_activation.numpy().astype(np.float64, copy=False)
    feature_influence = payload.feature_influence.numpy().astype(np.float64, copy=False)
    error_influence = payload.error_influence.numpy().astype(np.float64, copy=False)
    token_influence = payload.token_influence.numpy().astype(np.float64, copy=False)
    edge_influence = payload.edge_nonnegative_influence.numpy().astype(np.float64, copy=False)
    signed = payload.edge_signed_direct_effect.numpy().astype(np.float64, copy=False)
    positive = signed[signed > 0]
    negative = -signed[signed < 0]
    values = (
        float(payload.feature_count),
        *_sum_shape(activation),
        *_sum_shape(feature_influence),
        float(payload.error_count),
        *_sum_shape(error_influence),
        float(payload.token_count),
        *_sum_shape(token_influence),
        float(payload.edge_count),
        *_sum_shape(edge_influence),
        float(positive.sum()),
        float(negative.sum()),
        _fraction(float(positive.size), float(signed.size)),
        float(positive.max()) if positive.size else 0.0,
        float(negative.max()) if negative.size else 0.0,
        float(payload.root_influence),
    )
    result = torch.tensor(values, dtype=torch.float32)
    if result.shape != (30,):
        raise AssertionError("397D Global block has wrong shape")
    return result


def _layer_nodes_block(payload: S5GraphPayloadV2) -> Tensor:
    feature_layers = payload.feature_layer.numpy().astype(np.int64, copy=False)
    error_layers = payload.error_layer.numpy().astype(np.int64, copy=False)
    activation = payload.feature_activation.numpy().astype(np.float64, copy=False)
    feature_influence = payload.feature_influence.numpy().astype(np.float64, copy=False)
    error_influence = payload.error_influence.numpy().astype(np.float64, copy=False)
    activation_total = float(activation.sum())
    feature_influence_total = float(feature_influence.sum())
    error_influence_total = float(error_influence.sum())
    rows: list[float] = []
    for layer in range(8):
        feature_mask = feature_layers == layer
        error_mask = error_layers == layer
        layer_activation = activation[feature_mask]
        layer_feature_influence = feature_influence[feature_mask]
        layer_error_influence = error_influence[error_mask]
        rows.extend(
            (
                _fraction(float(feature_mask.sum()), float(payload.feature_count)),
                _fraction(float(layer_activation.sum()), activation_total),
                _fraction(float(layer_feature_influence.sum()), feature_influence_total),
                *_shape(layer_activation),
                *_shape(layer_feature_influence),
                _fraction(float(error_mask.sum()), float(payload.error_count)),
                _fraction(float(layer_error_influence.sum()), error_influence_total),
                *_shape(layer_error_influence),
            )
        )
    result = torch.tensor(rows, dtype=torch.float32)
    if result.shape != (112,):
        raise AssertionError("397D Layer Nodes block has wrong shape")
    return result


def _layer_flow_block(payload: S5GraphPayloadV2, typed: np.ndarray) -> Tensor:
    source = payload.edge_source.numpy().astype(np.int64, copy=False)
    target = payload.edge_target.numpy().astype(np.int64, copy=False)
    influence = payload.edge_nonnegative_influence.numpy().astype(np.float64, copy=False)
    signed = payload.edge_signed_direct_effect.numpy().astype(np.float64, copy=False)
    route = typed[source] * 18 + typed[target]
    influence_by_route = np.bincount(route, weights=influence, minlength=18 * 18)
    signed_by_route = np.bincount(route, weights=signed, minlength=18 * 18)
    absolute_by_route = np.bincount(route, weights=np.abs(signed), minlength=18 * 18)
    total_influence = float(influence.sum())
    rows: list[float] = []
    for source_category, target_category in FLOW_ALLOWED_PAIRS:
        index = source_category * 18 + target_category
        rows.extend(
            (
                _fraction(float(influence_by_route[index]), total_influence),
                _fraction(float(signed_by_route[index]), float(absolute_by_route[index])),
            )
        )
    result = torch.tensor(rows, dtype=torch.float32)
    if result.shape != (162,):
        raise AssertionError("397D Layer Flow block has wrong shape")
    return result


def _topology_arrays(
    node_count: int,
    source: np.ndarray,
    target: np.ndarray,
    root_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float]:
    indegree = np.bincount(target, minlength=node_count).astype(np.int64)
    outdegree = np.bincount(source, minlength=node_count).astype(np.int64)
    order = np.argsort(source, kind="stable")
    sorted_target = target[order]
    counts = np.bincount(source, minlength=node_count)
    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    work = indegree.copy()
    queue = deque(np.flatnonzero(work == 0).tolist())
    topological: list[int] = []
    while queue:
        node = int(queue.popleft())
        topological.append(node)
        neighbors = sorted_target[int(offsets[node]) : int(offsets[node + 1])]
        for neighbor in neighbors.tolist():
            work[neighbor] -= 1
            if work[neighbor] == 0:
                queue.append(int(neighbor))
    if len(topological) != node_count:
        raise ValueError("397D descriptor source Graph is not a DAG")

    shortest = np.full(node_count, np.inf, dtype=np.float64)
    longest = np.full(node_count, -np.inf, dtype=np.float64)
    shortest[root_index] = 0.0
    longest[root_index] = 0.0
    for node in reversed(topological):
        if node == root_index:
            continue
        neighbors = sorted_target[int(offsets[node]) : int(offsets[node + 1])]
        reachable = neighbors[np.isfinite(shortest[neighbors])]
        if reachable.size:
            shortest[node] = 1.0 + float(shortest[reachable].min())
            longest[node] = 1.0 + float(longest[reachable].max())

    parent = np.arange(node_count, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for left, right in zip(source.tolist(), target.tolist(), strict=True):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root != right_root:
            parent[right_root] = left_root
    components = len({find(node) for node in range(node_count)})
    finite_longest = longest[np.isfinite(longest)]
    longest_path = float(finite_longest.max()) if finite_longest.size else 0.0
    return (
        indegree,
        outdegree,
        shortest,
        np.asarray(topological, dtype=np.int64),
        components,
        longest_path,
    )


def _root_distance_fractions(distance: np.ndarray) -> tuple[float, ...]:
    if distance.size == 0:
        return (0.0,) * 6
    return (
        float(np.mean(distance == 1)),
        float(np.mean(distance == 2)),
        float(np.mean(distance == 3)),
        float(np.mean(distance == 4)),
        float(np.mean((distance >= 5) & np.isfinite(distance))),
        float(np.mean(~np.isfinite(distance))),
    )


def _motif_fractions(
    payload: S5GraphPayloadV2,
    kinds: np.ndarray,
) -> tuple[float, ...]:
    source = payload.edge_source.numpy().astype(np.int64, copy=False)
    target = payload.edge_target.numpy().astype(np.int64, copy=False)
    incoming = np.zeros((payload.node_count, 4), dtype=np.float64)
    outgoing = np.zeros((payload.node_count, 4), dtype=np.float64)
    np.add.at(incoming, (target, kinds[source]), 1.0)
    np.add.at(outgoing, (source, kinds[target]), 1.0)

    feature_nodes = np.arange(payload.feature_count, dtype=np.int64)
    two = []
    for first, _middle, last in MOTIF2_SEQUENCES:
        two.append(float((incoming[feature_nodes, first] * outgoing[feature_nodes, last]).sum()))

    feature_to_feature = (source < payload.feature_count) & (target < payload.feature_count)
    center_source = source[feature_to_feature]
    center_target = target[feature_to_feature]
    three = []
    for first, _second, _third, last in MOTIF3_SEQUENCES:
        three.append(
            float(
                (
                    incoming[center_source, first]
                    * outgoing[center_target, last]
                ).sum()
            )
        )
    two_total = sum(two)
    three_total = sum(three)
    return tuple(
        [(_fraction(value, two_total)) for value in two]
        + [(_fraction(value, three_total)) for value in three]
    )


def _topology_block(payload: S5GraphPayloadV2, kinds: np.ndarray) -> Tensor:
    source = payload.edge_source.numpy().astype(np.int64, copy=False)
    target = payload.edge_target.numpy().astype(np.int64, copy=False)
    indegree, outdegree, distance, _order, components, longest_path = _topology_arrays(
        payload.node_count,
        source,
        target,
        payload.root_index,
    )
    feature_end = payload.feature_count
    error_end = feature_end + payload.error_count
    token_end = error_end + payload.token_count
    feature_nodes = np.arange(0, feature_end, dtype=np.int64)
    error_nodes = np.arange(feature_end, error_end, dtype=np.int64)
    token_nodes = np.arange(error_end, token_end, dtype=np.int64)
    rows: list[float] = []
    for nodes in (token_nodes, feature_nodes, error_nodes):
        rows.extend(_root_distance_fractions(distance[nodes]))
    rows.extend(_shape(outdegree[token_nodes]))
    rows.extend(_shape(indegree[feature_nodes]))
    rows.extend(_shape(outdegree[feature_nodes]))
    rows.extend(_shape(outdegree[error_nodes]))
    rows.append(float(indegree[payload.root_index]))
    rows.extend(
        (
            float(np.mean(indegree[feature_nodes] == 0)) if feature_nodes.size else 0.0,
            float(np.mean(outdegree[feature_nodes] == 0)) if feature_nodes.size else 0.0,
            float(np.mean(outdegree[feature_nodes] > 1)) if feature_nodes.size else 0.0,
            float(np.mean(indegree[feature_nodes] > 1)) if feature_nodes.size else 0.0,
            float(np.mean(outdegree[token_nodes] > 1)) if token_nodes.size else 0.0,
            float(np.mean(outdegree[error_nodes] > 1)) if error_nodes.size else 0.0,
            float(components),
            longest_path,
            *_motif_fractions(payload, kinds),
        )
    )
    result = torch.tensor(rows, dtype=torch.float32)
    if result.shape != (51,):
        raise AssertionError("397D Topology block has wrong shape")
    return result


def _third_indices(positions: np.ndarray, *, start: int, length: int) -> np.ndarray:
    if positions.size == 0:
        return np.empty(0, dtype=np.int64)
    if length <= 0:
        return np.full(positions.shape, -1, dtype=np.int64)
    relative = positions - int(start)
    result = np.floor_divide(relative * 3, int(length)).astype(np.int64)
    return np.clip(result, 0, 2)


def _position_block(payload: S5GraphPayloadV2) -> Tensor:
    feature_position = payload.feature_position.numpy().astype(np.int64, copy=False)
    error_position = payload.error_position.numpy().astype(np.int64, copy=False)
    token_position = np.arange(payload.token_count, dtype=np.int64)
    activation = payload.feature_activation.numpy().astype(np.float64, copy=False)
    feature_influence = payload.feature_influence.numpy().astype(np.float64, copy=False)
    error_influence = payload.error_influence.numpy().astype(np.float64, copy=False)
    token_influence = payload.token_influence.numpy().astype(np.float64, copy=False)
    rows: list[float] = []
    regions = (
        (0, int(payload.condition_length)),
        (
            int(payload.condition_length),
            int(payload.token_count - payload.condition_length),
        ),
    )
    for start, length in regions:
        feature_region = (feature_position >= start) & (feature_position < start + length)
        error_region = (error_position >= start) & (error_position < start + length)
        token_region = (token_position >= start) & (token_position < start + length)
        feature_thirds = _third_indices(feature_position, start=start, length=length)
        error_thirds = _third_indices(error_position, start=start, length=length)
        token_thirds = _third_indices(token_position, start=start, length=length)
        denominators = (
            float(token_region.sum()),
            float(token_influence[token_region].sum()),
            float(feature_region.sum()),
            float(activation[feature_region].sum()),
            float(feature_influence[feature_region].sum()),
            float(error_region.sum()),
            float(error_influence[error_region].sum()),
        )
        for third in range(3):
            feature_bin = feature_region & (feature_thirds == third)
            error_bin = error_region & (error_thirds == third)
            token_bin = token_region & (token_thirds == third)
            numerators = (
                float(token_bin.sum()),
                float(token_influence[token_bin].sum()),
                float(feature_bin.sum()),
                float(activation[feature_bin].sum()),
                float(feature_influence[feature_bin].sum()),
                float(error_bin.sum()),
                float(error_influence[error_bin].sum()),
            )
            rows.extend(
                _fraction(numerator, denominator)
                for numerator, denominator in zip(
                    numerators, denominators, strict=True
                )
            )
    result = torch.tensor(rows, dtype=torch.float32)
    if result.shape != (42,):
        raise AssertionError("397D Relative Position block has wrong shape")
    return result


def vectorize_payload_v2_397_record(
    payload: S5GraphPayloadV2,
    *,
    n_layers: int = 8,
    d_feature: int,
    archive_identity_sha256: str | None = None,
) -> Descriptor397Record:
    """Convert one validated Payload v2 event into named raw 397D blocks."""

    if int(n_layers) != 8:
        raise ValueError("397D descriptor requires the frozen eight-layer S5 model")
    payload.validate(
        n_layers=n_layers,
        d_feature=d_feature,
        archive_identity_sha256=archive_identity_sha256,
    )
    kinds, typed = _node_arrays(payload)
    record = Descriptor397Record(
        event_key=payload.event_key,
        payload_artifact_id=payload.payload_artifact_id,
        schema_sha256=SCHEMA_397.sha256,
        blocks={
            "global": _global_block(payload),
            "layer_nodes": _layer_nodes_block(payload),
            "layer_flow": _layer_flow_block(payload, typed),
            "topology": _topology_block(payload, kinds),
            "position": _position_block(payload),
        },
    )
    record.validate()
    return record


def vectorize_payload_v2_397(
    payload: S5GraphPayloadV2,
    *,
    n_layers: int = 8,
    d_feature: int,
    archive_identity_sha256: str | None = None,
) -> Tensor:
    """Return one contiguous finite CPU ``torch.float32[397]`` raw vector."""

    return vectorize_payload_v2_397_record(
        payload,
        n_layers=n_layers,
        d_feature=d_feature,
        archive_identity_sha256=archive_identity_sha256,
    ).raw()


def fixed_log1p_397(raw: Tensor) -> Tensor:
    """Apply the frozen data-independent log1p treatment to a raw 397D vector."""

    value = raw.detach().cpu().to(torch.float32).contiguous()
    if value.shape != (DESCRIPTOR_DIM,) or not bool(torch.isfinite(value).all()):
        raise ValueError("fixed_log1p_397 expects a finite FP32[397] vector")
    mask = SCHEMA_397.log1p_mask
    if bool(torch.any(value[mask] < 0)):
        raise ValueError("397D log1p coordinate is unexpectedly negative")
    result = value.clone()
    result[mask] = torch.log1p(result[mask])
    return result.contiguous()


def vectorize_many_payload_v2_397(
    payloads: Sequence[S5GraphPayloadV2],
    *,
    n_layers: int = 8,
    d_feature: int,
    archive_identity_sha256: str | None = None,
) -> Tensor:
    """Vectorize a small in-memory fixture batch; production code must stream."""

    if not payloads:
        return torch.empty((0, DESCRIPTOR_DIM), dtype=torch.float32)
    return torch.stack(
        [
            vectorize_payload_v2_397(
                payload,
                n_layers=n_layers,
                d_feature=d_feature,
                archive_identity_sha256=archive_identity_sha256,
            )
            for payload in payloads
        ]
    ).contiguous()

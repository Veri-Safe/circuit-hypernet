from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .storage import load_safetensors_bundle, write_safetensors_bundle
from .influence import prune_direct_effect_graph


PAYLOAD_FORMAT = "nvarc_qwen3_joint_span_compact_v1"


class NodeType(IntEnum):
    FEATURE = 0
    ERROR = 1
    PROMPT_TOKEN = 2
    GENERATED_PREFIX_TOKEN = 3
    SPAN_ROOT = 4


def _is_directed_acyclic_edges(
    edge_source: Tensor, edge_target: Tensor, node_count: int
) -> bool:
    """Check a compact edge list with Kahn's algorithm.

    Payloads can contain millions of edges.  Building a NetworkX node/edge
    object for validation multiplies memory use and makes the validation cost
    dominate the GPU work.  The edge contract has already been checked by the
    caller, so a CSR-like outgoing index plus Kahn's algorithm is sufficient
    and preserves the same DAG predicate in O(E log E) time and O(E) memory.
    """

    if edge_source.numel() == 0:
        return True
    sources = edge_source.detach().cpu().numpy()
    targets = edge_target.detach().cpu().numpy()
    order = np.argsort(sources, kind="quicksort")
    sorted_targets = targets[order]
    del order

    counts = np.bincount(sources, minlength=node_count)
    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    indegree = np.bincount(targets, minlength=node_count).astype(
        np.int64, copy=False
    )
    queue = np.flatnonzero(indegree == 0).tolist()
    head = 0
    processed = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        processed += 1
        start = int(offsets[node])
        end = int(offsets[node + 1])
        neighbors = sorted_targets[start:end]
        if neighbors.size == 0:
            continue
        updated = indegree[neighbors] - 1
        indegree[neighbors] = updated
        queue.extend(neighbors[updated == 0].tolist())
    return processed == node_count


def _float_tensor(value: Tensor | Sequence[float]) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()


def _int_tensor(value: Tensor | Sequence[int], dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value, dtype=dtype).detach().cpu().contiguous()


@dataclass(frozen=True)
class CompactPayload:
    event_key: str
    artifact_id: str
    condition_length: int
    total_active_features: int
    feature_layer: Tensor
    feature_position: Tensor
    feature_id: Tensor
    feature_activation: Tensor
    feature_influence: Tensor
    feature_cumulative_score: Tensor
    error_layer: Tensor
    error_position: Tensor
    error_influence: Tensor
    error_cumulative_score: Tensor
    replay_token_ids: Tensor
    token_influence: Tensor
    token_cumulative_score: Tensor
    root_influence: float
    root_cumulative_score: float
    edge_source: Tensor
    edge_target: Tensor
    edge_signed_direct_effect: Tensor
    edge_nonnegative_influence: Tensor
    target_token_ids: Tensor
    target_positions: Tensor
    target_probabilities: Tensor
    target_entropies: Tensor
    target_log_probabilities: Tensor

    @property
    def feature_count(self) -> int:
        return int(self.feature_id.numel())

    @property
    def error_count(self) -> int:
        return int(self.error_layer.numel())

    @property
    def token_count(self) -> int:
        return int(self.replay_token_ids.numel())

    @property
    def target_count(self) -> int:
        return int(self.target_token_ids.numel())

    @property
    def node_count(self) -> int:
        return self.feature_count + self.error_count + self.token_count + 1

    @property
    def token_start(self) -> int:
        return self.feature_count + self.error_count

    @property
    def root_index(self) -> int:
        return self.node_count - 1

    def node_types(self) -> Tensor:
        token_positions = torch.arange(self.token_count, dtype=torch.int32)
        token_types = torch.where(
            token_positions < self.condition_length,
            int(NodeType.PROMPT_TOKEN),
            int(NodeType.GENERATED_PREFIX_TOKEN),
        ).to(torch.uint8)
        return torch.cat(
            (
                torch.full(
                    (self.feature_count,), int(NodeType.FEATURE), dtype=torch.uint8
                ),
                torch.full((self.error_count,), int(NodeType.ERROR), dtype=torch.uint8),
                token_types,
                torch.tensor([int(NodeType.SPAN_ROOT)], dtype=torch.uint8),
            )
        )

    def node_influences(self) -> Tensor:
        return torch.cat(
            (
                self.feature_influence.float(),
                self.error_influence.float(),
                self.token_influence.float(),
                torch.tensor([self.root_influence], dtype=torch.float32),
            )
        )

    def node_cumulative_scores(self) -> Tensor:
        return torch.cat(
            (
                self.feature_cumulative_score.float(),
                self.error_cumulative_score.float(),
                self.token_cumulative_score.float(),
                torch.tensor([self.root_cumulative_score], dtype=torch.float32),
            )
        )

    def validate(self) -> None:
        if len(self.event_key) != 64 or len(self.artifact_id) != 64:
            raise ValueError("Payload identities must be SHA256 values")
        if (
            self.condition_length <= 0
            or self.total_active_features < self.feature_count
        ):
            raise ValueError("Payload condition or active-feature count is invalid")
        aligned = (
            (
                self.feature_layer,
                self.feature_position,
                self.feature_id,
                self.feature_activation,
                self.feature_influence,
                self.feature_cumulative_score,
            ),
            (
                self.error_layer,
                self.error_position,
                self.error_influence,
                self.error_cumulative_score,
            ),
            (self.replay_token_ids, self.token_influence, self.token_cumulative_score),
            (
                self.edge_source,
                self.edge_target,
                self.edge_signed_direct_effect,
                self.edge_nonnegative_influence,
            ),
            (
                self.target_token_ids,
                self.target_positions,
                self.target_probabilities,
                self.target_entropies,
                self.target_log_probabilities,
            ),
        )
        for group in aligned:
            if len({int(item.numel()) for item in group}) != 1:
                raise ValueError("Payload array group is misaligned")
        if self.token_count != self.condition_length + self.target_count - 1:
            raise ValueError("Replay tokens must equal condition plus y_1..y_(K-1)")
        expected_positions = torch.arange(
            self.condition_length - 1,
            self.condition_length + self.target_count - 1,
            dtype=torch.int32,
        )
        if not torch.equal(self.target_positions.to(torch.int32), expected_positions):
            raise ValueError("Payload target positions changed")
        if self.feature_count and (
            int(self.feature_layer.min()) < 0
            or int(self.feature_layer.max()) >= 36
            or int(self.feature_position.min()) < 0
            or int(self.feature_position.max()) >= self.token_count
            or int(self.feature_id.min()) < 0
            or int(self.feature_id.max()) >= 163_840
        ):
            raise ValueError("Payload feature identity is out of range")
        if self.error_count and (
            int(self.error_layer.min()) < 0
            or int(self.error_layer.max()) >= 36
            or int(self.error_position.min()) < 0
            or int(self.error_position.max()) >= self.token_count
        ):
            raise ValueError("Payload error identity is out of range")
        if self.edge_source.numel():
            if (
                int(self.edge_source.min()) < 0
                or int(self.edge_target.min()) < 0
                or int(self.edge_source.max()) >= self.node_count
                or int(self.edge_target.max()) >= self.node_count
            ):
                raise ValueError("Payload edge endpoint is out of range")
            if torch.any(self.edge_source == self.edge_target):
                raise ValueError("Payload contains a self edge")
            pairs = torch.stack(
                (self.edge_source.long(), self.edge_target.long()), dim=1
            )
            if torch.unique(pairs, dim=0).shape[0] != pairs.shape[0]:
                raise ValueError("Payload contains duplicate directed edges")
            if torch.any(self.edge_source == self.root_index):
                raise ValueError("Virtual span root cannot be an edge source")
            token_targets = (self.edge_target >= self.token_start) & (
                self.edge_target < self.root_index
            )
            if torch.any(token_targets):
                raise ValueError("Token nodes cannot be edge targets")
        floating = (
            self.feature_activation,
            self.feature_influence,
            self.feature_cumulative_score,
            self.error_influence,
            self.error_cumulative_score,
            self.token_influence,
            self.token_cumulative_score,
            self.edge_signed_direct_effect,
            self.edge_nonnegative_influence,
            self.target_probabilities,
            self.target_entropies,
            self.target_log_probabilities,
            torch.tensor([self.root_influence, self.root_cumulative_score]),
        )
        if any(not torch.isfinite(value.float()).all() for value in floating):
            raise ValueError("Payload contains non-finite floating-point values")
        nonnegative = (
            self.feature_influence,
            self.error_influence,
            self.token_influence,
            self.edge_nonnegative_influence,
        )
        if any(torch.any(value < 0) for value in nonnegative):
            raise ValueError("Payload influence channel contains a negative value")
        cumulative = (
            self.feature_cumulative_score,
            self.error_cumulative_score,
            self.token_cumulative_score,
            torch.tensor([self.root_cumulative_score]),
        )
        if any(torch.any((value < 0) | (value > 1.0 + 1e-6)) for value in cumulative):
            raise ValueError("Payload cumulative pruning score lies outside [0, 1]")
        if torch.any((self.target_probabilities < 0) | (self.target_probabilities > 1)):
            raise ValueError("Payload target probability lies outside [0, 1]")
        if torch.any((self.target_token_ids < 0) | (self.target_token_ids > 10)):
            raise ValueError("Payload target contains a non-grid token")
        if torch.any((self.replay_token_ids < 0) | (self.replay_token_ids > 15)):
            raise ValueError(
                "Payload replay contains a token outside the 16-token vocabulary"
            )
        if not _is_directed_acyclic_edges(
            self.edge_source, self.edge_target, self.node_count
        ):
            raise ValueError("Payload retained graph is not a DAG")


def make_payload(**values: Any) -> CompactPayload:
    tensor_specs = {
        "feature_layer": torch.int16,
        "feature_position": torch.int32,
        "feature_id": torch.int32,
        "feature_activation": torch.float32,
        "feature_influence": torch.float32,
        "feature_cumulative_score": torch.float32,
        "error_layer": torch.int16,
        "error_position": torch.int32,
        "error_influence": torch.float32,
        "error_cumulative_score": torch.float32,
        "replay_token_ids": torch.int16,
        "token_influence": torch.float32,
        "token_cumulative_score": torch.float32,
        "edge_source": torch.int32,
        "edge_target": torch.int32,
        "edge_signed_direct_effect": torch.float32,
        "edge_nonnegative_influence": torch.float32,
        "target_token_ids": torch.int16,
        "target_positions": torch.int32,
        "target_probabilities": torch.float32,
        "target_entropies": torch.float32,
        "target_log_probabilities": torch.float32,
    }
    normalized = dict(values)
    for key, dtype in tensor_specs.items():
        normalized[key] = (
            torch.as_tensor(normalized[key], dtype=dtype).cpu().contiguous()
        )
    result = CompactPayload(**normalized)
    result.validate()
    return result


def build_payload_from_dense_attribution(
    *,
    event_key: str,
    artifact_id: str,
    condition_length: int,
    replay_token_ids: Tensor,
    active_features: Tensor,
    activation_values: Tensor,
    selected_features: Tensor,
    signed_direct_effect: Tensor,
    target_token_ids: Tensor,
    target_positions: Tensor,
    target_probabilities: Tensor,
    target_entropies: Tensor,
    target_log_probabilities: Tensor,
    node_threshold: float,
    edge_threshold: float,
    n_layers: int = 36,
) -> CompactPayload:
    """Prune a CodeCircuit-ordered dense reference graph into the compact schema."""

    replay_token_ids = replay_token_ids.detach().cpu().long()
    active_features = active_features.detach().cpu().long()
    activation_values = activation_values.detach().cpu().float()
    selected_features = selected_features.detach().cpu().long()
    matrix = signed_direct_effect.detach().cpu().float()
    n_pos = int(replay_token_ids.numel())
    n_selected = int(selected_features.numel())
    n_errors = n_layers * n_pos
    token_start = n_selected + n_errors
    root_index = token_start + n_pos
    expected_nodes = root_index + 1
    if matrix.shape != (expected_nodes, expected_nodes):
        raise ValueError(
            f"Dense attribution shape {tuple(matrix.shape)} != {(expected_nodes, expected_nodes)}"
        )
    if active_features.ndim != 2 or active_features.shape[1] != 3:
        raise ValueError(
            "active_features must have columns (layer, position, feature_id)"
        )
    if activation_values.shape != (active_features.shape[0],):
        raise ValueError("activation_values do not align with active_features")
    if selected_features.numel() and (
        int(selected_features.min()) < 0
        or int(selected_features.max()) >= active_features.shape[0]
    ):
        raise ValueError("selected_features index outside active feature table")
    token_indices = torch.arange(token_start, token_start + n_pos, dtype=torch.long)
    pruned = prune_direct_effect_graph(
        matrix,
        token_indices=token_indices,
        root_index=root_index,
        feature_count=n_selected,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )
    kept_features = torch.where(pruned.node_mask[:n_selected])[0]
    error_slice = pruned.node_mask[n_selected:token_start]
    kept_errors = torch.where(error_slice)[0]
    kept_old = torch.cat(
        (
            kept_features,
            kept_errors + n_selected,
            token_indices,
            torch.tensor([root_index], dtype=torch.long),
        )
    )
    old_to_new = torch.full((expected_nodes,), -1, dtype=torch.long)
    old_to_new[kept_old] = torch.arange(kept_old.numel(), dtype=torch.long)
    edge_rows, edge_cols = torch.where(pruned.edge_mask & (matrix != 0))
    keep_edges = pruned.node_mask[edge_rows] & pruned.node_mask[edge_cols]
    edge_rows = edge_rows[keep_edges]
    edge_cols = edge_cols[keep_edges]
    source = old_to_new[edge_cols]
    target = old_to_new[edge_rows]
    if torch.any(source < 0) or torch.any(target < 0):
        raise AssertionError("Pruned edge references a removed node")

    feature_active_indices = selected_features[kept_features]
    feature_records = active_features[feature_active_indices]
    error_layers = torch.div(kept_errors, n_pos, rounding_mode="floor")
    error_positions = kept_errors % n_pos
    token_scores = pruned.node_influence[token_indices]
    token_cumulative = pruned.node_cumulative_score[token_indices]
    return make_payload(
        event_key=event_key,
        artifact_id=artifact_id,
        condition_length=int(condition_length),
        total_active_features=int(active_features.shape[0]),
        feature_layer=feature_records[:, 0],
        feature_position=feature_records[:, 1],
        feature_id=feature_records[:, 2],
        feature_activation=activation_values[feature_active_indices],
        feature_influence=pruned.node_influence[kept_features],
        feature_cumulative_score=pruned.node_cumulative_score[kept_features],
        error_layer=error_layers,
        error_position=error_positions,
        error_influence=pruned.node_influence[n_selected + kept_errors],
        error_cumulative_score=pruned.node_cumulative_score[n_selected + kept_errors],
        replay_token_ids=replay_token_ids,
        token_influence=token_scores,
        token_cumulative_score=token_cumulative,
        root_influence=float(pruned.node_influence[root_index]),
        root_cumulative_score=float(pruned.node_cumulative_score[root_index]),
        edge_source=source,
        edge_target=target,
        edge_signed_direct_effect=matrix[edge_rows, edge_cols],
        edge_nonnegative_influence=pruned.edge_influence[edge_rows, edge_cols],
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_probabilities=target_probabilities,
        target_entropies=target_entropies,
        target_log_probabilities=target_log_probabilities,
    )


def build_payload_from_streaming_attribution(
    *,
    event_key: str,
    artifact_id: str,
    condition_length: int,
    attribution: Any,
) -> CompactPayload:
    """Convert the streaming exact result without restoring a pre-pruning dense graph."""

    graph = attribution.graph
    active_features = attribution.active_features.detach().cpu().long()
    activations = attribution.activation_values.detach().cpu().float()
    replay = attribution.replay_token_ids.detach().cpu().long()
    objective = attribution.objective
    if active_features.ndim != 2 or active_features.shape[1] != 3:
        raise ValueError("Streaming active-feature table changed shape")
    feature_records = active_features.index_select(0, graph.retained_features)
    feature_activations = activations.index_select(0, graph.retained_features)
    error_layers = torch.div(
        graph.retained_error_offsets, replay.numel(), rounding_mode="floor"
    )
    error_positions = graph.retained_error_offsets % replay.numel()
    if not torch.equal(
        graph.retained_token_offsets,
        torch.arange(replay.numel(), dtype=torch.long),
    ):
        raise ValueError("Streaming graph did not retain every logical replay token")
    rows, columns = torch.where(graph.signed_direct_effect != 0)
    feature_count = int(graph.retained_features.numel())
    error_count = int(graph.retained_error_offsets.numel())
    token_start = feature_count + error_count
    root_index = graph.root_index
    if (token_start, root_index) != (
        graph.token_start,
        graph.signed_direct_effect.shape[0] - 1,
    ):
        raise ValueError("Streaming graph node boundaries changed")
    node_influence = graph.node_influence
    cumulative = graph.node_cumulative_score
    return make_payload(
        event_key=event_key,
        artifact_id=artifact_id,
        condition_length=int(condition_length),
        total_active_features=int(active_features.shape[0]),
        feature_layer=feature_records[:, 0],
        feature_position=feature_records[:, 1],
        feature_id=feature_records[:, 2],
        feature_activation=feature_activations,
        feature_influence=node_influence[:feature_count],
        feature_cumulative_score=cumulative[:feature_count],
        error_layer=error_layers,
        error_position=error_positions,
        error_influence=node_influence[feature_count:token_start],
        error_cumulative_score=cumulative[feature_count:token_start],
        replay_token_ids=replay,
        token_influence=node_influence[token_start:root_index],
        token_cumulative_score=cumulative[token_start:root_index],
        root_influence=float(node_influence[root_index]),
        root_cumulative_score=float(cumulative[root_index]),
        edge_source=columns,
        edge_target=rows,
        edge_signed_direct_effect=graph.signed_direct_effect[rows, columns],
        edge_nonnegative_influence=graph.edge_nonnegative_influence[rows, columns],
        target_token_ids=objective.target_token_ids,
        target_positions=objective.target_positions,
        target_probabilities=objective.target_probabilities,
        target_entropies=objective.target_entropies,
        target_log_probabilities=objective.target_log_probabilities,
    )


_ARRAY_FIELDS = (
    "feature_layer",
    "feature_position",
    "feature_id",
    "feature_activation",
    "feature_influence",
    "feature_cumulative_score",
    "error_layer",
    "error_position",
    "error_influence",
    "error_cumulative_score",
    "replay_token_ids",
    "token_influence",
    "token_cumulative_score",
    "edge_source",
    "edge_target",
    "edge_signed_direct_effect",
    "edge_nonnegative_influence",
    "target_token_ids",
    "target_positions",
    "target_probabilities",
    "target_entropies",
    "target_log_probabilities",
)
_GROUPS = {
    "feature": _ARRAY_FIELDS[0:6],
    "error": _ARRAY_FIELDS[6:10],
    "token": _ARRAY_FIELDS[10:13],
    "edge": _ARRAY_FIELDS[13:17],
    "target": _ARRAY_FIELDS[17:22],
}


def write_payload_chunk(
    path: str | Path,
    payloads: Sequence[CompactPayload],
    *,
    config_sha256: str,
    production_policy_sha256: str,
    policy_eligibility: str | None = None,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("Cannot write an empty payload chunk")
    concatenated: dict[str, list[Tensor]] = {field: [] for field in _ARRAY_FIELDS}
    offsets = {group: [0] for group in _GROUPS}
    records = []
    for payload in payloads:
        payload.validate()
        for field in _ARRAY_FIELDS:
            concatenated[field].append(getattr(payload, field))
        for group, fields in _GROUPS.items():
            offsets[group].append(
                offsets[group][-1] + int(getattr(payload, fields[0]).numel())
            )
        records.append(
            {
                "event_key": payload.event_key,
                "artifact_id": payload.artifact_id,
                "condition_length": payload.condition_length,
                "total_active_features": payload.total_active_features,
                "root_influence": payload.root_influence,
                "root_cumulative_score": payload.root_cumulative_score,
            }
        )
    tensors = {}
    for field, chunks in concatenated.items():
        tensors[field] = torch.cat(chunks) if chunks else torch.empty(0)
    for group, values in offsets.items():
        tensors[f"{group}_offsets"] = torch.tensor(values, dtype=torch.int64)
    metadata = {
        "format": PAYLOAD_FORMAT,
        "config_sha256": config_sha256,
        "production_policy_sha256": production_policy_sha256,
        "root_view": "joint_signed_raw_sum",
        "events": len(payloads),
        "records": records,
    }
    if policy_eligibility is not None:
        metadata["policy_eligibility"] = policy_eligibility
    return write_safetensors_bundle(
        path,
        tensors=tensors,
        metadata=metadata,
    )


def read_payload_chunk(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> tuple[CompactPayload, ...]:
    tensors, metadata = load_safetensors_bundle(path, expected_format=PAYLOAD_FORMAT)
    if expected_identity is not None:
        mismatches = {
            key: {"expected": expected, "actual": metadata.get(key)}
            for key, expected in expected_identity.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Payload chunk identity changed: {mismatches}")
    records = metadata.get("records")
    if not isinstance(records, list):
        raise ValueError("Payload metadata has no records list")
    group_offsets = {group: tensors[f"{group}_offsets"].tolist() for group in _GROUPS}
    if any(len(value) != len(records) + 1 for value in group_offsets.values()):
        raise ValueError("Payload offsets do not align with records")
    result = []
    for index, record in enumerate(records):
        values: dict[str, Any] = dict(record)
        for group, fields in _GROUPS.items():
            start, end = group_offsets[group][index : index + 2]
            for field in fields:
                values[field] = tensors[field][start:end]
        result.append(make_payload(**values))
    return tuple(result)

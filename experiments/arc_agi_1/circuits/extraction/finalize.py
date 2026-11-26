from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .payload import CompactPayload
from .storage import load_safetensors_bundle, write_safetensors_bundle
from .trails import TRAIL_SCHEMA, TrailNode, TrailResult, extract_top_k_trails
from .vector59 import VECTOR_DIM, VECTOR_SCHEMA, VectorResult, vectorize_payload


FINAL_FORMAT = "nvarc_rearc_circuit_final_shard_v1"
TRAIL_MATERIALIZATION_COMPUTED = "computed"
TRAIL_MATERIALIZATION_SKIPPED = "skipped"
TRAIL_MATERIALIZATION_VALUES = frozenset(
    (TRAIL_MATERIALIZATION_COMPUTED, TRAIL_MATERIALIZATION_SKIPPED)
)


def _ascii_hashes(values: Sequence[str]) -> torch.Tensor:
    if any(len(value) != 64 or not value.isascii() for value in values):
        raise ValueError("Final artifact identity is not a 64-character ASCII hash")
    return torch.tensor(
        [list(value.encode("ascii")) for value in values], dtype=torch.uint8
    )


def _node_lookup(result: TrailResult) -> dict[int, TrailNode]:
    lookup = {node.logical_index: node for node in result.nodes}
    if len(lookup) != len(result.nodes):
        raise ValueError("Trail result repeats a logical node identity")
    return lookup


def _decode_ascii_hashes(value: torch.Tensor, *, label: str) -> tuple[str, ...]:
    if value.ndim != 2 or value.shape[1] != 64 or value.dtype != torch.uint8:
        raise ValueError(f"Final shard {label} table has an invalid shape or dtype")
    result = []
    for row in value:
        try:
            text = bytes(row.tolist()).decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError(f"Final shard {label} is not ASCII") from error
        if len(text) != 64 or any(
            character not in "0123456789abcdef" for character in text
        ):
            raise ValueError(f"Final shard {label} is not a lowercase SHA256")
        result.append(text)
    return tuple(result)


def _validate_offsets(
    value: torch.Tensor,
    *,
    expected_length: int,
    final_value: int,
    label: str,
) -> None:
    if value.dtype != torch.int64 or value.shape != (expected_length,):
        raise ValueError(f"Final shard {label} offsets changed shape or dtype")
    if int(value[0]) != 0 or int(value[-1]) != final_value:
        raise ValueError(f"Final shard {label} offsets changed coverage")
    if torch.any(value[1:] < value[:-1]):
        raise ValueError(f"Final shard {label} offsets are not monotonic")


def validate_final_shard(
    path: str | Path,
    *,
    expected_config_sha256: str | None = None,
    expected_policy_sha256: str | None = None,
) -> dict[str, Any]:
    tensors, metadata = load_safetensors_bundle(path, expected_format=FINAL_FORMAT)
    expected_metadata = {
        "config_sha256": expected_config_sha256,
        "production_policy_sha256": expected_policy_sha256,
        "vector_schema": VECTOR_SCHEMA,
        "trail_schema": TRAIL_SCHEMA,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_metadata.items()
        if expected is not None and metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Final shard metadata changed: {mismatches}")
    trail_materialization = metadata.get(
        "trail_materialization", TRAIL_MATERIALIZATION_COMPUTED
    )
    if trail_materialization not in TRAIL_MATERIALIZATION_VALUES:
        raise ValueError(
            "Final shard has an unsupported trail materialization mode: "
            f"{trail_materialization!r}"
        )
    event_keys = _decode_ascii_hashes(tensors["event_key_ascii"], label="event key")
    artifact_ids = _decode_ascii_hashes(
        tensors["artifact_id_ascii"], label="artifact ID"
    )
    event_count = len(event_keys)
    if len(artifact_ids) != event_count or len(set(event_keys)) != event_count:
        raise ValueError("Final shard event identities are duplicated or misaligned")
    if metadata.get("events") != event_count:
        raise ValueError("Final shard metadata event count changed")
    vectors = tensors["vectors"]
    if vectors.shape != (event_count, VECTOR_DIM) or vectors.dtype != torch.float32:
        raise ValueError("Final shard vector table changed shape or dtype")
    if not torch.isfinite(vectors).all():
        raise ValueError("Final shard vectors contain non-finite values")

    target_offsets = tensors["event_target_offsets"]
    target_total = int(tensors["target_token_ids"].numel())
    _validate_offsets(
        target_offsets,
        expected_length=event_count + 1,
        final_value=target_total,
        label="target",
    )
    target_fields = (
        "target_token_ids",
        "target_positions",
        "target_probabilities",
        "target_entropies",
        "target_log_probabilities",
    )
    if any(int(tensors[key].numel()) != target_total for key in target_fields):
        raise ValueError("Final shard target arrays are misaligned")
    target_count = tensors["target_count"].long()
    condition_length = tensors["condition_length"].long()
    if target_count.shape != (event_count,) or condition_length.shape != (event_count,):
        raise ValueError("Final shard event boundary arrays changed shape")
    if not torch.equal(target_count, torch.diff(target_offsets)):
        raise ValueError("Final shard target counts do not match offsets")
    if torch.any((target_count <= 0) | (target_count > 929)) or torch.any(
        condition_length <= 0
    ):
        raise ValueError("Final shard event target or condition lengths are invalid")
    if torch.any(
        (tensors["target_token_ids"] < 0) | (tensors["target_token_ids"] > 10)
    ):
        raise ValueError("Final shard contains a non-grid target token")
    target_float_fields = (
        "target_probabilities",
        "target_entropies",
        "target_log_probabilities",
    )
    if any(not torch.isfinite(tensors[key]).all() for key in target_float_fields):
        raise ValueError("Final shard target diagnostics contain non-finite values")
    for event_index in range(event_count):
        start, end = target_offsets[event_index : event_index + 2].tolist()
        expected_positions = torch.arange(
            int(condition_length[event_index]) - 1,
            int(condition_length[event_index]) + int(target_count[event_index]) - 1,
            dtype=torch.int32,
        )
        if not torch.equal(tensors["target_positions"][start:end], expected_positions):
            raise ValueError("Final shard target prediction positions changed")

    trail_count = int(tensors["trail_total_cost"].numel())
    node_count = int(tensors["trail_node_logical_index"].numel())
    trail_node_count = int(tensors["trail_node_local_index"].numel())
    edge_count = int(tensors["trail_edge_signed_direct_effect"].numel())
    _validate_offsets(
        tensors["event_trail_offsets"],
        expected_length=event_count + 1,
        final_value=trail_count,
        label="event-trail",
    )
    _validate_offsets(
        tensors["event_node_offsets"],
        expected_length=event_count + 1,
        final_value=node_count,
        label="event-node",
    )
    _validate_offsets(
        tensors["trail_node_offsets"],
        expected_length=trail_count + 1,
        final_value=trail_node_count,
        label="trail-node",
    )
    _validate_offsets(
        tensors["trail_edge_offsets"],
        expected_length=trail_count + 1,
        final_value=edge_count,
        label="trail-edge",
    )
    node_fields = (
        "trail_node_type",
        "trail_node_layer",
        "trail_node_position",
        "trail_node_identity",
        "trail_node_activation",
        "trail_node_influence",
    )
    if any(int(tensors[key].numel()) != node_count for key in node_fields):
        raise ValueError("Final shard trail node table is misaligned")
    edge_fields = (
        "trail_edge_nonnegative_influence",
        "trail_edge_normalized_strength",
        "trail_edge_cost",
    )
    if any(int(tensors[key].numel()) != edge_count for key in edge_fields):
        raise ValueError("Final shard trail edge table is misaligned")
    if tensors["trail_score"].numel() != trail_count:
        raise ValueError("Final shard trail score table is misaligned")
    event_no_trail = tensors["event_no_valid_trail"].bool()
    event_trail_counts = torch.diff(tensors["event_trail_offsets"])
    if event_no_trail.shape != (event_count,):
        raise ValueError("Final shard no-valid-trail flags changed shape")
    if trail_materialization == TRAIL_MATERIALIZATION_COMPUTED:
        if not torch.equal(event_no_trail, event_trail_counts == 0):
            raise ValueError("Final shard no-valid-trail flags changed semantics")
    elif (
        trail_count != 0
        or node_count != 0
        or trail_node_count != 0
        or edge_count != 0
        or torch.any(event_no_trail)
    ):
        raise ValueError("Skipped trail materialization must retain no trail data")
    if torch.any(event_trail_counts > 8):
        raise ValueError("Final shard retains more than eight trails for an event")
    for trail_index in range(trail_count):
        node_start, node_end = tensors["trail_node_offsets"][
            trail_index : trail_index + 2
        ].tolist()
        edge_start, edge_end = tensors["trail_edge_offsets"][
            trail_index : trail_index + 2
        ].tolist()
        if node_end - node_start != edge_end - edge_start + 1:
            raise ValueError("Final shard trail node/edge cardinality changed")
    for event_index in range(event_count):
        event_node_count = int(
            tensors["event_node_offsets"][event_index + 1]
            - tensors["event_node_offsets"][event_index]
        )
        trail_start, trail_end = tensors["event_trail_offsets"][
            event_index : event_index + 2
        ].tolist()
        node_start = int(tensors["trail_node_offsets"][trail_start])
        node_end = int(tensors["trail_node_offsets"][trail_end])
        local = tensors["trail_node_local_index"][node_start:node_end]
        if local.numel() and (
            int(local.min()) < 0 or int(local.max()) >= event_node_count
        ):
            raise ValueError("Final shard trail references an invalid event-local node")
    floating = (
        "trail_node_activation",
        "trail_node_influence",
        "trail_edge_signed_direct_effect",
        "trail_edge_nonnegative_influence",
        "trail_edge_normalized_strength",
        "trail_edge_cost",
        "trail_total_cost",
        "trail_score",
    )
    if any(not torch.isfinite(tensors[key]).all() for key in floating):
        raise ValueError("Final shard trail data contain non-finite values")
    return {
        "format": FINAL_FORMAT,
        "events": event_count,
        "event_keys": event_keys,
        "artifact_ids": artifact_ids,
        "targets": target_total,
        "trails": trail_count,
        "nodes": node_count,
        "vector_schema": VECTOR_SCHEMA,
        "trail_schema": TRAIL_SCHEMA,
        "trail_materialization": trail_materialization,
        "metadata": metadata,
    }


def materialize_final_shard(
    path: str | Path,
    payloads: Sequence[CompactPayload],
    *,
    config_sha256: str,
    production_policy_sha256: str,
    source_payload_sha256: Mapping[str, str],
    max_trails: int = 8,
    epsilon: float = 1e-12,
    policy_eligibility: str | None = None,
    include_trails: bool = True,
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("Cannot materialize an empty final shard")
    vectors: list[VectorResult] = []
    trail_results: list[TrailResult | None] = []
    event_trail_offsets = [0]
    event_node_offsets = [0]
    trail_node_offsets = [0]
    trail_edge_offsets = [0]
    node_logical: list[int] = []
    node_type: list[int] = []
    node_layer: list[int] = []
    node_position: list[int] = []
    node_identity: list[int] = []
    node_activation: list[float] = []
    node_influence: list[float] = []
    trail_node_local: list[int] = []
    edge_signed: list[float] = []
    edge_influence: list[float] = []
    edge_strength: list[float] = []
    edge_cost: list[float] = []
    trail_total_cost: list[float] = []
    trail_score: list[float] = []
    event_target_offsets = [0]
    target_token_ids: list[int] = []
    target_positions: list[int] = []
    target_probabilities: list[float] = []
    target_entropies: list[float] = []
    target_log_probabilities: list[float] = []
    for payload in payloads:
        payload.validate()
        vector = vectorize_payload(payload, epsilon=epsilon)
        trails = (
            extract_top_k_trails(payload, max_trails=max_trails, epsilon=epsilon)
            if include_trails
            else None
        )
        if (vector.event_key, vector.artifact_id) != (
            payload.event_key,
            payload.artifact_id,
        ):
            raise ValueError("CPU materialization changed an event identity")
        if trails is not None and (trails.event_key, trails.artifact_id) != (
            payload.event_key,
            payload.artifact_id,
        ):
            raise ValueError("CPU materialization changed an event identity")
        vectors.append(vector)
        trail_results.append(trails)
        target_token_ids.extend(payload.target_token_ids.tolist())
        target_positions.extend(payload.target_positions.tolist())
        target_probabilities.extend(payload.target_probabilities.tolist())
        target_entropies.extend(payload.target_entropies.tolist())
        target_log_probabilities.extend(payload.target_log_probabilities.tolist())
        event_target_offsets.append(len(target_token_ids))
        if trails is None:
            event_node_offsets.append(len(node_logical))
            event_trail_offsets.append(len(trail_total_cost))
            continue
        lookup = _node_lookup(trails)
        local_node_index = {
            node.logical_index: index for index, node in enumerate(trails.nodes)
        }
        for node in trails.nodes:
            node_logical.append(node.logical_index)
            node_type.append(node.node_type)
            node_layer.append(node.layer)
            node_position.append(node.position)
            node_identity.append(node.feature_or_token_id)
            node_activation.append(node.activation)
            node_influence.append(node.influence)
        event_node_offsets.append(len(node_logical))
        for trail in trails.trails:
            if len(trail.node_indices) != len(trail.signed_direct_effects) + 1:
                raise ValueError("Trail edge and node cardinalities do not align")
            for logical_index in trail.node_indices:
                if logical_index not in lookup:
                    raise ValueError(
                        "Trail references a node outside its event node table"
                    )
                trail_node_local.append(local_node_index[logical_index])
            edge_signed.extend(trail.signed_direct_effects)
            edge_influence.extend(trail.nonnegative_influences)
            edge_strength.extend(trail.normalized_strengths)
            edge_cost.extend(trail.edge_costs)
            trail_total_cost.append(trail.total_cost)
            trail_score.append(trail.score)
            trail_node_offsets.append(len(trail_node_local))
            trail_edge_offsets.append(len(edge_signed))
        event_trail_offsets.append(len(trail_total_cost))
    event_keys = [payload.event_key for payload in payloads]
    artifact_ids = [payload.artifact_id for payload in payloads]
    tensors = {
        "event_key_ascii": _ascii_hashes(event_keys),
        "artifact_id_ascii": _ascii_hashes(artifact_ids),
        "vectors": torch.from_numpy(np.stack([result.values for result in vectors])).to(
            torch.float32
        ),
        "target_count": torch.tensor(
            [payload.target_count for payload in payloads], dtype=torch.int16
        ),
        "condition_length": torch.tensor(
            [payload.condition_length for payload in payloads], dtype=torch.int32
        ),
        "event_target_offsets": torch.tensor(event_target_offsets, dtype=torch.int64),
        "target_token_ids": torch.tensor(target_token_ids, dtype=torch.int16),
        "target_positions": torch.tensor(target_positions, dtype=torch.int32),
        "target_probabilities": torch.tensor(target_probabilities, dtype=torch.float32),
        "target_entropies": torch.tensor(target_entropies, dtype=torch.float32),
        "target_log_probabilities": torch.tensor(
            target_log_probabilities, dtype=torch.float32
        ),
        "event_no_valid_trail": torch.tensor(
            [
                result.no_valid_trail if result is not None else False
                for result in trail_results
            ],
            dtype=torch.uint8,
        ),
        "event_trail_offsets": torch.tensor(event_trail_offsets, dtype=torch.int64),
        "event_node_offsets": torch.tensor(event_node_offsets, dtype=torch.int64),
        "trail_node_offsets": torch.tensor(trail_node_offsets, dtype=torch.int64),
        "trail_edge_offsets": torch.tensor(trail_edge_offsets, dtype=torch.int64),
        "trail_node_logical_index": torch.tensor(node_logical, dtype=torch.int32),
        "trail_node_type": torch.tensor(node_type, dtype=torch.uint8),
        "trail_node_layer": torch.tensor(node_layer, dtype=torch.int16),
        "trail_node_position": torch.tensor(node_position, dtype=torch.int32),
        "trail_node_identity": torch.tensor(node_identity, dtype=torch.int32),
        "trail_node_local_index": torch.tensor(trail_node_local, dtype=torch.int32),
        "trail_node_activation": torch.tensor(node_activation, dtype=torch.float32),
        "trail_node_influence": torch.tensor(node_influence, dtype=torch.float32),
        "trail_edge_signed_direct_effect": torch.tensor(
            edge_signed, dtype=torch.float32
        ),
        "trail_edge_nonnegative_influence": torch.tensor(
            edge_influence, dtype=torch.float32
        ),
        "trail_edge_normalized_strength": torch.tensor(
            edge_strength, dtype=torch.float32
        ),
        "trail_edge_cost": torch.tensor(edge_cost, dtype=torch.float32),
        "trail_total_cost": torch.tensor(trail_total_cost, dtype=torch.float32),
        "trail_score": torch.tensor(trail_score, dtype=torch.float32),
    }
    metadata_record = {
        "format": FINAL_FORMAT,
        "config_sha256": config_sha256,
        "production_policy_sha256": production_policy_sha256,
        "vector_schema": vectors[0].schema,
        "trail_schema": TRAIL_SCHEMA,
        "trail_materialization": (
            TRAIL_MATERIALIZATION_COMPUTED
            if include_trails
            else TRAIL_MATERIALIZATION_SKIPPED
        ),
        "events": len(payloads),
        "trails": len(trail_total_cost),
        "no_valid_trail_events": sum(
            int(result.no_valid_trail)
            for result in trail_results
            if result is not None
        ),
        "source_payload_sha256": dict(source_payload_sha256),
    }
    if policy_eligibility is not None:
        metadata_record["policy_eligibility"] = policy_eligibility
    metadata = write_safetensors_bundle(
        path,
        tensors=tensors,
        metadata=metadata_record,
    )
    validation = validate_final_shard(
        path,
        expected_config_sha256=config_sha256,
        expected_policy_sha256=production_policy_sha256,
    )
    if validation["events"] != len(payloads):
        raise ValueError("Published final shard changed event coverage")
    return metadata

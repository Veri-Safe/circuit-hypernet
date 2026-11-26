from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from s5_lean_circuit.circuit.descriptor_v2_397 import (
    BLOCK_DIMS,
    DESCRIPTOR_397_SCHEMA_SHA256,
    DESCRIPTOR_DIM,
    FLOW_ALLOWED_PAIRS,
    FLOW_CATEGORIES,
    MOTIF2_SEQUENCES,
    MOTIF3_SEQUENCES,
    SCHEMA_397,
    _third_indices,
    _topology_arrays,
    fixed_log1p_397,
    vectorize_payload_v2_397,
    vectorize_payload_v2_397_record,
)
from s5_lean_circuit.circuit.payload_v2 import (
    S5GraphPayloadV2,
    payload_artifact_id,
    read_payload_v2_shard,
    semantic_sha256,
    write_payload_v2_shard,
)


def _identity() -> dict[str, object]:
    return {
        "payload_schema_version": 2,
        "scope": "descriptor397-fixture",
        "n_layers": 8,
        "d_feature": 16_384,
        "model_identity": {"checkpoint_sha256": "a" * 64},
        "tokenizer_sha256": "b" * 64,
        "transcoder_bank_sha256": "c" * 64,
        "trace_corpus_identity_sha256": "d" * 64,
        "operator": "joint_signed_full_output",
        "feature_budget": 512,
        "node_threshold": 0.9,
        "edge_threshold": 0.99,
        "row_batch_size": 64,
        "backend": "streaming_cached_exact",
        "extraction_code_sha256": "e" * 64,
    }


def _payload(*, suffix: str = "0") -> S5GraphPayloadV2:
    archive_sha = semantic_sha256(_identity())
    event_key = suffix.rjust(64, "0")
    payload = S5GraphPayloadV2(
        event_key=event_key,
        payload_artifact_id=payload_artifact_id(
            event_key=event_key,
            archive_identity_sha256=archive_sha,
        ),
        condition_length=2,
        feature_layer=torch.tensor([0, 1], dtype=torch.int16),
        feature_position=torch.tensor([0, 1], dtype=torch.int16),
        feature_id=torch.tensor([7, 9], dtype=torch.int32),
        feature_activation=torch.tensor([1.0, 3.0], dtype=torch.float32),
        feature_influence=torch.tensor([0.3, 0.7], dtype=torch.float32),
        error_layer=torch.tensor([0], dtype=torch.int16),
        error_position=torch.tensor([1], dtype=torch.int16),
        error_influence=torch.tensor([0.2], dtype=torch.float32),
        token_influence=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32),
        root_influence=torch.tensor(1.0, dtype=torch.float32),
        edge_source=torch.tensor([3, 0, 2, 1, 5], dtype=torch.int32),
        edge_target=torch.tensor([0, 1, 1, 6, 6], dtype=torch.int32),
        edge_signed_direct_effect=torch.tensor(
            [0.1, 0.2, -0.3, 0.4, 0.5], dtype=torch.float32
        ),
        edge_nonnegative_influence=torch.tensor(
            [0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32
        ),
    )
    payload.validate(
        n_layers=8,
        d_feature=16_384,
        archive_identity_sha256=archive_sha,
    )
    return payload


def _empty_graph_payload(*, suffix: str = "f") -> S5GraphPayloadV2:
    source = _payload(suffix=suffix)
    payload = replace(
        source,
        condition_length=2,
        feature_layer=torch.empty(0, dtype=torch.int16),
        feature_position=torch.empty(0, dtype=torch.int16),
        feature_id=torch.empty(0, dtype=torch.int32),
        feature_activation=torch.empty(0, dtype=torch.float32),
        feature_influence=torch.empty(0, dtype=torch.float32),
        error_layer=torch.empty(0, dtype=torch.int16),
        error_position=torch.empty(0, dtype=torch.int16),
        error_influence=torch.empty(0, dtype=torch.float32),
        token_influence=torch.tensor([0.2, 0.4], dtype=torch.float32),
        edge_source=torch.empty(0, dtype=torch.int32),
        edge_target=torch.empty(0, dtype=torch.int32),
        edge_signed_direct_effect=torch.empty(0, dtype=torch.float32),
        edge_nonnegative_influence=torch.empty(0, dtype=torch.float32),
    )
    payload.validate(n_layers=8, d_feature=16_384)
    return payload


def test_descriptor397_schema_is_named_fixed_and_complete() -> None:
    assert DESCRIPTOR_DIM == 397
    assert BLOCK_DIMS == {
        "global": 30,
        "layer_nodes": 112,
        "layer_flow": 162,
        "topology": 51,
        "position": 42,
    }
    assert SCHEMA_397.block_offsets == {
        "global": (0, 30),
        "layer_nodes": (30, 142),
        "layer_flow": (142, 304),
        "topology": (304, 355),
        "position": (355, 397),
    }
    assert len(SCHEMA_397.coordinate_names) == 397
    assert len(set(SCHEMA_397.coordinate_names)) == 397
    assert len(SCHEMA_397.sha256) == 64
    assert SCHEMA_397.sha256 == DESCRIPTOR_397_SCHEMA_SHA256


def test_vectorize_payload_v2_397_is_deterministic_finite_and_contiguous() -> None:
    payload = _payload()
    first = vectorize_payload_v2_397(payload, d_feature=16_384)
    second = vectorize_payload_v2_397(payload, d_feature=16_384)
    assert first.shape == (397,)
    assert first.dtype == torch.float32
    assert first.device.type == "cpu"
    assert first.is_contiguous()
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_global_and_layer_node_coordinates_follow_the_frozen_rules() -> None:
    record = vectorize_payload_v2_397_record(_payload(), d_feature=16_384)
    global_block = record.blocks["global"]
    assert global_block[0].item() == 2
    assert global_block[1].item() == pytest.approx(4.0)
    assert global_block[2].item() == pytest.approx(2.0)
    assert global_block[3].item() == pytest.approx(2.8)
    assert global_block[4].item() == pytest.approx(3.0)
    assert global_block[19].item() == 5
    assert global_block[20].item() == pytest.approx(1.5)
    assert global_block[24].item() == pytest.approx(1.2)
    assert global_block[25].item() == pytest.approx(0.3)
    assert global_block[26].item() == pytest.approx(0.8)
    assert global_block[29].item() == pytest.approx(1.0)

    layer = record.blocks["layer_nodes"].reshape(8, 14)
    assert layer[0, 0].item() == pytest.approx(0.5)
    assert layer[0, 1].item() == pytest.approx(0.25)
    assert layer[0, 2].item() == pytest.approx(0.3)
    assert layer[0, 9].item() == pytest.approx(1.0)
    assert layer[1, 0].item() == pytest.approx(0.5)
    assert layer[1, 1].item() == pytest.approx(0.75)
    assert torch.count_nonzero(layer[2:]) == 0


def test_layer_flow_preserves_route_importance_and_signed_balance() -> None:
    record = vectorize_payload_v2_397_record(_payload(), d_feature=16_384)
    flow = record.blocks["layer_flow"].reshape(81, 2)
    indices = {pair: index for index, pair in enumerate(FLOW_ALLOWED_PAIRS)}
    token_to_l0 = indices[(0, 2)]
    feature_l0_to_l1 = indices[(2, 3)]
    error_l0_to_l1 = indices[(10, 3)]
    feature_l1_to_root = indices[(3, 1)]
    token_to_root = indices[(0, 1)]
    assert flow[token_to_l0].tolist() == pytest.approx([0.1 / 1.5, 1.0])
    assert flow[feature_l0_to_l1].tolist() == pytest.approx([0.2 / 1.5, 1.0])
    assert flow[error_l0_to_l1].tolist() == pytest.approx([0.3 / 1.5, -1.0])
    assert flow[feature_l1_to_root].tolist() == pytest.approx([0.4 / 1.5, 1.0])
    assert flow[token_to_root].tolist() == pytest.approx([0.5 / 1.5, 1.0])
    assert flow[:, 0].sum().item() == pytest.approx(1.0)
    assert FLOW_CATEGORIES[10] == "error_l0"


def test_topology_and_relative_position_have_expected_semantics() -> None:
    record = vectorize_payload_v2_397_record(_payload(), d_feature=16_384)
    topology = record.blocks["topology"]
    # Token distances: one direct root edge, one length-three path, one isolated.
    assert topology[0:6].tolist() == pytest.approx(
        [1 / 3, 0.0, 1 / 3, 0.0, 0.0, 1 / 3]
    )
    # Feature distances: one at distance one and one at distance two.
    assert topology[6:12].tolist() == pytest.approx([0.5, 0.5, 0, 0, 0, 0])
    # Error distance is two.
    assert topology[12:18].tolist() == pytest.approx([0, 1, 0, 0, 0, 0])
    assert topology[30].item() == pytest.approx(2.0)  # root indegree
    assert topology[37].item() == pytest.approx(2.0)  # weak components
    assert topology[38].item() == pytest.approx(3.0)  # longest root path

    motif2_index = {value: index for index, value in enumerate(MOTIF2_SEQUENCES)}
    motif3_index = {value: index for index, value in enumerate(MOTIF3_SEQUENCES)}
    motif2 = topology[39:45]
    motif3 = topology[45:51]
    assert motif2[motif2_index[(0, 1, 1)]].item() == pytest.approx(1 / 3)
    assert motif2[motif2_index[(1, 1, 3)]].item() == pytest.approx(1 / 3)
    assert motif2[motif2_index[(2, 1, 3)]].item() == pytest.approx(1 / 3)
    assert motif3[motif3_index[(0, 1, 1, 3)]].item() == pytest.approx(1.0)

    position = record.blocks["position"].reshape(6, 7)
    # Prompt positions 0 and 1 occupy early/middle thirds; position 2 is
    # generated-prefix early. Each populated region normalizes independently.
    assert position[0, 0].item() == pytest.approx(0.5)
    assert position[1, 0].item() == pytest.approx(0.5)
    assert position[2, 0].item() == pytest.approx(0.0)
    assert position[3, 0].item() == pytest.approx(1.0)
    assert torch.count_nonzero(position[4:]) == 0


def test_empty_feature_and_edge_groups_are_valid_zero_summaries() -> None:
    raw = vectorize_payload_v2_397(
        _empty_graph_payload(),
        d_feature=16_384,
    )
    assert raw.shape == (397,)
    assert torch.isfinite(raw).all()
    assert raw[0].item() == 0.0
    assert raw[19].item() == 0.0
    flow_start, flow_end = SCHEMA_397.block_offsets["layer_flow"]
    assert torch.count_nonzero(raw[flow_start:flow_end]) == 0


def test_fixed_log1p_changes_only_declared_coordinates() -> None:
    raw = vectorize_payload_v2_397(_payload(), d_feature=16_384)
    treated = fixed_log1p_397(raw)
    mask = SCHEMA_397.log1p_mask
    assert torch.equal(treated[~mask], raw[~mask])
    assert torch.allclose(treated[mask], torch.log1p(raw[mask]))
    assert treated.shape == raw.shape


def test_physical_shard_grouping_does_not_change_vector_bytes(tmp_path: Path) -> None:
    payloads = [_payload(suffix="1"), _payload(suffix="2")]
    together = tmp_path / "together.safetensors"
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    write_payload_v2_shard(together, payloads, archive_identity=_identity())
    write_payload_v2_shard(first, payloads[:1], archive_identity=_identity())
    write_payload_v2_shard(second, payloads[1:], archive_identity=_identity())
    together_payloads = read_payload_v2_shard(together)
    split_payloads = (*read_payload_v2_shard(first), *read_payload_v2_shard(second))
    together_vectors = [
        vectorize_payload_v2_397(payload, d_feature=16_384)
        for payload in together_payloads
    ]
    split_vectors = [
        vectorize_payload_v2_397(payload, d_feature=16_384)
        for payload in split_payloads
    ]
    assert all(
        torch.equal(left, right)
        for left, right in zip(together_vectors, split_vectors, strict=True)
    )


def test_topology_distinguishes_shortest_and_longest_root_paths() -> None:
    # Node 0 reaches root 3 directly and through 0->1->2->3.
    source = torch.tensor([0, 0, 1, 2]).numpy()
    target = torch.tensor([1, 3, 2, 3]).numpy()
    _indegree, _outdegree, shortest, _order, _components, longest = (
        _topology_arrays(4, source, target, root_index=3)
    )
    assert shortest[0] == pytest.approx(1.0)
    assert longest == pytest.approx(3.0)


def test_relative_thirds_freeze_nonuniform_length_four_and_five_bins() -> None:
    assert _third_indices(
        torch.arange(4).numpy(), start=0, length=4
    ).tolist() == [0, 0, 1, 2]
    assert _third_indices(
        torch.arange(5).numpy(), start=0, length=5
    ).tolist() == [0, 0, 1, 1, 2]

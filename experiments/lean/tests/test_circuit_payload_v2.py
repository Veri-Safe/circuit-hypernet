from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

from s5_lean_circuit.circuit.payload_v2 import (
    ByteBoundedPayloadV2Writer,
    S5GraphPayloadV2,
    payload_artifact_id,
    payload_v2_from_v1,
    payload_v2_equal,
    read_payload_v2_shard,
    semantic_sha256,
    verify_payload_v2_shard_record,
    write_payload_v2_shard,
)


def _identity() -> dict[str, object]:
    return {
        "payload_schema_version": 2,
        "scope": "formal_rl_union_phase1",
        "n_layers": 8,
        "d_feature": 16_384,
        "model_identity": {"checkpoint_sha256": "a" * 64},
        "tokenizer_sha256": "b" * 64,
        "transcoder_bank_sha256": "c" * 64,
        "trace_corpus_identity_sha256": "d" * 64,
        "operator": "joint_signed_full_output",
        "feature_budget": 2_048,
        "node_threshold": 0.8,
        "edge_threshold": 0.98,
        "backend": "streaming_cached_exact",
        "extraction_code_sha256": "e" * 64,
    }


def _payload(*, suffix: str = "0") -> S5GraphPayloadV2:
    identity = _identity()
    archive_sha = semantic_sha256(identity)
    event_key = suffix.rjust(64, "0")
    payload = S5GraphPayloadV2(
        event_key=event_key,
        payload_artifact_id=payload_artifact_id(
            event_key=event_key, archive_identity_sha256=archive_sha
        ),
        condition_length=2,
        feature_layer=torch.tensor([0, 1], dtype=torch.int16),
        feature_position=torch.tensor([0, 1], dtype=torch.int16),
        feature_id=torch.tensor([7, 9], dtype=torch.int32),
        feature_activation=torch.tensor([1.0, 2.0], dtype=torch.float32),
        feature_influence=torch.tensor([0.3, 0.4], dtype=torch.float32),
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


def test_payload_v2_is_condition_safe_and_valid() -> None:
    payload = _payload()
    assert not hasattr(payload, "replay_token_ids")
    assert not hasattr(payload, "target_token_ids")
    assert not hasattr(payload, "feature_cumulative_score")
    assert payload.node_count == 7
    assert payload.edge_count == 5


def test_v1_conversion_drops_diagnostics_without_changing_sparse_graph() -> None:
    expected = _payload()
    source = SimpleNamespace(
        event_key=expected.event_key,
        condition_length=expected.condition_length,
        feature_layer=expected.feature_layer,
        feature_position=expected.feature_position,
        feature_id=expected.feature_id,
        feature_activation=expected.feature_activation,
        feature_influence=expected.feature_influence,
        error_layer=expected.error_layer,
        error_position=expected.error_position,
        error_influence=expected.error_influence,
        token_influence=expected.token_influence,
        root_influence=float(expected.root_influence),
        edge_source=expected.edge_source,
        edge_target=expected.edge_target,
        edge_signed_direct_effect=expected.edge_signed_direct_effect,
        edge_nonnegative_influence=expected.edge_nonnegative_influence,
        replay_token_ids=torch.tensor([2, 4, 5], dtype=torch.int16),
        target_token_ids=torch.tensor([6], dtype=torch.int16),
        feature_cumulative_score=torch.tensor([0.2, 0.5]),
    )
    converted = payload_v2_from_v1(
        source,
        archive_identity_sha256=semantic_sha256(_identity()),
        n_layers=8,
        d_feature=16_384,
    )
    assert payload_v2_equal(converted, expected)
    assert not hasattr(converted, "replay_token_ids")


def test_payload_v2_rejects_direction_and_numeric_errors() -> None:
    with pytest.raises(ValueError, match="later layer"):
        replace(
            _payload(),
            feature_layer=torch.tensor([1, 0], dtype=torch.int16),
        ).validate(n_layers=8, d_feature=16_384)
    with pytest.raises(ValueError, match="positive"):
        replace(
            _payload(),
            edge_nonnegative_influence=torch.tensor(
                [0.1, 0.2, 0.0, 0.4, 0.5], dtype=torch.float32
            ),
        ).validate(n_layers=8, d_feature=16_384)
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            _payload(),
            feature_id=torch.tensor([7, 7], dtype=torch.int32),
            feature_position=torch.tensor([0, 0], dtype=torch.int16),
            feature_layer=torch.tensor([0, 0], dtype=torch.int16),
        ).validate(n_layers=8, d_feature=16_384)


def test_payload_v2_shard_round_trip_and_schema(tmp_path: Path) -> None:
    identity = _identity()
    path = tmp_path / "payload.safetensors"
    record = write_payload_v2_shard(
        path,
        [_payload(suffix="1"), _payload(suffix="2")],
        archive_identity=identity,
    )
    loaded = read_payload_v2_shard(path)
    assert [item.event_key for item in loaded] == ["1".rjust(64, "0"), "2".rjust(64, "0")]
    verified = verify_payload_v2_shard_record(
        record, expected_archive_identity_sha256=semantic_sha256(identity)
    )
    assert len(verified) == len(loaded)
    assert all(
        payload_v2_equal(first, second)
        for first, second in zip(verified, loaded, strict=True)
    )
    tensors = load_file(str(path), device="cpu")
    assert "replay_token_ids" not in tensors
    assert "target_token_ids" not in tensors
    assert "feature_cumulative_score" not in tensors
    sidecar = json.loads(Path(record["sidecar"]).read_text(encoding="utf-8"))
    assert sidecar["format"] == "s5_compact_circuit_graph_payload_v2"
    assert sidecar["records"][0]["node_count"] == 7


def test_byte_bounded_writer_resumes_verified_prefix(tmp_path: Path) -> None:
    identity = _identity()
    event_bytes = _payload().approximate_bytes()
    writer = ByteBoundedPayloadV2Writer(
        tmp_path,
        max_shard_bytes=event_bytes + 1,
        archive_identity=identity,
    )
    writer.append(_payload(suffix="1"))
    writer.append(_payload(suffix="2"))
    manifest = writer.close()
    assert manifest["status"] == "complete"
    assert manifest["total_events"] == 2
    assert len(manifest["shards"]) == 2
    resumed = ByteBoundedPayloadV2Writer(
        tmp_path,
        max_shard_bytes=event_bytes + 1,
        archive_identity=identity,
    )
    assert resumed.closed is True
    with pytest.raises(RuntimeError, match="completed"):
        resumed.append(_payload(suffix="3"))

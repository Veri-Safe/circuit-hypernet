"""Permanent condition-safe CompactPayload storage for S5 Circuit Graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .attribution import S5DecomposedForward, S5LinearizedForward


PAYLOAD_FORMAT = "s5_gpt8_joint_full_output_compact_v1"
GRAPH_POLICY_KEYS = {
    "operator",
    "feature_budget",
    "node_threshold",
    "edge_threshold",
    "queue_size",
    "row_batch_size",
    "backend",
}


class NodeType(IntEnum):
    FEATURE = 0
    ERROR = 1
    PROMPT_TOKEN = 2
    GENERATED_PREFIX_TOKEN = 3
    FULL_OUTPUT_ROOT = 4


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circuit_artifact_id(
    *,
    event_key: str,
    model_identity: Mapping[str, Any],
    tokenizer_sha256: str,
    transcoder_bank_sha256: str,
    graph_policy: Mapping[str, Any],
    code_identity: str,
) -> str:
    return semantic_sha256(
        {
            "format": PAYLOAD_FORMAT,
            "event_key": event_key,
            "model_identity": dict(model_identity),
            "tokenizer_sha256": tokenizer_sha256,
            "transcoder_bank_sha256": transcoder_bank_sha256,
            "graph_policy": dict(graph_policy),
            "code_identity": code_identity,
        }
    )


def validate_graph_policy(value: Mapping[str, Any]) -> None:
    if set(value) != GRAPH_POLICY_KEYS:
        raise ValueError("Graph policy must use the exact frozen schema")
    if value["operator"] != "joint_signed_full_output":
        raise ValueError("Graph policy operator changed")
    if value["backend"] != "streaming_cached_exact":
        raise ValueError("Graph policy backend changed")
    if int(value["feature_budget"]) <= 0:
        raise ValueError("Graph feature budget must be positive")
    if not 0 < float(value["node_threshold"]) <= 1 or not 0 < float(
        value["edge_threshold"]
    ) <= 1:
        raise ValueError("Graph cumulative threshold is outside (0, 1]")
    if int(value["queue_size"]) <= 0 or int(value["row_batch_size"]) <= 0:
        raise ValueError("Graph materialization batch sizes must be positive")


def graph_setting_label(value: Mapping[str, Any]) -> str:
    validate_graph_policy(value)
    node = str(float(value["node_threshold"])).replace(".", "p")
    edge = str(float(value["edge_threshold"])).replace(".", "p")
    return f"b{int(value['feature_budget'])}-n{node}-e{edge}"


def _is_dag(sources: Tensor, targets: Tensor, node_count: int) -> bool:
    if sources.numel() == 0:
        return True
    source_np = sources.detach().cpu().numpy().astype(np.int64, copy=False)
    target_np = targets.detach().cpu().numpy().astype(np.int64, copy=False)
    order = np.argsort(source_np, kind="quicksort")
    sorted_targets = target_np[order]
    counts = np.bincount(source_np, minlength=node_count)
    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.int64, out=offsets[1:])
    indegree = np.bincount(target_np, minlength=node_count).astype(np.int64)
    queue = np.flatnonzero(indegree == 0).tolist()
    cursor = 0
    processed = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        processed += 1
        neighbors = sorted_targets[int(offsets[node]) : int(offsets[node + 1])]
        if neighbors.size:
            updated = indegree[neighbors] - 1
            indegree[neighbors] = updated
            queue.extend(neighbors[updated == 0].tolist())
    return processed == node_count


@dataclass(frozen=True)
class S5CompactPayload:
    event_key: str
    artifact_id: str
    condition_length: int
    total_active_features: int
    n_layers: int
    d_feature: int
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

    def validate(self) -> None:
        if len(self.event_key) != 64 or len(self.artifact_id) != 64:
            raise ValueError("payload identities must be SHA256")
        if self.condition_length <= 0 or self.n_layers <= 0:
            raise ValueError("payload condition or layer count is invalid")
        if self.d_feature <= 0:
            raise ValueError("payload Transcoder width must be positive")
        groups = (
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
        if any(len({int(item.numel()) for item in group}) != 1 for group in groups):
            raise ValueError("payload aligned arrays have different lengths")
        if self.total_active_features < self.feature_count:
            raise ValueError("retained feature count exceeds the active feature count")
        if self.target_count < 1 or self.target_count > 50:
            raise ValueError("payload target count is outside [1, 50]")
        if self.token_count > 127 or self.condition_length > self.token_count:
            raise ValueError("payload causal replay exceeds the 128-token protocol")
        if self.token_count != self.condition_length + self.target_count - 1:
            raise ValueError("causal replay token count changed")
        if not torch.equal(
            self.replay_token_ids[self.condition_length :].to(torch.int64),
            self.target_token_ids[:-1].to(torch.int64),
        ):
            raise ValueError("generated replay prefix does not match target token IDs")
        expected_positions = torch.arange(
            self.condition_length - 1,
            self.condition_length + self.target_count - 1,
            dtype=torch.int32,
        )
        if not torch.equal(self.target_positions.to(torch.int32), expected_positions):
            raise ValueError("target prediction positions changed")
        if self.feature_count and (
            int(self.feature_layer.min()) < 0
            or int(self.feature_layer.max()) >= self.n_layers
            or int(self.feature_position.min()) < 0
            or int(self.feature_position.max()) >= self.token_count
            or int(self.feature_id.min()) < 0
            or int(self.feature_id.max()) >= self.d_feature
        ):
            raise ValueError("feature identity is outside the S5 Graph schema")
        if self.error_count and (
            int(self.error_layer.min()) < 0
            or int(self.error_layer.max()) >= self.n_layers
            or int(self.error_position.min()) < 0
            or int(self.error_position.max()) >= self.token_count
        ):
            raise ValueError("error identity is outside the S5 Graph schema")
        if self.edge_source.numel() and (
            int(self.edge_source.min()) < 0
            or int(self.edge_target.min()) < 0
            or int(self.edge_source.max()) >= self.node_count
            or int(self.edge_target.max()) >= self.node_count
            or torch.any(self.edge_source == self.edge_target)
        ):
            raise ValueError("payload edge identity is invalid")
        if self.edge_source.numel():
            pairs = self.edge_source.long() * self.node_count + self.edge_target.long()
            if torch.unique(pairs).numel() != pairs.numel():
                raise ValueError("payload contains duplicate directed edges")
            if torch.any(self.edge_source == self.root_index):
                raise ValueError("joint root cannot be an edge source")
            target_is_feature = self.edge_target < self.feature_count
            target_is_root = self.edge_target == self.root_index
            if torch.any(~(target_is_feature | target_is_root)):
                raise ValueError("only feature nodes or the joint root may be edge targets")
            targets = self.edge_target[target_is_feature].long()
            sources = self.edge_source[target_is_feature].long()
            source_is_feature = sources < self.feature_count
            source_is_error = (sources >= self.feature_count) & (
                sources < self.token_start
            )
            layered_sources = source_is_feature | source_is_error
            if torch.any(layered_sources):
                source_layers = torch.empty_like(sources)
                source_layers[source_is_feature] = self.feature_layer[
                    sources[source_is_feature]
                ].long()
                source_layers[source_is_error] = self.error_layer[
                    sources[source_is_error] - self.feature_count
                ].long()
                target_layers = self.feature_layer[targets].long()
                if torch.any(source_layers[layered_sources] >= target_layers[layered_sources]):
                    raise ValueError(
                        "feature/error edges must target a later-layer feature"
                    )
        if not _is_dag(self.edge_source, self.edge_target, self.node_count):
            raise ValueError("payload Graph is not a DAG")
        floats = (
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
        )
        if any(not torch.isfinite(value.float()).all() for value in floats):
            raise ValueError("payload contains non-finite values")
        if not np.isfinite(self.root_influence) or not np.isfinite(
            self.root_cumulative_score
        ):
            raise ValueError("payload root statistics are non-finite")
        nonnegative = (
            self.feature_activation,
            self.feature_influence,
            self.error_influence,
            self.token_influence,
            self.edge_nonnegative_influence,
            self.target_probabilities,
            self.target_entropies,
        )
        if any(torch.any(value.float() < 0) for value in nonnegative):
            raise ValueError("payload nonnegative channel contains a negative value")
        if self.root_influence < 0:
            raise ValueError("payload root influence is negative")
        if self.edge_nonnegative_influence.numel() and torch.any(
            self.edge_nonnegative_influence <= 0
        ):
            raise ValueError("retained payload edge influence must be strictly positive")
        if self.edge_signed_direct_effect.numel() and torch.any(
            self.edge_signed_direct_effect == 0
        ):
            raise ValueError("retained payload edge signed effect cannot be zero")
        cumulative = (
            self.feature_cumulative_score,
            self.error_cumulative_score,
            self.token_cumulative_score,
        )
        if any(
            torch.any((value.float() < -1e-6) | (value.float() > 1.0 + 1e-6))
            for value in cumulative
        ) or not -1e-6 <= self.root_cumulative_score <= 1.0 + 1e-6:
            raise ValueError("payload cumulative score is outside [0, 1]")
        if torch.any(self.target_probabilities > 1):
            raise ValueError("payload target probability exceeds one")
        if self.replay_token_ids.numel() and (
            int(self.replay_token_ids.min()) < 0
            or int(self.replay_token_ids.max()) >= 2048
        ):
            raise ValueError("payload replay token is outside the S5 vocabulary")
        if self.target_token_ids.numel() and (
            int(self.target_token_ids.min()) < 0
            or int(self.target_token_ids.max()) >= 2048
        ):
            raise ValueError("payload target token is outside the S5 vocabulary")

    def approximate_bytes(self) -> int:
        arrays = (
            self.feature_layer,
            self.feature_position,
            self.feature_id,
            self.feature_activation,
            self.feature_influence,
            self.feature_cumulative_score,
            self.error_layer,
            self.error_position,
            self.error_influence,
            self.error_cumulative_score,
            self.replay_token_ids,
            self.token_influence,
            self.token_cumulative_score,
            self.edge_source,
            self.edge_target,
            self.edge_signed_direct_effect,
            self.edge_nonnegative_influence,
            self.target_token_ids,
            self.target_positions,
            self.target_probabilities,
            self.target_entropies,
            self.target_log_probabilities,
        )
        return sum(item.numel() * item.element_size() for item in arrays) + 512


def build_compact_payload(
    context: S5DecomposedForward | S5LinearizedForward,
    graph: Any,
    *,
    event_key: str,
    artifact_id: str,
    condition_length: int,
    d_feature: int,
) -> S5CompactPayload:
    retained_feature_indices = graph.retained_features.detach().cpu().long()
    feature_records = context.active_features.detach().cpu().long().index_select(
        0, retained_feature_indices
    )
    feature_activations = context.activation_values.detach().cpu().float().index_select(
        0, retained_feature_indices
    )
    token_count = context.token_count
    error_offsets = graph.retained_error_offsets.detach().cpu().long()
    error_layers = error_offsets // token_count
    error_positions = error_offsets % token_count
    token_offsets = graph.retained_token_offsets.detach().cpu().long()
    if not torch.equal(token_offsets, torch.arange(token_count)):
        raise ValueError("all replay token nodes must survive Graph pruning")
    rows, columns = torch.nonzero(
        graph.edge_nonnegative_influence > 0, as_tuple=True
    )
    feature_count = int(feature_records.shape[0])
    error_count = int(error_offsets.numel())
    token_start = feature_count + error_count
    root_index = int(graph.root_index)
    node_influence = graph.node_influence.detach().cpu().float()
    cumulative = graph.node_cumulative_score.detach().cpu().float()
    payload = S5CompactPayload(
        event_key=event_key,
        artifact_id=artifact_id,
        condition_length=int(condition_length),
        total_active_features=context.feature_count,
        n_layers=context.n_layers,
        d_feature=int(d_feature),
        feature_layer=feature_records[:, 0].to(torch.int16),
        feature_position=feature_records[:, 1].to(torch.int16),
        feature_id=feature_records[:, 2].to(torch.int32),
        feature_activation=feature_activations,
        feature_influence=node_influence[:feature_count],
        feature_cumulative_score=cumulative[:feature_count],
        error_layer=error_layers.to(torch.int16),
        error_position=error_positions.to(torch.int16),
        error_influence=node_influence[feature_count:token_start],
        error_cumulative_score=cumulative[feature_count:token_start],
        replay_token_ids=context.replay_token_ids.detach().cpu().to(torch.int16),
        token_influence=node_influence[token_start:root_index],
        token_cumulative_score=cumulative[token_start:root_index],
        root_influence=float(node_influence[root_index]),
        root_cumulative_score=float(cumulative[root_index]),
        edge_source=columns.detach().cpu().to(torch.int32),
        edge_target=rows.detach().cpu().to(torch.int32),
        edge_signed_direct_effect=graph.signed_direct_effect[rows, columns]
        .detach()
        .cpu()
        .float(),
        edge_nonnegative_influence=graph.edge_nonnegative_influence[rows, columns]
        .detach()
        .cpu()
        .float(),
        target_token_ids=context.objective.target_token_ids.detach()
        .cpu()
        .to(torch.int16),
        target_positions=context.objective.target_positions.detach()
        .cpu()
        .to(torch.int16),
        target_probabilities=context.objective.target_probabilities.detach().cpu().float(),
        target_entropies=context.objective.target_entropies.detach().cpu().float(),
        target_log_probabilities=context.objective.target_log_probabilities.detach()
        .cpu()
        .float(),
    )
    payload.validate()
    return payload


ARRAY_FIELDS = (
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


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def write_payload_chunk(
    path: str | Path,
    payloads: Sequence[S5CompactPayload],
    *,
    archive_identity: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(path).resolve()
    if not payloads:
        raise ValueError("cannot write an empty CompactPayload chunk")
    for payload in payloads:
        payload.validate()
    tensors: dict[str, Tensor] = {}
    for field in ARRAY_FIELDS:
        values = [getattr(payload, field) for payload in payloads]
        offsets = [0]
        for value in values:
            offsets.append(offsets[-1] + int(value.numel()))
        tensors[f"{field}__offsets"] = torch.tensor(offsets, dtype=torch.int64)
        tensors[field] = torch.cat(values) if offsets[-1] else values[0].new_empty(0)
    tensors["condition_length"] = torch.tensor(
        [payload.condition_length for payload in payloads], dtype=torch.int16
    )
    tensors["total_active_features"] = torch.tensor(
        [payload.total_active_features for payload in payloads], dtype=torch.int32
    )
    tensors["n_layers"] = torch.tensor(
        [payload.n_layers for payload in payloads], dtype=torch.uint8
    )
    tensors["d_feature"] = torch.tensor(
        [payload.d_feature for payload in payloads], dtype=torch.int32
    )
    tensors["root_influence"] = torch.tensor(
        [payload.root_influence for payload in payloads], dtype=torch.float32
    )
    tensors["root_cumulative_score"] = torch.tensor(
        [payload.root_cumulative_score for payload in payloads], dtype=torch.float32
    )
    metadata = {
        "format": PAYLOAD_FORMAT,
        "archive_identity": dict(archive_identity),
        "archive_identity_sha256": semantic_sha256(dict(archive_identity)),
        "records": [
            {"event_key": payload.event_key, "artifact_id": payload.artifact_id}
            for payload in payloads
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    sidecar = _sidecar(destination)
    sidecar_tmp = _sidecar(temporary)
    save_file(tensors, str(temporary))
    tensor_sha256 = sha256_file(temporary)
    metadata["tensor_sha256"] = tensor_sha256
    sidecar_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reopened_payloads = read_payload_chunk(temporary)
    if len(reopened_payloads) != len(payloads):
        raise RuntimeError("CompactPayload full reopen validation failed")
    sidecar_sha256 = sha256_file(sidecar_tmp)
    os.replace(temporary, destination)
    os.replace(sidecar_tmp, sidecar)
    if sha256_file(destination) != tensor_sha256 or sha256_file(sidecar) != sidecar_sha256:
        raise RuntimeError("CompactPayload hash changed during atomic publication")
    read_payload_chunk(destination)
    return {
        "path": str(destination),
        "sidecar": str(sidecar),
        "records": len(payloads),
        "bytes": destination.stat().st_size + sidecar.stat().st_size,
        "tensor_sha256": tensor_sha256,
        "sidecar_sha256": sidecar_sha256,
        "archive_identity_sha256": metadata["archive_identity_sha256"],
    }


def read_payload_chunk(path: str | Path) -> tuple[S5CompactPayload, ...]:
    source = Path(path).resolve()
    tensors = load_file(str(source), device="cpu")
    metadata = json.loads(_sidecar(source).read_text(encoding="utf-8"))
    if metadata.get("format") != PAYLOAD_FORMAT:
        raise ValueError("unsupported S5 CompactPayload format")
    if metadata.get("tensor_sha256") != sha256_file(source):
        raise ValueError("CompactPayload tensor checksum mismatch")
    result = []
    for index, record in enumerate(metadata["records"]):
        arrays = {}
        for field in ARRAY_FIELDS:
            offsets = tensors[f"{field}__offsets"]
            start, end = int(offsets[index]), int(offsets[index + 1])
            arrays[field] = tensors[field][start:end]
        payload = S5CompactPayload(
            event_key=record["event_key"],
            artifact_id=record["artifact_id"],
            condition_length=int(tensors["condition_length"][index]),
            total_active_features=int(tensors["total_active_features"][index]),
            n_layers=int(tensors["n_layers"][index]),
            d_feature=int(tensors["d_feature"][index]),
            root_influence=float(tensors["root_influence"][index]),
            root_cumulative_score=float(tensors["root_cumulative_score"][index]),
            **arrays,
        )
        payload.validate()
        result.append(payload)
    return tuple(result)


class ByteBoundedPayloadWriter:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_chunk_bytes: int,
        archive_identity: Mapping[str, Any],
        setting_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if max_chunk_bytes <= 0:
            raise ValueError("max_chunk_bytes must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.max_chunk_bytes = int(max_chunk_bytes)
        self.archive_identity = dict(archive_identity)
        self.setting_identity = (
            None if setting_identity is None else dict(setting_identity)
        )
        if self.setting_identity is not None:
            validate_graph_policy(self.setting_identity)
        self.pending: list[S5CompactPayload] = []
        self.pending_bytes = 0
        self.manifest: list[dict[str, Any]] = []
        self.closed = False
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.is_file():
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            if value.get("format") != PAYLOAD_FORMAT:
                raise ValueError("existing CompactPayload manifest format changed")
            if value.get("archive_identity_sha256") != semantic_sha256(
                self.archive_identity
            ):
                raise ValueError("existing CompactPayload archive identity changed")
            expected_setting_sha = (
                None
                if self.setting_identity is None
                else semantic_sha256(self.setting_identity)
            )
            if value.get("setting_identity_sha256") != expected_setting_sha:
                raise ValueError("existing CompactPayload setting identity changed")
            for record in value.get("chunks", []):
                verify_payload_chunk_record(record)
                self.manifest.append(dict(record))
            self.closed = value.get("status") == "complete"

    def append(self, payload: S5CompactPayload) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a completed CompactPayload archive")
        payload.validate()
        estimated = payload.approximate_bytes()
        if self.pending and self.pending_bytes + estimated > self.max_chunk_bytes:
            self.flush()
        self.pending.append(payload)
        self.pending_bytes += estimated

    def flush(self) -> None:
        if not self.pending:
            return
        index = len(self.manifest)
        record = write_payload_chunk(
            self.output_dir / f"payload_chunk_{index:05d}.safetensors",
            self.pending,
            archive_identity=self.archive_identity,
        )
        self.manifest.append(record)
        self.pending = []
        self.pending_bytes = 0
        _write_manifest_atomic(
            self.output_dir / "manifest.json",
            self.manifest,
            archive_identity_sha256=semantic_sha256(self.archive_identity),
            status="running",
            setting_identity=self.setting_identity,
        )

    def close(self) -> tuple[dict[str, Any], ...]:
        if self.closed:
            return tuple(self.manifest)
        self.flush()
        _write_manifest_atomic(
            self.output_dir / "manifest.json",
            self.manifest,
            archive_identity_sha256=semantic_sha256(self.archive_identity),
            status="complete",
            setting_identity=self.setting_identity,
        )
        self.closed = True
        return tuple(self.manifest)


def _write_manifest_atomic(
    path: Path,
    chunks: Iterable[Mapping[str, Any]],
    *,
    archive_identity_sha256: str,
    status: str,
    setting_identity: Mapping[str, Any] | None = None,
) -> None:
    if status not in {"running", "complete"}:
        raise ValueError("CompactPayload manifest status is invalid")
    value = {
        "format": PAYLOAD_FORMAT,
        "status": status,
        "archive_identity_sha256": archive_identity_sha256,
        "chunks": [dict(item) for item in chunks],
    }
    if setting_identity is not None:
        validate_graph_policy(setting_identity)
        value["setting_identity"] = dict(setting_identity)
        value["setting_identity_sha256"] = semantic_sha256(dict(setting_identity))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{time.time_ns()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


@dataclass(frozen=True)
class VerifiedPayloadCorpus:
    """A complete, checksum-verified CompactPayload archive manifest."""

    manifest_path: Path
    manifest_sha256: str
    archive_identity: Mapping[str, Any]
    archive_identity_sha256: str
    setting_identity: Mapping[str, Any] | None
    setting_identity_sha256: str | None
    chunks: tuple[Mapping[str, Any], ...]
    event_keys: tuple[str, ...]
    payload_artifact_ids: tuple[str, ...]
    source_records: tuple[Mapping[str, Any], ...]

    @property
    def event_count(self) -> int:
        return len(self.event_keys)

    def iter_payloads(self) -> Iterable[S5CompactPayload]:
        for record in self.chunks:
            yield from verify_payload_chunk_record(
                record,
                expected_archive_identity_sha256=self.archive_identity_sha256,
            )

    def provenance(self) -> dict[str, Any]:
        return {
            "payload_archive_identity": dict(self.archive_identity),
            "payload_archive_identity_sha256": self.archive_identity_sha256,
            "payload_manifest_sha256": self.manifest_sha256,
            "payload_setting_identity": (
                None if self.setting_identity is None else dict(self.setting_identity)
            ),
            "payload_setting_identity_sha256": self.setting_identity_sha256,
            "payload_sources": [dict(record) for record in self.source_records],
            "payload_event_count": self.event_count,
            "payload_event_keys_sha256": semantic_sha256(list(self.event_keys)),
            "payload_artifact_ids_sha256": semantic_sha256(
                list(self.payload_artifact_ids)
            ),
        }


def _resolve_manifest_chunk_record(
    record: Mapping[str, Any], *, manifest_parent: Path
) -> dict[str, Any]:
    resolved = dict(record)
    for key in ("path", "sidecar"):
        source = Path(str(record[key]))
        candidates = (
            source,
            manifest_parent / source,
            manifest_parent / source.name,
        )
        match = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if match is None:
            raise FileNotFoundError(f"payload manifest {key} is missing: {source}")
        resolved[key] = str(match)
    return resolved


def load_verified_payload_corpus(path: str | Path) -> VerifiedPayloadCorpus:
    """Load a complete archive manifest and verify every declared chunk."""

    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("format") != PAYLOAD_FORMAT or value.get("status") != "complete":
        raise ValueError("CompactPayload corpus manifest is not complete")
    archive_sha = str(value.get("archive_identity_sha256", ""))
    if len(archive_sha) != 64:
        raise ValueError("CompactPayload corpus archive identity is invalid")
    setting_identity = value.get("setting_identity")
    setting_sha = value.get("setting_identity_sha256")
    if setting_identity is not None:
        if not isinstance(setting_identity, dict):
            raise ValueError("CompactPayload setting identity is invalid")
        validate_graph_policy(setting_identity)
        if semantic_sha256(setting_identity) != setting_sha:
            raise ValueError("CompactPayload setting identity hash changed")
    elif setting_sha is not None:
        raise ValueError("CompactPayload setting hash has no identity")
    raw_chunks = value.get("chunks")
    if not isinstance(raw_chunks, list) or not raw_chunks:
        raise ValueError("CompactPayload corpus manifest contains no chunks")
    chunks = tuple(
        _resolve_manifest_chunk_record(record, manifest_parent=manifest_path.parent)
        for record in raw_chunks
    )
    archive_identity: dict[str, Any] | None = None
    event_keys: list[str] = []
    artifact_ids: list[str] = []
    source_records: list[dict[str, Any]] = []
    for index, record in enumerate(chunks):
        payloads = verify_payload_chunk_record(
            record, expected_archive_identity_sha256=archive_sha
        )
        sidecar_value = json.loads(
            Path(str(record["sidecar"])).read_text(encoding="utf-8")
        )
        current_identity = sidecar_value.get("archive_identity")
        if not isinstance(current_identity, dict):
            raise ValueError("CompactPayload sidecar omits archive identity")
        if semantic_sha256(current_identity) != archive_sha:
            raise ValueError("CompactPayload sidecar archive identity hash changed")
        if archive_identity is None:
            archive_identity = dict(current_identity)
        elif current_identity != archive_identity:
            raise ValueError("CompactPayload corpus mixes archive identities")
        chunk_event_keys = [payload.event_key for payload in payloads]
        chunk_artifact_ids = [payload.artifact_id for payload in payloads]
        event_keys.extend(chunk_event_keys)
        artifact_ids.extend(chunk_artifact_ids)
        source_records.append(
            {
                "chunk_index": index,
                "tensor_sha256": str(record["tensor_sha256"]),
                "sidecar_sha256": str(record["sidecar_sha256"]),
                "archive_identity_sha256": archive_sha,
                "events": len(payloads),
                "event_keys_sha256": semantic_sha256(chunk_event_keys),
                "payload_artifact_ids_sha256": semantic_sha256(chunk_artifact_ids),
            }
        )
    if len(set(event_keys)) != len(event_keys):
        raise ValueError("CompactPayload corpus contains duplicate events")
    if archive_identity is None:
        raise AssertionError("verified CompactPayload corpus has no identity")
    if setting_identity is not None:
        for event_key, artifact_id in zip(event_keys, artifact_ids, strict=True):
            expected_artifact_id = circuit_artifact_id(
                event_key=event_key,
                model_identity=archive_identity["model_identity"],
                tokenizer_sha256=str(archive_identity["tokenizer_sha256"]),
                transcoder_bank_sha256=str(
                    archive_identity["transcoder_bank_sha256"]
                ),
                graph_policy=setting_identity,
                code_identity=str(archive_identity["code_identity"]),
            )
            if artifact_id != expected_artifact_id:
                raise ValueError("CompactPayload artifact ID does not bind Graph policy")
    return VerifiedPayloadCorpus(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        archive_identity=archive_identity,
        archive_identity_sha256=archive_sha,
        setting_identity=setting_identity,
        setting_identity_sha256=(None if setting_sha is None else str(setting_sha)),
        chunks=chunks,
        event_keys=tuple(event_keys),
        payload_artifact_ids=tuple(artifact_ids),
        source_records=tuple(source_records),
    )


def write_verified_payload_index(
    path: str | Path,
    payload_paths: Sequence[str | Path],
    *,
    expected_scope: str,
    expected_events: int,
    setting_identity: Mapping[str, Any],
) -> VerifiedPayloadCorpus:
    """Index independently published payload chunks into one complete corpus."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"CompactPayload index already exists: {destination}")
    if expected_events <= 0 or len(payload_paths) == 0:
        raise ValueError("CompactPayload index input is empty")
    validate_graph_policy(setting_identity)
    records: list[dict[str, Any]] = []
    archive_sha: str | None = None
    archive_identity: Mapping[str, Any] | None = None
    event_keys: list[str] = []
    for source_value in payload_paths:
        source = Path(source_value).resolve()
        sidecar = _sidecar(source)
        sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
        current_identity = sidecar_value.get("archive_identity")
        current_sha = str(sidecar_value.get("archive_identity_sha256", ""))
        if not isinstance(current_identity, Mapping) or semantic_sha256(
            dict(current_identity)
        ) != current_sha:
            raise ValueError("CompactPayload index source archive identity changed")
        if current_identity.get("scope") != expected_scope:
            raise ValueError("CompactPayload index source scope is unexpected")
        if archive_sha is None:
            archive_sha = current_sha
            archive_identity = current_identity
        elif current_sha != archive_sha or current_identity != archive_identity:
            raise ValueError("CompactPayload index mixes archive identities")
        payloads = read_payload_chunk(source)
        event_keys.extend(payload.event_key for payload in payloads)
        records.append(
            {
                "path": os.path.relpath(source, destination.parent),
                "sidecar": os.path.relpath(sidecar, destination.parent),
                "records": len(payloads),
                "bytes": source.stat().st_size + sidecar.stat().st_size,
                "tensor_sha256": sha256_file(source),
                "sidecar_sha256": sha256_file(sidecar),
                "archive_identity_sha256": current_sha,
            }
        )
    if len(event_keys) != expected_events or len(set(event_keys)) != expected_events:
        raise ValueError("CompactPayload index event coverage is incomplete")
    if archive_sha is None:
        raise AssertionError("CompactPayload index has no archive identity")
    _write_manifest_atomic(
        destination,
        records,
        archive_identity_sha256=archive_sha,
        status="complete",
        setting_identity=setting_identity,
    )
    return load_verified_payload_corpus(destination)


def verify_payload_chunk_record(
    record: Mapping[str, Any],
    *,
    expected_archive_identity_sha256: str | None = None,
) -> tuple[S5CompactPayload, ...]:
    path = Path(str(record["path"])).resolve()
    sidecar = Path(str(record["sidecar"])).resolve()
    if sha256_file(path) != record.get("tensor_sha256"):
        raise ValueError("CompactPayload manifest tensor checksum mismatch")
    if sha256_file(sidecar) != record.get("sidecar_sha256"):
        raise ValueError("CompactPayload manifest sidecar checksum mismatch")
    if (
        expected_archive_identity_sha256 is not None
        and record.get("archive_identity_sha256")
        != expected_archive_identity_sha256
    ):
        raise ValueError("CompactPayload archive identity mismatch")
    sidecar_value = json.loads(sidecar.read_text(encoding="utf-8"))
    if sidecar_value.get("archive_identity_sha256") != record.get(
        "archive_identity_sha256"
    ):
        raise ValueError("CompactPayload sidecar/archive identity mismatch")
    payloads = read_payload_chunk(path)
    if len(payloads) != int(record["records"]):
        raise ValueError("CompactPayload manifest record count mismatch")
    return payloads


def write_dense_reference_artifact(
    path: str | Path,
    *,
    signed_direct_effect: Tensor,
    selected_features: Tensor,
    selected_feature_records: Tensor,
    replay_token_ids: Tensor,
    event_key: str,
    artifact_id: str,
    archive_identity: Mapping[str, Any],
) -> dict[str, Any]:
    destination = Path(path).resolve()
    matrix = signed_direct_effect.detach().cpu().float().contiguous()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("dense reference matrix must be square")
    if not torch.isfinite(matrix).all():
        raise ValueError("dense reference matrix contains non-finite values")
    selected = selected_features.detach().cpu().long().contiguous()
    records = selected_feature_records.detach().cpu().to(torch.int32).contiguous()
    if records.shape != (selected.numel(), 3):
        raise ValueError("dense reference selected feature records do not align")
    tensors = {
        "signed_direct_effect": matrix,
        "selected_features": selected,
        "selected_feature_records": records,
        "replay_token_ids": replay_token_ids.detach().cpu().to(torch.int16),
    }
    metadata = {
        "format": "s5_dense_reference_direct_effect_v1",
        "event_key": event_key,
        "artifact_id": artifact_id,
        "archive_identity_sha256": semantic_sha256(dict(archive_identity)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name("." + destination.name + ".tmp")
    sidecar = _sidecar(destination)
    sidecar_tmp = _sidecar(temporary)
    save_file(tensors, str(temporary))
    metadata["tensor_sha256"] = sha256_file(temporary)
    sidecar_tmp.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reopened = load_file(str(temporary), device="cpu")
    if not torch.equal(reopened["signed_direct_effect"], matrix):
        raise RuntimeError("dense reference matrix reopen parity failed")
    sidecar_sha = sha256_file(sidecar_tmp)
    os.replace(temporary, destination)
    os.replace(sidecar_tmp, sidecar)
    return {
        "path": str(destination),
        "sidecar": str(sidecar),
        "tensor_sha256": metadata["tensor_sha256"],
        "sidecar_sha256": sidecar_sha,
        "archive_identity_sha256": metadata["archive_identity_sha256"],
        "event_key": event_key,
        "artifact_id": artifact_id,
        "nodes": int(matrix.shape[0]),
        "bytes": destination.stat().st_size + sidecar.stat().st_size,
    }


def verify_dense_reference_record(
    record: Mapping[str, Any], *, expected_archive_identity_sha256: str
) -> None:
    path = Path(str(record["path"])).resolve()
    sidecar = Path(str(record["sidecar"])).resolve()
    if sha256_file(path) != record.get("tensor_sha256"):
        raise ValueError("dense reference tensor checksum mismatch")
    if sha256_file(sidecar) != record.get("sidecar_sha256"):
        raise ValueError("dense reference sidecar checksum mismatch")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        metadata.get("format") != "s5_dense_reference_direct_effect_v1"
        or metadata.get("archive_identity_sha256")
        != expected_archive_identity_sha256
        or record.get("archive_identity_sha256")
        != expected_archive_identity_sha256
    ):
        raise ValueError("dense reference identity mismatch")
    tensors = load_file(str(path), device="cpu")
    matrix = tensors["signed_direct_effect"]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not torch.isfinite(matrix).all():
        raise ValueError("dense reference matrix validation failed")

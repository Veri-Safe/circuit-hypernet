"""Condition-safe sparse S5 Circuit Graph Payload v2 storage."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from .attribution import S5DecomposedForward, S5LinearizedForward


PAYLOAD_V2_FORMAT = "s5_compact_circuit_graph_payload_v2"
PAYLOAD_V2_SCHEMA_VERSION = 2
FLOAT_FIELDS = (
    "feature_activation",
    "feature_influence",
    "error_influence",
    "token_influence",
    "edge_signed_direct_effect",
    "edge_nonnegative_influence",
)


def semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_artifact_id(*, event_key: str, archive_identity_sha256: str) -> str:
    return semantic_sha256(
        {
            "payload_schema_version": PAYLOAD_V2_SCHEMA_VERSION,
            "event_key": str(event_key),
            "archive_identity_sha256": str(archive_identity_sha256),
        }
    )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _unique_rows(*columns: Tensor) -> bool:
    if not columns or columns[0].numel() == 0:
        return True
    rows = torch.stack([column.long() for column in columns], dim=1)
    return int(torch.unique(rows, dim=0).shape[0]) == int(rows.shape[0])


def _is_dag(edge_source: Tensor, edge_target: Tensor, node_count: int) -> bool:
    if edge_source.numel() == 0:
        return True
    source = edge_source.detach().cpu().numpy().astype(np.int64, copy=False)
    target = edge_target.detach().cpu().numpy().astype(np.int64, copy=False)
    order = np.argsort(source, kind="stable")
    sorted_source = source[order]
    sorted_target = target[order]
    counts = np.bincount(sorted_source, minlength=node_count)
    offsets = np.empty(node_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    indegree = np.bincount(target, minlength=node_count).astype(np.int64, copy=False)
    queue = deque(np.flatnonzero(indegree == 0).tolist())
    processed = 0
    while queue:
        node = queue.popleft()
        processed += 1
        neighbors = sorted_target[int(offsets[node]) : int(offsets[node + 1])]
        for neighbor in neighbors.tolist():
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(int(neighbor))
    return processed == int(node_count)


def _require_dtype(name: str, value: Tensor, dtype: torch.dtype) -> None:
    if value.device.type != "cpu" or value.dtype != dtype or value.ndim != 1:
        raise ValueError(
            f"Payload v2 {name} must be a CPU 1D {dtype} tensor, "
            f"got device={value.device} dtype={value.dtype} shape={tuple(value.shape)}"
        )


def payload_v2_equal(first: "S5GraphPayloadV2", second: "S5GraphPayloadV2") -> bool:
    if (
        first.event_key != second.event_key
        or first.payload_artifact_id != second.payload_artifact_id
        or first.condition_length != second.condition_length
    ):
        return False
    tensor_fields = (
        "feature_layer",
        "feature_position",
        "feature_id",
        "feature_activation",
        "feature_influence",
        "error_layer",
        "error_position",
        "error_influence",
        "token_influence",
        "root_influence",
        "edge_source",
        "edge_target",
        "edge_signed_direct_effect",
        "edge_nonnegative_influence",
    )
    return all(
        torch.equal(getattr(first, field), getattr(second, field))
        for field in tensor_fields
    )


@dataclass(frozen=True)
class S5GraphPayloadV2:
    event_key: str
    payload_artifact_id: str
    condition_length: int
    feature_layer: Tensor
    feature_position: Tensor
    feature_id: Tensor
    feature_activation: Tensor
    feature_influence: Tensor
    error_layer: Tensor
    error_position: Tensor
    error_influence: Tensor
    token_influence: Tensor
    root_influence: Tensor
    edge_source: Tensor
    edge_target: Tensor
    edge_signed_direct_effect: Tensor
    edge_nonnegative_influence: Tensor

    @property
    def feature_count(self) -> int:
        return int(self.feature_id.numel())

    @property
    def error_count(self) -> int:
        return int(self.error_layer.numel())

    @property
    def token_count(self) -> int:
        return int(self.token_influence.numel())

    @property
    def edge_count(self) -> int:
        return int(self.edge_source.numel())

    @property
    def token_start(self) -> int:
        return self.feature_count + self.error_count

    @property
    def root_index(self) -> int:
        return self.feature_count + self.error_count + self.token_count

    @property
    def node_count(self) -> int:
        return self.root_index + 1

    def validate(
        self,
        *,
        n_layers: int,
        d_feature: int,
        archive_identity_sha256: str | None = None,
    ) -> None:
        if not _is_sha256(self.event_key) or not _is_sha256(self.payload_artifact_id):
            raise ValueError("Payload v2 identities must be lowercase SHA256")
        if archive_identity_sha256 is not None:
            expected = payload_artifact_id(
                event_key=self.event_key,
                archive_identity_sha256=archive_identity_sha256,
            )
            if self.payload_artifact_id != expected:
                raise ValueError("Payload v2 artifact identity changed")
        if int(n_layers) <= 0 or int(d_feature) <= 0:
            raise ValueError("Payload v2 archive dimensions are invalid")

        int_fields = {
            "feature_layer": (self.feature_layer, torch.int16),
            "feature_position": (self.feature_position, torch.int16),
            "feature_id": (self.feature_id, torch.int32),
            "error_layer": (self.error_layer, torch.int16),
            "error_position": (self.error_position, torch.int16),
            "edge_source": (self.edge_source, torch.int32),
            "edge_target": (self.edge_target, torch.int32),
        }
        for name, (value, dtype) in int_fields.items():
            _require_dtype(name, value, dtype)
        for name in FLOAT_FIELDS:
            _require_dtype(name, getattr(self, name), torch.float32)
        if (
            self.root_influence.device.type != "cpu"
            or self.root_influence.dtype != torch.float32
            or self.root_influence.ndim != 0
        ):
            raise ValueError("Payload v2 root_influence must be a CPU FP32 scalar")

        feature_lengths = {
            int(value.numel())
            for value in (
                self.feature_layer,
                self.feature_position,
                self.feature_id,
                self.feature_activation,
                self.feature_influence,
            )
        }
        error_lengths = {
            int(value.numel())
            for value in (
                self.error_layer,
                self.error_position,
                self.error_influence,
            )
        }
        edge_lengths = {
            int(value.numel())
            for value in (
                self.edge_source,
                self.edge_target,
                self.edge_signed_direct_effect,
                self.edge_nonnegative_influence,
            )
        }
        if feature_lengths != {self.feature_count} or error_lengths != {self.error_count}:
            raise ValueError("Payload v2 node arrays are misaligned")
        if edge_lengths != {self.edge_count}:
            raise ValueError("Payload v2 edge arrays are misaligned")
        if not 0 < int(self.condition_length) <= self.token_count <= 127:
            raise ValueError("Payload v2 causal replay boundary is invalid")
        if self.feature_count and (
            int(self.feature_layer.min()) < 0
            or int(self.feature_layer.max()) >= int(n_layers)
            or int(self.feature_position.min()) < 0
            or int(self.feature_position.max()) >= self.token_count
            or int(self.feature_id.min()) < 0
            or int(self.feature_id.max()) >= int(d_feature)
        ):
            raise ValueError("Payload v2 feature identity is out of range")
        if self.error_count and (
            int(self.error_layer.min()) < 0
            or int(self.error_layer.max()) >= int(n_layers)
            or int(self.error_position.min()) < 0
            or int(self.error_position.max()) >= self.token_count
        ):
            raise ValueError("Payload v2 error identity is out of range")
        if not _unique_rows(self.feature_layer, self.feature_position, self.feature_id):
            raise ValueError("Payload v2 contains duplicate feature identities")
        if not _unique_rows(self.error_layer, self.error_position):
            raise ValueError("Payload v2 contains duplicate error identities")

        float_values = [getattr(self, name) for name in FLOAT_FIELDS] + [self.root_influence]
        if any(not bool(torch.isfinite(value).all()) for value in float_values):
            raise ValueError("Payload v2 contains non-finite values")
        for value in (
            self.feature_activation,
            self.feature_influence,
            self.error_influence,
            self.token_influence,
            self.root_influence,
        ):
            if bool(torch.any(value < 0)):
                raise ValueError("Payload v2 nonnegative node channel is negative")
        if self.edge_count and bool(torch.any(self.edge_nonnegative_influence <= 0)):
            raise ValueError("Payload v2 retained edge influence must be positive")
        if self.edge_count and bool(torch.any(self.edge_signed_direct_effect == 0)):
            raise ValueError("Payload v2 retained signed edge effect must be nonzero")

        if self.edge_count:
            source = self.edge_source.long()
            target = self.edge_target.long()
            if (
                int(source.min()) < 0
                or int(target.min()) < 0
                or int(source.max()) >= self.node_count
                or int(target.max()) >= self.node_count
                or bool(torch.any(source == target))
            ):
                raise ValueError("Payload v2 edge endpoint is invalid")
            if not _unique_rows(source, target):
                raise ValueError("Payload v2 contains duplicate directed edges")
            if bool(torch.any(source == self.root_index)):
                raise ValueError("Payload v2 root cannot be an edge source")
            target_is_feature = target < self.feature_count
            target_is_root = target == self.root_index
            if bool(torch.any(~(target_is_feature | target_is_root))):
                raise ValueError("Payload v2 edge target must be a feature or root")

            feature_targets = target_is_feature.nonzero(as_tuple=False).flatten()
            if feature_targets.numel():
                source_to_feature = source.index_select(0, feature_targets)
                target_features = target.index_select(0, feature_targets)
                source_is_feature = source_to_feature < self.feature_count
                source_is_error = (source_to_feature >= self.feature_count) & (
                    source_to_feature < self.token_start
                )
                source_is_token = (source_to_feature >= self.token_start) & (
                    source_to_feature < self.root_index
                )
                if bool(torch.any(~(source_is_feature | source_is_error | source_is_token))):
                    raise ValueError("Payload v2 feature target has an invalid source")
                target_layers = self.feature_layer[target_features].long()
                target_positions = self.feature_position[target_features].long()
                if bool(torch.any(source_is_feature)):
                    feature_sources = source_to_feature[source_is_feature]
                    if bool(
                        torch.any(
                            self.feature_layer[feature_sources].long()
                            >= target_layers[source_is_feature]
                        )
                    ):
                        raise ValueError("Payload v2 feature edge does not move to a later layer")
                    if bool(
                        torch.any(
                            self.feature_position[feature_sources].long()
                            > target_positions[source_is_feature]
                        )
                    ):
                        raise ValueError("Payload v2 feature edge moves backward in replay position")
                if bool(torch.any(source_is_error)):
                    error_sources = source_to_feature[source_is_error] - self.feature_count
                    if bool(
                        torch.any(
                            self.error_layer[error_sources].long()
                            >= target_layers[source_is_error]
                        )
                    ):
                        raise ValueError("Payload v2 error edge does not move to a later layer")
                    if bool(
                        torch.any(
                            self.error_position[error_sources].long()
                            > target_positions[source_is_error]
                        )
                    ):
                        raise ValueError("Payload v2 error edge moves backward in replay position")
                if bool(torch.any(source_is_token)):
                    token_positions = source_to_feature[source_is_token] - self.token_start
                    if bool(torch.any(token_positions > target_positions[source_is_token])):
                        raise ValueError("Payload v2 token edge moves backward in replay position")
        if not _is_dag(self.edge_source, self.edge_target, self.node_count):
            raise ValueError("Payload v2 Graph is not a DAG")

    def approximate_bytes(self) -> int:
        arrays = (
            self.feature_layer,
            self.feature_position,
            self.feature_id,
            self.feature_activation,
            self.feature_influence,
            self.error_layer,
            self.error_position,
            self.error_influence,
            self.token_influence,
            self.root_influence,
            self.edge_source,
            self.edge_target,
            self.edge_signed_direct_effect,
            self.edge_nonnegative_influence,
        )
        return sum(int(value.numel()) * int(value.element_size()) for value in arrays)


def build_payload_v2(
    context: S5DecomposedForward | S5LinearizedForward,
    graph: Any,
    *,
    event_key: str,
    payload_artifact_id_value: str,
    condition_length: int,
    d_feature: int,
    archive_identity_sha256: str | None = None,
) -> S5GraphPayloadV2:
    retained = graph.retained_features.detach().cpu().long()
    feature_records = context.active_features.detach().cpu().long().index_select(0, retained)
    feature_activation = context.activation_values.detach().cpu().float().index_select(0, retained)
    token_count = int(context.token_count)
    error_offsets = graph.retained_error_offsets.detach().cpu().long()
    token_offsets = graph.retained_token_offsets.detach().cpu().long()
    if not torch.equal(token_offsets, torch.arange(token_count)):
        raise ValueError("Payload v2 requires every replay token in canonical order")
    feature_count = int(feature_records.shape[0])
    error_count = int(error_offsets.numel())
    token_start = feature_count + error_count
    root_index = feature_count + error_count + token_count
    if int(graph.root_index) != root_index:
        raise ValueError("Payload v2 root index changed")
    node_influence = graph.node_influence.detach().cpu().float()
    if int(node_influence.numel()) != root_index + 1:
        raise ValueError("Payload v2 node influence does not cover canonical nodes")
    rows, columns = torch.nonzero(graph.edge_nonnegative_influence > 0, as_tuple=True)
    payload = S5GraphPayloadV2(
        event_key=str(event_key),
        payload_artifact_id=str(payload_artifact_id_value),
        condition_length=int(condition_length),
        feature_layer=feature_records[:, 0].to(torch.int16).contiguous(),
        feature_position=feature_records[:, 1].to(torch.int16).contiguous(),
        feature_id=feature_records[:, 2].to(torch.int32).contiguous(),
        feature_activation=feature_activation.to(torch.float32).contiguous(),
        feature_influence=node_influence[:feature_count].contiguous(),
        error_layer=(error_offsets // token_count).to(torch.int16).contiguous(),
        error_position=(error_offsets % token_count).to(torch.int16).contiguous(),
        error_influence=node_influence[feature_count:token_start].contiguous(),
        token_influence=node_influence[token_start:root_index].contiguous(),
        root_influence=node_influence[root_index].clone().to(torch.float32),
        edge_source=columns.detach().cpu().to(torch.int32).contiguous(),
        edge_target=rows.detach().cpu().to(torch.int32).contiguous(),
        edge_signed_direct_effect=graph.signed_direct_effect[rows, columns]
        .detach()
        .cpu()
        .float()
        .contiguous(),
        edge_nonnegative_influence=graph.edge_nonnegative_influence[rows, columns]
        .detach()
        .cpu()
        .float()
        .contiguous(),
    )
    payload.validate(
        n_layers=int(context.n_layers),
        d_feature=int(d_feature),
        archive_identity_sha256=archive_identity_sha256,
    )
    return payload


def payload_v2_from_v1(
    source: Any,
    *,
    archive_identity_sha256: str,
    n_layers: int,
    d_feature: int,
) -> S5GraphPayloadV2:
    """Drop v1-only diagnostics while preserving the complete retained sparse Graph."""

    event_key = str(source.event_key)
    payload = S5GraphPayloadV2(
        event_key=event_key,
        payload_artifact_id=payload_artifact_id(
            event_key=event_key,
            archive_identity_sha256=archive_identity_sha256,
        ),
        condition_length=int(source.condition_length),
        feature_layer=source.feature_layer.detach().cpu().to(torch.int16).contiguous(),
        feature_position=source.feature_position.detach()
        .cpu()
        .to(torch.int16)
        .contiguous(),
        feature_id=source.feature_id.detach().cpu().to(torch.int32).contiguous(),
        feature_activation=source.feature_activation.detach()
        .cpu()
        .float()
        .contiguous(),
        feature_influence=source.feature_influence.detach()
        .cpu()
        .float()
        .contiguous(),
        error_layer=source.error_layer.detach().cpu().to(torch.int16).contiguous(),
        error_position=source.error_position.detach()
        .cpu()
        .to(torch.int16)
        .contiguous(),
        error_influence=source.error_influence.detach().cpu().float().contiguous(),
        token_influence=source.token_influence.detach().cpu().float().contiguous(),
        root_influence=torch.tensor(float(source.root_influence), dtype=torch.float32),
        edge_source=source.edge_source.detach().cpu().to(torch.int32).contiguous(),
        edge_target=source.edge_target.detach().cpu().to(torch.int32).contiguous(),
        edge_signed_direct_effect=source.edge_signed_direct_effect.detach()
        .cpu()
        .float()
        .contiguous(),
        edge_nonnegative_influence=source.edge_nonnegative_influence.detach()
        .cpu()
        .float()
        .contiguous(),
    )
    payload.validate(
        n_layers=int(n_layers),
        d_feature=int(d_feature),
        archive_identity_sha256=archive_identity_sha256,
    )
    return payload


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _offsets(lengths: Sequence[int]) -> Tensor:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return torch.tensor(values, dtype=torch.int64)


def _concat(payloads: Sequence[S5GraphPayloadV2], field: str, empty: Tensor) -> Tensor:
    values = [getattr(payload, field) for payload in payloads]
    return torch.cat(values) if any(value.numel() for value in values) else empty


def write_payload_v2_shard(
    path: str | Path,
    payloads: Sequence[S5GraphPayloadV2],
    *,
    archive_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not payloads:
        raise ValueError("cannot write an empty Payload v2 shard")
    identity = dict(archive_identity)
    archive_sha = semantic_sha256(identity)
    n_layers = int(identity["n_layers"])
    d_feature = int(identity["d_feature"])
    for payload in payloads:
        payload.validate(
            n_layers=n_layers,
            d_feature=d_feature,
            archive_identity_sha256=archive_sha,
        )
    feature_lengths = [payload.feature_count for payload in payloads]
    error_lengths = [payload.error_count for payload in payloads]
    token_lengths = [payload.token_count for payload in payloads]
    edge_lengths = [payload.edge_count for payload in payloads]
    tensors = {
        "feature_offsets": _offsets(feature_lengths),
        "feature_layer": _concat(payloads, "feature_layer", torch.empty(0, dtype=torch.int16)),
        "feature_position": _concat(payloads, "feature_position", torch.empty(0, dtype=torch.int16)),
        "feature_id": _concat(payloads, "feature_id", torch.empty(0, dtype=torch.int32)),
        "feature_activation": _concat(payloads, "feature_activation", torch.empty(0, dtype=torch.float32)),
        "feature_influence": _concat(payloads, "feature_influence", torch.empty(0, dtype=torch.float32)),
        "error_offsets": _offsets(error_lengths),
        "error_layer": _concat(payloads, "error_layer", torch.empty(0, dtype=torch.int16)),
        "error_position": _concat(payloads, "error_position", torch.empty(0, dtype=torch.int16)),
        "error_influence": _concat(payloads, "error_influence", torch.empty(0, dtype=torch.float32)),
        "token_offsets": _offsets(token_lengths),
        "token_influence": _concat(payloads, "token_influence", torch.empty(0, dtype=torch.float32)),
        "edge_offsets": _offsets(edge_lengths),
        "edge_source": _concat(payloads, "edge_source", torch.empty(0, dtype=torch.int32)),
        "edge_target": _concat(payloads, "edge_target", torch.empty(0, dtype=torch.int32)),
        "edge_signed_direct_effect": _concat(payloads, "edge_signed_direct_effect", torch.empty(0, dtype=torch.float32)),
        "edge_nonnegative_influence": _concat(payloads, "edge_nonnegative_influence", torch.empty(0, dtype=torch.float32)),
        "condition_length": torch.tensor([payload.condition_length for payload in payloads], dtype=torch.int16),
        "root_influence": torch.stack([payload.root_influence for payload in payloads]).float(),
    }
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary_sidecar = _sidecar(temporary)
    sidecar = _sidecar(destination)
    save_file(tensors, str(temporary))
    tensor_sha = sha256_file(temporary)
    metadata = {
        "format": PAYLOAD_V2_FORMAT,
        "schema_version": PAYLOAD_V2_SCHEMA_VERSION,
        "archive_identity": identity,
        "archive_identity_sha256": archive_sha,
        "tensor_sha256": tensor_sha,
        "records": [
            {
                "event_key": payload.event_key,
                "payload_artifact_id": payload.payload_artifact_id,
                "feature_count": payload.feature_count,
                "error_count": payload.error_count,
                "token_count": payload.token_count,
                "edge_count": payload.edge_count,
                "node_count": payload.node_count,
            }
            for payload in payloads
        ],
    }
    temporary_sidecar.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    reopened = read_payload_v2_shard(temporary)
    if len(reopened) != len(payloads) or any(
        not payload_v2_equal(first, second)
        for first, second in zip(reopened, payloads, strict=True)
    ):
        raise RuntimeError("Payload v2 temporary reopen parity failed")
    sidecar_sha = sha256_file(temporary_sidecar)
    os.replace(temporary, destination)
    os.replace(temporary_sidecar, sidecar)
    record = {
        "path": str(destination),
        "sidecar": str(sidecar),
        "records": len(payloads),
        "bytes": destination.stat().st_size + sidecar.stat().st_size,
        "tensor_sha256": tensor_sha,
        "sidecar_sha256": sidecar_sha,
        "archive_identity_sha256": archive_sha,
        "event_keys_sha256": semantic_sha256([payload.event_key for payload in payloads]),
        "payload_artifact_ids_sha256": semantic_sha256(
            [payload.payload_artifact_id for payload in payloads]
        ),
    }
    verify_payload_v2_shard_record(record, expected_archive_identity_sha256=archive_sha)
    return record


def _slice(tensor: Tensor, offsets: Tensor, index: int) -> Tensor:
    start = int(offsets[index])
    end = int(offsets[index + 1])
    return tensor[start:end].contiguous()


def read_payload_v2_shard(path: str | Path) -> tuple[S5GraphPayloadV2, ...]:
    source = Path(path).resolve()
    tensors = load_file(str(source), device="cpu")
    metadata = json.loads(_sidecar(source).read_text(encoding="utf-8"))
    if metadata.get("format") != PAYLOAD_V2_FORMAT:
        raise ValueError("Payload v2 sidecar format changed")
    identity = metadata.get("archive_identity")
    if not isinstance(identity, dict):
        raise ValueError("Payload v2 sidecar omits archive identity")
    archive_sha = str(metadata.get("archive_identity_sha256", ""))
    if semantic_sha256(identity) != archive_sha:
        raise ValueError("Payload v2 archive identity changed")
    if sha256_file(source) != metadata.get("tensor_sha256"):
        raise ValueError("Payload v2 tensor checksum changed")
    records = metadata.get("records")
    if not isinstance(records, list):
        raise ValueError("Payload v2 sidecar records are invalid")
    required = {
        "feature_offsets",
        "feature_layer",
        "feature_position",
        "feature_id",
        "feature_activation",
        "feature_influence",
        "error_offsets",
        "error_layer",
        "error_position",
        "error_influence",
        "token_offsets",
        "token_influence",
        "edge_offsets",
        "edge_source",
        "edge_target",
        "edge_signed_direct_effect",
        "edge_nonnegative_influence",
        "condition_length",
        "root_influence",
    }
    if set(tensors) != required:
        raise ValueError("Payload v2 tensor schema changed")
    count = len(records)
    for field in ("feature_offsets", "error_offsets", "token_offsets", "edge_offsets"):
        value = tensors[field]
        if value.dtype != torch.int64 or value.shape != (count + 1,):
            raise ValueError(f"Payload v2 {field} is invalid")
        if int(value[0]) != 0 or bool(torch.any(value[1:] < value[:-1])):
            raise ValueError(f"Payload v2 {field} is not monotonic")
    if tensors["condition_length"].shape != (count,) or tensors["root_influence"].shape != (count,):
        raise ValueError("Payload v2 scalar arrays do not align")
    result = []
    for index, record in enumerate(records):
        payload = S5GraphPayloadV2(
            event_key=str(record["event_key"]),
            payload_artifact_id=str(record["payload_artifact_id"]),
            condition_length=int(tensors["condition_length"][index]),
            feature_layer=_slice(tensors["feature_layer"], tensors["feature_offsets"], index),
            feature_position=_slice(tensors["feature_position"], tensors["feature_offsets"], index),
            feature_id=_slice(tensors["feature_id"], tensors["feature_offsets"], index),
            feature_activation=_slice(tensors["feature_activation"], tensors["feature_offsets"], index),
            feature_influence=_slice(tensors["feature_influence"], tensors["feature_offsets"], index),
            error_layer=_slice(tensors["error_layer"], tensors["error_offsets"], index),
            error_position=_slice(tensors["error_position"], tensors["error_offsets"], index),
            error_influence=_slice(tensors["error_influence"], tensors["error_offsets"], index),
            token_influence=_slice(tensors["token_influence"], tensors["token_offsets"], index),
            root_influence=tensors["root_influence"][index].clone(),
            edge_source=_slice(tensors["edge_source"], tensors["edge_offsets"], index),
            edge_target=_slice(tensors["edge_target"], tensors["edge_offsets"], index),
            edge_signed_direct_effect=_slice(tensors["edge_signed_direct_effect"], tensors["edge_offsets"], index),
            edge_nonnegative_influence=_slice(tensors["edge_nonnegative_influence"], tensors["edge_offsets"], index),
        )
        payload.validate(
            n_layers=int(identity["n_layers"]),
            d_feature=int(identity["d_feature"]),
            archive_identity_sha256=archive_sha,
        )
        if record != {
            "event_key": payload.event_key,
            "payload_artifact_id": payload.payload_artifact_id,
            "feature_count": payload.feature_count,
            "error_count": payload.error_count,
            "token_count": payload.token_count,
            "edge_count": payload.edge_count,
            "node_count": payload.node_count,
        }:
            raise ValueError("Payload v2 sidecar counts do not match tensors")
        result.append(payload)
    return tuple(result)


def verify_payload_v2_shard_record(
    record: Mapping[str, Any], *, expected_archive_identity_sha256: str
) -> tuple[S5GraphPayloadV2, ...]:
    path = Path(str(record["path"])).resolve()
    sidecar = Path(str(record["sidecar"])).resolve()
    if sha256_file(path) != record.get("tensor_sha256"):
        raise ValueError("Payload v2 manifest tensor checksum mismatch")
    if sha256_file(sidecar) != record.get("sidecar_sha256"):
        raise ValueError("Payload v2 manifest sidecar checksum mismatch")
    if record.get("archive_identity_sha256") != expected_archive_identity_sha256:
        raise ValueError("Payload v2 manifest archive identity mismatch")
    payloads = read_payload_v2_shard(path)
    if len(payloads) != int(record["records"]):
        raise ValueError("Payload v2 manifest record count mismatch")
    if semantic_sha256([payload.event_key for payload in payloads]) != record.get(
        "event_keys_sha256"
    ):
        raise ValueError("Payload v2 event ordering changed")
    if semantic_sha256([payload.payload_artifact_id for payload in payloads]) != record.get(
        "payload_artifact_ids_sha256"
    ):
        raise ValueError("Payload v2 artifact ordering changed")
    return payloads


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05)


class ByteBoundedPayloadV2Writer:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_shard_bytes: int,
        archive_identity: Mapping[str, Any],
    ) -> None:
        if int(max_shard_bytes) <= 0:
            raise ValueError("Payload v2 max_shard_bytes must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.max_shard_bytes = int(max_shard_bytes)
        self.archive_identity = dict(archive_identity)
        self.archive_sha = semantic_sha256(self.archive_identity)
        self.pending: list[S5GraphPayloadV2] = []
        self.pending_bytes = 0
        self.chunks: list[dict[str, Any]] = []
        self.event_keys: list[str] = []
        self.artifact_ids: list[str] = []
        self.closed = False
        manifest_path = self.output_dir / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("format") != PAYLOAD_V2_FORMAT:
                raise ValueError("existing Payload v2 manifest format changed")
            if manifest.get("archive_identity") != self.archive_identity or manifest.get(
                "archive_identity_sha256"
            ) != self.archive_sha:
                raise ValueError("existing Payload v2 archive identity changed")
            for raw_record in manifest.get("shards", []):
                record = dict(raw_record)
                record["path"] = str((manifest_path.parent / record["path"]).resolve())
                record["sidecar"] = str((manifest_path.parent / record["sidecar"]).resolve())
                payloads = verify_payload_v2_shard_record(
                    record, expected_archive_identity_sha256=self.archive_sha
                )
                self.chunks.append(dict(raw_record))
                self.event_keys.extend(payload.event_key for payload in payloads)
                self.artifact_ids.extend(payload.payload_artifact_id for payload in payloads)
            if len(set(self.event_keys)) != len(self.event_keys):
                raise ValueError("existing Payload v2 archive contains duplicate events")
            if manifest.get("event_keys_sha256") != semantic_sha256(self.event_keys):
                raise ValueError("existing Payload v2 event-order digest changed")
            if manifest.get("payload_artifact_ids_sha256") != semantic_sha256(
                self.artifact_ids
            ):
                raise ValueError("existing Payload v2 artifact-order digest changed")
            self.closed = manifest.get("status") == "complete"

    def append(self, payload: S5GraphPayloadV2) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a completed Payload v2 archive")
        payload.validate(
            n_layers=int(self.archive_identity["n_layers"]),
            d_feature=int(self.archive_identity["d_feature"]),
            archive_identity_sha256=self.archive_sha,
        )
        if payload.event_key in set(self.event_keys) or any(
            item.event_key == payload.event_key for item in self.pending
        ):
            raise ValueError("Payload v2 archive received a duplicate event")
        estimated = payload.approximate_bytes()
        if estimated > self.max_shard_bytes:
            raise ValueError("one Payload v2 event exceeds the byte cap")
        if self.pending and self.pending_bytes + estimated > self.max_shard_bytes:
            self.flush()
        self.pending.append(payload)
        self.pending_bytes += estimated

    def flush(self) -> None:
        if not self.pending:
            return
        index = len(self.chunks)
        destination = self.output_dir / f"payload_shard_{index:05d}.safetensors"
        record = write_payload_v2_shard(
            destination,
            self.pending,
            archive_identity=self.archive_identity,
        )
        relative = {
            **record,
            "path": destination.name,
            "sidecar": _sidecar(destination).name,
        }
        self.chunks.append(relative)
        self.event_keys.extend(payload.event_key for payload in self.pending)
        self.artifact_ids.extend(payload.payload_artifact_id for payload in self.pending)
        self.pending = []
        self.pending_bytes = 0
        self._write_manifest(status="running")

    def _write_manifest(self, *, status: str) -> None:
        if status not in {"running", "complete"}:
            raise ValueError("Payload v2 manifest status is invalid")
        value = {
            "format": PAYLOAD_V2_FORMAT,
            "schema_version": PAYLOAD_V2_SCHEMA_VERSION,
            "status": status,
            "archive_identity": self.archive_identity,
            "archive_identity_sha256": self.archive_sha,
            "shards": self.chunks,
            "total_events": len(self.event_keys),
            "total_bytes": sum(int(record["bytes"]) for record in self.chunks),
            "event_keys_sha256": semantic_sha256(self.event_keys),
            "payload_artifact_ids_sha256": semantic_sha256(self.artifact_ids),
        }
        _atomic_json(self.output_dir / "manifest.json", value)

    def close(self) -> Mapping[str, Any]:
        if not self.closed:
            self.flush()
            self._write_manifest(status="complete")
            self.closed = True
        return json.loads((self.output_dir / "manifest.json").read_text(encoding="utf-8"))

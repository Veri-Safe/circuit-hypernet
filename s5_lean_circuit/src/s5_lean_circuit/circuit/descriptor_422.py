"""Additive 422D view: CodeCircuit-style25D prefix plus immutable raw397D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .descriptor_legacy25 import (
    LEGACY25_COORDINATES,
    LEGACY25_DIM,
    LEGACY25_SCHEMA_SHA256,
    LEGACY25_LOG1P_INDICES,
    LEGACY25_SIGNED_LOG1P_INDICES,
    Legacy25Computation,
    Legacy25Diagnostics,
    compute_legacy25,
    fixed_log1p_legacy25,
)
from .descriptor_v2_397 import (
    DESCRIPTOR_397_SCHEMA_SHA256,
    DESCRIPTOR_DIM,
    SCHEMA_397,
    vectorize_payload_v2_397,
)
from .payload_v2 import S5GraphPayloadV2, semantic_sha256


DESCRIPTOR422_FORMAT = "s5_codecircuit25_plus_s5_descriptor397_v1"
DESCRIPTOR422_VERSION = 1
DESCRIPTOR422_DIM = LEGACY25_DIM + DESCRIPTOR_DIM

@dataclass(frozen=True)
class Descriptor422Schema:
    format: str = DESCRIPTOR422_FORMAT
    version: int = DESCRIPTOR422_VERSION

    def identity(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "dtype": "float32",
            "dimension": DESCRIPTOR422_DIM,
            "prefix": {
                "name": "legacy25",
                "dimension": LEGACY25_DIM,
                "schema_sha256": LEGACY25_SCHEMA_SHA256,
                "coordinates": list(LEGACY25_COORDINATES),
            },
            "tail": {
                "name": "descriptor397",
                "dimension": DESCRIPTOR_DIM,
                "schema_sha256": DESCRIPTOR_397_SCHEMA_SHA256,
                "block_order": list(SCHEMA_397.block_order if hasattr(SCHEMA_397, "block_order") else ("global", "layer_nodes", "layer_flow", "topology", "position")),
            },
            "concatenation": "vector422[:25]=legacy25; vector422[25:]=raw397",
            "source": "one_S5GraphPayloadV2_event",
            "condition_safety": "inherits_legacy25_and_descriptor397_boundaries",
        }

    @property
    def sha256(self) -> str:
        return semantic_sha256(self.identity())

    @property
    def coordinate_names(self) -> tuple[str, ...]:
        return tuple(f"legacy25.{name}" for name in LEGACY25_COORDINATES) + tuple(
            f"descriptor397.{name}" for name in SCHEMA_397.coordinate_names
        )


SCHEMA_422 = Descriptor422Schema()
DESCRIPTOR422_SCHEMA_SHA256 = (
    "09960b93103332356dffba7e6260b90006827ee9cdb7c4a9d6c4c827bdb8c0c7"
)
if SCHEMA_422.sha256 != DESCRIPTOR422_SCHEMA_SHA256:
    raise RuntimeError("frozen 422 descriptor schema identity changed")


@dataclass(frozen=True)
class Descriptor422Record:
    event_key: str
    payload_artifact_id: str
    schema_sha256: str
    legacy25: Tensor
    raw397: Tensor
    diagnostics: Legacy25Diagnostics

    def raw(self) -> Tensor:
        return torch.cat((self.legacy25, self.raw397)).contiguous()

    def validate(self) -> None:
        if (
            len(self.event_key) != 64
            or len(self.payload_artifact_id) != 64
            or any(char not in "0123456789abcdef" for char in self.event_key)
            or any(char not in "0123456789abcdef" for char in self.payload_artifact_id)
        ):
            raise ValueError("422 source identities must be SHA256 strings")
        if self.schema_sha256 != SCHEMA_422.sha256:
            raise ValueError("422 schema identity changed")
        for name, value, shape in (
            ("legacy25", self.legacy25, (LEGACY25_DIM,)),
            ("raw397", self.raw397, (DESCRIPTOR_DIM,)),
        ):
            if (
                value.device.type != "cpu"
                or value.dtype != torch.float32
                or value.shape != shape
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"422 {name} must be finite CPU FP32{shape}")
        if self.raw().shape != (DESCRIPTOR422_DIM,) or not bool(torch.isfinite(self.raw()).all()):
            raise ValueError("422 concatenation is invalid")


def compose_vector422(
    legacy25: Tensor | np.ndarray,
    raw397: Tensor | np.ndarray,
    *,
    expected_event_key: str | None = None,
    expected_payload_artifact_id: str | None = None,
) -> Tensor:
    """Concatenate two independently validated raw views without reweighting."""

    prefix = torch.as_tensor(legacy25, dtype=torch.float32).detach().cpu().contiguous()
    tail = torch.as_tensor(raw397, dtype=torch.float32).detach().cpu().contiguous()
    if prefix.shape != (LEGACY25_DIM,) or tail.shape != (DESCRIPTOR_DIM,):
        raise ValueError("compose_vector422 expects FP32-compatible [25] and [397]")
    if not bool(torch.isfinite(prefix).all()) or not bool(torch.isfinite(tail).all()):
        raise ValueError("compose_vector422 inputs must be finite")
    # Optional identity arguments make accidental detached rows visible to
    # callers without putting those identities into the numeric condition.
    for name, value in (
        ("event_key", expected_event_key),
        ("payload_artifact_id", expected_payload_artifact_id),
    ):
        if value is not None and (len(str(value)) != 64 or any(c not in "0123456789abcdef" for c in str(value))):
            raise ValueError(f"{name} must be a lowercase SHA256")
    return torch.cat((prefix, tail)).contiguous()


def vectorize_payload_v2_422_record(
    payload: S5GraphPayloadV2,
    *,
    d_feature: int,
    n_layers: int = 8,
    archive_identity_sha256: str | None = None,
    archived_raw397: Tensor | np.ndarray | None = None,
) -> Descriptor422Record:
    """Compute legacy25 and append raw397, optionally checking archived397 parity."""

    legacy: Legacy25Computation = compute_legacy25(
        payload,
        n_layers=n_layers,
        d_feature=d_feature,
        archive_identity_sha256=archive_identity_sha256,
    )
    computed_raw397 = vectorize_payload_v2_397(
        payload,
        n_layers=n_layers,
        d_feature=d_feature,
        archive_identity_sha256=archive_identity_sha256,
    )
    if archived_raw397 is None:
        raw397 = computed_raw397
    else:
        raw397 = torch.as_tensor(archived_raw397, dtype=torch.float32).detach().cpu().contiguous()
        if raw397.shape != (DESCRIPTOR_DIM,) or not bool(torch.isfinite(raw397).all()):
            raise ValueError("archived raw397 is invalid")
        if not torch.equal(raw397, computed_raw397):
            raise ValueError("archived raw397 does not match the retained Payload")
    record = Descriptor422Record(
        event_key=payload.event_key,
        payload_artifact_id=payload.payload_artifact_id,
        schema_sha256=SCHEMA_422.sha256,
        legacy25=legacy.vector,
        raw397=raw397,
        diagnostics=legacy.diagnostics,
    )
    record.validate()
    return record


def vectorize_payload_v2_422(
    payload: S5GraphPayloadV2,
    *,
    d_feature: int,
    n_layers: int = 8,
    archive_identity_sha256: str | None = None,
    archived_raw397: Tensor | np.ndarray | None = None,
) -> Tensor:
    """Return one finite CPU FP32[422] vector for a validated Payload v2 event."""

    return vectorize_payload_v2_422_record(
        payload,
        d_feature=d_feature,
        n_layers=n_layers,
        archive_identity_sha256=archive_identity_sha256,
        archived_raw397=archived_raw397,
    ).raw()


def fixed_log1p_422(
    vector: Tensor | np.ndarray,
    *,
    diagnostics: Legacy25Diagnostics | None = None,
) -> Tensor:
    """Apply the separately declared fixed transform to an FP32[422] vector."""

    value = torch.as_tensor(vector, dtype=torch.float32).detach().cpu().contiguous()
    if value.shape != (DESCRIPTOR422_DIM,) or not bool(torch.isfinite(value).all()):
        raise ValueError("fixed_log1p_422 expects finite FP32[422]")
    prefix = fixed_log1p_legacy25(value[:LEGACY25_DIM], diagnostics=diagnostics)
    from .descriptor_v2_397 import fixed_log1p_397

    tail = fixed_log1p_397(value[LEGACY25_DIM:])
    return torch.cat((prefix, tail)).contiguous()


def vectorize_many_payload_v2_422(
    payloads: Sequence[S5GraphPayloadV2],
    *,
    d_feature: int,
    n_layers: int = 8,
    archive_identity_sha256: str | None = None,
) -> Tensor:
    if not payloads:
        return torch.empty((0, DESCRIPTOR422_DIM), dtype=torch.float32)
    return torch.stack(
        [
            vectorize_payload_v2_422(
                payload,
                d_feature=d_feature,
                n_layers=n_layers,
                archive_identity_sha256=archive_identity_sha256,
            )
            for payload in payloads
        ],
    ).contiguous()

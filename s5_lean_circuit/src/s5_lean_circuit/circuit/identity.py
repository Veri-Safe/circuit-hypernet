"""Portable semantic identities for durable S5 Circuit artifacts."""

from __future__ import annotations

from typing import Any, Mapping


MODEL_ARTIFACT_IDENTITY_KEYS = (
    "checkpoint_sha256",
    "checkpoint_step",
    "checkpoint_config_sha256",
    "current_architecture_config_sha256",
    "dtype",
)


def model_artifact_identity(runtime_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime-only fields from durable model provenance."""

    missing = [
        key for key in MODEL_ARTIFACT_IDENTITY_KEYS if key not in runtime_identity
    ]
    if missing:
        raise ValueError(f"source model artifact identity is missing fields: {missing}")
    identity = {
        key: runtime_identity[key] for key in MODEL_ARTIFACT_IDENTITY_KEYS
    }
    for key in (
        "checkpoint_sha256",
        "checkpoint_config_sha256",
        "current_architecture_config_sha256",
    ):
        if len(str(identity[key])) != 64:
            raise ValueError(f"source model artifact identity has invalid {key}")
    return identity

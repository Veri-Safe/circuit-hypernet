"""Explicit Vector422 views over immutable historical Vector397 data/model assets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from .composition_loco import EvalBank, load_eval_bank
from .composition_loco_vector422_spec import VECTOR422_MIGRATION, VECTOR422_PROTOCOL_ID


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def expand_circuit_input_422(core: Any) -> dict[str, Any]:
    """Zero-extend a strictly loaded 397-D A-P03; preserve every other weight.

    Only fresh atomic initialization uses this conversion. Resume loading is
    strict into the already expanded architecture and never resets the prefix.
    """
    encoder = core.circuit_encoder
    old = encoder.network[0]
    if not isinstance(old, nn.Linear) or old.in_features != 397 or old.out_features != 2048:
        raise ValueError("Vector422 migration requires the original Candidate-A 397->2048 input")
    if core.candidate != "A" or core.condition_source != "circuit" or core.fusion_mode != "append":
        raise ValueError("Vector422 supports only Candidate-A append Circuit HyperNet")
    # Keep the model/optimizer RNG stream identical to the historical run.
    devices = [old.weight.device] if old.weight.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        expanded = nn.Linear(422, 2048, bias=old.bias is not None).to(
            device=old.weight.device, dtype=old.weight.dtype
        )
    with torch.no_grad():
        expanded.weight.zero_()
        expanded.weight[:, 25:].copy_(old.weight)
        if old.bias is not None:
            expanded.bias.copy_(old.bias)
    expanded.weight.requires_grad_(old.weight.requires_grad)
    if old.bias is not None:
        expanded.bias.requires_grad_(old.bias.requires_grad)
    encoder.network[0] = expanded
    encoder.input_dim = 422
    core.circuit_input_dim = 422
    receipt = {
        "format": VECTOR422_MIGRATION,
        "source_input_dim": 397,
        "input_dim": 422,
        "prefix_columns": [0, 25],
        "tail_columns": [25, 422],
        "prefix_initialization": "zeros",
        "tail_initialization": "original_atomic_checkpoint",
        "added_trainable_scalars": 25 * 2048,
    }
    core.__dict__["_vector422_migration"] = receipt
    return receipt


def _sidecar(config: Any, key: str, source_path: Any, payload: Mapping[str, Tensor]):
    from .hypernet.vector422_source import load_vector422_sidecar

    if config.protocol_id != VECTOR422_PROTOCOL_ID:
        raise ValueError("Vector422 condition sources require their explicit protocol")
    return load_vector422_sidecar(
        _repo_path(config.data["vector422_sources"][key]),
        source_path=_repo_path(source_path),
        vectors=payload["vectors"],
        event_keys=payload.get("event_key"),
        payload_artifact_ids=payload.get("payload_artifact_id"),
    )


class Vector422DatasetView:
    """Delegate frozen text/fold identities and replace only condition lookup."""

    def __init__(self, dataset: Any, sidecar: Any) -> None:
        self._dataset = dataset
        self._vector422_source = sidecar
        self.vector422_identity = dict(sidecar.identity)
        if sidecar.size != dataset.payload["vectors"].shape[0]:
            raise ValueError("Vector422 sidecar canonical coverage differs from its cache")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def __len__(self) -> int:
        return len(self._dataset)

    def condition_vectors(self, canonical_indices: Tensor) -> Tensor:
        return self._vector422_source.condition_vectors(canonical_indices)


def vector422_dataset_view(config: Any, dataset: Any, key: str) -> Vector422DatasetView:
    source_path = config.data["cache_path"] if key == "composition" else config.data["atomic_replay"]["cache_path"]
    return Vector422DatasetView(dataset, _sidecar(config, key, source_path, dataset.payload))


def load_vector422_eval_bank(config: Any, key: str) -> EvalBank:
    """Load the audited prompt bank, binding its separate 422 condition overlay."""
    if key not in {"validation", "atomic_bank"}:
        raise ValueError("Vector422 final evaluation is not authorized")
    path_key = "validation_bank" if key == "validation" else "atomic_bank"
    source = _repo_path(config.data[path_key])
    original = load_eval_bank(source)
    sidecar = _sidecar(config, key, source, original.payload)
    payload = dict(original.payload)
    payload["vectors"] = sidecar.condition_vectors(torch.arange(sidecar.size))
    metadata = dict(original.metadata)
    # The historical identity continues to identify the unchanged source file.
    # The additional condition identity binds the runtime overlay explicitly.
    metadata["condition_view"] = {
        "format": "s5_composition_loco_vector422_eval_view_v1",
        "condition_dim": 422,
        "source_bank_identity_sha256": original.metadata["identity_sha256"],
        "vector422_source": dict(sidecar.identity),
    }
    return EvalBank(metadata=metadata, payload=payload, records=original.records)


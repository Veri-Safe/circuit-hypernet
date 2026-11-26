from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ImportedWeights:
    state_dict: dict[str, Tensor]
    source_keys: dict[str, str]
    source_path: str


class ReLUTranscoder(nn.Module):
    """CodeCircuit-compatible MLP input-to-output ReLU transcoder."""

    def __init__(self, d_model: int = 2560, d_feature: int = 163_840) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.d_feature = int(d_feature)
        self.W_enc = nn.Parameter(torch.empty(self.d_feature, self.d_model))
        self.b_enc = nn.Parameter(torch.empty(self.d_feature))
        self.W_dec = nn.Parameter(torch.empty(self.d_feature, self.d_model))
        self.b_dec = nn.Parameter(torch.empty(self.d_model))

    def encode(self, x: Tensor) -> Tensor:
        return torch.relu(F.linear(x, self.W_enc, self.b_enc))

    def decode(self, features: Tensor) -> Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encode(x)
        return self.decode(features), features

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Tensor]) -> "ReLUTranscoder":
        d_feature, d_model = state["W_enc"].shape
        with torch.device("meta"):
            model = cls(d_model=d_model, d_feature=d_feature)
        model.load_state_dict(dict(state), strict=True, assign=True)
        return model


def _choose_key(
    tensors: Mapping[str, Tensor], candidates: tuple[str, ...]
) -> tuple[str, Tensor] | None:
    for key in candidates:
        if key in tensors:
            return key, tensors[key]
    return None


def import_public_weights(
    path: str | Path,
    *,
    layer: int,
    d_model: int = 2560,
    d_feature: int = 163_840,
) -> ImportedWeights:
    from safetensors.torch import load_file

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Public transcoder layer is missing: {source}")
    tensors = load_file(str(source), device="cpu")
    candidates = {
        "W_enc": (
            f"W_enc_{layer}",
            "W_enc",
            "encoder.weight",
            "encoder.W",
            "w_enc",
        ),
        "b_enc": (f"b_enc_{layer}", "b_enc", "encoder.bias"),
        "W_dec": (
            f"W_dec_{layer}",
            "W_dec",
            "decoder.weight",
            "decoder.W",
            "w_dec",
        ),
        "b_dec": (f"b_dec_{layer}", "b_dec", "decoder.bias"),
    }
    selected: dict[str, Tensor] = {}
    source_keys: dict[str, str] = {}
    for canonical, names in candidates.items():
        result = _choose_key(tensors, names)
        if result is None:
            raise ValueError(f"Missing {canonical} in {source}; keys={sorted(tensors)}")
        source_key, tensor = result
        selected[canonical] = tensor
        source_keys[canonical] = source_key

    for forbidden in ("threshold", "thresholds", "W_skip", "w_skip"):
        if any(forbidden.lower() in key.lower() for key in tensors):
            raise ValueError(f"Public layer has unsupported {forbidden} parameter")

    if selected["W_enc"].shape == (d_model, d_feature):
        selected["W_enc"] = selected["W_enc"].T
    if selected["W_dec"].shape == (d_model, d_feature):
        selected["W_dec"] = selected["W_dec"].T
    expected_shapes = {
        "W_enc": (d_feature, d_model),
        "b_enc": (d_feature,),
        "W_dec": (d_feature, d_model),
        "b_dec": (d_model,),
    }
    mismatches = {
        key: {"expected": shape, "actual": tuple(selected[key].shape)}
        for key, shape in expected_shapes.items()
        if tuple(selected[key].shape) != shape
    }
    if mismatches:
        raise ValueError(f"Public transcoder shape mismatch: {mismatches}")
    state = {key: value.float().contiguous() for key, value in selected.items()}
    return ImportedWeights(state, source_keys, str(source))


def load_public_transcoder(
    path: str | Path,
    *,
    layer: int,
    d_model: int = 2560,
    d_feature: int = 163_840,
) -> tuple[ReLUTranscoder, ImportedWeights]:
    imported = import_public_weights(
        path, layer=layer, d_model=d_model, d_feature=d_feature
    )
    return ReLUTranscoder.from_state_dict(imported.state_dict), imported


def canonicalize_transcoder(
    transcoder: ReLUTranscoder,
    *,
    min_norm: float = 1e-8,
) -> tuple[ReLUTranscoder, Tensor]:
    with torch.no_grad():
        norms = torch.linalg.vector_norm(transcoder.W_dec.float(), dim=1)
        invalid = ~torch.isfinite(norms) | norms.le(float(min_norm))
        if invalid.any():
            ids = invalid.nonzero(as_tuple=False).flatten().tolist()
            preview = ids[:20]
            raise ValueError(
                f"Cannot canonicalize {len(ids)} decoder rows; feature ids={preview}"
            )
        canonical = ReLUTranscoder(transcoder.d_model, transcoder.d_feature).to(
            device=transcoder.W_enc.device,
            dtype=transcoder.W_enc.dtype,
        )
        canonical.W_enc.copy_(transcoder.W_enc * norms[:, None])
        canonical.b_enc.copy_(transcoder.b_enc * norms)
        canonical.W_dec.copy_(transcoder.W_dec / norms[:, None])
        canonical.b_dec.copy_(transcoder.b_dec)
    return canonical, norms


def forward_in_chunks(
    transcoder: ReLUTranscoder,
    x: Tensor,
    *,
    chunk_size: int,
) -> tuple[Tensor, Tensor]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    outputs = []
    features = []
    for start in range(0, x.shape[0], chunk_size):
        output, feature = transcoder(x[start : start + chunk_size])
        outputs.append(output)
        features.append(feature)
    return torch.cat(outputs), torch.cat(features)

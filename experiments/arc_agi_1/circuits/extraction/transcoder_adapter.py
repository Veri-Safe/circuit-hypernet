from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from s3circuit.graph_extract import ensure_codecircuit_on_path
from s3circuit.nvarc_transcoder.config import NvarcTranscoderConfig
from s3circuit.nvarc_transcoder.identity import (
    config_semantic_sha256_for_root,
    sha256_file,
)


EXPORT_FORMAT = "nvarc_transcoder_36_layer_export_v1"


@dataclass(frozen=True)
class CanonicalLayer:
    layer: int
    root: Path
    weight_path: Path
    weight_sha256: str


@dataclass(frozen=True)
class CanonicalTranscoderBank:
    manifest_path: Path
    manifest: dict[str, Any]
    layers: tuple[CanonicalLayer, ...]

    @property
    def identity_sha256(self) -> str:
        return sha256_file(self.manifest_path)


def _resolve_layer_root(manifest_path: Path, value: str, layer: int) -> Path:
    declared = Path(value)
    candidates = [declared]
    if not declared.is_absolute():
        candidates.append(manifest_path.parent / declared)
    candidates.extend(
        (
            manifest_path.parent / f"layer_{layer:02d}",
            manifest_path.parent / "canonical" / f"layer_{layer:02d}",
        )
    )
    for candidate in candidates:
        if (candidate / "transcoder.safetensors").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Cannot resolve canonical transcoder layer {layer} from {manifest_path}"
    )


def _config_hash_matches_mount_alias(
    *,
    manifest_path: Path,
    manifest: dict[str, Any],
    stage_a_config: NvarcTranscoderConfig,
) -> bool:
    root_value = os.environ.get("HC_ROOT")
    layers = manifest.get("layers")
    if not root_value or not isinstance(layers, list) or not layers:
        return False
    declared = Path(str(layers[0].get("canonical_snapshot", "")))
    if not declared.is_absolute():
        return False
    current_root = Path(root_value).resolve()
    current_layer = (manifest_path.parent / "layer_00").absolute()
    try:
        relative = current_layer.relative_to(current_root)
    except ValueError:
        try:
            relative = current_layer.resolve().relative_to(current_root)
        except ValueError:
            return False
    if tuple(declared.parts[-len(relative.parts) :]) != tuple(relative.parts):
        return False
    cached_root = declared
    for _ in relative.parts:
        cached_root = cached_root.parent
    expected = config_semantic_sha256_for_root(
        stage_a_config,
        current_root=current_root,
        identity_root=cached_root,
    )
    return manifest.get("config_sha256") == expected


def _validate_canonical_weight_metadata(
    root: Path,
    *,
    layer: int,
    d_model: int = 2560,
    d_feature: int = 163_840,
) -> dict[str, Any]:
    if not (root / "COMPLETE").is_file():
        raise ValueError(f"Canonical layer {layer} is incomplete: {root}")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical layer {layer} manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest = {
        "format": "nvarc_transcoder_canonical_v1",
        "layer": layer,
        "status": "complete",
    }
    manifest_mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if manifest_mismatches:
        raise ValueError(
            f"Canonical layer {layer} manifest mismatch: {manifest_mismatches}"
        )

    weight_path = root / "transcoder.safetensors"
    expected_shapes = {
        "W_enc": (d_feature, d_model),
        "b_enc": (d_feature,),
        "W_dec": (d_feature, d_model),
        "b_dec": (d_model,),
    }
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        actual_keys = set(handle.keys())
        if actual_keys != set(expected_shapes):
            raise ValueError(
                f"Canonical layer {layer} tensor keys changed: "
                f"missing={sorted(set(expected_shapes) - actual_keys)}, "
                f"unexpected={sorted(actual_keys - set(expected_shapes))}"
            )
        actual_shapes = {
            key: tuple(handle.get_slice(key).get_shape()) for key in expected_shapes
        }
    shape_mismatches = {
        key: {"expected": expected, "actual": actual_shapes[key]}
        for key, expected in expected_shapes.items()
        if actual_shapes[key] != expected
    }
    if shape_mismatches:
        raise ValueError(
            f"Canonical layer {layer} tensor shape mismatch: {shape_mismatches}"
        )
    return manifest


def load_canonical_bank(
    manifest_path: str | Path,
    *,
    stage_a_config: NvarcTranscoderConfig,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> CanonicalTranscoderBank:
    path = Path(manifest_path).absolute()
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "format": EXPORT_FORMAT,
        "num_layers": 36,
        "status": "complete",
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Canonical transcoder export mismatch: {mismatches}")
    if value.get("config_sha256") != stage_a_config.semantic_sha256 and not (
        _config_hash_matches_mount_alias(
            manifest_path=path,
            manifest=value,
            stage_a_config=stage_a_config,
        )
    ):
        raise ValueError("Canonical transcoder export config identity changed")
    layer_values = value.get("layers")
    if not isinstance(layer_values, list) or len(layer_values) != 36:
        raise ValueError("Canonical export must contain exactly 36 layers")
    layers = []
    for expected_layer, layer_value in enumerate(layer_values):
        if int(layer_value.get("layer", -1)) != expected_layer:
            raise ValueError("Canonical export layers are not ordered 0..35")
        root = _resolve_layer_root(
            path, str(layer_value["canonical_snapshot"]), expected_layer
        )
        weight_path = root / "transcoder.safetensors"
        actual_sha = sha256_file(weight_path)
        if actual_sha != layer_value.get("canonical_sha256"):
            raise ValueError(f"Canonical layer {expected_layer} checksum changed")
        _validate_canonical_weight_metadata(root, layer=expected_layer)
        layers.append(CanonicalLayer(expected_layer, root, weight_path, actual_sha))
        if progress is not None:
            progress(
                {
                    "layer": expected_layer,
                    "layers_complete": expected_layer + 1,
                    "layers_total": len(layer_values),
                    "weight_bytes": weight_path.stat().st_size,
                    "weight_sha256": actual_sha,
                }
            )
    return CanonicalTranscoderBank(path, value, tuple(layers))


def build_codecircuit_transcoder_set(
    bank: CanonicalTranscoderBank,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    lazy_encoder: bool = True,
    lazy_decoder: bool = True,
) -> Any:
    ensure_codecircuit_on_path()
    from circuit_tracer.transcoder.single_layer_transcoder import (
        TranscoderSet,
        load_relu_transcoder,
    )

    transcoders = {
        layer.layer: load_relu_transcoder(
            str(layer.weight_path),
            layer.layer,
            device=device,
            dtype=dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
        for layer in bank.layers
    }
    return TranscoderSet(
        transcoders,
        feature_input_hook="mlp.hook_in",
        feature_output_hook="mlp.hook_out",
        scan=f"{_scan_label(bank)}:{bank.identity_sha256}",
    )


def _scan_label(bank: CanonicalTranscoderBank) -> str:
    if bank.manifest.get("format") == "arc2_clean_devbest_transcoder_bank_v1":
        return "nvarc-arc2-clean-devbest-step3500"
    normalized = str(bank.manifest_path).replace("\\", "/")
    if "/rearc128_long_s42_step1400_outside_b0_b1/" in normalized:
        return "nvarc-rearc128-long-s42-step1400-outside-b0-b1"
    if "/rearc8_union_fast128/" in normalized:
        return "nvarc-rearc8-s42-step130"
    # Preserve the established identity for every existing step399 manifest.
    return "nvarc-rearc-step399"

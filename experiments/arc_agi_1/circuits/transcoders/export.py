from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .checkpoint import load_weight_snapshot, save_weight_snapshot
from .config import NvarcTranscoderConfig
from .identity import config_semantic_sha256_for_root, semantic_sha256, sha256_file
from .model import ReLUTranscoder, canonicalize_transcoder
from .selection import _resolve_repo_artifact_path
from .traces import write_json_atomic


_ACTIVATION_BOUNDARY_TOLERANCE = 1e-5


def _identity_root_from_repo_artifact(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        return None
    parts = path.parts
    for marker in ("outputs", "logs"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0:
                return Path(*parts[:index]).resolve()
    return None


def _config_sha256_matches_manifest(
    config: NvarcTranscoderConfig,
    *,
    actual: str | None,
    artifact_path: str | None,
) -> bool:
    if actual == config.semantic_sha256:
        return True
    root_value = os.environ.get("HC_ROOT")
    identity_root = _identity_root_from_repo_artifact(artifact_path)
    if not root_value or identity_root is None:
        return False
    expected = config_semantic_sha256_for_root(
        config,
        current_root=Path(root_value).resolve(),
        identity_root=identity_root,
    )
    return actual == expected


def _parity_check(
    raw: ReLUTranscoder,
    canonical: ReLUTranscoder,
    *,
    seed: int = 24,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn((32, raw.d_model), generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        raw_preactivations = F.linear(x, raw.W_enc, raw.b_enc)
        canonical_preactivations = F.linear(x, canonical.W_enc, canonical.b_enc)
        raw_features = torch.relu(raw_preactivations)
        canonical_features = torch.relu(canonical_preactivations)
        raw_y = raw_features @ raw.W_dec + raw.b_dec
        canonical_y = canonical_features @ canonical.W_dec + canonical.b_dec
    max_abs = float((raw_y - canonical_y).abs().max())
    scale = float(raw_y.abs().max())
    tolerance = 1e-5 + 1e-5 * scale
    raw_active = raw_features.gt(0)
    canonical_active = canonical_features.gt(0)
    active_mismatch = raw_active != canonical_active
    boundary_mismatch = (
        active_mismatch
        & raw_preactivations.abs().le(_ACTIVATION_BOUNDARY_TOLERANCE)
        & canonical_preactivations.abs().le(_ACTIVATION_BOUNDARY_TOLERANCE)
    )
    disallowed_active_mismatch = active_mismatch & ~boundary_mismatch
    active_exact_equal = not bool(active_mismatch.any())
    active_equal = not bool(disallowed_active_mismatch.any())
    if max_abs > tolerance:
        raise ValueError(
            f"Canonical forward parity failed: max_abs={max_abs}, tolerance={tolerance}"
        )
    if not active_equal:
        raise ValueError(
            "Canonical export changed active feature IDs outside ReLU boundary "
            f"tolerance {_ACTIVATION_BOUNDARY_TOLERANCE}: "
            f"count={int(disallowed_active_mismatch.sum())}"
        )
    return {
        "fixed_input_seed": seed,
        "fixed_input_rows": int(x.shape[0]),
        "forward_max_abs_error": max_abs,
        "forward_tolerance": tolerance,
        "active_feature_ids_exact_equal": active_exact_equal,
        "active_feature_ids_equal": active_equal,
        "active_feature_boundary_tolerance": _ACTIVATION_BOUNDARY_TOLERANCE,
        "active_feature_boundary_mismatch_count": int(boundary_mismatch.sum()),
        "active_feature_disallowed_mismatch_count": int(
            disallowed_active_mismatch.sum()
        ),
    }


def _canonicalize_snapshot(
    *,
    source: Path,
    destination: Path,
    config: NvarcTranscoderConfig,
    selection: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    raw, raw_manifest = load_weight_snapshot(source)
    norms = torch.linalg.vector_norm(raw.W_dec.float(), dim=1)
    invalid = ~torch.isfinite(norms) | norms.le(config.model.decoder_norm_min)
    if invalid.any():
        ids = invalid.nonzero(as_tuple=False).flatten()
        raise ValueError(
            json.dumps(
                {
                    "reason": "invalid_decoder_norm",
                    "feature_ids": ids.tolist(),
                    "norms": norms[ids].tolist(),
                },
                sort_keys=True,
            )
        )
    canonical, canonical_norms = canonicalize_transcoder(
        raw, min_norm=config.model.decoder_norm_min
    )
    parity = _parity_check(raw, canonical)
    manifest = {
        "format": "nvarc_transcoder_canonical_v1",
        "layer": int(selection["layer"]),
        "config_sha256": config.semantic_sha256,
        "selection_sha256": semantic_sha256(selection),
        "selection": selection,
        "source_role": role,
        "source_snapshot": str(source),
        "source_snapshot_sha256": sha256_file(source / "transcoder.safetensors"),
        "source_manifest": raw_manifest,
        "decoder_norm_min": float(canonical_norms.min()),
        "decoder_norm_max": float(canonical_norms.max()),
        "decoder_norm_mean": float(canonical_norms.mean()),
        "parity": parity,
        "status": "complete",
    }
    save_weight_snapshot(destination, transcoder=canonical, manifest=manifest)
    reloaded, reloaded_manifest = load_weight_snapshot(destination)
    reload_parity = _parity_check(canonical, reloaded)
    if reloaded_manifest != manifest:
        raise ValueError("Canonical circuit-tracer reload changed its manifest")
    result = {
        **manifest,
        "canonical_snapshot": str(destination),
        "canonical_sha256": sha256_file(destination / "transcoder.safetensors"),
        "reload_parity": reload_parity,
    }
    write_json_atomic(destination / "export_manifest.json", result)
    return result


def export_selected_candidate(
    config: NvarcTranscoderConfig,
    *,
    selection_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    selection_file = Path(selection_path).resolve()
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if selection.get("format") != "nvarc_transcoder_selection_v1":
        raise ValueError(f"Unsupported selection manifest: {selection_file}")
    if selection.get("status") != "complete":
        raise ValueError(f"Selection manifest is incomplete: {selection_file}")
    layer = int(selection.get("layer", -1))
    if not 0 <= layer < config.model.num_layers:
        raise ValueError(f"Selection layer is outside the configured model: {layer}")
    destination_path = Path(destination).resolve()
    failures = []
    for role in ("best", "runner_up"):
        selected = selection.get(role)
        if selected is None:
            continue
        source = _resolve_repo_artifact_path(selected["snapshot"])
        try:
            return _canonicalize_snapshot(
                source=source,
                destination=destination_path,
                config=config,
                selection=selection,
                role=role,
            )
        except (ValueError, RuntimeError) as error:
            failures.append({"role": role, "snapshot": str(source), "error": str(error)})
    failure_manifest = {
        "format": "nvarc_transcoder_canonical_failure_v1",
        "layer": selection.get("layer"),
        "selection": str(selection_file),
        "failures": failures,
        "status": "failed",
    }
    write_json_atomic(destination_path.parent / f"layer_{int(selection['layer']):02d}_failure.json", failure_manifest)
    raise RuntimeError(f"Best and eligible runner-up canonical exports failed: {failures}")


def finalize_exports(
    config: NvarcTranscoderConfig,
    *,
    canonical_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(canonical_root).resolve()
    layers = []
    for layer in range(config.model.num_layers):
        layer_root = root / f"layer_{layer:02d}"
        manifest_path = layer_root / "export_manifest.json"
        if not (layer_root / "COMPLETE").is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Canonical layer is incomplete: {layer_root}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "format": "nvarc_transcoder_canonical_v1",
            "layer": layer,
            "status": "complete",
        }
        mismatches = {
            key: {"expected": value, "actual": manifest.get(key)}
            for key, value in expected.items()
            if manifest.get(key) != value
        }
        if not _config_sha256_matches_manifest(
            config,
            actual=manifest.get("config_sha256"),
            artifact_path=manifest.get("canonical_snapshot")
            or manifest.get("source_snapshot"),
        ):
            mismatches["config_sha256"] = {
                "expected": config.semantic_sha256,
                "actual": manifest.get("config_sha256"),
            }
        if mismatches:
            raise ValueError(f"Canonical layer {layer} mismatch: {mismatches}")
        actual_sha = sha256_file(layer_root / "transcoder.safetensors")
        if manifest.get("canonical_sha256") != actual_sha:
            raise ValueError(f"Canonical layer {layer} weight hash changed")
        selection = manifest.get("selection")
        if not isinstance(selection, dict) or selection.get("status") != "complete":
            raise ValueError(f"Canonical layer {layer} has no completed selection record")
        source_role = manifest["source_role"]
        selected = selection.get(source_role)
        selected_snapshot = (
            _resolve_repo_artifact_path(selected["snapshot"])
            if isinstance(selected, dict) and "snapshot" in selected
            else None
        )
        raw_source = _resolve_repo_artifact_path(manifest["source_snapshot"])
        if selected_snapshot != raw_source:
            raise ValueError(f"Canonical layer {layer} does not match its selected raw source")
        if not (raw_source / "COMPLETE").is_file():
            raise FileNotFoundError(f"Canonical layer {layer} raw source is missing")
        if sha256_file(raw_source / "transcoder.safetensors") != manifest["source_snapshot_sha256"]:
            raise ValueError(f"Canonical layer {layer} raw source hash changed")
        layers.append(
            {
                "layer": layer,
                "canonical_snapshot": str(layer_root),
                "canonical_sha256": actual_sha,
                "raw_source_snapshot": manifest["source_snapshot"],
                "raw_source_sha256": manifest["source_snapshot_sha256"],
                "source_role": manifest["source_role"],
            }
        )
    result = {
        "format": "nvarc_transcoder_36_layer_export_v1",
        "config_sha256": config.semantic_sha256,
        "layers": layers,
        "num_layers": len(layers),
        "status": "complete",
    }
    write_json_atomic(output_path, result)
    return result

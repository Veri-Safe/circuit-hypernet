"""Canonical eight-layer Transcoder bank validation for Graph extraction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch

from .model_io import sha256_file
from .payload import semantic_sha256
from .transcoder_training import TRAINING_FORMAT, load_transcoder_checkpoint


def load_canonical_transcoder_bank(
    layer_paths: Sequence[str | Path],
    *,
    device: torch.device,
    expected_config_sha256: str,
    expected_seed: int = 42,
    verified_sha256: Mapping[int | str, str] | None = None,
) -> tuple[tuple[torch.nn.Module, ...], str]:
    if len(layer_paths) != 8:
        raise ValueError("canonical S5 bank requires exactly eight layer paths")
    registry_path = Path(__file__).resolve().parents[3] / "configs/selected_transcoder_hashes.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = registry["layers"] if expected_config_sha256 == registry["recipe_sha256"] else {}
    models = []
    identities = {}
    widths = set()
    for layer, value in enumerate(layer_paths):
        path = Path(value).resolve()
        model, checkpoint = load_transcoder_checkpoint(path, device=device)
        if checkpoint.get("format") != TRAINING_FORMAT:
            raise ValueError("canonical bank checkpoint format changed")
        manifest = checkpoint.get("manifest", {})
        if int(manifest.get("layer", -1)) != layer:
            raise ValueError("canonical bank layer identity mismatch")
        if int(manifest.get("seed", -1)) != int(expected_seed):
            raise ValueError("canonical production bank must use seed 42 only")
        if manifest.get("config_sha256") != expected_config_sha256:
            # Old envelopes carry the private recipe identity. Authenticate the
            # complete frozen weights for this specific public recipe and layer.
            pinned = selected.get(str(layer))
            if pinned is None or sha256_file(path) != pinned:
                raise ValueError("canonical bank mixes circuit config identities")
        canonicalization = checkpoint.get("canonicalization", {})
        if canonicalization.get("format") != "s5_rectangular_transcoder_unit_decoder_v1":
            raise ValueError("Graph extraction requires a reviewed canonical selection")
        norms = torch.linalg.vector_norm(model.W_dec.detach().float().cpu(), dim=1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise ValueError("canonical bank decoder rows are not unit normalized")
        widths.add(model.d_feature)
        identities[str(layer)] = {
            "path": str(path),
            "sha256": (
                str(verified_sha256[layer])
                if verified_sha256 is not None and layer in verified_sha256
                else str(verified_sha256[str(layer)])
                if verified_sha256 is not None and str(layer) in verified_sha256
                else sha256_file(path)
            ),
            "learning_rate": manifest.get("learning_rate"),
            "update": checkpoint.get("update"),
            "processed_tokens": checkpoint.get("processed_tokens"),
            "selection_manifest": canonicalization.get("selection_manifest"),
        }
        models.append(model)
    if len(widths) != 1:
        raise ValueError("canonical bank mixes Transcoder widths")
    return tuple(models), semantic_sha256(identities)

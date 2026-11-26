from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from torch import Tensor

from .identity import semantic_sha256, sha256_file


def write_json_atomic(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def write_safetensors_bundle(
    path: str | Path,
    *,
    tensors: Mapping[str, Tensor],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import load_file, save_file

    destination = Path(path)
    if destination.suffix != ".safetensors":
        raise ValueError("Safetensors bundle path must end in .safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sidecar = destination.with_suffix(".json")
    marker = Path(str(destination) + ".done")
    for item in (destination, sidecar, marker):
        if item.exists():
            raise FileExistsError(f"Refusing to overwrite committed artifact: {item}")

    normalized = {
        key: value.detach().cpu().contiguous() for key, value in tensors.items()
    }
    temporary_tensor = destination.with_name(destination.name + ".tmp")
    temporary_sidecar = sidecar.with_name(sidecar.name + ".tmp")
    temporary_marker = marker.with_name(marker.name + ".tmp")
    for temporary in (temporary_tensor, temporary_sidecar, temporary_marker):
        temporary.unlink(missing_ok=True)
    save_file(normalized, str(temporary_tensor))
    with temporary_tensor.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    tensor_sha256 = sha256_file(temporary_tensor)
    sidecar_value = {
        **dict(metadata),
        "tensor_sha256": tensor_sha256,
        "tensor_shapes": {key: list(value.shape) for key, value in normalized.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in normalized.items()},
    }
    sidecar_value["metadata_sha256"] = semantic_sha256(sidecar_value)
    with temporary_sidecar.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(sidecar_value, indent=2, sort_keys=True, ensure_ascii=True)
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    reloaded = load_file(str(temporary_tensor), device="cpu")
    if set(reloaded) != set(normalized):
        raise ValueError("Safetensors bundle changed keys on reload")
    for key, expected in normalized.items():
        actual = reloaded[key]
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError(f"Safetensors reload mismatch for {key}")
    os.replace(temporary_tensor, destination)
    os.replace(temporary_sidecar, sidecar)
    with temporary_marker.open("wb") as handle:
        handle.write(b"ok\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_marker, marker)
    return sidecar_value


def load_safetensors_bundle(
    path: str | Path,
    *,
    expected_format: str | None = None,
    verify_checksum: bool = True,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    from safetensors.torch import load_file

    source = Path(path)
    sidecar = source.with_suffix(".json")
    marker = Path(str(source) + ".done")
    if (
        not source.is_file()
        or not sidecar.is_file()
        or not marker.is_file()
        or marker.read_bytes() != b"ok\n"
    ):
        raise ValueError(f"Artifact bundle is incomplete: {source}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"Artifact sidecar is not an object: {sidecar}")
    if expected_format is not None and metadata.get("format") != expected_format:
        raise ValueError(
            f"Artifact format mismatch: {metadata.get('format')} != {expected_format}"
        )
    recorded_metadata_sha = metadata.pop("metadata_sha256", None)
    actual_metadata_sha = semantic_sha256(metadata)
    metadata["metadata_sha256"] = recorded_metadata_sha
    if recorded_metadata_sha != actual_metadata_sha:
        raise ValueError(f"Artifact metadata checksum changed: {sidecar}")
    if verify_checksum and sha256_file(source) != metadata.get("tensor_sha256"):
        raise ValueError(f"Artifact tensor checksum changed: {source}")
    tensors = load_file(str(source), device="cpu")
    expected_shapes = metadata.get("tensor_shapes")
    expected_dtypes = metadata.get("tensor_dtypes")
    if not isinstance(expected_shapes, dict) or not isinstance(expected_dtypes, dict):
        raise ValueError(f"Artifact tensor schema metadata is missing: {sidecar}")
    if set(tensors) != set(expected_shapes) or set(tensors) != set(expected_dtypes):
        raise ValueError(f"Artifact tensor keys changed: {source}")
    for key, tensor in tensors.items():
        if (
            list(tensor.shape) != expected_shapes[key]
            or str(tensor.dtype) != expected_dtypes[key]
        ):
            raise ValueError(f"Artifact tensor schema changed for {key}: {source}")
    return dict(tensors), metadata

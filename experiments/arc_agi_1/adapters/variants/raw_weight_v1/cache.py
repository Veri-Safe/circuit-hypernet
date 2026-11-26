from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import torch
from torch import Tensor

from s2_hypernet.nvarc_pg.checkpoint import sha256_file
from s2_hypernet.nvarc_pg.target_layout import QwenTargetLayout


RAW_CACHE_FORMAT = "nvarc_qwen3_raw_weight_cache_v1"
RMS_EPSILON = 1.0e-12
PINNED_RAW_WEIGHT_BASE_AGGREGATE_SHA256 = (
    "b74bcb895fe162511ab8d047648a9d3cbba110fead5f3d58b9a24e9456d0084c"
)


class TensorSource(Protocol):
    def get_tensor(self, key: str) -> Tensor: ...


@dataclass(frozen=True)
class RawMatrixTarget:
    target_key: str
    weight_key: str
    slot: int
    module_index: int
    module_name: str
    in_features: int
    out_features: int


def layout_matrix_targets(layout: QwenTargetLayout) -> tuple[RawMatrixTarget, ...]:
    targets: list[RawMatrixTarget] = []
    for layer in range(layout.num_layers):
        for module_index, module in enumerate(layout.modules):
            target_key = f"{layout.prefix}{layer}.{module.name}"
            targets.append(
                RawMatrixTarget(
                    target_key=target_key,
                    weight_key=f"{target_key}.weight",
                    slot=layer,
                    module_index=module_index,
                    module_name=module.name,
                    in_features=module.in_features,
                    out_features=module.out_features,
                )
            )
    for offset, module in enumerate(layout.global_modules):
        targets.append(
            RawMatrixTarget(
                target_key=module.name,
                weight_key=f"{module.name}.weight",
                slot=layout.num_layers + offset,
                module_index=len(layout.modules) + offset,
                module_name=module.name,
                in_features=module.in_features,
                out_features=module.out_features,
            )
        )
    if len(targets) != len(layout.target_keys):
        raise ValueError("Raw target enumeration no longer matches the Qwen target layout")
    return tuple(targets)


class MappingTensorSource:
    def __init__(self, tensors: Mapping[str, Tensor]) -> None:
        self.tensors = tensors

    def get_tensor(self, key: str) -> Tensor:
        try:
            return self.tensors[key]
        except KeyError as error:
            raise KeyError(f"Raw checkpoint tensor is missing: {key}") from error


class HFSafetensorSource:
    """Read selected tensors lazily from an offline HF safetensors checkpoint."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"HF checkpoint directory does not exist: {self.model_dir}")
        self.weight_map = self._build_weight_map()

    def _build_weight_map(self) -> dict[str, Path]:
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.is_file():
            value = json.loads(index_path.read_text(encoding="utf-8"))
            raw_map = value.get("weight_map")
            if not isinstance(raw_map, dict) or not raw_map:
                raise ValueError(f"Invalid safetensors weight_map: {index_path}")
            return {
                str(key): self.model_dir / str(filename)
                for key, filename in raw_map.items()
            }

        paths = sorted(self.model_dir.glob("*.safetensors"))
        if not paths:
            raise FileNotFoundError(
                "Raw-weight v1 requires HF safetensors; no *.safetensors files found in "
                f"{self.model_dir}"
            )
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("safetensors is required to read raw Qwen weights") from error
        weight_map: dict[str, Path] = {}
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in weight_map:
                        raise ValueError(f"Duplicate tensor key across safetensors shards: {key}")
                    weight_map[key] = path
        return weight_map

    def get_tensor(self, key: str) -> Tensor:
        resolved_key = self.resolve_key(key)
        try:
            path = self.weight_map[resolved_key]
        except KeyError as error:
            raise KeyError(f"Raw checkpoint tensor is missing: {key}") from error
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("safetensors is required to read raw Qwen weights") from error
        with safe_open(path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(resolved_key)

    def resolve_key(self, key: str) -> str:
        if key in self.weight_map:
            return key
        if key == "lm_head.weight" and "model.embed_tokens.weight" in self.weight_map:
            return "model.embed_tokens.weight"
        raise KeyError(f"Raw checkpoint tensor is missing: {key}")

    def artifact_hashes(self) -> dict[str, str]:
        return {
            path.name: sha256_file(path)
            for path in sorted(set(self.weight_map.values()), key=lambda item: item.name)
        }


def _canonical_projection_seed(*, base_seed: int, axis_length: int, width: int) -> int:
    payload = f"raw-weight-axis-v1:{base_seed}:{axis_length}:{width}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def semi_orthogonal_projection(
    axis_length: int,
    width: int,
    *,
    seed: int,
) -> Tensor:
    if axis_length <= 0 or width <= 0:
        raise ValueError("Projection dimensions must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        _canonical_projection_seed(
            base_seed=int(seed), axis_length=int(axis_length), width=int(width)
        )
    )
    if axis_length >= width:
        sample = torch.randn(axis_length, width, generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(sample, mode="reduced")
        signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
        projection = q * signs
    else:
        sample = torch.randn(width, axis_length, generator=generator, dtype=torch.float64)
        q, r = torch.linalg.qr(sample, mode="reduced")
        signs = torch.where(torch.diagonal(r) < 0, -1.0, 1.0)
        projection = (q * signs).transpose(0, 1)
    return projection.to(dtype=torch.float32).contiguous()


def normalize_raw_matrix(weight: Tensor, *, epsilon: float = RMS_EPSILON) -> tuple[Tensor, Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"Raw linear weight must be rank 2, found {tuple(weight.shape)}")
    value = weight.detach().to(device="cpu", dtype=torch.float32)
    rms = value.square().mean().add(float(epsilon)).sqrt()
    if not torch.isfinite(rms) or float(rms) <= 0:
        raise ValueError("Raw linear weight has an invalid matrix-wise RMS")
    return value / rms, rms.log()


def _standardize_log_rms(log_rms: Tensor) -> tuple[Tensor, float, float]:
    mean = log_rms.mean()
    std = log_rms.std(unbiased=False).clamp_min(1.0e-12)
    return (log_rms - mean) / std, float(mean), float(std)


def fixed_derangement(size: int, *, seed: int) -> Tensor:
    if size < 2:
        raise ValueError("A derangement requires at least two items")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    identity = torch.arange(size)
    for _ in range(100):
        candidate = torch.randperm(size, generator=generator)
        if bool(torch.all(candidate != identity)):
            return candidate
    return torch.roll(identity, shifts=1 + int(seed) % (size - 1))


def _manifest_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def _source_matrix_layout_sha256(layout: QwenTargetLayout) -> str:
    payload = [
        {
            "target_key": target.target_key,
            "weight_key": target.weight_key,
            "slot": target.slot,
            "module_index": target.module_index,
            "module_name": target.module_name,
            "shape": [target.out_features, target.in_features],
        }
        for target in layout_matrix_targets(layout)
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_tensor_audit(
    source_identity: Mapping[str, Any],
    *,
    layout: QwenTargetLayout,
) -> None:
    targets = layout_matrix_targets(layout)
    if source_identity.get("target_matrix_count") != len(targets):
        raise ValueError("Raw-weight cache source target count changed")

    expected_source_layout = _source_matrix_layout_sha256(layout)
    cached_source_layout = source_identity.get("source_matrix_layout_sha256")
    if (
        cached_source_layout is not None
        and cached_source_layout != expected_source_layout
    ):
        raise ValueError("Raw-weight cache source matrix layout changed")

    target_tensors = source_identity.get("target_tensors", {})
    if not isinstance(target_tensors, dict):
        raise ValueError("Raw-weight cache lacks the complete per-tensor source audit")
    expected_keys = {target.target_key for target in targets}
    actual_keys = set(target_tensors)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(
            "Raw-weight cache source tensor set changed: "
            f"missing={missing[:3]} extra={extra[:3]}"
        )

    for target in targets:
        audit = target_tensors[target.target_key]
        if not isinstance(audit, dict):
            raise ValueError(
                "Raw-weight cache has an invalid source tensor audit for "
                f"{target.target_key}"
            )
        if audit.get("requested_key") != target.weight_key:
            raise ValueError(
                "Raw-weight cache source tensor key changed for "
                f"{target.target_key}"
            )
        expected_shape = (target.out_features, target.in_features)
        try:
            actual_shape = tuple(int(value) for value in audit.get("shape", ()))
        except (TypeError, ValueError):
            actual_shape = ()
        if actual_shape != expected_shape:
            raise ValueError(
                "Raw-weight cache source matrix shape changed for "
                f"{target.target_key}: {actual_shape} != {expected_shape}"
            )
        if not isinstance(audit.get("resolved_key"), str) or not audit["resolved_key"]:
            raise ValueError(
                "Raw-weight cache lacks the resolved source key for "
                f"{target.target_key}"
            )
        tensor_sha256 = audit.get("tensor_sha256")
        if not isinstance(tensor_sha256, str) or len(tensor_sha256) != 64:
            raise ValueError(
                "Raw-weight cache lacks the source tensor digest for "
                f"{target.target_key}"
            )


def validate_cache_source_manifest(
    path: str | Path,
    *,
    kind: str,
    layout: QwenTargetLayout,
    projection_seed: int,
    expected_base_aggregate_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    manifest = json.loads(_manifest_path(source).read_text(encoding="utf-8"))
    if manifest.get("format") != RAW_CACHE_FORMAT or manifest.get("kind") != kind:
        raise ValueError(f"Raw-weight cache kind changed: {source}")
    if manifest.get("projection_seed") != projection_seed:
        raise ValueError("Raw-weight cache projection seed changed")
    source_identity = manifest.get("source", {})
    if not isinstance(source_identity, dict):
        raise ValueError("Raw-weight cache lacks its source identity")
    _validate_source_tensor_audit(source_identity, layout=layout)
    base_manifest = source_identity.get("base_checkpoint_manifest", {})
    expected_base_aggregate_sha256 = (
        PINNED_RAW_WEIGHT_BASE_AGGREGATE_SHA256
        if expected_base_aggregate_sha256 is None
        else expected_base_aggregate_sha256
    )
    if (
        base_manifest.get("hf_aggregate_sha256")
        != expected_base_aggregate_sha256
    ):
        raise ValueError(
            "Raw-weight cache was not built from the expected pinned Qwen3 checkpoint: "
            f"{base_manifest.get('hf_aggregate_sha256')!r} != "
            f"{expected_base_aggregate_sha256!r}"
        )
    if kind == "film_sketch32_v1":
        expected_keys = tuple(layout.target_keys)
    elif kind == "p2e_qv_axis128_v1":
        expected_keys = tuple(
            target.target_key
            for target in layout_matrix_targets(layout)
            if target.module_name in {"self_attn.q_proj", "self_attn.v_proj"}
        )
    else:
        raise ValueError(f"Unknown raw-weight cache kind: {kind!r}")
    if tuple(manifest.get("target_keys", ())) != expected_keys:
        raise ValueError("Raw-weight cache target order changed")
    return manifest


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_safetensors(path: Path, tensors: Mapping[str, Tensor]) -> str:
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to store raw-weight caches") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(temporary))
    os.replace(temporary, path)
    return sha256_file(path)


def _load_safetensors(path: Path) -> dict[str, Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("safetensors is required to load raw-weight caches") from error
    return load_file(str(path), device="cpu")


def _source_manifest_digest(value: Mapping[str, Any] | None) -> tuple[dict[str, Any], Tensor]:
    source = dict(value or {})
    payload = json.dumps(
        source,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return source, torch.tensor(list(digest), dtype=torch.uint8)


def _validate_embedded_source_digest(
    tensors: Mapping[str, Tensor],
    manifest: Mapping[str, Any],
) -> None:
    _, expected = _source_manifest_digest(manifest.get("source", {}))
    actual = tensors.get("source_manifest_sha256_bytes")
    if actual is None or not torch.equal(actual.cpu().to(torch.uint8), expected):
        raise ValueError("Raw-weight cache source manifest is not bound to its tensor file")


@dataclass(frozen=True)
class FilmFeatureCache:
    features: Tensor
    valid_mask: Tensor
    target_keys: tuple[str, ...]
    log_rms_mean: float
    log_rms_std: float
    projection_seed: int = 24

    def __post_init__(self) -> None:
        if self.features.shape != (37, 8, 1025):
            raise ValueError(f"FiLM cache must have shape [37, 8, 1025], found {tuple(self.features.shape)}")
        if self.valid_mask.shape != (37, 8) or self.valid_mask.dtype != torch.bool:
            raise ValueError("FiLM valid_mask must be bool [37, 8]")
        if int(self.valid_mask.sum().item()) != 253:
            raise ValueError("FiLM cache must contain exactly 253 valid matrices")
        if len(self.target_keys) != 253:
            raise ValueError("FiLM target key manifest must contain 253 entries")
        if not torch.isfinite(self.features).all():
            raise ValueError("FiLM cache contains nonfinite values")

    def save(self, path: str | Path, *, source_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        destination = Path(path).expanduser().resolve()
        source, source_digest = _source_manifest_digest(source_manifest)
        tensor_sha = _save_safetensors(
            destination,
            {
                "features": self.features.float(),
                "valid_mask": self.valid_mask.to(torch.uint8),
                "source_manifest_sha256_bytes": source_digest,
            },
        )
        manifest: dict[str, Any] = {
            "format": RAW_CACHE_FORMAT,
            "kind": "film_sketch32_v1",
            "tensor_sha256": tensor_sha,
            "projection_seed": self.projection_seed,
            "feature_shape": list(self.features.shape),
            "valid_matrix_count": int(self.valid_mask.sum().item()),
            "target_keys": list(self.target_keys),
            "log_rms_mean": self.log_rms_mean,
            "log_rms_std": self.log_rms_std,
            "source": source,
        }
        _write_json_atomic(_manifest_path(destination), manifest)
        return manifest

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str = "") -> "FilmFeatureCache":
        source = Path(path).expanduser().resolve()
        actual_sha256 = sha256_file(source)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError("FiLM raw-weight cache SHA-256 mismatch")
        manifest = json.loads(_manifest_path(source).read_text(encoding="utf-8"))
        if manifest.get("format") != RAW_CACHE_FORMAT or manifest.get("kind") != "film_sketch32_v1":
            raise ValueError("Not a FiLM sketch32 raw-weight cache")
        if actual_sha256 != manifest.get("tensor_sha256"):
            raise ValueError("FiLM cache tensor hash differs from its manifest")
        tensors = _load_safetensors(source)
        _validate_embedded_source_digest(tensors, manifest)
        return cls(
            features=tensors["features"].float(),
            valid_mask=tensors["valid_mask"].bool(),
            target_keys=tuple(str(key) for key in manifest["target_keys"]),
            log_rms_mean=float(manifest["log_rms_mean"]),
            log_rms_std=float(manifest["log_rms_std"]),
            projection_seed=int(manifest["projection_seed"]),
        )


@dataclass(frozen=True)
class P2EFeatureCache:
    row_values: Tensor
    row_offsets: Tensor
    column_values: Tensor
    column_offsets: Tensor
    standardized_log_rms: Tensor
    target_slots: Tensor
    target_modules: Tensor
    target_keys: tuple[str, ...]
    log_rms_mean: float
    log_rms_std: float
    projection_seed: int = 24

    def __post_init__(self) -> None:
        count = len(self.target_keys)
        if count != 72:
            raise ValueError(f"P2E-QV cache must contain 72 matrices, found {count}")
        if self.row_values.dim() != 2 or self.row_values.shape[1] != 128:
            raise ValueError("P2E row_values must have shape [N, 128]")
        if self.column_values.dim() != 2 or self.column_values.shape[1] != 128:
            raise ValueError("P2E column_values must have shape [N, 128]")
        for name, offsets, total in (
            ("row", self.row_offsets, self.row_values.shape[0]),
            ("column", self.column_offsets, self.column_values.shape[0]),
        ):
            if offsets.dtype != torch.long or offsets.shape != (count + 1,):
                raise ValueError(f"P2E {name}_offsets must be int64 [{count + 1}]")
            if int(offsets[0]) != 0 or int(offsets[-1]) != total:
                raise ValueError(f"P2E {name}_offsets do not span the packed tensor")
            if bool(torch.any(offsets[1:] < offsets[:-1])):
                raise ValueError(f"P2E {name}_offsets are not monotonic")
        if self.standardized_log_rms.shape != (count,):
            raise ValueError("P2E standardized_log_rms must have shape [72]")
        if self.target_slots.shape != (count,) or self.target_slots.dtype != torch.long:
            raise ValueError("P2E target_slots must be int64 [72]")
        if self.target_modules.shape != (count,) or self.target_modules.dtype != torch.long:
            raise ValueError("P2E target_modules must be int64 [72]")
        if not torch.isfinite(self.row_values).all() or not torch.isfinite(self.column_values).all():
            raise ValueError("P2E cache contains nonfinite values")

    @property
    def matrix_count(self) -> int:
        return len(self.target_keys)

    def axis_slice(self, matrix_index: int, *, axis: str) -> Tensor:
        if not 0 <= matrix_index < self.matrix_count:
            raise IndexError(matrix_index)
        if axis == "row":
            values, offsets = self.row_values, self.row_offsets
        elif axis == "column":
            values, offsets = self.column_values, self.column_offsets
        else:
            raise ValueError(f"Unknown P2E axis: {axis!r}")
        start, end = int(offsets[matrix_index]), int(offsets[matrix_index + 1])
        return values[start:end]

    def save(self, path: str | Path, *, source_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
        destination = Path(path).expanduser().resolve()
        source, source_digest = _source_manifest_digest(source_manifest)
        tensor_sha = _save_safetensors(
            destination,
            {
                "row_values": self.row_values.float(),
                "row_offsets": self.row_offsets,
                "column_values": self.column_values.float(),
                "column_offsets": self.column_offsets,
                "standardized_log_rms": self.standardized_log_rms.float(),
                "target_slots": self.target_slots,
                "target_modules": self.target_modules,
                "source_manifest_sha256_bytes": source_digest,
            },
        )
        manifest: dict[str, Any] = {
            "format": RAW_CACHE_FORMAT,
            "kind": "p2e_qv_axis128_v1",
            "tensor_sha256": tensor_sha,
            "projection_seed": self.projection_seed,
            "matrix_count": self.matrix_count,
            "target_keys": list(self.target_keys),
            "log_rms_mean": self.log_rms_mean,
            "log_rms_std": self.log_rms_std,
            "row_token_count": int(self.row_values.shape[0]),
            "column_token_count": int(self.column_values.shape[0]),
            "source": source,
        }
        _write_json_atomic(_manifest_path(destination), manifest)
        return manifest

    @classmethod
    def load(cls, path: str | Path, *, expected_sha256: str = "") -> "P2EFeatureCache":
        source = Path(path).expanduser().resolve()
        actual_sha256 = sha256_file(source)
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError("P2E raw-weight cache SHA-256 mismatch")
        manifest = json.loads(_manifest_path(source).read_text(encoding="utf-8"))
        if manifest.get("format") != RAW_CACHE_FORMAT or manifest.get("kind") != "p2e_qv_axis128_v1":
            raise ValueError("Not a P2E-QV raw-weight cache")
        if actual_sha256 != manifest.get("tensor_sha256"):
            raise ValueError("P2E cache tensor hash differs from its manifest")
        tensors = _load_safetensors(source)
        _validate_embedded_source_digest(tensors, manifest)
        return cls(
            row_values=tensors["row_values"].float(),
            row_offsets=tensors["row_offsets"].long(),
            column_values=tensors["column_values"].float(),
            column_offsets=tensors["column_offsets"].long(),
            standardized_log_rms=tensors["standardized_log_rms"].float(),
            target_slots=tensors["target_slots"].long(),
            target_modules=tensors["target_modules"].long(),
            target_keys=tuple(str(key) for key in manifest["target_keys"]),
            log_rms_mean=float(manifest["log_rms_mean"]),
            log_rms_std=float(manifest["log_rms_std"]),
            projection_seed=int(manifest["projection_seed"]),
        )


def _validate_weight_shape(weight: Tensor, target: RawMatrixTarget) -> None:
    expected = (target.out_features, target.in_features)
    if tuple(weight.shape) != expected:
        raise ValueError(
            f"Raw weight {target.weight_key} shape changed: {tuple(weight.shape)} != {expected}"
        )


def build_film_feature_cache(
    source: TensorSource,
    *,
    layout: QwenTargetLayout,
    projection_seed: int = 24,
) -> FilmFeatureCache:
    targets = layout_matrix_targets(layout)
    sketches: list[Tensor] = []
    log_rms_values: list[Tensor] = []
    for target in targets:
        weight = source.get_tensor(target.weight_key)
        _validate_weight_shape(weight, target)
        normalized, log_rms = normalize_raw_matrix(weight)
        if target.module_name == "lm_head":
            projection_in = semi_orthogonal_projection(
                target.in_features, 64, seed=projection_seed
            )
            sketch = normalized @ projection_in
            if sketch.shape != (16, 64):
                raise ValueError("lm_head FiLM sketch must be asymmetric [16, 64]")
        else:
            projection_out = semi_orthogonal_projection(
                target.out_features, 32, seed=projection_seed
            )
            projection_in = semi_orthogonal_projection(
                target.in_features, 32, seed=projection_seed
            )
            sketch = projection_out.transpose(0, 1) @ normalized @ projection_in
            if sketch.shape != (32, 32):
                raise ValueError("Transformer FiLM sketch must be [32, 32]")
        sketches.append(sketch.reshape(-1))
        log_rms_values.append(log_rms.reshape(()))

    standardized, mean, std = _standardize_log_rms(torch.stack(log_rms_values))
    features = torch.zeros(
        layout.generator_slot_count,
        len(layout.all_generator_modules),
        1025,
        dtype=torch.float32,
    )
    valid = torch.zeros(features.shape[:2], dtype=torch.bool)
    for index, target in enumerate(targets):
        features[target.slot, target.module_index, :1024] = sketches[index]
        features[target.slot, target.module_index, 1024] = standardized[index]
        valid[target.slot, target.module_index] = True
    return FilmFeatureCache(
        features=features,
        valid_mask=valid,
        target_keys=tuple(target.target_key for target in targets),
        log_rms_mean=mean,
        log_rms_std=std,
        projection_seed=projection_seed,
    )


def build_p2e_feature_cache(
    source: TensorSource,
    *,
    layout: QwenTargetLayout,
    projection_seed: int = 24,
) -> P2EFeatureCache:
    targets = layout_matrix_targets(layout)
    log_rms_by_key: dict[str, Tensor] = {}
    row_values: list[Tensor] = []
    column_values: list[Tensor] = []
    selected_targets: list[RawMatrixTarget] = []

    for target in targets:
        weight = source.get_tensor(target.weight_key)
        _validate_weight_shape(weight, target)
        normalized, log_rms = normalize_raw_matrix(weight)
        log_rms_by_key[target.target_key] = log_rms.reshape(())
        if target.module_name not in {"self_attn.q_proj", "self_attn.v_proj"}:
            continue
        projection_in = semi_orthogonal_projection(
            target.in_features, 128, seed=projection_seed
        )
        projection_out = semi_orthogonal_projection(
            target.out_features, 128, seed=projection_seed
        )
        row_values.append((normalized @ projection_in).contiguous())
        column_values.append((normalized.transpose(0, 1) @ projection_out).contiguous())
        selected_targets.append(target)

    all_log_rms = torch.stack([log_rms_by_key[target.target_key] for target in targets])
    _, mean, std = _standardize_log_rms(all_log_rms)
    selected_log_rms = torch.stack(
        [log_rms_by_key[target.target_key] for target in selected_targets]
    )
    selected_standardized = (selected_log_rms - mean) / max(std, 1.0e-12)

    row_offsets = [0]
    column_offsets = [0]
    for row, column in zip(row_values, column_values, strict=True):
        row_offsets.append(row_offsets[-1] + row.shape[0])
        column_offsets.append(column_offsets[-1] + column.shape[0])
    return P2EFeatureCache(
        row_values=torch.cat(row_values, dim=0),
        row_offsets=torch.tensor(row_offsets, dtype=torch.long),
        column_values=torch.cat(column_values, dim=0),
        column_offsets=torch.tensor(column_offsets, dtype=torch.long),
        standardized_log_rms=selected_standardized.float(),
        target_slots=torch.tensor([target.slot for target in selected_targets], dtype=torch.long),
        target_modules=torch.tensor(
            [target.module_index for target in selected_targets], dtype=torch.long
        ),
        target_keys=tuple(target.target_key for target in selected_targets),
        log_rms_mean=mean,
        log_rms_std=std,
        projection_seed=projection_seed,
    )


def build_source_manifest(source: HFSafetensorSource, *, layout: QwenTargetLayout) -> dict[str, Any]:
    targets = layout_matrix_targets(layout)
    missing: list[str] = []
    resolved_keys: dict[str, str] = {}
    tensors: dict[str, dict[str, Any]] = {}
    for target in targets:
        try:
            resolved_key = source.resolve_key(target.weight_key)
            weight = source.get_tensor(target.weight_key)
            _validate_weight_shape(weight, target)
            resolved_keys[target.weight_key] = resolved_key
            contiguous = weight.detach().cpu().contiguous()
            tensors[target.target_key] = {
                "requested_key": target.weight_key,
                "resolved_key": resolved_key,
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "tensor_sha256": hashlib.sha256(
                    contiguous.view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
        except KeyError:
            missing.append(target.weight_key)
    if missing:
        raise KeyError(f"HF checkpoint is missing raw target weights: {missing[:10]}")
    return {
        "hf_checkpoint": str(source.model_dir),
        "hf_safetensors": source.artifact_hashes(),
        "target_layout_sha256": layout.layout_sha256,
        "source_matrix_layout_sha256": _source_matrix_layout_sha256(layout),
        "target_matrix_count": len(targets),
        "resolved_weight_keys": resolved_keys,
        "target_tensors": tensors,
        "normalization": "matrix_rms_fp32_v1",
        "rms_epsilon": RMS_EPSILON,
    }

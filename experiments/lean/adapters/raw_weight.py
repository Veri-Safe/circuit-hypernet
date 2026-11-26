"""S5 Query+Raw-Weight Safe-P2E conditioning.

This module is deliberately independent of the historical Qwen3 raw-weight
implementation.  S5 has eight transformer layers, four target modules, and a
34-token Parameter-Token row, so reusing a Q/V/37-slot conditioner would
silently route only part of the generated adapter.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from transformers import PreTrainedModel

from s2_hypernet.lora_tokenizer import Tokenizer2DBatchedLoRA
from s2_hypernet.parameter_generator import TransformerModel
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.cache import (
    semi_orthogonal_projection,
)

from .config import base_content_identity_matches
from .parameter_generator import (
    S5_PG_D_MODEL,
    S5ParameterGenerator,
)
from .target_layout import S5TargetLayout, build_s5_target_layout


S5_RAW_CACHE_FORMAT = "s5_raw_weight_cache_v1"
S5_RAW_CACHE_KIND = "p2e_all_targets_axis128_v1"
S5_RAW_PROJECTION_SEED = 24
S5_RAW_WIDTH = 128
S5_RAW_QUERY_COUNT = 8
S5_RAW_COMPRESSOR_HEADS = 4
S5_RAW_COMPRESSOR_FFN = 512
S5_RAW_RESIDUAL_CAP = 0.025
S5_RAW_RESIDUAL_RMS_CEILING = 0.005
S5_RAW_RMS_EPSILON = 1.0e-12


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@dataclass(frozen=True)
class S5RawMatrixTarget:
    target_key: str
    slot: int
    module_index: int
    module_name: str
    in_features: int
    out_features: int


def s5_raw_matrix_targets(
    layout: S5TargetLayout | None = None,
) -> tuple[S5RawMatrixTarget, ...]:
    layout = layout or build_s5_target_layout()
    return tuple(
        S5RawMatrixTarget(
            target_key=f"{layout.prefix}{layer}.{module.name}",
            slot=layer,
            module_index=module_index,
            module_name=module.name,
            in_features=module.in_features,
            out_features=module.out_features,
        )
        for layer in range(layout.num_layers)
        for module_index, module in enumerate(layout.modules)
    )


@dataclass(frozen=True)
class S5RawWeightCache:
    """Packed, CPU-only FP32 raw-weight sketches for all 32 S5 targets."""

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
    projection_seed: int = S5_RAW_PROJECTION_SEED
    base_identity: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        count = len(self.target_keys)
        if count != 32:
            raise ValueError(f"S5 raw cache must contain 32 matrices, found {count}")
        if self.row_values.ndim != 2 or self.row_values.shape[1] != S5_RAW_WIDTH:
            raise ValueError("S5 raw row_values must have shape [N,128]")
        if self.column_values.ndim != 2 or self.column_values.shape[1] != S5_RAW_WIDTH:
            raise ValueError("S5 raw column_values must have shape [N,128]")
        for name, offsets, total in (
            ("row", self.row_offsets, self.row_values.shape[0]),
            ("column", self.column_offsets, self.column_values.shape[0]),
        ):
            if offsets.dtype != torch.long or offsets.shape != (count + 1,):
                raise ValueError(f"S5 raw {name}_offsets must have shape [33]")
            if int(offsets[0]) != 0 or int(offsets[-1]) != total:
                raise ValueError(f"S5 raw {name}_offsets do not span packed values")
            if bool(torch.any(offsets[1:] < offsets[:-1])):
                raise ValueError(f"S5 raw {name}_offsets are not monotonic")
        if self.standardized_log_rms.shape != (count,):
            raise ValueError("S5 standardized_log_rms must have shape [32]")
        if self.target_slots.shape != (count,) or self.target_slots.dtype != torch.long:
            raise ValueError("S5 target_slots must be int64 [32]")
        if self.target_modules.shape != (count,) or self.target_modules.dtype != torch.long:
            raise ValueError("S5 target_modules must be int64 [32]")
        if not (
            torch.isfinite(self.row_values).all()
            and torch.isfinite(self.column_values).all()
            and torch.isfinite(self.standardized_log_rms).all()
        ):
            raise ValueError("S5 raw cache contains non-finite values")
        expected = tuple(target.target_key for target in s5_raw_matrix_targets())
        if self.target_keys != expected:
            raise ValueError("S5 raw target order changed")
        expected_slots = torch.tensor(
            [target.slot for target in s5_raw_matrix_targets()], dtype=torch.long
        )
        expected_modules = torch.tensor(
            [target.module_index for target in s5_raw_matrix_targets()], dtype=torch.long
        )
        if not torch.equal(self.target_slots.cpu(), expected_slots):
            raise ValueError("S5 raw target slots/order changed")
        if not torch.equal(self.target_modules.cpu(), expected_modules):
            raise ValueError("S5 raw target modules/order changed")

    @property
    def matrix_count(self) -> int:
        return len(self.target_keys)

    def axis_slice(self, matrix_index: int, *, axis: str) -> Tensor:
        if not 0 <= int(matrix_index) < self.matrix_count:
            raise IndexError(matrix_index)
        if axis == "row":
            values, offsets = self.row_values, self.row_offsets
        elif axis == "column":
            values, offsets = self.column_values, self.column_offsets
        else:
            raise ValueError(f"unknown axis: {axis}")
        start, end = int(offsets[matrix_index]), int(offsets[matrix_index + 1])
        return values[start:end]

    def manifest(self, *, tensor_sha256: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "format": S5_RAW_CACHE_FORMAT,
            "kind": S5_RAW_CACHE_KIND,
            "matrix_count": self.matrix_count,
            "target_keys": list(self.target_keys),
            "target_slots": self.target_slots.tolist(),
            "target_modules": self.target_modules.tolist(),
            "row_token_count": int(self.row_values.shape[0]),
            "column_token_count": int(self.column_values.shape[0]),
            "projection_seed": int(self.projection_seed),
            "storage_dtype": "torch.float32",
            "normalization": "matrix_rms_fp32_v1",
            "rms_epsilon": S5_RAW_RMS_EPSILON,
            "log_rms_population_statistics": True,
            "log_rms_mean": float(self.log_rms_mean),
            "log_rms_std": float(self.log_rms_std),
            "layout_sha256": build_s5_target_layout().layout_sha256,
            "base_identity": dict(self.base_identity or {}),
            "target_tensor_sha256": list(self.target_tensor_hashes()),
        }
        if tensor_sha256 is not None:
            body["tensor_sha256"] = str(tensor_sha256)
        return body

    def save(self, path: str | Path) -> dict[str, Any]:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": S5_RAW_CACHE_FORMAT,
            "kind": S5_RAW_CACHE_KIND,
            "row_values": self.row_values.float().cpu(),
            "row_offsets": self.row_offsets.long().cpu(),
            "column_values": self.column_values.float().cpu(),
            "column_offsets": self.column_offsets.long().cpu(),
            "standardized_log_rms": self.standardized_log_rms.float().cpu(),
            "target_slots": self.target_slots.long().cpu(),
            "target_modules": self.target_modules.long().cpu(),
            "target_keys": list(self.target_keys),
        }
        temporary = destination.with_name(f".{destination.name}.{__import__('os').getpid()}.tmp")
        torch.save(payload, temporary)
        temporary.replace(destination)
        tensor_sha = _file_sha256(destination)
        manifest = self.manifest(tensor_sha256=tensor_sha)
        manifest["manifest_sha256"] = _json_sha256(manifest)
        manifest_path = destination.with_suffix(destination.suffix + ".json")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str = "",
        expected_base_identity: Mapping[str, Any] | None = None,
    ) -> "S5RawWeightCache":
        source = Path(path).expanduser().resolve()
        actual = _file_sha256(source)
        if expected_sha256 and actual != expected_sha256:
            raise ValueError("S5 raw cache tensor hash changed")
        manifest_path = source.with_suffix(source.suffix + ".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded = manifest.pop("manifest_sha256", None)
        if manifest.get("format") != S5_RAW_CACHE_FORMAT or manifest.get("kind") != S5_RAW_CACHE_KIND:
            raise ValueError("not an S5 raw-weight cache")
        if recorded != _json_sha256(manifest) or actual != manifest.get("tensor_sha256"):
            raise ValueError("S5 raw cache manifest identity changed")
        if expected_base_identity is not None and not base_content_identity_matches(
            manifest.get("base_identity", {}), expected_base_identity
        ):
            raise ValueError("S5 raw cache Base identity changed")
        payload = torch.load(source, map_location="cpu", weights_only=True)
        value = cls(
            row_values=payload["row_values"].float(),
            row_offsets=payload["row_offsets"].long(),
            column_values=payload["column_values"].float(),
            column_offsets=payload["column_offsets"].long(),
            standardized_log_rms=payload["standardized_log_rms"].float(),
            target_slots=payload["target_slots"].long(),
            target_modules=payload["target_modules"].long(),
            target_keys=tuple(str(value) for value in payload["target_keys"]),
            log_rms_mean=float(manifest["log_rms_mean"]),
            log_rms_std=float(manifest["log_rms_std"]),
            projection_seed=int(manifest["projection_seed"]),
            base_identity=manifest.get("base_identity", {}),
        )
        if manifest.get("target_tensor_sha256") != list(value.target_tensor_hashes()):
            raise ValueError("S5 raw cache per-target tensor identity changed")
        return value

    def target_tensor_hashes(self) -> tuple[str, ...]:
        """Hash each target's row/column sketch and standardized log-RMS."""

        hashes: list[str] = []
        for index in range(self.matrix_count):
            digest = hashlib.sha256()
            for tensor in (
                self.axis_slice(index, axis="row"),
                self.axis_slice(index, axis="column"),
                self.standardized_log_rms[index : index + 1],
            ):
                value = tensor.detach().cpu().contiguous()
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(str(tuple(value.shape)).encode("ascii"))
                digest.update(value.numpy().tobytes())
            hashes.append(digest.hexdigest())
        return tuple(hashes)


def _normalize_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    value = weight.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if value.ndim != 2:
        raise ValueError("S5 raw target weight must be rank 2")
    rms = value.square().mean().add(S5_RAW_RMS_EPSILON).sqrt()
    if not bool(torch.isfinite(rms)) or float(rms) <= 0.0:
        raise ValueError("S5 raw target weight RMS is invalid")
    return value / rms, rms.log()


def build_s5_raw_weight_cache(
    base_model: nn.Module,
    *,
    layout: S5TargetLayout | None = None,
    projection_seed: int = S5_RAW_PROJECTION_SEED,
    base_identity: Mapping[str, Any] | None = None,
) -> S5RawWeightCache:
    layout = layout or build_s5_target_layout()
    targets = s5_raw_matrix_targets(layout)
    rows: list[Tensor] = []
    columns: list[Tensor] = []
    log_rms: list[Tensor] = []
    for target in targets:
        module = base_model.get_submodule(target.target_key)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor):
            raise ValueError(f"S5 Base target has no tensor weight: {target.target_key}")
        if tuple(weight.shape) != (target.out_features, target.in_features):
            raise ValueError(
                f"S5 raw target shape changed for {target.target_key}: "
                f"{tuple(weight.shape)} != {(target.out_features, target.in_features)}"
            )
        normalized, log_value = _normalize_weight(weight)
        rows.append(
            (normalized @ semi_orthogonal_projection(
                target.in_features, S5_RAW_WIDTH, seed=projection_seed
            )).contiguous()
        )
        columns.append(
            (normalized.transpose(0, 1) @ semi_orthogonal_projection(
                target.out_features, S5_RAW_WIDTH, seed=projection_seed
            )).contiguous()
        )
        log_rms.append(log_value.reshape(()))
    logs = torch.stack(log_rms).float()
    mean = logs.mean()
    std = logs.std(unbiased=False).clamp_min(S5_RAW_RMS_EPSILON)
    row_offsets = [0]
    column_offsets = [0]
    for row, column in zip(rows, columns, strict=True):
        row_offsets.append(row_offsets[-1] + row.shape[0])
        column_offsets.append(column_offsets[-1] + column.shape[0])
    return S5RawWeightCache(
        row_values=torch.cat(rows, dim=0),
        row_offsets=torch.tensor(row_offsets, dtype=torch.long),
        column_values=torch.cat(columns, dim=0),
        column_offsets=torch.tensor(column_offsets, dtype=torch.long),
        standardized_log_rms=(logs - mean) / std,
        target_slots=torch.tensor([target.slot for target in targets], dtype=torch.long),
        target_modules=torch.tensor([target.module_index for target in targets], dtype=torch.long),
        target_keys=tuple(target.target_key for target in targets),
        log_rms_mean=float(mean),
        log_rms_std=float(std),
        projection_seed=projection_seed,
        base_identity=base_identity,
    )


class _AxisQueryCompressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(S5_RAW_QUERY_COUNT, S5_RAW_WIDTH))
        self.query_norm = nn.RMSNorm(S5_RAW_WIDTH, eps=1.0e-6)
        self.value_norm = nn.RMSNorm(S5_RAW_WIDTH, eps=1.0e-6)
        self.attention = nn.MultiheadAttention(
            S5_RAW_WIDTH,
            S5_RAW_COMPRESSOR_HEADS,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.RMSNorm(S5_RAW_WIDTH, eps=1.0e-6)
        self.ffn = nn.Sequential(
            nn.Linear(S5_RAW_WIDTH, S5_RAW_COMPRESSOR_FFN),
            nn.SiLU(),
            nn.Linear(S5_RAW_COMPRESSOR_FFN, S5_RAW_WIDTH),
        )
        self.final_norm = nn.RMSNorm(S5_RAW_WIDTH, elementwise_affine=False, eps=1.0e-6)
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        queries = self.queries.to(dtype=values.dtype).unsqueeze(0).expand(values.shape[0], -1, -1)
        normalized = self.value_norm(values)
        attended, _ = self.attention(self.query_norm(queries), normalized, normalized, need_weights=False)
        hidden = queries + attended
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return self.final_norm(hidden)


def _fourier_coordinates(length: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    position = (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / float(length)
    frequency = torch.arange(1, 65, device=device, dtype=torch.float32)
    phase = 2.0 * torch.pi * position[:, None] * frequency[None, :]
    return torch.stack((phase.sin(), phase.cos()), dim=-1).reshape(length, S5_RAW_WIDTH).to(dtype=dtype)


class S5SafeP2ERawWeightConditioner(nn.Module):
    """Query-aware all-target Safe-P2E residual for the S5 8x34x1024 grid."""

    method = "s5_safe_p2e_all_targets"

    def __init__(
        self,
        cache: S5RawWeightCache,
        *,
        d_model: int = S5_PG_D_MODEL,
        residual_cap: float = S5_RAW_RESIDUAL_CAP,
        residual_rms_ceiling: float | None = S5_RAW_RESIDUAL_RMS_CEILING,
    ) -> None:
        super().__init__()
        if d_model != S5_PG_D_MODEL:
            raise ValueError("S5 raw conditioner width is fixed at 1024")
        if residual_cap != S5_RAW_RESIDUAL_CAP:
            raise ValueError("S5 raw residual cap is fixed at 0.025")
        if residual_rms_ceiling is not None and not 0.0 < residual_rms_ceiling <= residual_cap:
            raise ValueError("invalid S5 raw residual RMS ceiling")
        self.d_model = int(d_model)
        self.residual_cap = float(residual_cap)
        self.residual_rms_ceiling = (
            None if residual_rms_ceiling is None else float(residual_rms_ceiling)
        )
        self.register_buffer("row_values", cache.row_values.float(), persistent=False)
        self.register_buffer("row_offsets", cache.row_offsets.long(), persistent=False)
        self.register_buffer("column_values", cache.column_values.float(), persistent=False)
        self.register_buffer("column_offsets", cache.column_offsets.long(), persistent=False)
        self.register_buffer("standardized_log_rms", cache.standardized_log_rms.float(), persistent=False)
        self.register_buffer("target_slots", cache.target_slots.long(), persistent=False)
        self.register_buffer("target_modules", cache.target_modules.long(), persistent=False)
        self.token_to_module = self._build_token_to_module()
        self.axis_type_embeddings = nn.Parameter(torch.empty(2, S5_RAW_WIDTH))
        self.scale_direction = nn.Linear(1, S5_RAW_WIDTH, bias=False)
        self.input_norm = nn.RMSNorm(S5_RAW_WIDTH, elementwise_affine=False, eps=1.0e-6)
        self.row_compressor = _AxisQueryCompressor()
        self.column_compressor = _AxisQueryCompressor()
        self.memory_projection = nn.Linear(S5_RAW_WIDTH, d_model, bias=False)
        self.memory_norm = nn.RMSNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.weight_cross_attention = nn.MultiheadAttention(d_model, 8, dropout=0.0, batch_first=True)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.axis_type_embeddings, mean=0.0, std=0.02)
        self.scale_direction.reset_parameters()
        self.memory_projection.reset_parameters()
        nn.init.zeros_(self.output_projection.weight)
        self._axis_segments = {
            "row": self._build_axis_segments(self.row_offsets),
            "column": self._build_axis_segments(self.column_offsets),
        }
        self._coordinate_buffers: dict[int, str] = {}
        lengths = {
            int(value)
            for offsets in (self.row_offsets, self.column_offsets)
            for value in (offsets[1:] - offsets[:-1]).tolist()
        }
        for length in sorted(lengths):
            name = f"coordinate_{length}"
            self.register_buffer(name, _fourier_coordinates(length, device=torch.device("cpu"), dtype=torch.float32), persistent=False)
            self._coordinate_buffers[length] = name
        self._route_buffers: list[tuple[str, str]] = []
        for module_index in range(4):
            token_name = f"raw_token_indices_{module_index}"
            matrix_name = f"raw_matrix_indices_{module_index}"
            token_indices = torch.nonzero(self.token_to_module == module_index, as_tuple=False).flatten()
            matrix_indices = torch.nonzero(self.target_modules == module_index, as_tuple=False).flatten()
            self.register_buffer(token_name, token_indices, persistent=False)
            self.register_buffer(matrix_name, matrix_indices, persistent=False)
            self._route_buffers.append((token_name, matrix_name))
        self._eval_memory: Tensor | None = None
        self._eval_memory_signature: tuple[Any, ...] | None = None
        self._last_residual_rms: Tensor | None = None
        self._last_residual_max_abs: Tensor | None = None

    @staticmethod
    def _build_axis_segments(offsets: Tensor) -> dict[int, tuple[tuple[int, int, int], ...]]:
        groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        values = offsets.detach().cpu().tolist()
        for index, (start, end) in enumerate(zip(values[:-1], values[1:], strict=True)):
            groups[int(end - start)].append((index, int(start), int(end)))
        return {length: tuple(items) for length, items in groups.items()}

    def _build_token_to_module(self) -> Tensor:
        layout = build_s5_target_layout()
        tokenizer = Tokenizer2DBatchedLoRA(
            token_dim=layout.token_dim,
            rank=layout.rank,
            alpha=layout.alpha,
            pg_mapping=layout.pg_mapping,
        )
        module_to_index = {module.name: index for index, module in enumerate(layout.modules)}
        token_area = layout.rank * layout.token_dim
        route: list[int] = []
        for key, shape in tokenizer.shape_state_dict.items():
            module_name, _, factor = key.rpartition(".lora_")
            if factor not in {"A", "B"} or module_name not in module_to_index:
                raise ValueError(f"unexpected S5 tokenizer route: {key}")
            route.extend([module_to_index[module_name]] * math.ceil(math.prod(shape) / token_area))
        expected = (9, 4, 11, 10)
        actual = tuple(route.count(index) for index in range(4))
        if actual != expected or len(route) != 34:
            raise ValueError(f"S5 raw token route changed: {actual}, expected {expected}")
        return torch.tensor(route, dtype=torch.long)

    def restore_fp32_output_path(self) -> None:
        self.output_projection.float()

    def _memory_signature(self) -> tuple[Any, ...]:
        tensors = [
            self.row_values,
            self.column_values,
            self.standardized_log_rms,
            self.axis_type_embeddings,
            *self.scale_direction.parameters(),
            *self.row_compressor.parameters(),
            *self.column_compressor.parameters(),
            *self.memory_projection.parameters(),
        ]
        return tuple((item.device, item.dtype, item._version) for item in tensors)

    def train(self, mode: bool = True) -> "S5SafeP2ERawWeightConditioner":
        if mode:
            self._eval_memory = None
            self._eval_memory_signature = None
        return super().train(mode)

    def _coordinate(self, length: int, reference: Tensor) -> Tensor:
        return getattr(self, self._coordinate_buffers[length]).to(device=reference.device, dtype=reference.dtype)

    def _compress_axis(self, values: Tensor, axis: str, compressor: _AxisQueryCompressor, axis_index: int) -> Tensor:
        output = torch.empty((32, S5_RAW_QUERY_COUNT, S5_RAW_WIDTH), device=values.device, dtype=values.dtype)
        scale = self.scale_direction(self.standardized_log_rms.to(device=values.device, dtype=values.dtype)[:, None])
        for length, segments in self._axis_segments[axis].items():
            indices = [item[0] for item in segments]
            packed = torch.stack([values[start:end] for _, start, end in segments], dim=0)
            packed = self.input_norm(
                packed
                + self._coordinate(length, packed)[None]
                + self.axis_type_embeddings[axis_index].to(dtype=packed.dtype)[None, None]
                + scale[indices, None]
            )
            output[indices] = compressor(packed)
        return output

    def build_weight_memory(self) -> Tensor:
        row = self._compress_axis(self.row_values, "row", self.row_compressor, 0)
        column = self._compress_axis(self.column_values, "column", self.column_compressor, 1)
        return self.memory_norm(self.memory_projection(torch.cat((row, column), dim=1)))

    def weight_memory_for_forward(self) -> Tensor:
        if self.training or torch.is_grad_enabled():
            return self.build_weight_memory()
        signature = self._memory_signature()
        if self._eval_memory is None or self._eval_memory_signature != signature:
            self._eval_memory = self.build_weight_memory().detach()
            self._eval_memory_signature = signature
        return self._eval_memory

    def _controlled_weight_memory(
        self,
        *,
        control: str,
        layer_permutation: tuple[int, ...] | None,
        module_permutation: tuple[int, ...] | None,
    ) -> Tensor:
        """Return the frozen weight memory under one explicit structure control."""

        memories = self.weight_memory_for_forward()
        if control == "matched":
            if layer_permutation is not None or module_permutation is not None:
                raise ValueError("matched Raw-Weight control cannot carry a permutation")
            return memories
        if control == "layer_permuted":
            permutation = (
                tuple(range(1, 8)) + (0,)
                if layer_permutation is None
                else tuple(int(value) for value in layer_permutation)
            )
            if len(permutation) != 8 or sorted(permutation) != list(range(8)):
                raise ValueError("S5 raw layer permutation must be a permutation of 0..7")
            if module_permutation is not None:
                raise ValueError("layer-permuted Raw-Weight control cannot carry a module permutation")
            permutation_tensor = torch.tensor(
                permutation,
                dtype=torch.long,
                device=memories.device,
            )
            source_indices = (
                permutation_tensor.index_select(0, self.target_slots) * 4
                + self.target_modules
            )
            return memories.index_select(0, source_indices)
        if control == "module_permuted":
            permutation = (
                (1, 2, 3, 0)
                if module_permutation is None
                else tuple(int(value) for value in module_permutation)
            )
            if len(permutation) != 4 or sorted(permutation) != list(range(4)):
                raise ValueError("S5 raw module permutation must be a permutation of 0..3")
            if layer_permutation is not None:
                raise ValueError("module-permuted Raw-Weight control cannot carry a layer permutation")
            permutation_tensor = torch.tensor(
                permutation,
                dtype=torch.long,
                device=memories.device,
            )
            source_indices = (
                self.target_slots * 4
                + permutation_tensor.index_select(0, self.target_modules)
            )
            return memories.index_select(0, source_indices)
        raise ValueError(f"unknown S5 raw-weight control: {control}")

    def forward(
        self,
        parameter_tokens: Tensor,
        *,
        control: str = "matched",
        layer_permutation: tuple[int, ...] | None = None,
        module_permutation: tuple[int, ...] | None = None,
    ) -> Tensor:
        control = str(control).lower()
        if control not in {"matched", "zero", "layer_permuted", "module_permuted"}:
            raise ValueError(f"unknown S5 raw-weight control: {control}")
        if parameter_tokens.shape[1:] != (8, 34, self.d_model):
            raise ValueError(
                "S5 raw conditioner expects Parameter Tokens [B,8,34,1024], "
                f"found {tuple(parameter_tokens.shape)}"
            )
        if control == "zero":
            if layer_permutation is not None or module_permutation is not None:
                raise ValueError("zero Raw-Weight control cannot carry a permutation")
            return parameter_tokens
        memories = self._controlled_weight_memory(
            control=control,
            layer_permutation=layer_permutation,
            module_permutation=module_permutation,
        ).to(device=parameter_tokens.device, dtype=parameter_tokens.dtype)
        result = parameter_tokens.clone()
        batch_size = parameter_tokens.shape[0]
        deltas: list[tuple[Tensor, Tensor]] = []
        residual_square_sum = torch.zeros((), device=parameter_tokens.device, dtype=torch.float32)
        residual_count = 0
        for token_name, matrix_name in self._route_buffers:
            token_indices = getattr(self, token_name)
            matrix_indices = getattr(self, matrix_name)
            query = parameter_tokens[:, :, token_indices, :]
            query_flat = query.reshape(batch_size * 8, token_indices.numel(), self.d_model)
            memory = memories[matrix_indices]
            memory_flat = memory[None].expand(batch_size, -1, -1, -1).reshape(
                batch_size * 8, memory.shape[1], self.d_model
            )
            attended, _ = self.weight_cross_attention(query_flat, memory_flat, memory_flat, need_weights=False)
            bounded = self.residual_cap * torch.tanh(self.output_projection(attended.float()))
            deltas.append((token_indices, bounded.reshape_as(query)))
            residual_square_sum = residual_square_sum + bounded.float().square().sum()
            residual_count += bounded.numel()
        mean_square = residual_square_sum / max(residual_count, 1)
        pre_rms = mean_square.sqrt()
        if self.residual_rms_ceiling is None:
            ceiling_scale = torch.ones_like(pre_rms)
        else:
            ceiling_scale = self.residual_rms_ceiling / mean_square.clamp_min(self.residual_rms_ceiling**2).sqrt()
        post_square_sum = torch.zeros((), device=parameter_tokens.device, dtype=torch.float32)
        max_abs = torch.zeros((), device=parameter_tokens.device, dtype=torch.float32)
        for token_indices, bounded in deltas:
            scaled = bounded * ceiling_scale
            result[:, :, token_indices, :] = (
                parameter_tokens[:, :, token_indices, :].float() + scaled
            ).to(dtype=parameter_tokens.dtype)
            detached = scaled.detach().float()
            post_square_sum = post_square_sum + detached.square().sum()
            max_abs = torch.maximum(max_abs, detached.abs().max())
        self._last_residual_rms = (post_square_sum / max(residual_count, 1)).sqrt().detach()
        self._last_residual_max_abs = max_abs.detach()
        return result

    def diagnostic_metrics(self) -> dict[str, float]:
        value = {
            "raw_weight_residual_cap": self.residual_cap,
            "raw_weight_output_weight_rms": float(self.output_projection.weight.detach().float().square().mean().sqrt().cpu()),
        }
        if self._last_residual_rms is not None:
            value["raw_weight_last_residual_rms"] = float(self._last_residual_rms.cpu())
            value["raw_weight_last_residual_max_abs"] = float(self._last_residual_max_abs.cpu())
        return value


class S5RawQueryEncoderConditioner(S5SafeP2ERawWeightConditioner):
    """Inject compressed Base-weight memory into Query Encoder output tokens.

    Unlike Safe-P2E, this route does not touch Parameter Tokens after PG block
    one.  It compresses every target matrix to one token, lets the encoded
    Query attend to those 32 tokens, and returns a bounded zero-initialized
    residual to the ordinary Query-only PG memory.
    """

    method = "s5_raw_query_cross_all_targets"

    def forward(
        self,
        query_tokens: Tensor,
        query_mask: Tensor,
        *,
        control: str = "matched",
        layer_permutation: tuple[int, ...] | None = None,
        module_permutation: tuple[int, ...] | None = None,
    ) -> Tensor:
        control = str(control).lower()
        if control not in {"matched", "zero", "layer_permuted", "module_permuted"}:
            raise ValueError(f"unknown S5 raw-weight control: {control}")
        if query_tokens.ndim != 3 or query_tokens.shape[-1] != self.d_model:
            raise ValueError("Raw Query conditioner expects Query tokens [B,T,1024]")
        if query_mask.shape != query_tokens.shape[:2]:
            raise ValueError("Raw Query conditioner mask must have shape [B,T]")
        valid = query_mask.to(device=query_tokens.device, dtype=torch.bool)
        if control == "zero":
            if layer_permutation is not None or module_permutation is not None:
                raise ValueError("zero Raw-Weight control cannot carry a permutation")
            return query_tokens * valid[:, :, None].to(dtype=query_tokens.dtype)
        memories = self._controlled_weight_memory(
            control=control,
            layer_permutation=layer_permutation,
            module_permutation=module_permutation,
        ).to(device=query_tokens.device, dtype=query_tokens.dtype)
        # The axis compressors already emit 16 learned memory tokens per
        # matrix.  Their mean is a compact, fixed-cardinality matrix token that
        # keeps Query cross-attention tractable for large batches.
        matrix_tokens = memories.mean(dim=1)
        memory = matrix_tokens.unsqueeze(0).expand(query_tokens.shape[0], -1, -1)
        attended, _ = self.weight_cross_attention(
            query_tokens,
            memory,
            memory,
            need_weights=False,
        )
        bounded = self.residual_cap * torch.tanh(self.output_projection(attended.float()))
        bounded = bounded * valid[:, :, None].to(dtype=bounded.dtype)
        valid_count = valid.sum().to(dtype=torch.float32) * float(self.d_model)
        mean_square = bounded.float().square().sum() / valid_count.clamp_min(1.0)
        if self.residual_rms_ceiling is None:
            ceiling_scale = torch.ones_like(mean_square)
        else:
            ceiling_scale = self.residual_rms_ceiling / mean_square.clamp_min(
                self.residual_rms_ceiling**2
            ).sqrt()
        scaled = bounded * ceiling_scale
        self._last_residual_rms = (
            scaled.float().square().sum().div(valid_count.clamp_min(1.0)).sqrt().detach()
        )
        self._last_residual_max_abs = scaled.detach().float().abs().max()
        return (
            query_tokens.float() + scaled
        ).to(dtype=query_tokens.dtype) * valid[:, :, None].to(dtype=query_tokens.dtype)


class _S5RawTransformerModel(TransformerModel):
    def __init__(self, *args: Any, raw_weight_conditioner: S5SafeP2ERawWeightConditioner, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_weight_conditioner = raw_weight_conditioner

    def forward(
        self,
        encoder_hidden_states: Tensor,
        condition_attention_mask: Tensor | None = None,
        *,
        raw_weight_control: str = "matched",
        raw_weight_layer_permutation: tuple[int, ...] | None = None,
        raw_weight_module_permutation: tuple[int, ...] | None = None,
    ) -> Tensor:
        if encoder_hidden_states.dim() == 4 and encoder_hidden_states.shape[1] == 1:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        encoder_hidden_states = self.norm_input(encoder_hidden_states)
        hw_freqs, n_freqs, l_freqs, all_freqs = self._get_pos_embed(
            device=encoder_hidden_states.device,
            cond_seq_length=encoder_hidden_states.shape[1],
        )
        x = self.get_tokens(batch_size=encoder_hidden_states.shape[0], device=encoder_hidden_states.device)
        x = self.apply_condition_summary(x=x, encoder_hidden_states=encoder_hidden_states, condition_attention_mask=condition_attention_mask)
        for index, block in enumerate(self.blocks, start=1):
            x = block(
                x=x,
                encoder_hidden_states=encoder_hidden_states,
                hw_freqs=hw_freqs,
                n_freqs=n_freqs,
                l_freqs=l_freqs,
                all_freqs=all_freqs,
                condition_attention_mask=condition_attention_mask,
            )
            if index == 1:
                x = self.raw_weight_conditioner(
                    x,
                    control=raw_weight_control,
                    layer_permutation=raw_weight_layer_permutation,
                    module_permutation=raw_weight_module_permutation,
                )
        return self.post_process(x)


class S5RawParameterGenerator(S5ParameterGenerator):
    """S5 PG whose Parameter Tokens receive Safe-P2E after block 1."""

    def __init__(self, config: Any, raw_weight_conditioner: S5SafeP2ERawWeightConditioner) -> None:
        # This mirrors s2_hypernet.ParameterGenerator.__init__ while replacing
        # only the TransformerModel implementation.  The state names of the
        # shared PG trunk remain unchanged; the extra conditioner is explicit.
        PreTrainedModel.__init__(self, config)
        self.tokenizer = Tokenizer2DBatchedLoRA(
            token_dim=config.token_dim,
            rank=config.rank,
            alpha=config.alpha,
            pg_mapping=config.pg_mapping,
        )
        self.lora_A_token_count = self.tokenizer.lora_A_token_count
        self.lora_B_token_count = self.tokenizer.lora_B_token_count
        self.model = _S5RawTransformerModel(
            d_model=config.d_model,
            num_base_model_layers=config.num_base_model_layers,
            num_token_per_layer=self.lora_A_token_count + self.lora_B_token_count,
            lora_rank=config.rank,
            output_dim=config.output_dim,
            head_dim=config.head_dim,
            num_blocks=config.num_pg_layers,
            dim_accumulation=config.dim_accumulation,
            lora_A_token_count=self.lora_A_token_count,
            lora_B_token_count=self.lora_B_token_count,
            condition_injection_mode=config.condition_injection_mode,
            condition_summary_scale_init=config.condition_summary_scale_init,
            raw_weight_conditioner=raw_weight_conditioner,
        )
        self.layer_num = config.num_base_model_layers
        self.target_layer_indices = list(config.target_layer_indices)
        self.prefix = config.prefix
        self.hidden_in = nn.Linear(config.input_dim, config.d_model)
        self.post_init()
        self.model.reset_condition_injection()
        nn.init.zeros_(self.model.proj_out_B.weight)
        nn.init.zeros_(self.model.proj_out_B.bias)
        nn.init.zeros_(self.model.raw_weight_conditioner.output_projection.weight)

    def forward(
        self,
        condition: Tensor,
        condition_attention_mask: Tensor | None = None,
        *,
        raw_weight_control: str = "matched",
        raw_weight_layer_permutation: tuple[int, ...] | None = None,
        raw_weight_module_permutation: tuple[int, ...] | None = None,
    ) -> dict[str, Tensor]:
        embeddings = self.hidden_in(condition)
        output = self.model(
            encoder_hidden_states=embeddings,
            condition_attention_mask=condition_attention_mask,
            raw_weight_control=raw_weight_control,
            raw_weight_layer_permutation=raw_weight_layer_permutation,
            raw_weight_module_permutation=raw_weight_module_permutation,
        )
        all_layer_state_dict: dict[str, Tensor] = {}
        for seq_idx, layer_idx in enumerate(self.target_layer_indices):
            layer_state = self.tokenizer.detokenize(self.tokenizer.shape_state_dict, output[:, seq_idx])
            for key, value in layer_state.items():
                all_layer_state_dict[f"{self.prefix}{layer_idx}.{key}"] = value
        all_layer_state_dict["scale"] = torch.tensor(self.tokenizer.lora_scale, device=output.device, dtype=output.dtype)
        return all_layer_state_dict


def build_s5_raw_parameter_generator(
    *,
    cache: S5RawWeightCache,
    layout: S5TargetLayout | None = None,
) -> S5RawParameterGenerator:
    from .parameter_generator import build_s5_parameter_generator_config

    layout = layout or build_s5_target_layout()
    config = build_s5_parameter_generator_config(layout=layout)
    conditioner = S5SafeP2ERawWeightConditioner(cache)
    return S5RawParameterGenerator(config, conditioner)


__all__ = [
    "S5RawMatrixTarget",
    "S5RawWeightCache",
    "S5SafeP2ERawWeightConditioner",
    "S5RawParameterGenerator",
    "S5RawQueryEncoderConditioner",
    "build_s5_raw_parameter_generator",
    "build_s5_raw_weight_cache",
    "s5_raw_matrix_targets",
]

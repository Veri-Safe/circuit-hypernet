from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import torch
from torch import Tensor, nn

from .cache import FilmFeatureCache, P2EFeatureCache, fixed_derangement
from .routing import ParameterTokenRouting


def _derived_seed(seed: int, *parts: object) -> int:
    payload = ":".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _random_like_by_coordinate(values: Tensor, *, seed: int) -> Tensor:
    if values.dim() < 2:
        raise ValueError("Coordinate-matched random values require a leading sample dimension")
    mean = values.float().mean(dim=0)
    std = values.float().std(dim=0, unbiased=False)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(values.shape, generator=generator, dtype=torch.float32)
    return (noise * std + mean).to(dtype=values.dtype)


class IdentityRawWeightConditioner(nn.Module):
    method = "none"

    def forward(self, parameter_tokens: Tensor) -> Tensor:
        return parameter_tokens


class FilmRawWeightConditioner(nn.Module):
    method = "film"

    def __init__(
        self,
        cache: FilmFeatureCache,
        *,
        routing: ParameterTokenRouting,
        d_model: int = 1280,
        encoder_hidden_dim: int = 512,
        gate_init: float = 0.05,
        content: str = "real",
        control_seed: int = 24,
    ) -> None:
        super().__init__()
        if content not in {"real", "zero", "random", "shuffled_layer"}:
            raise ValueError(f"Unsupported FiLM content mode: {content!r}")
        if d_model <= 0 or encoder_hidden_dim <= 0:
            raise ValueError("FiLM dimensions must be positive")
        if routing.valid_slot_module_mask.shape != cache.valid_mask.shape:
            raise ValueError("FiLM cache mask does not match Parameter Token routing")
        if not torch.equal(routing.valid_slot_module_mask, cache.valid_mask):
            raise ValueError("FiLM cache valid targets differ from the Qwen route mask")

        prepared = self._prepare_features(
            cache.features,
            cache.valid_mask,
            content=content,
            seed=control_seed,
        )
        self.content = content
        self.d_model = int(d_model)
        self.register_buffer("features", prepared, persistent=False)
        self.register_buffer("valid_mask", cache.valid_mask.clone(), persistent=False)
        self.register_buffer(
            "token_to_module", routing.token_to_module.clone(), persistent=False
        )
        self.encoder = nn.Sequential(
            nn.Linear(1025, encoder_hidden_dim),
            nn.SiLU(),
            nn.Linear(encoder_hidden_dim, d_model),
            nn.RMSNorm(d_model, elementwise_affine=False, eps=1.0e-6),
        )
        self.film_head = nn.Linear(d_model, d_model * 2, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self.reset_parameters()

    @staticmethod
    def _prepare_features(
        features: Tensor,
        valid_mask: Tensor,
        *,
        content: str,
        seed: int,
    ) -> Tensor:
        prepared = features.detach().clone().float()
        if content in {"real", "zero"}:
            return prepared
        valid_values = prepared[valid_mask]
        random_values = _random_like_by_coordinate(valid_values, seed=seed)
        random_prepared = torch.zeros_like(prepared)
        random_prepared[valid_mask] = random_values
        if content == "random":
            return random_prepared

        result = torch.zeros_like(prepared)
        layer_count = valid_mask.shape[0] - 1
        transformer_module_count = valid_mask.shape[1] - 1
        for module_index in range(transformer_module_count):
            permutation = fixed_derangement(
                layer_count,
                seed=_derived_seed(seed, "film", module_index),
            )
            result[:layer_count, module_index] = prepared[
                permutation, module_index
            ]
        result[layer_count:] = random_prepared[layer_count:]
        return result

    def reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        nn.init.normal_(
            self.film_head.weight,
            mean=0.0,
            std=1.0 / math.sqrt(2.0 * self.d_model),
        )

    def forward(self, parameter_tokens: Tensor) -> Tensor:
        if parameter_tokens.dim() != 4:
            raise ValueError("Parameter Tokens must have shape [batch, slot, token, channel]")
        if parameter_tokens.shape[1:3] != (
            self.valid_mask.shape[0],
            self.token_to_module.numel(),
        ):
            raise ValueError("Parameter Token grid does not match the FiLM route")
        if parameter_tokens.shape[-1] != self.d_model:
            raise ValueError("Parameter Token width does not match the FiLM conditioner")
        if self.content == "zero":
            return parameter_tokens

        descriptors = self.encoder(
            self.features.to(device=parameter_tokens.device, dtype=parameter_tokens.dtype)
        )
        gamma, beta = self.film_head(descriptors).chunk(2, dim=-1)
        mask = self.valid_mask.to(device=parameter_tokens.device)[..., None]
        gamma = gamma.masked_fill(~mask, 0.0)
        beta = beta.masked_fill(~mask, 0.0)
        token_route = self.token_to_module.to(device=parameter_tokens.device)
        gamma = gamma[:, token_route]
        beta = beta[:, token_route]
        gate = self.gate.to(device=parameter_tokens.device, dtype=parameter_tokens.dtype)
        return parameter_tokens * (1.0 + gate * gamma[None]) + gate * beta[None]


class AxisQueryCompressor(nn.Module):
    def __init__(
        self,
        *,
        width: int = 128,
        query_count: int = 4,
        heads: int = 4,
        ffn_dim: int = 512,
    ) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("Axis compressor width must be divisible by its head count")
        self.queries = nn.Parameter(torch.empty(query_count, width))
        self.query_norm = nn.RMSNorm(width, elementwise_affine=True, eps=1.0e-6)
        self.value_norm = nn.RMSNorm(width, elementwise_affine=True, eps=1.0e-6)
        self.cross_attention = nn.MultiheadAttention(
            width,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn_norm = nn.RMSNorm(width, elementwise_affine=True, eps=1.0e-6)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.SiLU(),
            nn.Linear(ffn_dim, width),
        )
        self.final_norm = nn.RMSNorm(width, elementwise_affine=False, eps=1.0e-6)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.queries, mean=0.0, std=0.02)

    def forward(self, values: Tensor) -> Tensor:
        if values.dim() != 3:
            raise ValueError("Axis values must have shape [matrix, axis_position, width]")
        queries = self.queries.to(dtype=values.dtype).unsqueeze(0).expand(
            values.shape[0], -1, -1
        )
        normalized_values = self.value_norm(values)
        attended, _ = self.cross_attention(
            self.query_norm(queries),
            normalized_values,
            normalized_values,
            need_weights=False,
        )
        hidden = queries + attended
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return self.final_norm(hidden)


def normalized_fourier_coordinates(
    length: int,
    *,
    width: int = 128,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if width != 128:
        raise ValueError("P2E-QV v1 freezes the coordinate width at 128")
    if length <= 0:
        raise ValueError("Coordinate length must be positive")
    position = (torch.arange(length, device=device, dtype=torch.float32) + 0.5) / length
    frequency = torch.arange(1, 65, device=device, dtype=torch.float32)
    phase = 2.0 * torch.pi * position[:, None] * frequency[None, :]
    return torch.stack((phase.sin(), phase.cos()), dim=-1).reshape(length, width).to(dtype=dtype)


class P2ERawWeightConditioner(nn.Module):
    method = "p2e_qv"

    def __init__(
        self,
        cache: P2EFeatureCache,
        *,
        routing: ParameterTokenRouting,
        d_model: int = 1280,
        width: int = 128,
        query_count: int = 4,
        compressor_heads: int = 4,
        compressor_ffn_dim: int = 512,
        weight_attention_heads: int | None = None,
        gate_init: float = 0.01,
        content: str = "real",
        control_seed: int = 24,
    ) -> None:
        super().__init__()
        if content not in {"real", "zero", "random", "shuffled_layer"}:
            raise ValueError(f"Unsupported P2E content mode: {content!r}")
        if width != 128 or compressor_heads != 4 or compressor_ffn_dim != 512:
            raise ValueError("P2E-QV freezes width/head/FFN to 128/4/512")
        if query_count not in {4, 8, 16}:
            raise ValueError("P2E-QV query_count must be one of 4, 8, or 16")
        if weight_attention_heads is None:
            if d_model % 128:
                raise ValueError("P2E weight attention requires d_model divisible by head_dim=128")
            weight_attention_heads = d_model // 128
        if d_model % weight_attention_heads:
            raise ValueError("P2E d_model must be divisible by weight_attention_heads")

        row_values, column_values, log_rms, source_indices = self._prepare_content(
            cache,
            content=content,
            seed=control_seed,
        )
        self.content = content
        self.d_model = int(d_model)
        self.width = int(width)
        self.query_count = int(query_count)
        self.register_buffer("row_values", row_values, persistent=False)
        self.register_buffer("row_offsets", cache.row_offsets.clone(), persistent=False)
        self.register_buffer("column_values", column_values, persistent=False)
        self.register_buffer("column_offsets", cache.column_offsets.clone(), persistent=False)
        self.register_buffer("standardized_log_rms", log_rms, persistent=False)
        self.register_buffer("target_slots", cache.target_slots.clone(), persistent=False)
        self.register_buffer("target_modules", cache.target_modules.clone(), persistent=False)
        self.register_buffer("source_indices", source_indices, persistent=False)
        self.register_buffer(
            "token_to_module", routing.token_to_module.clone(), persistent=False
        )

        self.axis_type_embeddings = nn.Parameter(torch.empty(2, width))
        self.scale_direction = nn.Linear(1, width, bias=False)
        self.input_norm = nn.RMSNorm(width, elementwise_affine=False, eps=1.0e-6)
        self.row_compressor = AxisQueryCompressor(
            width=width,
            query_count=query_count,
            heads=compressor_heads,
            ffn_dim=compressor_ffn_dim,
        )
        self.column_compressor = AxisQueryCompressor(
            width=width,
            query_count=query_count,
            heads=compressor_heads,
            ffn_dim=compressor_ffn_dim,
        )
        self.memory_projection = nn.Linear(width, d_model, bias=False)
        self.memory_norm = nn.RMSNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.weight_cross_attention = nn.MultiheadAttention(
            d_model,
            weight_attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))
        self._coordinate_buffer_names: dict[int, str] = {}
        self._axis_segments = {
            "row": self._build_axis_segments(cache.row_offsets),
            "column": self._build_axis_segments(cache.column_offsets),
        }
        lengths = set((cache.row_offsets[1:] - cache.row_offsets[:-1]).tolist())
        lengths.update((cache.column_offsets[1:] - cache.column_offsets[:-1]).tolist())
        for length in sorted(int(item) for item in lengths):
            name = f"coordinate_{length}"
            self.register_buffer(
                name,
                normalized_fourier_coordinates(length, width=width),
                persistent=False,
            )
            self._coordinate_buffer_names[length] = name
        self._weight_route_buffer_names = self._register_weight_routes()
        self._eval_weight_memory: Tensor | None = None
        self._eval_weight_memory_signature: tuple[object, ...] | None = None
        self.reset_parameters()
        self._validate_qv_routes(routing)

    def _register_weight_routes(self) -> tuple[tuple[str, str], ...]:
        route_names: list[tuple[str, str]] = []
        module_indices = sorted(
            int(value) for value in self.target_modules.unique().tolist()
        )
        for module_index in module_indices:
            token_name = f"p2e_route_token_indices_{module_index}"
            matrix_name = f"p2e_route_matrix_indices_{module_index}"
            token_indices = torch.nonzero(
                self.token_to_module == module_index,
                as_tuple=False,
            ).flatten()
            matrix_indices = torch.nonzero(
                self.target_modules == module_index,
                as_tuple=False,
            ).flatten()
            order = torch.argsort(self.target_slots[matrix_indices])
            self.register_buffer(token_name, token_indices, persistent=False)
            self.register_buffer(
                matrix_name,
                matrix_indices[order],
                persistent=False,
            )
            route_names.append((token_name, matrix_name))
        return tuple(route_names)

    def _weight_routes(self) -> tuple[tuple[Tensor, Tensor], ...]:
        return tuple(
            (getattr(self, token_name), getattr(self, matrix_name))
            for token_name, matrix_name in self._weight_route_buffer_names
        )

    def _weight_memory_cache_signature(self) -> tuple[object, ...]:
        tensors = [
            self.row_values,
            self.column_values,
            self.standardized_log_rms,
            self.axis_type_embeddings,
            *self.scale_direction.parameters(),
            *self.row_compressor.parameters(),
            *self.column_compressor.parameters(),
            *self.memory_projection.parameters(),
            *(
                getattr(self, name)
                for name in self._coordinate_buffer_names.values()
            ),
        ]
        return tuple(
            (tensor.device.type, tensor.device.index, tensor.dtype, tensor._version)
            for tensor in tensors
        )

    def _clear_weight_memory_cache(self) -> None:
        self._eval_weight_memory = None
        self._eval_weight_memory_signature = None

    def train(self, mode: bool = True) -> P2ERawWeightConditioner:
        if mode:
            self._clear_weight_memory_cache()
        return super().train(mode)

    @staticmethod
    def _build_axis_segments(offsets: Tensor) -> dict[int, tuple[tuple[int, int, int], ...]]:
        groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        offsets_cpu = offsets.detach().cpu().tolist()
        for matrix_index, (start, end) in enumerate(
            zip(offsets_cpu[:-1], offsets_cpu[1:], strict=True)
        ):
            groups[int(end - start)].append((matrix_index, int(start), int(end)))
        return {length: tuple(items) for length, items in groups.items()}

    @staticmethod
    def _prepare_content(
        cache: P2EFeatureCache,
        *,
        content: str,
        seed: int,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        row = cache.row_values.detach().clone().float()
        column = cache.column_values.detach().clone().float()
        log_rms = cache.standardized_log_rms.detach().clone().float()
        source_indices = torch.arange(cache.matrix_count, dtype=torch.long)
        if content in {"real", "zero"}:
            return row, column, log_rms, source_indices

        groups: dict[int, list[int]] = defaultdict(list)
        for index, module_index in enumerate(cache.target_modules.tolist()):
            groups[int(module_index)].append(index)
        if content == "shuffled_layer":
            for module_index, indices in groups.items():
                permutation = fixed_derangement(
                    len(indices),
                    seed=_derived_seed(seed, "p2e", module_index),
                )
                for target_offset, source_offset in enumerate(permutation.tolist()):
                    source_indices[indices[target_offset]] = indices[source_offset]
            row_result = torch.empty_like(row)
            column_result = torch.empty_like(column)
            log_result = torch.empty_like(log_rms)
            for target_index, source_index in enumerate(source_indices.tolist()):
                row_start, row_end = (
                    int(cache.row_offsets[target_index]),
                    int(cache.row_offsets[target_index + 1]),
                )
                source_row_start, source_row_end = (
                    int(cache.row_offsets[source_index]),
                    int(cache.row_offsets[source_index + 1]),
                )
                col_start, col_end = (
                    int(cache.column_offsets[target_index]),
                    int(cache.column_offsets[target_index + 1]),
                )
                source_col_start, source_col_end = (
                    int(cache.column_offsets[source_index]),
                    int(cache.column_offsets[source_index + 1]),
                )
                if (row_end - row_start, col_end - col_start) != (
                    source_row_end - source_row_start,
                    source_col_end - source_col_start,
                ):
                    raise ValueError("P2E shuffled control crossed incompatible matrix shapes")
                row_result[row_start:row_end] = row[source_row_start:source_row_end]
                column_result[col_start:col_end] = column[source_col_start:source_col_end]
                log_result[target_index] = log_rms[source_index]
            return row_result, column_result, log_result, source_indices

        row_result = torch.empty_like(row)
        column_result = torch.empty_like(column)
        log_result = torch.empty_like(log_rms)
        for module_index, indices in groups.items():
            row_stack = torch.stack([cache.axis_slice(index, axis="row") for index in indices])
            column_stack = torch.stack(
                [cache.axis_slice(index, axis="column") for index in indices]
            )
            random_rows = _random_like_by_coordinate(
                row_stack,
                seed=_derived_seed(seed, "p2e-random-row", module_index),
            )
            random_columns = _random_like_by_coordinate(
                column_stack,
                seed=_derived_seed(seed, "p2e-random-column", module_index),
            )
            random_log = _random_like_by_coordinate(
                log_rms[indices, None],
                seed=_derived_seed(seed, "p2e-random-log", module_index),
            ).squeeze(-1)
            for offset, target_index in enumerate(indices):
                row_start, row_end = (
                    int(cache.row_offsets[target_index]),
                    int(cache.row_offsets[target_index + 1]),
                )
                col_start, col_end = (
                    int(cache.column_offsets[target_index]),
                    int(cache.column_offsets[target_index + 1]),
                )
                row_result[row_start:row_end] = random_rows[offset]
                column_result[col_start:col_end] = random_columns[offset]
                log_result[target_index] = random_log[offset]
        return row_result, column_result, log_result, source_indices

    def _validate_qv_routes(self, routing: ParameterTokenRouting) -> None:
        expected_names = {"self_attn.q_proj", "self_attn.v_proj"}
        actual_names = {
            routing.module_names[index]
            for index in self.target_modules.unique().tolist()
        }
        if actual_names != expected_names:
            raise ValueError(f"P2E-QV cache routes changed: {actual_names} != {expected_names}")
        for name in expected_names:
            module_index = routing.module_names.index(name)
            slots = self.target_slots[self.target_modules == module_index]
            if not torch.equal(slots.cpu(), torch.arange(36)):
                raise ValueError(f"P2E-QV {name} cache must contain layers 0..35 in order")

    def reset_parameters(self) -> None:
        nn.init.normal_(self.axis_type_embeddings, mean=0.0, std=0.02)
        self.scale_direction.reset_parameters()
        self.memory_projection.reset_parameters()

    def _coordinate(self, length: int, *, reference: Tensor) -> Tensor:
        return getattr(self, self._coordinate_buffer_names[length]).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    def _compress_axis(
        self,
        *,
        values: Tensor,
        axis: str,
        axis_index: int,
        compressor: AxisQueryCompressor,
    ) -> Tensor:
        output = torch.empty(
            self.target_slots.numel(),
            self.query_count,
            self.width,
            device=values.device,
            dtype=values.dtype,
        )
        scale = self.scale_direction(
            self.standardized_log_rms.to(device=values.device, dtype=values.dtype)[:, None]
        )
        for length, segments in self._axis_segments[axis].items():
            matrix_indices = [item[0] for item in segments]
            packed = torch.stack(
                [values[start:end] for _, start, end in segments],
                dim=0,
            )
            coordinate = self._coordinate(length, reference=packed)
            type_embedding = self.axis_type_embeddings[axis_index].to(dtype=packed.dtype)
            packed = self.input_norm(
                packed
                + coordinate[None]
                + type_embedding[None, None]
                + scale[matrix_indices, None]
            )
            compressed = compressor(packed)
            output[matrix_indices] = compressed
        return output

    def build_weight_memory(self) -> Tensor:
        row_values = self.row_values
        column_values = self.column_values
        row_latents = self._compress_axis(
            values=row_values,
            axis="row",
            axis_index=0,
            compressor=self.row_compressor,
        )
        column_latents = self._compress_axis(
            values=column_values,
            axis="column",
            axis_index=1,
            compressor=self.column_compressor,
        )
        latents = torch.cat((row_latents, column_latents), dim=1)
        return self.memory_norm(self.memory_projection(latents))

    def weight_memory_for_forward(self) -> Tensor:
        """Reuse fixed weight memory only in gradient-free evaluation."""

        if self.training or torch.is_grad_enabled():
            return self.build_weight_memory()
        signature = self._weight_memory_cache_signature()
        if (
            self._eval_weight_memory is None
            or self._eval_weight_memory_signature != signature
        ):
            self._eval_weight_memory = self.build_weight_memory().detach()
            self._eval_weight_memory_signature = signature
        return self._eval_weight_memory

    def forward(self, parameter_tokens: Tensor) -> Tensor:
        if parameter_tokens.dim() != 4:
            raise ValueError("Parameter Tokens must have shape [batch, slot, token, channel]")
        if parameter_tokens.shape[1] != 37 or parameter_tokens.shape[2] != self.token_to_module.numel():
            raise ValueError("Parameter Token grid does not match the P2E route")
        if parameter_tokens.shape[-1] != self.d_model:
            raise ValueError("Parameter Token width does not match the P2E conditioner")
        if self.content == "zero":
            return parameter_tokens

        memories = self.weight_memory_for_forward().to(dtype=parameter_tokens.dtype)
        result = parameter_tokens.clone()
        batch_size = parameter_tokens.shape[0]
        for token_indices, matrix_indices in self._weight_routes():
            query = parameter_tokens[:, :36, token_indices, :]
            query_flat = query.reshape(batch_size * 36, token_indices.numel(), self.d_model)
            memory = memories[matrix_indices]
            memory_flat = memory[None].expand(batch_size, -1, -1, -1).reshape(
                batch_size * 36,
                memory.shape[1],
                self.d_model,
            )
            delta, _ = self.weight_cross_attention(
                query_flat,
                memory_flat,
                memory_flat,
                need_weights=False,
            )
            delta = delta.reshape_as(query)
            gate = self.gate.to(device=parameter_tokens.device, dtype=parameter_tokens.dtype)
            result[:, :36, token_indices, :] = query + gate * delta
        return result

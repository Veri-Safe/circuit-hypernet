from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from s2_hypernet.nvarc_pg.target_layout import QwenTargetLayout
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.cache import (
    FilmFeatureCache,
    RMS_EPSILON,
    layout_matrix_targets,
    semi_orthogonal_projection,
)
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.conditioning import (
    AxisQueryCompressor,
    normalized_fourier_coordinates,
)
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.routing import (
    ParameterTokenRouting,
)


HISTORY_FEATURE_WIDTH = 1025


def merge_projected_sketch(
    base_sketch: Tensor,
    lora_a: Tensor,
    lora_b: Tensor,
    projection_in: Tensor,
    projection_out: Tensor,
    *,
    scale: float | Tensor,
) -> Tensor:
    """Compute P_out^T (W0 + scale * B A) P_in without materializing W_eff."""

    if lora_a.dim() != 3 or lora_b.dim() != 3:
        raise ValueError("LoRA factors must have shapes [batch, rank, in] and [batch, out, rank]")
    if lora_a.shape[0] != lora_b.shape[0] or lora_a.shape[1] != lora_b.shape[2]:
        raise ValueError("LoRA A/B batch or rank dimensions do not align")
    if lora_a.shape[2] != projection_in.shape[0]:
        raise ValueError("LoRA A input dimension does not match P_in")
    if lora_b.shape[1] != projection_out.shape[0]:
        raise ValueError("LoRA B output dimension does not match P_out")
    expected = (projection_out.shape[1], projection_in.shape[1])
    if tuple(base_sketch.shape) != expected:
        raise ValueError(f"Base sketch shape {tuple(base_sketch.shape)} != {expected}")

    a_projected = torch.matmul(lora_a.float(), projection_in.float())
    b_projected = torch.matmul(
        projection_out.float().transpose(0, 1),
        lora_b.float(),
    )
    delta = torch.matmul(b_projected, a_projected)
    scale_tensor = torch.as_tensor(scale, device=delta.device, dtype=delta.dtype)
    return base_sketch.float().unsqueeze(0) + scale_tensor * delta


def pack_history_feature_batch(
    histories: Sequence[Sequence[Tensor]],
    *,
    matrix_count: int,
    max_history: int,
    history_mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Right-pad detached per-event features into [batch, history, matrix, 1025]."""

    if not histories:
        raise ValueError("histories must contain at least one batch item")
    if history_mode not in {"all", "last1"}:
        raise ValueError(f"Unknown history_mode: {history_mode!r}")
    selected: list[tuple[Tensor, ...]] = []
    for history in histories:
        items = tuple(history[-1:] if history_mode == "last1" else history)
        if len(items) > max_history:
            raise ValueError(
                f"History length {len(items)} exceeds configured max_history={max_history}"
            )
        for item in items:
            if item.shape != (matrix_count, HISTORY_FEATURE_WIDTH):
                raise ValueError(
                    "History feature shape changed: "
                    f"{tuple(item.shape)} != {(matrix_count, HISTORY_FEATURE_WIDTH)}"
                )
        selected.append(items)

    width = max((len(items) for items in selected), default=0)
    if width == 0:
        return (
            torch.zeros(
                len(selected),
                0,
                matrix_count,
                HISTORY_FEATURE_WIDTH,
                device=device,
                dtype=dtype,
            ),
            torch.zeros(len(selected), 0, device=device, dtype=torch.bool),
        )
    packed = torch.zeros(
        len(selected),
        width,
        matrix_count,
        HISTORY_FEATURE_WIDTH,
        device=device,
        dtype=dtype,
    )
    mask = torch.zeros(len(selected), width, device=device, dtype=torch.bool)
    for batch_index, items in enumerate(selected):
        for history_index, item in enumerate(items):
            packed[batch_index, history_index].copy_(
                item.detach().to(device=device, dtype=dtype)
            )
            mask[batch_index, history_index] = True
    return packed, mask


class HistoryEffectiveWeightConditioner(nn.Module):
    """Condition Parameter Tokens on detached historical W0 + 2BA sketches."""

    method = "safe_history_p2e"
    output_parameter_names = ("output_projection.weight",)

    def __init__(
        self,
        cache: FilmFeatureCache,
        *,
        layout: QwenTargetLayout,
        routing: ParameterTokenRouting,
        d_model: int = 1280,
        width: int = 128,
        query_count: int = 8,
        compressor_heads: int = 4,
        compressor_ffn_dim: int = 512,
        weight_attention_heads: int | None = None,
        residual_cap: float = 0.025,
        residual_rms_ceiling: float | None = 0.005,
        content: str = "matched_history",
        history_mode: str = "all",
        max_history: int = 8,
        projection_seed: int = 24,
    ) -> None:
        super().__init__()
        if content not in {"matched_history", "static_history"}:
            raise ValueError(f"Unsupported history content: {content!r}")
        if history_mode not in {"all", "last1"}:
            raise ValueError(f"Unsupported history_mode: {history_mode!r}")
        if not 1 <= int(max_history) <= 8:
            raise ValueError("max_history must be in 1..8")
        if width != 128 or compressor_heads != 4 or compressor_ffn_dim != 512:
            raise ValueError("History-P2E freezes width/head/FFN to 128/4/512")
        if query_count not in {8, 16}:
            raise ValueError("History-P2E query_count must be 8 or 16")
        if residual_cap != 0.025:
            raise ValueError("History-P2E freezes the element residual cap at 0.025")
        if residual_rms_ceiling is not None and not (
            0.0 < float(residual_rms_ceiling) <= residual_cap
        ):
            raise ValueError("residual_rms_ceiling must be positive and <= residual_cap")
        if projection_seed != 24:
            raise ValueError("History-P2E freezes projection_seed=24")
        if routing.valid_slot_module_mask.shape != cache.valid_mask.shape:
            raise ValueError("History-P2E cache mask does not match Parameter Token routing")
        if not torch.equal(routing.valid_slot_module_mask, cache.valid_mask):
            raise ValueError("History-P2E cache target mask differs from the route mask")
        targets = layout_matrix_targets(layout)
        if tuple(target.target_key for target in targets) != cache.target_keys:
            raise ValueError("History-P2E cache target order differs from the Qwen layout")
        if weight_attention_heads is None:
            if d_model % 128:
                raise ValueError("History weight attention requires d_model divisible by 128")
            weight_attention_heads = d_model // 128
        if d_model % weight_attention_heads:
            raise ValueError("d_model must be divisible by weight_attention_heads")

        self.content = content
        self.history_mode = history_mode
        self.max_history = int(max_history)
        self.d_model = int(d_model)
        self.width = int(width)
        self.query_count = int(query_count)
        self.residual_cap = float(residual_cap)
        self.residual_rms_ceiling = (
            None if residual_rms_ceiling is None else float(residual_rms_ceiling)
        )
        self.scale = float(layout.scale)
        self.targets = targets
        self.matrix_count = len(targets)
        self.register_buffer(
            "target_slots",
            torch.tensor([target.slot for target in targets], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "target_modules",
            torch.tensor([target.module_index for target in targets], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "token_to_module", routing.token_to_module.clone(), persistent=False
        )
        self.module_names = routing.module_names
        self._matrix_indices_by_module = tuple(
            tuple(
                index
                for index, target in enumerate(targets)
                if target.module_index == module_index
            )
            for module_index in range(len(routing.module_names))
        )
        for module_index, matrix_indices in enumerate(self._matrix_indices_by_module):
            self.register_buffer(
                f"matrix_indices_{module_index}",
                torch.tensor(matrix_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"parameter_token_indices_{module_index}",
                torch.nonzero(
                    routing.token_to_module == module_index,
                    as_tuple=False,
                ).flatten(),
                persistent=False,
            )

        base_sketches: list[Tensor] = []
        for target in targets:
            normalized = cache.features[
                target.slot, target.module_index, :1024
            ].reshape(16, 64) if target.module_name == "lm_head" else cache.features[
                target.slot, target.module_index, :1024
            ].reshape(32, 32)
            standardized_log_rms = cache.features[
                target.slot, target.module_index, 1024
            ]
            log_weight_rms = standardized_log_rms * cache.log_rms_std + cache.log_rms_mean
            base_sketches.append(normalized.float() * log_weight_rms.exp())
        base_actual = torch.stack([item.reshape(-1) for item in base_sketches])
        base_rms = base_actual.square().mean(dim=1).add(RMS_EPSILON).sqrt()
        base_log_rms = base_rms.log()
        log_mean = base_log_rms.mean()
        log_std = base_log_rms.std(unbiased=False).clamp_min(1.0e-12)
        base_features = torch.cat(
            (
                base_actual / base_rms[:, None],
                ((base_log_rms - log_mean) / log_std)[:, None],
            ),
            dim=1,
        )
        self.register_buffer("base_sketches", base_actual, persistent=False)
        self.register_buffer("base_features", base_features, persistent=False)
        self.register_buffer("base_log_sketch_rms_mean", log_mean, persistent=False)
        self.register_buffer("base_log_sketch_rms_std", log_std, persistent=False)

        for module_index, module in enumerate(layout.all_generator_modules):
            input_width = 64 if module.name == "lm_head" else 32
            output_width = 16 if module.name == "lm_head" else 32
            self.register_buffer(
                f"projection_in_{module_index}",
                semi_orthogonal_projection(
                    module.in_features, input_width, seed=projection_seed
                ),
                persistent=False,
            )
            projection_out = (
                torch.eye(16, dtype=torch.float32)
                if module.name == "lm_head"
                else semi_orthogonal_projection(
                    module.out_features, output_width, seed=projection_seed
                )
            )
            self.register_buffer(
                f"projection_out_{module_index}",
                projection_out,
                persistent=False,
            )

        self.square_value_projection = nn.Linear(32, width)
        self.lm_row_projection = nn.Linear(64, width)
        self.lm_column_projection = nn.Linear(16, width)
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
        self.layer_embeddings = nn.Embedding(layout.generator_slot_count, d_model)
        self.module_embeddings = nn.Embedding(len(routing.module_names), d_model)
        self.age_embeddings = nn.Embedding(self.max_history + 1, d_model)
        self.memory_norm = nn.RMSNorm(d_model, elementwise_affine=False, eps=1.0e-6)
        self.weight_cross_attention = nn.MultiheadAttention(
            d_model,
            weight_attention_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self._coordinate_names: dict[int, str] = {}
        for length in (16, 32, 64):
            name = f"coordinate_{length}"
            self.register_buffer(
                name,
                normalized_fourier_coordinates(length, width=width),
                persistent=False,
            )
            self._coordinate_names[length] = name
        self._last_pre_ceiling_residual_rms: Tensor | None = None
        self._last_rms_ceiling_scale: Tensor | None = None
        self._last_residual_rms: Tensor | None = None
        self._last_residual_max_abs: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.axis_type_embeddings, mean=0.0, std=0.02)
        self.square_value_projection.reset_parameters()
        self.lm_row_projection.reset_parameters()
        self.lm_column_projection.reset_parameters()
        self.scale_direction.reset_parameters()
        self.memory_projection.reset_parameters()
        self.layer_embeddings.reset_parameters()
        self.module_embeddings.reset_parameters()
        self.age_embeddings.reset_parameters()
        nn.init.zeros_(self.output_projection.weight)

    def restore_fp32_output_path(self) -> None:
        self.output_projection.float()

    def _projection(self, module_index: int, *, axis: str, reference: Tensor) -> Tensor:
        value = getattr(self, f"projection_{axis}_{module_index}")
        return value.to(device=reference.device, dtype=torch.float32)

    def _indices(self, module_index: int, *, kind: str, device: torch.device) -> Tensor:
        return getattr(self, f"{kind}_{module_index}").to(device=device)

    @staticmethod
    def _batch_size_from_state(state: Mapping[str, Tensor]) -> int:
        for key, value in state.items():
            if key != "scale" and value.dim() >= 1:
                return int(value.shape[0])
        raise ValueError("Generated LoRA state contains no batched factor")

    @torch.no_grad()
    def project_effective_features(
        self,
        state: Mapping[str, Tensor],
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> Tensor:
        """Detach and compress the current Generated LoRA into all-target sketches."""

        batch_size = self._batch_size_from_state(state)
        reference = next(value for key, value in state.items() if key != "scale")
        if self.content == "static_history":
            return self.base_features.to(
                device=reference.device, dtype=output_dtype
            ).unsqueeze(0).expand(batch_size, -1, -1).clone()
        scale = state.get("scale")
        if scale is None or scale.numel() != 1:
            raise ValueError("Generated LoRA state must contain one scalar scale")
        # The generated state inherits this immutable scale from the same target
        # layout.  Do not read the CUDA scalar back here: this method runs after
        # every visible event and a host read would serialize the training loop.

        actual = self.base_sketches.to(
            device=reference.device, dtype=torch.float32
        ).unsqueeze(0).expand(batch_size, -1, -1).clone()
        for module_index, matrix_indices in enumerate(self._matrix_indices_by_module):
            module_name = self.module_names[module_index]
            projection_in = self._projection(
                module_index, axis="in", reference=reference
            )
            projection_out = self._projection(
                module_index, axis="out", reference=reference
            )
            if module_name == "lm_head":
                target_keys = ("lm_head",)
            else:
                target_keys = tuple(
                    self.targets[index].target_key for index in matrix_indices
                )
            lora_a = torch.stack(
                [state[f"{target_key}.lora_A"].detach().float() for target_key in target_keys],
                dim=1,
            )
            lora_b = torch.stack(
                [state[f"{target_key}.lora_B"].detach().float() for target_key in target_keys],
                dim=1,
            )
            a_projected = torch.matmul(lora_a, projection_in)
            b_projected = torch.einsum(
                "ok,bsor->bskr", projection_out, lora_b
            )
            delta = torch.matmul(b_projected, a_projected)
            flat_delta = delta.reshape(batch_size, len(matrix_indices), 1024)
            indices = self._indices(
                module_index,
                kind="matrix_indices",
                device=actual.device,
            )
            actual[:, indices] = actual[:, indices] + self.scale * flat_delta

        rms = actual.square().mean(dim=-1).add(RMS_EPSILON).sqrt()
        standardized_log_rms = (
            rms.log() - self.base_log_sketch_rms_mean.to(rms.device)
        ) / self.base_log_sketch_rms_std.to(rms.device)
        features = torch.cat(
            (actual / rms[..., None], standardized_log_rms[..., None]), dim=-1
        )
        return features.detach().to(dtype=output_dtype)

    def pack_histories(
        self,
        histories: Sequence[Sequence[Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        return pack_history_feature_batch(
            histories,
            matrix_count=self.matrix_count,
            max_history=self.max_history,
            history_mode=self.history_mode,
            device=device,
            dtype=dtype,
        )

    def _coordinate(self, length: int, *, reference: Tensor) -> Tensor:
        return getattr(self, self._coordinate_names[length]).to(
            device=reference.device, dtype=reference.dtype
        )

    def _compress_axis(
        self,
        values: Tensor,
        *,
        axis_index: int,
        log_rms: Tensor,
        projection: nn.Linear,
        compressor: AxisQueryCompressor,
    ) -> Tensor:
        batch, history, matrices, positions, _ = values.shape
        projected = projection(values)
        coordinate = self._coordinate(positions, reference=projected)
        scale = self.scale_direction(log_rms[..., None]).unsqueeze(-2)
        type_embedding = self.axis_type_embeddings[axis_index].to(
            device=projected.device, dtype=projected.dtype
        )
        projected = self.input_norm(
            projected
            + coordinate[None, None, None]
            + scale
            + type_embedding[None, None, None, None]
        )
        compressed = compressor(
            projected.reshape(batch * history * matrices, positions, self.width)
        )
        return compressed.reshape(
            batch, history, matrices, self.query_count, self.width
        )

    def _module_memory(
        self,
        features: Tensor,
        history_mask: Tensor,
        *,
        module_index: int,
    ) -> Tensor:
        index = self._indices(
            module_index,
            kind="matrix_indices",
            device=features.device,
        )
        selected = features.index_select(2, index)
        sketch = selected[..., :1024]
        log_rms = selected[..., 1024]
        module_name = self.module_names[module_index]
        if module_name == "lm_head":
            matrix = sketch.reshape(*sketch.shape[:-1], 16, 64)
            row_projection = self.lm_row_projection
            column_projection = self.lm_column_projection
        else:
            matrix = sketch.reshape(*sketch.shape[:-1], 32, 32)
            row_projection = self.square_value_projection
            column_projection = self.square_value_projection
        row_latents = self._compress_axis(
            matrix,
            axis_index=0,
            log_rms=log_rms,
            projection=row_projection,
            compressor=self.row_compressor,
        )
        column_latents = self._compress_axis(
            matrix.transpose(-1, -2),
            axis_index=1,
            log_rms=log_rms,
            projection=column_projection,
            compressor=self.column_compressor,
        )
        latents = torch.cat((row_latents, column_latents), dim=-2)
        memory = self.memory_projection(latents)
        slots = self.target_slots.index_select(0, index)
        module_ids = self.target_modules.index_select(0, index)
        identity = self.layer_embeddings(slots) + self.module_embeddings(module_ids)
        counts = history_mask.long().sum(dim=1)
        positions = torch.arange(
            history_mask.shape[1], device=history_mask.device
        ).unsqueeze(0)
        ages = (counts[:, None] - positions).clamp(min=0, max=self.max_history)
        age = self.age_embeddings(ages)
        memory = memory + identity[None, None, :, None] + age[:, :, None, None]
        return self.memory_norm(memory)

    def _zero_dependency(self, parameter_tokens: Tensor) -> Tensor:
        dependency = torch.zeros((), device=parameter_tokens.device, dtype=torch.float32)
        for parameter in self.parameters():
            if parameter.requires_grad:
                dependency = dependency + parameter.reshape(-1)[0].float() * 0.0
        return parameter_tokens + dependency.to(dtype=parameter_tokens.dtype)

    def diagnostic_metrics(self) -> dict[str, float]:
        weight = self.output_projection.weight.detach().float()
        metrics = {
            "raw_weight_residual_cap": self.residual_cap,
            "raw_weight_output_weight_rms": float(weight.square().mean().sqrt().cpu()),
            "raw_weight_output_weight_max_abs": float(weight.abs().max().cpu()),
        }
        if self.residual_rms_ceiling is not None:
            metrics["raw_weight_rms_ceiling"] = self.residual_rms_ceiling
        if self._last_pre_ceiling_residual_rms is not None:
            metrics["raw_weight_pre_ceiling_residual_rms"] = float(
                self._last_pre_ceiling_residual_rms.cpu()
            )
            metrics["raw_weight_rms_ceiling_scale"] = float(
                self._last_rms_ceiling_scale.cpu()
            )
        if self._last_residual_rms is not None:
            metrics["raw_weight_last_residual_rms"] = float(
                self._last_residual_rms.cpu()
            )
            metrics["raw_weight_last_residual_max_abs"] = float(
                self._last_residual_max_abs.cpu()
            )
        return metrics

    def forward(
        self,
        parameter_tokens: Tensor,
        *,
        history_features: Tensor | None,
        history_mask: Tensor | None,
    ) -> Tensor:
        if parameter_tokens.dim() != 4:
            raise ValueError("Parameter Tokens must be [batch, slot, token, channel]")
        if parameter_tokens.shape[1] != 37:
            raise ValueError("History-P2E requires the 37-slot Qwen layout")
        if parameter_tokens.shape[2] != self.token_to_module.numel():
            raise ValueError("Parameter Token route width changed")
        if parameter_tokens.shape[-1] != self.d_model:
            raise ValueError("Parameter Token channel width changed")
        if history_features is None or history_mask is None:
            return self._zero_dependency(parameter_tokens)
        if history_features.shape[:2] != history_mask.shape:
            raise ValueError("history_features and history_mask do not align")
        if history_features.shape[0] != parameter_tokens.shape[0]:
            raise ValueError("History batch does not match Parameter Tokens")
        if history_features.shape[2:] != (
            self.matrix_count,
            HISTORY_FEATURE_WIDTH,
        ):
            raise ValueError("History feature matrix shape changed")
        history_mask = history_mask.to(device=parameter_tokens.device, dtype=torch.bool)
        if history_features.shape[1] == 0:
            return self._zero_dependency(parameter_tokens)

        if self.history_mode == "last1" and history_features.shape[1] > 1:
            counts = history_mask.long().sum(dim=1)
            source = (counts - 1).clamp_min(0)
            gather = source[:, None, None, None].expand(
                -1, 1, self.matrix_count, HISTORY_FEATURE_WIDTH
            )
            history_features = history_features.gather(1, gather)
            history_mask = counts.gt(0)[:, None]
        history_features = history_features.to(
            device=parameter_tokens.device, dtype=parameter_tokens.dtype
        )
        if self.content == "static_history":
            history_features = self.base_features.to(
                device=parameter_tokens.device, dtype=parameter_tokens.dtype
            )[None, None].expand(
                parameter_tokens.shape[0],
                history_features.shape[1],
                -1,
                -1,
            )

        active_indices = torch.nonzero(history_mask.any(dim=1), as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return self._zero_dependency(parameter_tokens)
        active_tokens = parameter_tokens.index_select(0, active_indices)
        active_features = history_features.index_select(0, active_indices)
        active_mask = history_mask.index_select(0, active_indices)
        routed: list[tuple[Tensor, Tensor, Tensor, Tensor]] = []
        residual_square_sum = torch.zeros(
            (), device=parameter_tokens.device, dtype=torch.float32
        )
        residual_count = 0

        for module_index, matrix_indices in enumerate(self._matrix_indices_by_module):
            matrix_index = self._indices(
                module_index,
                kind="matrix_indices",
                device=parameter_tokens.device,
            )
            slot_indices = self.target_slots.index_select(0, matrix_index)
            token_indices = self._indices(
                module_index,
                kind="parameter_token_indices",
                device=parameter_tokens.device,
            )
            query = active_tokens.index_select(1, slot_indices).index_select(
                2, token_indices
            )
            memory = self._module_memory(
                active_features,
                active_mask,
                module_index=module_index,
            )
            batch, history, matrices, queries, channels = memory.shape
            memory_flat = memory.permute(0, 2, 1, 3, 4).reshape(
                batch * matrices, history * queries, channels
            )
            query_flat = query.reshape(
                batch * matrices, token_indices.numel(), self.d_model
            )
            key_padding_mask = ~active_mask[:, None, :, None].expand(
                batch, matrices, history, queries
            ).reshape(batch * matrices, history * queries)
            attended, _ = self.weight_cross_attention(
                query_flat,
                memory_flat,
                memory_flat,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            projected = self.output_projection(attended.float())
            bounded_delta = self.residual_cap * torch.tanh(projected)
            bounded_delta = bounded_delta.reshape_as(query)
            routed.append((slot_indices, token_indices, query, bounded_delta))
            residual_square_sum = (
                residual_square_sum + bounded_delta.float().square().sum()
            )
            residual_count += bounded_delta.numel()

        pre_ceiling_rms = (residual_square_sum / residual_count).sqrt()
        if self.residual_rms_ceiling is None:
            ceiling_scale = torch.ones_like(pre_ceiling_rms)
        else:
            denominator = (residual_square_sum / residual_count).clamp_min(
                self.residual_rms_ceiling**2
            ).sqrt()
            ceiling_scale = self.residual_rms_ceiling / denominator

        updated_active = active_tokens.clone()
        post_square_sum = torch.zeros_like(residual_square_sum)
        residual_max_abs = torch.zeros_like(pre_ceiling_rms)
        for slot_indices, token_indices, query, bounded_delta in routed:
            delta = bounded_delta * ceiling_scale
            detached = delta.detach().float()
            post_square_sum = post_square_sum + detached.square().sum()
            residual_max_abs = torch.maximum(residual_max_abs, detached.abs().max())
            updated = query.float() + delta
            updated_active[
                :,
                slot_indices[:, None],
                token_indices[None, :],
                :,
            ] = updated.to(dtype=updated_active.dtype)

        result = parameter_tokens.clone()
        result.index_copy_(0, active_indices, updated_active)
        self._last_pre_ceiling_residual_rms = pre_ceiling_rms.detach().float()
        self._last_rms_ceiling_scale = ceiling_scale.detach().float()
        self._last_residual_rms = (post_square_sum / residual_count).sqrt()
        self._last_residual_max_abs = residual_max_abs
        return result

from __future__ import annotations

import torch
from torch import Tensor, nn

from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.cache import (
    FilmFeatureCache,
    P2EFeatureCache,
)
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.conditioning import (
    FilmRawWeightConditioner,
    P2ERawWeightConditioner,
)
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.routing import (
    ParameterTokenRouting,
)


class SafeFilmRawWeightConditioner(nn.Module):
    """Scale-only bounded FiLM with an exact no-op initialization."""

    method = "safe_film"
    output_parameter_names = ("scale_head.weight",)

    def __init__(
        self,
        cache: FilmFeatureCache,
        *,
        routing: ParameterTokenRouting,
        d_model: int = 1280,
        encoder_hidden_dim: int = 512,
        residual_cap: float = 0.025,
        content: str = "real",
        control_seed: int = 24,
    ) -> None:
        super().__init__()
        if content not in {"real", "zero", "random", "shuffled_layer"}:
            raise ValueError(f"Unsupported Safe-FiLM content mode: {content!r}")
        if residual_cap not in {0.005, 0.01, 0.025, 0.05}:
            raise ValueError(
                "Safe-FiLM residual_cap must be one of 0.005, 0.01, 0.025, or 0.05"
            )
        if routing.valid_slot_module_mask.shape != cache.valid_mask.shape:
            raise ValueError("Safe-FiLM cache mask does not match Parameter Token routing")
        if not torch.equal(routing.valid_slot_module_mask, cache.valid_mask):
            raise ValueError("Safe-FiLM cache targets differ from the Qwen route mask")

        prepared = FilmRawWeightConditioner._prepare_features(
            cache.features,
            cache.valid_mask,
            content=content,
            seed=control_seed,
        )
        self.content = content
        self.d_model = int(d_model)
        self.residual_cap = float(residual_cap)
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
        self.scale_head = nn.Linear(d_model, d_model, bias=False)
        self._last_residual_rms: Tensor | None = None
        self._last_residual_max_abs: Tensor | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        nn.init.zeros_(self.scale_head.weight)

    def restore_fp32_output_path(self) -> None:
        self.scale_head.float()

    def diagnostic_metrics(self) -> dict[str, float]:
        weight = self.scale_head.weight.detach().float()
        metrics = {
            "raw_weight_residual_cap": self.residual_cap,
            "raw_weight_output_weight_rms": float(weight.square().mean().sqrt().cpu()),
            "raw_weight_output_weight_max_abs": float(weight.abs().max().cpu()),
        }
        if self._last_residual_rms is not None:
            metrics["raw_weight_last_residual_rms"] = float(
                self._last_residual_rms.cpu()
            )
            metrics["raw_weight_last_residual_max_abs"] = float(
                self._last_residual_max_abs.cpu()
            )
        return metrics

    def forward(self, parameter_tokens: Tensor) -> Tensor:
        if parameter_tokens.dim() != 4:
            raise ValueError("Parameter Tokens must have shape [batch, slot, token, channel]")
        if parameter_tokens.shape[1:3] != (
            self.valid_mask.shape[0],
            self.token_to_module.numel(),
        ):
            raise ValueError("Parameter Token grid does not match the Safe-FiLM route")
        if parameter_tokens.shape[-1] != self.d_model:
            raise ValueError("Parameter Token width does not match Safe-FiLM")
        if self.content == "zero":
            return parameter_tokens

        encoder_dtype = next(self.encoder.parameters()).dtype
        descriptors = self.encoder(
            self.features.to(device=parameter_tokens.device, dtype=encoder_dtype)
        )
        scale = self.scale_head(descriptors.float())
        mask = self.valid_mask.to(device=parameter_tokens.device)[..., None]
        scale = scale.masked_fill(~mask, 0.0)
        token_route = self.token_to_module.to(device=parameter_tokens.device)
        scale = scale[:, token_route]
        bounded_scale = self.residual_cap * torch.tanh(scale)
        detached_scale = bounded_scale.detach().float()
        self._last_residual_rms = detached_scale.square().mean().sqrt()
        self._last_residual_max_abs = detached_scale.abs().max()
        output = parameter_tokens.float() * (1.0 + bounded_scale[None])
        return output.to(dtype=parameter_tokens.dtype)


class SafeP2ERawWeightConditioner(P2ERawWeightConditioner):
    """Prompt-aware module-masked weight attention with a bounded zero-init residual."""

    method = "safe_p2e_qv"
    output_parameter_names = ("output_projection.weight",)

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
        residual_cap: float = 0.025,
        residual_rms_ceiling: float | None = None,
        content: str = "real",
        control_seed: int = 24,
    ) -> None:
        if residual_cap not in {0.005, 0.01, 0.025, 0.05}:
            raise ValueError(
                "Safe-P2E residual_cap must be one of 0.005, 0.01, 0.025, or 0.05"
            )
        super().__init__(
            cache,
            routing=routing,
            d_model=d_model,
            width=width,
            query_count=query_count,
            compressor_heads=compressor_heads,
            compressor_ffn_dim=compressor_ffn_dim,
            weight_attention_heads=weight_attention_heads,
            gate_init=0.01,
            content=content,
            control_seed=control_seed,
        )
        self.register_parameter("gate", None)
        self.residual_cap = float(residual_cap)
        if residual_rms_ceiling is not None:
            residual_rms_ceiling = float(residual_rms_ceiling)
            if not 0.0 < residual_rms_ceiling <= self.residual_cap:
                raise ValueError(
                    "Safe-P2E residual_rms_ceiling must be positive and must not "
                    "exceed residual_cap"
                )
        self.residual_rms_ceiling = residual_rms_ceiling
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self._last_pre_ceiling_residual_rms: Tensor | None = None
        self._last_rms_ceiling_scale: Tensor | None = None
        self._last_residual_rms: Tensor | None = None
        self._last_residual_max_abs: Tensor | None = None
        nn.init.zeros_(self.output_projection.weight)

    def restore_fp32_output_path(self) -> None:
        self.output_projection.float()

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
            metrics["raw_weight_rms_ceiling_triggered"] = float(
                self._last_rms_ceiling_scale.cpu() < 1.0
            )
        if self._last_residual_rms is not None:
            metrics["raw_weight_last_residual_rms"] = float(
                self._last_residual_rms.cpu()
            )
            metrics["raw_weight_last_residual_max_abs"] = float(
                self._last_residual_max_abs.cpu()
            )
        return metrics

    def forward(self, parameter_tokens: Tensor) -> Tensor:
        if parameter_tokens.dim() != 4:
            raise ValueError("Parameter Tokens must have shape [batch, slot, token, channel]")
        if parameter_tokens.shape[1] != 37 or parameter_tokens.shape[2] != self.token_to_module.numel():
            raise ValueError("Parameter Token grid does not match the Safe-P2E route")
        if parameter_tokens.shape[-1] != self.d_model:
            raise ValueError("Parameter Token width does not match Safe-P2E")
        if self.content == "zero":
            return parameter_tokens

        memories = self.weight_memory_for_forward().to(
            device=parameter_tokens.device,
            dtype=parameter_tokens.dtype,
        )
        result = parameter_tokens.clone()
        batch_size = parameter_tokens.shape[0]
        routed_deltas: list[tuple[Tensor, Tensor, Tensor]] = []
        residual_square_sum = torch.zeros(
            (), device=parameter_tokens.device, dtype=torch.float32
        )
        residual_count = 0
        for token_indices, matrix_indices in self._weight_routes():
            query = parameter_tokens[:, :36, token_indices, :]
            query_flat = query.reshape(
                batch_size * 36,
                token_indices.numel(),
                self.d_model,
            )
            memory = memories[matrix_indices]
            memory_flat = memory[None].expand(batch_size, -1, -1, -1).reshape(
                batch_size * 36,
                memory.shape[1],
                self.d_model,
            )
            attended, _ = self.weight_cross_attention(
                query_flat,
                memory_flat,
                memory_flat,
                need_weights=False,
            )
            projected = self.output_projection(attended.float())
            bounded_delta = self.residual_cap * torch.tanh(projected)
            routed_deltas.append((query, token_indices, bounded_delta))
            residual_square_sum = residual_square_sum + bounded_delta.float().square().sum()
            residual_count += bounded_delta.numel()

        residual_mean_square = residual_square_sum / residual_count
        pre_ceiling_rms = residual_mean_square.sqrt()
        if self.residual_rms_ceiling is None:
            ceiling_scale = torch.ones_like(pre_ceiling_rms)
        else:
            # Clamp before sqrt so the exact-zero output initialization has a
            # finite backward path.  Below the ceiling this is exactly one;
            # above it the residual is scaled to the configured global RMS.
            denominator = residual_mean_square.clamp_min(
                self.residual_rms_ceiling**2
            ).sqrt()
            ceiling_scale = self.residual_rms_ceiling / denominator

        residual_max_abs = torch.zeros((), device=parameter_tokens.device)
        post_ceiling_square_sum = torch.zeros(
            (), device=parameter_tokens.device, dtype=torch.float32
        )
        for query, token_indices, bounded_delta in routed_deltas:
            scaled_delta = bounded_delta * ceiling_scale
            detached_delta = scaled_delta.detach().float()
            post_ceiling_square_sum = (
                post_ceiling_square_sum + detached_delta.square().sum()
            )
            residual_max_abs = torch.maximum(residual_max_abs, detached_delta.abs().max())
            updated = query.reshape(
                batch_size * 36,
                token_indices.numel(),
                self.d_model,
            ).float() + scaled_delta
            result[:, :36, token_indices, :] = updated.reshape_as(query).to(
                dtype=parameter_tokens.dtype
            )
        self._last_pre_ceiling_residual_rms = pre_ceiling_rms.detach().float()
        self._last_rms_ceiling_scale = ceiling_scale.detach().float()
        self._last_residual_rms = (post_ceiling_square_sum / residual_count).sqrt()
        self._last_residual_max_abs = residual_max_abs
        return result

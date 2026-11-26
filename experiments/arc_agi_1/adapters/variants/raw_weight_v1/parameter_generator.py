from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from s2_hypernet.config import ParameterGeneratorConfig
from s2_hypernet.nvarc_pg.parameter_generator import NvarcParameterGenerator
from s2_hypernet.nvarc_pg.target_layout import QwenTargetLayout
from s2_hypernet.parameter_generator import ParameterGenerator, TransformerModel

from .cache import FilmFeatureCache, P2EFeatureCache, validate_cache_source_manifest
from .conditioning import (
    FilmRawWeightConditioner,
    IdentityRawWeightConditioner,
    P2ERawWeightConditioner,
)
from .config import RawWeightConditioningConfig
from .routing import build_parameter_token_routing


class RawWeightTransformerModel(TransformerModel):
    """TransformerModel with one explicit pre-block-1 raw-weight injection."""

    def __init__(self, *args, raw_weight_conditioner: nn.Module, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.raw_weight_conditioner = raw_weight_conditioner

    def forward(
        self,
        encoder_hidden_states: Tensor,
        condition_attention_mask: Tensor | None = None,
    ) -> Tensor:
        if encoder_hidden_states.dim() == 4 and encoder_hidden_states.shape[1] == 1:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        encoder_hidden_states = self.norm_input(encoder_hidden_states)
        hw_freqs, n_freqs, l_freqs, all_freqs = self._get_pos_embed(
            device=encoder_hidden_states.device,
            cond_seq_length=encoder_hidden_states.shape[1],
        )
        x = self.get_tokens(
            batch_size=encoder_hidden_states.shape[0],
            device=encoder_hidden_states.device,
        )
        x = self.raw_weight_conditioner(x)
        x = self.apply_condition_summary(
            x=x,
            encoder_hidden_states=encoder_hidden_states,
            condition_attention_mask=condition_attention_mask,
        )
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                if condition_attention_mask is None:

                    def block_forward(
                        x_arg: Tensor,
                        encoder_arg: Tensor,
                        hw_arg: Tensor,
                        n_arg: Tensor,
                        l_arg: Tensor,
                        all_arg: Tensor,
                        block_module: nn.Module = block,
                    ) -> Tensor:
                        return block_module(
                            x=x_arg,
                            encoder_hidden_states=encoder_arg,
                            hw_freqs=hw_arg,
                            n_freqs=n_arg,
                            l_freqs=l_arg,
                            all_freqs=all_arg,
                            condition_attention_mask=None,
                        )

                    x = checkpoint(
                        block_forward,
                        x,
                        encoder_hidden_states,
                        hw_freqs,
                        n_freqs,
                        l_freqs,
                        all_freqs,
                        use_reentrant=False,
                    )
                else:

                    def block_forward_with_mask(
                        x_arg: Tensor,
                        encoder_arg: Tensor,
                        hw_arg: Tensor,
                        n_arg: Tensor,
                        l_arg: Tensor,
                        all_arg: Tensor,
                        mask_arg: Tensor,
                        block_module: nn.Module = block,
                    ) -> Tensor:
                        return block_module(
                            x=x_arg,
                            encoder_hidden_states=encoder_arg,
                            hw_freqs=hw_arg,
                            n_freqs=n_arg,
                            l_freqs=l_arg,
                            all_freqs=all_arg,
                            condition_attention_mask=mask_arg,
                        )

                    x = checkpoint(
                        block_forward_with_mask,
                        x,
                        encoder_hidden_states,
                        hw_freqs,
                        n_freqs,
                        l_freqs,
                        all_freqs,
                        condition_attention_mask,
                        use_reentrant=False,
                    )
            else:
                x = block(
                    x=x,
                    encoder_hidden_states=encoder_hidden_states,
                    hw_freqs=hw_freqs,
                    n_freqs=n_freqs,
                    l_freqs=l_freqs,
                    all_freqs=all_freqs,
                    condition_attention_mask=condition_attention_mask,
                )
        return self.post_process(x)


def _replace_transformer_model(
    generator: ParameterGenerator,
    *,
    conditioner: nn.Module,
    injection_after_block: int = 0,
) -> None:
    config = generator.config
    old_model = generator.model
    old_state = old_model.state_dict()
    model_class: type[nn.Module]
    extra_kwargs: dict[str, int] = {}
    if injection_after_block == 0:
        model_class = RawWeightTransformerModel
    else:
        from s2_hypernet.nvarc_pg_experiments.raw_weight_safe_v2.parameter_generator import (
            SafeRawWeightTransformerModel,
        )

        model_class = SafeRawWeightTransformerModel
        extra_kwargs["injection_after_block"] = injection_after_block
    new_model = model_class(
        d_model=config.d_model,
        num_base_model_layers=config.num_base_model_layers,
        num_token_per_layer=generator.lora_A_token_count + generator.lora_B_token_count,
        lora_rank=config.rank,
        output_dim=config.output_dim,
        head_dim=config.head_dim,
        num_blocks=config.num_pg_layers,
        dim_accumulation=config.dim_accumulation,
        lora_A_token_count=generator.lora_A_token_count,
        lora_B_token_count=generator.lora_B_token_count,
        condition_injection_mode=config.condition_injection_mode,
        condition_summary_scale_init=config.condition_summary_scale_init,
        raw_weight_conditioner=conditioner,
        **extra_kwargs,
    )
    incompatible = new_model.load_state_dict(old_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected base PG keys while adding raw conditioner: {incompatible.unexpected_keys}"
        )
    expected_missing_prefix = "raw_weight_conditioner."
    if any(not key.startswith(expected_missing_prefix) for key in incompatible.missing_keys):
        raise RuntimeError(
            f"Base PG keys were lost while adding raw conditioner: {incompatible.missing_keys}"
        )
    new_model.gradient_checkpointing = old_model.gradient_checkpointing
    generator.model = new_model


def build_raw_weight_parameter_generator(
    pg_config: ParameterGeneratorConfig,
    *,
    layout: QwenTargetLayout,
    raw_config: RawWeightConditioningConfig,
    expected_base_aggregate_sha256: str | None = None,
) -> nn.Module:
    if pg_config.condition_injection_mode != "none":
        raise ValueError("Raw-weight v1 requires the base prompt-summary injection mode to remain none")
    if raw_config.method == "safe_history_p2e":
        from s2_hypernet.nvarc_pg_experiments.history_effweight_v1.parameter_generator import (
            build_history_parameter_generator,
        )

        return build_history_parameter_generator(
            pg_config,
            layout=layout,
            raw_config=raw_config,
            expected_base_aggregate_sha256=expected_base_aggregate_sha256,
        )
    generator = ParameterGenerator(pg_config)
    if raw_config.method == "none":
        return NvarcParameterGenerator(config=pg_config, generator=generator)

    routing = build_parameter_token_routing(generator.tokenizer, layout=layout)
    cache_path = raw_config.validate_cache_file()
    assert cache_path is not None
    if raw_config.method in {"film", "safe_film"}:
        validate_cache_source_manifest(
            cache_path,
            kind="film_sketch32_v1",
            layout=layout,
            projection_seed=raw_config.projection_seed,
            expected_base_aggregate_sha256=expected_base_aggregate_sha256,
        )
        cache = FilmFeatureCache.load(
            cache_path,
            expected_sha256=raw_config.cache_sha256,
        )
        if raw_config.method == "film":
            conditioner: nn.Module = FilmRawWeightConditioner(
                cache,
                routing=routing,
                d_model=pg_config.d_model,
                encoder_hidden_dim=raw_config.film_encoder_hidden_dim,
                gate_init=raw_config.resolved_gate_init,
                content=raw_config.content,
                control_seed=raw_config.projection_seed,
            )
        else:
            from s2_hypernet.nvarc_pg_experiments.raw_weight_safe_v2.conditioning import (
                SafeFilmRawWeightConditioner,
            )

            conditioner = SafeFilmRawWeightConditioner(
                cache,
                routing=routing,
                d_model=pg_config.d_model,
                encoder_hidden_dim=raw_config.film_encoder_hidden_dim,
                residual_cap=raw_config.resolved_residual_cap,
                content=raw_config.content,
                control_seed=raw_config.projection_seed,
            )
    elif raw_config.method in {"p2e_qv", "safe_p2e_qv"}:
        validate_cache_source_manifest(
            cache_path,
            kind="p2e_qv_axis128_v1",
            layout=layout,
            projection_seed=raw_config.projection_seed,
            expected_base_aggregate_sha256=expected_base_aggregate_sha256,
        )
        cache = P2EFeatureCache.load(
            cache_path,
            expected_sha256=raw_config.cache_sha256,
        )
        if raw_config.method == "p2e_qv":
            conditioner = P2ERawWeightConditioner(
                cache,
                routing=routing,
                d_model=pg_config.d_model,
                width=raw_config.p2e_width,
                query_count=raw_config.p2e_queries_per_axis,
                compressor_heads=raw_config.p2e_heads,
                compressor_ffn_dim=raw_config.p2e_ffn_dim,
                gate_init=raw_config.resolved_gate_init,
                content=raw_config.content,
                control_seed=raw_config.projection_seed,
            )
        else:
            from s2_hypernet.nvarc_pg_experiments.raw_weight_safe_v2.conditioning import (
                SafeP2ERawWeightConditioner,
            )

            conditioner = SafeP2ERawWeightConditioner(
                cache,
                routing=routing,
                d_model=pg_config.d_model,
                width=raw_config.p2e_width,
                query_count=raw_config.p2e_queries_per_axis,
                compressor_heads=raw_config.p2e_heads,
                compressor_ffn_dim=raw_config.p2e_ffn_dim,
                residual_cap=raw_config.resolved_residual_cap,
                residual_rms_ceiling=raw_config.residual_rms_ceiling,
                content=raw_config.content,
                control_seed=raw_config.projection_seed,
            )
    else:
        conditioner = IdentityRawWeightConditioner()

    _replace_transformer_model(
        generator,
        conditioner=conditioner,
        injection_after_block=(
            raw_config.resolved_injection_after_block
            if raw_config.method.startswith("safe_")
            else 0
        ),
    )
    return NvarcParameterGenerator(config=pg_config, generator=generator)


def load_canonical_pg_backbone(
    parameter_generator: NvarcParameterGenerator,
    canonical_state: Mapping[str, Tensor],
) -> tuple[str, ...]:
    """Load the common PG tensors and allow only raw-conditioner keys to be absent."""

    incompatible = parameter_generator.load_state_dict(dict(canonical_state), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Canonical PG state contains unexpected keys: {incompatible.unexpected_keys}"
        )
    allowed_prefix = "generator.model.raw_weight_conditioner."
    forbidden_missing = [
        key for key in incompatible.missing_keys if not key.startswith(allowed_prefix)
    ]
    if forbidden_missing:
        raise ValueError(
            f"Canonical PG state is missing base-backbone keys: {forbidden_missing}"
        )
    return tuple(incompatible.missing_keys)

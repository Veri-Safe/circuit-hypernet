from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

from torch import Tensor, nn

from s2_hypernet.config import ParameterGeneratorConfig
from s2_hypernet.nvarc_pg.target_layout import QwenTargetLayout
from s2_hypernet.nvarc_pg_experiments.raw_weight_safe_v2.parameter_generator import (
    SafeRawWeightTransformerModel,
)
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.cache import (
    FilmFeatureCache,
    validate_cache_source_manifest,
)
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.config import (
    RawWeightConditioningConfig,
)
from s2_hypernet.nvarc_pg_experiments.raw_weight_v1.routing import (
    build_parameter_token_routing,
)
from s2_hypernet.parameter_generator import ParameterGenerator

from .conditioning import HistoryEffectiveWeightConditioner


class HistoryEffectiveWeightTransformerModel(SafeRawWeightTransformerModel):
    """Safe PG model whose post-block-1 injection receives per-query history."""

    def forward(
        self,
        encoder_hidden_states: Tensor,
        condition_attention_mask: Tensor | None = None,
        *,
        history_features: Tensor | None = None,
        history_mask: Tensor | None = None,
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
        x = self.apply_condition_summary(
            x=x,
            encoder_hidden_states=encoder_hidden_states,
            condition_attention_mask=condition_attention_mask,
        )
        for block_index, block in enumerate(self.blocks, start=1):
            x = self._run_block(
                block,
                x=x,
                encoder_hidden_states=encoder_hidden_states,
                hw_freqs=hw_freqs,
                n_freqs=n_freqs,
                l_freqs=l_freqs,
                all_freqs=all_freqs,
                condition_attention_mask=condition_attention_mask,
            )
            if block_index == self.injection_after_block:
                x = self.raw_weight_conditioner(
                    x,
                    history_features=history_features,
                    history_mask=history_mask,
                )
        return self.post_process(x)


class HistoryParameterGenerator(ParameterGenerator):
    """HY-WU Parameter Generator with an explicit detached-history input."""

    def forward(
        self,
        condition: Tensor,
        condition_attention_mask: Tensor | None = None,
        *,
        history_features: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        embeddings = self.hidden_in(condition)
        output = self.model(
            encoder_hidden_states=embeddings,
            condition_attention_mask=condition_attention_mask,
            history_features=history_features,
            history_mask=history_mask,
        )

        all_layer_state_dict: dict[str, Tensor] = {}
        shape_state_dict = self.tokenizer.shape_state_dict
        for seq_idx, layer_idx in enumerate(self.target_layer_indices):
            layer_output = output[:, seq_idx]
            layer_state_dict: OrderedDict[str, Tensor] = self.tokenizer.detokenize(
                shape_state_dict,
                layer_output,
            )
            for key, value in layer_state_dict.items():
                all_layer_state_dict[f"{self.prefix}{layer_idx}.{key}"] = value
        all_layer_state_dict["scale"] = output.new_tensor(self.tokenizer.lora_scale)
        return all_layer_state_dict


class HistoryNvarcParameterGenerator(nn.Module):
    """Select the real 253 targets from the homogeneous 37-slot PG grid."""

    def __init__(
        self,
        config: ParameterGeneratorConfig,
        *,
        generator: HistoryParameterGenerator,
        lm_head_slot: int = 36,
    ) -> None:
        super().__init__()
        self.generator = generator
        self.config = config
        self.lm_head_slot = int(lm_head_slot)
        self.layer_target_keys = tuple(
            f"{config.prefix}{layer}.{module}"
            for layer in range(lm_head_slot)
            for module in config.pg_mapping
            if module != "lm_head"
        )

    def enable_gradient_checkpointing(self) -> None:
        self.generator.enable_gradient_checkpointing()

    def forward(
        self,
        condition: Tensor,
        condition_attention_mask: Tensor | None = None,
        *,
        history_features: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        raw: Mapping[str, Tensor] = self.generator(
            condition,
            condition_attention_mask=condition_attention_mask,
            history_features=history_features,
            history_mask=history_mask,
        )
        selected: dict[str, Tensor] = {}
        for target_key in self.layer_target_keys:
            for factor in ("lora_A", "lora_B"):
                selected[f"{target_key}.{factor}"] = raw[
                    f"{target_key}.{factor}"
                ]
        source_prefix = f"model.layers.{self.lm_head_slot}.lm_head"
        for factor in ("lora_A", "lora_B"):
            selected[f"lm_head.{factor}"] = raw[f"{source_prefix}.{factor}"]
        selected["scale"] = raw["scale"]
        return selected


def _replace_history_model(
    generator: HistoryParameterGenerator,
    conditioner: HistoryEffectiveWeightConditioner,
) -> None:
    config = generator.config
    old_model = generator.model
    old_state = old_model.state_dict()
    new_model = HistoryEffectiveWeightTransformerModel(
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
        injection_after_block=1,
    )
    incompatible = new_model.load_state_dict(old_state, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected base PG keys while adding history conditioner: "
            f"{incompatible.unexpected_keys}"
        )
    if any(
        not key.startswith("raw_weight_conditioner.")
        for key in incompatible.missing_keys
    ):
        raise RuntimeError(
            "Base PG keys were lost while adding history conditioner: "
            f"{incompatible.missing_keys}"
        )
    new_model.gradient_checkpointing = old_model.gradient_checkpointing
    generator.model = new_model


def build_history_parameter_generator(
    pg_config: ParameterGeneratorConfig,
    *,
    layout: QwenTargetLayout,
    raw_config: RawWeightConditioningConfig,
    expected_base_aggregate_sha256: str | None = None,
) -> HistoryNvarcParameterGenerator:
    if raw_config.method != "safe_history_p2e":
        raise ValueError("History builder requires method=safe_history_p2e")
    if pg_config.condition_injection_mode != "none":
        raise ValueError("History-P2E keeps the base prompt-summary mode disabled")
    cache_path = raw_config.validate_cache_file()
    assert cache_path is not None
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
    generator = HistoryParameterGenerator(pg_config)
    routing = build_parameter_token_routing(generator.tokenizer, layout=layout)
    conditioner = HistoryEffectiveWeightConditioner(
        cache,
        layout=layout,
        routing=routing,
        d_model=pg_config.d_model,
        width=raw_config.p2e_width,
        query_count=raw_config.p2e_queries_per_axis,
        compressor_heads=raw_config.p2e_heads,
        compressor_ffn_dim=raw_config.p2e_ffn_dim,
        residual_cap=raw_config.resolved_residual_cap,
        residual_rms_ceiling=raw_config.residual_rms_ceiling,
        content=raw_config.content,
        history_mode=str(raw_config.history_mode),
        max_history=int(raw_config.max_history),
        projection_seed=raw_config.projection_seed,
    )
    _replace_history_model(generator, conditioner)
    return HistoryNvarcParameterGenerator(pg_config, generator=generator)

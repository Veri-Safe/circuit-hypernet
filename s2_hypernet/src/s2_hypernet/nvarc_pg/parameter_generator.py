from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor, nn

from s2_hypernet.config import ParameterGeneratorConfig
from s2_hypernet.parameter_generator import ParameterGenerator


class NvarcParameterGenerator(nn.Module):
    """Expose 36 layer adapters plus one singleton LM-head adapter.

    The HY-WU generator uses a homogeneous output grid. Slot 36 is therefore a
    synthetic global slot: only its ``lm_head`` tensors are retained, while the
    first 36 slots retain only Transformer-layer targets.
    """

    def __init__(
        self,
        config: ParameterGeneratorConfig | None = None,
        *,
        generator: nn.Module | None = None,
        layer_target_keys: tuple[str, ...] | None = None,
        lm_head_slot: int = 36,
    ) -> None:
        super().__init__()
        if generator is None:
            if config is None:
                raise ValueError("config is required when generator is not supplied")
            generator = ParameterGenerator(config)
        if layer_target_keys is None:
            if config is None:
                raise ValueError("layer_target_keys is required without a config")
            layer_target_keys = tuple(
                f"{config.prefix}{layer}.{module}"
                for layer in range(lm_head_slot)
                for module in config.pg_mapping
                if module != "lm_head"
            )
        self.generator = generator
        self.config = config
        self.layer_target_keys = tuple(layer_target_keys)
        self.lm_head_slot = int(lm_head_slot)

    def enable_gradient_checkpointing(self) -> None:
        enable = getattr(self.generator, "enable_gradient_checkpointing", None)
        if enable is None:
            raise TypeError("Wrapped Parameter Generator has no checkpointing support")
        enable()

    def forward(
        self,
        condition: Tensor,
        condition_attention_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        raw: Mapping[str, Tensor] = self.generator(
            condition,
            condition_attention_mask=condition_attention_mask,
        )
        selected: dict[str, Tensor] = {}
        for target_key in self.layer_target_keys:
            for factor in ("lora_A", "lora_B"):
                key = f"{target_key}.{factor}"
                selected[key] = raw[key]
        source_prefix = f"model.layers.{self.lm_head_slot}.lm_head"
        for factor in ("lora_A", "lora_B"):
            selected[f"lm_head.{factor}"] = raw[f"{source_prefix}.{factor}"]
        selected["scale"] = raw["scale"]
        return selected

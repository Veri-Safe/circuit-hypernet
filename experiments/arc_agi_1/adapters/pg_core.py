from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from s2_hypernet.config import ParameterGeneratorConfig
from s2_hypernet.nvarc_pg.parameter_generator import NvarcParameterGenerator
from s2_hypernet.parameter_generator import MultiHeadCrossAttention, TransformerBlock

from .fusion import CircuitFusion, FusionInputs, append_masked_memory


class TrainableGlobalLoRAScale(nn.Module):
    """One checkpointed scalar shared by every generated LoRA."""

    def __init__(self, *, initial: float, maximum: float) -> None:
        super().__init__()
        if not 0.0 < initial < maximum:
            raise ValueError("Global LoRA scale initial value must be inside (0, max)")
        ratio = torch.tensor(initial / maximum, dtype=torch.float32)
        self.logit = nn.Parameter(torch.logit(ratio))
        self.maximum = float(maximum)

    def forward(self) -> Tensor:
        return torch.sigmoid(self.logit) * self.maximum


class S4CircuitParameterGenerator(nn.Module):
    """Fresh S2 PG trunk with an explicit, hook-free circuit fusion path."""

    def __init__(
        self,
        config: ParameterGeneratorConfig,
        *,
        variant_id: str,
        encoder_hidden_dim: int = 128,
        fusion_seed: int | None = None,
        trainable_global_lora_scale: bool = False,
        global_lora_scale_initial: float = 1.0,
        global_lora_scale_max: float = 1.5,
        circuit_input_mode: str = "raw_v1",
    ) -> None:
        super().__init__()
        self.variant_id = variant_id
        self.base = NvarcParameterGenerator(config)
        self.gradient_checkpointing = False
        self.global_lora_scale_control = (
            TrainableGlobalLoRAScale(
                initial=global_lora_scale_initial,
                maximum=global_lora_scale_max,
            )
            if trainable_global_lora_scale
            else None
        )
        with torch.random.fork_rng(devices=[]):
            if fusion_seed is not None:
                torch.manual_seed(fusion_seed)
            self.fusion = CircuitFusion(
                variant_id,
                encoder_hidden_dim=encoder_hidden_dim,
                pg_hidden_dim=config.d_model,
                generator_slots=config.num_base_model_layers,
                input_mode=circuit_input_mode,
            )
            num_blocks = config.num_pg_layers
            if variant_id == "f3_independent_attention":
                self.circuit_attentions = nn.ModuleList(
                    [
                        MultiHeadCrossAttention(
                            d_model=config.d_model,
                            num_heads=config.d_model // config.head_dim,
                        )
                        for _ in range(num_blocks)
                    ]
                )
            else:
                self.circuit_attentions = nn.ModuleList()
            if variant_id in {"f3_shared_attention", "f3_independent_attention"}:
                self.circuit_output_projections = nn.ModuleList(
                    [nn.Linear(config.d_model, config.d_model) for _ in range(num_blocks)]
                )
                for projection in self.circuit_output_projections:
                    nn.init.zeros_(projection.weight)
                    nn.init.zeros_(projection.bias)
            else:
                self.circuit_output_projections = nn.ModuleList()

    @property
    def generator(self) -> nn.Module:
        return self.base.generator

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False

    def prepare_runtime(
        self,
        *,
        device: torch.device,
        pg_dtype: torch.dtype = torch.bfloat16,
    ) -> "S4CircuitParameterGenerator":
        """Keep circuit MLP math FP32 while the PG/attention path runs in BF16."""

        self.base.to(device=device, dtype=pg_dtype)
        self.fusion.to(device=device, dtype=torch.float32)
        self.circuit_attentions.to(device=device, dtype=pg_dtype)
        self.circuit_output_projections.to(device=device, dtype=pg_dtype)
        if self.global_lora_scale_control is not None:
            self.global_lora_scale_control.to(device=device, dtype=torch.float32)
        return self

    def global_lora_scale_value(self) -> Tensor:
        if self.global_lora_scale_control is None:
            return torch.tensor(1.0)
        return self.global_lora_scale_control()

    def common_trunk_state_dict(self) -> OrderedDict[str, Tensor]:
        return OrderedDict(
            (key, value.detach().cpu().contiguous())
            for key, value in self.base.state_dict().items()
        )

    def load_common_trunk_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    @staticmethod
    def _cross_mask(mask: Tensor | None, *, device: torch.device) -> Tensor | None:
        if mask is None:
            return None
        return mask.to(device=device, dtype=torch.bool)[:, None, None, :]

    def _f3_block(
        self,
        *,
        block: TransformerBlock,
        block_index: int,
        x: Tensor,
        prompt_memory: Tensor,
        prompt_mask: Tensor,
        circuit_memory: Tensor,
        circuit_mask: Tensor,
        hw_freqs: Tensor,
        n_freqs: Tensor,
        prompt_freqs: Tensor,
        circuit_freqs: Tensor,
        all_freqs: Tensor,
    ) -> Tensor:
        batch_size, seq_n, seq_hw, channels = x.shape

        norm_x = block.norm1(x)
        norm_x = norm_x.permute(0, 2, 1, 3).reshape(
            batch_size * seq_hw, seq_n, channels
        )
        out = block.self_attn_n(norm_x, freqs=n_freqs)
        out = out.reshape(batch_size, seq_hw, seq_n, channels).permute(0, 2, 1, 3)
        x = x + out

        norm_x = block.norm2(x)
        norm_x = norm_x.reshape(batch_size * seq_n, seq_hw, channels)
        out = block.self_attn_hw(norm_x, freqs=hw_freqs)
        out = out.reshape(batch_size, seq_n, seq_hw, channels)
        x = x + out

        # Both branches read the same normalized state. Their softmaxes are separate.
        norm_x = block.norm3(x).reshape(batch_size, seq_n * seq_hw, channels)
        prompt_out = block.cross_attn(
            norm_x,
            prompt_memory,
            q_freqs=all_freqs,
            kv_freqs=prompt_freqs,
            attn_mask=self._cross_mask(prompt_mask, device=norm_x.device),
        )
        if self.variant_id == "f3_shared_attention":
            raw_circuit_out = block.cross_attn(
                norm_x,
                circuit_memory,
                q_freqs=all_freqs,
                kv_freqs=circuit_freqs,
                attn_mask=self._cross_mask(circuit_mask, device=norm_x.device),
            )
        else:
            raw_circuit_out = self.circuit_attentions[block_index](
                norm_x,
                circuit_memory,
                q_freqs=all_freqs,
                kv_freqs=circuit_freqs,
                attn_mask=self._cross_mask(circuit_mask, device=norm_x.device),
            )
        circuit_out = self.circuit_output_projections[block_index](raw_circuit_out)
        present = circuit_mask.any(dim=1).to(dtype=circuit_out.dtype)[:, None, None]
        circuit_out = circuit_out * present
        x = x + (prompt_out + circuit_out).reshape(
            batch_size, seq_n, seq_hw, channels
        )
        return x + block.feed_forward(block.norm4(x))

    def _run_f3_block_checkpointed(self, **kwargs: Any) -> Tensor:
        block = kwargs.pop("block")
        block_index = int(kwargs.pop("block_index"))
        keys = tuple(kwargs)
        values = tuple(kwargs[key] for key in keys)

        def block_forward(*args: Tensor) -> Tensor:
            return self._f3_block(
                block=block,
                block_index=block_index,
                **dict(zip(keys, args, strict=True)),
            )

        return checkpoint(block_forward, *values, use_reentrant=False)

    def _condition_memory(
        self, prompt: Tensor, prompt_mask: Tensor, fusion: FusionInputs,
    ) -> tuple[Tensor, Tensor]:
        if self.variant_id in {"f0_single_token", "f1_layer_shared_memory"}:
            assert fusion.memory_tokens is not None and fusion.memory_mask is not None
            return append_masked_memory(
                prompt, prompt_mask, fusion.memory_tokens, fusion.memory_mask,
            )
        return prompt, prompt_mask

    def _forward_tokens(
        self,
        condition: Tensor,
        *,
        condition_attention_mask: Tensor,
        circuit_vector: Tensor,
        circuit_present: Tensor,
    ) -> Tensor:
        generator = self.base.generator
        core = generator.model
        prompt_memory = core.norm_input(generator.hidden_in(condition))
        fusion: FusionInputs = self.fusion.encode(
            circuit_vector,
            circuit_present=circuit_present,
            output_dtype=prompt_memory.dtype,
        )
        x = core.get_tokens(
            batch_size=prompt_memory.shape[0],
            device=prompt_memory.device,
        )
        x = x + fusion.direct_residual[:, :, None, :]

        memory, memory_mask = self._condition_memory(
            prompt_memory, condition_attention_mask, fusion,
        )

        hw_freqs, n_freqs, memory_freqs, all_freqs = core._get_pos_embed(
            device=memory.device,
            cond_seq_length=memory.shape[1],
        )
        is_f3 = self.variant_id in {
            "f3_shared_attention",
            "f3_independent_attention",
        }
        circuit_freqs = None
        if is_f3:
            assert fusion.memory_tokens is not None and fusion.memory_mask is not None
            _, _, circuit_freqs, _ = core._get_pos_embed(
                device=memory.device,
                cond_seq_length=2,
            )

        for block_index, block in enumerate(core.blocks):
            if is_f3:
                kwargs = {
                    "block": block,
                    "block_index": block_index,
                    "x": x,
                    "prompt_memory": memory,
                    "prompt_mask": memory_mask,
                    "circuit_memory": fusion.memory_tokens,
                    "circuit_mask": fusion.memory_mask,
                    "hw_freqs": hw_freqs,
                    "n_freqs": n_freqs,
                    "prompt_freqs": memory_freqs,
                    "circuit_freqs": circuit_freqs,
                    "all_freqs": all_freqs,
                }
                if self.gradient_checkpointing and self.training:
                    x = self._run_f3_block_checkpointed(**kwargs)
                else:
                    x = self._f3_block(**kwargs)
            else:
                if self.gradient_checkpointing and self.training:
                    def block_forward(
                        x_arg: Tensor,
                        memory_arg: Tensor,
                        hw_arg: Tensor,
                        n_arg: Tensor,
                        memory_freq_arg: Tensor,
                        all_arg: Tensor,
                        mask_arg: Tensor,
                        block_module: TransformerBlock = block,
                    ) -> Tensor:
                        return block_module(
                            x=x_arg,
                            encoder_hidden_states=memory_arg,
                            hw_freqs=hw_arg,
                            n_freqs=n_arg,
                            l_freqs=memory_freq_arg,
                            all_freqs=all_arg,
                            condition_attention_mask=mask_arg,
                        )

                    x = checkpoint(
                        block_forward,
                        x,
                        memory,
                        hw_freqs,
                        n_freqs,
                        memory_freqs,
                        all_freqs,
                        memory_mask,
                        use_reentrant=False,
                    )
                else:
                    x = block(
                        x=x,
                        encoder_hidden_states=memory,
                        hw_freqs=hw_freqs,
                        n_freqs=n_freqs,
                        l_freqs=memory_freqs,
                        all_freqs=all_freqs,
                        condition_attention_mask=memory_mask,
                    )
        return core.post_process(x)

    def _detokenize(self, output: Tensor) -> dict[str, Tensor]:
        generator = self.base.generator
        raw: dict[str, Tensor] = {}
        shape_state_dict = generator.tokenizer.shape_state_dict
        for sequence_index, layer_index in enumerate(generator.target_layer_indices):
            layer_state = generator.tokenizer.detokenize(
                shape_state_dict,
                output[:, sequence_index],
            )
            for key, value in layer_state.items():
                raw[f"{generator.prefix}{layer_index}.{key}"] = value
        selected: dict[str, Tensor] = {}
        for target_key in self.base.layer_target_keys:
            for factor in ("lora_A", "lora_B"):
                key = f"{target_key}.{factor}"
                selected[key] = raw[key]
        source_prefix = f"model.layers.{self.base.lm_head_slot}.lm_head"
        for factor in ("lora_A", "lora_B"):
            selected[f"lm_head.{factor}"] = raw[f"{source_prefix}.{factor}"]
        base_scale = torch.tensor(
            generator.tokenizer.lora_scale,
            device=output.device,
            dtype=output.dtype,
        )
        global_scale = self.global_lora_scale_value().to(
            device=output.device,
            dtype=output.dtype,
        )
        selected["scale"] = base_scale * global_scale
        return selected

    def forward(
        self,
        condition: Tensor,
        *,
        condition_attention_mask: Tensor,
        circuit_vector: Tensor,
        circuit_present: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if condition.ndim != 3:
            raise ValueError("condition must have shape [B,T,input_dim]")
        if condition_attention_mask.shape != condition.shape[:2]:
            raise ValueError("condition mask does not align with condition")
        if circuit_vector.shape != (condition.shape[0], 59):
            raise ValueError("circuit_vector must have shape [B,59]")
        if circuit_present is None:
            circuit_present = torch.ones(
                condition.shape[0], device=condition.device, dtype=torch.bool
            )
        output = self._forward_tokens(
            condition,
            condition_attention_mask=condition_attention_mask,
            circuit_vector=circuit_vector,
            circuit_present=circuit_present,
        )
        return self._detokenize(output)

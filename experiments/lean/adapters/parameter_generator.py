"""S5 adapter around the shared S2 Generated-LoRA Parameter Generator.

The default S5 path remains the upstream append-memory generator.  The
Circuit-F2/F3i variants in this module add a separate Circuit-to-Parameter
Token path without changing the shared S2 generator implementation:

* F2 injects one zero-initialized residual after PG block 1.  Candidate A
  uses a masked Circuit summary; Candidate B uses one block-aware
  Circuit-only attention read.
* F3i uses one independent Circuit multi-head attention module after PG block
  1 and reuses it through the remaining PG blocks.

The zero-initialized output projections make the new routes exact no-ops at
step 0, which is required for the Query-only parity gate.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor, nn

from s2_hypernet.config import ParameterGeneratorConfig
from s2_hypernet.lora_tokenizer import Tokenizer2DBatchedLoRA
from s2_hypernet.parameter_generator import TransformerModel
from s2_hypernet.parameter_generator import ParameterGenerator as S2ParameterGenerator

from .target_layout import S5TargetLayout, build_s5_target_layout


S5_PG_D_MODEL = 1024
S5_PG_HEAD_DIM = 128
S5_PG_NUM_BLOCKS = 12
S5_PG_INPUT_DIM = 1024
S5_FUSION_MODES = (
    "append",
    "f2",
    "f3i",
    "dual_f3i",
    "circuit_p2e",
)
S5_MODULE_TOKEN_COUNTS = (9, 4, 11, 10)
S5_MODULE_TOKEN_INDICES = (
    (0, 1, 2, 16, 17, 18, 19, 20, 21),
    (3, 4, 22, 23),
    (5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31),
    (8, 9, 10, 11, 12, 13, 14, 15, 32, 33),
)
if (
    tuple(map(len, S5_MODULE_TOKEN_INDICES)) != S5_MODULE_TOKEN_COUNTS
    or sorted(index for route in S5_MODULE_TOKEN_INDICES for index in route)
    != list(range(34))
):
    raise RuntimeError("S5 module-token routes do not cover the 34-token layout")
S5_CIRCUIT_RESIDUAL_CAP = 0.025
S5_CIRCUIT_RESIDUAL_RMS_CEILING = 0.005


def _zero_attention_output(attention: nn.MultiheadAttention) -> None:
    """Zero only the residual output projection of a Circuit attention."""

    nn.init.zeros_(attention.out_proj.weight)
    if attention.out_proj.bias is not None:
        nn.init.zeros_(attention.out_proj.bias)


class CircuitP2EConditioner(nn.Module):
    """Route per-query Circuit tokens to matching layer/module LoRA slots.

    The current Vector397 encoders emit one global token plus five structural
    block tokens.  This conditioner adds explicit target-layer and target-
    module addresses before a shared cross-attention, then applies each result
    only to the Parameter-Token slice for that module.  It mirrors the useful
    routing bias of Raw Safe-P2E while retaining per-query Circuit content.
    """

    def __init__(
        self,
        *,
        d_model: int,
        circuit_token_count: int,
        residual_cap: float = S5_CIRCUIT_RESIDUAL_CAP,
        residual_rms_ceiling: float = S5_CIRCUIT_RESIDUAL_RMS_CEILING,
    ) -> None:
        super().__init__()
        if int(circuit_token_count) != 6:
            raise ValueError("Circuit-P2E requires one global plus five block tokens")
        if d_model <= 0 or d_model % 8:
            raise ValueError("Circuit-P2E width must be positive and divisible by eight")
        if not 0.0 < float(residual_rms_ceiling) <= float(residual_cap):
            raise ValueError("Circuit-P2E residual ceiling must be in (0, cap]")
        self.d_model = int(d_model)
        self.circuit_token_count = int(circuit_token_count)
        self.residual_cap = float(residual_cap)
        self.residual_rms_ceiling = float(residual_rms_ceiling)
        self.query_norm = nn.RMSNorm(d_model, eps=1.0e-6)
        self.memory_norm = nn.RMSNorm(d_model, eps=1.0e-6)
        self.layer_embeddings = nn.Parameter(torch.empty(8, d_model))
        self.module_embeddings = nn.Parameter(torch.empty(4, d_model))
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            num_heads=8,
            dropout=0.0,
            batch_first=True,
        )
        nn.init.normal_(self.layer_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.module_embeddings, mean=0.0, std=0.02)
        for module_index, route in enumerate(S5_MODULE_TOKEN_INDICES):
            self.register_buffer(
                f"module_token_indices_{module_index}",
                torch.tensor(route, dtype=torch.long),
                persistent=False,
            )
        _zero_attention_output(self.cross_attention)
        self._last_residual_rms: Tensor | None = None
        self._last_residual_max_abs: Tensor | None = None

    def validate_step0_initialization(self) -> None:
        output = self.cross_attention.out_proj
        if not bool(torch.equal(output.weight, torch.zeros_like(output.weight))):
            raise RuntimeError("Circuit-P2E output projection is not zero initialized")
        if output.bias is not None and not bool(
            torch.equal(output.bias, torch.zeros_like(output.bias))
        ):
            raise RuntimeError("Circuit-P2E output bias is not zero initialized")

    def forward(
        self,
        parameter_tokens: Tensor,
        circuit_tokens: Tensor,
        circuit_mask: Tensor,
    ) -> Tensor:
        if parameter_tokens.ndim != 4 or parameter_tokens.shape[1:3] != (8, 34):
            raise ValueError("Circuit-P2E expects Parameter Tokens [B,8,34,H]")
        if parameter_tokens.shape[-1] != self.d_model:
            raise ValueError("Circuit-P2E Parameter-Token width changed")
        if circuit_tokens.shape != (
            parameter_tokens.shape[0],
            self.circuit_token_count,
            self.d_model,
        ):
            raise ValueError("Circuit-P2E Circuit token shape changed")
        if circuit_mask.shape != circuit_tokens.shape[:2]:
            raise ValueError("Circuit-P2E mask must have shape [B,6]")
        mask = circuit_mask.to(device=circuit_tokens.device, dtype=torch.bool)
        present = mask.any(dim=1)
        safe_mask = mask.clone()
        safe_tokens = circuit_tokens
        empty = ~present
        if bool(empty.any()):
            safe_tokens = circuit_tokens.clone()
            safe_mask[empty, 0] = True
            safe_tokens[empty, 0] = 0
        batch_size = parameter_tokens.shape[0]
        memory_base = safe_tokens[:, None, :, :] + self.layer_embeddings.to(
            device=safe_tokens.device,
            dtype=safe_tokens.dtype,
        )[None, :, None, :]
        result = parameter_tokens.clone()
        deltas: list[tuple[Tensor, Tensor]] = []
        residual_square_sum = torch.zeros(
            (), device=parameter_tokens.device, dtype=torch.float32
        )
        residual_count = 0
        for module_index, route in enumerate(S5_MODULE_TOKEN_INDICES):
            token_indices = getattr(self, f"module_token_indices_{module_index}")
            count = token_indices.numel()
            query = self.query_norm(
                parameter_tokens.index_select(2, token_indices)
            )
            query_flat = query.reshape(batch_size * 8, count, self.d_model)
            memory = memory_base + self.module_embeddings[module_index].to(
                device=memory_base.device,
                dtype=memory_base.dtype,
            )[None, None, None, :]
            memory = self.memory_norm(memory)
            memory_flat = memory.reshape(
                batch_size * 8,
                self.circuit_token_count,
                self.d_model,
            )
            memory_mask = safe_mask[:, None, :].expand(-1, 8, -1).reshape(
                batch_size * 8,
                self.circuit_token_count,
            )
            attended, _ = self.cross_attention(
                query_flat,
                memory_flat,
                memory_flat,
                key_padding_mask=~memory_mask,
                need_weights=False,
            )
            bounded = self.residual_cap * torch.tanh(attended.float())
            bounded = bounded.reshape(batch_size, 8, count, self.d_model)
            bounded = bounded * present.to(dtype=bounded.dtype)[:, None, None, None]
            deltas.append((token_indices, bounded))
            residual_square_sum = residual_square_sum + bounded.square().sum()
            residual_count += bounded.numel()
        mean_square = residual_square_sum / max(residual_count, 1)
        ceiling_scale = self.residual_rms_ceiling / mean_square.clamp_min(
            self.residual_rms_ceiling**2
        ).sqrt()
        post_square_sum = torch.zeros_like(residual_square_sum)
        max_abs = torch.zeros_like(residual_square_sum)
        for token_indices, bounded in deltas:
            scaled = bounded * ceiling_scale
            result[:, :, token_indices, :] = (
                parameter_tokens.index_select(2, token_indices).float() + scaled
            ).to(dtype=parameter_tokens.dtype)
            detached = scaled.detach().float()
            post_square_sum = post_square_sum + detached.square().sum()
            max_abs = torch.maximum(max_abs, detached.abs().max())
        self._last_residual_rms = (
            post_square_sum / max(residual_count, 1)
        ).sqrt().detach()
        self._last_residual_max_abs = max_abs.detach()
        return result


class CircuitFusionTransformerModel(TransformerModel):
    """S2 PG Transformer with an explicit Circuit residual/attention path."""

    def __init__(
        self,
        *,
        fusion_mode: str,
        circuit_token_count: int,
        circuit_residual_scale: float = 1.0,
        circuit_global_residual_scale: float = 1.0,
        circuit_block_residual_scale: float = 1.0,
        fusion_injection_blocks: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        mode = str(fusion_mode).lower()
        if mode not in {"f2", "f3i", "dual_f3i", "circuit_p2e"}:
            raise ValueError(
                "CircuitFusionTransformerModel requires f2/f3i/dual_f3i/circuit_p2e"
            )
        if int(circuit_token_count) not in {1, 5, 6}:
            raise ValueError("Circuit fusion token count must be 1, 5, or 6")
        if mode in {"dual_f3i", "circuit_p2e"} and int(circuit_token_count) != 6:
            raise ValueError(f"{mode} requires six Candidate-AB Circuit tokens")
        residual_scale = float(circuit_residual_scale)
        if not torch.isfinite(torch.tensor(residual_scale)) or residual_scale <= 0.0:
            raise ValueError("circuit_residual_scale must be finite and positive")
        global_scale = float(circuit_global_residual_scale)
        block_scale = float(circuit_block_residual_scale)
        if any(
            not torch.isfinite(torch.tensor(value)) or value <= 0.0
            for value in (global_scale, block_scale)
        ):
            raise ValueError("Circuit global/block residual scales must be finite and positive")
        super().__init__(**kwargs)
        self.fusion_mode = mode
        self.circuit_token_count = int(circuit_token_count)
        default_blocks = (
            (0,)
            if mode in {"f2", "circuit_p2e"}
            else tuple(range(1, len(self.blocks)))
        )
        injection_blocks = (
            default_blocks
            if fusion_injection_blocks is None
            else tuple(int(index) for index in fusion_injection_blocks)
        )
        if not injection_blocks or any(index < 0 or index >= len(self.blocks) for index in injection_blocks):
            raise ValueError("fusion_injection_blocks must contain valid PG block indices")
        self.fusion_injection_blocks = injection_blocks
        self.circuit_residual_scale = residual_scale
        self.circuit_global_residual_scale = global_scale
        self.circuit_block_residual_scale = block_scale
        d_model = int(self.d_model)
        # The attention uses 8 heads at d_model=1024, matching the S5 PG
        # attention width.  A single module is reused across PG depth for
        # F3i so its parameter count is independent of the number of blocks.
        if not (mode == "f2" and self.circuit_token_count == 1) and mode != "circuit_p2e":
            self.fusion_query_norm = nn.RMSNorm(d_model, eps=1e-6)
        if mode != "circuit_p2e":
            self.fusion_circuit_norm = nn.RMSNorm(d_model, eps=1e-6)
        if mode == "f2" and self.circuit_token_count == 1:
            self.fusion_f2_summary = nn.Linear(d_model, d_model)
            self.fusion_f2_output = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.fusion_f2_output.weight)
            nn.init.zeros_(self.fusion_f2_output.bias)
        elif mode in {"f2", "f3i"}:
            attention = nn.MultiheadAttention(
                d_model,
                num_heads=8,
                dropout=0.0,
                batch_first=True,
            )
            _zero_attention_output(attention)
            if mode == "f2":
                self.fusion_f2_attention = attention
            else:
                self.fusion_f3i_attention = attention
        elif mode == "dual_f3i":
            self.fusion_global_attention = nn.MultiheadAttention(
                d_model,
                num_heads=8,
                dropout=0.0,
                batch_first=True,
            )
            self.fusion_block_attention = nn.MultiheadAttention(
                d_model,
                num_heads=8,
                dropout=0.0,
                batch_first=True,
            )
            _zero_attention_output(self.fusion_global_attention)
            _zero_attention_output(self.fusion_block_attention)
        else:
            self.fusion_circuit_p2e = CircuitP2EConditioner(
                d_model=d_model,
                circuit_token_count=self.circuit_token_count,
            )

    def fusion_parameters(self) -> list[nn.Parameter]:
        """Return parameters owned by the Circuit fusion path."""

        names = (
            "fusion_query_norm",
            "fusion_circuit_norm",
            "fusion_f2_summary",
            "fusion_f2_output",
            "fusion_f2_attention",
            "fusion_f3i_attention",
            "fusion_global_attention",
            "fusion_block_attention",
            "fusion_circuit_p2e",
        )
        result: list[nn.Parameter] = []
        seen: set[int] = set()
        for name in names:
            module = getattr(self, name, None)
            if module is None:
                continue
            for parameter in module.parameters():
                if id(parameter) not in seen:
                    result.append(parameter)
                    seen.add(id(parameter))
        return result

    @staticmethod
    def _prepare_circuit(
        circuit_tokens: Tensor | None,
        circuit_mask: Tensor | None,
        *,
        batch_size: int,
        d_model: int,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if circuit_tokens is None:
            return None, None, None
        if circuit_tokens.ndim != 3 or circuit_tokens.shape[0] != batch_size:
            raise ValueError("circuit_tokens must have shape [B,C,d_model]")
        if circuit_tokens.shape[2] != d_model:
            raise ValueError("circuit_tokens width does not match PG d_model")
        if circuit_mask is None:
            mask = torch.ones(
                circuit_tokens.shape[:2], device=circuit_tokens.device, dtype=torch.bool
            )
        else:
            if circuit_mask.shape != circuit_tokens.shape[:2]:
                raise ValueError("circuit_mask must have shape [B,C]")
            mask = circuit_mask.to(device=circuit_tokens.device, dtype=torch.bool)
        has_circuit = mask.any(dim=1)
        # MultiheadAttention returns NaNs when every key is masked.  Keep one
        # zero key for absent controls, then explicitly zero the residual for
        # those rows so absent remains a true no-Circuit control.
        safe_mask = mask.clone()
        safe_tokens = circuit_tokens
        empty = ~has_circuit
        if bool(empty.any()):
            safe_tokens = circuit_tokens.clone()
            safe_mask[empty, 0] = True
            safe_tokens[empty, 0] = 0
        return safe_tokens, safe_mask, has_circuit

    def _f2_residual(
        self,
        x: Tensor,
        circuit_tokens: Tensor,
        circuit_mask: Tensor,
        has_circuit: Tensor,
    ) -> Tensor:
        batch_size = x.shape[0]
        if self.circuit_token_count == 1:
            mask = circuit_mask.to(dtype=circuit_tokens.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            summary = (circuit_tokens * mask).sum(dim=1) / denom
            residual = self.fusion_f2_output(
                self.fusion_f2_summary(self.fusion_circuit_norm(summary))
            )
            residual = residual * has_circuit.to(dtype=residual.dtype).unsqueeze(-1)
            residual = residual * self.circuit_residual_scale
            return residual[:, None, None, :].expand_as(x)
        query = self.fusion_query_norm(x.reshape(batch_size, -1, self.d_model))
        memory = self.fusion_circuit_norm(circuit_tokens)
        residual = self.fusion_f2_attention(
            query,
            memory,
            memory,
            key_padding_mask=~circuit_mask,
            need_weights=False,
        )[0]
        residual = residual.reshape_as(x)
        residual = residual * self.circuit_residual_scale
        return residual * has_circuit.to(dtype=residual.dtype)[:, None, None, None]

    def _f3i_residual(
        self,
        x: Tensor,
        circuit_tokens: Tensor,
        circuit_mask: Tensor,
        has_circuit: Tensor,
    ) -> Tensor:
        batch_size = x.shape[0]
        query = self.fusion_query_norm(x.reshape(batch_size, -1, self.d_model))
        memory = self.fusion_circuit_norm(circuit_tokens)
        residual = self.fusion_f3i_attention(
            query,
            memory,
            memory,
            key_padding_mask=~circuit_mask,
            need_weights=False,
        )[0]
        residual = residual.reshape_as(x)
        residual = residual * self.circuit_residual_scale
        return residual * has_circuit.to(dtype=residual.dtype)[:, None, None, None]

    def _dual_f3i_residual(
        self,
        x: Tensor,
        circuit_tokens: Tensor,
        circuit_mask: Tensor,
        has_circuit: Tensor,
    ) -> Tensor:
        if circuit_tokens.shape[1] != 6:
            raise ValueError("dual-F3i requires one global plus five block tokens")
        batch_size = x.shape[0]
        query = self.fusion_query_norm(x.reshape(batch_size, -1, self.d_model))
        memory = self.fusion_circuit_norm(circuit_tokens)
        block_memory = memory[:, 1:]
        block_mask = circuit_mask[:, 1:].clone()
        empty = ~has_circuit
        if bool(empty.any()):
            block_memory = block_memory.clone()
            block_mask[empty, 0] = True
            block_memory[empty, 0] = 0
        global_residual = self.fusion_global_attention(
            query,
            memory[:, :1],
            memory[:, :1],
            key_padding_mask=~circuit_mask[:, :1],
            need_weights=False,
        )[0]
        block_residual = self.fusion_block_attention(
            query,
            block_memory,
            block_memory,
            key_padding_mask=~block_mask,
            need_weights=False,
        )[0]
        residual = (
            self.circuit_global_residual_scale * global_residual
            + self.circuit_block_residual_scale * block_residual
        ).reshape_as(x)
        residual = residual * self.circuit_residual_scale
        return residual * has_circuit.to(dtype=residual.dtype)[:, None, None, None]

    def forward(
        self,
        encoder_hidden_states: Tensor,
        condition_attention_mask: Tensor | None = None,
        *,
        circuit_tokens: Tensor | None = None,
        circuit_mask: Tensor | None = None,
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
        prepared = self._prepare_circuit(
            circuit_tokens,
            circuit_mask,
            batch_size=encoder_hidden_states.shape[0],
            d_model=self.d_model,
        )
        prepared_tokens, prepared_mask, has_circuit = prepared
        if prepared_tokens is None or prepared_mask is None or has_circuit is None:
            raise ValueError("Circuit fusion requires circuit tokens and a mask")
        # Formal S5 Circuit runs keep PG checkpointing disabled.  The plain
        # block call below deliberately mirrors TransformerModel.forward's
        # non-checkpoint path, preserving exact block order and positional
        # treatment while making the fusion insertion point explicit.
        for block_index, block in enumerate(self.blocks):
            x = block(
                x=x,
                encoder_hidden_states=encoder_hidden_states,
                hw_freqs=hw_freqs,
                n_freqs=n_freqs,
                l_freqs=l_freqs,
                all_freqs=all_freqs,
                condition_attention_mask=condition_attention_mask,
            )
            if self.fusion_mode == "f2" and block_index in self.fusion_injection_blocks:
                x = x + self._f2_residual(
                    x,
                    prepared_tokens,
                    prepared_mask,
                    has_circuit,
                )
            elif self.fusion_mode == "f3i" and block_index in self.fusion_injection_blocks:
                x = x + self._f3i_residual(
                    x,
                    prepared_tokens,
                    prepared_mask,
                    has_circuit,
                )
            elif self.fusion_mode == "dual_f3i" and block_index in self.fusion_injection_blocks:
                x = x + self._dual_f3i_residual(
                    x,
                    prepared_tokens,
                    prepared_mask,
                    has_circuit,
                )
            elif self.fusion_mode == "circuit_p2e" and block_index in self.fusion_injection_blocks:
                x = self.fusion_circuit_p2e(
                    x,
                    prepared_tokens,
                    prepared_mask,
                )
        return self.post_process(x)


def build_s5_parameter_generator_config(
    *,
    input_dim: int = S5_PG_INPUT_DIM,
    layout: S5TargetLayout | None = None,
    condition_injection_mode: str = "none",
    condition_summary_scale_init: float = 0.01,
    **overrides: Any,
) -> ParameterGeneratorConfig:
    """Create the S2 config with the S5 target and PG protocol values."""

    canonical = build_s5_target_layout()
    if layout is None:
        layout = canonical
    if layout != canonical:
        raise ValueError("S5 PG target layout is protocol-frozen")
    if int(input_dim) != S5_PG_INPUT_DIM:
        raise ValueError(
            f"S5 PG condition input_dim must be exactly {S5_PG_INPUT_DIM}"
        )
    if condition_injection_mode != "none":
        raise ValueError("S5 PG condition_injection_mode must be exactly 'none'")
    if float(condition_summary_scale_init) != 0.01:
        raise ValueError("S5 PG condition_summary_scale_init must be exactly 0.01")
    if overrides:
        unsupported = ", ".join(sorted(overrides))
        raise TypeError(f"S5 PG protocol values cannot be overridden: {unsupported}")

    return ParameterGeneratorConfig(
        token_dim=layout.token_dim,
        rank=layout.rank,
        alpha=layout.alpha,
        input_dim=int(input_dim),
        dim_accumulation=layout.dim_accumulation,
        d_model=S5_PG_D_MODEL,
        head_dim=S5_PG_HEAD_DIM,
        num_base_model_layers=layout.num_layers,
        output_dim=layout.token_dim,
        num_pg_layers=S5_PG_NUM_BLOCKS,
        prefix=layout.prefix,
        pg_mapping=layout.pg_mapping,
        target_layer_indices=list(range(layout.num_layers)),
        # S2's constructor uses this value to implement the step-0 contract.
        # Zero B output is retained while both A and B output heads remain
        # ordinary trainable Linear parameters.
        near_zero_b_init_std=0.0,
        condition_injection_mode=condition_injection_mode,
        condition_summary_scale_init=condition_summary_scale_init,
    )


def _layout_from_config(config: ParameterGeneratorConfig) -> S5TargetLayout:
    return S5TargetLayout(
        rank=int(config.rank),
        alpha=int(config.alpha),
        token_dim=int(config.token_dim),
        dim_accumulation=int(config.dim_accumulation),
    )


def validate_s5_parameter_generator_config(
    config: ParameterGeneratorConfig,
) -> S5TargetLayout:
    """Validate that a shared S2 config is exactly an S5 PG config."""

    canonical = build_s5_target_layout()
    checks = {
        "input_dim": (int(config.input_dim), S5_PG_INPUT_DIM),
        "token_dim": (int(config.token_dim), canonical.token_dim),
        "rank": (int(config.rank), canonical.rank),
        "alpha": (int(config.alpha), canonical.alpha),
        "dim_accumulation": (
            int(config.dim_accumulation),
            canonical.dim_accumulation,
        ),
        "d_model": (int(config.d_model), S5_PG_D_MODEL),
        "head_dim": (int(config.head_dim), S5_PG_HEAD_DIM),
        "num_base_model_layers": (
            int(config.num_base_model_layers),
            canonical.num_layers,
        ),
        "num_pg_layers": (int(config.num_pg_layers), S5_PG_NUM_BLOCKS),
        "output_dim": (int(config.output_dim), canonical.token_dim),
        "prefix": (str(config.prefix), canonical.prefix),
        "target_layer_indices": (
            tuple(int(index) for index in config.target_layer_indices),
            tuple(range(canonical.num_layers)),
        ),
        "condition_injection_mode": (str(config.condition_injection_mode), "none"),
        "condition_summary_scale_init": (
            float(config.condition_summary_scale_init),
            0.01,
        ),
    }
    errors = [
        f"{name}={actual!r}, expected {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if tuple(config.pg_mapping.items()) != tuple(canonical.pg_mapping.items()):
        errors.append("pg_mapping/order does not match the S5 2080/2048 target map")
    if float(config.near_zero_b_init_std) != 0.0:
        errors.append("near_zero_b_init_std must be 0.0 for step-0 B-zero init")
    if errors:
        raise ValueError("Invalid S5 ParameterGenerator config: " + "; ".join(errors))
    return canonical


class S5ParameterGenerator(S2ParameterGenerator):
    """Shared S2 PG with the S5 target layout and initialization contract."""

    def __init__(self, config: ParameterGeneratorConfig):
        self.target_layout = validate_s5_parameter_generator_config(config)
        super().__init__(config)
        if not self.model.proj_out_A.weight.requires_grad:
            raise RuntimeError("S5 PG output A must remain trainable")
        if not self.model.proj_out_B.weight.requires_grad:
            raise RuntimeError("S5 PG output B must remain trainable")
        if not bool((self.model.proj_out_B.weight == 0).all()):
            raise RuntimeError("S5 PG step-0 B output must be initialized to zero")
        if not bool((self.model.proj_out_B.bias == 0).all()):
            raise RuntimeError("S5 PG step-0 B bias must be initialized to zero")


class S5CircuitFusionParameterGenerator(S5ParameterGenerator):
    """S5 PG variant that receives Circuit tokens through F2 or F3i."""

    def __init__(
        self,
        config: ParameterGeneratorConfig,
        *,
        fusion_mode: str,
        circuit_token_count: int,
        circuit_residual_scale: float = 1.0,
        circuit_global_residual_scale: float = 1.0,
        circuit_block_residual_scale: float = 1.0,
        fusion_injection_blocks: tuple[int, ...] | None = None,
    ) -> None:
        mode = str(fusion_mode).lower()
        if mode not in {"f2", "f3i", "dual_f3i", "circuit_p2e"}:
            raise ValueError(
                "Circuit fusion parameter generator requires f2/f3i/dual_f3i/circuit_p2e"
            )
        if int(circuit_token_count) not in {1, 5, 6}:
            raise ValueError("Circuit fusion token count must be 1, 5, or 6")
        super().__init__(config)
        base_model = self.model
        base_state = base_model.state_dict()
        self.model = CircuitFusionTransformerModel(
            fusion_mode=mode,
            circuit_token_count=int(circuit_token_count),
            circuit_residual_scale=circuit_residual_scale,
            circuit_global_residual_scale=circuit_global_residual_scale,
            circuit_block_residual_scale=circuit_block_residual_scale,
            fusion_injection_blocks=fusion_injection_blocks,
            d_model=base_model.d_model,
            num_base_model_layers=base_model.seq_n,
            num_token_per_layer=base_model.seq_h,
            lora_rank=base_model.seq_w,
            dim_accumulation=base_model.dim_accumulation,
            output_dim=base_model.output_dim,
            head_dim=base_model.head_dim,
            num_blocks=len(base_model.blocks),
            lora_A_token_count=base_model.lora_A_token_count,
            lora_B_token_count=base_model.lora_B_token_count,
            condition_injection_mode=base_model.condition_injection_mode,
            condition_summary_scale_init=0.01,
        )
        missing, unexpected = self.model.load_state_dict(base_state, strict=False)
        if unexpected or any(not key.startswith("fusion_") for key in missing):
            raise RuntimeError(
                "Circuit fusion PG failed to inherit the common S2 PG state: "
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )
        self.fusion_mode = mode
        self.circuit_token_count = int(circuit_token_count)

    def fusion_parameters(self) -> list[nn.Parameter]:
        return self.model.fusion_parameters()

    def validate_fusion_step0_initialization(self) -> None:
        attentions = [
            attention
            for name in (
                "fusion_f2_attention",
                "fusion_f3i_attention",
                "fusion_global_attention",
                "fusion_block_attention",
            )
            if (attention := getattr(self.model, name, None)) is not None
        ]
        output = getattr(self.model, "fusion_f2_output", None)
        if output is not None:
            if not bool(torch.equal(output.weight, torch.zeros_like(output.weight))):
                raise RuntimeError("S5 F2 output projection is not zero initialized")
            if not bool(torch.equal(output.bias, torch.zeros_like(output.bias))):
                raise RuntimeError("S5 F2 output bias is not zero initialized")
        for attention in attentions:
            if not bool(
                torch.equal(attention.out_proj.weight, torch.zeros_like(attention.out_proj.weight))
            ):
                raise RuntimeError("S5 Circuit attention output projection is not zero initialized")
            if attention.out_proj.bias is not None and not bool(
                torch.equal(attention.out_proj.bias, torch.zeros_like(attention.out_proj.bias))
            ):
                raise RuntimeError("S5 Circuit attention output bias is not zero initialized")
        p2e = getattr(self.model, "fusion_circuit_p2e", None)
        if p2e is not None:
            p2e.validate_step0_initialization()

    def forward(
        self,
        condition: Tensor,
        condition_attention_mask: Tensor | None = None,
        *,
        circuit_tokens: Tensor | None = None,
        circuit_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if circuit_tokens is None:
            raise ValueError("Circuit fusion PG requires circuit_tokens")
        embeddings = self.hidden_in(condition)
        output = self.model(
            encoder_hidden_states=embeddings,
            condition_attention_mask=condition_attention_mask,
            circuit_tokens=circuit_tokens,
            circuit_mask=circuit_mask,
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
        all_layer_state_dict["scale"] = torch.tensor(
            self.tokenizer.lora_scale,
            device=output.device,
            dtype=output.dtype,
        )
        return all_layer_state_dict


# Keep the S2 names available from the S5 adapter module for callers that use
# the shared implementation directly, while making the default S5 import safe.
ParameterGenerator = S5ParameterGenerator


def build_s5_parameter_generator(
    *, input_dim: int = S5_PG_INPUT_DIM, layout: S5TargetLayout | None = None
) -> S5ParameterGenerator:
    """Build a protocol-default S5 PG ready to emit batched LoRA state."""

    config = build_s5_parameter_generator_config(input_dim=input_dim, layout=layout)
    return S5ParameterGenerator(config)


def build_s5_circuit_fusion_parameter_generator(
    *,
    fusion_mode: str,
    circuit_token_count: int,
    circuit_residual_scale: float = 1.0,
    circuit_global_residual_scale: float = 1.0,
    circuit_block_residual_scale: float = 1.0,
    fusion_injection_blocks: tuple[int, ...] | None = None,
    input_dim: int = S5_PG_INPUT_DIM,
    layout: S5TargetLayout | None = None,
) -> S5CircuitFusionParameterGenerator:
    """Build a zero-initialized F2/F3i PG with a separate Circuit path."""

    config = build_s5_parameter_generator_config(input_dim=input_dim, layout=layout)
    return S5CircuitFusionParameterGenerator(
        config,
        fusion_mode=fusion_mode,
        circuit_token_count=circuit_token_count,
        circuit_residual_scale=circuit_residual_scale,
        circuit_global_residual_scale=circuit_global_residual_scale,
        circuit_block_residual_scale=circuit_block_residual_scale,
        fusion_injection_blocks=fusion_injection_blocks,
    )


__all__ = [
    "ParameterGenerator",
    "ParameterGeneratorConfig",
    "CircuitP2EConditioner",
    "CircuitFusionTransformerModel",
    "S2ParameterGenerator",
    "S5CircuitFusionParameterGenerator",
    "S5ParameterGenerator",
    "S5_FUSION_MODES",
    "S5_PG_D_MODEL",
    "S5_PG_HEAD_DIM",
    "S5_PG_INPUT_DIM",
    "S5_PG_NUM_BLOCKS",
    "Tokenizer2DBatchedLoRA",
    "build_s5_parameter_generator",
    "build_s5_circuit_fusion_parameter_generator",
    "build_s5_parameter_generator_config",
    "validate_s5_parameter_generator_config",
]

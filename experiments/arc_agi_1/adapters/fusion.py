from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .contracts import FUSION_VARIANTS
from .encoders import EncodedCircuit, SingleCircuitEncoder, StructuredCircuitEncoder
from .representations import validate_raw_vector59


@dataclass(frozen=True)
class FusionInputs:
    direct_residual: Tensor
    memory_tokens: Tensor | None
    memory_mask: Tensor | None
    encoded: EncodedCircuit | None


def append_masked_memory(
    prompt: Tensor,
    prompt_mask: Tensor,
    extra: Tensor,
    extra_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Append extra tokens after each row's valid prompt tokens, then right-pad."""

    if prompt.ndim != 3 or prompt_mask.shape != prompt.shape[:2]:
        raise ValueError("Prompt memory and mask are misaligned")
    if extra.ndim != 3 or extra.shape[0] != prompt.shape[0]:
        raise ValueError("Extra memory must have shape [B, E, H]")
    if extra.shape[2] != prompt.shape[2] or extra_mask.shape != extra.shape[:2]:
        raise ValueError("Extra memory and mask are misaligned")
    prompt_bool = prompt_mask.to(device=prompt.device, dtype=torch.bool)
    extra_bool = extra_mask.to(device=prompt.device, dtype=torch.bool)
    rows: list[Tensor] = []
    lengths: list[int] = []
    for row_index in range(prompt.shape[0]):
        valid_prompt = prompt[row_index][prompt_bool[row_index]]
        valid_extra = extra[row_index][extra_bool[row_index]]
        row = torch.cat((valid_prompt, valid_extra), dim=0)
        rows.append(row)
        lengths.append(row.shape[0])
    memory = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
    positions = torch.arange(memory.shape[1], device=memory.device).unsqueeze(0)
    mask = positions.lt(torch.tensor(lengths, device=memory.device).unsqueeze(1))
    return memory, mask.to(dtype=prompt_mask.dtype)


class CircuitFusion(nn.Module):
    def __init__(
        self,
        variant_id: str,
        *,
        encoder_hidden_dim: int = 128,
        pg_hidden_dim: int = 1_280,
        generator_slots: int = 37,
        input_mode: str = "raw_v1",
    ) -> None:
        super().__init__()
        if variant_id not in FUSION_VARIANTS:
            raise ValueError(f"Unsupported circuit fusion: {variant_id}")
        if generator_slots != 37:
            raise ValueError("S4 R1 requires 36 layer slots plus lm_head")
        self.variant_id = variant_id
        self.pg_hidden_dim = pg_hidden_dim
        self.generator_slots = generator_slots
        if variant_id == "f0_single_token":
            self.single_encoder: SingleCircuitEncoder | None = SingleCircuitEncoder(
                hidden_dim=encoder_hidden_dim,
                output_dim=pg_hidden_dim,
            )
            self.structured_encoder: StructuredCircuitEncoder | None = None
        else:
            self.single_encoder = None
            self.structured_encoder = StructuredCircuitEncoder(
                hidden_dim=encoder_hidden_dim,
                output_dim=pg_hidden_dim,
                # F3 already has a zero residual projection. Zeroing both maps
                # would make the Global/Topology branch permanently untrainable.
                zero_memory_outputs=variant_id
                not in {"f3_shared_attention", "f3_independent_attention"},
                input_mode=input_mode,
            )

    def encode(
        self,
        vectors: Tensor,
        *,
        circuit_present: Tensor | None = None,
        output_dtype: torch.dtype,
    ) -> FusionInputs:
        validate_raw_vector59(vectors)
        batch_size = vectors.shape[0]
        if circuit_present is None:
            present = torch.ones(batch_size, device=vectors.device, dtype=torch.bool)
        else:
            present = circuit_present.to(device=vectors.device, dtype=torch.bool)
        direct = torch.zeros(
            batch_size,
            self.generator_slots,
            self.pg_hidden_dim,
            device=vectors.device,
            dtype=output_dtype,
        )

        if self.variant_id == "f0_single_token":
            assert self.single_encoder is not None
            token = self.single_encoder(vectors).to(dtype=output_dtype).unsqueeze(1)
            return FusionInputs(
                direct_residual=direct,
                memory_tokens=token,
                memory_mask=present[:, None],
                encoded=None,
            )

        assert self.structured_encoder is not None
        encoded = self.structured_encoder(
            vectors,
            circuit_present=present,
        )
        global_context = encoded.global_context.to(dtype=output_dtype)
        topology_context = encoded.topology_context.to(dtype=output_dtype)
        layer_context = encoded.layer_context.to(dtype=output_dtype)

        if self.variant_id in {
            "f1_layer_shared_memory",
            "f3_shared_attention",
            "f3_independent_attention",
        }:
            direct[:, :36] = layer_context
        elif self.variant_id == "f2_fully_direct":
            shared = global_context + topology_context
            direct[:, :36] = shared[:, None, :] + layer_context
            direct[:, 36] = shared

        memory_tokens = None
        memory_mask = None
        if self.variant_id in {
            "f1_layer_shared_memory",
            "f3_shared_attention",
            "f3_independent_attention",
        }:
            memory_tokens = torch.stack((global_context, topology_context), dim=1)
            memory_mask = present[:, None].expand(-1, 2)

        return FusionInputs(
            direct_residual=direct,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            encoded=encoded,
        )

    def encoder_parameters(self):
        if self.single_encoder is not None:
            yield from self.single_encoder.parameters()
        if self.structured_encoder is not None:
            yield from self.structured_encoder.parameters()

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from s2_hypernet.nvarc_pg.model import EncodedRow, FrozenQwenPerQueryLoRA

from .conditioning import HistoryEffectiveWeightConditioner


class HistoryEffectiveWeightSystem(FrozenQwenPerQueryLoRA):
    """Frozen Qwen + PG runtime with detached per-row effective-weight history."""

    def _conditioner(self) -> HistoryEffectiveWeightConditioner:
        parameter_generator: nn.Module = getattr(
            self.parameter_generator, "module", self.parameter_generator
        )
        generator = getattr(parameter_generator, "generator", None)
        model = getattr(generator, "model", None)
        conditioner = getattr(model, "raw_weight_conditioner", None)
        if not isinstance(conditioner, HistoryEffectiveWeightConditioner):
            raise TypeError("History system has no HistoryEffectiveWeightConditioner")
        return conditioner

    def pack_histories(
        self,
        histories: Sequence[Sequence[Tensor]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        return self._conditioner().pack_histories(
            histories,
            device=device,
            dtype=dtype,
        )

    def prepare_history_features(
        self,
        state: Mapping[str, Tensor],
        *,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> Tensor:
        return self._conditioner().project_effective_features(
            state,
            output_dtype=output_dtype,
        )

    def generate_lora_from_condition(
        self,
        condition: Tensor,
        attention_mask: Tensor,
        *,
        histories: Sequence[Sequence[Tensor]],
    ) -> Mapping[str, Tensor]:
        history_features, history_mask = self.pack_histories(
            histories,
            device=condition.device,
            dtype=condition.dtype,
        )
        return self.parameter_generator(
            condition,
            condition_attention_mask=attention_mask,
            history_features=history_features,
            history_mask=history_mask,
        )

    def generate_lora_batch_with_histories(
        self,
        encoded_rows: Sequence[EncodedRow],
        *,
        event_indices: Sequence[int],
        histories: Sequence[Sequence[Tensor]],
    ) -> Mapping[str, Tensor]:
        if (
            not encoded_rows
            or len(encoded_rows) != len(event_indices)
            or len(encoded_rows) != len(histories)
        ):
            raise ValueError(
                "encoded_rows, event_indices, and histories must have equal non-zero length"
            )
        sequences = []
        for encoded, event_index in zip(encoded_rows, event_indices, strict=True):
            event = encoded.row.events[int(event_index)]
            sequences.append(encoded.hidden_states[0, : event.condition_end])
        condition = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences],
            device=condition.device,
            dtype=torch.long,
        )
        positions = torch.arange(
            condition.shape[1], device=condition.device
        ).unsqueeze(0)
        attention_mask = positions.lt(lengths.unsqueeze(1)).to(dtype=torch.long)
        return self.generate_lora_from_condition(
            condition,
            attention_mask,
            histories=histories,
        )

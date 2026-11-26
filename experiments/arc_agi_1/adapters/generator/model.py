from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from .events import TokenizedRow
from .generated_lora import GeneratedLoRAController
from .objective import causal_loss_sum


@dataclass(frozen=True)
class EncodedRow:
    row: TokenizedRow
    hidden_states: Tensor


def _pad_long_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    padding_value: int,
    device: torch.device,
) -> Tensor:
    """Pack Python token sequences on CPU and issue one device transfer."""

    padded = torch.nn.utils.rnn.pad_sequence(
        [
            torch.tensor(sequence, device="cpu", dtype=torch.long)
            for sequence in sequences
        ],
        batch_first=True,
        padding_value=padding_value,
    )
    return padded.to(device=device)


class FrozenQwenPerQueryLoRA:
    """Own the no-LoRA condition pass and scoped adapted event pass."""

    def __init__(
        self,
        base_model: nn.Module,
        parameter_generator: nn.Module,
        *,
        target_keys: tuple[str, ...],
    ) -> None:
        self.base_model = base_model
        self.parameter_generator = parameter_generator
        self.controller = GeneratedLoRAController(
            base_model,
            target_keys=target_keys,
        )
        self.freeze_base_model()

    def freeze_base_model(self) -> None:
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

    def encode_row(self, row: TokenizedRow, *, device: torch.device) -> EncodedRow:
        return self.encode_rows((row,), device=device)[0]

    def encode_rows(
        self,
        rows: Sequence[TokenizedRow],
        *,
        device: torch.device,
    ) -> tuple[EncodedRow, ...]:
        """Condition-encode a real row microbatch with right padding."""

        if not rows:
            raise ValueError("rows must not be empty")
        row_tuple = tuple(rows)
        lengths = torch.tensor(
            [row.length for row in row_tuple],
            device=device,
            dtype=torch.long,
        )
        input_ids = _pad_long_sequences(
            [row.input_ids for row in row_tuple],
            padding_value=0,
            device=device,
        )
        positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        attention_mask = positions.lt(lengths.unsqueeze(1)).to(dtype=torch.long)
        with torch.no_grad():
            backbone = getattr(self.base_model, "model", None)
            if backbone is None:
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                if not outputs.hidden_states:
                    raise RuntimeError(
                        "Frozen Qwen condition pass did not return hidden states"
                    )
                hidden_states = outputs.hidden_states[-1].detach()
            else:
                outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                if outputs.last_hidden_state is None:
                    raise RuntimeError(
                        "Frozen Qwen backbone condition pass did not return last_hidden_state"
                    )
                hidden_states = outputs.last_hidden_state.detach()
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("Frozen Qwen hidden states do not align with row tokens")
        return tuple(
            EncodedRow(
                row=row,
                hidden_states=hidden_states[
                    batch_index : batch_index + 1, : row.length
                ],
            )
            for batch_index, row in enumerate(row_tuple)
        )

    def condition_batch(
        self,
        encoded: EncodedRow,
        *,
        event_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        if not event_indices:
            raise ValueError("event_indices must not be empty")
        sequences = []
        for event_index in event_indices:
            event = encoded.row.events[int(event_index)]
            sequences.append(encoded.hidden_states[0, : event.condition_end])
        condition = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        attention_mask = torch.zeros(
            condition.shape[:2],
            device=condition.device,
            dtype=torch.long,
        )
        for batch_index, sequence in enumerate(sequences):
            attention_mask[batch_index, : sequence.shape[0]] = 1
        return condition, attention_mask

    def generate_lora(
        self,
        encoded: EncodedRow,
        *,
        event_indices: Sequence[int],
    ) -> Mapping[str, Tensor]:
        condition, attention_mask = self.condition_batch(
            encoded,
            event_indices=event_indices,
        )
        return self.parameter_generator(
            condition,
            condition_attention_mask=attention_mask,
        )

    def generate_lora_batch(
        self,
        encoded_rows: Sequence[EncodedRow],
        *,
        event_indices: Sequence[int],
    ) -> Mapping[str, Tensor]:
        """Generate one distinct LoRA for each row/event pair in a batch."""

        if not encoded_rows or len(encoded_rows) != len(event_indices):
            raise ValueError("encoded_rows and event_indices must have equal non-zero length")
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
        positions = torch.arange(condition.shape[1], device=condition.device).unsqueeze(0)
        attention_mask = positions.lt(lengths.unsqueeze(1)).to(dtype=torch.long)
        return self.parameter_generator(
            condition,
            condition_attention_mask=attention_mask,
        )

    def activate(self, state: Mapping[str, Tensor]) -> AbstractContextManager[Any]:
        return self.controller.activate(state)

    def set_lora_scale_multiplier(self, value: float) -> None:
        """Set an evaluation-time multiplier on the generated LoRA residual."""

        self.controller.set_scale_multiplier(value)

    def set_lora_target_scope(self, scope: str) -> None:
        """Restrict which generated LoRA targets affect evaluation forwards."""

        self.controller.set_target_scope(scope)

    def event_tensors(
        self,
        row: TokenizedRow,
        *,
        event_index: int,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        event = row.events[int(event_index)]
        if device is None:
            device = next(self.base_model.parameters()).device
        input_ids = torch.tensor(
            row.input_ids[: event.response_end], device=device, dtype=torch.long
        ).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        labels = torch.tensor(event.labels, device=device, dtype=torch.long).unsqueeze(0)
        return input_ids, attention_mask, labels

    def event_batch_tensors(
        self,
        rows: Sequence[TokenizedRow],
        *,
        event_indices: Sequence[int],
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not rows or len(rows) != len(event_indices):
            raise ValueError("rows and event_indices must have equal non-zero length")
        if device is None:
            device = next(self.base_model.parameters()).device
        inputs: list[Sequence[int]] = []
        labels: list[Sequence[int]] = []
        for row, event_index in zip(rows, event_indices, strict=True):
            event = row.events[int(event_index)]
            inputs.append(row.input_ids[: event.response_end])
            labels.append(event.labels)
        input_ids = _pad_long_sequences(
            inputs,
            padding_value=0,
            device=device,
        )
        padded_labels = _pad_long_sequences(
            labels,
            padding_value=-100,
            device=device,
        )
        lengths = torch.tensor(
            [len(value) for value in inputs],
            device=device,
            dtype=torch.long,
        )
        positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        attention_mask = positions.lt(lengths.unsqueeze(1)).to(dtype=torch.long)
        return input_ids, attention_mask, padded_labels

    def forward_event(self, row: TokenizedRow, *, event_index: int) -> Any:
        input_ids, attention_mask, _ = self.event_tensors(row, event_index=event_index)
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    def forward_event_batch(
        self,
        rows: Sequence[TokenizedRow],
        *,
        event_indices: Sequence[int],
    ) -> Any:
        input_ids, attention_mask, _ = self.event_batch_tensors(
            rows,
            event_indices=event_indices,
        )
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )

    def event_loss_sum(
        self,
        row: TokenizedRow,
        *,
        event_index: int,
    ) -> tuple[Tensor, Tensor]:
        _, _, labels = self.event_tensors(row, event_index=event_index)
        outputs = self.forward_event(row, event_index=event_index)
        return causal_loss_sum(outputs.logits, labels)

    def event_batch_loss_sum(
        self,
        rows: Sequence[TokenizedRow],
        *,
        event_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        input_ids, attention_mask, labels = self.event_batch_tensors(
            rows,
            event_indices=event_indices,
        )
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
        return causal_loss_sum(outputs.logits, labels)

    def activation_summary(self) -> dict[str, object]:
        return self.controller.activation_summary()

    def close(self) -> None:
        self.controller.close()

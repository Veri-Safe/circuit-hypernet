from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor
from torch import nn

from s2_hypernet.nvarc_pg.events import TokenizedRow
from s2_hypernet.nvarc_pg.model import EncodedRow, FrozenQwenPerQueryLoRA
from s2_hypernet.nvarc_pg.objective import IGNORE_INDEX


FULL_LOGITS = "full_logits_v1"
RESPONSE_ONLY_LOGITS = "response_only_logits_v1"


class FrozenQwenCircuitPerQueryLoRA(FrozenQwenPerQueryLoRA):
    """S2 two-pass model with an explicit circuit vector for each event."""

    def __init__(
        self,
        base_model: nn.Module,
        parameter_generator: nn.Module,
        *,
        target_keys: tuple[str, ...],
        loss_backend: str = FULL_LOGITS,
    ) -> None:
        super().__init__(
            base_model,
            parameter_generator,
            target_keys=target_keys,
        )
        if loss_backend not in {FULL_LOGITS, RESPONSE_ONLY_LOGITS}:
            raise ValueError(f"Unsupported S4 loss backend: {loss_backend}")
        self.loss_backend = loss_backend

    def generate_lora_batch_with_circuits(
        self,
        encoded_rows: Sequence[EncodedRow],
        *,
        event_indices: Sequence[int],
        circuit_vectors: Tensor,
        circuit_present: Tensor | None = None,
    ) -> Mapping[str, Tensor]:
        if not encoded_rows or len(encoded_rows) != len(event_indices):
            raise ValueError(
                "encoded_rows and event_indices must have equal non-zero length"
            )
        if circuit_vectors.shape != (len(encoded_rows), 59):
            raise ValueError("circuit_vectors must have shape [batch,59]")
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
        positions = torch.arange(condition.shape[1], device=condition.device).unsqueeze(
            0
        )
        attention_mask = positions.lt(lengths.unsqueeze(1)).to(dtype=torch.long)
        return self.parameter_generator(
            condition,
            condition_attention_mask=attention_mask,
            circuit_vector=circuit_vectors.to(
                device=condition.device, dtype=torch.float32
            ),
            circuit_present=circuit_present,
        )

    def _response_only_event_batch_loss_sum(
        self,
        rows: Sequence[TokenizedRow],
        *,
        event_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        input_ids, attention_mask, _ = self.event_batch_tensors(
            rows,
            event_indices=event_indices,
        )
        backbone = getattr(self.base_model, "model", None)
        lm_head = getattr(self.base_model, "lm_head", None)
        if backbone is None or lm_head is None:
            raise TypeError(
                "response_only_logits_v1 requires a causal LM with model and lm_head"
            )
        outputs = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.last_hidden_state
        if hidden_states.shape[:2] != input_ids.shape:
            raise ValueError("Base hidden states do not align with event inputs")

        selected_hidden = []
        targets = []
        for batch_index, (row, event_index) in enumerate(
            zip(rows, event_indices, strict=True)
        ):
            event = row.events[int(event_index)]
            if event.response_start <= 0:
                raise ValueError("Response-only next-token loss needs a prompt token")
            prediction_start = event.response_start - 1
            prediction_end = event.response_end - 1
            selected_hidden.append(
                hidden_states[batch_index, prediction_start:prediction_end]
            )
            target = torch.tensor(
                event.labels[event.response_start : event.response_end],
                device=input_ids.device,
                dtype=torch.long,
            )
            expected = input_ids[batch_index, event.response_start : event.response_end]
            if target.numel() <= 0 or target.eq(IGNORE_INDEX).any():
                raise ValueError("Response event has invalid supervised targets")
            if not torch.equal(target, expected):
                raise ValueError("Response labels no longer match event input IDs")
            targets.append(target)

        padded_hidden = torch.nn.utils.rnn.pad_sequence(
            selected_hidden,
            batch_first=True,
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        logits = lm_head(padded_hidden)
        if logits.shape[:2] != padded_targets.shape:
            raise ValueError("Response-only logits do not align with response targets")
        token_count = padded_targets.ne(IGNORE_INDEX).sum()
        loss_sum = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            padded_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        return loss_sum, token_count

    def event_batch_loss_sum(
        self,
        rows: Sequence[TokenizedRow],
        *,
        event_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        if self.loss_backend == FULL_LOGITS:
            return super().event_batch_loss_sum(rows, event_indices=event_indices)
        return self._response_only_event_batch_loss_sum(
            rows,
            event_indices=event_indices,
        )

    def frozen_base_loss_sum_from_encoded(
        self,
        encoded_rows: Sequence[EncodedRow],
        *,
        event_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        """Compute no-LoRA response CE from the existing frozen hidden cache."""

        if not encoded_rows or len(encoded_rows) != len(event_indices):
            raise ValueError(
                "encoded_rows and event_indices must have equal non-zero length"
            )
        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            raise TypeError("Base-preservation CE requires a causal LM head")
        selected_hidden = []
        targets = []
        for encoded, event_index in zip(
            encoded_rows,
            event_indices,
            strict=True,
        ):
            event = encoded.row.events[int(event_index)]
            if event.response_start <= 0:
                raise ValueError("Base-preservation CE needs a prompt token")
            selected_hidden.append(
                encoded.hidden_states[
                    0,
                    event.response_start - 1 : event.response_end - 1,
                ]
            )
            target = torch.tensor(
                event.labels[event.response_start : event.response_end],
                device=encoded.hidden_states.device,
                dtype=torch.long,
            )
            if target.numel() <= 0 or target.eq(IGNORE_INDEX).any():
                raise ValueError("Base-preservation event has invalid targets")
            targets.append(target)
        padded_hidden = torch.nn.utils.rnn.pad_sequence(
            selected_hidden,
            batch_first=True,
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        with torch.no_grad():
            logits = lm_head(padded_hidden)
            loss_sums, token_counts = self._per_example_loss_sums(
                logits,
                padded_targets,
            )
        return loss_sums.sum(), token_counts.sum()

    @staticmethod
    def _per_example_loss_sums(
        logits: Tensor,
        targets: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if logits.shape[:2] != targets.shape:
            raise ValueError("Per-example logits and targets do not align")
        mask = targets.ne(IGNORE_INDEX)
        token_counts = mask.sum(dim=1)
        if token_counts.le(0).any():
            raise ValueError("Per-example response contains no supervised tokens")
        losses = torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape_as(targets)
        return (losses * mask).sum(dim=1), token_counts

    def frozen_base_loss_sums_from_encoded(
        self,
        encoded_rows: Sequence[EncodedRow],
        *,
        event_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        """Return one cached-Base response CE numerator per event."""

        if not encoded_rows or len(encoded_rows) != len(event_indices):
            raise ValueError(
                "encoded_rows and event_indices must have equal non-zero length"
            )
        lm_head = getattr(self.base_model, "lm_head", None)
        if lm_head is None:
            raise TypeError("Base-preservation CE requires a causal LM head")
        selected_hidden = []
        targets = []
        for encoded, event_index in zip(
            encoded_rows,
            event_indices,
            strict=True,
        ):
            event = encoded.row.events[int(event_index)]
            if event.response_start <= 0:
                raise ValueError("Base-preservation CE needs a prompt token")
            selected_hidden.append(
                encoded.hidden_states[
                    0,
                    event.response_start - 1 : event.response_end - 1,
                ]
            )
            target = torch.tensor(
                event.labels[event.response_start : event.response_end],
                device=encoded.hidden_states.device,
                dtype=torch.long,
            )
            if target.numel() <= 0 or target.eq(IGNORE_INDEX).any():
                raise ValueError("Base-preservation event has invalid targets")
            targets.append(target)
        padded_hidden = torch.nn.utils.rnn.pad_sequence(
            selected_hidden,
            batch_first=True,
        )
        padded_targets = torch.nn.utils.rnn.pad_sequence(
            targets,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        with torch.no_grad():
            return self._per_example_loss_sums(
                lm_head(padded_hidden),
                padded_targets,
            )

    def event_batch_loss_sums(
        self,
        rows: Sequence[TokenizedRow],
        *,
        event_indices: Sequence[int],
    ) -> tuple[Tensor, Tensor]:
        """Return one adapted response CE numerator per event."""

        if self.loss_backend != FULL_LOGITS:
            raise ValueError(
                "Per-event base preservation currently requires full_logits_v1"
            )
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
        return self._per_example_loss_sums(
            outputs.logits[:, :-1, :].contiguous(),
            labels[:, 1:].contiguous(),
        )

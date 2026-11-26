from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Sequence

import torch
from torch import Tensor


EXACT_VJP_EXECUTOR = "codecircuit_exact_vjp_executor_v1"


def _inject_exact(
    grads: Tensor,
    *,
    batch_indices: Tensor,
    pos_indices: Tensor,
    values: Tensor,
) -> Tensor:
    """Preserve CodeCircuit's gradient replacement expression exactly."""

    grads_out = grads.clone().to(values.dtype)
    grads_out.index_put_((batch_indices, pos_indices), values)
    return grads_out.to(grads.dtype)


@dataclass(frozen=True)
class ExactVJPExecutorStats:
    identity: str
    calls: int
    rows: int
    layer_groups: int
    buffer_allocations: int
    buffer_reuses: int
    seed_allocations: int
    seed_reuses: int
    batch_index_allocations: int
    batch_index_reuses: int
    avoided_layer_syncs: int
    cuda_milliseconds: float

    def as_metrics(self) -> dict[str, Any]:
        return {f"exact_vjp_{key}": value for key, value in asdict(self).items()}


class ExactVJPExecutor:
    """CodeCircuit-equivalent VJP execution with reusable scratch tensors."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self._batch_buffer: Tensor | None = None
        self._zero_seed: Tensor | None = None
        self._batch_indices: Tensor | None = None
        self._timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.calls = 0
        self.rows = 0
        self.layer_groups = 0
        self.buffer_allocations = 0
        self.buffer_reuses = 0
        self.seed_allocations = 0
        self.seed_reuses = 0
        self.batch_index_allocations = 0
        self.batch_index_reuses = 0
        self.avoided_layer_syncs = 0

    @staticmethod
    def _same_storage_contract(left: Tensor | None, right: Tensor) -> bool:
        return left is not None and (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.layout == right.layout
            and left.stride() == right.stride()
        )

    def _row_buffer(self, *, batch_size: int, values: Tensor) -> Tensor:
        shape = (int(self.context._row_size), batch_size)
        reusable = self._batch_buffer
        if reusable is None or (
            reusable.shape != shape
            or reusable.dtype != values.dtype
            or reusable.device != values.device
        ):
            reusable = torch.zeros(shape, dtype=values.dtype, device=values.device)
            self._batch_buffer = reusable
            self.buffer_allocations += 1
        else:
            reusable.zero_()
            self.buffer_reuses += 1
        return reusable

    def _backward_seed(self, residual: Tensor) -> Tensor:
        if not self._same_storage_contract(self._zero_seed, residual):
            self._zero_seed = torch.zeros_like(residual)
            self.seed_allocations += 1
        else:
            self.seed_reuses += 1
        return self._zero_seed

    def _all_batch_indices(self, *, batch_size: int, device: torch.device) -> Tensor:
        indices = self._batch_indices
        if indices is None or indices.numel() != batch_size or indices.device != device:
            indices = torch.arange(batch_size, device=device)
            self._batch_indices = indices
            self.batch_index_allocations += 1
        else:
            self.batch_index_reuses += 1
        return indices

    def compute_batch(
        self,
        *,
        layer_ids: Sequence[int],
        positions: Tensor,
        inject_values: Tensor,
        retain_graph: bool = True,
    ) -> Tensor:
        row_count = len(layer_ids)
        if row_count <= 0:
            raise ValueError("Exact VJP requires at least one target row")
        if positions.ndim != 1 or positions.numel() != row_count:
            raise ValueError("Exact VJP target positions changed shape")
        if inject_values.ndim != 2 or inject_values.shape[0] != row_count:
            raise ValueError("Exact VJP injection values changed shape")
        if min(layer_ids) < 0 or max(layer_ids) >= int(self.context.n_layers):
            raise ValueError("Exact VJP target layer is outside the model")

        first_residual = self.context._resid_activations[0]
        if first_residual is None:
            raise RuntimeError("Residual activations are not cached")
        batch_size = int(first_residual.shape[0])
        if row_count > batch_size:
            raise ValueError("Exact VJP target count exceeds the physical batch")

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream(inject_values.device))

        buffer = self._row_buffer(batch_size=batch_size, values=inject_values)
        self.context._batch_buffer = buffer
        batch_indices = self._all_batch_indices(
            batch_size=batch_size, device=positions.device
        )
        grouped_offsets: dict[int, list[int]] = {}
        for offset, layer in enumerate(layer_ids):
            grouped_offsets.setdefault(int(layer), []).append(offset)

        handles = []
        try:
            for layer in sorted(grouped_offsets):
                offsets = torch.tensor(
                    grouped_offsets[layer], dtype=torch.long, device=positions.device
                )
                fn = partial(
                    _inject_exact,
                    batch_indices=batch_indices.index_select(0, offsets),
                    pos_indices=positions.index_select(0, offsets),
                    values=inject_values.index_select(0, offsets),
                )
                residual = self.context._resid_activations[layer]
                if residual is None:
                    raise RuntimeError("Residual activation is missing")
                handles.append(residual.register_hook(fn))

            last_layer = max(grouped_offsets)
            last_residual = self.context._resid_activations[last_layer]
            if last_residual is None:
                raise RuntimeError("Last residual activation is missing")
            last_residual.backward(
                gradient=self._backward_seed(last_residual),
                retain_graph=retain_graph,
            )
        finally:
            for handle in handles:
                handle.remove()
            self.context._batch_buffer = None

        end_event.record(torch.cuda.current_stream(inject_values.device))
        self._timing_events.append((start_event, end_event))
        self.calls += 1
        self.rows += row_count
        self.layer_groups += len(grouped_offsets)
        self.avoided_layer_syncs += 1
        return buffer.T[:row_count]

    def stats(self) -> ExactVJPExecutorStats:
        cuda_milliseconds = 0.0
        for start_event, end_event in self._timing_events:
            end_event.synchronize()
            cuda_milliseconds += float(start_event.elapsed_time(end_event))
        return ExactVJPExecutorStats(
            identity=EXACT_VJP_EXECUTOR,
            calls=self.calls,
            rows=self.rows,
            layer_groups=self.layer_groups,
            buffer_allocations=self.buffer_allocations,
            buffer_reuses=self.buffer_reuses,
            seed_allocations=self.seed_allocations,
            seed_reuses=self.seed_reuses,
            batch_index_allocations=self.batch_index_allocations,
            batch_index_reuses=self.batch_index_reuses,
            avoided_layer_syncs=self.avoided_layer_syncs,
            cuda_milliseconds=cuda_milliseconds,
        )

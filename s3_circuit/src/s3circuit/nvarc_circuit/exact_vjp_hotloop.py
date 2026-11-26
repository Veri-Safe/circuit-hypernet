from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from .exact_vjp import ExactVJPExecutor, _inject_exact


EXACT_VJP_HOTLOOP_EXECUTOR = "codecircuit_exact_vjp_hotloop_executor_v2"


@dataclass(frozen=True)
class ExactVJPHotLoopStats:
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
    hook_registrations: int
    hook_removals: int
    avoided_hook_registrations: int
    avoided_hook_removals: int
    plan_builds: int
    fallbacks: int

    def as_metrics(self) -> dict[str, Any]:
        return {f"exact_vjp_{key}": value for key, value in asdict(self).items()}


class ExactVJPHotLoopExecutor(ExactVJPExecutor):
    """Reuse residual hooks within one event without changing VJP expressions."""

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        self._active_plans: list[tuple[Tensor, Tensor, Tensor] | None] = [
            None
        ] * int(context.n_layers)
        self._handles: list[Any] = []
        self._closed = False
        self._in_compute = False
        self.hook_registrations = 0
        self.hook_removals = 0
        self.avoided_hook_registrations = 0
        self.avoided_hook_removals = 0
        self.plan_builds = 0
        self.fallbacks = 0
        try:
            for layer in range(int(context.n_layers)):
                residual = context._resid_activations[layer]
                if residual is None:
                    raise RuntimeError(
                        f"Residual activation is missing at layer {layer}"
                    )
                self._handles.append(residual.register_hook(self._hook_for(layer)))
                self.hook_registrations += 1
        except Exception:
            self.close()
            raise

    def _hook_for(self, layer: int):
        def inject(grads: Tensor) -> Tensor | None:
            plan = self._active_plans[layer]
            if plan is None:
                return None
            batch_indices, pos_indices, values = plan
            return _inject_exact(
                grads,
                batch_indices=batch_indices,
                pos_indices=pos_indices,
                values=values,
            )

        return inject

    def compute_batch(
        self,
        *,
        layer_ids: Sequence[int],
        positions: Tensor,
        inject_values: Tensor,
        retain_graph: bool = True,
    ) -> Tensor:
        if self._closed:
            raise RuntimeError("Exact VJP hot-loop executor is closed")
        if self._in_compute:
            raise RuntimeError("Exact VJP hot-loop executor is not reentrant")
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

        timing = None
        if inject_values.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(torch.cuda.current_stream(inject_values.device))
            timing = (start_event, end_event)

        buffer = self._row_buffer(batch_size=batch_size, values=inject_values)
        self.context._batch_buffer = buffer
        batch_indices = self._all_batch_indices(
            batch_size=batch_size, device=positions.device
        )
        grouped_offsets: dict[int, list[int]] = {}
        for offset, layer in enumerate(layer_ids):
            grouped_offsets.setdefault(int(layer), []).append(offset)

        self._in_compute = True
        try:
            for layer, layer_offsets in grouped_offsets.items():
                offsets = torch.tensor(
                    layer_offsets, dtype=torch.long, device=positions.device
                )
                self._active_plans[layer] = (
                    batch_indices.index_select(0, offsets),
                    positions.index_select(0, offsets),
                    inject_values.index_select(0, offsets),
                )
                self.plan_builds += 1

            last_layer = max(grouped_offsets)
            last_residual = self.context._resid_activations[last_layer]
            if last_residual is None:
                raise RuntimeError("Last residual activation is missing")
            last_residual.backward(
                gradient=self._backward_seed(last_residual),
                retain_graph=retain_graph,
            )
        finally:
            for layer in grouped_offsets:
                self._active_plans[layer] = None
            self.context._batch_buffer = None
            self._in_compute = False

        if timing is not None:
            timing[1].record(torch.cuda.current_stream(inject_values.device))
            self._timing_events.append(timing)
        groups = len(grouped_offsets)
        self.calls += 1
        self.rows += row_count
        self.layer_groups += groups
        self.avoided_layer_syncs += 1
        self.avoided_hook_registrations += groups
        self.avoided_hook_removals += groups
        return buffer.T[:row_count]

    def close(self) -> None:
        if self._closed:
            return
        if self._in_compute:
            raise RuntimeError("Cannot close exact VJP executor during a VJP")
        for handle in self._handles:
            handle.remove()
            self.hook_removals += 1
        self._handles.clear()
        self._closed = True

    def stats(self) -> ExactVJPHotLoopStats:
        cuda_milliseconds = 0.0
        for start_event, end_event in self._timing_events:
            end_event.synchronize()
            cuda_milliseconds += float(start_event.elapsed_time(end_event))
        return ExactVJPHotLoopStats(
            identity=EXACT_VJP_HOTLOOP_EXECUTOR,
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
            hook_registrations=self.hook_registrations,
            hook_removals=self.hook_removals,
            avoided_hook_registrations=self.avoided_hook_registrations,
            avoided_hook_removals=self.avoided_hook_removals,
            plan_builds=self.plan_builds,
            fallbacks=self.fallbacks,
        )

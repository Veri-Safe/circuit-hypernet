from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from s3circuit.graph_extract import ensure_codecircuit_on_path

from .batching import (
    FIXED_CAP_POLICY,
    AttributionBatchingPlan,
    resolve_attribution_batching,
)
from .exact_vjp import ExactVJPExecutor
from .exact_vjp_hotloop import ExactVJPHotLoopExecutor
from .span_objective import SpanObjective, build_span_objective, causal_replay_token_ids
from .streaming import (
    CachedFeatureSelection,
    ROOT_TARGET,
    StreamingPrunedGraph,
    materialize_cached_exact_graph,
    materialize_streaming_exact_graph,
    select_features_cached_exact,
    select_features_streaming_exact,
)


REFERENCE_BACKEND = "codecircuit_autograd_reference_v1"
STREAMING_BACKEND = "streaming_exact_v1"
CACHED_STREAMING_BACKEND = "streaming_cached_exact_v1"
CACHED_STREAMING_OPTIMIZED_BACKEND = "streaming_cached_exact_v2"
CACHED_STREAMING_PIPELINE_BACKEND = "streaming_cached_pipeline_exact_v1"
CACHED_STREAMING_HOTLOOP_BACKEND = "streaming_cached_pipeline_hotloop_exact_v1"
CACHED_STREAMING_DUAL_EVENT_BACKEND = "streaming_cached_dual_event_exact_v1"
CACHED_STREAMING_VJP_BACKEND = "streaming_cached_pipeline_vjp_exact_v1"
CACHED_STREAMING_DUAL_VJP_BACKEND = "streaming_cached_dual_event_vjp_exact_v1"
CACHED_STREAMING_VJP_HOTLOOP_V2_BACKEND = (
    "streaming_cached_pipeline_vjp_hotloop_exact_v2"
)
CACHED_STREAMING_DUAL_VJP_HOTLOOP_V2_BACKEND = (
    "streaming_cached_dual_event_vjp_hotloop_exact_v2"
)
CODECIRCUIT_ROW_EXECUTOR = "codecircuit"
EXACT_VJP_ROW_EXECUTOR = "exact_vjp"
HOTLOOP_ROW_EXECUTOR = "exact_vjp_hotloop"
ROW_EXECUTORS = frozenset(
    {CODECIRCUIT_ROW_EXECUTOR, EXACT_VJP_ROW_EXECUTOR, HOTLOOP_ROW_EXECUTOR}
)
DUAL_EVENT_BACKENDS = frozenset(
    {
        CACHED_STREAMING_DUAL_EVENT_BACKEND,
        CACHED_STREAMING_DUAL_VJP_BACKEND,
        CACHED_STREAMING_DUAL_VJP_HOTLOOP_V2_BACKEND,
    }
)
PIPELINE_BACKENDS = frozenset(
    {
        CACHED_STREAMING_PIPELINE_BACKEND,
        CACHED_STREAMING_HOTLOOP_BACKEND,
        CACHED_STREAMING_DUAL_EVENT_BACKEND,
        CACHED_STREAMING_VJP_BACKEND,
        CACHED_STREAMING_DUAL_VJP_BACKEND,
        CACHED_STREAMING_VJP_HOTLOOP_V2_BACKEND,
        CACHED_STREAMING_DUAL_VJP_HOTLOOP_V2_BACKEND,
    }
)


@dataclass(frozen=True)
class DenseAttribution:
    replay_token_ids: Tensor
    active_features: Tensor
    activation_values: Tensor
    selected_features: Tensor
    signed_direct_effect: Tensor
    objective: SpanObjective
    backend: str = REFERENCE_BACKEND


@dataclass(frozen=True)
class StreamingAttribution:
    replay_token_ids: Tensor
    active_features: Tensor
    activation_values: Tensor
    graph: StreamingPrunedGraph
    objective: SpanObjective
    batching: AttributionBatchingPlan | None = None
    row_cache_estimated_bytes: int = 0
    row_cache_resident_bytes: int = 0
    unique_cached_rows: int = 0
    cached_row_reads: int = 0
    cached_full_source_sum_reads: int = 0
    cached_submatrix_read_elements: int = 0
    provider_row_compute_seconds: float = 0.0
    row_cache_write_seconds: float = 0.0
    row_cache_read_seconds: float = 0.0
    feature_ranking_seconds: float = 0.0
    graph_materialization_seconds: float = 0.0
    async_d2h_enabled: bool = False
    async_d2h_blocks: int = 0
    async_d2h_bytes: int = 0
    exact_vjp_metrics: dict[str, Any] | None = None
    backend: str = STREAMING_BACKEND


@dataclass(frozen=True)
class CachedAttributionPreparation:
    replay_token_ids: Tensor
    active_features: Tensor
    activation_values: Tensor
    objective: SpanObjective
    selection: CachedFeatureSelection
    feature_count: int
    source_count: int
    error_count: int
    token_count: int
    row_batch_size: int
    node_threshold: float
    edge_threshold: float
    max_node_selected_dense_bytes: int | None
    batching: AttributionBatchingPlan
    backend: str
    async_d2h_enabled: bool = False
    async_d2h_blocks: int = 0
    async_d2h_bytes: int = 0
    exact_vjp_metrics: dict[str, Any] | None = None


@dataclass(frozen=True)
class _CachedProviderShape:
    feature_count: int
    source_count: int
    row_batch_size: int


class CodeCircuitRowProvider:
    """Recompute exact CodeCircuit rows while keeping only one requested block."""

    def __init__(
        self,
        model: Any,
        *,
        condition_token_ids: Sequence[int],
        target_token_ids: Sequence[int],
        logical_row_batch_size: int,
        compute_microbatch_size: int,
        async_cpu_copy: bool = False,
        single_allocation_host_copy: bool = False,
        optimized_exact_vjp: bool = False,
        optimized_exact_vjp_hotloop: bool = False,
        gpu_lock: Any | None = None,
    ) -> None:
        if min(logical_row_batch_size, compute_microbatch_size) <= 0:
            raise ValueError("Attribution row batch sizes must be positive")
        if compute_microbatch_size > logical_row_batch_size:
            raise ValueError("Compute microbatch cannot exceed logical row batch")
        replay_ids = causal_replay_token_ids(condition_token_ids, target_token_ids)
        self.model = model
        self.device = torch.device(model.cfg.device)
        self.logical_row_batch_size = int(logical_row_batch_size)
        # Streaming uses this public attribute to preserve the original logical
        # row blocking. GPU execution is split independently inside rows().
        self.row_batch_size = self.logical_row_batch_size
        self.compute_microbatch_size = int(compute_microbatch_size)
        self.async_cpu_copy = bool(async_cpu_copy)
        self.single_allocation_host_copy = bool(single_allocation_host_copy)
        self.optimized_exact_vjp = bool(optimized_exact_vjp)
        self.optimized_exact_vjp_hotloop = bool(optimized_exact_vjp_hotloop)
        if self.optimized_exact_vjp and self.optimized_exact_vjp_hotloop:
            raise ValueError("Only one exact VJP executor may be selected")
        self.gpu_lock = gpu_lock
        self.async_d2h_blocks = 0
        self.async_d2h_bytes = 0
        with gpu_lock if gpu_lock is not None else nullcontext():
            self.replay_token_ids = torch.tensor(
                replay_ids, dtype=torch.long, device=self.device
            )
            self.context = model.setup_attribution(self.replay_token_ids)
            self.activation_matrix = self.context.activation_matrix.coalesce()
            self.active_features = self.activation_matrix.indices().T.detach().cpu()
            self.feature_layers_cpu = (
                self.active_features[:, 0].long()
                if self.optimized_exact_vjp or self.optimized_exact_vjp_hotloop
                else None
            )
            self.activation_values = (
                self.activation_matrix.values().detach().float().cpu()
            )
            self.feature_count = int(self.activation_matrix._nnz())
            self.source_count = int(self.context._row_size)
            n_layers, n_positions, _ = self.activation_matrix.shape
            if n_layers != 36 or n_positions != len(replay_ids):
                raise ValueError("CodeCircuit row-provider activation shape changed")
            self.n_layers = int(n_layers)
            self.n_positions = int(n_positions)
            self.feature_layers, self.feature_positions, _ = (
                self.activation_matrix.indices()
            )
            self.objective = build_span_objective(
                self.context.logits[0],
                model.unembed.W_U,
                condition_length=len(condition_token_ids),
                target_token_ids=target_token_ids,
            )
            with self.context.install_hooks(model):
                expanded = self.replay_token_ids.expand(
                    self.compute_microbatch_size, -1
                )
                residual = model.forward(expanded, stop_at_layer=n_layers)
                self.context._resid_activations[-1] = model.ln_final(residual)
            if self.optimized_exact_vjp_hotloop:
                self.exact_vjp_executor = ExactVJPHotLoopExecutor(self.context)
            elif self.optimized_exact_vjp:
                self.exact_vjp_executor = ExactVJPExecutor(self.context)
            else:
                self.exact_vjp_executor = None

    def rows(self, target_feature_indices: Sequence[int]) -> Tensor:
        with self.gpu_lock if self.gpu_lock is not None else nullcontext():
            return self._rows_unlocked(target_feature_indices)

    def root_row_for_seed(self, replay_seed: Tensor) -> Tensor:
        """Compute a canonical root row for one explicit replay-position seed."""

        with self.gpu_lock if self.gpu_lock is not None else nullcontext():
            return (
                compute_joint_root_row(
                    self.context,
                    replay_seed=replay_seed,
                    retain_graph=True,
                )
                .detach()
                .float()
                .cpu()
            )

    def _rows_unlocked(self, target_feature_indices: Sequence[int]) -> Tensor:
        targets = [int(item) for item in target_feature_indices]
        async_cpu_copy = bool(getattr(self, "async_cpu_copy", False))
        if not targets:
            return torch.empty((0, self.source_count), dtype=torch.float32)
        if targets.count(ROOT_TARGET) > 1:
            raise ValueError("A row request may contain the virtual root only once")
        invalid = [
            target
            for target in targets
            if target != ROOT_TARGET and not 0 <= target < self.feature_count
        ]
        if invalid:
            raise ValueError(
                f"Feature-row request is outside the activation table: {invalid}"
            )
        result = torch.empty(
            (len(targets), self.source_count),
            dtype=torch.float32,
            device="cpu",
            pin_memory=async_cpu_copy,
        )
        root_positions = [
            index for index, target in enumerate(targets) if target == ROOT_TARGET
        ]
        if root_positions:
            result[root_positions[0]] = (
                compute_joint_root_row(
                    self.context,
                    replay_seed=self.objective.replay_seed,
                    retain_graph=True,
                )
                .detach()
                .float()
                .cpu()
            )
        feature_requests = [
            (index, target)
            for index, target in enumerate(targets)
            if target != ROOT_TARGET
        ]
        pending_copies: list[tuple[torch.cuda.Event, Tensor]] = []
        copy_stream = (
            torch.cuda.Stream(device=self.device) if async_cpu_copy else None
        )
        for start in range(0, len(feature_requests), self.compute_microbatch_size):
            request = feature_requests[start : start + self.compute_microbatch_size]
            output_indices = [item[0] for item in request]
            requested_features = [item[1] for item in request]
            if self.exact_vjp_executor is None:
                feature_indices = torch.tensor(
                    requested_features, dtype=torch.long, device=self.device
                )
                rows = self.context.compute_batch(
                    layers=self.feature_layers.index_select(0, feature_indices),
                    positions=self.feature_positions.index_select(
                        0, feature_indices
                    ),
                    inject_values=self.context.encoder_vecs.index_select(
                        0, feature_indices
                    ),
                    retain_graph=True,
                )
            else:
                if self.feature_layers_cpu is None:
                    raise RuntimeError("Exact VJP CPU layer metadata is missing")
                feature_indices_cpu = torch.tensor(
                    requested_features, dtype=torch.long
                )
                feature_indices = feature_indices_cpu.to(device=self.device)
                rows = self.exact_vjp_executor.compute_batch(
                    layer_ids=self.feature_layers_cpu.index_select(
                        0, feature_indices_cpu
                    ).tolist(),
                    positions=self.feature_positions.index_select(
                        0, feature_indices
                    ),
                    inject_values=self.context.encoder_vecs.index_select(
                        0, feature_indices
                    ),
                    retain_graph=True,
                )
            # CodeCircuit may reuse its row buffer across compute_batch calls.
            # Async D2H must copy from storage that remains stable while the
            # next microbatch is already computing on the default stream.
            float_rows = (
                rows.detach().to(dtype=torch.float32, copy=True)
                if self.single_allocation_host_copy
                else rows.detach().float().clone()
            )
            if copy_stream is None:
                result[torch.tensor(output_indices, dtype=torch.long)] = float_rows.cpu()
                continue
            contiguous = output_indices == list(
                range(output_indices[0], output_indices[0] + len(output_indices))
            )
            if not contiguous:
                raise ValueError("Async attribution row destinations must be contiguous")
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(copy_stream):
                copy_stream.wait_event(ready)
                result[output_indices[0] : output_indices[-1] + 1].copy_(
                    float_rows, non_blocking=True
                )
                copied = torch.cuda.Event()
                copied.record(copy_stream)
            pending_copies.append((copied, float_rows))
            self.async_d2h_blocks += 1
            self.async_d2h_bytes += int(float_rows.numel() * float_rows.element_size())
            if len(pending_copies) > 1:
                completed, _ = pending_copies.pop(0)
                completed.synchronize()
        for completed, _ in pending_copies:
            completed.synchronize()
        return result

    def exact_vjp_runtime_metrics(self) -> dict[str, Any] | None:
        if self.exact_vjp_executor is None:
            return None
        return self.exact_vjp_executor.stats().as_metrics()

    def close_exact_vjp_executor(self) -> None:
        if self.exact_vjp_executor is None:
            return
        close = getattr(self.exact_vjp_executor, "close", None)
        if close is not None:
            close()


def build_code_circuit_row_provider(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    logical_row_batch_size: int,
    compute_microbatch_size: int,
    executor: str,
    async_cpu_copy: bool = False,
    single_allocation_host_copy: bool = False,
    gpu_lock: Any | None = None,
) -> CodeCircuitRowProvider:
    """Build every reference and production provider through one locked path."""

    if executor not in ROW_EXECUTORS:
        raise ValueError(f"Unsupported attribution row executor: {executor}")
    return CodeCircuitRowProvider(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        logical_row_batch_size=logical_row_batch_size,
        compute_microbatch_size=compute_microbatch_size,
        async_cpu_copy=async_cpu_copy,
        single_allocation_host_copy=single_allocation_host_copy,
        optimized_exact_vjp=executor == EXACT_VJP_ROW_EXECUTOR,
        optimized_exact_vjp_hotloop=executor == HOTLOOP_ROW_EXECUTOR,
        gpu_lock=gpu_lock,
    )


def attribute_joint_span_streaming_exact(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str = FIXED_CAP_POLICY,
    compute_microbatch_cap: int | None = None,
    max_feature_nodes: int,
    update_interval: int,
    node_threshold: float,
    edge_threshold: float,
    max_node_selected_dense_bytes: int | None = None,
) -> StreamingAttribution:
    """Production attribution with exact row recomputation and compact materialization."""

    ensure_codecircuit_on_path()
    replay_length = len(causal_replay_token_ids(condition_token_ids, target_token_ids))
    batching = resolve_attribution_batching(
        replay_length=replay_length,
        logical_row_batch_size=row_batch_size,
        update_interval=update_interval,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=(
            row_batch_size if compute_microbatch_cap is None else compute_microbatch_cap
        ),
    )
    provider = CodeCircuitRowProvider(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        logical_row_batch_size=batching.logical_row_batch_size,
        compute_microbatch_size=batching.compute_microbatch_size,
    )
    selected = select_features_streaming_exact(
        provider,
        feature_budget=max_feature_nodes,
        queue_size=batching.logical_queue_size,
    )
    graph = materialize_streaming_exact_graph(
        provider,
        selected_features=selected,
        error_count=provider.n_layers * provider.n_positions,
        token_count=provider.n_positions,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_dense_bytes=max_node_selected_dense_bytes,
    )
    return StreamingAttribution(
        replay_token_ids=provider.replay_token_ids.detach().cpu(),
        active_features=provider.active_features,
        activation_values=provider.activation_values,
        graph=graph,
        objective=provider.objective,
        batching=batching,
        backend=STREAMING_BACKEND,
    )


def attribute_joint_span_cached_exact(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str = FIXED_CAP_POLICY,
    compute_microbatch_cap: int | None = None,
    max_feature_nodes: int,
    update_interval: int,
    node_threshold: float,
    edge_threshold: float,
    max_node_selected_dense_bytes: int | None = None,
    max_exact_row_cache_bytes: int | None = None,
) -> StreamingAttribution:
    """Exact attribution that host-caches each selected signed row once."""

    return _attribute_joint_span_cached_exact(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        row_batch_size=row_batch_size,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=compute_microbatch_cap,
        max_feature_nodes=max_feature_nodes,
        update_interval=update_interval,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_node_selected_dense_bytes=max_node_selected_dense_bytes,
        max_exact_row_cache_bytes=max_exact_row_cache_bytes,
        cache_full_source_sums=False,
        backend=CACHED_STREAMING_BACKEND,
    )


def attribute_joint_span_cached_exact_v2(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str = FIXED_CAP_POLICY,
    compute_microbatch_cap: int | None = None,
    max_feature_nodes: int,
    update_interval: int,
    node_threshold: float,
    edge_threshold: float,
    max_node_selected_dense_bytes: int | None = None,
    max_exact_row_cache_bytes: int | None = None,
) -> StreamingAttribution:
    """Exact cached attribution with immutable full-source row sums."""

    return _attribute_joint_span_cached_exact(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        row_batch_size=row_batch_size,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=compute_microbatch_cap,
        max_feature_nodes=max_feature_nodes,
        update_interval=update_interval,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_node_selected_dense_bytes=max_node_selected_dense_bytes,
        max_exact_row_cache_bytes=max_exact_row_cache_bytes,
        cache_full_source_sums=True,
        backend=CACHED_STREAMING_OPTIMIZED_BACKEND,
    )


def prepare_joint_span_cached_pipeline_exact(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str = FIXED_CAP_POLICY,
    compute_microbatch_cap: int | None = None,
    max_feature_nodes: int,
    update_interval: int,
    node_threshold: float,
    edge_threshold: float,
    max_node_selected_dense_bytes: int | None = None,
    max_exact_row_cache_bytes: int | None = None,
) -> CachedAttributionPreparation:
    """Finish GPU rows and return a CPU-only exact materialization state."""

    return _prepare_joint_span_cached_exact(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        row_batch_size=row_batch_size,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=compute_microbatch_cap,
        max_feature_nodes=max_feature_nodes,
        update_interval=update_interval,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_node_selected_dense_bytes=max_node_selected_dense_bytes,
        max_exact_row_cache_bytes=max_exact_row_cache_bytes,
        cache_full_source_sums=True,
        async_cpu_copy=True,
        single_allocation_host_copy=False,
        gpu_lock=None,
        backend=CACHED_STREAMING_PIPELINE_BACKEND,
    )


def prepare_joint_span_cached_hotloop_exact(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str = FIXED_CAP_POLICY,
    compute_microbatch_cap: int | None = None,
    max_feature_nodes: int,
    update_interval: int,
    node_threshold: float,
    edge_threshold: float,
    max_node_selected_dense_bytes: int | None = None,
    max_exact_row_cache_bytes: int | None = None,
    gpu_lock: Any | None = None,
    backend: str = CACHED_STREAMING_HOTLOOP_BACKEND,
) -> CachedAttributionPreparation:
    """Prepare exact rows with one stable float32 allocation per async D2H block."""

    if backend not in {
        CACHED_STREAMING_HOTLOOP_BACKEND,
        CACHED_STREAMING_DUAL_EVENT_BACKEND,
        CACHED_STREAMING_VJP_BACKEND,
        CACHED_STREAMING_DUAL_VJP_BACKEND,
        CACHED_STREAMING_VJP_HOTLOOP_V2_BACKEND,
        CACHED_STREAMING_DUAL_VJP_HOTLOOP_V2_BACKEND,
    }:
        raise ValueError(f"Unsupported hot-loop backend: {backend}")
    return _prepare_joint_span_cached_exact(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        row_batch_size=row_batch_size,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=compute_microbatch_cap,
        max_feature_nodes=max_feature_nodes,
        update_interval=update_interval,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_node_selected_dense_bytes=max_node_selected_dense_bytes,
        max_exact_row_cache_bytes=max_exact_row_cache_bytes,
        cache_full_source_sums=True,
        async_cpu_copy=True,
        single_allocation_host_copy=True,
        optimized_exact_vjp=backend
        in {CACHED_STREAMING_VJP_BACKEND, CACHED_STREAMING_DUAL_VJP_BACKEND},
        optimized_exact_vjp_hotloop=backend
        in {
            CACHED_STREAMING_VJP_HOTLOOP_V2_BACKEND,
            CACHED_STREAMING_DUAL_VJP_HOTLOOP_V2_BACKEND,
        },
        gpu_lock=gpu_lock,
        backend=backend,
    )


def attribute_joint_span_cached_pipeline_exact(
    model: Any,
    **kwargs: Any,
) -> StreamingAttribution:
    """Sequential compatibility path for the independently pipelined backend."""

    return finalize_joint_span_cached_exact(
        prepare_joint_span_cached_pipeline_exact(model, **kwargs)
    )


def _attribute_joint_span_cached_exact(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str,
    compute_microbatch_cap: int | None,
    max_feature_nodes: int,
    update_interval: int,
    node_threshold: float,
    edge_threshold: float,
    max_node_selected_dense_bytes: int | None,
    max_exact_row_cache_bytes: int | None,
    cache_full_source_sums: bool,
    backend: str,
) -> StreamingAttribution:
    ensure_codecircuit_on_path()
    replay_length = len(causal_replay_token_ids(condition_token_ids, target_token_ids))
    batching = resolve_attribution_batching(
        replay_length=replay_length,
        logical_row_batch_size=row_batch_size,
        update_interval=update_interval,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=(
            row_batch_size if compute_microbatch_cap is None else compute_microbatch_cap
        ),
    )
    provider = CodeCircuitRowProvider(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        logical_row_batch_size=batching.logical_row_batch_size,
        compute_microbatch_size=batching.compute_microbatch_size,
    )
    selection = select_features_cached_exact(
        provider,
        feature_budget=max_feature_nodes,
        queue_size=batching.logical_queue_size,
        max_cache_bytes=max_exact_row_cache_bytes,
        cache_full_source_sums=cache_full_source_sums,
    )
    graph = materialize_cached_exact_graph(
        provider,
        selection=selection,
        error_count=provider.n_layers * provider.n_positions,
        token_count=provider.n_positions,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_dense_bytes=max_node_selected_dense_bytes,
    )
    cache = selection.cache
    return StreamingAttribution(
        replay_token_ids=provider.replay_token_ids.detach().cpu(),
        active_features=provider.active_features,
        activation_values=provider.activation_values,
        graph=graph,
        objective=provider.objective,
        batching=batching,
        row_cache_estimated_bytes=cache.estimated_bytes,
        row_cache_resident_bytes=cache.resident_bytes,
        unique_cached_rows=cache.cached_rows,
        cached_row_reads=cache.read_rows,
        cached_full_source_sum_reads=cache.full_source_sum_reads,
        cached_submatrix_read_elements=cache.submatrix_read_elements,
        provider_row_compute_seconds=cache.provider_compute_seconds,
        row_cache_write_seconds=cache.cache_write_seconds,
        row_cache_read_seconds=cache.cache_read_seconds,
        feature_ranking_seconds=cache.ranking_seconds,
        graph_materialization_seconds=cache.materialization_seconds,
        backend=backend,
    )


def _prepare_joint_span_cached_exact(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str,
    compute_microbatch_cap: int | None,
    max_feature_nodes: int,
    update_interval: int,
    node_threshold: float,
    edge_threshold: float,
    max_node_selected_dense_bytes: int | None,
    max_exact_row_cache_bytes: int | None,
    cache_full_source_sums: bool,
    async_cpu_copy: bool,
    backend: str,
    single_allocation_host_copy: bool = False,
    optimized_exact_vjp: bool = False,
    optimized_exact_vjp_hotloop: bool = False,
    gpu_lock: Any | None = None,
) -> CachedAttributionPreparation:
    ensure_codecircuit_on_path()
    replay_length = len(causal_replay_token_ids(condition_token_ids, target_token_ids))
    batching = resolve_attribution_batching(
        replay_length=replay_length,
        logical_row_batch_size=row_batch_size,
        update_interval=update_interval,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=(
            row_batch_size if compute_microbatch_cap is None else compute_microbatch_cap
        ),
    )
    executor = CODECIRCUIT_ROW_EXECUTOR
    if optimized_exact_vjp:
        executor = EXACT_VJP_ROW_EXECUTOR
    elif optimized_exact_vjp_hotloop:
        executor = HOTLOOP_ROW_EXECUTOR
    provider = build_code_circuit_row_provider(
        model,
        condition_token_ids=condition_token_ids,
        target_token_ids=target_token_ids,
        logical_row_batch_size=batching.logical_row_batch_size,
        compute_microbatch_size=batching.compute_microbatch_size,
        async_cpu_copy=async_cpu_copy,
        single_allocation_host_copy=single_allocation_host_copy,
        executor=executor,
        gpu_lock=gpu_lock,
    )
    try:
        selection = select_features_cached_exact(
            provider,
            feature_budget=max_feature_nodes,
            queue_size=batching.logical_queue_size,
            max_cache_bytes=max_exact_row_cache_bytes,
            cache_full_source_sums=cache_full_source_sums,
        )
    finally:
        provider.close_exact_vjp_executor()
    exact_vjp_metrics = provider.exact_vjp_runtime_metrics()
    objective = SpanObjective(
        target_token_ids=provider.objective.target_token_ids.detach().cpu(),
        target_positions=provider.objective.target_positions.detach().cpu(),
        target_probabilities=provider.objective.target_probabilities.detach().cpu(),
        target_entropies=provider.objective.target_entropies.detach().cpu(),
        target_log_probabilities=(
            provider.objective.target_log_probabilities.detach().cpu()
        ),
        residual_directions=provider.objective.residual_directions.detach().cpu(),
        replay_seed=provider.objective.replay_seed.detach().cpu(),
    )
    return CachedAttributionPreparation(
        replay_token_ids=provider.replay_token_ids.detach().cpu(),
        active_features=provider.active_features,
        activation_values=provider.activation_values,
        objective=objective,
        selection=selection,
        feature_count=provider.feature_count,
        source_count=provider.source_count,
        error_count=provider.n_layers * provider.n_positions,
        token_count=provider.n_positions,
        row_batch_size=provider.row_batch_size,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_node_selected_dense_bytes=max_node_selected_dense_bytes,
        batching=batching,
        backend=backend,
        async_d2h_enabled=provider.async_cpu_copy,
        async_d2h_blocks=provider.async_d2h_blocks,
        async_d2h_bytes=provider.async_d2h_bytes,
        exact_vjp_metrics=exact_vjp_metrics,
    )


def finalize_joint_span_cached_exact(
    preparation: CachedAttributionPreparation,
) -> StreamingAttribution:
    """Materialize and prune a prepared cache without touching CUDA state."""

    provider_shape = _CachedProviderShape(
        feature_count=preparation.feature_count,
        source_count=preparation.source_count,
        row_batch_size=preparation.row_batch_size,
    )
    graph = materialize_cached_exact_graph(
        provider_shape,
        selection=preparation.selection,
        error_count=preparation.error_count,
        token_count=preparation.token_count,
        node_threshold=preparation.node_threshold,
        edge_threshold=preparation.edge_threshold,
        max_dense_bytes=preparation.max_node_selected_dense_bytes,
    )
    cache = preparation.selection.cache
    return StreamingAttribution(
        replay_token_ids=preparation.replay_token_ids,
        active_features=preparation.active_features,
        activation_values=preparation.activation_values,
        graph=graph,
        objective=preparation.objective,
        batching=preparation.batching,
        row_cache_estimated_bytes=cache.estimated_bytes,
        row_cache_resident_bytes=cache.resident_bytes,
        unique_cached_rows=cache.cached_rows,
        cached_row_reads=cache.read_rows,
        cached_full_source_sum_reads=cache.full_source_sum_reads,
        cached_submatrix_read_elements=cache.submatrix_read_elements,
        provider_row_compute_seconds=cache.provider_compute_seconds,
        row_cache_write_seconds=cache.cache_write_seconds,
        row_cache_read_seconds=cache.cache_read_seconds,
        feature_ranking_seconds=cache.ranking_seconds,
        graph_materialization_seconds=cache.materialization_seconds,
        async_d2h_enabled=preparation.async_d2h_enabled,
        async_d2h_blocks=preparation.async_d2h_blocks,
        async_d2h_bytes=preparation.async_d2h_bytes,
        exact_vjp_metrics=preparation.exact_vjp_metrics,
        backend=preparation.backend,
    )


def estimate_dense_reference_bytes(
    *,
    total_active_features: int,
    replay_tokens: int,
    feature_budget: int,
    n_layers: int = 36,
    dtype_bytes: int = 4,
) -> dict[str, int]:
    if (
        min(total_active_features, replay_tokens, feature_budget, n_layers, dtype_bytes)
        <= 0
    ):
        raise ValueError("Capacity-estimate arguments must be positive")
    selected = min(total_active_features, feature_budget)
    source_nodes = total_active_features + (n_layers + 1) * replay_tokens + 1
    working_rows = selected + 1
    compact_nodes = selected + (n_layers + 1) * replay_tokens + 1
    return {
        "total_active_features": int(total_active_features),
        "selected_features": int(selected),
        "source_nodes_before_selection": int(source_nodes),
        "working_rows": int(working_rows),
        "working_edge_bytes": int(working_rows * source_nodes * dtype_bytes),
        "compact_dense_bytes": int(compact_nodes * compact_nodes * dtype_bytes),
    }


def compute_joint_root_row(
    context: Any,
    *,
    replay_seed: Tensor,
    retain_graph: bool,
) -> Tensor:
    """Inject one position-indexed span seed into CodeCircuit's final residual cache."""

    final_residual = context._resid_activations[-1]
    if final_residual is None:
        raise ValueError("Attribution context has no cached final residual")
    if replay_seed.shape != final_residual.shape[1:]:
        raise ValueError(
            f"Span seed shape {tuple(replay_seed.shape)} does not match final residual "
            f"{tuple(final_residual.shape[1:])}"
        )
    batch_size = int(final_residual.shape[0])
    seed = replay_seed.to(device=final_residual.device, dtype=torch.float32)
    context._batch_buffer = torch.zeros(
        context._row_size,
        batch_size,
        device=final_residual.device,
        dtype=seed.dtype,
    )

    def inject(grads: Tensor) -> Tensor:
        result = grads.clone().to(seed.dtype)
        result[0] = seed
        if batch_size > 1:
            result[1:] = 0
        return result.to(grads.dtype)

    handle = final_residual.register_hook(inject)
    try:
        final_residual.backward(
            gradient=torch.zeros_like(final_residual),
            retain_graph=retain_graph,
        )
    finally:
        handle.remove()
    buffer, context._batch_buffer = context._batch_buffer, None
    if buffer is None:
        raise RuntimeError("Joint root attribution buffer was not populated")
    return buffer[:, 0]


def attribute_joint_span_dense_reference(
    model: Any,
    *,
    condition_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    row_batch_size: int,
    compute_microbatch_policy: str = FIXED_CAP_POLICY,
    compute_microbatch_cap: int | None = None,
    max_feature_nodes: int,
    update_interval: int = 4,
) -> DenseAttribution:
    """CodeCircuit-compatible dense audit backend with one joint span root."""

    ensure_codecircuit_on_path()
    from circuit_tracer.attribution.attribute import compute_partial_influences

    if row_batch_size <= 0 or max_feature_nodes <= 0 or update_interval <= 0:
        raise ValueError(
            "Attribution batch, feature, and update limits must be positive"
        )
    replay_ids = causal_replay_token_ids(condition_token_ids, target_token_ids)
    batching = resolve_attribution_batching(
        replay_length=len(replay_ids),
        logical_row_batch_size=row_batch_size,
        update_interval=update_interval,
        compute_microbatch_policy=compute_microbatch_policy,
        compute_microbatch_cap=(
            row_batch_size if compute_microbatch_cap is None else compute_microbatch_cap
        ),
    )
    device = torch.device(model.cfg.device)
    input_ids = torch.tensor(replay_ids, dtype=torch.long, device=device)
    context = model.setup_attribution(input_ids)
    activation_matrix = context.activation_matrix.coalesce()
    total_active = int(activation_matrix._nnz())
    selected_limit = min(max_feature_nodes, total_active)
    if selected_limit <= 0:
        raise RuntimeError(
            "No active transcoder features are available for attribution"
        )
    n_layers, n_pos, _ = activation_matrix.shape
    if n_layers != 36 or n_pos != len(replay_ids):
        raise ValueError("Attribution activation shape changed")
    objective = build_span_objective(
        context.logits[0],
        model.unembed.W_U,
        condition_length=len(condition_token_ids),
        target_token_ids=target_token_ids,
    )
    base_node_count = total_active + (n_layers + 1) * n_pos
    root_global_index = base_node_count
    working = torch.zeros(
        (selected_limit + 1, base_node_count + 1), dtype=torch.float32, device="cpu"
    )
    row_to_node = torch.full((selected_limit + 1,), -1, dtype=torch.long)

    with context.install_hooks(model):
        expanded = input_ids.expand(batching.compute_microbatch_size, -1)
        residual = model.forward(expanded, stop_at_layer=n_layers)
        context._resid_activations[-1] = model.ln_final(residual)
        root_row = compute_joint_root_row(
            context,
            replay_seed=objective.replay_seed,
            retain_graph=True,
        )
        working[0, :base_node_count] = root_row.detach().float().cpu()
        row_to_node[0] = root_global_index

        feature_layers, feature_positions, _ = activation_matrix.indices()
        visited = torch.zeros(total_active, dtype=torch.bool)
        written = 1
        visited_count = 0
        while visited_count < selected_limit:
            if selected_limit == total_active:
                pending = torch.where(~visited)[0]
            else:
                influences = compute_partial_influences(
                    working[:written],
                    torch.ones(1, dtype=torch.float32),
                    row_to_node[:written].to(device),
                    device=device,
                )
                rank = torch.argsort(influences[:total_active], descending=True).cpu()
                queue_size = min(
                    batching.logical_queue_size,
                    selected_limit - visited_count,
                )
                pending = rank[~visited[rank]][:queue_size]
            if pending.numel() == 0:
                raise RuntimeError("Feature-priority queue became empty before budget")
            for start in range(0, len(pending), batching.compute_microbatch_size):
                feature_indices = pending[
                    start : start + batching.compute_microbatch_size
                ]
                feature_indices_device = feature_indices.to(device)
                batch_length = int(feature_indices.numel())
                visited_count += batch_length
                rows = context.compute_batch(
                    layers=feature_layers.index_select(0, feature_indices_device),
                    positions=feature_positions.index_select(0, feature_indices_device),
                    inject_values=context.encoder_vecs.index_select(
                        0, feature_indices_device
                    ),
                    retain_graph=visited_count < selected_limit,
                )
                working[written : written + batch_length, :base_node_count] = (
                    rows.detach().float().cpu()
                )
                row_to_node[written : written + batch_length] = feature_indices
                visited[feature_indices] = True
                written += batch_length

    selected_features = torch.where(visited)[0]
    nonfeatures = torch.arange(total_active, base_node_count, dtype=torch.long)
    retained_global = torch.cat(
        (selected_features, nonfeatures, torch.tensor([root_global_index]))
    )
    global_to_local = torch.full((base_node_count + 1,), -1, dtype=torch.long)
    global_to_local[retained_global] = torch.arange(retained_global.numel())
    compact = torch.zeros(
        (retained_global.numel(), retained_global.numel()), dtype=torch.float32
    )
    for row_index in range(written):
        target_global = int(row_to_node[row_index])
        target_local = int(global_to_local[target_global])
        if target_local < 0:
            continue
        compact[target_local] = working[row_index].index_select(0, retained_global)
    return DenseAttribution(
        replay_token_ids=input_ids.detach().cpu(),
        active_features=activation_matrix.indices().T.detach().cpu(),
        activation_values=activation_matrix.values().detach().float().cpu(),
        selected_features=selected_features,
        signed_direct_effect=compact,
        objective=objective,
    )

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import time
from typing import Any, Callable, Protocol, Sequence

import torch
from torch import Tensor

from .influence import (
    cumulative_rank_scores,
    propagate_influence,
    prune_edges_after_node_selection,
    threshold_for_cumulative_fraction,
)


ROOT_TARGET = -1

_ROW_CACHE_ADMISSION = ContextVar("exact_row_cache_admission", default=None)


@contextmanager
def exact_row_cache_admission(callback: Callable[[int], None]):
    """Reserve actual host allocation bytes before an exact cache is allocated.

    The caller owns the reservation until CPU finalization releases the cache.
    Context-local scope keeps independent worker threads and nested uses apart.
    """
    token = _ROW_CACHE_ADMISSION.set(callback)
    try:
        yield
    finally:
        _ROW_CACHE_ADMISSION.reset(token)


class DirectEffectRowProvider(Protocol):
    feature_count: int
    source_count: int

    def rows(self, target_feature_indices: Sequence[int]) -> Tensor:
        """Return signed rows over all base sources; -1 denotes the span root."""


@dataclass(frozen=True)
class StreamingPrunedGraph:
    selected_features: Tensor
    retained_features: Tensor
    retained_error_offsets: Tensor
    retained_token_offsets: Tensor
    signed_direct_effect: Tensor
    edge_nonnegative_influence: Tensor
    node_influence: Tensor
    node_cumulative_score: Tensor
    token_start: int
    root_index: int


class ExactEffectRowCache:
    """Keep exact signed rows in host memory for ranking and final edges."""

    def __init__(
        self,
        provider: DirectEffectRowProvider,
        *,
        feature_capacity: int,
        max_cache_bytes: int | None = None,
        cache_full_source_sums: bool = False,
    ) -> None:
        if feature_capacity <= 0:
            raise ValueError("Feature-row cache capacity must be positive")
        self.feature_count = int(provider.feature_count)
        self.source_count = int(provider.source_count)
        self.row_batch_size = max(1, int(getattr(provider, "row_batch_size", 64)))
        self.feature_capacity = min(int(feature_capacity), self.feature_count)
        self.cache_full_source_sums = bool(cache_full_source_sums)
        scalar_cache_bytes = (
            (self.feature_capacity + 1)
            * torch.tensor([], dtype=torch.float32).element_size()
            if self.cache_full_source_sums
            else 0
        )
        self.estimated_bytes = (
            (self.feature_capacity + 1)
            * self.source_count
            * torch.tensor([], dtype=torch.float32).element_size()
            + scalar_cache_bytes
        )
        if max_cache_bytes is not None and self.estimated_bytes > max_cache_bytes:
            raise MemoryError(
                "Exact row cache exceeds the audited host-memory capacity: "
                f"{self.estimated_bytes} > {max_cache_bytes} bytes"
            )
        self.allocation_bytes = (
            self.estimated_bytes
            + self.feature_count * torch.tensor([], dtype=torch.long).element_size()
        )
        admission = _ROW_CACHE_ADMISSION.get()
        if admission is not None:
            admission(self.allocation_bytes)
        try:
            self._rows = torch.empty(
                (self.feature_capacity + 1, self.source_count),
                dtype=torch.float32,
                device="cpu",
            )
        except RuntimeError as error:
            raise MemoryError(
                "Unable to allocate the exact host row cache "
                f"({self.estimated_bytes} bytes)"
            ) from error
        self._feature_slots = torch.full(
            (self.feature_count,), -1, dtype=torch.long
        )
        self._absolute_row_sums = (
            torch.empty(self.feature_capacity + 1, dtype=torch.float32)
            if self.cache_full_source_sums
            else None
        )
        self._root_slot = -1
        self._size = 0
        self._read_rows = 0
        self._full_source_sum_reads = 0
        self._submatrix_read_elements = 0
        self.provider_compute_seconds = 0.0
        self.cache_write_seconds = 0.0
        self.cache_read_seconds = 0.0
        self.ranking_seconds = 0.0
        self.materialization_seconds = 0.0

    @property
    def cached_rows(self) -> int:
        return self._size

    @property
    def resident_bytes(self) -> int:
        row_bytes = (
            self._size
            * self.source_count
            * torch.tensor([], dtype=torch.float32).element_size()
        )
        scalar_bytes = (
            self._size * torch.tensor([], dtype=torch.float32).element_size()
            if self._absolute_row_sums is not None
            else 0
        )
        return row_bytes + scalar_bytes

    @property
    def read_rows(self) -> int:
        return self._read_rows

    @property
    def full_source_sum_reads(self) -> int:
        return self._full_source_sum_reads

    @property
    def submatrix_read_elements(self) -> int:
        return self._submatrix_read_elements

    def append(self, targets: Sequence[int], rows: Tensor) -> None:
        values = [int(target) for target in targets]
        if rows.shape != (len(values), self.source_count):
            raise ValueError("Cached direct-effect row shape changed")
        if self._size + len(values) > self.feature_capacity + 1:
            raise MemoryError("Exact row cache capacity was exceeded")
        for target in values:
            if target == ROOT_TARGET:
                if self._root_slot >= 0:
                    raise ValueError("The virtual-root row is already cached")
            elif not 0 <= target < self.feature_count:
                raise ValueError(f"Cached feature target is invalid: {target}")
            elif int(self._feature_slots[target]) >= 0:
                raise ValueError(f"Feature row {target} is already cached")

        start = self._size
        end = start + len(values)
        started = time.perf_counter()
        self._rows[start:end].copy_(rows.detach().float().cpu())
        if self._absolute_row_sums is not None:
            self._absolute_row_sums[start:end].copy_(
                self._rows[start:end].abs().sum(dim=1)
            )
        self.cache_write_seconds += time.perf_counter() - started
        for offset, target in enumerate(values):
            slot = start + offset
            if target == ROOT_TARGET:
                self._root_slot = slot
            else:
                self._feature_slots[target] = slot
        self._size = end

    def _slots(self, targets: Sequence[int]) -> Tensor:
        slots: list[int] = []
        for item in targets:
            target = int(item)
            slot = (
                self._root_slot
                if target == ROOT_TARGET
                else int(self._feature_slots[target])
                if 0 <= target < self.feature_count
                else -1
            )
            if slot < 0:
                raise KeyError(f"Direct-effect row is not cached: {target}")
            slots.append(slot)
        return torch.tensor(slots, dtype=torch.long)

    def _read(self, targets: Sequence[int], *, absolute: bool) -> Tensor:
        slots = self._slots(targets)
        if slots.numel() == 0:
            return torch.empty((0, self.source_count), dtype=torch.float32)
        self._read_rows += int(slots.numel())
        started = time.perf_counter()
        result = self._rows.index_select(0, slots)
        if absolute:
            result.abs_()
        self.cache_read_seconds += time.perf_counter() - started
        return result

    def absolute_rows(self, targets: Sequence[int]) -> Tensor:
        return self._read(targets, absolute=True)

    def signed_rows(self, targets: Sequence[int]) -> Tensor:
        return self._read(targets, absolute=False)

    def absolute_full_source_sums(self, targets: Sequence[int]) -> Tensor:
        if self._absolute_row_sums is None:
            raise RuntimeError("Full-source row sums were not cached")
        slots = self._slots(targets)
        self._full_source_sum_reads += int(slots.numel())
        started = time.perf_counter()
        result = self._absolute_row_sums.index_select(0, slots)
        self.cache_read_seconds += time.perf_counter() - started
        return result

    def absolute_submatrix(
        self,
        targets: Sequence[int],
        source_indices: Tensor,
    ) -> Tensor:
        slots = self._slots(targets)
        sources = source_indices.detach().cpu().long()
        if slots.numel() == 0 or sources.numel() == 0:
            return torch.empty(
                (int(slots.numel()), int(sources.numel())), dtype=torch.float32
            )
        started = time.perf_counter()
        # Select the narrow source set before target-row reordering so building
        # the propagation core never copies every source column.
        result = (
            self._rows[: self._size]
            .index_select(1, sources)
            .index_select(0, slots)
            .abs_()
        )
        self._submatrix_read_elements += int(result.numel())
        self.cache_read_seconds += time.perf_counter() - started
        return result

    def compute_and_append(
        self, provider: DirectEffectRowProvider, targets: Sequence[int]
    ) -> None:
        started = time.perf_counter()
        rows = provider.rows(targets)
        self.provider_compute_seconds += time.perf_counter() - started
        self.append(targets, rows)

    def selected_features_at_capacity(self, capacity: int) -> Tensor:
        """Recover the exact prefix of the deterministic priority expansion."""

        limit = min(int(capacity), self.feature_count)
        if limit <= 0 or self._size < limit + 1:
            raise ValueError("Requested feature-selection prefix is not cached")
        slots = self._feature_slots
        selected = torch.where((slots > self._root_slot) & (slots <= limit))[0]
        if selected.numel() != limit:
            raise ValueError("Cached feature-selection prefix changed cardinality")
        return selected


@dataclass(frozen=True)
class CachedFeatureSelection:
    selected_features: Tensor
    cache: ExactEffectRowCache


def selected_feature_influence_cached(
    provider: DirectEffectRowProvider,
    *,
    selection: CachedFeatureSelection,
) -> Tensor:
    """Return pre-pruning CodeCircuit influence aligned to selected_features."""

    selected = torch.sort(selection.selected_features.detach().cpu().long()).values
    cache = selection.cache
    targets = [int(item) for item in selected.tolist()] + [ROOT_TARGET]
    denominators, output = _cached_normalized_core_and_output(
        cache,
        targets=targets,
        source_indices=selected,
        selected_feature_sources=selected,
        epsilon=1e-10,
    )
    influence = torch.zeros(selected.numel(), dtype=torch.float32)
    for start, end, rows in _cached_target_blocks(cache, targets):
        restricted = rows.index_select(1, selected)
        influence += output[start:end] @ (restricted / denominators[start:end, None])
    return influence


def _target_blocks(provider: DirectEffectRowProvider, targets: Sequence[int]):
    block_size = max(1, int(getattr(provider, "row_batch_size", 64)))
    for start in range(0, len(targets), block_size):
        end = min(start + block_size, len(targets))
        rows = provider.rows(targets[start:end]).float().cpu()
        if rows.shape != (end - start, provider.source_count):
            raise ValueError("Direct-effect row provider returned a changed shape")
        yield start, end, rows


def _cached_target_blocks(
    cache: ExactEffectRowCache, targets: Sequence[int]
):
    for start in range(0, len(targets), cache.row_batch_size):
        end = min(start + cache.row_batch_size, len(targets))
        yield start, end, cache.absolute_rows(targets[start:end])


def _normalized_core_and_output(
    provider: DirectEffectRowProvider,
    *,
    targets: Sequence[int],
    source_indices: Tensor,
    selected_feature_sources: Tensor,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    row_count = len(targets)
    core = torch.zeros((row_count, row_count), dtype=torch.float32)
    denominators = torch.empty(row_count, dtype=torch.float32)
    for start, end, rows in _target_blocks(provider, targets):
        restricted = rows.index_select(1, source_indices)
        block_denominators = restricted.abs().sum(dim=1).clamp_min(epsilon)
        denominators[start:end] = block_denominators
        if selected_feature_sources.numel():
            feature_columns = rows.index_select(1, selected_feature_sources).abs()
            core[start:end, : selected_feature_sources.numel()] = (
                feature_columns / block_denominators[:, None]
            )
    root_index = row_count - 1
    root_weights = torch.zeros(row_count, dtype=torch.float32)
    root_weights[root_index] = 1.0
    output_influence = propagate_influence(core, root_weights) + root_weights
    return denominators, output_influence


def _rank_candidate_features(
    provider: DirectEffectRowProvider,
    selected: Tensor,
) -> Tensor:
    targets = [int(item) for item in selected.tolist()] + [ROOT_TARGET]
    all_sources = torch.arange(provider.source_count, dtype=torch.long)
    denominators, output = _normalized_core_and_output(
        provider,
        targets=targets,
        source_indices=all_sources,
        selected_feature_sources=selected,
        epsilon=1e-8,
    )
    scores = torch.zeros(provider.feature_count, dtype=torch.float32)
    for start, end, rows in _target_blocks(provider, targets):
        normalized_features = (
            rows[:, : provider.feature_count].abs() / denominators[start:end, None]
        )
        scores += output[start:end] @ normalized_features
    if selected.numel():
        scores[selected] = float("-inf")
    return torch.argsort(scores, descending=True, stable=True)


def _cached_normalized_core_and_output(
    cache: ExactEffectRowCache,
    *,
    targets: Sequence[int],
    source_indices: Tensor | None,
    selected_feature_sources: Tensor,
    epsilon: float,
    use_cached_full_source_sums: bool = False,
) -> tuple[Tensor, Tensor]:
    row_count = len(targets)
    core = torch.zeros((row_count, row_count), dtype=torch.float32)
    if source_indices is None and use_cached_full_source_sums:
        denominators = cache.absolute_full_source_sums(targets).clamp_min(epsilon)
        if selected_feature_sources.numel():
            feature_columns = cache.absolute_submatrix(
                targets, selected_feature_sources
            )
            core[:, : selected_feature_sources.numel()] = (
                feature_columns / denominators[:, None]
            )
        root_index = row_count - 1
        root_weights = torch.zeros(row_count, dtype=torch.float32)
        root_weights[root_index] = 1.0
        output_influence = propagate_influence(core, root_weights) + root_weights
        return denominators, output_influence

    denominators = torch.empty(row_count, dtype=torch.float32)
    for start, end, absolute_rows in _cached_target_blocks(cache, targets):
        # Ranking retains every source. Avoid materializing an identity
        # index_select of the full cached block in that common path.
        restricted = (
            absolute_rows
            if source_indices is None
            else absolute_rows.index_select(1, source_indices)
        )
        block_denominators = restricted.sum(dim=1).clamp_min(epsilon)
        denominators[start:end] = block_denominators
        if selected_feature_sources.numel():
            feature_columns = absolute_rows.index_select(
                1, selected_feature_sources
            )
            core[start:end, : selected_feature_sources.numel()] = (
                feature_columns / block_denominators[:, None]
            )
    root_index = row_count - 1
    root_weights = torch.zeros(row_count, dtype=torch.float32)
    root_weights[root_index] = 1.0
    output_influence = propagate_influence(core, root_weights) + root_weights
    return denominators, output_influence


def _rank_candidate_features_cached(
    cache: ExactEffectRowCache,
    selected: Tensor,
    *,
    use_cached_full_source_sums: bool = False,
) -> Tensor:
    targets = [int(item) for item in selected.tolist()] + [ROOT_TARGET]
    denominators, output = _cached_normalized_core_and_output(
        cache,
        targets=targets,
        source_indices=None,
        selected_feature_sources=selected,
        epsilon=1e-8,
        use_cached_full_source_sums=use_cached_full_source_sums,
    )
    scores = torch.zeros(cache.feature_count, dtype=torch.float32)
    for start, end, absolute_rows in _cached_target_blocks(cache, targets):
        normalized_features = (
            absolute_rows[:, : cache.feature_count]
            / denominators[start:end, None]
        )
        scores += output[start:end] @ normalized_features
    if selected.numel():
        scores[selected] = float("-inf")
    return torch.argsort(scores, descending=True, stable=True)


def select_features_streaming_exact(
    provider: DirectEffectRowProvider,
    *,
    feature_budget: int,
    queue_size: int,
) -> Tensor:
    """Match CodeCircuit priority expansion without retaining its dense row matrix."""

    if feature_budget <= 0 or queue_size <= 0:
        raise ValueError("Feature budget and queue size must be positive")
    limit = min(int(feature_budget), int(provider.feature_count))
    if limit == provider.feature_count:
        return torch.arange(provider.feature_count, dtype=torch.long)
    selected = torch.empty(0, dtype=torch.long)
    selected_mask = torch.zeros(provider.feature_count, dtype=torch.bool)
    while selected.numel() < limit:
        ranking = _rank_candidate_features(provider, selected)
        pending = ranking[~selected_mask[ranking]][
            : min(queue_size, limit - selected.numel())
        ]
        if pending.numel() == 0:
            raise RuntimeError("Streaming feature-priority queue became empty")
        selected_mask[pending] = True
        selected = torch.where(selected_mask)[0]
    return selected


def select_features_cached_exact(
    provider: DirectEffectRowProvider,
    *,
    feature_budget: int,
    queue_size: int,
    max_cache_bytes: int | None = None,
    cache_full_source_sums: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CachedFeatureSelection:
    """Select the exact CodeCircuit queue while computing every row once."""

    if feature_budget <= 0 or queue_size <= 0:
        raise ValueError("Feature budget and queue size must be positive")
    limit = min(int(feature_budget), int(provider.feature_count))
    cache = ExactEffectRowCache(
        provider,
        feature_capacity=limit,
        max_cache_bytes=max_cache_bytes,
        cache_full_source_sums=cache_full_source_sums,
    )
    cache.compute_and_append(provider, [ROOT_TARGET])
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "root_row",
                "selected_features": 0,
                "cached_rows": cache.cached_rows,
                "feature_budget": limit,
                "provider_compute_seconds": cache.provider_compute_seconds,
            }
        )
    selected = torch.empty(0, dtype=torch.long)
    selected_mask = torch.zeros(provider.feature_count, dtype=torch.bool)

    while selected.numel() < limit:
        if limit == provider.feature_count:
            pending = torch.where(~selected_mask)[0]
        else:
            ranking_started = time.perf_counter()
            ranking = _rank_candidate_features_cached(
                cache,
                selected,
                use_cached_full_source_sums=cache_full_source_sums,
            )
            cache.ranking_seconds += time.perf_counter() - ranking_started
            pending = ranking[~selected_mask[ranking]][
                : min(queue_size, limit - selected.numel())
            ]
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "ranking_complete",
                        "selected_features": int(selected.numel()),
                        "pending_features": int(pending.numel()),
                        "cached_rows": cache.cached_rows,
                        "ranking_seconds": cache.ranking_seconds,
                    }
                )
        if pending.numel() == 0:
            raise RuntimeError("Cached feature-priority queue became empty")
        pending_targets = [int(item) for item in pending.tolist()]
        for start in range(0, len(pending_targets), cache.row_batch_size):
            end = min(start + cache.row_batch_size, len(pending_targets))
            cache.compute_and_append(provider, pending_targets[start:end])
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "row_batch_complete",
                        "selected_features": int(selected.numel()),
                        "cached_rows": cache.cached_rows,
                        "row_batch_size": end - start,
                        "provider_compute_seconds": cache.provider_compute_seconds,
                    }
                )
        selected_mask[pending] = True
        selected = torch.where(selected_mask)[0]
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "selection_round_complete",
                    "selected_features": int(selected.numel()),
                    "cached_rows": cache.cached_rows,
                    "feature_budget": limit,
                    "provider_compute_seconds": cache.provider_compute_seconds,
                    "ranking_seconds": cache.ranking_seconds,
                }
            )

    return CachedFeatureSelection(selected_features=selected, cache=cache)


def materialize_streaming_exact_graph(
    provider: DirectEffectRowProvider,
    *,
    selected_features: Tensor,
    error_count: int,
    token_count: int,
    node_threshold: float,
    edge_threshold: float,
    max_dense_bytes: int | None = None,
) -> StreamingPrunedGraph:
    """Materialize only the node-selected graph and exactly prune its edges."""

    selected = torch.sort(selected_features.detach().cpu().long()).values
    if selected.numel() and (
        int(selected.min()) < 0 or int(selected.max()) >= provider.feature_count
    ):
        raise ValueError("Selected feature lies outside the provider")
    expected_sources = provider.feature_count + error_count + token_count
    if provider.source_count != expected_sources:
        raise ValueError(
            f"Provider sources {provider.source_count} != expected {expected_sources}"
        )
    nonfeatures = torch.arange(
        provider.feature_count, provider.source_count, dtype=torch.long
    )
    retained_sources = torch.cat((selected, nonfeatures))
    targets = [int(item) for item in selected.tolist()] + [ROOT_TARGET]
    denominators, output = _normalized_core_and_output(
        provider,
        targets=targets,
        source_indices=retained_sources,
        selected_feature_sources=selected,
        epsilon=1e-10,
    )
    source_influence = torch.zeros(retained_sources.numel(), dtype=torch.float32)
    for start, end, rows in _target_blocks(provider, targets):
        restricted = rows.index_select(1, retained_sources)
        normalized = restricted.abs() / denominators[start:end, None]
        source_influence += output[start:end] @ normalized
    all_node_influence = torch.cat((source_influence, torch.zeros(1)))
    all_cumulative = cumulative_rank_scores(all_node_influence)

    initial_mask = all_node_influence >= threshold_for_cumulative_fraction(
        all_node_influence, node_threshold
    )
    token_start = int(selected.numel()) + error_count
    token_indices = torch.arange(token_start, token_start + token_count)
    root_index = int(all_node_influence.numel() - 1)
    initial_mask[token_indices] = True
    initial_mask[root_index] = True
    kept = torch.where(initial_mask)[0]
    kept_features = kept[kept < selected.numel()]
    error_start = int(selected.numel())
    kept_errors = kept[(kept >= error_start) & (kept < token_start)] - error_start
    kept_tokens = kept[(kept >= token_start) & (kept < root_index)] - token_start
    if kept_tokens.numel() != token_count:
        raise AssertionError("All logical token nodes must survive node pruning")

    kept_source_globals = torch.cat(
        (
            selected.index_select(0, kept_features),
            provider.feature_count + kept_errors,
            provider.feature_count + error_count + kept_tokens,
        )
    )
    feature_targets = selected.index_select(0, kept_features)
    target_rows = [int(item) for item in feature_targets.tolist()] + [ROOT_TARGET]
    compact_count = int(kept_source_globals.numel()) + 1
    estimated_dense_bytes = (
        compact_count
        * compact_count
        * (
            torch.tensor([], dtype=torch.float32).element_size() * 2
            + torch.tensor([], dtype=torch.bool).element_size()
        )
    )
    if max_dense_bytes is not None and estimated_dense_bytes > max_dense_bytes:
        raise MemoryError(
            "Node-selected exact graph exceeds the audited materialization capacity: "
            f"{estimated_dense_bytes} > {max_dense_bytes} bytes"
        )
    direct = torch.zeros((compact_count, compact_count), dtype=torch.float32)
    for start, end, rows in _target_blocks(provider, target_rows):
        restricted = rows.index_select(1, kept_source_globals)
        for offset, target in enumerate(target_rows[start:end]):
            destination = compact_count - 1 if target == ROOT_TARGET else start + offset
            direct[destination, :-1] = restricted[offset]
    compact_token_start = int(feature_targets.numel() + kept_errors.numel())
    compact_token_indices = torch.arange(
        compact_token_start, compact_token_start + token_count
    )
    secondary_mask, edge_mask, edge_influence = prune_edges_after_node_selection(
        direct,
        token_indices=compact_token_indices,
        root_index=compact_count - 1,
        feature_count=int(feature_targets.numel()),
        edge_threshold=edge_threshold,
    )
    final_indices = torch.where(secondary_mask)[0]
    final_feature_mask = final_indices < feature_targets.numel()
    final_error_start = int(feature_targets.numel())
    final_token_start = final_error_start + int(kept_errors.numel())
    final_features_local = final_indices[final_feature_mask]
    final_errors_local = (
        final_indices[
            (final_indices >= final_error_start) & (final_indices < final_token_start)
        ]
        - final_error_start
    )
    final_tokens_local = (
        final_indices[
            (final_indices >= final_token_start) & (final_indices < compact_count - 1)
        ]
        - final_token_start
    )
    if final_tokens_local.numel() != token_count:
        raise AssertionError("Edge pruning removed a protected token")
    final_direct = direct.index_select(0, final_indices).index_select(1, final_indices)
    final_edge_mask = edge_mask.index_select(0, final_indices).index_select(
        1, final_indices
    )
    final_edge_influence = edge_influence.index_select(0, final_indices).index_select(
        1, final_indices
    )
    final_direct[~final_edge_mask] = 0
    final_edge_influence[~final_edge_mask] = 0
    return StreamingPrunedGraph(
        selected_features=selected,
        retained_features=feature_targets.index_select(0, final_features_local),
        retained_error_offsets=kept_errors.index_select(0, final_errors_local),
        retained_token_offsets=kept_tokens.index_select(0, final_tokens_local),
        signed_direct_effect=final_direct,
        edge_nonnegative_influence=final_edge_influence,
        node_influence=all_node_influence.index_select(0, kept).index_select(
            0, final_indices
        ),
        node_cumulative_score=all_cumulative.index_select(0, kept).index_select(
            0, final_indices
        ),
        token_start=int(final_features_local.numel() + final_errors_local.numel()),
        root_index=int(final_indices.numel() - 1),
    )


def materialize_cached_exact_graph(
    provider: DirectEffectRowProvider,
    *,
    selection: CachedFeatureSelection,
    error_count: int,
    token_count: int,
    node_threshold: float,
    edge_threshold: float,
    max_dense_bytes: int | None = None,
) -> StreamingPrunedGraph:
    """Prune an exact graph using absolute and signed views of cached rows."""

    materialization_started = time.perf_counter()
    selected = torch.sort(
        selection.selected_features.detach().cpu().long()
    ).values
    cache = selection.cache
    if (
        cache.feature_count != provider.feature_count
        or cache.source_count != provider.source_count
    ):
        raise ValueError("Exact row cache belongs to a different provider")
    if selected.numel() and (
        int(selected.min()) < 0 or int(selected.max()) >= provider.feature_count
    ):
        raise ValueError("Selected feature lies outside the provider")
    expected_sources = provider.feature_count + error_count + token_count
    if provider.source_count != expected_sources:
        raise ValueError(
            f"Provider sources {provider.source_count} != expected {expected_sources}"
        )
    nonfeatures = torch.arange(
        provider.feature_count, provider.source_count, dtype=torch.long
    )
    retained_sources = torch.cat((selected, nonfeatures))
    targets = [int(item) for item in selected.tolist()] + [ROOT_TARGET]
    denominators, output = _cached_normalized_core_and_output(
        cache,
        targets=targets,
        source_indices=retained_sources,
        selected_feature_sources=selected,
        epsilon=1e-10,
    )
    source_influence = torch.zeros(retained_sources.numel(), dtype=torch.float32)
    for start, end, absolute_rows in _cached_target_blocks(cache, targets):
        restricted = absolute_rows.index_select(1, retained_sources)
        normalized = restricted / denominators[start:end, None]
        source_influence += output[start:end] @ normalized
    all_node_influence = torch.cat((source_influence, torch.zeros(1)))
    all_cumulative = cumulative_rank_scores(all_node_influence)

    initial_mask = all_node_influence >= threshold_for_cumulative_fraction(
        all_node_influence, node_threshold
    )
    token_start = int(selected.numel()) + error_count
    token_indices = torch.arange(token_start, token_start + token_count)
    root_index = int(all_node_influence.numel() - 1)
    initial_mask[token_indices] = True
    initial_mask[root_index] = True
    kept = torch.where(initial_mask)[0]
    kept_features = kept[kept < selected.numel()]
    error_start = int(selected.numel())
    kept_errors = kept[(kept >= error_start) & (kept < token_start)] - error_start
    kept_tokens = kept[(kept >= token_start) & (kept < root_index)] - token_start
    if kept_tokens.numel() != token_count:
        raise AssertionError("All logical token nodes must survive node pruning")

    kept_source_globals = torch.cat(
        (
            selected.index_select(0, kept_features),
            provider.feature_count + kept_errors,
            provider.feature_count + error_count + kept_tokens,
        )
    )
    feature_targets = selected.index_select(0, kept_features)
    target_rows = [int(item) for item in feature_targets.tolist()] + [ROOT_TARGET]
    compact_count = int(kept_source_globals.numel()) + 1
    estimated_dense_bytes = (
        compact_count
        * compact_count
        * (
            torch.tensor([], dtype=torch.float32).element_size() * 2
            + torch.tensor([], dtype=torch.bool).element_size()
        )
    )
    if max_dense_bytes is not None and estimated_dense_bytes > max_dense_bytes:
        raise MemoryError(
            "Node-selected exact graph exceeds the audited materialization capacity: "
            f"{estimated_dense_bytes} > {max_dense_bytes} bytes"
        )
    direct = torch.zeros((compact_count, compact_count), dtype=torch.float32)
    for start in range(0, len(target_rows), cache.row_batch_size):
        end = min(start + cache.row_batch_size, len(target_rows))
        rows = cache.signed_rows(target_rows[start:end])
        restricted = rows.index_select(1, kept_source_globals)
        for offset, target in enumerate(target_rows[start:end]):
            destination = compact_count - 1 if target == ROOT_TARGET else start + offset
            direct[destination, :-1] = restricted[offset]
    compact_token_start = int(feature_targets.numel() + kept_errors.numel())
    compact_token_indices = torch.arange(
        compact_token_start, compact_token_start + token_count
    )
    secondary_mask, edge_mask, edge_influence = prune_edges_after_node_selection(
        direct,
        token_indices=compact_token_indices,
        root_index=compact_count - 1,
        feature_count=int(feature_targets.numel()),
        edge_threshold=edge_threshold,
    )
    final_indices = torch.where(secondary_mask)[0]
    final_feature_mask = final_indices < feature_targets.numel()
    final_error_start = int(feature_targets.numel())
    final_token_start = final_error_start + int(kept_errors.numel())
    final_features_local = final_indices[final_feature_mask]
    final_errors_local = (
        final_indices[
            (final_indices >= final_error_start) & (final_indices < final_token_start)
        ]
        - final_error_start
    )
    final_tokens_local = (
        final_indices[
            (final_indices >= final_token_start) & (final_indices < compact_count - 1)
        ]
        - final_token_start
    )
    if final_tokens_local.numel() != token_count:
        raise AssertionError("Edge pruning removed a protected token")
    final_direct = direct.index_select(0, final_indices).index_select(1, final_indices)
    final_edge_mask = edge_mask.index_select(0, final_indices).index_select(
        1, final_indices
    )
    final_edge_influence = edge_influence.index_select(0, final_indices).index_select(
        1, final_indices
    )
    final_direct[~final_edge_mask] = 0
    final_edge_influence[~final_edge_mask] = 0
    result = StreamingPrunedGraph(
        selected_features=selected,
        retained_features=feature_targets.index_select(0, final_features_local),
        retained_error_offsets=kept_errors.index_select(0, final_errors_local),
        retained_token_offsets=kept_tokens.index_select(0, final_tokens_local),
        signed_direct_effect=final_direct,
        edge_nonnegative_influence=final_edge_influence,
        node_influence=all_node_influence.index_select(0, kept).index_select(
            0, final_indices
        ),
        node_cumulative_score=all_cumulative.index_select(0, kept).index_select(
            0, final_indices
        ),
        token_start=int(final_features_local.numel() + final_errors_local.numel()),
        root_index=int(final_indices.numel() - 1),
    )
    cache.materialization_seconds += time.perf_counter() - materialization_started
    return result

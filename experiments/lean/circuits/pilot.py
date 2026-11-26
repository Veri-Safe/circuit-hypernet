"""Pilot/audit orchestration for S5 S3-style Circuit Graph extraction."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor

from .attribution import (
    S5DirectEffectRowProvider,
    decomposed_forward,
    feature_intervention_logits,
    linearized_forward,
)
from .payload import S5CompactPayload, build_compact_payload
from .s3_reuse import (
    ROOT_TARGET,
    materialize_cached_exact_graph,
    materialize_streaming_exact_graph,
    select_features_cached_exact,
    select_features_streaming_exact,
    selected_feature_influence_cached,
    propagate_influence,
)
from .traces import CircuitTrace
from .transcoder import RectangularReLUTranscoder


@dataclass(frozen=True)
class ExtractedGraph:
    payload: S5CompactPayload
    selected_features: Tensor
    retained_features: Tensor
    selected_feature_influence: Tensor | None
    backend: str
    seconds: float
    cache_stats: Mapping[str, Any]
    dense_reference_signed_direct_effect: Tensor | None = None
    dense_reference_selected_records: Tensor | None = None


@dataclass
class PreparedCachedGraph:
    context: Any
    provider: S5DirectEffectRowProvider
    selection: Any
    selected_feature_influence: Tensor
    preparation_seconds: float


def prepare_cached_graph(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    *,
    device: torch.device,
    feature_budget: int,
    queue_size: int,
    row_batch_size: int,
    max_cache_bytes: int | None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> PreparedCachedGraph:
    started = time.perf_counter()
    context = linearized_forward(model, transcoders, trace, device=device)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "linearized_forward_complete",
                "feature_count": context.feature_count,
                "source_count": context.source_count,
                "error_count": context.error_count,
                "token_count": context.token_count,
            }
        )
    provider = S5DirectEffectRowProvider(
        context,
        row_batch_size=row_batch_size,
        use_batched_vjp=True,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "cache_allocation_started",
                "feature_budget": min(feature_budget, context.feature_count),
                "estimated_cache_bytes": (
                    (min(feature_budget, context.feature_count) + 1)
                    * context.source_count
                    * torch.tensor([], dtype=torch.float32).element_size()
                    + (min(feature_budget, context.feature_count) + 1)
                    * torch.tensor([], dtype=torch.float32).element_size()
                ),
            }
        )
    selection = select_features_cached_exact(
        provider,
        feature_budget=feature_budget,
        queue_size=queue_size,
        max_cache_bytes=max_cache_bytes,
        cache_full_source_sums=True,
        progress_callback=progress_callback,
    )
    influence = selected_feature_influence_cached(provider, selection=selection)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "selection_complete",
                "selected_features": int(selection.selected_features.numel()),
                "preparation_seconds": time.perf_counter() - started,
                "provider": provider.diagnostics(),
                "cache": {
                    "estimated_bytes": selection.cache.estimated_bytes,
                    "resident_bytes": selection.cache.resident_bytes,
                    "cached_rows": selection.cache.cached_rows,
                },
            }
        )
    return PreparedCachedGraph(
        context=context,
        provider=provider,
        selection=selection,
        selected_feature_influence=influence.detach().cpu(),
        preparation_seconds=time.perf_counter() - started,
    )


def materialize_prepared_graph(
    prepared: PreparedCachedGraph,
    *,
    event_key: str,
    artifact_id: str,
    condition_length: int,
    d_feature: int,
    node_threshold: float,
    edge_threshold: float,
    max_selected_dense_bytes: int | None,
) -> ExtractedGraph:
    started = time.perf_counter()
    graph = materialize_cached_exact_graph(
        prepared.provider,
        selection=prepared.selection,
        error_count=prepared.context.error_count,
        token_count=prepared.context.token_count,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        max_dense_bytes=max_selected_dense_bytes,
    )
    payload = build_compact_payload(
        prepared.context,
        graph,
        event_key=event_key,
        artifact_id=artifact_id,
        condition_length=condition_length,
        d_feature=d_feature,
    )
    cache = prepared.selection.cache
    return ExtractedGraph(
        payload=payload,
        selected_features=prepared.selection.selected_features.detach().cpu(),
        retained_features=graph.retained_features.detach().cpu(),
        selected_feature_influence=prepared.selected_feature_influence,
        backend="streaming_cached_exact",
        seconds=prepared.preparation_seconds + time.perf_counter() - started,
        cache_stats={
            "estimated_bytes": cache.estimated_bytes,
            "resident_bytes": cache.resident_bytes,
            "cached_rows": cache.cached_rows,
            "read_rows": cache.read_rows,
            "provider_compute_seconds": cache.provider_compute_seconds,
            "cache_write_seconds": cache.cache_write_seconds,
            "cache_read_seconds": cache.cache_read_seconds,
            "ranking_seconds": cache.ranking_seconds,
            "materialization_seconds": cache.materialization_seconds,
            "provider": prepared.provider.diagnostics(),
        },
    )


def selected_feature_influence_recompute(
    provider: S5DirectEffectRowProvider, selected_features: Tensor
) -> Tensor:
    selected = torch.sort(selected_features.detach().cpu().long()).values
    targets = [int(item) for item in selected.tolist()] + [ROOT_TARGET]
    rows = provider.rows(targets).float().cpu()
    selected_columns = rows.abs().index_select(1, selected)
    denominators = selected_columns.sum(dim=1).clamp_min(1e-10)
    core = torch.zeros(
        (len(targets), len(targets)), dtype=torch.float32
    )
    core[:, : selected.numel()] = (
        selected_columns / denominators[:, None]
    )
    root_weights = torch.zeros(len(targets), dtype=torch.float32)
    root_weights[-1] = 1.0
    output = propagate_influence(core, root_weights) + root_weights
    return output @ core[:, : selected.numel()]


def extract_graph(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    *,
    device: torch.device,
    event_key: str,
    artifact_id: str,
    feature_budget: int,
    node_threshold: float,
    edge_threshold: float,
    queue_size: int = 640,
    row_batch_size: int = 8,
    backend: str = "streaming_cached_exact",
    max_cache_bytes: int | None = None,
    max_dense_reference_bytes: int = 64 * 1024**3,
    max_selected_dense_bytes: int | None = None,
) -> ExtractedGraph:
    started = time.perf_counter()
    context = linearized_forward(model, transcoders, trace, device=device)
    provider = S5DirectEffectRowProvider(
        context, row_batch_size=row_batch_size, use_batched_vjp=True
    )
    cache_stats: dict[str, Any] = {}
    if backend == "dense_autograd_reference":
        selected = select_features_streaming_exact(
            provider, feature_budget=feature_budget, queue_size=queue_size
        )
        selected = torch.sort(selected.detach().cpu().long()).values
        selected_influence = selected_feature_influence_recompute(provider, selected)
        nonfeatures = torch.arange(
            provider.feature_count, provider.source_count, dtype=torch.long
        )
        dense_sources = torch.cat((selected, nonfeatures))
        dense_targets = [int(item) for item in selected.tolist()] + [ROOT_TARGET]
        dense_rows = provider.rows(dense_targets).index_select(1, dense_sources)
        dense_count = int(dense_sources.numel()) + 1
        dense_bytes = dense_count * dense_count * 4
        if dense_bytes > int(max_dense_reference_bytes):
            raise MemoryError(
                "selected-node dense reference exceeds its audited capacity: "
                f"{dense_bytes} > {max_dense_reference_bytes}"
            )
        dense_reference = torch.zeros(
            (dense_count, dense_count), dtype=torch.float32
        )
        dense_reference[: selected.numel(), :-1] = dense_rows[:-1]
        dense_reference[-1, :-1] = dense_rows[-1]
        dense_selected_records = context.active_features.detach().cpu().index_select(
            0, selected
        )
        graph = materialize_streaming_exact_graph(
            provider,
            selected_features=selected,
            error_count=context.error_count,
            token_count=context.token_count,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            max_dense_bytes=max_selected_dense_bytes,
        )
        cache_stats["dense_reference_bytes"] = dense_bytes
    elif backend == "streaming_cached_exact":
        dense_reference = None
        dense_selected_records = None
        selection = select_features_cached_exact(
            provider,
            feature_budget=feature_budget,
            queue_size=queue_size,
            max_cache_bytes=max_cache_bytes,
            cache_full_source_sums=True,
        )
        graph = materialize_cached_exact_graph(
            provider,
            selection=selection,
            error_count=context.error_count,
            token_count=context.token_count,
            node_threshold=node_threshold,
            edge_threshold=edge_threshold,
            max_dense_bytes=max_selected_dense_bytes,
        )
        selected = selection.selected_features
        selected_influence = selected_feature_influence_cached(
            provider, selection=selection
        )
        cache = selection.cache
        cache_stats = {
            "estimated_bytes": cache.estimated_bytes,
            "resident_bytes": cache.resident_bytes,
            "cached_rows": cache.cached_rows,
            "read_rows": cache.read_rows,
            "provider_compute_seconds": cache.provider_compute_seconds,
            "cache_write_seconds": cache.cache_write_seconds,
            "cache_read_seconds": cache.cache_read_seconds,
            "ranking_seconds": cache.ranking_seconds,
            "materialization_seconds": cache.materialization_seconds,
        }
    else:
        raise ValueError(f"unsupported S5 circuit backend: {backend}")

    payload = build_compact_payload(
        context,
        graph,
        event_key=event_key,
        artifact_id=artifact_id,
        condition_length=trace.condition_length,
        d_feature=transcoders[0].d_feature,
    )
    return ExtractedGraph(
        payload=payload,
        selected_features=selected.detach().cpu(),
        retained_features=graph.retained_features.detach().cpu(),
        selected_feature_influence=(
            selected_influence.detach().cpu()
            if selected_influence is not None
            else None
        ),
        backend=backend,
        seconds=time.perf_counter() - started,
        cache_stats=cache_stats,
        dense_reference_signed_direct_effect=dense_reference,
        dense_reference_selected_records=dense_selected_records,
    )


def compare_parity(
    reference: ExtractedGraph,
    candidate: ExtractedGraph,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    left, right = reference.payload, candidate.payload
    selected_identity_equal = bool(
        torch.equal(reference.selected_features, candidate.selected_features)
    )
    selected_influence_close = bool(
        reference.selected_feature_influence is not None
        and candidate.selected_feature_influence is not None
        and reference.selected_feature_influence.shape
        == candidate.selected_feature_influence.shape
        and torch.allclose(
            reference.selected_feature_influence.float(),
            candidate.selected_feature_influence.float(),
            atol=float(atol),
            rtol=float(rtol),
        )
    )
    identity_fields = (
        "feature_layer",
        "feature_position",
        "feature_id",
        "error_layer",
        "error_position",
        "replay_token_ids",
        "edge_source",
        "edge_target",
        "target_token_ids",
        "target_positions",
    )
    identity_equal = {
        field: bool(torch.equal(getattr(left, field), getattr(right, field)))
        for field in identity_fields
    }
    numeric_fields = (
        "feature_activation",
        "feature_influence",
        "feature_cumulative_score",
        "error_influence",
        "error_cumulative_score",
        "token_influence",
        "token_cumulative_score",
        "edge_signed_direct_effect",
        "edge_nonnegative_influence",
        "target_probabilities",
        "target_entropies",
        "target_log_probabilities",
    )
    numeric_close = {
        field: bool(
            torch.allclose(
                getattr(left, field).float(),
                getattr(right, field).float(),
                atol=float(atol),
                rtol=float(rtol),
            )
        )
        for field in numeric_fields
        if getattr(left, field).shape == getattr(right, field).shape
    }
    shapes_equal = {
        field: getattr(left, field).shape == getattr(right, field).shape
        for field in numeric_fields
    }
    passes = (
        selected_identity_equal
        and selected_influence_close
        and all(identity_equal.values())
        and all(shapes_equal.values())
        and all(numeric_close.values())
    )
    return {
        "passes": passes,
        "identity_equal": identity_equal,
        "shapes_equal": shapes_equal,
        "numeric_close": numeric_close,
        "selected_features_identity_equal": selected_identity_equal,
        "selected_feature_influence_close": selected_influence_close,
        "reference_seconds": reference.seconds,
        "candidate_seconds": candidate.seconds,
    }


def _mean_output_nll(logits: Tensor, trace: CircuitTrace) -> float:
    rows = logits[0] if logits.ndim == 3 else logits
    positions = torch.arange(
        trace.condition_length - 1,
        trace.condition_length + trace.target_count - 1,
        device=rows.device,
    )
    targets = torch.tensor(trace.generated_token_ids, device=rows.device)
    selected = rows.index_select(0, positions).float()
    return float(
        -torch.log_softmax(selected, dim=-1)
        .gather(1, targets[:, None])
        .mean()
        .item()
    )


def _matched_controls(
    active_features: Tensor,
    activations: Tensor,
    selected: Sequence[int],
    *,
    control_index: int,
) -> list[int]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, record in enumerate(active_features.tolist()):
        groups.setdefault((int(record[0]), int(record[1])), []).append(index)
    log_activation = activations.float().abs().clamp_min(1e-20).log()
    excluded = set(int(item) for item in selected)
    used: set[int] = set()
    output = []
    for feature in selected:
        layer, position, _feature_id = active_features[int(feature)].tolist()
        candidates = [
            item
            for item in groups[(int(layer), int(position))]
            if item not in excluded and item not in used
        ]
        if not candidates:
            raise ValueError("cannot construct exact layer/position matched control")
        candidates.sort(
            key=lambda item: (
                abs(float(log_activation[item] - log_activation[int(feature)])),
                item,
            )
        )
        quartile = max(1, math.ceil(len(candidates) / 4))
        pool = candidates[:quartile]
        choice = pool[control_index % len(pool)]
        used.add(choice)
        output.append(choice)
    return output


def intervention_score(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    extraction: ExtractedGraph,
    *,
    device: torch.device,
    intervention_fraction: float = 0.05,
    random_controls: int = 3,
) -> dict[str, Any]:
    baseline_context = decomposed_forward(
        model, transcoders, trace, device=device, requires_grad=False
    )
    baseline_nll = _mean_output_nll(baseline_context.logits, trace)
    if extraction.selected_feature_influence is None:
        raise ValueError("causal intervention requires cached pre-pruning influence")
    selected = torch.sort(extraction.selected_features.long()).values
    count = max(1, math.ceil(float(intervention_fraction) * len(selected)))
    ranking = torch.argsort(
        extraction.selected_feature_influence.float(), descending=True, stable=True
    )[:count]
    selected_global = selected.index_select(0, ranking).tolist()

    def run(active_indices: Sequence[int]) -> float:
        interventions = []
        for index in active_indices:
            layer, position, feature = baseline_context.active_features[int(index)].tolist()
            interventions.append((layer, position, feature, 0.0))
        logits = feature_intervention_logits(
            model, transcoders, trace, interventions, device=device
        )
        return _mean_output_nll(logits, trace) - baseline_nll

    top_delta = run(selected_global)
    random_deltas = [
        run(
            _matched_controls(
                baseline_context.active_features.cpu(),
                baseline_context.activation_values.cpu(),
                selected_global,
                control_index=index,
            )
        )
        for index in range(int(random_controls))
    ]
    return {
        "feature_count": count,
        "baseline_mean_nll": baseline_nll,
        "top_ablation_delta_nll": top_delta,
        "random_control_delta_nll": random_deltas,
        "matched_effect": top_delta - statistics.mean(random_deltas),
    }


def tokenwise_feature_scores(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    *,
    device: torch.device,
    feature_budget: int,
    queue_size: int,
    row_batch_size: int,
    max_cache_bytes: int | None,
) -> dict[int, float]:
    """Run S3's tokenwise nonnegative-sum reference on one frozen replay."""

    context = linearized_forward(model, transcoders, trace, device=device)
    totals: dict[int, float] = {}
    for target_log_probability in context.objective.target_log_probabilities:
        provider = S5DirectEffectRowProvider(
            context,
            row_batch_size=row_batch_size,
            use_batched_vjp=True,
            root_scalar=target_log_probability,
        )
        selection = select_features_cached_exact(
            provider,
            feature_budget=feature_budget,
            queue_size=queue_size,
            max_cache_bytes=max_cache_bytes,
            cache_full_source_sums=False,
        )
        influence = selected_feature_influence_cached(provider, selection=selection)
        selected = torch.sort(selection.selected_features.cpu().long()).values
        for feature, score in zip(
            selected.tolist(), influence.tolist(), strict=True
        ):
            totals[int(feature)] = totals.get(int(feature), 0.0) + float(score)
        del influence, selected, selection, provider
        gc.collect()
    return totals


def operator_causal_comparison(
    model: torch.nn.Module,
    transcoders: Sequence[RectangularReLUTranscoder],
    trace: CircuitTrace,
    extraction: ExtractedGraph,
    *,
    device: torch.device,
    feature_budget: int,
    queue_size: int,
    row_batch_size: int,
    max_cache_bytes: int | None,
    intervention_fraction: float = 0.05,
    random_controls: int = 3,
    precomputed_tokenwise_scores: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    if extraction.selected_feature_influence is None:
        raise ValueError("joint operator has no pre-pruning feature influence")
    joint_selected = torch.sort(extraction.selected_features.long()).values
    joint_scores = {
        int(feature): float(score)
        for feature, score in zip(
            joint_selected.tolist(),
            extraction.selected_feature_influence.tolist(),
            strict=True,
        )
    }
    tokenwise_scores = (
        dict(precomputed_tokenwise_scores)
        if precomputed_tokenwise_scores is not None
        else tokenwise_feature_scores(
            model,
            transcoders,
            trace,
            device=device,
            feature_budget=feature_budget,
            queue_size=queue_size,
            row_batch_size=row_batch_size,
            max_cache_bytes=max_cache_bytes,
        )
    )
    count = max(
        1,
        math.ceil(
            float(intervention_fraction)
            * min(len(joint_scores), len(tokenwise_scores), int(feature_budget))
        ),
    )

    def top(scores: Mapping[int, float]) -> list[int]:
        return [
            feature
            for feature, _score in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )[:count]
        ]

    baseline_context = decomposed_forward(
        model, transcoders, trace, device=device, requires_grad=False
    )
    baseline_nll = _mean_output_nll(baseline_context.logits, trace)

    def score(selected_global: Sequence[int]) -> dict[str, Any]:
        def run(active_indices: Sequence[int]) -> float:
            interventions = []
            for index in active_indices:
                layer, position, feature = baseline_context.active_features[
                    int(index)
                ].tolist()
                interventions.append((layer, position, feature, 0.0))
            logits = feature_intervention_logits(
                model, transcoders, trace, interventions, device=device
            )
            return _mean_output_nll(logits, trace) - baseline_nll

        top_delta = run(selected_global)
        controls = [
            run(
                _matched_controls(
                    baseline_context.active_features.cpu(),
                    baseline_context.activation_values.cpu(),
                    selected_global,
                    control_index=index,
                )
            )
            for index in range(int(random_controls))
        ]
        return {
            "feature_count": len(selected_global),
            "baseline_mean_nll": baseline_nll,
            "top_ablation_delta_nll": top_delta,
            "random_control_delta_nll": controls,
            "matched_effect": top_delta - statistics.mean(controls),
        }

    return {
        "joint_signed_span": score(top(joint_scores)),
        "tokenwise_nonnegative_sum": score(top(tokenwise_scores)),
        "feature_count": count,
    }

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from .attribution import (
    CACHED_STREAMING_BACKEND,
    CACHED_STREAMING_DUAL_EVENT_BACKEND,
    CACHED_STREAMING_DUAL_VJP_BACKEND,
    CACHED_STREAMING_DUAL_VJP_HOTLOOP_V2_BACKEND,
    CACHED_STREAMING_HOTLOOP_BACKEND,
    CACHED_STREAMING_OPTIMIZED_BACKEND,
    CACHED_STREAMING_PIPELINE_BACKEND,
    CACHED_STREAMING_VJP_BACKEND,
    CACHED_STREAMING_VJP_HOTLOOP_V2_BACKEND,
    PIPELINE_BACKENDS,
    STREAMING_BACKEND,
    CachedAttributionPreparation,
    StreamingAttribution,
    attribute_joint_span_cached_exact,
    attribute_joint_span_cached_exact_v2,
    attribute_joint_span_streaming_exact,
    finalize_joint_span_cached_exact,
    prepare_joint_span_cached_hotloop_exact,
    prepare_joint_span_cached_pipeline_exact,
)
from .config import NvarcCircuitConfig
from .events import CircuitEvent
from .generation import GenerationRecord
from .identity import artifact_id
from .parity import validate_tl_frozen_greedy_replay
from .payload import CompactPayload, build_payload_from_streaming_attribution
from .policy import ProductionPolicy
from .span_objective import validate_generation_diagnostics


@dataclass(frozen=True)
class PreparedEventPayload:
    event_key: str
    artifact_id: str
    condition_length: int
    attribution: CachedAttributionPreparation
    replay_metrics: dict[str, Any]


def _replay_metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "replay_argmax_required": bool(report["expected_greedy_required"]),
        "replay_argmax_exact": bool(report["argmax_exact"]),
        "replay_argmax_mismatches": int(report["argmax_mismatches"]),
        "replay_argmax_mismatch_max_gap": float(
            report["argmax_mismatch_max_gap"]
        ),
        "replay_target_diagnostic_shape": str(
            report["target_diagnostic_replay_shape"]
        ),
        "replay_allowed_argmax_sha256": str(
            report["full_replay_allowed_argmax_sha256"]
        ),
        "replay_bf16_ambiguities_accepted": int(
            report["bf16_logit_ambiguities_accepted"]
        ),
        "replay_probabilities_match_native_hf": bool(
            report["probabilities_match_native_hf"]
        ),
        "replay_native_hf_probability_max_abs": float(
            report["native_hf_probability_max_abs"]
        ),
        "replay_native_hf_probability_mean_abs": float(
            report["native_hf_probability_mean_abs"]
        ),
        "replay_native_hf_probability_max_rel": float(
            report["native_hf_probability_max_rel"]
        ),
    }


def build_event_artifact_id(
    *,
    event: CircuitEvent,
    generation: GenerationRecord,
    config: NvarcCircuitConfig,
    policy: ProductionPolicy,
) -> str:
    return artifact_id(
        event_key_value=event.event_key,
        condition_token_ids=event.condition_input_ids,
        generated_token_ids=generation.generated_token_ids,
        model_identity_sha256=policy.model_identity_sha256,
        tokenizer_identity_sha256=policy.tokenizer_identity_sha256,
        transcoder_identity_sha256=policy.transcoder_identity_sha256,
        operator_identity=policy.operator_identity,
        backend_identity=policy.backend_identity,
        production_policy_sha256=policy.semantic_sha256,
        payload_schema=config.artifacts.payload_schema,
        vector_schema=config.artifacts.vector_schema,
        trail_schema=config.artifacts.trail_schema,
    )


def _update_runtime_metrics(
    runtime_metrics: MutableMapping[str, Any] | None,
    attribution: StreamingAttribution,
) -> None:
    if runtime_metrics is None:
        return
    runtime_metrics.update(
        {
            "row_cache_estimated_bytes": attribution.row_cache_estimated_bytes,
            "row_cache_resident_bytes": attribution.row_cache_resident_bytes,
            "unique_cached_rows": attribution.unique_cached_rows,
            "cached_row_reads": attribution.cached_row_reads,
            "cached_full_source_sum_reads": (
                attribution.cached_full_source_sum_reads
            ),
            "cached_submatrix_read_elements": (
                attribution.cached_submatrix_read_elements
            ),
            "provider_row_compute_seconds": attribution.provider_row_compute_seconds,
            "row_cache_write_seconds": attribution.row_cache_write_seconds,
            "row_cache_read_seconds": attribution.row_cache_read_seconds,
            "feature_ranking_seconds": attribution.feature_ranking_seconds,
            "graph_materialization_seconds": (
                attribution.graph_materialization_seconds
            ),
            "async_d2h_enabled": attribution.async_d2h_enabled,
            "async_d2h_blocks": attribution.async_d2h_blocks,
            "async_d2h_bytes": attribution.async_d2h_bytes,
        }
    )
    if attribution.exact_vjp_metrics is not None:
        runtime_metrics.update(attribution.exact_vjp_metrics)


def prepare_event_payload_pipeline(
    *,
    model,
    event: CircuitEvent,
    generation: GenerationRecord,
    config: NvarcCircuitConfig,
    policy: ProductionPolicy,
    token_contract: Any,
    grammar_factory: Callable[[], Any],
    max_exact_row_cache_bytes: int | None = None,
    gpu_lock: Any | None = None,
    require_replay_argmax: bool = True,
    align_target_diagnostics_to_span_replay: bool = False,
) -> PreparedEventPayload:
    """Finish one event's GPU work and retain only CPU materialization state."""

    policy.validate(config)
    if policy.backend_identity not in PIPELINE_BACKENDS:
        raise ValueError("Pipeline preparation requires the pipeline backend")
    event.validate(max_sequence_length=config.data.max_sequence_length)
    generation.validate(condition_length=event.condition_length)
    if generation.event_key != event.event_key:
        raise ValueError("Event catalog and frozen generation do not align")
    with gpu_lock if gpu_lock is not None else nullcontext():
        replay_diagnostics = validate_tl_frozen_greedy_replay(
            tl_model=model,
            condition_token_ids=event.condition_input_ids,
            generation=generation,
            token_contract=token_contract,
            grammar_factory=grammar_factory,
            atol=policy.diagnostic_atol,
            rtol=policy.diagnostic_rtol,
            require_expected_greedy=require_replay_argmax,
            align_target_diagnostics_to_span_replay=(
                align_target_diagnostics_to_span_replay
            ),
        )
    identifier = build_event_artifact_id(
        event=event,
        generation=generation,
        config=config,
        policy=policy,
    )
    preparation_function = (
        prepare_joint_span_cached_pipeline_exact
        if policy.backend_identity == CACHED_STREAMING_PIPELINE_BACKEND
        else prepare_joint_span_cached_hotloop_exact
    )
    preparation_kwargs: dict[str, Any] = {
        "model": model,
        "condition_token_ids": event.condition_input_ids,
        "target_token_ids": generation.generated_token_ids,
        "row_batch_size": policy.row_batch_size,
        "compute_microbatch_policy": policy.compute_microbatch_policy,
        "compute_microbatch_cap": policy.compute_microbatch_cap,
        "max_feature_nodes": policy.feature_budget,
        "update_interval": policy.update_interval,
        "node_threshold": policy.node_threshold,
        "edge_threshold": policy.edge_threshold,
        "max_node_selected_dense_bytes": policy.max_node_selected_dense_bytes,
        "max_exact_row_cache_bytes": max_exact_row_cache_bytes,
    }
    if policy.backend_identity in {
        CACHED_STREAMING_HOTLOOP_BACKEND,
        CACHED_STREAMING_DUAL_EVENT_BACKEND,
        CACHED_STREAMING_VJP_BACKEND,
        CACHED_STREAMING_DUAL_VJP_BACKEND,
        CACHED_STREAMING_VJP_HOTLOOP_V2_BACKEND,
        CACHED_STREAMING_DUAL_VJP_HOTLOOP_V2_BACKEND,
    }:
        preparation_kwargs.update(
            gpu_lock=gpu_lock,
            backend=policy.backend_identity,
        )
    attribution = preparation_function(**preparation_kwargs)
    validate_generation_diagnostics(
        attribution.objective,
        target_probabilities=replay_diagnostics["target_probabilities"],
        target_entropies=replay_diagnostics["target_entropies"],
        target_log_probabilities=replay_diagnostics["target_log_probabilities"],
        atol=policy.diagnostic_atol,
        rtol=policy.diagnostic_rtol,
    )
    return PreparedEventPayload(
        event_key=event.event_key,
        artifact_id=identifier,
        condition_length=event.condition_length,
        attribution=attribution,
        replay_metrics=_replay_metric_summary(replay_diagnostics),
    )


def finalize_event_payload_pipeline(
    prepared: PreparedEventPayload,
    *,
    runtime_metrics: MutableMapping[str, Any] | None = None,
) -> CompactPayload:
    attribution = finalize_joint_span_cached_exact(prepared.attribution)
    _update_runtime_metrics(runtime_metrics, attribution)
    return build_payload_from_streaming_attribution(
        event_key=prepared.event_key,
        artifact_id=prepared.artifact_id,
        condition_length=prepared.condition_length,
        attribution=attribution,
    )


def extract_event_payload(
    *,
    model,
    event: CircuitEvent,
    generation: GenerationRecord,
    config: NvarcCircuitConfig,
    policy: ProductionPolicy,
    token_contract: Any,
    grammar_factory: Callable[[], Any],
    runtime_metrics: MutableMapping[str, Any] | None = None,
    max_exact_row_cache_bytes: int | None = None,
    require_replay_argmax: bool = True,
    align_target_diagnostics_to_span_replay: bool = False,
) -> CompactPayload:
    """Extract one identity-locked production payload from frozen generation."""

    if policy.backend_identity in PIPELINE_BACKENDS:
        prepared = prepare_event_payload_pipeline(
            model=model,
            event=event,
            generation=generation,
            config=config,
            policy=policy,
            token_contract=token_contract,
            grammar_factory=grammar_factory,
            max_exact_row_cache_bytes=max_exact_row_cache_bytes,
            require_replay_argmax=require_replay_argmax,
            align_target_diagnostics_to_span_replay=(
                align_target_diagnostics_to_span_replay
            ),
        )
        if runtime_metrics is not None:
            runtime_metrics.update(prepared.replay_metrics)
        return finalize_event_payload_pipeline(
            prepared, runtime_metrics=runtime_metrics
        )

    policy.validate(config)
    event.validate(max_sequence_length=config.data.max_sequence_length)
    generation.validate(condition_length=event.condition_length)
    if generation.event_key != event.event_key:
        raise ValueError("Event catalog and frozen generation do not align")
    replay_diagnostics = validate_tl_frozen_greedy_replay(
        tl_model=model,
        condition_token_ids=event.condition_input_ids,
        generation=generation,
        token_contract=token_contract,
        grammar_factory=grammar_factory,
        atol=policy.diagnostic_atol,
        rtol=policy.diagnostic_rtol,
        require_expected_greedy=require_replay_argmax,
        align_target_diagnostics_to_span_replay=(
            align_target_diagnostics_to_span_replay
        ),
    )
    if runtime_metrics is not None:
        runtime_metrics.update(_replay_metric_summary(replay_diagnostics))
    identifier = build_event_artifact_id(
        event=event,
        generation=generation,
        config=config,
        policy=policy,
    )
    attribution_function = {
        STREAMING_BACKEND: attribute_joint_span_streaming_exact,
        CACHED_STREAMING_BACKEND: attribute_joint_span_cached_exact,
        CACHED_STREAMING_OPTIMIZED_BACKEND: attribute_joint_span_cached_exact_v2,
    }.get(policy.backend_identity)
    if attribution_function is None:
        raise ValueError(
            f"Unsupported Stage-B attribution backend: {policy.backend_identity}"
        )
    attribution_kwargs = {
        "condition_token_ids": event.condition_input_ids,
        "target_token_ids": generation.generated_token_ids,
        "row_batch_size": policy.row_batch_size,
        "compute_microbatch_policy": policy.compute_microbatch_policy,
        "compute_microbatch_cap": policy.compute_microbatch_cap,
        "max_feature_nodes": policy.feature_budget,
        "update_interval": policy.update_interval,
        "node_threshold": policy.node_threshold,
        "edge_threshold": policy.edge_threshold,
        "max_node_selected_dense_bytes": policy.max_node_selected_dense_bytes,
    }
    if policy.backend_identity in {
        CACHED_STREAMING_BACKEND,
        CACHED_STREAMING_OPTIMIZED_BACKEND,
    }:
        attribution_kwargs["max_exact_row_cache_bytes"] = max_exact_row_cache_bytes
    attribution = attribution_function(model, **attribution_kwargs)
    validate_generation_diagnostics(
        attribution.objective,
        target_probabilities=replay_diagnostics["target_probabilities"],
        target_entropies=replay_diagnostics["target_entropies"],
        target_log_probabilities=replay_diagnostics["target_log_probabilities"],
        atol=policy.diagnostic_atol,
        rtol=policy.diagnostic_rtol,
    )
    _update_runtime_metrics(runtime_metrics, attribution)
    return build_payload_from_streaming_attribution(
        event_key=event.event_key,
        artifact_id=identifier,
        condition_length=event.condition_length,
        attribution=attribution,
    )

"""Configuration and frozen protocol checks for the S5 circuit pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..protocol import ALL_TASKS


CIRCUIT_CONFIG_FORMAT = "s5_s3_style_circuit_pilot_v1"
CIRCUIT_CONFIG_FORMAT_W8 = "s5_s3_style_circuit_pilot_w8_v2"
LEGACY_WIDTH_MULTIPLIERS = (32, 64)
WIDTH_MULTIPLIERS = (8,)
AUDIT_LAYERS = (0, 3, 7)
TRANSCODER_LEARNING_RATES = (5e-6, 1e-5, 2e-5)
FEATURE_BUDGETS = (2048, 4096, 8192, 16384)
NODE_THRESHOLDS = (0.8, 0.9, 0.95)
EDGE_THRESHOLDS = (0.98, 0.99, 0.995)
CIRCUIT_SOURCE_CHECKPOINT_SHA256 = (
    "ee2a7efdf77262ee1fbdf7d2080f930c2bedc27e45b7712537fba964ee05931c"
)
CIRCUIT_TOKENIZER_SHA256 = (
    "1053e327f96ee13ceca0c4ab4d9e48eefa9eb2c2eddf65359715642ec701199f"
)
CIRCUIT_ARCHITECTURE_CONFIG_SHA256 = (
    "c4d88a4c8e722804e53702990b01d84f5e267e8dffd28e9a7a4e5fa6eaefb58f"
)


def _semantic_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_circuit_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    validate_circuit_config(value)
    value["_config_path"] = str(source)
    value["_config_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    value["_semantic_sha256"] = _semantic_sha256(
        {key: item for key, item in value.items() if not key.startswith("_")}
    )
    return value


def trace_source_config_sha256(config: dict[str, Any]) -> str:
    """Return the frozen config identity carried by the reused Trace archive."""
    value = config.get("traces", {}).get("source_config_sha256")
    return str(value) if value is not None else str(config["_config_sha256"])


def _tuple(value: Any, cast: type) -> tuple[Any, ...]:
    return tuple(cast(item) for item in value)


def validate_circuit_config(config: dict[str, Any]) -> None:
    config_format = config.get("format")
    if config_format not in (CIRCUIT_CONFIG_FORMAT, CIRCUIT_CONFIG_FORMAT_W8):
        raise ValueError(
            f"format must be {CIRCUIT_CONFIG_FORMAT} or {CIRCUIT_CONFIG_FORMAT_W8}"
        )
    source = config["source_model"]
    if source.get("protocol_id") != "s5_narrow_atomic_only_sft_dense_bigram_off_v1":
        raise ValueError("circuit pilot is locked to the dense, bigram-disabled base")
    if int(source.get("checkpoint_step", -1)) != 38_000:
        raise ValueError("circuit pilot is locked to checkpoint step 38000")
    expected_checkpoint = (
        CIRCUIT_SOURCE_CHECKPOINT_SHA256 if config_format == CIRCUIT_CONFIG_FORMAT_W8
        else "aee775e5691d7b386542e55d89be9fecedab0137d6d6c3e758ef66deed8bcb56"
    )
    if source.get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError("circuit source raw step-38000 checkpoint SHA256 changed")
    if source.get("tokenizer_sha256") != CIRCUIT_TOKENIZER_SHA256:
        raise ValueError("circuit tokenizer SHA256 changed")
    if (
        source.get("architecture_config_sha256")
        != CIRCUIT_ARCHITECTURE_CONFIG_SHA256
    ):
        raise ValueError("circuit architecture-config SHA256 changed")
    checkpoint_config_sha256 = str(source.get("checkpoint_config_sha256", ""))
    if len(checkpoint_config_sha256) != 64:
        raise ValueError("source_model.checkpoint_config_sha256 must be SHA256")
    if bool(source.get("enable_bigram_table", True)):
        raise ValueError("initial circuit source must have the bigram table disabled")

    generation = config["generation"]
    if generation.get("decoding") != "greedy_argmax":
        raise ValueError("generation.decoding must be greedy_argmax")
    if int(generation.get("max_new_tokens", -1)) != 50:
        raise ValueError("generation.max_new_tokens must equal 50")
    if int(generation.get("context_length", -1)) != 128:
        raise ValueError("generation.context_length must equal 128")
    if tuple(generation.get("stop_markers", ())) != ("proof complete", "no goals"):
        raise ValueError("generation.stop_markers changed from the evaluator contract")
    if generation.get("target") != "self_generated_full_output":
        raise ValueError("generation target must be self_generated_full_output")

    traces = config["traces"]
    if "source_config_sha256" in traces:
        raise ValueError("public trace generation and replay must use the same recipe")
    if tuple(traces.get("tasks", ())) != ALL_TASKS:
        raise ValueError("Transcoder traces must cover all nine tasks in canonical order")
    if int(traces.get("train_seed_base", -1)) != 555_555_000:
        raise ValueError("traces.train_seed_base must equal 555555000")
    if int(traces.get("validation_seed_base", -1)) != 666_666_000:
        raise ValueError("traces.validation_seed_base must equal 666666000")
    if int(traces.get("train_tokens_per_task", -1)) != 10_000_000:
        raise ValueError("initial train token quota must equal 10M per task")
    if int(traces.get("max_train_tokens_per_task", -1)) != 20_000_000:
        raise ValueError("maximum train token quota must equal 20M per task")
    if int(traces.get("validation_tokens_per_task", -1)) != 200_000:
        raise ValueError("validation token quota must equal 200k per task")
    if traces.get("balance_unit") != "eligible_tokens":
        raise ValueError("Transcoder traces must be balanced by eligible tokens")
    if int(traces.get("n_distractors", -1)) != 2:
        raise ValueError("traces.n_distractors must equal 2")
    trace_generation_exact = {
        "generation_backend": "same_length_batched_canonical_verified_v1",
        "generation_batch_size": 128,
        "generation_pool_size": 2048,
        "teacher_forced_verify_all": True,
        "canonical_regeneration_interval": 4096,
    }
    for key, expected_value in trace_generation_exact.items():
        if traces.get(key) != expected_value:
            raise ValueError(f"traces.{key} must equal {expected_value}")
    expected_shards = {
        "a": list(ALL_TASKS[:5]),
        "b": list(ALL_TASKS[5:]),
    }
    if traces.get("formal_single_node_shards") != expected_shards:
        raise ValueError(
            "traces.formal_single_node_shards must be the frozen disjoint 5+4 split"
        )

    transcoder = config["transcoder"]
    if int(transcoder.get("d_in", -1)) != 2080:
        raise ValueError("transcoder.d_in must equal 2080")
    if int(transcoder.get("d_out", -1)) != 2048:
        raise ValueError("transcoder.d_out must equal 2048")
    if int(transcoder.get("layers", -1)) != 8:
        raise ValueError("transcoder.layers must equal 8")
    expected_widths = (
        WIDTH_MULTIPLIERS
        if config_format == CIRCUIT_CONFIG_FORMAT_W8
        else LEGACY_WIDTH_MULTIPLIERS
    )
    if _tuple(transcoder.get("width_multipliers", ()), int) != expected_widths:
        raise ValueError(f"width multipliers must equal {expected_widths}")
    if _tuple(transcoder.get("audit_layers", ()), int) != AUDIT_LAYERS:
        raise ValueError(f"audit layers must equal {AUDIT_LAYERS}")
    if _tuple(transcoder.get("production_and_audit_seeds", ()), int) != (42, 43, 44):
        raise ValueError("Transcoder seeds must equal [42, 43, 44]")
    if transcoder.get("production_seed") != 42:
        raise ValueError("Transcoder production seed must equal 42")
    if transcoder.get("initialization") != "random_circuit_tracer_reference":
        raise ValueError("Transcoder initialization contract changed")
    if float(transcoder.get("sparsity_coeff", -1.0)) != 8.0:
        raise ValueError("Transcoder sparsity_coeff must equal 8.0")
    if float(transcoder.get("c_coeff", -1.0)) != 4.0:
        raise ValueError("Transcoder c_coeff must equal 4.0")

    optimization = config["optimization"]
    if _tuple(optimization.get("learning_rates", ()), float) != TRANSCODER_LEARNING_RATES:
        raise ValueError(f"learning-rate candidates must equal {TRANSCODER_LEARNING_RATES}")
    expected = {
        "effective_tokens": 8192,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "warmup_fraction": 0.02,
        "cooldown_fraction": 0.20,
        "checkpoint_percent": 10,
    }
    for key, expected_value in expected.items():
        actual = optimization.get(key)
        if float(actual) != float(expected_value):
            raise ValueError(f"optimization.{key} must equal {expected_value}")

    graph = config["graph"]
    if graph.get("operator") != "joint_signed_full_output":
        raise ValueError("graph.operator must be joint_signed_full_output")
    if graph.get("reference_operator") != "tokenwise_nonnegative_sum":
        raise ValueError("graph.reference_operator changed")
    if _tuple(graph.get("feature_budgets", ()), int) != FEATURE_BUDGETS:
        raise ValueError(f"feature budgets must equal {FEATURE_BUDGETS}")
    if _tuple(graph.get("node_thresholds", ()), float) != NODE_THRESHOLDS:
        raise ValueError(f"node thresholds must equal {NODE_THRESHOLDS}")
    if _tuple(graph.get("edge_thresholds", ()), float) != EDGE_THRESHOLDS:
        raise ValueError(f"edge thresholds must equal {EDGE_THRESHOLDS}")
    if int(graph.get("audit_examples", -1)) != 64:
        raise ValueError("graph.audit_examples must equal 64")
    if int(graph.get("audit_escalation_examples", -1)) != 256:
        raise ValueError("graph.audit_escalation_examples must equal 256")
    if int(graph.get("parity_examples", -1)) != 12:
        raise ValueError("graph.parity_examples must equal 12")
    if int(graph.get("audit_seed_base", -1)) != 777_777_000:
        raise ValueError("graph.audit_seed_base must equal 777777000")
    if float(graph.get("intervention_fraction", -1.0)) != 0.05:
        raise ValueError("graph.intervention_fraction must equal 0.05")
    if int(graph.get("random_controls", -1)) != 3:
        raise ValueError("graph.random_controls must equal 3")
    if graph.get("production_backend") != "streaming_cached_exact":
        raise ValueError("production backend must be streaming_cached_exact")
    if graph.get("reference_backend") != "dense_autograd_reference":
        raise ValueError("reference backend must be dense_autograd_reference")
    graph_exact = {
        "queue_size": 640,
        "row_batch_size": 8,
        "max_cache_bytes": 137_438_953_472,
        "max_dense_reference_bytes": 68_719_476_736,
        "max_selected_dense_bytes": 68_719_476_736,
    }
    for key, expected_value in graph_exact.items():
        if int(graph.get(key, -1)) != expected_value:
            raise ValueError(f"graph.{key} must equal {expected_value}")

    artifacts = config["artifacts"]
    if artifacts.get("payload_retention") != "permanent":
        raise ValueError("CompactPayload retention must be permanent")
    if bool(artifacts.get("materialize_trails", True)):
        raise ValueError("first pilot must not materialize trails")
    if bool(artifacts.get("per_token_attribution", True)):
        raise ValueError("first pilot stores joint attribution only")
    if artifacts.get("chunking") != "byte_bounded_from_audit":
        raise ValueError("payload chunking must be byte-bounded from audit evidence")
    if artifacts.get("label_storage") != "separate_analysis_join":
        raise ValueError("labels and correctness must use a separate analysis join")
    deferred = config["deferred_gates"]
    if any(
        deferred.get(key) is not None
        for key in (
            "absolute_transcoder_fidelity",
            "fail_closed_production",
            "action_if_budget_16384_unstable",
        )
    ):
        raise ValueError("pilot config must keep deferred production gates null")
    if bool(deferred.get("production_900_authorized", True)):
        raise ValueError("pilot config must keep production_900 unauthorized")

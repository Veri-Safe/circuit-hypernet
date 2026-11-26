from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from .batching import SUPPORTED_COMPUTE_MICROBATCH_POLICIES
from .config import (
    NvarcCircuitConfig,
    OUTSIDE_B0_DATA_CONTRACT,
    OUTSIDE_B1_DATA_CONTRACT,
    REARC8_DATA_CONTRACT,
    REARC8_PUBLIC_EVAL_DATA_CONTRACT,
    REARC8_SEED24_DATA_CONTRACT,
)
from .identity import semantic_sha256

POLICY_FORMAT = "nvarc_rearc_circuit_production_policy_v2"
REARC8_POLICY_FORMAT = "nvarc_rearc8_circuit_production_policy_v1"
REARC128_OUTSIDE_POLICY_FORMAT = (
    "nvarc_rearc128_outside_circuit_production_policy_v1"
)
REARC8_V4_NUMERICS_BACKEND = "qwen3_native_hf_fa2_reused_rope_v4"
REARC128_OUTSIDE_NUMERICS_BACKEND = "qwen3_native_hf_fa2_reused_rope_v4"
REARC128_OUTSIDE_REQUIRED_GATES = frozenset(
    {
        "dataset_event_identity",
        "gold_prefix_fidelity",
        "hf_tl_parity",
        "backend_equivalence",
        "compact_vector_exactness",
        "transcoder_fidelity",
        "capacity",
        "feature_pruning",
        "operator_causal",
    }
)


@dataclass(frozen=True)
class ProductionPolicy:
    config_sha256: str
    model_identity_sha256: str
    tokenizer_identity_sha256: str
    transcoder_identity_sha256: str
    backend_identity: str
    operator_identity: str
    generation_batch_size: int
    row_batch_size: int
    compute_microbatch_policy: str
    compute_microbatch_cap: int
    feature_budget: int
    update_interval: int
    node_threshold: float
    edge_threshold: float
    path_epsilon: float
    diagnostic_atol: float
    diagnostic_rtol: float
    max_node_selected_dense_bytes: int
    gate_sha256: Mapping[str, str]

    @property
    def semantic_sha256(self) -> str:
        return semantic_sha256({"format": POLICY_FORMAT, **asdict(self)})

    def validate(self, config: NvarcCircuitConfig) -> None:
        validate_policy_settings(self, config)
        hashes = {
            "config_sha256": self.config_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "transcoder_identity_sha256": self.transcoder_identity_sha256,
            **dict(self.gate_sha256),
        }
        if any(
            not isinstance(value, str) or len(value) != 64 for value in hashes.values()
        ):
            raise ValueError("Production policy contains a malformed identity hash")
        required_gates = {
            "gold_prefix_fidelity",
            "hf_tl_parity",
            "backend_equivalence",
            "capacity",
            "feature_pruning",
            "operator_causal",
        }
        if set(self.gate_sha256) != required_gates:
            raise ValueError(
                f"Production policy gate coverage changed: {set(self.gate_sha256)}"
            )


@dataclass(frozen=True)
class Rearc8ProductionPolicy(ProductionPolicy):
    numerics_backend: str

    @property
    def semantic_sha256(self) -> str:
        return semantic_sha256({"format": REARC8_POLICY_FORMAT, **asdict(self)})

    def validate(self, config: NvarcCircuitConfig) -> None:
        validate_policy_settings(self, config)
        if config.data.contract not in {
            REARC8_DATA_CONTRACT,
            REARC8_PUBLIC_EVAL_DATA_CONTRACT,
            REARC8_SEED24_DATA_CONTRACT,
        }:
            raise ValueError("RE-ARC8 production policy requires the RE-ARC8 contract")
        if self.numerics_backend != REARC8_V4_NUMERICS_BACKEND:
            raise ValueError("RE-ARC8 production numerics backend changed")
        hashes = {
            "config_sha256": self.config_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "transcoder_identity_sha256": self.transcoder_identity_sha256,
            **dict(self.gate_sha256),
        }
        if any(
            not isinstance(value, str) or len(value) != 64 for value in hashes.values()
        ):
            raise ValueError("RE-ARC8 production policy has a malformed identity hash")
        required_gates = {"stage_b_preflight", "v4_closure"}
        if set(self.gate_sha256) != required_gates:
            raise ValueError(
                "RE-ARC8 production policy gate coverage changed: "
                f"{set(self.gate_sha256)}"
            )


@dataclass(frozen=True)
class Rearc128OutsideProductionPolicy(ProductionPolicy):
    """Checkpoint-specific policy unlocked only by the RE-ARC128 gate pack."""

    numerics_backend: str
    execution_mode: str
    validation_manifest_sha256: str

    @property
    def semantic_sha256(self) -> str:
        return semantic_sha256(
            {"format": REARC128_OUTSIDE_POLICY_FORMAT, **asdict(self)}
        )

    def validate(self, config: NvarcCircuitConfig) -> None:
        validate_policy_settings(self, config)
        if config.data.contract not in {
            OUTSIDE_B0_DATA_CONTRACT,
            OUTSIDE_B1_DATA_CONTRACT,
        }:
            raise ValueError(
                "RE-ARC128 outside policy requires a B0 or B1 data contract"
            )
        if self.numerics_backend != REARC128_OUTSIDE_NUMERICS_BACKEND:
            raise ValueError("RE-ARC128 outside production numerics backend changed")
        if self.execution_mode != "dual_worker_exact_with_single_fallback_v1":
            raise ValueError("RE-ARC128 outside execution mode changed")
        hashes = {
            "config_sha256": self.config_sha256,
            "model_identity_sha256": self.model_identity_sha256,
            "tokenizer_identity_sha256": self.tokenizer_identity_sha256,
            "transcoder_identity_sha256": self.transcoder_identity_sha256,
            "validation_manifest_sha256": self.validation_manifest_sha256,
            **dict(self.gate_sha256),
        }
        if any(
            not isinstance(value, str) or len(value) != 64
            for value in hashes.values()
        ):
            raise ValueError(
                "RE-ARC128 outside production policy has a malformed identity hash"
            )
        if set(self.gate_sha256) != REARC128_OUTSIDE_REQUIRED_GATES:
            raise ValueError(
                "RE-ARC128 outside production policy gate coverage changed: "
                f"{set(self.gate_sha256)}"
            )


def validate_policy_settings(
    policy: ProductionPolicy,
    config: NvarcCircuitConfig,
) -> None:
    """Validate shared scientific settings without granting production status."""

    if policy.config_sha256 != config.semantic_sha256:
        raise ValueError("Policy was resolved for a different Stage-B config")
    if policy.backend_identity != config.attribution.backend:
        raise ValueError("Backend identity changed")
    if policy.operator_identity != config.attribution.operator:
        raise ValueError("Operator identity changed")
    if policy.generation_batch_size not in config.generation.batch_candidates:
        raise ValueError("Generation batch was not a configured candidate")
    if policy.feature_budget not in config.attribution.feature_budget_candidates:
        raise ValueError("Feature budget was not a configured candidate")
    if policy.node_threshold not in config.attribution.node_threshold_candidates:
        raise ValueError("Node threshold was not a configured candidate")
    if policy.edge_threshold not in config.attribution.edge_threshold_candidates:
        raise ValueError("Edge threshold was not a configured candidate")
    if policy.row_batch_size != config.attribution.row_batch_size or (
        policy.update_interval != config.attribution.attribute_update_interval
    ):
        raise ValueError("Attribution batching policy is invalid")
    if (
        policy.compute_microbatch_policy
        not in SUPPORTED_COMPUTE_MICROBATCH_POLICIES
        or not 0 < policy.compute_microbatch_cap <= policy.row_batch_size
    ):
        raise ValueError("Attribution compute microbatch policy is invalid")
    if policy.path_epsilon != config.attribution.path_epsilon:
        raise ValueError("Path epsilon changed after policy resolution")
    if min(policy.diagnostic_atol, policy.diagnostic_rtol) < 0.0:
        raise ValueError("Diagnostic replay tolerances must be nonnegative")
    if policy.max_node_selected_dense_bytes <= 0:
        raise ValueError("Materialization capacity must be positive")


def load_production_policy(
    path: str | Path, *, config: NvarcCircuitConfig
) -> ProductionPolicy | Rearc8ProductionPolicy | Rearc128OutsideProductionPolicy:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Unsupported production-policy format")
    format_version = value.pop("format", None)
    recorded_sha = value.pop("policy_sha256", None)
    if format_version == POLICY_FORMAT:
        policy = ProductionPolicy(**value)
    elif format_version == REARC8_POLICY_FORMAT:
        policy = Rearc8ProductionPolicy(**value)
    elif format_version == REARC128_OUTSIDE_POLICY_FORMAT:
        policy = Rearc128OutsideProductionPolicy(**value)
    else:
        raise ValueError("Unsupported production-policy format")
    policy.validate(config)
    if recorded_sha != policy.semantic_sha256:
        raise ValueError("Production-policy checksum changed")
    return policy

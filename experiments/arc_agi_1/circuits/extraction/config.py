from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from s3circuit.nvarc_transcoder.config import NvarcTranscoderConfig
from s3circuit.nvarc_transcoder.config import load_config as load_stage_a_config
from s3circuit.nvarc_transcoder.identity import config_semantic_sha256_for_root


FORMAT_VERSION = "nvarc_rearc_stage_b_v1"
VECTOR_DIM = 59
LEGACY_DATA_CONTRACT = "rearc_step399_v1"
REARC8_DATA_CONTRACT = "rearc8_union_fast128_v1"
REARC8_SEED24_DATA_CONTRACT = "rearc8_seed24_exclusive_v1"
REARC8_PUBLIC_EVAL_DATA_CONTRACT = "rearc8_public_eval_v1"
REARC8_SEED24_SOURCE = "rearc8_pg_s24_excl_base_s42_v1"
REARC8_PUBLIC_CANONICAL_SOURCE = "arc1_public_canonical"
REARC8_PUBLIC_AUTHOR_VIEW_SOURCE = "arc1_public_author_view"
OUTSIDE_UNION_DATA_CONTRACT = "rearc128_outside_b0_b1_union_v1"
OUTSIDE_B0_DATA_CONTRACT = "rearc128_outside_b0_v1"
OUTSIDE_B1_DATA_CONTRACT = "rearc128_outside_b1_v1"
OUTSIDE_B0_SOURCE = "outside_r128_b0"
OUTSIDE_B1_SOURCE = "outside_r128_b1"
REARC8_SUBSET_DATA_CONTRACTS = {
    REARC8_SEED24_DATA_CONTRACT,
    REARC8_PUBLIC_EVAL_DATA_CONTRACT,
}
OUTSIDE_SUBSET_DATA_CONTRACTS = {
    OUTSIDE_B0_DATA_CONTRACT,
    OUTSIDE_B1_DATA_CONTRACT,
}
MANIFEST_DATA_CONTRACTS = {
    REARC8_DATA_CONTRACT,
    *REARC8_SUBSET_DATA_CONTRACTS,
    OUTSIDE_UNION_DATA_CONTRACT,
    *OUTSIDE_SUBSET_DATA_CONTRACTS,
}
SOURCE_FILTERED_DATA_CONTRACTS = {
    *REARC8_SUBSET_DATA_CONTRACTS,
    *OUTSIDE_SUBSET_DATA_CONTRACTS,
}


def _required(value: Mapping[str, Any], key: str) -> Any:
    if key not in value:
        raise ValueError(f"Missing required Stage-B configuration field: {key}")
    return value[key]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"Unresolved environment variable: {expanded}")
        return expanded
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _strict_tuple(value: Any, expected: tuple[Any, ...], label: str) -> tuple[Any, ...]:
    actual = tuple(value)
    if actual != expected:
        raise ValueError(f"{label} changed: {actual} != {expected}")
    return actual


@dataclass(frozen=True)
class StageAConfig:
    profile: str
    production_lr_policy: str
    canonical_export_manifest: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "StageAConfig":
        return cls(
            profile=str(_required(value, "profile")),
            production_lr_policy=str(_required(value, "production_lr_policy")),
            canonical_export_manifest=str(
                _required(value, "canonical_export_manifest")
            ),
        )


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    validation_path: str
    expected_train_rows: int
    expected_train_puzzles: int
    expected_train_events: int
    max_sequence_length: int
    seed: int
    contract: str = LEGACY_DATA_CONTRACT
    manifest_path: str | None = None
    source_names: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DataConfig":
        contract = str(value.get("contract", LEGACY_DATA_CONTRACT))
        result = cls(
            train_path=str(_required(value, "train_path")),
            validation_path=str(_required(value, "validation_path")),
            expected_train_rows=int(_required(value, "expected_train_rows")),
            expected_train_puzzles=int(_required(value, "expected_train_puzzles")),
            expected_train_events=int(_required(value, "expected_train_events")),
            max_sequence_length=int(_required(value, "max_sequence_length")),
            seed=int(_required(value, "seed")),
            contract=contract,
            manifest_path=(
                str(value["manifest_path"])
                if value.get("manifest_path") is not None
                else None
            ),
            source_names=tuple(str(item) for item in value.get("source_names", [])),
        )
        actual = (
            result.expected_train_rows,
            result.expected_train_puzzles,
            result.expected_train_events,
            result.max_sequence_length,
            result.seed,
        )
        contracts = {
            LEGACY_DATA_CONTRACT: {
                "values": (102_392, 400, 665_371, 16_384, 24),
                "suffix": "/grids_v15/rearc",
                "manifest_required": False,
            },
            REARC8_DATA_CONTRACT: {
                "values": (13_523, 800, 48_692, 16_384, 24),
                "suffix": "/rearc8_union_fast128_v1/hf/train",
                "manifest_required": True,
                "source_names": (),
            },
            REARC8_SEED24_DATA_CONTRACT: {
                "values": (3_200, 400, 20_826, 16_384, 24),
                "suffix": "/rearc8_union_fast128_v1/hf/train",
                "manifest_required": True,
                "source_names": (REARC8_SEED24_SOURCE,),
            },
            REARC8_PUBLIC_EVAL_DATA_CONTRACT: {
                "values": (7_123, 400, 7_123, 16_384, 24),
                "suffix": "/rearc8_union_fast128_v1/hf/train",
                "manifest_required": True,
                "source_names": (
                    REARC8_PUBLIC_CANONICAL_SOURCE,
                    REARC8_PUBLIC_AUTHOR_VIEW_SOURCE,
                ),
            },
            OUTSIDE_UNION_DATA_CONTRACT: {
                "values": (3_200, 400, 20_810, 16_384, 24),
                "suffix": "/rearc128_outside_delta4_b0_b1_union_v1/hf/train",
                "manifest_required": True,
                "source_names": (),
            },
            OUTSIDE_B0_DATA_CONTRACT: {
                "values": (1_600, 400, 10_430, 16_384, 24),
                "suffix": "/rearc128_outside_delta4_s42/hf/outside_r128_b0",
                "manifest_required": True,
                "source_names": (OUTSIDE_B0_SOURCE,),
            },
            OUTSIDE_B1_DATA_CONTRACT: {
                "values": (1_600, 400, 10_380, 16_384, 24),
                "suffix": "/rearc128_outside_delta4_s42/hf/outside_r128_b1",
                "manifest_required": True,
                "source_names": (OUTSIDE_B1_SOURCE,),
            },
        }
        if contract not in contracts:
            raise ValueError(f"Unsupported Stage-B data contract: {contract}")
        contract_value = contracts[contract]
        expected = contract_value["values"]
        if actual != expected:
            raise ValueError(
                f"{contract} Stage-B data contract changed: {actual} != {expected}"
            )
        suffix = str(contract_value["suffix"])
        if not result.train_path.replace("\\", "/").endswith(suffix):
            raise ValueError(
                f"Stage-B training path for {contract} must end with {suffix}"
            )
        if bool(contract_value["manifest_required"]) != bool(result.manifest_path):
            requirement = (
                "required" if contract_value["manifest_required"] else "forbidden"
            )
            raise ValueError(
                f"Stage-B manifest_path is {requirement} for {contract}"
            )
        if result.source_names != contract_value.get("source_names", ()):
            raise ValueError(
                f"Stage-B source_names for {contract} changed: "
                f"{result.source_names} != {contract_value.get('source_names', ())}"
            )
        return result


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int
    batch_candidates: tuple[int, ...]
    dtype: str
    attention_backend: str
    decoder: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationConfig":
        result = cls(
            max_new_tokens=int(_required(value, "max_new_tokens")),
            batch_candidates=tuple(
                int(item) for item in _required(value, "batch_candidates")
            ),
            dtype=str(_required(value, "dtype")),
            attention_backend=str(_required(value, "attention_backend")),
            decoder=str(_required(value, "decoder")),
        )
        if result.max_new_tokens != 1024:
            raise ValueError("Stage-B generation must keep max_new_tokens=1024")
        _strict_tuple(result.batch_candidates, (1, 4, 8, 16), "generation batches")
        if (result.dtype, result.attention_backend, result.decoder) != (
            "bfloat16",
            "flash_attention_2",
            "constrained_greedy_full_softmax_v1",
        ):
            raise ValueError("Stage-B generation contract changed")
        return result


@dataclass(frozen=True)
class AttributionConfig:
    operator: str
    reference_operator: str
    backend: str
    reference_backend: str
    feature_budget_candidates: tuple[int, ...]
    node_threshold_candidates: tuple[float, ...]
    edge_threshold_candidates: tuple[float, ...]
    row_batch_size: int
    attribute_update_interval: int
    path_epsilon: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AttributionConfig":
        result = cls(
            operator=str(_required(value, "operator")),
            reference_operator=str(_required(value, "reference_operator")),
            backend=str(_required(value, "backend")),
            reference_backend=str(_required(value, "reference_backend")),
            feature_budget_candidates=tuple(
                int(item) for item in _required(value, "feature_budget_candidates")
            ),
            node_threshold_candidates=tuple(
                float(item) for item in _required(value, "node_threshold_candidates")
            ),
            edge_threshold_candidates=tuple(
                float(item) for item in _required(value, "edge_threshold_candidates")
            ),
            row_batch_size=int(_required(value, "row_batch_size")),
            attribute_update_interval=int(
                _required(value, "attribute_update_interval")
            ),
            path_epsilon=float(_required(value, "path_epsilon")),
        )
        identities = (
            result.operator,
            result.reference_operator,
            result.reference_backend,
        )
        if identities != (
            "joint_signed_span",
            "tokenwise_nonnegative_sum",
            "codecircuit_autograd_reference_v1",
        ) or result.backend not in {
            "streaming_exact_v1",
            "streaming_cached_exact_v1",
            "streaming_cached_exact_v2",
            "streaming_cached_pipeline_exact_v1",
            "streaming_cached_pipeline_hotloop_exact_v1",
            "streaming_cached_dual_event_exact_v1",
            "streaming_cached_pipeline_vjp_exact_v1",
            "streaming_cached_dual_event_vjp_exact_v1",
            "streaming_cached_pipeline_vjp_hotloop_exact_v2",
            "streaming_cached_dual_event_vjp_hotloop_exact_v2",
        }:
            raise ValueError(
                "Stage-B attribution identity changed: "
                f"{(*identities, result.backend)}"
            )
        _strict_tuple(
            result.feature_budget_candidates,
            (8192, 16384, 32768),
            "feature budgets",
        )
        _strict_tuple(
            result.node_threshold_candidates,
            (0.8, 0.9, 0.95),
            "node thresholds",
        )
        _strict_tuple(
            result.edge_threshold_candidates,
            (0.98, 0.99, 0.995),
            "edge thresholds",
        )
        if result.row_batch_size <= 0 or result.attribute_update_interval <= 0:
            raise ValueError("Attribution batching values must be positive")
        if not 0.0 < result.path_epsilon < 1.0:
            raise ValueError("path_epsilon must lie strictly between zero and one")
        return result


@dataclass(frozen=True)
class AuditConfig:
    master_events: int
    parity_events: int
    escalation_events: int
    fidelity_events: int
    intervention_fraction: float
    random_controls: int
    noninferiority_ratio: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AuditConfig":
        result = cls(
            **{
                "master_events": int(_required(value, "master_events")),
                "parity_events": int(_required(value, "parity_events")),
                "escalation_events": int(_required(value, "escalation_events")),
                "fidelity_events": int(_required(value, "fidelity_events")),
                "intervention_fraction": float(
                    _required(value, "intervention_fraction")
                ),
                "random_controls": int(_required(value, "random_controls")),
                "noninferiority_ratio": float(_required(value, "noninferiority_ratio")),
            }
        )
        if asdict(result) != {
            "master_events": 64,
            "parity_events": 12,
            "escalation_events": 256,
            "fidelity_events": 64,
            "intervention_fraction": 0.05,
            "random_controls": 3,
            "noninferiority_ratio": 0.9,
        }:
            raise ValueError("Stage-B audit contract changed")
        return result


@dataclass(frozen=True)
class ArtifactConfig:
    output_root: str
    num_work_units: int
    payload_target_bytes: int
    final_events_per_shard: int
    max_trails: int
    payload_schema: str
    vector_schema: str
    trail_schema: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactConfig":
        result = cls(
            output_root=str(_required(value, "output_root")),
            num_work_units=int(_required(value, "num_work_units")),
            payload_target_bytes=int(_required(value, "payload_target_bytes")),
            final_events_per_shard=int(_required(value, "final_events_per_shard")),
            max_trails=int(_required(value, "max_trails")),
            payload_schema=str(_required(value, "payload_schema")),
            vector_schema=str(_required(value, "vector_schema")),
            trail_schema=str(_required(value, "trail_schema")),
        )
        if result.num_work_units <= 0 or result.payload_target_bytes <= 0:
            raise ValueError("Artifact work-unit and byte limits must be positive")
        if result.final_events_per_shard != 4096 or result.max_trails != 8:
            raise ValueError("Final shard/trail contract changed")
        expected_schemas = (
            "nvarc_qwen3_joint_span_compact_v1",
            "nvarc_qwen3_36l_joint_signed_span_mean_59d_v1",
            "nvarc_qwen3_joint_span_trails_v1",
        )
        if (
            result.payload_schema,
            result.vector_schema,
            result.trail_schema,
        ) != expected_schemas:
            raise ValueError("Stage-B artifact schema changed")
        return result


@dataclass(frozen=True)
class NvarcCircuitConfig:
    format_version: str
    stage_a: StageAConfig
    data: DataConfig
    generation: GenerationConfig
    attribution: AttributionConfig
    audit: AuditConfig
    artifacts: ArtifactConfig
    stage_a_config: NvarcTranscoderConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NvarcCircuitConfig":
        expanded = _expand(dict(value))
        if str(_required(expanded, "format_version")) != FORMAT_VERSION:
            raise ValueError("Unsupported NVARC Stage-B configuration format")
        stage_a = StageAConfig.from_mapping(_required(expanded, "stage_a"))
        stage_a_config = load_stage_a_config(stage_a.profile)
        result = cls(
            format_version=FORMAT_VERSION,
            stage_a=stage_a,
            data=DataConfig.from_mapping(_required(expanded, "data")),
            generation=GenerationConfig.from_mapping(_required(expanded, "generation")),
            attribution=AttributionConfig.from_mapping(
                _required(expanded, "attribution")
            ),
            audit=AuditConfig.from_mapping(_required(expanded, "audit")),
            artifacts=ArtifactConfig.from_mapping(_required(expanded, "artifacts")),
            stage_a_config=stage_a_config,
        )
        result._validate_stage_a_alignment()
        return result

    def _validate_stage_a_alignment(self) -> None:
        stage_a = self.stage_a_config
        checks = {
            "validation_path": (
                self.data.validation_path,
                stage_a.data.validation_path,
            ),
            "n_ctx": (self.data.max_sequence_length, stage_a.data.max_sequence_length),
            "max_new_tokens": (
                self.generation.max_new_tokens,
                stage_a.traces.max_new_tokens,
            ),
        }
        if self.data.contract not in OUTSIDE_SUBSET_DATA_CONTRACTS:
            checks["train_path"] = (self.data.train_path, stage_a.data.train_path)
        mismatches = {
            key: {"stage_b": left, "stage_a": right}
            for key, (left, right) in checks.items()
            if left != right
        }
        if mismatches:
            raise ValueError(f"Stage-A/Stage-B contract mismatch: {mismatches}")
        if self.data.contract in SOURCE_FILTERED_DATA_CONTRACTS:
            expected_stage_a = (
                (13_523, 48_692)
                if self.data.contract in REARC8_SUBSET_DATA_CONTRACTS
                else (3_200, 20_810)
            )
            actual_stage_a = (
                stage_a.data.expected_train_rows,
                stage_a.data.expected_train_events,
            )
            if actual_stage_a != expected_stage_a:
                raise ValueError(
                    "Subset Stage-B requires its frozen full-union "
                    f"Stage-A bank: {actual_stage_a} != {expected_stage_a}"
                )
        else:
            expected_data = (
                self.data.expected_train_rows,
                self.data.expected_train_events,
            )
            actual_stage_a = (
                stage_a.data.expected_train_rows,
                stage_a.data.expected_train_events,
            )
            if expected_data != actual_stage_a:
                raise ValueError(
                    "Stage-A/Stage-B row-event contract mismatch: "
                    f"{expected_data} != {actual_stage_a}"
                )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("stage_a_config")
        if self.data.contract == LEGACY_DATA_CONTRACT:
            # Keep every existing step399 policy/config identity byte-for-byte stable.
            value["data"].pop("contract")
            value["data"].pop("manifest_path")
        if self.data.contract not in SOURCE_FILTERED_DATA_CONTRACTS:
            # Preserve every existing config and policy identity.
            value["data"].pop("source_names")
        root_value = os.environ.get("HC_ROOT")
        if not root_value:
            raise ValueError("HC_ROOT is required for a mount-stable Stage-B identity")
        current_root = Path(root_value).resolve()
        identity_root = Path("${HC_ROOT}")
        value["stage_a_config_sha256"] = config_semantic_sha256_for_root(
            self.stage_a_config,
            current_root=current_root,
            identity_root=identity_root,
        )

        def rebase(item: Any) -> Any:
            if isinstance(item, dict):
                return {key: rebase(child) for key, child in item.items()}
            if isinstance(item, list):
                return [rebase(child) for child in item]
            if isinstance(item, tuple):
                return tuple(rebase(child) for child in item)
            if not isinstance(item, str):
                return item
            path = Path(item)
            if not path.is_absolute():
                return item
            try:
                relative = path.relative_to(current_root)
            except ValueError:
                try:
                    relative = path.resolve().relative_to(current_root)
                except ValueError:
                    return item
            return str(identity_root).rstrip("/\\") + "/" + relative.as_posix()

        return rebase(value)

    @property
    def semantic_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_config(path: str | Path) -> NvarcCircuitConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Stage-B config must contain a mapping")
    return NvarcCircuitConfig.from_mapping(value)

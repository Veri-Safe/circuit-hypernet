from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .contracts import (
    CONFIG_FORMAT,
    EXPECTED_BASE_AGGREGATE_SHA256,
    EXPECTED_EVENTS,
    EXPECTED_GLOBAL_ROW_BATCH,
    EXPECTED_LAYERS,
    EXPECTED_MAX_STEPS,
    EXPECTED_PUZZLES,
    EXPECTED_ROWS,
    EXPECTED_S2_PROFILE,
    EXPECTED_SCHEDULE_SHA256,
    FUSION_VARIANTS,
    expand_environment,
    semantic_sha256,
)


def workspace_relative_path(value: str | Path) -> str:
    """Record the configured path relative to the launcher's working directory."""
    return Path(os.path.relpath(value)).as_posix()


def _required(value: Mapping[str, Any], key: str) -> Any:
    if key not in value:
        raise ValueError(f"Missing required S4 config field: {key}")
    return value[key]


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if key == "common_config":
            continue
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class ArtifactBankConfig:
    catalog_root: str
    final_root: str
    root_manifest: str
    root_manifest_sha256: str
    expected_events: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ArtifactBankConfig":
        from s2_hypernet.nvarc_pg.config import workspace_path

        return cls(
            catalog_root=workspace_path(str(_required(value, "catalog_root"))),
            final_root=workspace_path(str(_required(value, "final_root"))),
            root_manifest=workspace_path(str(_required(value, "root_manifest"))),
            root_manifest_sha256=str(_required(value, "root_manifest_sha256")),
            expected_events=int(_required(value, "expected_events")),
        )


@dataclass(frozen=True)
class ContinuationConfig:
    """Explicit optimizer-preserving fork from a completed S4 checkpoint."""

    source_checkpoint: str
    source_config: str
    source_step: int
    scheduler: str
    trunk_lr_scale: float
    circuit_lr_scale: float

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "ContinuationConfig | None":
        from s2_hypernet.nvarc_pg.config import workspace_path

        if value is None:
            return None
        return cls(
            source_checkpoint=workspace_path(str(_required(value, "source_checkpoint"))),
            source_config=workspace_path(str(_required(value, "source_config"))),
            source_step=int(_required(value, "source_step")),
            scheduler=str(_required(value, "scheduler")),
            trunk_lr_scale=float(_required(value, "trunk_lr_scale")),
            circuit_lr_scale=float(_required(value, "circuit_lr_scale")),
        )

    def validate(self) -> None:
        if self.source_step <= 0:
            raise ValueError("S4 continuation source step must be positive")
        if self.scheduler != "constant_lr_v1":
            raise ValueError("S4 continuation requires constant_lr_v1")
        for name, value in (
            ("trunk_lr_scale", self.trunk_lr_scale),
            ("circuit_lr_scale", self.circuit_lr_scale),
        ):
            if value not in {0.1, 0.25, 0.5, 1.0}:
                raise ValueError(f"S4 continuation {name} must be 0.1, 0.25, 0.5, or 1")


@dataclass(frozen=True)
class CircuitTrainingConfig:
    """Optional circuit-specific objective applied in addition to matched CE."""

    negative_mode: str = "none"
    negative_fraction: float = 0.0
    negative_coefficient: float = 0.0
    negative_margin: float = 0.0
    negative_seed: int = 42

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "CircuitTrainingConfig":
        source = value or {}
        return cls(
            negative_mode=str(source.get("negative_mode", "none")),
            negative_fraction=float(source.get("negative_fraction", 0.0)),
            negative_coefficient=float(source.get("negative_coefficient", 0.0)),
            negative_margin=float(source.get("negative_margin", 0.0)),
            negative_seed=int(source.get("negative_seed", 42)),
        )

    def validate(self) -> None:
        if self.negative_mode == "none":
            if any(
                value != 0.0
                for value in (
                    self.negative_fraction,
                    self.negative_coefficient,
                    self.negative_margin,
                )
            ):
                raise ValueError("Disabled S4 circuit negatives must have zero settings")
            return
        if self.negative_mode != "deterministic_shuffled_one_sided_margin_v1":
            raise ValueError("Unsupported S4 circuit negative mode")
        if self.negative_fraction != 0.25:
            raise ValueError("S4 shuffled-margin fraction must be 0.25")
        if self.negative_coefficient != 0.1:
            raise ValueError("S4 shuffled-margin coefficient must be 0.1")
        if self.negative_margin != 0.01:
            raise ValueError("S4 shuffled-margin loss margin must be 0.01")
        if self.negative_seed != 42:
            raise ValueError("S4 shuffled-margin seed must remain 42")

    @property
    def is_default(self) -> bool:
        return self == CircuitTrainingConfig()


@dataclass(frozen=True)
class OptimizationConfig:
    global_lr_multiplier: float = 1.0
    circuit_lr_multiplier: float = 1.0
    factor_l2: float = 0.0
    effective_delta_l2: float = 0.0
    lora_gate_max: float = 1.0
    lora_gate_warmup_steps: int = 0
    trainable_global_lora_scale: bool = False
    global_lora_scale_initial: float = 1.0
    global_lora_scale_max: float = 1.5
    base_preservation_coefficient: float = 0.0
    freeze_trunk: bool = False

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "OptimizationConfig":
        source = value or {}
        return cls(
            global_lr_multiplier=float(source.get("global_lr_multiplier", 1.0)),
            circuit_lr_multiplier=float(source.get("circuit_lr_multiplier", 1.0)),
            factor_l2=float(source.get("factor_l2", 0.0)),
            effective_delta_l2=float(source.get("effective_delta_l2", 0.0)),
            lora_gate_max=float(source.get("lora_gate_max", 1.0)),
            lora_gate_warmup_steps=int(source.get("lora_gate_warmup_steps", 0)),
            trainable_global_lora_scale=bool(
                source.get("trainable_global_lora_scale", False)
            ),
            global_lora_scale_initial=float(
                source.get("global_lora_scale_initial", 1.0)
            ),
            global_lora_scale_max=float(source.get("global_lora_scale_max", 1.5)),
            base_preservation_coefficient=float(
                source.get("base_preservation_coefficient", 0.0)
            ),
            freeze_trunk=bool(source.get("freeze_trunk", False)),
        )

    def validate(self) -> None:
        if self.global_lr_multiplier not in {
            0.1,
            0.25,
            0.45,
            0.5,
            0.55,
            1.0,
            2.0,
        }:
            raise ValueError("Unsupported S4 global LR multiplier")
        if self.circuit_lr_multiplier not in {1.0, 1.8, 2.0, 2.2, 4.0}:
            raise ValueError(
                "S4 R1 circuit LR multiplier must be 1, 1.8, 2, 2.2, or 4"
            )
        if self.factor_l2 not in {0.0, 1.0e-4}:
            raise ValueError("S4 R1 factor L2 must be 0 or 1e-4")
        if self.effective_delta_l2 not in {0.0, 1.0e-4, 2.0e-4, 3.0e-4, 1.0e-3}:
            raise ValueError(
                "S4 R1 effective delta L2 must be 0, 1e-4, 3e-4, or 1e-3"
            )
        if self.factor_l2 and self.effective_delta_l2:
            raise ValueError(
                "S4 R1 factor L2 and effective delta L2 must be isolated"
            )
        if self.lora_gate_max not in {0.125, 0.1875, 0.21875, 0.25, 0.5, 1.0}:
            raise ValueError(
                "S4 R1 LoRA gate max must be 0.125, 0.1875, 0.25, 0.5, or 1"
            )
        if self.lora_gate_warmup_steps < 0:
            raise ValueError("S4 R1 LoRA gate warmup must be non-negative")
        if self.lora_gate_warmup_steps == 0 and self.lora_gate_max != 1.0:
            raise ValueError("A sub-unit S4 LoRA gate requires a positive warmup")
        if not 0.0 < self.global_lora_scale_initial < self.global_lora_scale_max:
            raise ValueError(
                "S4 global LoRA scale initial value must be inside (0, max)"
            )
        if not self.trainable_global_lora_scale and (
            self.global_lora_scale_initial != 1.0
            or self.global_lora_scale_max != 1.5
        ):
            raise ValueError(
                "Fixed-scale S4 configs cannot declare trainable-scale parameters"
            )
        if self.base_preservation_coefficient not in {0.0, 0.25, 0.5, 0.75}:
            raise ValueError(
                "S4 base-preservation coefficient must be 0, 0.25, 0.5, or 0.75"
            )

    def lora_gate_at_step(self, step: int) -> float:
        if step < 0:
            raise ValueError("S4 LoRA gate step must be non-negative")
        if self.lora_gate_warmup_steps == 0:
            return self.lora_gate_max
        progress = min(float(step) / float(self.lora_gate_warmup_steps), 1.0)
        return self.lora_gate_max * progress

    @property
    def semantic_dict(self) -> dict[str, float | int]:
        value: dict[str, float | int] = {
            "global_lr_multiplier": self.global_lr_multiplier,
            "circuit_lr_multiplier": self.circuit_lr_multiplier,
            "factor_l2": self.factor_l2,
        }
        if self.effective_delta_l2:
            value["effective_delta_l2"] = self.effective_delta_l2
        if self.lora_gate_max != 1.0:
            value["lora_gate_max"] = self.lora_gate_max
        if self.lora_gate_warmup_steps:
            value["lora_gate_warmup_steps"] = self.lora_gate_warmup_steps
        if self.trainable_global_lora_scale:
            value["trainable_global_lora_scale"] = True
            value["global_lora_scale_initial"] = self.global_lora_scale_initial
            value["global_lora_scale_max"] = self.global_lora_scale_max
        if self.base_preservation_coefficient:
            value["base_preservation_coefficient"] = (
                self.base_preservation_coefficient
            )
        if self.freeze_trunk:
            value["freeze_trunk"] = True
        return value

    @property
    def is_default(self) -> bool:
        return self == OptimizationConfig()


@dataclass(frozen=True)
class RuntimeOptimizationConfig:
    """Explicit, auditable runtime choices that may affect training numerics."""

    loss_backend: str = "full_logits_v1"
    batching_mode: str = "fixed_rows_v1"
    max_micro_rows: int = 1
    padded_token_budget: int = 0
    pg_gradient_checkpointing: bool = True

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "RuntimeOptimizationConfig":
        source = value or {}
        return cls(
            loss_backend=str(source.get("loss_backend", "full_logits_v1")),
            batching_mode=str(source.get("batching_mode", "fixed_rows_v1")),
            max_micro_rows=int(source.get("max_micro_rows", 1)),
            padded_token_budget=int(source.get("padded_token_budget", 0)),
            pg_gradient_checkpointing=bool(
                source.get("pg_gradient_checkpointing", True)
            ),
        )

    def validate(self, *, canonical_max_micro_rows: int) -> None:
        if self.loss_backend not in {
            "full_logits_v1",
            "response_only_logits_v1",
        }:
            raise ValueError(f"Unsupported S4 loss backend: {self.loss_backend}")
        if self.batching_mode not in {"fixed_rows_v1", "token_budget_v1"}:
            raise ValueError(f"Unsupported S4 batching mode: {self.batching_mode}")
        if not 1 <= self.max_micro_rows <= canonical_max_micro_rows:
            raise ValueError(
                "S4 runtime max_micro_rows must be between 1 and the canonical "
                f"micro row batch ({canonical_max_micro_rows})"
            )
        if self.batching_mode == "fixed_rows_v1":
            if self.max_micro_rows != 1 or self.padded_token_budget != 0:
                raise ValueError(
                    "Fixed-row batching must keep max_micro_rows=1 and "
                    "padded_token_budget=0"
                )
        elif self.max_micro_rows <= 1 or self.padded_token_budget <= 0:
            raise ValueError(
                "Token-budget batching requires max_micro_rows>1 and a positive "
                "padded_token_budget"
            )

    @property
    def is_default(self) -> bool:
        return self == RuntimeOptimizationConfig()


@dataclass(frozen=True)
class LossProtocolConfig:
    """Select which assistant events contribute to the S4 objective."""

    name: str = "all_assistant_events_v1"
    supervised_assistant_start_index: int = 0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "LossProtocolConfig":
        source = value or {}
        return cls(
            name=str(source.get("name", "all_assistant_events_v1")),
            supervised_assistant_start_index=int(
                source.get("supervised_assistant_start_index", 0)
            ),
        )

    def validate(self) -> None:
        supported = {
            "all_assistant_events_v1": 0,
            "first_pair_demonstration_only_v1": 1,
        }
        expected_start = supported.get(self.name)
        if expected_start is None:
            raise ValueError(f"Unsupported S4 loss protocol: {self.name}")
        if self.supervised_assistant_start_index != expected_start:
            raise ValueError(
                "S4 loss protocol name/start index mismatch: "
                f"{self.name} requires {expected_start}"
            )

    @property
    def is_default(self) -> bool:
        return self == LossProtocolConfig()


@dataclass(frozen=True)
class ObjectiveWeightingConfig:
    """Weight supervised assistant events without changing their labels."""

    mode: str = "token_balanced_v1"
    expected_selected_events: int = 17_626
    depth_min_weight: float = 1.0
    depth_max_weight: float = 1.0

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
    ) -> "ObjectiveWeightingConfig":
        source = value or {}
        return cls(
            mode=str(source.get("mode", "token_balanced_v1")),
            expected_selected_events=int(
                source.get("expected_selected_events", 17_626)
            ),
            depth_min_weight=float(source.get("depth_min_weight", 1.0)),
            depth_max_weight=float(source.get("depth_max_weight", 1.0)),
        )

    def validate(self) -> None:
        expected_counts = {
            "token_balanced_v1": 17_626,
            "event_balanced_v1": 17_626,
            "depth_weighted_event_v1": 17_626,
            "last_two_event_balanced_v1": 6_400,
        }
        expected = expected_counts.get(self.mode)
        if expected is None:
            raise ValueError(f"Unsupported S4 objective weighting: {self.mode}")
        if self.expected_selected_events != expected:
            raise ValueError(
                "S4 objective selected-event count changed: "
                f"{self.expected_selected_events} != {expected}"
            )
        if self.mode == "depth_weighted_event_v1":
            if (self.depth_min_weight, self.depth_max_weight) != (1.0, 2.0):
                raise ValueError("S4 depth weighting must linearly span 1.0 to 2.0")
        elif (self.depth_min_weight, self.depth_max_weight) != (1.0, 1.0):
            raise ValueError("Non-depth S4 objectives cannot declare depth weights")

    @property
    def is_default(self) -> bool:
        return self == ObjectiveWeightingConfig()


@dataclass(frozen=True)
class ReplicationConfig:
    """Opt-in ARC1 repetitions without changing the upstream artifact identity."""

    protocol: str
    seed: int
    data_seed: int
    world_size: int
    full_schedule_sha256: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any] | None
    ) -> "ReplicationConfig | None":
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ValueError("S4 replication must be a mapping")
        fields = {
            "protocol", "seed", "data_seed", "world_size", "full_schedule_sha256"
        }
        if set(value) != fields:
            raise ValueError("S4 replication requires exactly the declared protocol fields")
        result = cls(
            protocol=value["protocol"],
            seed=value["seed"],
            data_seed=value["data_seed"],
            world_size=value["world_size"],
            full_schedule_sha256=value["full_schedule_sha256"],
        )
        result.validate()
        return result

    def validate(self) -> None:
        for field in ("seed", "data_seed", "world_size"):
            if type(getattr(self, field)) is not int:
                raise ValueError(f"S4 replication {field} must be an integer without coercion")
        for field in ("protocol", "full_schedule_sha256"):
            if type(getattr(self, field)) is not str:
                raise ValueError(f"S4 replication {field} must be a string without coercion")
        if self.protocol != "arc1_paired_multiseed_v1":
            raise ValueError("Unsupported S4 replication protocol")
        if self.seed not in range(42, 47) or self.data_seed != self.seed:
            raise ValueError("S4 paired replication requires matching seeds 42 through 46")
        if self.world_size <= 0:
            raise ValueError("S4 paired replication requires a positive process count")
        if len(self.full_schedule_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.full_schedule_sha256
        ):
            raise ValueError("S4 replication requires a lowercase full schedule SHA256")
        if self.seed == 42 and self.full_schedule_sha256 != EXPECTED_SCHEDULE_SHA256:
            raise ValueError("S4 seed42 replication must retain the original row schedule")
        if self.seed != 42 and self.full_schedule_sha256 == EXPECTED_SCHEDULE_SHA256:
            raise ValueError("S4 fresh data seeds cannot reuse the seed42 row schedule")


@dataclass(frozen=True)
class S4R1Config:
    format_version: str
    variant_id: str
    s2_config: str
    base_aggregate_sha256: str
    vector_schema: str
    vector_dim: int
    train_artifacts: ArtifactBankConfig
    eval_artifacts: ArtifactBankConfig
    output_root: str
    fresh_init_root: str
    batch_manifest: str
    fusion_hidden_dim: int
    circuit_encoder_hidden_dim: int
    fusion_dropout: float
    circuit_input_mode: str
    optimization: OptimizationConfig
    runtime_optimization: RuntimeOptimizationConfig
    loss_protocol: LossProtocolConfig
    objective_weighting: ObjectiveWeightingConfig
    circuit_training: CircuitTrainingConfig
    continuation: ContinuationConfig | None
    seed: int
    source_path: str
    training_micro_row_batch_override: int | None
    training_max_grad_norm_override: float | None
    diagnostic_logical_step_indices: tuple[int, ...]
    replication: ReplicationConfig | None = None

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        source_path: str | Path,
    ) -> "S4R1Config":
        from s2_hypernet.nvarc_pg.config import workspace_path

        artifacts = _required(value, "artifacts")
        fusion = _required(value, "fusion")
        result = cls(
            format_version=str(_required(value, "format_version")),
            variant_id=str(_required(value, "variant_id")),
            s2_config=workspace_path(str(_required(value, "s2_config"))),
            base_aggregate_sha256=str(_required(value, "base_aggregate_sha256")),
            vector_schema=str(_required(value, "vector_schema")),
            vector_dim=int(_required(value, "vector_dim")),
            train_artifacts=ArtifactBankConfig.from_mapping(
                _required(artifacts, "train")
            ),
            eval_artifacts=ArtifactBankConfig.from_mapping(
                _required(artifacts, "evaluation")
            ),
            output_root=workspace_path(str(_required(value, "output_root"))),
            fresh_init_root=workspace_path(str(_required(value, "fresh_init_root"))),
            batch_manifest=workspace_path(str(_required(value, "batch_manifest"))),
            fusion_hidden_dim=int(_required(fusion, "pg_hidden_dim")),
            circuit_encoder_hidden_dim=int(_required(fusion, "encoder_hidden_dim")),
            fusion_dropout=float(_required(fusion, "dropout")),
            circuit_input_mode=str(fusion.get("input_mode", "raw_v1")),
            optimization=OptimizationConfig.from_mapping(value.get("optimization")),
            runtime_optimization=RuntimeOptimizationConfig.from_mapping(
                value.get("runtime_optimization")
            ),
            loss_protocol=LossProtocolConfig.from_mapping(value.get("loss_protocol")),
            objective_weighting=ObjectiveWeightingConfig.from_mapping(
                value.get("objective_weighting")
            ),
            circuit_training=CircuitTrainingConfig.from_mapping(
                value.get("circuit_training")
            ),
            continuation=ContinuationConfig.from_mapping(value.get("continuation")),
            seed=int(_required(value, "seed")),
            source_path=str(Path(source_path)),
            training_micro_row_batch_override=(
                None
                if value.get("training_micro_row_batch_override") is None
                else int(value["training_micro_row_batch_override"])
            ),
            training_max_grad_norm_override=(
                None
                if value.get("training_max_grad_norm_override") is None
                else float(value["training_max_grad_norm_override"])
            ),
            diagnostic_logical_step_indices=tuple(
                int(step) for step in value.get("diagnostic_logical_step_indices", ())
            ),
            replication=ReplicationConfig.from_mapping(value.get("replication")),
        )
        result.validate()
        return result

    def load_s2_config(self, *, apply_training_override: bool = False) -> Any:
        """Load the canonical S2 config with declared S4 training-only overrides.

        The S2 config path is part of the immutable S4 artifact contract.  A
        runtime/training adjustment therefore belongs to the S4 layer and must
        not be represented by a different S2 config path.
        """

        from dataclasses import replace

        from s2_hypernet.nvarc_pg.config import load_experiment_config

        s2 = load_experiment_config(self.s2_config)
        if not apply_training_override:
            return s2
        training_overrides: dict[str, int | float] = {}
        if self.replication is not None:
            training_overrides["world_size"] = self.replication.world_size
        if self.training_micro_row_batch_override is not None:
            training_overrides["micro_row_batch"] = (
                self.training_micro_row_batch_override
            )
        if self.training_max_grad_norm_override is not None:
            training_overrides["max_grad_norm"] = (
                self.training_max_grad_norm_override
            )
        if not training_overrides:
            return s2
        return replace(
            s2,
            training=replace(s2.training, **training_overrides),
        )

    @property
    def effective_world_size(self) -> int:
        if self.replication is not None:
            return self.replication.world_size
        return self.load_s2_config().training.world_size

    @property
    def expected_schedule_sha256(self) -> str:
        if self.replication is not None:
            return self.replication.full_schedule_sha256
        return EXPECTED_SCHEDULE_SHA256

    @property
    def effective_data_seed(self) -> int:
        if self.replication is not None:
            return self.replication.data_seed
        return self.load_s2_config().data.seed

    def logical_step_for_active_step(self, active_step: int) -> int:
        """Map a checkpoint/training step to its immutable logical schedule step.

        Warm-start diagnostics use a short active checkpoint sequence while
        consuming rows from a later point in the canonical schedule.  Gate
        reconstruction must use that same logical step; using the short
        checkpoint number would replay the gate warmup during evaluation.
        """

        active_step = int(active_step)
        if active_step <= 0:
            raise ValueError("S4 active step must be positive")
        if self.diagnostic_logical_step_indices:
            if active_step > len(self.diagnostic_logical_step_indices):
                raise ValueError(
                    "S4 active step exceeds the diagnostic logical schedule"
                )
            return self.diagnostic_logical_step_indices[active_step - 1]
        return active_step

    def validate(self) -> None:
        if self.format_version != CONFIG_FORMAT:
            raise ValueError(f"Unsupported S4 config format: {self.format_version}")
        if self.variant_id not in FUSION_VARIANTS:
            raise ValueError(f"Unsupported S4 fusion variant: {self.variant_id}")
        if self.base_aggregate_sha256 != EXPECTED_BASE_AGGREGATE_SHA256:
            raise ValueError("S4 R1 base checkpoint identity changed")
        if self.vector_schema != "nvarc_qwen3_36l_joint_signed_span_mean_59d_v1":
            raise ValueError("S4 R1 vector schema changed")
        if self.vector_dim != 59:
            raise ValueError("S4 R1 requires raw 59D circuit vectors")
        if self.train_artifacts.expected_events != EXPECTED_EVENTS:
            raise ValueError("S4 R1 train circuit coverage changed")
        if self.eval_artifacts.expected_events != 7_123:
            raise ValueError("S4 R1 evaluation circuit coverage changed")
        if self.fusion_hidden_dim != 1_280:
            raise ValueError("S4 R1 PG hidden size must remain 1280")
        if self.circuit_encoder_hidden_dim != 128:
            raise ValueError("S4 R1 circuit encoder hidden size must remain 128")
        if self.fusion_dropout != 0.0:
            raise ValueError("S4 R1 circuit fusion dropout must remain zero")
        if self.circuit_input_mode not in {
            "raw_v1",
            "dual_raw_signed_log_layernorm_v1",
        }:
            raise ValueError("Unsupported S4 circuit input mode")
        self.optimization.validate()
        self.loss_protocol.validate()
        self.objective_weighting.validate()
        self.circuit_training.validate()
        if self.continuation is not None:
            self.continuation.validate()
            if self.optimization.is_default or self.optimization.freeze_trunk:
                raise ValueError(
                    "S4 continuation requires split, trainable trunk/circuit groups"
                )
            if self.diagnostic_logical_step_indices:
                raise ValueError(
                    "S4 continuation cannot use diagnostic logical step remapping"
                )
        if self.replication is not None:
            self.replication.validate()
            if self.seed != self.replication.seed:
                raise ValueError("S4 initialization seed does not match replication seed")
            if self.continuation is not None or self.diagnostic_logical_step_indices:
                raise ValueError("S4 paired replication requires a fresh unremapped run")
        elif self.seed != 42:
            raise ValueError("S4 R1 common seed must remain 42")

        s2 = self.load_s2_config()
        if self.replication is not None and (
            s2.data.seed != 42
        ):
            raise ValueError(
                "S4 replication must retain the canonical seed42 S2 artifact"
            )
        if self.optimization.lora_gate_warmup_steps > s2.training.max_steps:
            raise ValueError("S4 LoRA gate warmup exceeds the training schedule")
        self.runtime_optimization.validate(
            canonical_max_micro_rows=s2.training.micro_row_batch
        )
        mismatches = {}
        expected = {
            "profile_id": EXPECTED_S2_PROFILE,
            "base_aggregate_sha256": EXPECTED_BASE_AGGREGATE_SHA256,
            "train_rows": EXPECTED_ROWS,
            "train_puzzles": EXPECTED_PUZZLES,
            "expected_events": EXPECTED_EVENTS,
            "layers": EXPECTED_LAYERS,
            "max_steps": EXPECTED_MAX_STEPS,
            "global_row_batch": EXPECTED_GLOBAL_ROW_BATCH,
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
        }
        actual = {
            "profile_id": s2.profile.profile_id,
            "base_aggregate_sha256": s2.profile.hf_aggregate_sha256,
            "train_rows": s2.profile.train_rows,
            "train_puzzles": s2.profile.train_puzzles,
            "expected_events": s2.profile.expected_event_count,
            "layers": s2.target_layout.num_layers,
            "max_steps": s2.training.max_steps,
            "global_row_batch": s2.training.global_row_batch,
            "schedule_sha256": s2.profile.schedule_sha256,
        }
        for key, expected_value in expected.items():
            if actual[key] != expected_value:
                mismatches[key] = {"expected": expected_value, "actual": actual[key]}
        method_expected = {
            "input_dim": 2_560,
            "d_pg": 1_280,
            "blocks": 12,
            "head_dim": 128,
            "token_dim": 1_024,
            "dim_accumulation": 8,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.0,
        }
        for key, expected_value in method_expected.items():
            actual_value = getattr(s2.method, key)
            if actual_value != expected_value:
                mismatches[f"method.{key}"] = {
                    "expected": expected_value,
                    "actual": actual_value,
                }
        if self.training_micro_row_batch_override is not None:
            override = self.training_micro_row_batch_override
            if override <= 0:
                raise ValueError(
                    "S4 training micro row batch override must be positive"
                )
            if override > s2.training.micro_row_batch:
                raise ValueError(
                    "S4 training micro row batch override cannot exceed canonical micro row batch"
                )
            if s2.training.global_row_batch % (self.effective_world_size * override):
                raise ValueError(
                    "S4 training micro row batch override does not divide global row batch"
                )
        if self.training_max_grad_norm_override is not None:
            if self.training_max_grad_norm_override not in {0.25, 0.5, 0.75, 1.0}:
                raise ValueError(
                    "S4 training max-grad-norm override must be 0.25, 0.5, 0.75, or 1"
                )
        if self.diagnostic_logical_step_indices:
            if len(set(self.diagnostic_logical_step_indices)) != len(
                self.diagnostic_logical_step_indices
            ):
                raise ValueError("S4 diagnostic logical steps must be unique")
            if any(
                step <= 0 or step > s2.training.max_steps
                for step in self.diagnostic_logical_step_indices
            ):
                raise ValueError("S4 diagnostic logical step is outside training")
        if mismatches:
            raise ValueError(f"S4 R1 S2 contract changed: {mismatches}")

    @property
    def semantic_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("source_path", None)
        value.pop("optimization")
        if not self.optimization.is_default:
            value["optimization"] = self.optimization.semantic_dict
        runtime_optimization = value.pop("runtime_optimization")
        if not self.runtime_optimization.is_default:
            value["runtime_optimization"] = runtime_optimization
        loss_protocol = value.pop("loss_protocol")
        if not self.loss_protocol.is_default:
            value["loss_protocol"] = loss_protocol
        objective_weighting = value.pop("objective_weighting")
        if not self.objective_weighting.is_default:
            value["objective_weighting"] = objective_weighting
        circuit_training = value.pop("circuit_training")
        if not self.circuit_training.is_default:
            value["circuit_training"] = circuit_training
        circuit_input_mode = value.pop("circuit_input_mode")
        if circuit_input_mode != "raw_v1":
            value["circuit_input_mode"] = circuit_input_mode
        continuation = value.pop("continuation")
        if continuation is not None:
            value["continuation"] = continuation
        replication = value.pop("replication")
        if replication is not None:
            value["replication"] = replication
        diagnostic_steps = value.pop("diagnostic_logical_step_indices")
        if diagnostic_steps:
            value["diagnostic_logical_step_indices"] = diagnostic_steps
        max_grad_norm_override = value.pop("training_max_grad_norm_override")
        if max_grad_norm_override is not None:
            value["training_max_grad_norm_override"] = max_grad_norm_override
        return value

    @property
    def semantic_sha256(self) -> str:
        return semantic_sha256(self.semantic_dict)

    @property
    def training_semantic_dict(self) -> dict[str, Any]:
        """Identity of everything allowed to affect training checkpoints."""

        value = self.semantic_dict
        value.pop("eval_artifacts", None)
        return value

    @property
    def training_semantic_sha256(self) -> str:
        return semantic_sha256(self.training_semantic_dict)

    @property
    def expected_supervised_train_events(self) -> int:
        result = self.objective_weighting.expected_selected_events
        if result <= 0 or result > self.train_artifacts.expected_events:
            raise ValueError("S4 loss protocol leaves no expected supervised events")
        return result

    @property
    def artifact_contract_sha256(self) -> str:
        return semantic_sha256(
            {
                "format_version": self.format_version,
                "s2_config": self.s2_config,
                "base_aggregate_sha256": self.base_aggregate_sha256,
                "vector_schema": self.vector_schema,
                "vector_dim": self.vector_dim,
                "train_artifacts": asdict(self.train_artifacts),
            }
        )

    @property
    def eval_artifact_contract_sha256(self) -> str:
        return semantic_sha256(
            {
                "format_version": self.format_version,
                "s2_config": self.s2_config,
                "base_aggregate_sha256": self.base_aggregate_sha256,
                "vector_schema": self.vector_schema,
                "vector_dim": self.vector_dim,
                "eval_artifacts": asdict(self.eval_artifacts),
            }
        )


def load_config(path: str | Path) -> S4R1Config:
    config_path = Path(path).resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"S4 config must contain a mapping: {config_path}")
    common_value: Mapping[str, Any] = {}
    common_path = value.get("common_config")
    if common_path is not None:
        common_file = Path(str(common_path))
        if not common_file.is_absolute():
            common_file = config_path.parent / common_file
        loaded = yaml.safe_load(common_file.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"S4 common config must contain a mapping: {common_file}")
        common_value = loaded
    merged = expand_environment(_deep_merge(common_value, value))
    return S4R1Config.from_mapping(merged, source_path=config_path)

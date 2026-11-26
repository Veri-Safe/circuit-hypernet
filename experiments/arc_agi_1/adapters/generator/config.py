from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from s2_hypernet.config import ParameterGeneratorConfig

from .method_config import MethodConfig
from .profiles import ExperimentProfile, profile_for_format, profile_for_id
from .target_layout import QwenTargetLayout, build_qwen3_4b_target_layout
from .variants import ExperimentVariant, variant_for_id


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration field: {key}")
    return mapping[key]


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class BaseCheckpointConfig:
    runtime: str
    source_kind: str
    source_step: int
    hf_checkpoint: str
    tokenizer: str
    source_run_manifest: str = ""

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        profile: ExperimentProfile,
    ) -> "BaseCheckpointConfig":
        result = cls(
            runtime=str(_required(value, "runtime")),
            source_kind=str(_required(value, "source_kind")),
            source_step=int(_required(value, "source_step")),
            hf_checkpoint=str(_required(value, "hf_checkpoint")),
            tokenizer=str(_required(value, "tokenizer")),
            source_run_manifest=str(value.get("source_run_manifest", "")),
        )
        if result.runtime not in {"hf_only", "hf_separated"}:
            raise ValueError("NVARC PG requires an HF checkpoint runtime")
        if (
            result.source_kind != profile.source_kind
            or result.source_step != profile.source_step
        ):
            raise ValueError(
                f"{profile.profile_id} requires source_kind={profile.source_kind!r} "
                f"at step {profile.source_step}"
            )
        if profile.source_run_name is not None and not result.source_run_manifest:
            raise ValueError(f"{profile.profile_id} requires source_run_manifest")
        return result


@dataclass(frozen=True)
class DataConfig:
    train_paths: tuple[str, ...]
    dev_path: str
    split_manifest: str
    max_sequence_length: int
    expected_rows: int
    expected_puzzles: int
    seed: int
    audit_manifest: str = ""
    num_workers: int = 4

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        profile: ExperimentProfile,
    ) -> "DataConfig":
        if "train_paths" in value and "train_path" in value:
            raise ValueError("Set only one of train_path or train_paths")
        raw_train_paths = value.get("train_paths")
        if raw_train_paths is None:
            raw_train_paths = [_required(value, "train_path")]
        if not isinstance(raw_train_paths, (list, tuple)) or not raw_train_paths:
            raise ValueError("train_paths must be a non-empty sequence")
        result = cls(
            train_paths=tuple(str(path) for path in raw_train_paths),
            dev_path=str(value.get("dev_path", "")),
            split_manifest=str(_required(value, "split_manifest")),
            max_sequence_length=int(_required(value, "max_sequence_length")),
            expected_rows=int(_required(value, "expected_rows")),
            expected_puzzles=int(_required(value, "expected_puzzles")),
            seed=int(_required(value, "seed")),
            audit_manifest=str(value.get("audit_manifest", "")),
            num_workers=int(value.get("num_workers", 4)),
        )
        if (
            result.max_sequence_length != 16_384
            or result.expected_rows != profile.train_rows
            or result.expected_puzzles != profile.train_puzzles
            or result.seed != profile.data_seed
        ):
            raise ValueError(f"{profile.profile_id} data contract changed")
        if len(result.train_paths) != len(profile.train_splits):
            raise ValueError(f"{profile.profile_id} training split count changed")
        for path, split in zip(result.train_paths, profile.train_splits, strict=True):
            if not path.replace("\\", "/").endswith(split.path_suffix):
                raise ValueError(
                    f"{profile.profile_id} training path must end with {split.path_suffix}"
                )
        if profile.dev_path_suffix and not result.dev_path.replace("\\", "/").endswith(
            profile.dev_path_suffix
        ):
            raise ValueError(
                f"{profile.profile_id} dev path must end with {profile.dev_path_suffix}"
            )
        if not profile.dev_path_suffix and result.dev_path:
            raise ValueError(f"{profile.profile_id} does not use teacher-forced dev data")
        return result

    @property
    def train_path(self) -> str:
        if len(self.train_paths) != 1:
            raise ValueError("This profile has multiple training paths; use train_paths")
        return self.train_paths[0]


@dataclass(frozen=True)
class TrainingConfig:
    max_steps: int
    global_row_batch: int
    micro_row_batch: int
    world_size: int
    peak_lr: float
    min_lr: float
    warmup_fraction: float
    weight_decay: float
    beta1: float
    beta2: float
    max_grad_norm: float
    save_every: int
    factor_l2: float = 0.0
    base_gradient_checkpointing: bool = True
    pg_gradient_checkpointing: bool = True
    log_every: int = 1

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        profile: ExperimentProfile,
        variant: ExperimentVariant,
    ) -> "TrainingConfig":
        result = cls(
            max_steps=int(_required(value, "max_steps")),
            global_row_batch=int(_required(value, "global_row_batch")),
            micro_row_batch=int(_required(value, "micro_row_batch")),
            world_size=int(value.get("world_size", os.environ.get("WORLD_SIZE", "1"))),
            peak_lr=float(_required(value, "peak_lr")),
            min_lr=float(_required(value, "min_lr")),
            warmup_fraction=float(_required(value, "warmup_fraction")),
            weight_decay=float(_required(value, "weight_decay")),
            beta1=float(_required(value, "beta1")),
            beta2=float(_required(value, "beta2")),
            max_grad_norm=float(_required(value, "max_grad_norm")),
            save_every=int(_required(value, "save_every")),
            factor_l2=float(value.get("factor_l2", 0.0)),
            base_gradient_checkpointing=bool(
                value.get("base_gradient_checkpointing", True)
            ),
            pg_gradient_checkpointing=bool(
                value.get("pg_gradient_checkpointing", True)
            ),
            log_every=int(value.get("log_every", 1)),
        )
        if (
            result.max_steps != profile.max_steps
            or result.global_row_batch != profile.global_row_batch
        ):
            raise ValueError(
                f"{profile.profile_id} requires {profile.max_steps} steps "
                f"and global row batch {profile.global_row_batch}"
            )
        if result.world_size <= 0 or result.micro_row_batch <= 0:
            raise ValueError("Process and microbatch sizes must be positive")
        optimizer_contract = {
            "peak_lr": variant.peak_lr,
            "min_lr": variant.min_lr,
            "warmup_fraction": 0.03,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.98,
            "max_grad_norm": 0.5,
            "factor_l2": variant.factor_l2,
        }
        actual_optimizer = {
            key: getattr(result, key) for key in optimizer_contract
        }
        if actual_optimizer != optimizer_contract:
            raise ValueError(
                f"NVARC PG {variant.variant_id} optimizer contract changed: "
                f"{actual_optimizer} != {optimizer_contract}"
            )
        if result.global_row_batch % (result.micro_row_batch * result.world_size):
            raise ValueError("global row batch is not divisible by micro row batch * world size")
        rows_per_epoch = profile.rows_per_logical_epoch or profile.train_rows
        if profile.train_epochs > 1 and result.max_steps * result.global_row_batch != (
            rows_per_epoch * profile.train_epochs
        ):
            raise ValueError(
                f"{profile.profile_id} step horizon does not equal "
                f"{profile.train_epochs} complete data epochs"
            )
        return result

    @property
    def gradient_accumulation_steps(self) -> int:
        return self.global_row_batch // (self.micro_row_batch * self.world_size)

    @property
    def warmup_steps(self) -> int:
        return round(self.max_steps * self.warmup_fraction)


@dataclass(frozen=True)
class EvaluationConfig:
    protocol_config: str
    milestone_steps: tuple[int, ...]
    modes: tuple[str, ...]
    protocol_kind: str
    challenges: str = ""
    solutions: str = ""
    shuffle_seed: int = 24

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        profile: ExperimentProfile,
    ) -> "EvaluationConfig":
        result = cls(
            protocol_config=str(_required(value, "protocol_config")),
            milestone_steps=tuple(int(item) for item in _required(value, "milestone_steps")),
            modes=tuple(str(item) for item in _required(value, "modes")),
            challenges=str(value.get("challenges", "")),
            solutions=str(value.get("solutions", "")),
            shuffle_seed=int(value.get("shuffle_seed", 24)),
            protocol_kind=profile.protocol_kind,
        )
        if result.milestone_steps != profile.milestone_steps:
            raise ValueError(f"{profile.profile_id} milestone schedule changed")
        if result.modes != ("frozen", "matched", "shuffled"):
            raise ValueError("NVARC PG v1 evaluation modes changed")
        return result


@dataclass(frozen=True)
class NvarcPGConfig:
    format_version: str
    profile_id: str
    variant_id: str
    base: BaseCheckpointConfig
    method: MethodConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NvarcPGConfig":
        expanded = _expand_environment(dict(value))
        format_version = str(_required(expanded, "format_version"))
        profile = profile_for_format(format_version)
        variant = variant_for_id(str(expanded.get("variant_id", "base")))
        return cls(
            format_version=format_version,
            profile_id=profile.profile_id,
            variant_id=variant.variant_id,
            base=BaseCheckpointConfig.from_mapping(
                _required(expanded, "base"), profile=profile
            ),
            method=MethodConfig.from_mapping(
                _required(expanded, "method"), variant=variant
            ),
            data=DataConfig.from_mapping(_required(expanded, "data"), profile=profile),
            training=TrainingConfig.from_mapping(
                _required(expanded, "training"), profile=profile, variant=variant
            ),
            evaluation=EvaluationConfig.from_mapping(
                _required(expanded, "evaluation"), profile=profile
            ),
        )

    @property
    def profile(self) -> ExperimentProfile:
        return profile_for_id(self.profile_id)

    @property
    def variant(self) -> ExperimentVariant:
        return variant_for_id(self.variant_id)

    @property
    def target_layout(self) -> QwenTargetLayout:
        return build_qwen3_4b_target_layout(
            rank=self.method.rank,
            alpha=self.method.alpha,
            token_dim=self.method.token_dim,
            dim_accumulation=self.method.dim_accumulation,
        )

    def build_parameter_generator_config(self) -> ParameterGeneratorConfig:
        return ParameterGeneratorConfig(
            token_dim=self.method.token_dim,
            rank=self.method.rank,
            alpha=self.method.alpha,
            input_dim=self.method.input_dim,
            dim_accumulation=self.method.dim_accumulation,
            d_model=self.method.d_pg,
            head_dim=self.method.head_dim,
            num_base_model_layers=self.target_layout.generator_slot_count,
            output_dim=self.method.token_dim,
            num_pg_layers=self.method.blocks,
            prefix=self.target_layout.prefix,
            pg_mapping=self.target_layout.pg_mapping,
            target_layer_indices=list(range(self.target_layout.generator_slot_count)),
            near_zero_b_init_std=0.0,
            condition_injection_mode="none",
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.variant_id == "base":
            value.pop("variant_id")
        if value["training"].get("factor_l2") == 0.0:
            value["training"].pop("factor_l2")
        if value["training"].get("base_gradient_checkpointing") is True:
            value["training"].pop("base_gradient_checkpointing")
        if value["training"].get("pg_gradient_checkpointing") is True:
            value["training"].pop("pg_gradient_checkpointing")
        return value


def load_experiment_config(path: str | Path) -> NvarcPGConfig:
    config_path = Path(path)
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Experiment config must contain a mapping: {config_path}")
    return NvarcPGConfig.from_mapping(value)

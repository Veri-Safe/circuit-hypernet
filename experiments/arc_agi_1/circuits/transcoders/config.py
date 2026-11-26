from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


LEGACY_FORMAT_VERSION = "nvarc_rearc_transcoder_step399_v1"
HF_FORMAT_VERSION = "nvarc_rearc_transcoder_hf_v2"
UNION_FORMAT_VERSION = "nvarc_rearc_transcoder_union_v3"
TARGET_FORMAT_VERSION = "nvarc_rearc_transcoder_target_v4"
OUTSIDE_UNION_FORMAT_VERSION = "nvarc_rearc128_outside_transcoder_union_v5"
FORMAT_VERSION = LEGACY_FORMAT_VERSION
SUPPORTED_FORMAT_VERSIONS = {
    LEGACY_FORMAT_VERSION,
    HF_FORMAT_VERSION,
    UNION_FORMAT_VERSION,
    TARGET_FORMAT_VERSION,
    OUTSIDE_UNION_FORMAT_VERSION,
}
LOCKED_RUN_NAME = "rearc_step399_base"


def expected_train_generation_modes(
    format_version: str,
) -> dict[str, int] | None:
    if format_version == UNION_FORMAT_VERSION:
        return {
            "cumulative_generated": 6_400,
            "final_only_gold_prefix": 7_123,
        }
    if format_version == OUTSIDE_UNION_FORMAT_VERSION:
        return {"cumulative_generated": 3_200}
    return None


def _required(value: Mapping[str, Any], key: str) -> Any:
    if key not in value:
        raise ValueError(f"Missing required configuration field: {key}")
    return value[key]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise ValueError(f"Unresolved environment variable in path: {expanded}")
        return expanded
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class BaseConfig:
    run_name: str
    source_step: int
    run_manifest: str
    megatron_checkpoint: str
    hf_checkpoint: str
    tokenizer: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BaseConfig":
        result = cls(
            run_name=str(_required(value, "run_name")),
            source_step=int(_required(value, "source_step")),
            run_manifest=str(_required(value, "run_manifest")),
            megatron_checkpoint=str(_required(value, "megatron_checkpoint")),
            hf_checkpoint=str(_required(value, "hf_checkpoint")),
            tokenizer=str(_required(value, "tokenizer")),
        )
        if result.run_name != LOCKED_RUN_NAME or result.source_step != 399:
            raise ValueError(
                "Stage A is locked to the completed RE-ARC FullFT step_399 run"
            )
        if Path(result.megatron_checkpoint).name != "step_399":
            raise ValueError("Megatron checkpoint must end in step_399")
        if Path(result.hf_checkpoint).name != "step_399-hf":
            raise ValueError("HF checkpoint must end in step_399-hf")
        return result


def _validate_sha256(value: str, *, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{label} must be a lowercase SHA256")
    return digest


@dataclass(frozen=True)
class PinnedHfBaseConfig:
    label: str
    run_name: str
    source_step: int
    hf_checkpoint: str
    tokenizer: str
    expected_files: dict[str, str]
    aggregate_sha256: str
    tokenizer_config_sha256: str
    token_mapping_sha256: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PinnedHfBaseConfig":
        files = {
            str(name): _validate_sha256(str(digest), label=f"expected_files.{name}")
            for name, digest in dict(_required(value, "expected_files")).items()
        }
        result = cls(
            label=str(_required(value, "label")),
            run_name=str(_required(value, "run_name")),
            source_step=int(_required(value, "source_step")),
            hf_checkpoint=str(_required(value, "hf_checkpoint")),
            tokenizer=str(_required(value, "tokenizer")),
            expected_files=files,
            aggregate_sha256=_validate_sha256(
                str(_required(value, "aggregate_sha256")), label="aggregate_sha256"
            ),
            tokenizer_config_sha256=_validate_sha256(
                str(_required(value, "tokenizer_config_sha256")),
                label="tokenizer_config_sha256",
            ),
            token_mapping_sha256=_validate_sha256(
                str(_required(value, "token_mapping_sha256")),
                label="token_mapping_sha256",
            ),
        )
        if not result.label or not result.run_name or result.source_step < 0:
            raise ValueError(
                "Pinned HF checkpoint label, run name, and step are required"
            )
        expected_name = f"step_{result.source_step}-hf"
        if Path(result.hf_checkpoint).name != expected_name:
            raise ValueError(f"HF checkpoint must end in {expected_name}")
        required = {
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        }
        missing = required - set(files)
        if missing:
            raise ValueError(
                f"Pinned HF checkpoint is missing artifacts: {sorted(missing)}"
            )
        return result


@dataclass(frozen=True)
class DataConfig:
    train_path: str
    validation_path: str
    expected_train_rows: int
    expected_train_puzzles: int
    expected_train_events: int
    expected_validation_rows: int
    expected_validation_puzzles: int
    expected_validation_events: int
    max_sequence_length: int
    seed: int

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        format_version: str = LEGACY_FORMAT_VERSION,
    ) -> "DataConfig":
        result = cls(
            train_path=str(_required(value, "train_path")),
            validation_path=str(_required(value, "validation_path")),
            expected_train_rows=int(_required(value, "expected_train_rows")),
            expected_train_puzzles=int(_required(value, "expected_train_puzzles")),
            expected_train_events=int(_required(value, "expected_train_events")),
            expected_validation_rows=int(_required(value, "expected_validation_rows")),
            expected_validation_puzzles=int(_required(value, "expected_validation_puzzles")),
            expected_validation_events=int(_required(value, "expected_validation_events")),
            max_sequence_length=int(_required(value, "max_sequence_length")),
            seed=int(_required(value, "seed")),
        )
        expected_by_format = {
            UNION_FORMAT_VERSION: (
                13_523,
                800,
                48_692,
                684,
                114,
                3_006,
                16_384,
                24,
            ),
            TARGET_FORMAT_VERSION: (
                1_600,
                400,
                10_394,
                684,
                114,
                3_006,
                16_384,
                24,
            ),
            OUTSIDE_UNION_FORMAT_VERSION: (
                3_200,
                400,
                20_810,
                684,
                114,
                3_006,
                16_384,
                24,
            ),
        }
        expected = expected_by_format.get(
            format_version,
            (102_392, 400, 665_371, 684, 114, 3_006, 16_384, 24),
        )
        actual = (
            result.expected_train_rows,
            result.expected_train_puzzles,
            result.expected_train_events,
            result.expected_validation_rows,
            result.expected_validation_puzzles,
            result.expected_validation_events,
            result.max_sequence_length,
            result.seed,
        )
        if actual != expected:
            raise ValueError(f"RE-ARC Stage-A data contract changed: {actual} != {expected}")
        normalized_train_path = result.train_path.replace("\\", "/")
        expected_train_suffix = {
            UNION_FORMAT_VERSION: "/rearc8_union_fast128_v1/hf/train",
            TARGET_FORMAT_VERSION: (
                "/rearc128_minus64_delta4_s42/hf/rearc128_minus64_delta4_b1"
            ),
            OUTSIDE_UNION_FORMAT_VERSION: (
                "/rearc128_outside_delta4_b0_b1_union_v1/hf/train"
            ),
        }.get(format_version, "/grids_v15/rearc")
        if not normalized_train_path.endswith(expected_train_suffix):
            raise ValueError(
                f"Training path must end with {expected_train_suffix}"
            )
        if not result.validation_path.replace("\\", "/").endswith(
            "/arc1_clean_sft_v1/arc1_safe_validation"
        ):
            raise ValueError("Validation path must be arc1_safe_validation")
        return result


@dataclass(frozen=True)
class TraceConfig:
    train_root: str
    validation_root: str
    max_new_tokens: int
    rows_per_file: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TraceConfig":
        result = cls(
            train_root=str(_required(value, "train_root")),
            validation_root=str(_required(value, "validation_root")),
            max_new_tokens=int(_required(value, "max_new_tokens")),
            rows_per_file=int(value.get("rows_per_file", 256)),
        )
        if result.max_new_tokens != 1024 or result.rows_per_file <= 0:
            raise ValueError("Trace generation contract changed")
        return result


@dataclass(frozen=True)
class ModelConfig:
    public_transcoder_root: str
    native_model: str
    output_root: str
    d_model: int
    d_feature: int
    num_layers: int
    sparsity_coeff: float
    c_coeff: float
    decoder_norm_min: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ModelConfig":
        result = cls(
            public_transcoder_root=str(_required(value, "public_transcoder_root")),
            native_model=str(_required(value, "native_model")),
            output_root=str(_required(value, "output_root")),
            d_model=int(_required(value, "d_model")),
            d_feature=int(_required(value, "d_feature")),
            num_layers=int(_required(value, "num_layers")),
            sparsity_coeff=float(_required(value, "sparsity_coeff")),
            c_coeff=float(_required(value, "c_coeff")),
            decoder_norm_min=float(_required(value, "decoder_norm_min")),
        )
        expected = (2560, 163_840, 36, 8.0, 4.0, 1e-8)
        actual = (
            result.d_model,
            result.d_feature,
            result.num_layers,
            result.sparsity_coeff,
            result.c_coeff,
            result.decoder_norm_min,
        )
        if actual != expected:
            raise ValueError(f"Transcoder contract changed: {actual} != {expected}")
        return result


@dataclass(frozen=True)
class TrainingConfig:
    effective_tokens: int
    updates_per_block: int
    beta1: float
    beta2: float
    eps: float
    weight_decay: float
    max_grad_norm: float
    warmup_fraction: float
    cooldown_fraction: float
    resumable_percent: int
    candidate_percent: int
    learning_rate_candidates: tuple[float, ...]
    activation_dtype: str
    matrix_compute: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TrainingConfig":
        result = cls(
            effective_tokens=int(_required(value, "effective_tokens")),
            updates_per_block=int(_required(value, "updates_per_block")),
            beta1=float(_required(value, "beta1")),
            beta2=float(_required(value, "beta2")),
            eps=float(_required(value, "eps")),
            weight_decay=float(_required(value, "weight_decay")),
            max_grad_norm=float(_required(value, "max_grad_norm")),
            warmup_fraction=float(_required(value, "warmup_fraction")),
            cooldown_fraction=float(_required(value, "cooldown_fraction")),
            resumable_percent=int(_required(value, "resumable_percent")),
            candidate_percent=int(_required(value, "candidate_percent")),
            learning_rate_candidates=tuple(
                float(item) for item in _required(value, "learning_rate_candidates")
            ),
            activation_dtype=str(_required(value, "activation_dtype")),
            matrix_compute=str(_required(value, "matrix_compute")),
        )
        expected = {
            "effective_tokens": 8192,
            "updates_per_block": 32,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "warmup_fraction": 0.02,
            "cooldown_fraction": 0.20,
            "resumable_percent": 5,
            "candidate_percent": 10,
            "learning_rate_candidates": (5e-6, 1e-5, 2e-5),
            "activation_dtype": "bfloat16",
        }
        for key, expected_value in expected.items():
            if getattr(result, key) != expected_value:
                raise ValueError(f"Training contract changed for {key}")
        if result.matrix_compute not in {"fp32_tf32", "bf16_autocast"}:
            raise ValueError("matrix_compute must be fp32_tf32 or bf16_autocast")
        return result

    @property
    def tokens_per_block(self) -> int:
        return self.effective_tokens * self.updates_per_block


@dataclass(frozen=True)
class NvarcTranscoderConfig:
    format_version: str
    base: BaseConfig | PinnedHfBaseConfig
    data: DataConfig
    traces: TraceConfig
    model: ModelConfig
    training: TrainingConfig

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "NvarcTranscoderConfig":
        expanded = _expand(dict(value))
        format_version = str(_required(expanded, "format_version"))
        if format_version not in SUPPORTED_FORMAT_VERSIONS:
            raise ValueError(f"Unsupported Stage-A format: {format_version}")
        base_value = _required(expanded, "base")
        base = (
            BaseConfig.from_mapping(base_value)
            if format_version == LEGACY_FORMAT_VERSION
            else PinnedHfBaseConfig.from_mapping(base_value)
        )
        return cls(
            format_version=format_version,
            base=base,
            data=DataConfig.from_mapping(
                _required(expanded, "data"), format_version=format_version
            ),
            traces=TraceConfig.from_mapping(_required(expanded, "traces")),
            model=ModelConfig.from_mapping(_required(expanded, "model")),
            training=TrainingConfig.from_mapping(_required(expanded, "training")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def semantic_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> NvarcTranscoderConfig:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Stage-A config must contain a mapping")
    return NvarcTranscoderConfig.from_mapping(value)

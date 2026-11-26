from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


RawWeightMethod = Literal[
    "none",
    "film",
    "p2e_qv",
    "safe_film",
    "safe_p2e_qv",
    "safe_history_p2e",
]
RawWeightContent = Literal[
    "real",
    "zero",
    "random",
    "shuffled_layer",
    "matched_history",
    "static_history",
]


@dataclass(frozen=True)
class RawWeightConditioningConfig:
    """Architecture-only configuration for the isolated raw-weight branch."""

    method: RawWeightMethod = "none"
    content: RawWeightContent = "real"
    cache_path: str = ""
    cache_sha256: str = ""
    projection_seed: int = 24
    gate_init: float | None = None
    film_encoder_hidden_dim: int = 512
    p2e_width: int = 128
    p2e_queries_per_axis: int = 4
    p2e_heads: int = 4
    p2e_ffn_dim: int = 512
    injection_after_block: int | None = None
    residual_cap: float | None = None
    residual_rms_ceiling: float | None = None
    raw_encoder_peak_lr: float | None = None
    raw_output_peak_lr: float | None = None
    raw_warmup_steps: int | None = None
    history_mode: Literal["all", "last1"] | None = None
    max_history: int | None = None

    def __post_init__(self) -> None:
        if self.method not in {
            "none",
            "film",
            "p2e_qv",
            "safe_film",
            "safe_p2e_qv",
            "safe_history_p2e",
        }:
            raise ValueError(f"Unknown raw-weight method: {self.method!r}")
        if self.content not in {
            "real",
            "zero",
            "random",
            "shuffled_layer",
            "matched_history",
            "static_history",
        }:
            raise ValueError(f"Unknown raw-weight content mode: {self.content!r}")
        if self.projection_seed != 24:
            raise ValueError("raw-weight v1 freezes projection_seed=24")
        if self.method == "none":
            if self.cache_path or self.cache_sha256:
                raise ValueError("method=none must not reference a raw-weight cache")
            if self.content != "real":
                raise ValueError("method=none requires content=real")
            if self.gate_init is not None:
                raise ValueError("method=none must not set gate_init")
            self._validate_no_safe_fields()
            self._validate_no_history_fields()
            return

        if self.method == "safe_history_p2e":
            if self.content not in {"matched_history", "static_history"}:
                raise ValueError(
                    "safe_history_p2e content must be matched_history or static_history"
                )
            if self.history_mode not in {"all", "last1"}:
                raise ValueError("safe_history_p2e history_mode must be all or last1")
            if (
                self.max_history is None
                or isinstance(self.max_history, bool)
                or int(self.max_history) != self.max_history
                or not 1 <= int(self.max_history) <= 8
            ):
                raise ValueError("safe_history_p2e max_history must be an integer in 1..8")
        else:
            if self.content in {"matched_history", "static_history"}:
                raise ValueError(
                    f"content={self.content} is reserved for safe_history_p2e"
                )
            self._validate_no_history_fields()

        if not self.cache_path:
            raise ValueError(f"method={self.method} requires cache_path")
        if len(self.cache_sha256) != 64:
            raise ValueError(
                f"method={self.method} requires a 64-character cache_sha256"
            )
        if self.film_encoder_hidden_dim != 512:
            raise ValueError("FiLM v1 freezes film_encoder_hidden_dim=512")
        if self.p2e_width != 128 or self.p2e_heads != 4 or self.p2e_ffn_dim != 512:
            raise ValueError("P2E-QV freezes width/head/FFN to 128/4/512")
        if self.p2e_queries_per_axis not in {4, 8, 16}:
            raise ValueError("P2E-QV queries_per_axis must be one of 4, 8, or 16")

        if self.method.startswith("safe_"):
            if self.gate_init is not None:
                raise ValueError("safe raw-weight methods use a fixed residual cap, not gate_init")
            if self.resolved_injection_after_block != 1:
                raise ValueError("safe raw-weight v2 freezes injection_after_block=1")
            if self.resolved_residual_cap not in {0.005, 0.01, 0.025, 0.05}:
                raise ValueError(
                    "safe raw-weight residual_cap must be one of "
                    "0.005, 0.01, 0.025, or 0.05"
                )
            if self.method not in {
                "safe_p2e_qv",
                "safe_history_p2e",
            } and self.residual_rms_ceiling is not None:
                raise ValueError(
                    "Safe-P2E is the only method that supports residual_rms_ceiling"
                )
            if self.residual_rms_ceiling is not None:
                ceiling = self.resolved_residual_rms_ceiling
                if not math.isfinite(ceiling) or ceiling <= 0.0:
                    raise ValueError("residual_rms_ceiling must be finite and positive")
                if ceiling > self.resolved_residual_cap:
                    raise ValueError(
                        "residual_rms_ceiling must not exceed residual_cap"
                    )
            if self.resolved_raw_encoder_peak_lr <= 0.0:
                raise ValueError("raw_encoder_peak_lr must be positive")
            if self.resolved_raw_output_peak_lr <= 0.0:
                raise ValueError("raw_output_peak_lr must be positive")
            if self.raw_warmup_steps is not None:
                if (
                    isinstance(self.raw_warmup_steps, bool)
                    or int(self.raw_warmup_steps) != self.raw_warmup_steps
                    or int(self.raw_warmup_steps) < 0
                ):
                    raise ValueError("raw_warmup_steps must be a non-negative integer")
            return

        self._validate_no_safe_fields()
        expected_gate = 0.05 if self.method == "film" else 0.01
        gate = expected_gate if self.gate_init is None else float(self.gate_init)
        allowed = {"film": {0.05, 0.10}, "p2e_qv": {0.01, 0.02}}[self.method]
        if gate not in allowed:
            raise ValueError(
                f"{self.method} gate_init must be one of {sorted(allowed)}, found {gate}"
            )

    def _validate_no_safe_fields(self) -> None:
        values = {
            "injection_after_block": self.injection_after_block,
            "residual_cap": self.residual_cap,
            "residual_rms_ceiling": self.residual_rms_ceiling,
            "raw_encoder_peak_lr": self.raw_encoder_peak_lr,
            "raw_output_peak_lr": self.raw_output_peak_lr,
            "raw_warmup_steps": self.raw_warmup_steps,
        }
        unexpected = sorted(key for key, value in values.items() if value is not None)
        if unexpected:
            raise ValueError(
                f"method={self.method} does not accept safe-v2 fields: {unexpected}"
            )

    def _validate_no_history_fields(self) -> None:
        unexpected = sorted(
            key
            for key, value in {
                "history_mode": self.history_mode,
                "max_history": self.max_history,
            }.items()
            if value is not None
        )
        if unexpected:
            raise ValueError(
                f"method={self.method} does not accept history fields: {unexpected}"
            )

    @property
    def resolved_gate_init(self) -> float:
        if self.method not in {"film", "p2e_qv"}:
            raise ValueError(f"method={self.method} has no trainable raw-weight gate")
        if self.gate_init is not None:
            return float(self.gate_init)
        return 0.05 if self.method == "film" else 0.01

    @property
    def resolved_injection_after_block(self) -> int:
        if not self.method.startswith("safe_"):
            raise ValueError(f"method={self.method} has no safe-v2 injection point")
        return 1 if self.injection_after_block is None else int(self.injection_after_block)

    @property
    def resolved_residual_cap(self) -> float:
        if not self.method.startswith("safe_"):
            raise ValueError(f"method={self.method} has no bounded residual")
        return 0.025 if self.residual_cap is None else float(self.residual_cap)

    @property
    def resolved_residual_rms_ceiling(self) -> float:
        if self.method not in {
            "safe_p2e_qv",
            "safe_history_p2e",
        } or self.residual_rms_ceiling is None:
            raise ValueError(
                f"method={self.method} has no configured residual RMS ceiling"
            )
        return float(self.residual_rms_ceiling)

    @property
    def resolved_raw_encoder_peak_lr(self) -> float:
        if not self.method.startswith("safe_"):
            raise ValueError(f"method={self.method} has no differential raw encoder LR")
        return (
            2.0e-4
            if self.raw_encoder_peak_lr is None
            else float(self.raw_encoder_peak_lr)
        )

    @property
    def resolved_raw_output_peak_lr(self) -> float:
        if not self.method.startswith("safe_"):
            raise ValueError(f"method={self.method} has no differential raw output LR")
        return (
            5.0e-4
            if self.raw_output_peak_lr is None
            else float(self.raw_output_peak_lr)
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RawWeightConditioningConfig":
        if value is None:
            return cls()
        allowed = set(cls.__annotations__)
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"Unknown raw-weight configuration fields: {unknown}")
        expanded = {
            key: os.path.expandvars(item) if isinstance(item, str) else item
            for key, item in value.items()
        }
        return cls(**expanded)

    def validate_cache_file(self) -> Path | None:
        if self.method == "none":
            return None
        path = Path(self.cache_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Raw-weight cache does not exist: {path}")
        return path

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.method.startswith("safe_"):
            value["injection_after_block"] = self.resolved_injection_after_block
            value["residual_cap"] = self.resolved_residual_cap
            value["raw_encoder_peak_lr"] = self.resolved_raw_encoder_peak_lr
            value["raw_output_peak_lr"] = self.resolved_raw_output_peak_lr
            value.pop("gate_init")
            if self.method not in {
                "safe_p2e_qv",
                "safe_history_p2e",
            } or self.residual_rms_ceiling is None:
                value.pop("residual_rms_ceiling")
            if self.raw_warmup_steps is None:
                value.pop("raw_warmup_steps")
            if self.method != "safe_history_p2e":
                value.pop("history_mode")
                value.pop("max_history")
        else:
            for key in (
                "injection_after_block",
                "residual_cap",
                "residual_rms_ceiling",
                "raw_encoder_peak_lr",
                "raw_output_peak_lr",
                "raw_warmup_steps",
                "history_mode",
                "max_history",
            ):
                value.pop(key)
        if self.gate_init is None and self.method in {"film", "p2e_qv"}:
            value["gate_init"] = self.resolved_gate_init
        return value


def load_raw_weight_config(path: str | Path) -> RawWeightConditioningConfig:
    config_path = Path(path).expanduser().resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Raw-weight config must contain a mapping: {config_path}")
    if "raw_weight" in value:
        value = value["raw_weight"]
    if not isinstance(value, dict):
        raise ValueError(f"raw_weight must contain a mapping: {config_path}")
    return RawWeightConditioningConfig.from_mapping(value)


def raw_weight_architecture_identity(
    value: RawWeightConditioningConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the portable identity of a raw-weight conditioner architecture.

    The cache content hash is part of the identity; its absolute path is not because
    the same shared volume is mounted as /volume on login nodes and /mnt in Siflow.
    """

    normalized = value.to_dict() if isinstance(value, RawWeightConditioningConfig) else dict(value)
    normalized.pop("content", None)
    normalized.pop("cache_path", None)
    normalized.pop("raw_encoder_peak_lr", None)
    normalized.pop("raw_output_peak_lr", None)
    normalized.pop("raw_warmup_steps", None)
    return normalized


def raw_weight_family_init_identity(
    value: RawWeightConditioningConfig | Mapping[str, Any],
) -> dict[str, Any]:
    """Return the subset that determines conditioner parameter initialization.

    ``residual_rms_ceiling`` changes the forward-time trust region but does not
    change, initialize, or resize a conditioner parameter.  Excluding it here
    lets LR/ceiling ablations share one bit-identical family initialization
    while the full architecture identity still protects checkpoint evaluation
    from a mismatched ceiling.
    """

    normalized = raw_weight_architecture_identity(value)
    normalized.pop("residual_rms_ceiling", None)
    normalized.pop("history_mode", None)
    normalized.pop("max_history", None)
    return normalized

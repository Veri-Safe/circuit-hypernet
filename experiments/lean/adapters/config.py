"""Configuration and immutable identities for the S5 HyperNet experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


BASE_STEP = 38_000
BASE_CHECKPOINT_SHA256 = (
    "ee2a7efdf77262ee1fbdf7d2080f930c2bedc27e45b7712537fba964ee05931c"
)
BASE_MODEL_STATE_SHA256 = (
    "bd0277b72e7f17f864bd9e8064e1dff989a81254c5353550c6f912944ecf1afd"
)
BASE_CONFIG_SHA256 = (
    "d87e874cdee838e7d07b99631d06ee64ec3466d24012a08af75d032eddf7169e"
)
PROTOCOL_ID = "s5_hypernet_vector397_ab_fusion_v1"
SCREEN_PREFIX = 120_000
FORMAL_PREFIX = 1_200_000
BLOCK_SIZE = 128
GLOBAL_BATCH_SIZE = 512
TOKENS_PER_STEP = BLOCK_SIZE * GLOBAL_BATCH_SIZE

FORMAL_BUDGETS = (500_000_000, 1_000_000_000, 2_490_368_000)
PAIRED_GROUPS: dict[str, dict[str, float]] = {
    "G0": {"pg_lr": 5.0e-5, "gate_max": 0.25},
    "G1": {"pg_lr": 2.5e-5, "gate_max": 0.25},
    "G2": {"pg_lr": 5.0e-5, "gate_max": 0.125},
    "G3": {"pg_lr": 5.0e-5, "gate_max": 0.5},
}
FORMAL_GRID_GROUPS: dict[str, dict[str, float]] = {
    "P00": {"pg_lr": 5.0e-5, "gate_max": 0.25},
    "P01": {"pg_lr": 2.5e-5, "gate_max": 0.25},
    "P02": {"pg_lr": 5.0e-5, "gate_max": 0.125},
    "P03": {"pg_lr": 5.0e-5, "gate_max": 0.5},
    "P04": {"pg_lr": 1.25e-5, "gate_max": 0.125},
    "P05": {"pg_lr": 1.25e-5, "gate_max": 0.25},
    "P06": {"pg_lr": 1.25e-5, "gate_max": 0.375},
    "P07": {"pg_lr": 1.25e-5, "gate_max": 0.5},
    "P08": {"pg_lr": 2.5e-5, "gate_max": 0.125},
    "P09": {"pg_lr": 2.5e-5, "gate_max": 0.375},
    "P10": {"pg_lr": 2.5e-5, "gate_max": 0.5},
    "P11": {"pg_lr": 5.0e-5, "gate_max": 0.375},
    "P12": {"pg_lr": 7.5e-5, "gate_max": 0.125},
    "P13": {"pg_lr": 7.5e-5, "gate_max": 0.25},
    "P14": {"pg_lr": 7.5e-5, "gate_max": 0.375},
    "P15": {"pg_lr": 7.5e-5, "gate_max": 0.5},
}
# Query/Circuit relative-learning-rate follow-up around the two selected
# anchors.  Each candidate gets its own group namespace because Candidate A
# and B intentionally keep different gate caps (A=.5, B=.125).  The sweep
# changes only the query/circuit LR multipliers; pg_lr and gate_max stay at the
# corresponding anchor values.
FORMAL_QC_SWEEP_GROUPS: dict[str, dict[str, float]] = {
    **{f"AQC{index:02d}": {"pg_lr": 5.0e-5, "gate_max": 0.5} for index in range(8)},
    **{f"BQC{index:02d}": {"pg_lr": 5.0e-5, "gate_max": 0.125} for index in range(8)},
}
# Query-only atomic-specialization cells.  The corpus, optimizer contract,
# and token budget remain the formal 1.2M/2.490368B protocol; this matrix only
# explores stronger PG learning rates and LoRA gates.
ATOMIC_SPECIALIZATION_GROUPS: dict[str, dict[str, float]] = {
    "QO00": {"pg_lr": 1.0e-4, "gate_max": 0.5},
    "QO01": {"pg_lr": 1.25e-4, "gate_max": 0.5},
    "QO02": {"pg_lr": 1.5e-4, "gate_max": 0.5},
    "QO03": {"pg_lr": 1.0e-4, "gate_max": 0.75},
    "QO04": {"pg_lr": 1.25e-4, "gate_max": 0.75},
    "QO05": {"pg_lr": 1.5e-4, "gate_max": 0.75},
    "QO06": {"pg_lr": 7.5e-5, "gate_max": 0.5},
    "QO07": {"pg_lr": 7.5e-5, "gate_max": 0.625},
    "QO08": {"pg_lr": 7.5e-5, "gate_max": 0.75},
    "QO09": {"pg_lr": 7.5e-5, "gate_max": 0.875},
    "QO10": {"pg_lr": 1.0e-4, "gate_max": 0.625},
    "QO11": {"pg_lr": 1.0e-4, "gate_max": 0.875},
    "QO12": {"pg_lr": 1.25e-4, "gate_max": 0.625},
    "QO13": {"pg_lr": 1.25e-4, "gate_max": 0.875},
    "QO14": {"pg_lr": 1.5e-4, "gate_max": 0.625},
    "QO15": {"pg_lr": 1.5e-4, "gate_max": 0.875},
    "QO16": {"pg_lr": 8.75e-5, "gate_max": 0.5625},
    "QO17": {"pg_lr": 8.75e-5, "gate_max": 0.6875},
    "QO18": {"pg_lr": 8.75e-5, "gate_max": 0.8125},
    "QO19": {"pg_lr": 1.125e-4, "gate_max": 0.5625},
    "QO20": {"pg_lr": 1.125e-4, "gate_max": 0.6875},
    "QO21": {"pg_lr": 1.125e-4, "gate_max": 0.8125},
    "QO22": {"pg_lr": 1.375e-4, "gate_max": 0.5625},
    "QO23": {"pg_lr": 1.375e-4, "gate_max": 0.6875},
    "QO24": {"pg_lr": 1.375e-4, "gate_max": 0.8125},
    "QO25": {"pg_lr": 1.625e-4, "gate_max": 0.5625},
    "QO26": {"pg_lr": 1.625e-4, "gate_max": 0.6875},
    "QO27": {"pg_lr": 1.625e-4, "gate_max": 0.8125},
}
CIRCUIT_HEAVY80_GROUPS: dict[str, dict[str, float]] = {
    "C80_4E5_G050": {"pg_lr": 4.0e-5, "gate_max": 0.5},
    "C80_6E5_G050": {"pg_lr": 6.0e-5, "gate_max": 0.5},
}
HYBRID_APPEND6_GROUPS: dict[str, dict[str, float]] = {
    "H00": {"pg_lr": 5.0e-5, "gate_max": 0.375},
    "H01": {"pg_lr": 5.0e-5, "gate_max": 0.5},
    "H02": {"pg_lr": 6.25e-5, "gate_max": 0.375},
    "H03": {"pg_lr": 6.25e-5, "gate_max": 0.5},
    "H04": {"pg_lr": 7.5e-5, "gate_max": 0.375},
    "H05": {"pg_lr": 7.5e-5, "gate_max": 0.5},
}
VECTOR397_SCHEMA_SHA256 = "e6c5528d1a06d2979babe776ff22e28ef08e40cf9a2fd9a5d69751418b337bda"
VECTOR397_INPUT_TREATMENTS = (
    "fixed_log1p_v1",
    "train_zscore_v1",
    "train_zscore_compact381_v1",
)

# Paths are provenance and launch-location metadata, not immutable artifact
# identity. The same volume may have different mount points on training and
# preparation hosts. Hashes and the frozen step must agree across mounts.
BASE_CONTENT_IDENTITY_FIELDS = (
    "checkpoint_sha256",
    "model_state_sha256",
    "config_sha256",
    "step",
)


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


@dataclass(frozen=True)
class S5BaseIdentity:
    checkpoint_path: str
    # ``config_sha256`` is the protocol identity stored inside the resolved
    # architecture/checkpoint envelope.  It is deliberately different from
    # the byte hash of ``resolved_config.json``.  Keep the path explicit so a
    # launcher cannot silently select a nearby architecture file.
    architecture_config_path: str = ""
    checkpoint_sha256: str = BASE_CHECKPOINT_SHA256
    model_state_sha256: str = BASE_MODEL_STATE_SHA256
    config_sha256: str = BASE_CONFIG_SHA256
    step: int = BASE_STEP

    def __post_init__(self) -> None:
        if not str(self.checkpoint_path):
            raise ValueError("checkpoint_path must not be empty")
        for name in ("checkpoint_sha256", "model_state_sha256", "config_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        if int(self.step) != BASE_STEP:
            raise ValueError("S5 HyperNet Base step must be 38000")


def base_content_identity(value: Mapping[str, Any] | S5BaseIdentity) -> dict[str, Any]:
    """Return the mount-path-independent Base identity fields."""

    if isinstance(value, S5BaseIdentity):
        source: Mapping[str, Any] = asdict(value)
    else:
        source = value
    return {field: source.get(field) for field in BASE_CONTENT_IDENTITY_FIELDS}


def base_content_identity_matches(
    actual: Mapping[str, Any] | S5BaseIdentity,
    expected: Mapping[str, Any] | S5BaseIdentity,
) -> bool:
    """Compare Base identity without rejecting /volume versus /mnt aliases."""

    return base_content_identity(actual) == base_content_identity(expected)


@dataclass(frozen=True)
class S5HyperNetModelConfig:
    candidate: str
    group: str = "G0"
    seed: int = 42
    query_width: int = 1024
    circuit_width: int = 1024
    n_ctx: int = 128
    circuit_token_count: int = 1
    pg_lr: float = 5e-5
    factor_l2: float = 0.0
    effective_delta_l2: float = 1e-4
    base_preservation_coefficient: float = 0.5
    gate_max: float = 0.25
    gate_warmup_ratio: float = 0.20
    query_lr_multiplier: float = 0.5
    circuit_lr_multiplier: float = 2.0
    warmup_ratio: float = 0.03
    cosine_min_lr_ratio: float = 0.1
    total_tokens: int = 1_000_000_000
    global_batch_size: int = GLOBAL_BATCH_SIZE
    expected_world_size: int = field(default_factory=lambda: int(os.environ.get("WORLD_SIZE", "1")))
    micro_batch_size: int | None = None
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    weight_decay: float = 0.0
    max_grad_norm: float = 0.5
    bf16: bool = True

    def __post_init__(self) -> None:
        if self.candidate not in {"A", "B", "AB", "a", "b", "ab"}:
            raise ValueError("candidate must be A, B, or AB")
        object.__setattr__(self, "candidate", str(self.candidate).upper())
        from ..performance_completion import approved_training_seeds
        if int(self.seed) not in approved_training_seeds():
            raise ValueError("formal matrix seed must belong to a registered repetition study")
        if self.query_width != 1024 or self.circuit_width != 1024:
            raise ValueError("Query/Circuit widths are fixed at 1024")
        expected_tokens = {"A": 1, "B": 5, "AB": 6}[self.candidate]
        if int(self.circuit_token_count) != expected_tokens:
            raise ValueError("circuit_token_count does not match candidate")
        normalized_group = str(self.group).upper()
        all_groups = {
            **PAIRED_GROUPS,
            **FORMAL_GRID_GROUPS,
            **FORMAL_QC_SWEEP_GROUPS,
            **ATOMIC_SPECIALIZATION_GROUPS,
            **CIRCUIT_HEAVY80_GROUPS,
            **HYBRID_APPEND6_GROUPS,
        }
        if normalized_group not in all_groups:
            raise ValueError("group is not an approved HyperNet parameter cell")
        if normalized_group.startswith("AQC") and self.candidate != "A":
            raise ValueError("AQC groups are reserved for Candidate A")
        if normalized_group.startswith("BQC") and self.candidate != "B":
            raise ValueError("BQC groups are reserved for Candidate B")
        object.__setattr__(self, "group", normalized_group)
        group = all_groups[normalized_group]
        if float(self.pg_lr) != group["pg_lr"] or float(self.gate_max) != group["gate_max"]:
            raise ValueError("pg_lr/gate_max do not match the paired group contract")
        if self.n_ctx != BLOCK_SIZE or self.global_batch_size != GLOBAL_BATCH_SIZE:
            raise ValueError("invalid S5 training dimensions")
        if int(self.total_tokens) not in FORMAL_BUDGETS:
            raise ValueError("total_tokens is not one of the user-approved budget candidates")
        if self.expected_world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        if self.global_batch_size % self.expected_world_size:
            raise ValueError("global batch must divide the formal world size")
        local_batch = self.global_batch_size // self.expected_world_size
        if self.micro_batch_size is None:
            requested = int(os.environ.get("HYPERNET_MICROBATCH", str(local_batch)))
            object.__setattr__(self, "micro_batch_size", requested)
        if self.micro_batch_size <= 0 or local_batch % self.micro_batch_size:
            raise ValueError("micro_batch_size must divide the per-rank logical batch")
        if self.pg_lr <= 0 or self.factor_l2 != 0.0 or self.effective_delta_l2 < 0:
            raise ValueError("invalid S5 optimizer values")
        if self.effective_delta_l2 != 1.0e-4:
            raise ValueError("effective_delta_l2 is fixed at 1e-4 in approved cells")
        if self.base_preservation_coefficient != 0.5:
            raise ValueError("base_preservation_coefficient is fixed at 0.5")
        if self.gate_warmup_ratio != 0.20:
            raise ValueError("gate_warmup_ratio is fixed at 0.20")
        optimizer_contract = (
            self.query_lr_multiplier,
            self.circuit_lr_multiplier,
            self.warmup_ratio,
            self.cosine_min_lr_ratio,
            self.adam_beta1,
            self.adam_beta2,
            self.weight_decay,
            self.max_grad_norm,
            self.bf16,
        )
        allowed_query_multipliers = (
            {0.25, 0.5, 1.0}
            if normalized_group in FORMAL_QC_SWEEP_GROUPS
            else {0.5}
        )
        allowed_circuit_multipliers = (
            {1.0, 2.0, 4.0}
            if normalized_group in FORMAL_QC_SWEEP_GROUPS
            else {2.0, 4.0}
        )
        if float(optimizer_contract[0]) not in allowed_query_multipliers or optimizer_contract[2:] != (
            0.03,
            0.1,
            0.9,
            0.98,
            0.0,
            0.5,
            True,
        ) or float(self.circuit_lr_multiplier) not in allowed_circuit_multipliers:
            raise ValueError(
                "formal HyperNet optimizer/precision contract changed; "
                "query/circuit LR multipliers are outside the approved cell"
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0,1)")
        if not 0.0 < self.cosine_min_lr_ratio <= 1.0:
            raise ValueError("cosine_min_lr_ratio must be in (0,1]")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Execution topology is supplied by the launcher and recorded in checkpoints.
        payload.pop("expected_world_size")
        payload.pop("micro_batch_size")
        return payload

    @property
    def total_steps(self) -> int:
        return int(self.total_tokens) // TOKENS_PER_STEP

    @property
    def gradient_accumulation_steps(self) -> int:
        local_batch = self.global_batch_size // self.expected_world_size
        return local_batch // self.micro_batch_size


@dataclass(frozen=True)
class S5HyperNetDataConfig:
    archive_manifest: str
    tokenizer_path: str
    base_ce_path: str
    prefix: int = FORMAL_PREFIX
    n_ctx: int = 128
    metadata_root: str | None = None
    cache_path: str | None = None
    descriptor_schema_sha256: str = VECTOR397_SCHEMA_SHA256
    input_treatment: str = "fixed_log1p_v1"
    # A normalized fingerprint bank is a condition-only replacement for the
    # vectors in ``cache_path``.  The token/cache payload itself is retained
    # unchanged so Frozen-Base CE and the query/supervision stream remain
    # byte-identical to the original formal protocol.
    vector397_fingerprint_manifest: str | None = None
    vector397_normalizer_path: str | None = None
    # Compatibility fields used by the first train-only z-score configs.
    # New remote rerun configs use the explicit vector397_* spelling above.
    fingerprint_manifest: str | None = None
    fingerprint_normalizer: str | None = None
    fingerprint_scope: str | None = None
    fingerprint_normalizer_artifact_sha256: str | None = None

    def __post_init__(self) -> None:
        if int(self.prefix) <= 0 or int(self.prefix) > FORMAL_PREFIX:
            raise ValueError("S5 data prefix is outside the frozen corpus range")
        if int(self.prefix) != FORMAL_PREFIX:
            raise ValueError("formal A/B runs must use the complete 1.2M bank")
        if self.input_treatment not in VECTOR397_INPUT_TREATMENTS:
            raise ValueError("unsupported Vector397 input treatment")
        canonical = (
            self.vector397_fingerprint_manifest,
            self.vector397_normalizer_path,
        )
        legacy = (self.fingerprint_manifest, self.fingerprint_normalizer)
        if any(value is not None for value in canonical) and not all(
            isinstance(value, str) and value for value in canonical
        ):
            raise ValueError(
                "vector397_fingerprint_manifest and vector397_normalizer_path "
                "must be supplied together"
            )
        if any(value is not None for value in legacy) and not all(
            isinstance(value, str) and value for value in legacy
        ):
            raise ValueError(
                "fingerprint_manifest and fingerprint_normalizer must be supplied together"
            )
        if all(isinstance(value, str) and value for value in canonical) and all(
            isinstance(value, str) and value for value in legacy
        ) and canonical != legacy:
            raise ValueError("duplicate Vector397 fingerprint paths disagree")
        if self.input_treatment in {"train_zscore_v1", "train_zscore_compact381_v1"} and not (
            all(isinstance(value, str) and value for value in canonical)
            or all(isinstance(value, str) and value for value in legacy)
        ):
            raise ValueError(
                f"{self.input_treatment} requires a fingerprint manifest and normalizer path"
            )
        if self.input_treatment == "fixed_log1p_v1" and any(
            value is not None
            for value in (
                *canonical,
                *legacy,
                self.fingerprint_scope,
                self.fingerprint_normalizer_artifact_sha256,
            )
        ):
            raise ValueError(
                "fixed_log1p_v1 cannot declare a normalized fingerprint source"
            )
        if self.input_treatment in {"train_zscore_v1", "train_zscore_compact381_v1"}:
            if self.fingerprint_scope is not None and not str(self.fingerprint_scope):
                raise ValueError("fingerprint_scope must not be empty")
            if self.fingerprint_normalizer_artifact_sha256 is not None:
                artifact = str(self.fingerprint_normalizer_artifact_sha256)
                if len(artifact) != 64 or any(
                    char not in "0123456789abcdef" for char in artifact
                ):
                    raise ValueError(
                        "fingerprint_normalizer_artifact_sha256 must be a lowercase SHA256"
                    )
        if self.descriptor_schema_sha256 != VECTOR397_SCHEMA_SHA256:
            raise ValueError("formal A/B Vector397 schema identity changed")
        for name in ("archive_manifest", "tokenizer_path", "base_ce_path"):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must not be empty")
        if not self.cache_path:
            raise ValueError("formal A/B runs require one shared packed cache_path")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for name in (
            "vector397_fingerprint_manifest",
            "vector397_normalizer_path",
            "fingerprint_manifest",
            "fingerprint_normalizer",
            "fingerprint_scope",
            "fingerprint_normalizer_artifact_sha256",
        ):
            if value.get(name) is None:
                value.pop(name, None)
        return value

    @property
    def resolved_fingerprint_manifest(self) -> str | None:
        return self.vector397_fingerprint_manifest or self.fingerprint_manifest

    @property
    def resolved_fingerprint_normalizer(self) -> str | None:
        return self.vector397_normalizer_path or self.fingerprint_normalizer


@dataclass(frozen=True)
class S5HyperNetRunConfig:
    model: S5HyperNetModelConfig
    data: S5HyperNetDataConfig
    base: S5BaseIdentity
    output_dir: str
    run_name: str
    protocol_id: str = PROTOCOL_ID
    input_treatment_identity: str | None = None
    common_init_manifest: str | None = None
    candidate_init_manifest: str | None = None
    assignment_seed: int = 42
    eval_batch_size: int = 8
    save_every_epoch: bool = True
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError("S5 HyperNet protocol identity changed")
        if not self.run_name or not self.output_dir:
            raise ValueError("run_name and output_dir are required")
        if self.eval_batch_size <= 0:
            raise ValueError("eval_batch_size must be positive")
        from ..performance_completion import validate_atomic
        validate_atomic(self)

    def to_dict(self) -> dict[str, Any]:
        # Keep historical fixed-log1p config hashes stable: optional fields are
        # omitted by S5HyperNetDataConfig when they are unused.
        return {
            "model": self.model.to_dict(),
            "data": self.data.to_dict(),
            "base": asdict(self.base),
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "protocol_id": self.protocol_id,
            "input_treatment_identity": self.input_treatment_identity,
            "common_init_manifest": self.common_init_manifest,
            "candidate_init_manifest": self.candidate_init_manifest,
            "assignment_seed": self.assignment_seed,
            "eval_batch_size": self.eval_batch_size,
            "save_every_epoch": self.save_every_epoch,
            "extra": dict(self.extra),
        }

    @property
    def config_sha256(self) -> str:
        return semantic_sha256(self.to_dict())


def load_run_config(path: str | Path) -> S5HyperNetRunConfig:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("S5 HyperNet config must be a JSON object")
    model = S5HyperNetModelConfig(**dict(value["model"]))
    data = S5HyperNetDataConfig(**dict(value["data"]))
    base_value = dict(value["base"])
    # Accept the explicit top-level spelling for hand-authored configs, but
    # never fall back to the old ``extra`` escape hatch during training.
    if not base_value.get("architecture_config_path"):
        base_value["architecture_config_path"] = value.get(
            "architecture_config_path",
            dict(value.get("extra", {})).get("architecture_config_path", ""),
        )
    base = S5BaseIdentity(**base_value)
    config = S5HyperNetRunConfig(
        model=model,
        data=data,
        base=base,
        output_dir=str(value["output_dir"]),
        run_name=str(value["run_name"]),
        protocol_id=str(value.get("protocol_id", PROTOCOL_ID)),
        input_treatment_identity=value.get("input_treatment_identity"),
        common_init_manifest=value.get("common_init_manifest"),
        candidate_init_manifest=value.get("candidate_init_manifest"),
        assignment_seed=int(value.get("assignment_seed", 42)),
        eval_batch_size=int(value.get("eval_batch_size", 8)),
        save_every_epoch=bool(value.get("save_every_epoch", True)),
        extra=dict(value.get("extra", {})),
    )
    recorded = value.get("config_sha256")
    if recorded is not None and str(recorded) != config.config_sha256:
        raise ValueError("S5 HyperNet config checksum changed")
    return config


def write_run_config(path: str | Path, config: S5HyperNetRunConfig) -> dict[str, Any]:
    destination = Path(path)
    payload = config.to_dict()
    payload["config_sha256"] = config.config_sha256
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


__all__ = [
    "BASE_CHECKPOINT_SHA256",
    "BASE_CONFIG_SHA256",
    "BASE_CONTENT_IDENTITY_FIELDS",
    "BASE_MODEL_STATE_SHA256",
    "BASE_STEP",
    "BLOCK_SIZE",
    "FORMAL_BUDGETS",
    "FORMAL_GRID_GROUPS",
    "FORMAL_QC_SWEEP_GROUPS",
    "ATOMIC_SPECIALIZATION_GROUPS",
    "CIRCUIT_HEAVY80_GROUPS",
    "FORMAL_PREFIX",
    "GLOBAL_BATCH_SIZE",
    "HYBRID_APPEND6_GROUPS",
    "PAIRED_GROUPS",
    "PROTOCOL_ID",
    "SCREEN_PREFIX",
    "TOKENS_PER_STEP",
    "VECTOR397_SCHEMA_SHA256",
    "VECTOR397_INPUT_TREATMENTS",
    "S5BaseIdentity",
    "S5HyperNetDataConfig",
    "S5HyperNetModelConfig",
    "S5HyperNetRunConfig",
    "base_content_identity",
    "base_content_identity_matches",
    "load_run_config",
    "semantic_sha256",
    "write_run_config",
]

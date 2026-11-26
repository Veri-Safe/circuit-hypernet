"""Strict JSON configuration for the composition leave-one-out SFT round."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .composition_loco import (
    COMPOSITION_TASKS,
    GLOBAL_BATCH_SIZE,
    N_CTX,
    PROTOCOL_ID,
    STAGE_A_BASE_IDENTITY,
    ROUTES,
    TOTAL_PADDED_TOKENS,
    TOTAL_STEPS,
    VALIDATION_STEPS,
    task_fold,
)
from .composition_loco_manual160_spec import (
    MANUAL160_OUTPUT_ROOT,
    MANUAL160_PROFILES,
    MANUAL160_PROTOCOL_ID,
    MANUAL160_ROUTES,
    profile_values,
)
from .composition_loco_direct_lr32_spec import (
    DIRECT_LR32_OUTPUT_ROOT,
    DIRECT_LR32_PROFILES,
    DIRECT_LR32_PROTOCOL_ID,
    direct_lr32_profile_values,
)
from .composition_loco_circuit_sweep40_spec import (
    CIRCUIT_SWEEP40_BASE_STEP,
    CIRCUIT_SWEEP40_CONFIG_ROOT,
    CIRCUIT_SWEEP40_INITIAL_CHECKPOINT,
    CIRCUIT_SWEEP40_INITIAL_CHECKPOINT_SHA256,
    CIRCUIT_SWEEP40_INITIAL_STEP,
    CIRCUIT_SWEEP40_OUTPUT_ROOT,
    CIRCUIT_T_AND_OR40_OUTPUT_ROOT,
    CIRCUIT_SWEEP40_PROFILES,
    CIRCUIT_SWEEP40_PROTOCOL_ID,
    CIRCUIT_SWEEP40_ROUTES,
    circuit_sweep40_profile_label,
    circuit_sweep40_profile_values,
)
from .composition_loco_query_weight_spec import (
    QUERY_WEIGHT_HELDOUT_TASK,
    QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256,
    QUERY_WEIGHT_OUTPUT_ROOT,
    QUERY_WEIGHT_PROFILES,
    QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT,
    QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256,
    QUERY_WEIGHT_RAW_CACHE_PATH,
    QUERY_WEIGHT_RAW_CACHE_SHA256,
    QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT,
    QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256,
    QUERY_WEIGHT_PROTOCOL_ID,
    QUERY_WEIGHT_ROUTES,
    profile_label as query_weight_profile_label,
    profile_values as query_weight_profile_values,
)
from .composition_loco_query_circuit_spec import (
    QUERY_CIRCUIT_EVAL_BATCH_SIZE,
    QUERY_CIRCUIT_HELDOUT_TASK,
    QUERY_CIRCUIT_OUTPUT_ROOT,
    QUERY_CIRCUIT_PROFILES,
    QUERY_CIRCUIT_PROTOCOL_ID,
    QUERY_CIRCUIT_ROUTES,
    initial_checkpoint as query_circuit_initial_checkpoint,
    profile_label as query_circuit_profile_label,
    profile_values as query_circuit_profile_values,
)
from .composition_loco_weight_circuit_spec import (
    WEIGHT_CIRCUIT_EVAL_BATCH_SIZE,
    WEIGHT_CIRCUIT_HELDOUT_TASKS,
    WEIGHT_CIRCUIT_OUTPUT_ROOT,
    WEIGHT_CIRCUIT_PROFILES,
    WEIGHT_CIRCUIT_PROTOCOL_ID,
    WEIGHT_CIRCUIT_ROUTES,
    QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256,
    QUERY_WEIGHT_RAW_CACHE_PATH,
    QUERY_WEIGHT_RAW_CACHE_SHA256,
    heldout_task as weight_circuit_heldout_task,
    initial_checkpoint as weight_circuit_initial_checkpoint,
    profile_label as weight_circuit_profile_label,
    profile_values as weight_circuit_profile_values,
)
from .composition_loco_t_nested_and_query_weight_spec import (
    NESTED_QUERY_WEIGHT_EVAL_BATCH_SIZE,
    NESTED_QUERY_WEIGHT_HELDOUT_TASK,
    NESTED_QUERY_WEIGHT_OUTPUT_ROOT,
    NESTED_QUERY_WEIGHT_PROFILES,
    NESTED_QUERY_WEIGHT_PROTOCOL_ID,
    NESTED_QUERY_WEIGHT_ROUTES,
    QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256 as NESTED_QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256,
    QUERY_WEIGHT_RAW_CACHE_PATH as NESTED_QUERY_WEIGHT_RAW_CACHE_PATH,
    QUERY_WEIGHT_RAW_CACHE_SHA256 as NESTED_QUERY_WEIGHT_RAW_CACHE_SHA256,
    initial_checkpoint as nested_query_weight_initial_checkpoint,
    profile_label as nested_query_weight_profile_label,
    profile_values as nested_query_weight_profile_values,
)
from .composition_loco_direct_retention4_spec import (
    DIRECT_RETENTION4_ATOMIC_STEPS,
    DIRECT_RETENTION4_CONFIG_ROOT,
    DIRECT_RETENTION4_GROUP_IDS,
    DIRECT_RETENTION4_OUTPUT_ROOT,
    DIRECT_RETENTION4_PROTOCOL_ID,
    DIRECT_RETENTION4_VALIDATION_STEPS,
    RETENTION_COMPOSITION_TASKS,
    RETENTION_HELDOUT_TASK,
    RETENTION_KD_TEMPERATURE,
    direct_retention4_expected_output,
    direct_retention4_group_values,
)
from .composition_loco_direct_retention32_spec import (
    DIRECT_RETENTION32_ATOMIC_STEPS,
    DIRECT_RETENTION32_CONFIG_ROOT,
    DIRECT_RETENTION32_GROUP_IDS,
    DIRECT_RETENTION32_OUTPUT_ROOT,
    DIRECT_RETENTION32_PROTOCOL_ID,
    DIRECT_RETENTION32_SCHEDULER_STEPS,
    DIRECT_RETENTION32_TRAIN_STEPS,
    DIRECT_RETENTION32_VALIDATION_STEPS,
    DIRECT_RETENTION32_WARMUP_STEPS,
    direct_retention32_expected_output,
    direct_retention32_group_values,
    direct_retention32_train_tasks,
)
from .composition_loco_circuit_retention20_spec import (
    CIRCUIT_RETENTION20_ATOMIC_STEPS,
    CIRCUIT_RETENTION20_CONFIG_ROOT,
    CIRCUIT_RETENTION20_GROUP_IDS,
    CIRCUIT_RETENTION20_HELDOUT_TASKS,
    CIRCUIT_RETENTION20_OUTPUT_ROOT,
    CIRCUIT_RETENTION20_PROTOCOL_ID,
    CIRCUIT_RETENTION20_SCHEDULER_STEPS,
    CIRCUIT_RETENTION20_TRAIN_STEPS,
    CIRCUIT_RETENTION20_VALIDATION_STEPS,
    CIRCUIT_RETENTION20_WARMUP_STEPS,
    circuit_retention20_expected_output,
    circuit_retention20_group_values,
    circuit_retention20_train_tasks,
)
from .composition_loco_query_retention20_spec import (
    QUERY_RETENTION20_ATOMIC_STEPS,
    QUERY_RETENTION20_CONFIG_ROOT,
    QUERY_RETENTION20_GROUP_IDS,
    QUERY_RETENTION20_HELDOUT_TASKS,
    QUERY_RETENTION20_OUTPUT_ROOT,
    QUERY_RETENTION20_PROTOCOL_ID,
    QUERY_RETENTION20_SCHEDULER_STEPS,
    QUERY_RETENTION20_TRAIN_STEPS,
    QUERY_RETENTION20_VALIDATION_STEPS,
    QUERY_RETENTION20_WARMUP_STEPS,
    query_retention20_expected_output,
    query_retention20_group_values,
    query_retention20_train_tasks,
)
from .composition_loco_raw_retention20_spec import (
    RAW_RETENTION20_CHECKPOINT,
    RAW_RETENTION20_CHECKPOINT_SHA256,
    RAW_RETENTION20_CACHE_PATH,
    RAW_RETENTION20_CACHE_SHA256,
    RAW_RETENTION20_HISTORICAL_CACHE_SHA256,
    RAW_RETENTION20_ATOMIC_STEPS,
    RAW_RETENTION20_CONFIG_ROOT,
    RAW_RETENTION20_GROUP_IDS,
    RAW_RETENTION20_HELDOUT_TASKS,
    RAW_RETENTION20_OUTPUT_ROOT,
    RAW_RETENTION20_PROTOCOL_ID,
    RAW_RETENTION20_SCHEDULER_STEPS,
    RAW_RETENTION20_TRAIN_STEPS,
    RAW_RETENTION20_VALIDATION_STEPS,
    RAW_RETENTION20_WARMUP_STEPS,
    raw_retention20_expected_output,
    raw_retention20_group_values,
    raw_retention20_train_tasks,
)


CONFIG_FORMAT = "s5_composition_loco_run_config_v1"
LR_HALF_PROTOCOL_ID = "s5_composition_loco_lr_half_v1"
LR_HALF_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_lr_half_v1"
RECREATED_INPUTS_PROTOCOL_ID = "s5_composition_loco_lr_half_recreated_inputs_v1"
RECREATED_INPUTS_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_lr_half_recreated_v1"
ORIGINAL_LR_RECREATED_PROTOCOL_ID = "s5_composition_loco_original_lr_recreated_v1"
ORIGINAL_LR_RECREATED_OUTPUT_ROOT = "outputs/s5_lean_circuit/composition_loco_original_lr_recreated_v1"

# These are deliberately separate from the frozen LR-half identities.  The
# recreated-input wave may consume them only through the new protocol and
# output root; it must never rewrite or relabel the historical inputs.
RECREATED_DIRECT_CHECKPOINT = (
    "outputs/s5_lean_circuit/direct_base_sft_lr_sweep_rebuilt_0907/"
    "lr3e5/training/best_model.pt"
)
RECREATED_DIRECT_CHECKPOINT_SHA256 = "480b29e898c9cf76f3e2d2835fc3e3e1f317b2ede24aff0deaca00ac8f05794e"
RECREATED_DIRECT_CONFIG_SHA256 = "2e362f07469e30b757860d07dfb9b5dfc6b8b2c1358f10c143613929c68d0a55"
RECREATED_RAW_CACHE = (
    "outputs/s5_lean_circuit/hypernet_raw_weight_formal_1p2m_seed42/"
    "shared/raw_weight_cache.pt"
)
RECREATED_RAW_CACHE_SHA256 = "886479c121f743184f5bb89eef2af30afb86a0278cb0b94dbcebfc0c98428334"
HISTORICAL_RAW_CACHE_SHA256 = "a70eb76788bcef9c2ca301ad4a4d3febc94eb247ce1f02cf41ebdccda8bb87e1"


def semantic_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@dataclass(frozen=True)
class CompositionLOCOConfig:
    route: str
    fold_id: str
    heldout_task: str
    train_tasks: tuple[str, ...]
    data: Mapping[str, Any]
    base: Mapping[str, Any]
    initialization: Mapping[str, Any]
    model: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    output_dir: str
    protocol_id: str = PROTOCOL_ID
    format: str = CONFIG_FORMAT
    config_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "protocol_id": self.protocol_id,
            "route": self.route,
            "fold_id": self.fold_id,
            "heldout_task": self.heldout_task,
            "train_tasks": list(self.train_tasks),
            "data": dict(self.data),
            "base": dict(self.base),
            "initialization": dict(self.initialization),
            "model": dict(self.model),
            "training": dict(self.training),
            "evaluation": dict(self.evaluation),
            "output_dir": self.output_dir,
        }

    @property
    def computed_sha256(self) -> str:
        return semantic_sha256(self.to_dict())


def matches_checkpoint_config(public_sha: str, checkpoint_recipe: str | None, verified_file_sha: str) -> bool:
    """Accept a new checkpoint's exact recipe or the recorded immutable model file."""
    if not checkpoint_recipe or not verified_file_sha:
        return False
    if public_sha == checkpoint_recipe:
        return True
    manifest = Path(__file__).resolve().parents[2] / "configs/selected_checkpoint_hashes.json"
    selected = json.loads(manifest.read_text(encoding="utf-8"))
    return selected.get(public_sha) == verified_file_sha


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"composition LOCO config {name} must be an object")
    return value


def _validate_config(config: CompositionLOCOConfig) -> None:
    if "stage2_permuted_control" in config.training:
        from .composition_stage2_permuted import validate_config
        validate_config(config)
        return
    if "performance_completion_run_id" in config.training:
        from .performance_completion import validate_composition
        validate_composition(config)
        return
    from .composition_loco_vector422_spec import VECTOR422_PROTOCOL_ID, validate_vector422_config
    if config.protocol_id == VECTOR422_PROTOCOL_ID:
        validate_vector422_config(config)
        return
    if config.protocol_id == CIRCUIT_SWEEP40_PROTOCOL_ID:
        _validate_circuit_sweep40_config(config)
        return
    if config.protocol_id == MANUAL160_PROTOCOL_ID:
        _validate_manual160_config(config)
        return
    if config.protocol_id == DIRECT_LR32_PROTOCOL_ID:
        _validate_direct_lr32_config(config)
        return
    if config.protocol_id == QUERY_WEIGHT_PROTOCOL_ID:
        _validate_query_weight_config(config)
        return
    if config.protocol_id == QUERY_CIRCUIT_PROTOCOL_ID:
        _validate_query_circuit_config(config)
        return
    if config.protocol_id == WEIGHT_CIRCUIT_PROTOCOL_ID:
        _validate_weight_circuit_config(config)
        return
    if config.protocol_id == NESTED_QUERY_WEIGHT_PROTOCOL_ID:
        _validate_nested_query_weight_config(config)
        return
    if config.protocol_id == DIRECT_RETENTION4_PROTOCOL_ID:
        _validate_direct_retention4_config(config)
        return
    if config.protocol_id == DIRECT_RETENTION32_PROTOCOL_ID:
        _validate_direct_retention32_config(config)
        return
    if config.protocol_id == CIRCUIT_RETENTION20_PROTOCOL_ID:
        if "circuit_refine16_id" in config.training:
            from .composition_loco_circuit_refine16 import validate
            validate(config)
            return
        _validate_circuit_retention20_config(config)
        return
    if config.protocol_id == RAW_RETENTION20_PROTOCOL_ID:
        _validate_raw_retention20_config(config)
        return
    if config.protocol_id == QUERY_RETENTION20_PROTOCOL_ID:
        _validate_query_retention20_config(config)
        return
    if config.protocol_id == LR_HALF_PROTOCOL_ID:
        _validate_lr_half_config(config)
        return
    if config.protocol_id == RECREATED_INPUTS_PROTOCOL_ID:
        _validate_recreated_inputs_config(config)
        return
    if config.protocol_id == ORIGINAL_LR_RECREATED_PROTOCOL_ID:
        _validate_original_lr_recreated_config(config)
        return
    if config.format != CONFIG_FORMAT or config.protocol_id != PROTOCOL_ID:
        raise ValueError("composition LOCO config format/protocol changed")
    if config.route not in ROUTES:
        raise ValueError(f"unknown composition LOCO route: {config.route}")
    expected_train = task_fold(config.heldout_task)
    if tuple(config.train_tasks) != expected_train:
        raise ValueError("composition LOCO train task order changed")
    expected_fold_id = f"holdout_{config.heldout_task}"
    if config.fold_id != expected_fold_id:
        raise ValueError("composition LOCO fold id changed")
    data = config.data
    required_data = {"cache_path", "cache_manifest", "fold_index", "base_ce_path", "validation_bank", "final_bank", "atomic_bank"}
    if required_data - set(data):
        raise ValueError("composition LOCO data paths are incomplete")
    input_treatment = str(data.get("input_treatment", "fixed_log1p_v1"))
    if input_treatment not in {"fixed_log1p_v1", "train_zscore_v1"}:
        raise ValueError("composition LOCO input_treatment is unsupported")
    fingerprint_fields = (
        data.get("fingerprint_manifest"),
        data.get("fingerprint_normalizer"),
        data.get("fingerprint_scope"),
        data.get("fingerprint_normalizer_artifact_sha256"),
    )
    if config.route == "circuit_a" and input_treatment == "train_zscore_v1":
        if not all(isinstance(value, str) and value for value in fingerprint_fields):
            raise ValueError(
                "composition z-score circuit_a requires manifest, normalizer, scope, and artifact identity"
            )
        expected_scope = f"loco_holdout_{config.heldout_task}"
        if data.get("fingerprint_scope") != expected_scope:
            raise ValueError("composition z-score circuit_a must use its matching LOCO scope")
        artifact = str(data.get("fingerprint_normalizer_artifact_sha256"))
        if len(artifact) != 64 or any(char not in "0123456789abcdef" for char in artifact):
            raise ValueError("fingerprint_normalizer_artifact_sha256 must be a lowercase SHA256")
    elif input_treatment != "fixed_log1p_v1" or any(value is not None for value in fingerprint_fields):
        raise ValueError(
            "only circuit_a may opt into train_zscore_v1; historical routes remain fixed-log1p"
        )
    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"composition LOCO training.{key} must equal {expected!r}")
    if float(training.get("warmup_ratio", -1.0)) != 0.03:
        raise ValueError("composition LOCO warmup_ratio must be 0.03")
    if float(training.get("cosine_min_lr_ratio", -1.0)) != 0.1:
        raise ValueError("composition LOCO cosine_min_lr_ratio must be 0.1")
    if training.get("gate_schedule") != "fixed_init":
        raise ValueError("composition LOCO gate_schedule must be fixed_init")
    if config.route == "direct_sft":
        if training.get("precision") != "fp16":
            raise ValueError("Direct-SFT composition LOCO requires FP16+GradScaler")
        if config.model.get("condition_source") not in (None, "none"):
            raise ValueError("Direct-SFT config cannot declare a HyperNet condition")
        if int(config.initialization.get("step", -1)) != 7_629 or float(training.get("learning_rate", -1.0)) != 1.5e-5:
            raise ValueError("Direct-SFT continuation init/LR contract changed")
    else:
        if training.get("precision") != "bf16":
            raise ValueError("HyperNet composition LOCO requires BF16")
        if config.model.get("candidate") != "A":
            raise ValueError("composition LOCO HyperNet routes use Candidate A")
        if config.model.get("condition_source") not in {"query_only", "raw_weight", "circuit"}:
            raise ValueError("HyperNet condition_source is invalid")
        if config.route == "circuit_a" and config.model.get("condition_source") != "circuit":
            raise ValueError("circuit_a must use condition_source=circuit")
        if config.route == "query_only" and config.model.get("condition_source") != "query_only":
            raise ValueError("query_only must use condition_source=query_only")
        if config.route == "raw_weight" and config.model.get("condition_source") != "raw_weight":
            raise ValueError("raw_weight must use condition_source=raw_weight")
        if float(config.model.get("base_preservation_coefficient", -1.0)) != 0.5:
            raise ValueError("base-preservation coefficient must be 0.5")
        if float(config.model.get("effective_delta_l2", -1.0)) != 1.0e-4:
            raise ValueError("effective-delta L2 must be 1e-4")
        expected_steps = {
            "query_only": 30_516,
            "raw_weight": 30_516,
            "circuit_a": 15_258,
        }
        if int(config.initialization.get("step", -1)) != expected_steps[config.route]:
            raise ValueError("HyperNet continuation init step changed")
        expected_pg = {"query_only": 7.5e-5, "raw_weight": 3.75e-5, "circuit_a": 2.5e-5}
        if float(config.model.get("pg_lr", -1.0)) != expected_pg[config.route]:
            raise ValueError("HyperNet continuation PG learning rate changed")
        expected_gate = {"query_only": 0.5, "raw_weight": 0.375, "circuit_a": 0.5}
        if float(config.model.get("gate_max", -1.0)) != expected_gate[config.route]:
            raise ValueError("HyperNet continuation gate contract changed")
        if config.route == "raw_weight" and (
            float(config.model.get("raw_encoder_lr", -1.0)) != 2.0e-4
            or float(config.model.get("raw_output_lr", -1.0)) != 5.0e-4
            or config.model.get("raw_fusion_mode") != "query_cross"
        ):
            raise ValueError("Raw-query continuation LR/fusion contract changed")
    if not config.output_dir:
        raise ValueError("composition LOCO output_dir is required")


def _validate_manual160_config(config: CompositionLOCOConfig) -> None:
    """Validate one member of the explicit 8-profile x 5-fold matrix.

    The historical composition-SFT guard above remains intentionally strict.
    manual160 is a separately namespaced protocol whose only permitted
    differences are the values in ``composition_loco_manual160_spec`` and the
    dedicated output/config roots.  Its wave number is a scheduling detail,
    not a profile number: the training profile is independent of the scheduler.
    """

    if config.format != CONFIG_FORMAT or config.protocol_id != MANUAL160_PROTOCOL_ID:
        raise ValueError("manual160 config format/protocol changed")
    if config.route not in MANUAL160_ROUTES or config.heldout_task not in COMPOSITION_TASKS:
        raise ValueError("invalid manual160 route/fold")
    expected_train = task_fold(config.heldout_task)
    if tuple(config.train_tasks) != expected_train:
        raise ValueError("manual160 train task order changed")
    if config.fold_id != f"holdout_{config.heldout_task}":
        raise ValueError("manual160 fold id changed")
    profile_id = str(config.training.get("manual160_profile_id", ""))
    if profile_id not in MANUAL160_PROFILES:
        raise ValueError("manual160 profile id is missing or invalid")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
    }
    if required_data - set(config.data):
        raise ValueError("manual160 data paths are incomplete")
    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
        "cosine_min_lr_ratio": 0.1,
        "gate_schedule": "fixed_init",
        "manual160_profile_id": profile_id,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"manual160 training.{key} must equal {expected!r}")
    if float(training.get("adam_beta1", -1.0)) != 0.9:
        raise ValueError("manual160 Adam beta1 must be 0.9")
    if float(training.get("max_grad_norm", -1.0)) <= 0:
        raise ValueError("manual160 max_grad_norm must be positive")
    warmup = float(training.get("warmup_ratio", -1.0))
    if warmup not in {0.03, 0.10}:
        raise ValueError("manual160 warmup_ratio must be 0.03 or 0.10")

    expected = profile_values(config.route, profile_id)
    for key in ("learning_rate", "weight_decay", "adam_beta2", "warmup_ratio", "max_grad_norm"):
        if key in expected and float(training.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"manual160 training.{key} does not match profile {profile_id}")
    if config.route == "direct_sft":
        if training.get("precision") != "fp16" or config.model.get("condition_source") not in (None, "none"):
            raise ValueError("manual160 Direct route precision/condition changed")
        if int(config.initialization.get("step", -1)) != 7_629:
            raise ValueError("manual160 Direct initialization step changed")
        if config.initialization.get("checkpoint_path") != RECREATED_DIRECT_CHECKPOINT:
            raise ValueError("manual160 Direct must use the rebuilt 38000-base SFT checkpoint")
        if config.initialization.get("checkpoint_sha256") != RECREATED_DIRECT_CHECKPOINT_SHA256:
            raise ValueError("manual160 Direct checkpoint identity changed")
        if config.initialization.get("checkpoint_config_sha256") != RECREATED_DIRECT_CONFIG_SHA256:
            raise ValueError("manual160 Direct checkpoint config identity changed")
    else:
        if training.get("precision") != "bf16":
            raise ValueError("manual160 HyperNet routes require BF16")
        if config.model.get("candidate") != "A":
            raise ValueError("manual160 HyperNet routes use Candidate A")
        source = {"query_only": "query_only", "raw_weight": "raw_weight", "circuit_a": "circuit"}[config.route]
        if config.model.get("condition_source") != source:
            raise ValueError("manual160 HyperNet condition source changed")
        expected_steps = {"query_only": 30_516, "raw_weight": 30_516, "circuit_a": 15_258}
        if int(config.initialization.get("step", -1)) != expected_steps[config.route]:
            raise ValueError("manual160 HyperNet initialization step changed")
        if config.route == "raw_weight":
            if config.model.get("raw_weight_cache_path") != RECREATED_RAW_CACHE or config.model.get("raw_weight_cache_sha256") != RECREATED_RAW_CACHE_SHA256:
                raise ValueError("manual160 Raw route must use the rebuilt raw cache")
            if config.initialization.get("historical_raw_weight_cache_sha256") != HISTORICAL_RAW_CACHE_SHA256:
                raise ValueError("manual160 Raw historical-cache provenance changed")
            if config.model.get("raw_fusion_mode") != "query_cross":
                raise ValueError("manual160 Raw fusion mode changed")
        elif config.model.get("raw_fusion_mode") != "p2e":
            raise ValueError("manual160 non-Raw fusion mode changed")

    model_keys = (
        "pg_lr", "query_lr_multiplier", "circuit_lr_multiplier", "raw_encoder_lr",
        "raw_output_lr", "gate_max", "base_preservation_coefficient",
        "effective_delta_l2",
    )
    for key in model_keys:
        if key in expected and float(config.model.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"manual160 model.{key} does not match profile {profile_id}")
    if config.route == "direct_sft" and "learning_rate" not in expected:
        raise ValueError("manual160 Direct profile is incomplete")
    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or evaluation.get("selection_tasks") != list(config.train_tasks)
    ):
        raise ValueError("manual160 evaluation bank/selection contract changed")
    if not config.output_dir.startswith(MANUAL160_OUTPUT_ROOT + "/runs/"):
        raise ValueError("manual160 output must use the dedicated output root")


def _validate_query_weight_config(config: CompositionLOCOConfig) -> None:
    """Validate one of the 24-profile Query/Raw one-wave rows.

    QUERY_WEIGHT is a fresh namespace for the single held-out ``T_and_or`` fold.  It
    deliberately shares the frozen four-task training view and route-anchor
    checkpoints with the earlier experiments, but its profile table and
    output/control roots are independent.  Only values explicitly listed in
    ``composition_loco_query_weight_spec`` are allowed to differ from an anchor.
    """

    if config.format != CONFIG_FORMAT or config.protocol_id != QUERY_WEIGHT_PROTOCOL_ID:
        raise ValueError("query_weight config format/protocol changed")
    if config.route not in QUERY_WEIGHT_ROUTES or config.heldout_task != QUERY_WEIGHT_HELDOUT_TASK:
        raise ValueError("invalid query_weight route/fold")
    if tuple(config.train_tasks) != task_fold(QUERY_WEIGHT_HELDOUT_TASK):
        raise ValueError("query_weight train task order changed")
    if config.fold_id != f"holdout_{QUERY_WEIGHT_HELDOUT_TASK}":
        raise ValueError("query_weight fold id changed")

    profile_id = str(config.training.get("query_weight_profile_id", ""))
    if profile_id not in QUERY_WEIGHT_PROFILES:
        raise ValueError("query_weight profile id is missing or invalid")
    if config.training.get("query_weight_profile_label") != query_weight_profile_label(profile_id):
        raise ValueError("query_weight profile label changed")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
    }
    if required_data - set(config.data):
        raise ValueError("query_weight data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("query_weight must use the existing fixed-log1p Vector397 treatment")
    fingerprint_fields = (
        config.data.get("fingerprint_manifest"),
        config.data.get("fingerprint_normalizer"),
        config.data.get("fingerprint_scope"),
        config.data.get("fingerprint_normalizer_artifact_sha256"),
    )
    if any(value is not None for value in fingerprint_fields):
        raise ValueError("query_weight cannot declare a train-fitted fingerprint normalizer")

    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
        "precision": "bf16",
        "gate_schedule": "fixed_init",
        "query_weight_profile_id": profile_id,
        "query_weight_profile_label": query_weight_profile_label(profile_id),
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"query_weight training.{key} must equal {expected!r}")
    expected = query_weight_profile_values(config.route, profile_id)
    for key in (
        "weight_decay", "adam_beta1", "adam_beta2", "warmup_ratio",
        "max_grad_norm", "cosine_min_lr_ratio",
    ):
        if float(training.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"query_weight training.{key} does not match profile {profile_id}")
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise ValueError("query_weight max_grad_norm must be positive")

    base = dict(config.base)
    for key, expected_value in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected_value:
            raise ValueError(f"query_weight base identity field changed: {key}")
    initialization = dict(config.initialization)
    if config.route == "query_only":
        expected_checkpoint = QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT
        expected_checkpoint_sha = QUERY_WEIGHT_QUERY_INITIAL_CHECKPOINT_SHA256
        expected_group = "QO02"
        expected_source = "query_only"
        expected_fusion = "p2e"
    else:
        expected_checkpoint = QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT
        expected_checkpoint_sha = QUERY_WEIGHT_RAW_INITIAL_CHECKPOINT_SHA256
        expected_group = "P14"
        expected_source = "raw_weight"
        expected_fusion = "query_cross"
    if (
        int(initialization.get("step", -1)) != 30_516
        or initialization.get("checkpoint_path") != expected_checkpoint
        or initialization.get("checkpoint_sha256") != expected_checkpoint_sha
        or config.model.get("candidate") != "A"
        or config.model.get("group") != expected_group
        or config.model.get("condition_source") != expected_source
        or config.model.get("raw_fusion_mode") != expected_fusion
    ):
        raise ValueError("query_weight route-anchor initialization or fusion identity changed")
    if config.route == "raw_weight":
        if (
            config.model.get("raw_weight_cache_path") != QUERY_WEIGHT_RAW_CACHE_PATH
            or config.model.get("raw_weight_cache_sha256") != QUERY_WEIGHT_RAW_CACHE_SHA256
            or initialization.get("historical_raw_weight_cache_sha256")
            != QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256
        ):
            raise ValueError("query_weight Raw route must use the rebuilt raw cache and provenance")

    model_keys = (
        "pg_lr", "query_lr_multiplier", "raw_encoder_lr", "raw_output_lr",
        "gate_max", "base_preservation_coefficient", "effective_delta_l2",
    )
    for key in model_keys:
        if key in expected and float(config.model.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"query_weight model.{key} does not match profile {profile_id}")
    if float(config.model.get("gate_max", -1.0)) != float(expected["gate_max"]):
        raise ValueError("query_weight gate must remain at the route anchor")

    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or evaluation.get("selection_tasks") != list(config.train_tasks)
        or evaluation.get("simple_nested_diagnostic") is not True
    ):
        raise ValueError("query_weight evaluation bank/selection contract changed")
    expected_output = f"{QUERY_WEIGHT_OUTPUT_ROOT}/runs/profile_{profile_id}/{QUERY_WEIGHT_HELDOUT_TASK}/{config.route}"
    if config.output_dir != expected_output:
        raise ValueError("query_weight output path changed")


def _validate_query_circuit_config(config: CompositionLOCOConfig) -> None:
    """Validate one paired Query-only/Circuit-A row for the Query/Circuit sweep."""

    if config.format != CONFIG_FORMAT or config.protocol_id != QUERY_CIRCUIT_PROTOCOL_ID:
        raise ValueError("query_circuit config format/protocol changed")
    if config.route not in QUERY_CIRCUIT_ROUTES or config.heldout_task != QUERY_CIRCUIT_HELDOUT_TASK:
        raise ValueError("invalid query_circuit route/fold")
    if tuple(config.train_tasks) != task_fold(QUERY_CIRCUIT_HELDOUT_TASK):
        raise ValueError("query_circuit train task order changed")
    if config.fold_id != f"holdout_{QUERY_CIRCUIT_HELDOUT_TASK}":
        raise ValueError("query_circuit fold id changed")

    profile_id = str(config.training.get("query_circuit_profile_id", ""))
    if profile_id not in QUERY_CIRCUIT_PROFILES:
        raise ValueError("query_circuit profile id is missing or invalid")
    if config.training.get("query_circuit_profile_label") != query_circuit_profile_label(config.route, profile_id):
        raise ValueError("query_circuit profile label changed")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
    }
    if required_data - set(config.data):
        raise ValueError("query_circuit data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("query_circuit must use the existing fixed-log1p Vector397 treatment")
    fingerprint_fields = (
        config.data.get("fingerprint_manifest"),
        config.data.get("fingerprint_normalizer"),
        config.data.get("fingerprint_scope"),
        config.data.get("fingerprint_normalizer_artifact_sha256"),
    )
    if any(value is not None for value in fingerprint_fields):
        raise ValueError("query_circuit cannot declare a train-fitted fingerprint normalizer")

    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
        "precision": "bf16",
        "gate_schedule": "fixed_init",
        "query_circuit_profile_id": profile_id,
        "query_circuit_profile_label": query_circuit_profile_label(config.route, profile_id),
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"query_circuit training.{key} must equal {expected!r}")
    if int(config.evaluation.get("eval_batch_size", -1)) != QUERY_CIRCUIT_EVAL_BATCH_SIZE:
        raise ValueError("query_circuit evaluation batch size must be the declared batched value")

    expected = query_circuit_profile_values(config.route, profile_id)
    for key in (
        "weight_decay", "adam_beta1", "adam_beta2", "warmup_ratio",
        "max_grad_norm", "cosine_min_lr_ratio",
    ):
        if float(training.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"query_circuit training.{key} does not match profile {profile_id}")
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise ValueError("query_circuit max_grad_norm must be positive")

    base = dict(config.base)
    for key, expected_value in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected_value:
            raise ValueError(f"query_circuit base identity field changed: {key}")
    checkpoint_path, checkpoint_sha, checkpoint_step = query_circuit_initial_checkpoint(config.route)
    initialization = dict(config.initialization)
    expected_source = "query_only" if config.route == "query_only" else "circuit"
    expected_fusion = "p2e"
    if (
        int(initialization.get("step", -1)) != checkpoint_step
        or initialization.get("checkpoint_path") != checkpoint_path
        or initialization.get("checkpoint_sha256") != checkpoint_sha
        or config.model.get("candidate") != "A"
        or config.model.get("condition_source") != expected_source
        or config.model.get("raw_fusion_mode") != expected_fusion
    ):
        raise ValueError("query_circuit route-anchor initialization or fusion identity changed")

    model_keys = (
        "pg_lr", "query_lr_multiplier", "circuit_lr_multiplier",
        "gate_max", "base_preservation_coefficient", "effective_delta_l2",
    )
    for key in model_keys:
        if key in expected and float(config.model.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"query_circuit model.{key} does not match profile {profile_id}")
    if float(config.model.get("gate_max", -1.0)) != 0.5:
        raise ValueError("query_circuit gate must remain fixed at 0.5")
    if config.route == "query_only" and "circuit_lr_multiplier" in config.model:
        raise ValueError("query_circuit Query-only config cannot declare a Circuit LR multiplier")
    if config.route == "circuit_a" and "circuit_lr_multiplier" not in config.model:
        raise ValueError("query_circuit Circuit-A config must declare a Circuit LR multiplier")

    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or evaluation.get("selection_tasks") != list(config.train_tasks)
        or evaluation.get("simple_nested_diagnostic") is not True
    ):
        raise ValueError("query_circuit evaluation bank/selection contract changed")
    expected_output = f"{QUERY_CIRCUIT_OUTPUT_ROOT}/runs/profile_{profile_id}/{QUERY_CIRCUIT_HELDOUT_TASK}/{config.route}"
    if config.output_dir != expected_output:
        raise ValueError("query_circuit output path changed")


def _validate_weight_circuit_config(config: CompositionLOCOConfig) -> None:
    """Validate one WEIGHT_CIRCUIT row with a route-specific held-out composition.

    Unlike QUERY_WEIGHT/QUERY_CIRCUIT, WEIGHT_CIRCUIT intentionally has two folds in one matrix.  The
    route-to-fold mapping is sealed in ``composition_loco_weight_circuit_spec`` and is
    checked for every config so a controller cannot accidentally train/evaluate
    a route on the other route's held-out task.
    """

    if config.format != CONFIG_FORMAT or config.protocol_id != WEIGHT_CIRCUIT_PROTOCOL_ID:
        raise ValueError("weight_circuit config format/protocol changed")
    if config.route not in WEIGHT_CIRCUIT_ROUTES:
        raise ValueError("invalid weight_circuit route")
    expected_heldout = weight_circuit_heldout_task(config.route)
    if config.heldout_task != expected_heldout:
        raise ValueError("weight_circuit route/held-out task mapping changed")
    if tuple(config.train_tasks) != task_fold(expected_heldout):
        raise ValueError("weight_circuit train task order changed")
    if config.fold_id != f"holdout_{expected_heldout}":
        raise ValueError("weight_circuit fold id changed")

    profile_id = str(config.training.get("weight_circuit_profile_id", ""))
    if profile_id not in WEIGHT_CIRCUIT_PROFILES:
        raise ValueError("weight_circuit profile id is missing or invalid")
    if config.training.get("weight_circuit_profile_label") != weight_circuit_profile_label(config.route, profile_id):
        raise ValueError("weight_circuit profile label changed")
    if config.training.get("weight_circuit_heldout_route") != config.route:
        raise ValueError("weight_circuit route metadata changed")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
    }
    if required_data - set(config.data):
        raise ValueError("weight_circuit data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("weight_circuit must use the existing fixed-log1p Vector397 treatment")
    fingerprint_fields = (
        config.data.get("fingerprint_manifest"),
        config.data.get("fingerprint_normalizer"),
        config.data.get("fingerprint_scope"),
        config.data.get("fingerprint_normalizer_artifact_sha256"),
    )
    if any(value is not None for value in fingerprint_fields):
        raise ValueError("weight_circuit cannot declare a train-fitted fingerprint normalizer")

    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
        "precision": "bf16",
        "gate_schedule": "fixed_init",
        "weight_circuit_profile_id": profile_id,
        "weight_circuit_profile_label": weight_circuit_profile_label(config.route, profile_id),
        "weight_circuit_heldout_route": config.route,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"weight_circuit training.{key} must equal {expected!r}")
    expected = weight_circuit_profile_values(config.route, profile_id)
    for key in (
        "weight_decay", "adam_beta1", "adam_beta2", "warmup_ratio",
        "max_grad_norm", "cosine_min_lr_ratio",
    ):
        if float(training.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"weight_circuit training.{key} does not match profile {profile_id}")
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise ValueError("weight_circuit max_grad_norm must be positive")

    base = dict(config.base)
    for key, expected_value in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected_value:
            raise ValueError(f"weight_circuit base identity field changed: {key}")
    checkpoint_path, checkpoint_sha, checkpoint_step = weight_circuit_initial_checkpoint(config.route)
    initialization = dict(config.initialization)
    expected_source = "raw_weight" if config.route == "raw_weight" else "circuit"
    expected_fusion = "query_cross" if config.route == "raw_weight" else "p2e"
    expected_group = "P14" if config.route == "raw_weight" else "P03"
    if (
        int(initialization.get("step", -1)) != checkpoint_step
        or initialization.get("checkpoint_path") != checkpoint_path
        or initialization.get("checkpoint_sha256") != checkpoint_sha
        or config.model.get("candidate") != "A"
        or config.model.get("group") != expected_group
        or config.model.get("condition_source") != expected_source
        or config.model.get("raw_fusion_mode") != expected_fusion
    ):
        raise ValueError("weight_circuit route-anchor initialization or fusion identity changed")
    if config.route == "raw_weight":
        if (
            config.model.get("raw_weight_cache_path") != QUERY_WEIGHT_RAW_CACHE_PATH
            or config.model.get("raw_weight_cache_sha256") != QUERY_WEIGHT_RAW_CACHE_SHA256
            or initialization.get("historical_raw_weight_cache_sha256")
            != QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256
        ):
            raise ValueError("weight_circuit Raw-weight route must use the rebuilt cache with historical provenance")
    else:
        if "raw_weight_cache_path" in config.model or "raw_encoder_lr" in config.model:
            raise ValueError("weight_circuit Circuit-A config contains Raw-weight-only fields")

    model_keys = (
        "pg_lr", "query_lr_multiplier", "circuit_lr_multiplier",
        "raw_encoder_lr", "raw_output_lr", "gate_max",
        "base_preservation_coefficient", "effective_delta_l2", "raw_fusion_mode",
    )
    for key in model_keys:
        if key in expected:
            if key == "raw_fusion_mode":
                if config.model.get(key) != expected[key]:
                    raise ValueError(f"weight_circuit model.{key} does not match profile {profile_id}")
            elif float(config.model.get(key, float("nan"))) != float(expected[key]):
                raise ValueError(f"weight_circuit model.{key} does not match profile {profile_id}")
    expected_gate = 0.375 if config.route == "raw_weight" else 0.5
    if float(config.model.get("gate_max", -1.0)) != expected_gate:
        raise ValueError("weight_circuit gate must remain fixed at the route anchor")
    if config.route == "raw_weight" and "circuit_lr_multiplier" in config.model:
        raise ValueError("weight_circuit Raw-weight config cannot declare a Circuit LR multiplier")
    if config.route == "circuit_a" and "circuit_lr_multiplier" not in config.model:
        raise ValueError("weight_circuit Circuit-A config must declare a Circuit LR multiplier")

    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or evaluation.get("selection_tasks") != list(config.train_tasks)
        or evaluation.get("simple_nested_diagnostic") is not True
        or int(evaluation.get("eval_batch_size", -1)) != WEIGHT_CIRCUIT_EVAL_BATCH_SIZE
    ):
        raise ValueError("weight_circuit evaluation/selection bank contract changed")
    expected_output = f"{WEIGHT_CIRCUIT_OUTPUT_ROOT}/runs/profile_{profile_id}/{expected_heldout}/{config.route}"
    if config.output_dir != expected_output:
        raise ValueError("weight_circuit output path changed")


def _validate_nested_query_weight_config(config: CompositionLOCOConfig) -> None:
    """Validate a Query-only/Raw-weight row for the nested-AND fold.

    NESTED_QUERY_WEIGHT reuses the first 16 approved QUERY_WEIGHT profiles, but it owns a separate
    held-out fold, protocol identity, output root, and eight-task selector.
    Keeping this guard separate prevents a T_and_or config from being
    relabelled as a T_nested_and run.
    """

    if config.format != CONFIG_FORMAT or config.protocol_id != NESTED_QUERY_WEIGHT_PROTOCOL_ID:
        raise ValueError("nested_query_weight config format/protocol changed")
    if config.route not in NESTED_QUERY_WEIGHT_ROUTES or config.heldout_task != NESTED_QUERY_WEIGHT_HELDOUT_TASK:
        raise ValueError("invalid nested_query_weight route/fold")
    if tuple(config.train_tasks) != task_fold(NESTED_QUERY_WEIGHT_HELDOUT_TASK):
        raise ValueError("nested_query_weight train task order changed")
    if config.fold_id != f"holdout_{NESTED_QUERY_WEIGHT_HELDOUT_TASK}":
        raise ValueError("nested_query_weight fold id changed")

    profile_id = str(config.training.get("nested_query_weight_profile_id", ""))
    if profile_id not in NESTED_QUERY_WEIGHT_PROFILES:
        raise ValueError("nested_query_weight profile id is missing or invalid")
    if config.training.get("nested_query_weight_profile_label") != nested_query_weight_profile_label(config.route, profile_id):
        raise ValueError("nested_query_weight profile label changed")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
    }
    if required_data - set(config.data):
        raise ValueError("nested_query_weight data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("nested_query_weight must use the existing fixed-log1p Vector397 treatment")
    fingerprint_fields = (
        config.data.get("fingerprint_manifest"),
        config.data.get("fingerprint_normalizer"),
        config.data.get("fingerprint_scope"),
        config.data.get("fingerprint_normalizer_artifact_sha256"),
    )
    if any(value is not None for value in fingerprint_fields):
        raise ValueError("nested_query_weight cannot declare a train-fitted fingerprint normalizer")

    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
        "precision": "bf16",
        "gate_schedule": "fixed_init",
        "nested_query_weight_profile_id": profile_id,
        "nested_query_weight_profile_label": nested_query_weight_profile_label(config.route, profile_id),
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"nested_query_weight training.{key} must equal {expected!r}")
    if int(config.evaluation.get("eval_batch_size", -1)) != NESTED_QUERY_WEIGHT_EVAL_BATCH_SIZE:
        raise ValueError("nested_query_weight evaluation batch size must be the declared batched value")

    expected = nested_query_weight_profile_values(config.route, profile_id)
    for key in (
        "weight_decay", "adam_beta1", "adam_beta2", "warmup_ratio",
        "max_grad_norm", "cosine_min_lr_ratio",
    ):
        if float(training.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"nested_query_weight training.{key} does not match profile {profile_id}")
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise ValueError("nested_query_weight max_grad_norm must be positive")

    base = dict(config.base)
    for key, expected_value in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected_value:
            raise ValueError(f"nested_query_weight base identity field changed: {key}")

    checkpoint_path, checkpoint_sha, checkpoint_step = nested_query_weight_initial_checkpoint(config.route)
    initialization = dict(config.initialization)
    expected_source = config.route
    expected_fusion = "p2e" if config.route == "query_only" else "query_cross"
    expected_group = "QO02" if config.route == "query_only" else "P14"
    if (
        int(initialization.get("step", -1)) != checkpoint_step
        or initialization.get("checkpoint_path") != checkpoint_path
        or initialization.get("checkpoint_sha256") != checkpoint_sha
        or config.model.get("candidate") != "A"
        or config.model.get("group") != expected_group
        or config.model.get("condition_source") != expected_source
        or config.model.get("raw_fusion_mode") != expected_fusion
    ):
        raise ValueError("nested_query_weight route-anchor initialization or fusion identity changed")

    if config.route == "raw_weight":
        if (
            config.model.get("raw_weight_cache_path") != NESTED_QUERY_WEIGHT_RAW_CACHE_PATH
            or config.model.get("raw_weight_cache_sha256") != NESTED_QUERY_WEIGHT_RAW_CACHE_SHA256
            or initialization.get("historical_raw_weight_cache_sha256")
            != NESTED_QUERY_WEIGHT_HISTORICAL_RAW_CACHE_SHA256
        ):
            raise ValueError("nested_query_weight Raw-weight route must use the rebuilt cache with provenance")
    else:
        if any(key in config.model for key in ("raw_weight_cache_path", "raw_weight_cache_sha256", "raw_encoder_lr", "raw_output_lr")):
            raise ValueError("nested_query_weight Query-only config contains Raw-weight-only fields")

    model_keys = (
        "pg_lr", "query_lr_multiplier", "raw_encoder_lr", "raw_output_lr",
        "gate_max", "base_preservation_coefficient", "effective_delta_l2",
        "raw_fusion_mode",
    )
    for key in model_keys:
        if key not in expected:
            continue
        observed = config.model.get(key)
        if key == "raw_fusion_mode":
            if observed != expected[key]:
                raise ValueError(f"nested_query_weight model.{key} does not match profile {profile_id}")
        elif float(observed if observed is not None else float("nan")) != float(expected[key]):
            raise ValueError(f"nested_query_weight model.{key} does not match profile {profile_id}")
    expected_gate = 0.5 if config.route == "query_only" else 0.375
    if float(config.model.get("gate_max", -1.0)) != expected_gate:
        raise ValueError("nested_query_weight gate must remain at the route anchor")
    if "circuit_lr_multiplier" in config.model:
        raise ValueError("nested_query_weight Query/Raw config cannot declare a Circuit LR multiplier")

    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or evaluation.get("selection_tasks") != list(config.train_tasks)
        or evaluation.get("simple_nested_diagnostic") is not True
    ):
        raise ValueError("nested_query_weight evaluation/selection bank contract changed")
    expected_output = f"{NESTED_QUERY_WEIGHT_OUTPUT_ROOT}/runs/profile_{profile_id}/{NESTED_QUERY_WEIGHT_HELDOUT_TASK}/{config.route}"
    if config.output_dir != expected_output:
        raise ValueError("nested_query_weight output path changed")


def _validate_circuit_sweep40_config(config: CompositionLOCOConfig) -> None:
    """Validate one fold cell in the 40-lane Circuit-A optimizer sweep.

    The sweep has one profile-level SiFlow job per profile.  The job invokes
    this same config contract for the five folds serially; the fold index is
    therefore a data identity, not a scheduling wave number.
    """

    if config.format != CONFIG_FORMAT or config.protocol_id != CIRCUIT_SWEEP40_PROTOCOL_ID:
        raise ValueError("circuit sweep40 config format/protocol changed")
    if config.route not in CIRCUIT_SWEEP40_ROUTES or config.heldout_task not in COMPOSITION_TASKS:
        raise ValueError("invalid circuit sweep40 route/fold")
    if tuple(config.train_tasks) != task_fold(config.heldout_task):
        raise ValueError("circuit sweep40 train task order changed")
    if config.fold_id != f"holdout_{config.heldout_task}":
        raise ValueError("circuit sweep40 fold id changed")

    profile_id = str(config.training.get("circuit_sweep40_profile_id", ""))
    if profile_id not in CIRCUIT_SWEEP40_PROFILES:
        raise ValueError("circuit sweep40 profile id is missing or invalid")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
    }
    if required_data - set(config.data):
        raise ValueError("circuit sweep40 data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("circuit sweep40 must use the existing fixed-log1p Vector397 treatment")
    fingerprint_fields = (
        config.data.get("fingerprint_manifest"),
        config.data.get("fingerprint_normalizer"),
        config.data.get("fingerprint_scope"),
        config.data.get("fingerprint_normalizer_artifact_sha256"),
    )
    if any(value is not None for value in fingerprint_fields):
        raise ValueError("circuit sweep40 cannot declare a train-fitted fingerprint normalizer")

    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
        "gate_schedule": "fixed_init",
        "precision": "bf16",
        "circuit_sweep40_profile_id": profile_id,
        "circuit_sweep40_profile_label": circuit_sweep40_profile_label(profile_id),
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"circuit sweep40 training.{key} must equal {expected!r}")

    expected = circuit_sweep40_profile_values(profile_id)
    for key in (
        "weight_decay", "adam_beta1", "adam_beta2", "warmup_ratio",
        "max_grad_norm", "cosine_min_lr_ratio",
    ):
        if float(training.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"circuit sweep40 training.{key} does not match profile {profile_id}")
    if float(training.get("max_grad_norm", 0.0)) <= 0:
        raise ValueError("circuit sweep40 max_grad_norm must be positive")

    base = dict(config.base)
    initialization = dict(config.initialization)
    if (
        int(base.get("step", -1)) != CIRCUIT_SWEEP40_BASE_STEP
        or int(initialization.get("step", -1)) != CIRCUIT_SWEEP40_INITIAL_STEP
        or initialization.get("checkpoint_path") != CIRCUIT_SWEEP40_INITIAL_CHECKPOINT
        or initialization.get("checkpoint_sha256") != CIRCUIT_SWEEP40_INITIAL_CHECKPOINT_SHA256
        or config.model.get("candidate") != "A"
        or config.model.get("group") != "P03"
        or config.model.get("condition_source") != "circuit"
        or config.model.get("raw_fusion_mode") != "p2e"
    ):
        raise ValueError("circuit sweep40 Base/A-P03 initialization or route identity changed")
    for key, expected_value in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected_value:
            raise ValueError(f"circuit sweep40 base identity field changed: {key}")

    model_keys = (
        "pg_lr", "query_lr_multiplier", "circuit_lr_multiplier", "gate_max",
        "base_preservation_coefficient", "effective_delta_l2",
    )
    for key in model_keys:
        if float(config.model.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"circuit sweep40 model.{key} does not match profile {profile_id}")
    if float(config.model.get("gate_max", -1.0)) != 0.5:
        raise ValueError("circuit sweep40 gate must remain fixed at 0.5")

    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or evaluation.get("selection_tasks") != list(config.train_tasks)
    ):
        raise ValueError("circuit sweep40 evaluation bank/selection contract changed")
    expected_outputs = {
        f"{root}/runs/profile_{profile_id}/{config.heldout_task}/circuit_a"
        for root in (CIRCUIT_SWEEP40_OUTPUT_ROOT, CIRCUIT_T_AND_OR40_OUTPUT_ROOT)
    }
    if config.output_dir not in expected_outputs:
        raise ValueError("circuit sweep40 output path changed")


def _validate_direct_lr32_config(config: CompositionLOCOConfig) -> None:
    """Validate one row of the 32-profile Direct-SFT sweep.

    Unlike the historical recreated Direct route, this protocol starts from
    the immutable Stage-A Base checkpoint itself.  The only swept values are
    the ordinary Direct optimizer settings sealed in the profile table.
    """

    if config.format != CONFIG_FORMAT or config.protocol_id != DIRECT_LR32_PROTOCOL_ID:
        raise ValueError("direct_lr32 config format/protocol changed")
    if config.route != "direct_sft" or config.heldout_task not in COMPOSITION_TASKS:
        raise ValueError("direct_lr32 route/fold is invalid")
    if tuple(config.train_tasks) != task_fold(config.heldout_task):
        raise ValueError("direct_lr32 train task order changed")
    if config.fold_id != f"holdout_{config.heldout_task}":
        raise ValueError("direct_lr32 fold id changed")

    profile_id = str(config.training.get("direct_lr32_profile_id", ""))
    if profile_id not in DIRECT_LR32_PROFILES:
        raise ValueError("direct_lr32 profile id is missing or invalid")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "validation_bank",
        "final_bank", "atomic_bank", "tokenizer_path",
    }
    if required_data - set(config.data):
        raise ValueError("direct_lr32 data paths are incomplete")

    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(VALIDATION_STEPS),
        "seed": 42,
        "cosine_min_lr_ratio": 0.1,
        "gate_schedule": "fixed_init",
        "direct_lr32_profile_id": profile_id,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"direct_lr32 training.{key} must equal {expected!r}")
    if training.get("precision") != "fp16":
        raise ValueError("direct_lr32 Direct route requires FP16")

    expected = direct_lr32_profile_values(profile_id)
    for key in (
        "learning_rate", "weight_decay", "adam_beta1", "adam_beta2",
        "warmup_ratio", "max_grad_norm",
    ):
        if float(training.get(key, float("nan"))) != float(expected[key]):
            raise ValueError(f"direct_lr32 training.{key} does not match profile {profile_id}")

    base = dict(config.base)
    initialization = dict(config.initialization)
    if (
        int(base.get("step", -1)) != 38_000
        or int(initialization.get("step", -1)) != 38_000
        or initialization.get("checkpoint_path") != base.get("checkpoint_path")
        or initialization.get("checkpoint_sha256") != base.get("checkpoint_sha256")
        or initialization.get("checkpoint_config_sha256") != base.get("config_sha256")
        or initialization.get("historical_direct_checkpoint") is not False
    ):
        raise ValueError("direct_lr32 must initialize directly from the Stage-A Base checkpoint")
    if config.model.get("condition_source") not in (None, "none"):
        raise ValueError("direct_lr32 cannot declare a HyperNet condition")
    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or evaluation.get("selection_tasks") != list(config.train_tasks)
    ):
        raise ValueError("direct_lr32 evaluation contract changed")
    if not config.output_dir.startswith(DIRECT_LR32_OUTPUT_ROOT + "/runs/"):
        raise ValueError("direct_lr32 output must use the dedicated output root")


def _validate_circuit_retention20_config(config: CompositionLOCOConfig) -> None:
    """Validate one Circuit-A row of the 20-cell retention protocol."""

    if config.format != CONFIG_FORMAT or config.protocol_id != CIRCUIT_RETENTION20_PROTOCOL_ID:
        raise ValueError("circuit_retention20 config format/protocol changed")
    if config.route != "circuit_a" or config.heldout_task not in CIRCUIT_RETENTION20_HELDOUT_TASKS:
        raise ValueError("circuit_retention20 route/heldout task is invalid")
    expected_tasks = circuit_retention20_train_tasks(config.heldout_task)
    if tuple(config.train_tasks) != expected_tasks:
        raise ValueError("circuit_retention20 train task order changed")
    if config.fold_id != f"holdout_{config.heldout_task}":
        raise ValueError("circuit_retention20 fold id changed")
    group_id = str(config.training.get("circuit_retention20_group_id", ""))
    if group_id not in CIRCUIT_RETENTION20_GROUP_IDS:
        raise ValueError("circuit_retention20 group id is missing or invalid")
    values = circuit_retention20_group_values(group_id)
    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
        "atomic_replay",
    }
    if required_data - set(config.data):
        raise ValueError("circuit_retention20 data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("circuit_retention20 must use fixed-log1p Vector397")
    if any(
        config.data.get(key) is not None
        for key in (
            "fingerprint_manifest", "fingerprint_normalizer", "fingerprint_scope",
            "fingerprint_normalizer_artifact_sha256",
        )
    ):
        raise ValueError("circuit_retention20 does not use a train-fitted fingerprint normalizer")
    replay = config.data.get("atomic_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("circuit_retention20 atomic_replay must be an object")
    replay_required = {
        "cache_path", "cache_manifest", "source_config_path",
        "expected_source_config_sha256", "expected_prefix", "expected_n_ctx",
    }
    if replay_required - set(replay):
        raise ValueError("circuit_retention20 atomic replay provenance is incomplete")
    if int(replay.get("expected_prefix", -1)) != 1_200_000 or int(replay.get("expected_n_ctx", -1)) != N_CTX:
        raise ValueError("circuit_retention20 atomic replay dimensions changed")

    training = config.training
    exact = {
        "total_steps": CIRCUIT_RETENTION20_TRAIN_STEPS,
        "scheduler_total_steps": CIRCUIT_RETENTION20_SCHEDULER_STEPS,
        "total_tokens": CIRCUIT_RETENTION20_TRAIN_STEPS * GLOBAL_BATCH_SIZE * N_CTX,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(CIRCUIT_RETENTION20_VALIDATION_STEPS),
        "seed": 42,
        "precision": "bf16",
        "warmup_ratio": 0.03,
        "warmup_steps": CIRCUIT_RETENTION20_WARMUP_STEPS,
        "cosine_min_lr_ratio": 0.01,
        "gate_schedule": "fixed_init",
        "circuit_retention20_group_id": group_id,
        "replay_composition_per_batch": int(values["composition_per_batch"]),
        "replay_atomic_per_batch": int(values["atomic_per_batch"]),
        "replay_composition_per_task": int(values["composition_per_task"]),
        "replay_atomic_per_task": int(values["atomic_per_task"]),
        "kd_temperature": 2.0,
        "kd_lambda": 0.25,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"circuit_retention20 training.{key} must equal {expected!r}")
    for key, expected in {
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
    }.items():
        if float(training.get(key, float("nan"))) != expected:
            raise ValueError(f"circuit_retention20 training.{key} changed")

    model = config.model
    model_expected = {
        "candidate": "A",
        "condition_source": "circuit",
        "group": "P03",
        "raw_fusion_mode": "p2e",
        "pg_lr": float(values["pg_lr"]),
        "query_lr_multiplier": float(values["query_lr_multiplier"]),
        "circuit_lr_multiplier": float(values["circuit_lr_multiplier"]),
        "gate_max": 0.5,
        "base_preservation_coefficient": 0.5,
        "effective_delta_l2": 1.0e-4,
    }
    for key, expected in model_expected.items():
        observed = model.get(key)
        if observed != expected and not (
            isinstance(expected, float) and float(observed if observed is not None else float("nan")) == expected
        ):
            raise ValueError(f"circuit_retention20 model.{key} changed")

    base = dict(config.base)
    initialization = dict(config.initialization)
    if (
        int(base.get("step", -1)) != 38_000
        or int(initialization.get("step", -1)) != 15_258
        or initialization.get("checkpoint_path") != (
            "outputs/s5_lean_circuit/hypernet_ab_formal_1p2m_seed42/"
            "formal_lr_gate_r4a8/a_p03/checkpoints/model_step0015258.pt"
        )
        or initialization.get("checkpoint_sha256") != "d948792b7206d36f1a06697428fcb928aefaf7db120558529fae3c38ab957cde"
    ):
        raise ValueError("circuit_retention20 must initialize from A-P03 step15258")
    for key, expected in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected:
            raise ValueError(f"circuit_retention20 base identity field changed: {key}")
    evaluation = config.evaluation
    if (
        evaluation.get("heldout_task") != config.heldout_task
        or int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or evaluation.get("selection_tasks") != list(expected_tasks)
        or evaluation.get("atomic_steps") != list(CIRCUIT_RETENTION20_ATOMIC_STEPS)
    ):
        raise ValueError("circuit_retention20 evaluation contract changed")
    if config.output_dir != circuit_retention20_expected_output(group_id, config.heldout_task):
        raise ValueError("circuit_retention20 output path changed")


def _validate_query_retention20_config(config: CompositionLOCOConfig) -> None:
    """Validate one Query-only row of the 20-cell retention protocol."""

    if config.format != CONFIG_FORMAT or config.protocol_id != QUERY_RETENTION20_PROTOCOL_ID:
        raise ValueError("query_retention20 config format/protocol changed")
    if config.route != "query_only" or config.heldout_task not in QUERY_RETENTION20_HELDOUT_TASKS:
        raise ValueError("query_retention20 route/heldout task is invalid")
    expected_tasks = query_retention20_train_tasks(config.heldout_task)
    if tuple(config.train_tasks) != expected_tasks:
        raise ValueError("query_retention20 train task order changed")
    if config.fold_id != f"holdout_{config.heldout_task}":
        raise ValueError("query_retention20 fold id changed")
    group_id = str(config.training.get("query_retention20_group_id", ""))
    if group_id not in QUERY_RETENTION20_GROUP_IDS:
        raise ValueError("query_retention20 group id is missing or invalid")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
        "atomic_replay",
    }
    if required_data - set(config.data):
        raise ValueError("query_retention20 data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("query_retention20 must use fixed-log1p cache treatment")
    if any(
        config.data.get(key) is not None
        for key in (
            "fingerprint_manifest", "fingerprint_normalizer", "fingerprint_scope",
            "fingerprint_normalizer_artifact_sha256",
        )
    ):
        raise ValueError("query_retention20 cannot declare a train-fitted fingerprint normalizer")
    replay = config.data.get("atomic_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("query_retention20 atomic_replay must be an object")
    replay_required = {
        "cache_path", "cache_manifest", "source_config_path",
        "expected_source_config_sha256", "expected_prefix", "expected_n_ctx",
    }
    if replay_required - set(replay):
        raise ValueError("query_retention20 atomic replay provenance is incomplete")
    if int(replay.get("expected_prefix", -1)) != 1_200_000 or int(replay.get("expected_n_ctx", -1)) != N_CTX:
        raise ValueError("query_retention20 atomic replay dimensions changed")

    values = query_retention20_group_values(group_id)
    training = config.training
    exact = {
        "total_steps": QUERY_RETENTION20_TRAIN_STEPS,
        "scheduler_total_steps": QUERY_RETENTION20_SCHEDULER_STEPS,
        "total_tokens": QUERY_RETENTION20_TRAIN_STEPS * GLOBAL_BATCH_SIZE * N_CTX,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(QUERY_RETENTION20_VALIDATION_STEPS),
        "seed": 42,
        "precision": "bf16",
        "warmup_ratio": 0.03,
        "warmup_steps": QUERY_RETENTION20_WARMUP_STEPS,
        "cosine_min_lr_ratio": 0.01,
        "gate_schedule": "fixed_init",
        "query_retention20_group_id": group_id,
        "replay_composition_per_batch": int(values["composition_per_batch"]),
        "replay_atomic_per_batch": int(values["atomic_per_batch"]),
        "replay_composition_per_task": int(values["composition_per_task"]),
        "replay_atomic_per_task": int(values["atomic_per_task"]),
        "kd_temperature": 2.0,
        "kd_lambda": 0.25,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"query_retention20 training.{key} must equal {expected!r}")
    for key, expected in {
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
    }.items():
        if float(training.get(key, float("nan"))) != expected:
            raise ValueError(f"query_retention20 training.{key} changed")

    model = config.model
    model_expected = {
        "candidate": "A",
        "condition_source": "query_only",
        "group": "QO02",
        "raw_fusion_mode": "p2e",
        "pg_lr": float(values["pg_lr"]),
        "query_lr_multiplier": float(values["query_lr_multiplier"]),
        "gate_max": 0.5,
        "base_preservation_coefficient": 0.5,
        "effective_delta_l2": 1.0e-4,
    }
    for key, expected in model_expected.items():
        observed = model.get(key)
        if observed != expected and not (
            isinstance(expected, float)
            and float(observed if observed is not None else float("nan")) == expected
        ):
            raise ValueError(f"query_retention20 model.{key} changed")
    for forbidden in (
        "circuit_lr_multiplier", "raw_encoder_lr", "raw_output_lr",
        "raw_weight_cache_path", "raw_weight_cache_sha256",
    ):
        if forbidden in model:
            raise ValueError(f"query_retention20 Query-only config contains {forbidden}")

    base = dict(config.base)
    initialization = dict(config.initialization)
    if (
        int(base.get("step", -1)) != 38_000
        or int(initialization.get("step", -1)) != 30_516
        or initialization.get("checkpoint_path") != (
            "outputs/s5_lean_circuit/hypernet_query_only_atomic_specialization_seed42/"
            "qo02/checkpoints/model_step0030516.pt"
        )
        or initialization.get("checkpoint_sha256") != (
            "80b51d000c9e48e1d5a8abe3bcb2a8e9ce9c86481db1977021dd6a0876c366d4"
        )
    ):
        raise ValueError("query_retention20 must initialize from QO02 step30516")
    for key, expected in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected:
            raise ValueError(f"query_retention20 base identity field changed: {key}")

    evaluation = config.evaluation
    if (
        evaluation.get("heldout_task") != config.heldout_task
        or int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or evaluation.get("selection_tasks") != list(expected_tasks)
        or evaluation.get("atomic_steps") != list(QUERY_RETENTION20_ATOMIC_STEPS)
    ):
        raise ValueError("query_retention20 evaluation contract changed")
    if config.output_dir != query_retention20_expected_output(group_id, config.heldout_task):
        raise ValueError("query_retention20 output path changed")



def _validate_raw_retention20_config(config: CompositionLOCOConfig) -> None:
    """Validate one Raw-Weight row of the 20-cell retention protocol."""

    if config.format != CONFIG_FORMAT or config.protocol_id != RAW_RETENTION20_PROTOCOL_ID:
        raise ValueError("raw_retention20 config format/protocol changed")
    if config.route != "raw_weight" or config.heldout_task not in RAW_RETENTION20_HELDOUT_TASKS:
        raise ValueError("raw_retention20 route/heldout task is invalid")
    expected_tasks = raw_retention20_train_tasks(config.heldout_task)
    if tuple(config.train_tasks) != expected_tasks:
        raise ValueError("raw_retention20 train task order changed")
    if config.fold_id != f"holdout_{config.heldout_task}":
        raise ValueError("raw_retention20 fold id changed")
    group_id = str(config.training.get("raw_retention20_group_id", ""))
    if group_id not in RAW_RETENTION20_GROUP_IDS:
        raise ValueError("raw_retention20 group id is missing or invalid")

    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
        "atomic_replay",
    }
    if required_data - set(config.data):
        raise ValueError("raw_retention20 data paths are incomplete")
    if str(config.data.get("input_treatment", "fixed_log1p_v1")) != "fixed_log1p_v1":
        raise ValueError("raw_retention20 must use fixed-log1p cache treatment")
    if any(
        config.data.get(key) is not None
        for key in (
            "fingerprint_manifest", "fingerprint_normalizer", "fingerprint_scope",
            "fingerprint_normalizer_artifact_sha256",
        )
    ):
        raise ValueError("raw_retention20 cannot declare a train-fitted fingerprint normalizer")
    replay = config.data.get("atomic_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("raw_retention20 atomic_replay must be an object")
    replay_required = {
        "cache_path", "cache_manifest", "source_config_path",
        "expected_source_config_sha256", "expected_prefix", "expected_n_ctx",
    }
    if replay_required - set(replay):
        raise ValueError("raw_retention20 atomic replay provenance is incomplete")
    if int(replay.get("expected_prefix", -1)) != 1_200_000 or int(replay.get("expected_n_ctx", -1)) != N_CTX:
        raise ValueError("raw_retention20 atomic replay dimensions changed")

    values = raw_retention20_group_values(group_id)
    training = config.training
    exact = {
        "total_steps": RAW_RETENTION20_TRAIN_STEPS,
        "scheduler_total_steps": RAW_RETENTION20_SCHEDULER_STEPS,
        "total_tokens": RAW_RETENTION20_TRAIN_STEPS * GLOBAL_BATCH_SIZE * N_CTX,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(RAW_RETENTION20_VALIDATION_STEPS),
        "seed": 42,
        "precision": "bf16",
        "warmup_ratio": 0.03,
        "warmup_steps": RAW_RETENTION20_WARMUP_STEPS,
        "cosine_min_lr_ratio": 0.01,
        "gate_schedule": "fixed_init",
        "raw_retention20_group_id": group_id,
        "replay_composition_per_batch": int(values["composition_per_batch"]),
        "replay_atomic_per_batch": int(values["atomic_per_batch"]),
        "replay_composition_per_task": int(values["composition_per_task"]),
        "replay_atomic_per_task": int(values["atomic_per_task"]),
        "kd_temperature": 2.0,
        "kd_lambda": 0.25,
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"raw_retention20 training.{key} must equal {expected!r}")
    for key, expected in {
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "weight_decay": 0.0,
        "max_grad_norm": 0.5,
    }.items():
        if float(training.get(key, float("nan"))) != expected:
            raise ValueError(f"raw_retention20 training.{key} changed")

    model = config.model
    model_expected = {
        "candidate": "A",
        "condition_source": "raw_weight",
        "group": "P14",
        "raw_fusion_mode": "query_cross",
        "pg_lr": float(values["pg_lr"]),
        "raw_encoder_lr": float(values["raw_encoder_lr"]),
        "raw_output_lr": float(values["raw_output_lr"]),
        "raw_weight_cache_path": RAW_RETENTION20_CACHE_PATH,
        "raw_weight_cache_sha256": RAW_RETENTION20_CACHE_SHA256,
        "query_lr_multiplier": float(values["query_lr_multiplier"]),
        "gate_max": 0.375,
        "base_preservation_coefficient": 0.5,
        "effective_delta_l2": 1.0e-4,
    }
    for key, expected in model_expected.items():
        observed = model.get(key)
        if observed != expected and not (
            isinstance(expected, float)
            and float(observed if observed is not None else float("nan")) == expected
        ):
            raise ValueError(f"raw_retention20 model.{key} changed")
    for forbidden in (
        "circuit_lr_multiplier", "circuit_lr_multipliers",
    ):
        if forbidden in model:
            raise ValueError(f"raw_retention20 Raw-Weight config contains {forbidden}")

    base = dict(config.base)
    initialization = dict(config.initialization)
    if (
        int(base.get("step", -1)) != 38_000
        or int(initialization.get("step", -1)) != 30_516
        or initialization.get("checkpoint_path") != RAW_RETENTION20_CHECKPOINT
        or initialization.get("checkpoint_sha256") != RAW_RETENTION20_CHECKPOINT_SHA256
        or initialization.get("historical_raw_weight_cache_sha256") != RAW_RETENTION20_HISTORICAL_CACHE_SHA256
    ):
        raise ValueError("raw_retention20 must initialize from RQ14/P14 step30516")
    for key, expected in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected:
            raise ValueError(f"raw_retention20 base identity field changed: {key}")

    evaluation = config.evaluation
    if (
        evaluation.get("heldout_task") != config.heldout_task
        or int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or evaluation.get("selection_tasks") != list(expected_tasks)
        or evaluation.get("atomic_steps") != list(RAW_RETENTION20_ATOMIC_STEPS)
    ):
        raise ValueError("raw_retention20 evaluation contract changed")
    if config.output_dir != raw_retention20_expected_output(group_id, config.heldout_task):
        raise ValueError("raw_retention20 output path changed")


def _validate_direct_retention4_config(config: CompositionLOCOConfig) -> None:
    """Validate the four fixed replay/KD cells without weakening Direct-LR32."""

    if config.format != CONFIG_FORMAT or config.protocol_id != DIRECT_RETENTION4_PROTOCOL_ID:
        raise ValueError("direct_retention4 config format/protocol changed")
    if config.route != "direct_sft" or config.heldout_task != RETENTION_HELDOUT_TASK:
        raise ValueError("direct_retention4 route/heldout task is invalid")
    if tuple(config.train_tasks) != RETENTION_COMPOSITION_TASKS:
        raise ValueError("direct_retention4 must train on the four visible composition tasks")
    if config.fold_id != f"holdout_{RETENTION_HELDOUT_TASK}":
        raise ValueError("direct_retention4 fold id changed")
    group_id = str(config.training.get("retention4_group_id", ""))
    if group_id not in DIRECT_RETENTION4_GROUP_IDS:
        raise ValueError("direct_retention4 group id is missing or invalid")
    values = direct_retention4_group_values(group_id)
    required_data = {
        "cache_path", "cache_manifest", "fold_index", "validation_bank",
        "final_bank", "atomic_bank", "tokenizer_path", "atomic_replay",
    }
    if required_data - set(config.data):
        raise ValueError("direct_retention4 data paths are incomplete")
    replay = config.data.get("atomic_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("direct_retention4 atomic_replay must be an object")
    replay_required = {
        "cache_path", "cache_manifest", "source_config_path",
        "expected_source_config_sha256", "expected_prefix", "expected_n_ctx",
    }
    if replay_required - set(replay):
        raise ValueError("direct_retention4 atomic replay provenance is incomplete")
    if int(replay.get("expected_prefix", -1)) != 1_200_000 or int(replay.get("expected_n_ctx", -1)) != N_CTX:
        raise ValueError("direct_retention4 atomic replay dimensions changed")
    training = config.training
    exact = {
        "total_steps": TOTAL_STEPS,
        "total_tokens": TOTAL_PADDED_TOKENS,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(DIRECT_RETENTION4_VALIDATION_STEPS),
        "seed": 42,
        "cosine_min_lr_ratio": 0.1,
        "warmup_ratio": 0.03,
        "gate_schedule": "fixed_init",
        "precision": "fp16",
        "retention4_group_id": group_id,
        "replay_composition_per_batch": int(values["composition_per_batch"]),
        "replay_atomic_per_batch": int(values["atomic_per_batch"]),
        "replay_composition_per_task": int(values["composition_per_task"]),
        "replay_atomic_per_task": int(values["atomic_per_task"]),
        "kd_temperature": RETENTION_KD_TEMPERATURE,
        "kd_lambda": float(values["kd_lambda"]),
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"direct_retention4 training.{key} must equal {expected!r}")
    for key, expected in {
        "learning_rate": 1.5e-5,
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "max_grad_norm": 1.0,
    }.items():
        if float(training.get(key, float("nan"))) != expected:
            raise ValueError(f"direct_retention4 training.{key} changed")
    base = dict(config.base)
    initialization = dict(config.initialization)
    if (
        int(base.get("step", -1)) != 38_000
        or int(initialization.get("step", -1)) != 38_000
        or initialization.get("artifact_identity_class")
        != "stage_a_base_direct_retention4_v1"
        or initialization.get("checkpoint_path") != base.get("checkpoint_path")
        or initialization.get("checkpoint_sha256") != base.get("checkpoint_sha256")
        or initialization.get("checkpoint_config_sha256") != base.get("config_sha256")
        or initialization.get("historical_direct_checkpoint") is not False
    ):
        raise ValueError("direct_retention4 must initialize directly from Stage-A Base step38000")
    for key, expected in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected:
            raise ValueError(f"direct_retention4 base identity field changed: {key}")
    if config.model.get("condition_source") not in (None, "none"):
        raise ValueError("direct_retention4 Direct route cannot declare a condition")
    evaluation = config.evaluation
    if (
        int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or evaluation.get("selection_tasks") != list(RETENTION_COMPOSITION_TASKS)
        or evaluation.get("atomic_steps") != list(DIRECT_RETENTION4_ATOMIC_STEPS)
    ):
        raise ValueError("direct_retention4 evaluation contract changed")
    if config.output_dir != direct_retention4_expected_output(group_id):
        raise ValueError("direct_retention4 output path changed")


def _validate_direct_retention32_config(config: CompositionLOCOConfig) -> None:
    """Validate one row of the four-fold, eight-cell 12k retention matrix."""

    if config.format != CONFIG_FORMAT or config.protocol_id != DIRECT_RETENTION32_PROTOCOL_ID:
        raise ValueError("direct_retention32 config format/protocol changed")
    if config.route != "direct_sft":
        raise ValueError("direct_retention32 route must be direct_sft")
    if config.heldout_task == "T_and_or" or config.heldout_task not in COMPOSITION_TASKS:
        raise ValueError("direct_retention32 held-out task must be one of the four new folds")
    expected_tasks = direct_retention32_train_tasks(config.heldout_task)
    if tuple(config.train_tasks) != expected_tasks:
        raise ValueError("direct_retention32 train task order changed")
    if config.fold_id != f"holdout_{config.heldout_task}":
        raise ValueError("direct_retention32 fold id changed")
    group_id = str(config.training.get("retention32_group_id", ""))
    if group_id not in DIRECT_RETENTION32_GROUP_IDS:
        raise ValueError("direct_retention32 group id is missing or invalid")
    values = direct_retention32_group_values(group_id)
    required_data = {
        "cache_path", "cache_manifest", "fold_index", "base_ce_path",
        "validation_bank", "final_bank", "atomic_bank", "tokenizer_path",
        "atomic_replay",
    }
    if required_data - set(config.data):
        raise ValueError("direct_retention32 data paths are incomplete")
    replay = config.data.get("atomic_replay")
    if not isinstance(replay, Mapping):
        raise ValueError("direct_retention32 atomic_replay must be an object")
    replay_required = {
        "cache_path", "cache_manifest", "source_config_path",
        "expected_source_config_sha256", "expected_prefix", "expected_n_ctx",
    }
    if replay_required - set(replay):
        raise ValueError("direct_retention32 atomic replay provenance is incomplete")
    if int(replay.get("expected_prefix", -1)) != 1_200_000 or int(replay.get("expected_n_ctx", -1)) != N_CTX:
        raise ValueError("direct_retention32 atomic replay dimensions changed")
    training = config.training
    exact = {
        "total_steps": DIRECT_RETENTION32_TRAIN_STEPS,
        "scheduler_total_steps": DIRECT_RETENTION32_SCHEDULER_STEPS,
        "total_tokens": DIRECT_RETENTION32_TRAIN_STEPS * GLOBAL_BATCH_SIZE * N_CTX,
        "global_batch_size": GLOBAL_BATCH_SIZE,
        "n_ctx": N_CTX,
        "save_model_every": 1_000,
        "save_resume_every": 5_000,
        "validation_steps": list(DIRECT_RETENTION32_VALIDATION_STEPS),
        "seed": 42,
        "cosine_min_lr_ratio": 0.1,
        "warmup_ratio": 0.03,
        "warmup_steps": DIRECT_RETENTION32_WARMUP_STEPS,
        "gate_schedule": "fixed_init",
        "precision": "fp16",
        "retention32_group_id": group_id,
        "replay_composition_per_batch": int(values["composition_per_batch"]),
        "replay_atomic_per_batch": int(values["atomic_per_batch"]),
        "replay_composition_per_task": int(values["composition_per_task"]),
        "replay_atomic_per_task": int(values["atomic_per_task"]),
        "kd_temperature": 2.0,
        "kd_lambda": float(values["kd_lambda"]),
    }
    for key, expected in exact.items():
        if training.get(key) != expected:
            raise ValueError(f"direct_retention32 training.{key} must equal {expected!r}")
    for key, expected in {
        "learning_rate": float(values["learning_rate"]),
        "weight_decay": 0.01,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "max_grad_norm": 1.0,
    }.items():
        if float(training.get(key, float("nan"))) != expected:
            raise ValueError(f"direct_retention32 training.{key} changed")
    base = dict(config.base)
    initialization = dict(config.initialization)
    if (
        int(base.get("step", -1)) != 38_000
        or int(initialization.get("step", -1)) != 38_000
        or initialization.get("artifact_identity_class")
        != "stage_a_base_direct_retention32_v1"
        or initialization.get("checkpoint_path") != base.get("checkpoint_path")
        or initialization.get("checkpoint_sha256") != base.get("checkpoint_sha256")
        or initialization.get("checkpoint_config_sha256") != base.get("config_sha256")
        or initialization.get("historical_direct_checkpoint") is not False
    ):
        raise ValueError("direct_retention32 must initialize directly from Stage-A Base step38000")
    for key, expected in STAGE_A_BASE_IDENTITY.items():
        if base.get(key) != expected:
            raise ValueError(f"direct_retention32 base identity field changed: {key}")
    if config.model.get("condition_source") not in (None, "none"):
        raise ValueError("direct_retention32 Direct route cannot declare a condition")
    evaluation = config.evaluation
    if (
        evaluation.get("heldout_task") != config.heldout_task
        or int(evaluation.get("visible_samples_per_task", -1)) != 500
        or int(evaluation.get("final_samples_per_task", -1)) != 2_000
        or int(evaluation.get("atomic_samples_per_task", -1)) != 500
        or evaluation.get("selection_tasks") != list(expected_tasks)
        or evaluation.get("atomic_steps") != list(DIRECT_RETENTION32_ATOMIC_STEPS)
    ):
        raise ValueError("direct_retention32 evaluation contract changed")
    if not config.output_dir.startswith(DIRECT_RETENTION32_OUTPUT_ROOT + "/runs/"):
        raise ValueError("direct_retention32 output must use the dedicated output root")
    if config.output_dir != direct_retention32_expected_output(group_id, config.heldout_task):
        raise ValueError("direct_retention32 output path changed")


def make_lr_half_config(baseline: CompositionLOCOConfig) -> CompositionLOCOConfig:
    """Exactly one approved perturbation, never a generic v1 guard bypass."""
    from dataclasses import replace

    if baseline.protocol_id != PROTOCOL_ID:
        raise ValueError("LR-half requires an original composition-SFT baseline")
    _validate_config(baseline)
    training, model = dict(baseline.training), dict(baseline.model)
    if baseline.route == "direct_sft":
        training["learning_rate"] *= 0.5
    else:
        model["pg_lr"] *= 0.5
        for name in ("raw_encoder_lr", "raw_output_lr"):
            if name in model:
                model[name] *= 0.5
    return replace(
        baseline, protocol_id=LR_HALF_PROTOCOL_ID, training=training, model=model,
        output_dir=f"{LR_HALF_OUTPUT_ROOT}/runs/{baseline.heldout_task}/{baseline.route}",
        config_sha256=None,
    )


def make_recreated_inputs_config(baseline: CompositionLOCOConfig) -> CompositionLOCOConfig:
    """Build an explicitly non-historical config for missing-input recovery."""
    from dataclasses import replace

    if baseline.protocol_id != PROTOCOL_ID or baseline.route not in {"direct_sft", "raw_weight"}:
        raise ValueError("recreated-input recovery supports only the Direct and Raw routes")
    candidate = make_lr_half_config(baseline)
    initialization = dict(candidate.initialization)
    model = dict(candidate.model)
    if baseline.route == "direct_sft":
        initialization.update({
            "checkpoint_path": RECREATED_DIRECT_CHECKPOINT,
            "checkpoint_sha256": RECREATED_DIRECT_CHECKPOINT_SHA256,
            "checkpoint_config_sha256": RECREATED_DIRECT_CONFIG_SHA256,
        })
    else:
        model.update({
            "raw_weight_cache_path": RECREATED_RAW_CACHE,
            "raw_weight_cache_sha256": RECREATED_RAW_CACHE_SHA256,
        })
        initialization["historical_raw_weight_cache_sha256"] = HISTORICAL_RAW_CACHE_SHA256
    return replace(
        candidate,
        protocol_id=RECREATED_INPUTS_PROTOCOL_ID,
        initialization=initialization,
        model=model,
        output_dir=f"{RECREATED_INPUTS_OUTPUT_ROOT}/runs/{baseline.heldout_task}/{baseline.route}",
        config_sha256=None,
    )


def make_original_lr_recreated_config(baseline: CompositionLOCOConfig) -> CompositionLOCOConfig:
    """Build the original-LR Raw route against the explicitly rebuilt cache."""
    from dataclasses import replace

    if baseline.protocol_id != PROTOCOL_ID or baseline.route != "raw_weight":
        raise ValueError("original-LR recovery supports only the original Raw route")
    _validate_config(baseline)
    initialization = dict(baseline.initialization)
    initialization["historical_raw_weight_cache_sha256"] = HISTORICAL_RAW_CACHE_SHA256
    model = dict(baseline.model)
    model.update(
        {
            "raw_weight_cache_path": RECREATED_RAW_CACHE,
            "raw_weight_cache_sha256": RECREATED_RAW_CACHE_SHA256,
        }
    )
    return replace(
        baseline,
        protocol_id=ORIGINAL_LR_RECREATED_PROTOCOL_ID,
        initialization=initialization,
        model=model,
        output_dir=(
            f"{ORIGINAL_LR_RECREATED_OUTPUT_ROOT}/runs/"
            f"{baseline.heldout_task}/raw_weight"
        ),
        config_sha256=None,
    )


def _validate_lr_half_config(config: CompositionLOCOConfig) -> None:
    from .composition_loco import COMPOSITION_TASKS

    if config.route not in ROUTES or config.heldout_task not in COMPOSITION_TASKS:
        raise ValueError("invalid LR-half fold/route")
    root = Path(__file__).resolve().parents[2]
    baseline = load_config(root / "configs/composition_loco_sft_v1" /
                           config.heldout_task / f"{config.route}.json")
    expected = make_lr_half_config(baseline)
    if config.to_dict() != expected.to_dict():
        raise ValueError("LR-half must change only all peak LRs, protocol ID and dedicated output")


def _validate_recreated_inputs_config(config: CompositionLOCOConfig) -> None:
    from .composition_loco import COMPOSITION_TASKS

    if config.route not in {"direct_sft", "raw_weight"} or config.heldout_task not in COMPOSITION_TASKS:
        raise ValueError("invalid recreated-input fold/route")
    root = Path(__file__).resolve().parents[2]
    baseline = load_config(root / "configs/composition_loco_sft_v1" /
                           config.heldout_task / f"{config.route}.json")
    expected = make_recreated_inputs_config(baseline)
    if config.to_dict() != expected.to_dict():
        raise ValueError(
            "recreated-input config must use only the approved rebuilt Direct checkpoint "
            "or rebuilt Raw cache and a separate output root"
        )


def _validate_original_lr_recreated_config(config: CompositionLOCOConfig) -> None:
    from .composition_loco import COMPOSITION_TASKS

    if config.route != "raw_weight" or config.heldout_task not in COMPOSITION_TASKS:
        raise ValueError("original-LR rebuilt-cache recovery is restricted to Raw folds")
    root = Path(__file__).resolve().parents[2]
    baseline = load_config(
        root / "configs/composition_loco_sft_v1" /
        config.heldout_task / "raw_weight.json"
    )
    expected = make_original_lr_recreated_config(baseline)
    if config.to_dict() != expected.to_dict():
        raise ValueError(
            "original-LR recreated Raw config must preserve the original optimizer LRs "
            "and use only the approved rebuilt cache and separate output"
        )


def load_config(path: str | Path) -> CompositionLOCOConfig:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("composition LOCO config must be an object")
    config = CompositionLOCOConfig(
        route=str(value.get("route", "")),
        fold_id=str(value.get("fold_id", "")),
        heldout_task=str(value.get("heldout_task", "")),
        train_tasks=tuple(str(item) for item in value.get("train_tasks", ())),
        data=_require_mapping(value.get("data"), "data"),
        base=_require_mapping(value.get("base"), "base"),
        initialization=_require_mapping(value.get("initialization"), "initialization"),
        model=_require_mapping(value.get("model"), "model"),
        training=_require_mapping(value.get("training"), "training"),
        evaluation=_require_mapping(value.get("evaluation"), "evaluation"),
        output_dir=str(value.get("output_dir", "")),
        protocol_id=str(value.get("protocol_id", PROTOCOL_ID)),
        format=str(value.get("format", CONFIG_FORMAT)),
        config_sha256=(None if value.get("config_sha256") is None else str(value["config_sha256"])),
    )
    _validate_config(config)
    if config.config_sha256 is not None and config.config_sha256 != config.computed_sha256:
        raise ValueError("composition LOCO config checksum changed")
    return config


def write_config(path: str | Path, config: CompositionLOCOConfig) -> dict[str, Any]:
    _validate_config(config)
    payload = config.to_dict()
    payload["config_sha256"] = config.computed_sha256
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return payload


__all__ = [
    "CONFIG_FORMAT", "CompositionLOCOConfig", "load_config", "semantic_sha256", "write_config",
    "LR_HALF_PROTOCOL_ID", "LR_HALF_OUTPUT_ROOT", "make_lr_half_config",
    "RECREATED_INPUTS_PROTOCOL_ID", "RECREATED_INPUTS_OUTPUT_ROOT", "make_recreated_inputs_config",
    "ORIGINAL_LR_RECREATED_PROTOCOL_ID", "ORIGINAL_LR_RECREATED_OUTPUT_ROOT",
    "make_original_lr_recreated_config",
    "MANUAL160_PROTOCOL_ID", "MANUAL160_OUTPUT_ROOT",
    "DIRECT_LR32_PROTOCOL_ID", "DIRECT_LR32_OUTPUT_ROOT",
    "CIRCUIT_SWEEP40_PROTOCOL_ID", "CIRCUIT_SWEEP40_OUTPUT_ROOT",
    "CIRCUIT_SWEEP40_CONFIG_ROOT",
    "QUERY_WEIGHT_PROTOCOL_ID", "QUERY_WEIGHT_OUTPUT_ROOT", "QUERY_WEIGHT_CONFIG_ROOT",
    "QUERY_CIRCUIT_PROTOCOL_ID", "QUERY_CIRCUIT_OUTPUT_ROOT", "QUERY_CIRCUIT_CONFIG_ROOT",
    "WEIGHT_CIRCUIT_PROTOCOL_ID", "WEIGHT_CIRCUIT_OUTPUT_ROOT", "WEIGHT_CIRCUIT_CONFIG_ROOT",
    "NESTED_QUERY_WEIGHT_PROTOCOL_ID", "NESTED_QUERY_WEIGHT_OUTPUT_ROOT", "NESTED_QUERY_WEIGHT_CONFIG_ROOT",
    "DIRECT_RETENTION4_PROTOCOL_ID", "DIRECT_RETENTION4_OUTPUT_ROOT", "DIRECT_RETENTION4_CONFIG_ROOT",
    "DIRECT_RETENTION32_PROTOCOL_ID", "DIRECT_RETENTION32_OUTPUT_ROOT", "DIRECT_RETENTION32_CONFIG_ROOT",
    "CIRCUIT_RETENTION20_PROTOCOL_ID", "CIRCUIT_RETENTION20_OUTPUT_ROOT", "CIRCUIT_RETENTION20_CONFIG_ROOT",
    "RAW_RETENTION20_PROTOCOL_ID", "RAW_RETENTION20_OUTPUT_ROOT", "RAW_RETENTION20_CONFIG_ROOT",
    "QUERY_RETENTION20_PROTOCOL_ID", "QUERY_RETENTION20_OUTPUT_ROOT", "QUERY_RETENTION20_CONFIG_ROOT",
    "RECREATED_DIRECT_CHECKPOINT", "RECREATED_DIRECT_CHECKPOINT_SHA256", "RECREATED_DIRECT_CONFIG_SHA256",
    "RECREATED_RAW_CACHE", "RECREATED_RAW_CACHE_SHA256", "HISTORICAL_RAW_CACHE_SHA256",
]

"""Configuration loading and protocol invariants for S5 reproduction runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .protocol import (
    ALL_TASKS,
    ATOMIC_TASKS,
    COMPOSITION_TASKS,
    DEPTH3_TASKS,
    STAGE1_SCHEDULE_TYPES,
    STAGE_B_SUPERVISED_SEED_BASE,
    STAGE_B_RENDERER_VERSION,
    STAGE_B_RENDERER_VERSION_V1,
    STAGE_B_TASK_SEED_STRIDE,
    validate_sampling_counts,
    validate_task_types,
)
from .schedules import (
    CONSTANT_SCHEDULE,
    COSINE_SCHEDULE,
    LINEAR_DECAY_SCHEDULE,
)
from .alias_tasks import FAMILIES as ALIAS_SUBTYPES, FORMAT as ALIAS_DATA_FORMAT


STAGE_A_DENSE_BGOFF_CHECKPOINT_SHA256 = (
    "ee2a7efdf77262ee1fbdf7d2080f930c2bedc27e45b7712537fba964ee05931c"
)
STAGE_A_DENSE_BGOFF_CONFIG_SHA256 = (
    "d87e874cdee838e7d07b99631d06ee64ec3466d24012a08af75d032eddf7169e"
)
HYPERNET_MATCHED_DIRECT_SFT_TRACK = "hypernet_matched_direct_sft"
HYPERNET_MATCHED_DIRECT_SFT_TOKENS = 2_490_368_000
HYPERNET_MATCHED_DIRECT_SFT_CANDIDATE_STEPS = [7_629, 15_258, 38_000]
ALIAS_INTERMEDIATE_CONTINUATION_TRACK = "alias_intermediate_continuation"
ALIAS_INTERMEDIATE_TASK = "S_alias"
ALIAS_INTERMEDIATE_TOTAL_SAMPLES = 4_864_000
ALIAS_INTERMEDIATE_SAMPLES_PER_SUBTYPE = 1_216_000
ALIAS_INTERMEDIATE_TOTAL_TOKENS = 622_592_000
ALIAS_INTERMEDIATE_TOTAL_STEPS = 9_500
ALIAS_INTERMEDIATE_TOKENIZER_SHA256 = (
    "1053e327f96ee13ceca0c4ab4d9e48eefa9eb2c2eddf65359715642ec701199f"
)
ALIAS_INTERMEDIATE_VALIDATION_SUITE_SHA256 = (
    "c6e0ef0b2379d792beaaec7107dff29d4ebe93a9514cd7e3efe90cbd91465a2a"
)
ALIAS_INTERMEDIATE_TEST_SUITE_SHA256 = (
    "1a2b1087404a56605b0056cccdad2ef3e3d8dc089e01d10cd5833f12ab1f65a2"
)

FORMAL_LOCO_FOLDS = {
    "loo_and_or": ("LOO-and-or", "T_and_or"),
    "loo_or_and": ("LOO-or-and", "T_or_and"),
    "loo_and_or_and": ("LOO-and-or-and", "T_and_or_and"),
    "loo_nested_and": ("LOO-nested-and", "T_nested_and"),
    "loo_nested_or": ("LOO-nested-or", "T_nested_or"),
}


def _validate_formal_stage_b_data_config(config: dict[str, Any]) -> None:
    data_id = str(config.get("data_config_id", ""))
    tasks = tuple(config["train_tasks"])
    counts = config["task_schedule"]["counts_per_100"]
    expected_tasks: tuple[str, ...]
    expected_counts: dict[str, int]
    expected_fold = None
    expected_heldout = None

    if data_id == "four_atomic":
        expected_tasks = ATOMIC_TASKS
        expected_counts = {task: 25 for task in ATOMIC_TASKS}
    elif data_id == "atomic_inclusive_full":
        expected_tasks = ALL_TASKS
        expected_counts = {
            **{task: 10 for task in ATOMIC_TASKS},
            **{task: 12 for task in COMPOSITION_TASKS},
        }
    elif data_id == "composition_only_full":
        expected_tasks = COMPOSITION_TASKS
        expected_counts = {task: 20 for task in COMPOSITION_TASKS}
    else:
        prefix = next(
            (
                candidate
                for candidate in ("atomic_inclusive_", "composition_only_")
                if data_id.startswith(candidate)
            ),
            None,
        )
        suffix = data_id[len(prefix) :] if prefix is not None else ""
        if prefix is None or suffix not in FORMAL_LOCO_FOLDS:
            raise ValueError(f"unknown formal Stage-B data_config_id: {data_id}")
        expected_fold, expected_heldout = FORMAL_LOCO_FOLDS[suffix]
        visible = tuple(
            task for task in COMPOSITION_TASKS if task != expected_heldout
        )
        if prefix == "atomic_inclusive_":
            expected_tasks = ATOMIC_TASKS + visible
            expected_counts = {
                **{task: 10 for task in ATOMIC_TASKS},
                **{task: 15 for task in visible},
            }
        else:
            expected_tasks = visible
            expected_counts = {task: 25 for task in visible}

    if tasks != expected_tasks:
        raise ValueError(f"formal Stage-B train_tasks drifted for {data_id}")
    if counts != expected_counts:
        raise ValueError(f"formal Stage-B task weights drifted for {data_id}")
    if config.get("fold_id") != expected_fold:
        raise ValueError(f"formal Stage-B fold_id drifted for {data_id}")
    if config.get("training_heldout_task") != expected_heldout:
        raise ValueError(
            f"formal Stage-B training_heldout_task drifted for {data_id}"
        )


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_config(config)
    config["_config_path"] = str(config_path.resolve())
    config["_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return config


def validate_config(config: dict[str, Any]) -> None:
    protocol_track = str(config.get("protocol_track", "stage_a_reproduction"))
    if protocol_track not in {
        "stage_a_reproduction",
        "stage1_paper_grid",
        "stage_b_method",
        HYPERNET_MATCHED_DIRECT_SFT_TRACK,
        ALIAS_INTERMEDIATE_CONTINUATION_TRACK,
    }:
        raise ValueError(f"unknown protocol_track: {protocol_track}")
    is_stage_b = protocol_track == "stage_b_method"
    is_stage1_grid = protocol_track == "stage1_paper_grid"
    is_hypernet_matched_direct = (
        protocol_track == HYPERNET_MATCHED_DIRECT_SFT_TRACK
    )
    is_alias_intermediate = (
        protocol_track == ALIAS_INTERMEDIATE_CONTINUATION_TRACK
    )
    run_kind = str(config.get("run_kind", "stage_a_formal"))
    surface_family = str(config.get("surface_family", ""))
    if is_stage1_grid:
        if surface_family not in {"narrow", "broad_medium", "broad_full"}:
            raise ValueError("Stage-1 grid surface_family is unsupported")
    elif surface_family != "narrow":
        raise ValueError("the frozen reproduction requires surface_family=narrow")
    if int(config.get("seed", -1)) != 42:
        raise ValueError("the frozen reproduction uses model seed 42")
    if is_alias_intermediate:
        train_tasks = tuple(config.get("train_tasks", ()))
        if train_tasks != (ALIAS_INTERMEDIATE_TASK,):
            raise ValueError("alias continuation exposes exactly one S_alias task")
    else:
        train_tasks = validate_task_types(config.get("train_tasks", ()))
        if not set(train_tasks).issubset(ALL_TASKS):
            raise ValueError("depth-3 tasks are evaluation-only and cannot train")
    if is_stage_b or is_hypernet_matched_direct or is_alias_intermediate:
        initialization = config.get("initialization", {})
        if initialization.get("type") != "stage_a_model_weights":
            raise ValueError(
                "continued SFT initialization.type must be stage_a_model_weights"
            )
        if (
            initialization.get("expected_protocol_id")
            != "s5_narrow_atomic_only_sft_dense_bigram_off_v1"
        ):
            raise ValueError("continued SFT Base protocol ID must be frozen")
        if int(initialization.get("expected_step", -1)) != 38_000:
            raise ValueError("continued SFT Base checkpoint step must equal 38,000")
        expected_source_hash = str(initialization.get("expected_config_sha256", ""))
        if expected_source_hash != STAGE_A_DENSE_BGOFF_CONFIG_SHA256:
            raise ValueError("continued SFT Base config SHA256 must match frozen Base")
        expected_checkpoint_hash = str(
            initialization.get("expected_checkpoint_sha256", "")
        )
        if expected_checkpoint_hash != STAGE_A_DENSE_BGOFF_CHECKPOINT_SHA256:
            raise ValueError(
                "continued SFT Base checkpoint SHA256 must match frozen Base"
            )
    if is_stage_b:
        if run_kind not in {
            "scaling_benchmark",
            "lr_pilot",
            "lr_tuning",
            "warmup_tuning",
            "schedule_tuning",
            "formal_sft",
            "formal_sft_v2",
        }:
            raise ValueError(f"unsupported Stage-B run_kind: {run_kind}")
        schedule = config.get("task_schedule", {})
        if schedule.get("type") != "weighted_cycle":
            raise ValueError("Stage-B task_schedule.type must be weighted_cycle")
        validate_sampling_counts(
            schedule.get("counts_per_100", {}),
            train_tasks=train_tasks,
        )
        if int(schedule.get("sampling_seed", -1)) != 42:
            raise ValueError("Stage-B sampling_seed must equal 42")
        if int(schedule.get("seed_base", -1)) != STAGE_B_SUPERVISED_SEED_BASE:
            raise ValueError("Stage-B supervised seed_base must equal 50B")
        if int(schedule.get("task_seed_stride", -1)) != STAGE_B_TASK_SEED_STRIDE:
            raise ValueError("Stage-B task_seed_stride must equal 100M")
    elif is_stage1_grid:
        schedule = config.get("task_schedule", {})
        schedule_type = str(schedule.get("type", ""))
        if schedule_type not in STAGE1_SCHEDULE_TYPES:
            raise ValueError("Stage-1 grid task_schedule.type is unsupported")
        if schedule_type == "atomic_only":
            if schedule != {"type": "atomic_only"}:
                raise ValueError("Atomic-only Stage-1 schedule has extra fields")
            expected_train_tasks = ATOMIC_TASKS
        elif schedule_type == "composition_only":
            if schedule != {"type": "composition_only"}:
                raise ValueError("Composition-only Stage-1 schedule has extra fields")
            expected_train_tasks = COMPOSITION_TASKS
        else:
            expected_schedule = {
                "type": "curriculum",
                "simple_fraction": 0.3,
                "gradual_fraction": 0.3,
                "full_fraction": 0.4,
                "gradual_mix": "linear_to_uniform_all_tasks",
            }
            if schedule != expected_schedule:
                raise ValueError("Stage-1 curriculum schedule drifted")
            expected_train_tasks = ALL_TASKS
        if train_tasks != expected_train_tasks:
            raise ValueError(
                f"Stage-1 {schedule_type} train_tasks drifted"
            )
    elif not is_alias_intermediate and train_tasks != ATOMIC_TASKS:
        raise ValueError("train_tasks must be exactly the four atomic tasks")
    if is_alias_intermediate:
        if config.get("selection_tasks") != [ALIAS_INTERMEDIATE_TASK]:
            raise ValueError("alias continuation selects only on S_alias")
        if config.get("test_tasks") != [
            ALIAS_INTERMEDIATE_TASK,
            "original_nine",
            "wrapped_five",
        ]:
            raise ValueError("alias continuation final evaluation suite drifted")
        alias_data = config.get("alias_data", {})
        expected_alias_data = {
            "format": ALIAS_DATA_FORMAT,
            "task_family": ALIAS_INTERMEDIATE_TASK,
            "subtypes": list(ALIAS_SUBTYPES),
            "schedule": "balanced_64_row_blocks_v1",
            "seed": 42,
            "train_split": "train",
            "validation_split": "validation",
            "test_split": "test",
            "total_samples": ALIAS_INTERMEDIATE_TOTAL_SAMPLES,
            "samples_per_subtype": ALIAS_INTERMEDIATE_SAMPLES_PER_SUBTYPE,
        }
        if alias_data != expected_alias_data:
            raise ValueError("alias continuation data contract drifted")
    else:
        if validate_task_types(config.get("selection_tasks", ())) != ALL_TASKS:
            raise ValueError("selection_tasks must be exactly all nine public tasks")
        if validate_task_types(config.get("test_tasks", ())) != ALL_TASKS:
            raise ValueError("test_tasks must be exactly all nine public tasks")

    model = config["model"]
    expected_model = {
        "block_size": 128,
        "vocab_size": 2048,
        "n_layer": 8,
        "n_head": 128,
        "d_head": 16,
        "d_model": 2048,
        "d_mlp": 8192,
        "d_pos_emb": 32,
    }
    for key, expected in expected_model.items():
        if int(model.get(key, -1)) != expected:
            raise ValueError(f"model.{key} must equal {expected}")
    exact_model_behavior = {
        "bias": True,
        "ln_bias": True,
        "rms_norm": True,
        "activation_type": "gelu",
        "residual_activation_type": "identity",
        "tied_unembed": False,
        "sink": True,
        "flash": True,
    }
    for key, expected in exact_model_behavior.items():
        if model.get(key) != expected:
            raise ValueError(f"model.{key} must equal {expected!r}")
    keep_fraction = model.get("activation_keep_fraction")
    if keep_fraction not in (None, 0.25, 1.0):
        raise ValueError("activation_keep_fraction must be 0.25 or disabled")
    if bool(model.get("enable_bigram_table")) != bool(
        model.get("learnable_bigram_table")
    ):
        raise ValueError(
            "enable_bigram_table and learnable_bigram_table must match"
        )
    if is_stage_b or is_hypernet_matched_direct or is_alias_intermediate:
        if keep_fraction is not None:
            raise ValueError("continued SFT must keep dense activations")
        if bool(model.get("enable_bigram_table")):
            raise ValueError("continued SFT must keep bigram OFF")
        if float(model.get("dropout", -1.0)) != 0.0:
            raise ValueError("continued SFT dropout must equal zero")
        if is_stage_b and not bool(model.get("grad_checkpointing")):
            raise ValueError("Stage-B baseline requires gradient checkpointing")
        if (is_hypernet_matched_direct or is_alias_intermediate) and bool(
            model.get("grad_checkpointing")
        ):
            raise ValueError(
                "continuations keep gradient checkpointing off"
            )
    if is_stage1_grid:
        if keep_fraction != 0.25:
            raise ValueError("Stage-1 dense grid must keep activation fraction 0.25")
        if not bool(model.get("enable_bigram_table")):
            raise ValueError("Stage-1 dense grid must keep the public bigram table ON")

    training = config["training"]
    exact_training = {
        "global_batch_size": 512,
        "eval_interval": 2000,
        "validation_samples_per_task": 50,
        "max_new_tokens": 50,
        "n_distractors": 2,
    }
    for key, expected in exact_training.items():
        if int(training.get(key, -1)) != expected:
            raise ValueError(f"training.{key} must equal {expected}")
    expected_checkpoint_interval = (
        7_629 if is_hypernet_matched_direct else 2_000
    )
    if int(training.get("checkpoint_interval", -1)) != expected_checkpoint_interval:
        raise ValueError(
            f"training.checkpoint_interval must equal {expected_checkpoint_interval}"
        )
    exact_optimizer = {
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
    }
    for key, expected in exact_optimizer.items():
        if float(training.get(key, -1.0)) != expected:
            raise ValueError(f"training.{key} must equal {expected}")
    if is_hypernet_matched_direct:
        if run_kind not in {
            "hypernet_matched_direct_sft_v1",
            "hypernet_matched_direct_sft_lr_sweep_v1",
        }:
            raise ValueError("direct-SFT run_kind is unsupported")
        if int(training.get("warmup_steps", -1)) != 1000:
            raise ValueError("direct-SFT warmup must equal 1000")
        if training.get("schedule") != COSINE_SCHEDULE:
            raise ValueError("direct-SFT schedule must be cosine decay")
        if float(training.get("final_learning_rate_ratio", -1.0)) != 0.1:
            raise ValueError("direct-SFT final LR ratio must equal 0.1")
        if int(training.get("decay_end_step", -1)) != 38_000:
            raise ValueError("direct-SFT decay_end_step must equal 38,000")
        if training.get("candidate_steps") != HYPERNET_MATCHED_DIRECT_SFT_CANDIDATE_STEPS:
            raise ValueError("direct-SFT candidate steps drifted")
    elif is_alias_intermediate:
        if run_kind != "alias_intermediate_continued_sft_v1":
            raise ValueError("alias continuation run_kind is unsupported")
        if int(training.get("warmup_steps", -1)) != 1000:
            raise ValueError("alias continuation warmup must equal 1000")
        if training.get("schedule") != COSINE_SCHEDULE:
            raise ValueError("alias continuation must use cosine decay")
        if float(training.get("final_learning_rate_ratio", -1.0)) != 0.1:
            raise ValueError("alias continuation final LR ratio must equal 0.1")
        if int(training.get("decay_end_step", -1)) != ALIAS_INTERMEDIATE_TOTAL_STEPS:
            raise ValueError("alias continuation decay_end_step must equal 9,500")
        if training.get("candidate_steps") != [2_000, 4_000, 6_000, 8_000, 9_500]:
            raise ValueError("alias continuation candidate steps drifted")
    elif run_kind == "formal_sft_v2":
        if int(training.get("warmup_steps", -1)) != 1000:
            raise ValueError("formal Stage-B v2 warmup must equal 1000")
        if training.get("schedule") != COSINE_SCHEDULE:
            raise ValueError("formal Stage-B v2 schedule must be cosine decay")
        if float(training.get("final_learning_rate_ratio", -1.0)) != 0.1:
            raise ValueError("formal Stage-B v2 final LR ratio must equal 0.1")
        if int(training.get("decay_end_step", -1)) != 15_258:
            raise ValueError("formal Stage-B v2 decay_end_step must equal 15,258")
    elif run_kind == "warmup_tuning":
        if int(training.get("warmup_steps", -1)) != 250:
            raise ValueError("warmup tuning candidate must use 250 warmup steps")
        if training.get("schedule") != CONSTANT_SCHEDULE:
            raise ValueError("warmup tuning keeps the constant LR schedule")
    elif run_kind == "schedule_tuning":
        if int(training.get("warmup_steps", -1)) != 1000:
            raise ValueError("schedule tuning warmup must equal frozen 1000")
        schedule_name = training.get("schedule")
        if schedule_name not in {
            CONSTANT_SCHEDULE,
            COSINE_SCHEDULE,
            LINEAR_DECAY_SCHEDULE,
        }:
            raise ValueError("schedule tuning uses constant/cosine/linear only")
        expected_final_ratio = 1.0 if schedule_name == CONSTANT_SCHEDULE else 0.1
        if float(training.get("final_learning_rate_ratio", -1.0)) != (
            expected_final_ratio
        ):
            raise ValueError(
                "schedule tuning final LR ratio must be 1.0 constant or 0.1 decay"
            )
        if int(training.get("decay_end_step", -1)) != 15_258:
            raise ValueError("schedule tuning decay_end_step must equal 15,258")
    else:
        if int(training.get("warmup_steps", -1)) != 1000:
            raise ValueError("training.warmup_steps must equal 1000")
        if training.get("schedule") != CONSTANT_SCHEDULE:
            raise ValueError("training.schedule must be linear_warmup_then_constant")
    expected_total_tokens = 5_000_000_000
    if is_stage_b and run_kind in {
        "lr_pilot",
        "lr_tuning",
        "warmup_tuning",
    }:
        expected_total_tokens = 500_000_000
    if is_stage_b and run_kind == "schedule_tuning":
        expected_total_tokens = 1_000_000_000
    if is_hypernet_matched_direct:
        expected_total_tokens = HYPERNET_MATCHED_DIRECT_SFT_TOKENS
    if is_alias_intermediate:
        expected_total_tokens = ALIAS_INTERMEDIATE_TOTAL_TOKENS
    if int(training.get("total_tokens", -1)) != expected_total_tokens:
        raise ValueError(
            f"training.total_tokens must equal {expected_total_tokens}"
        )
    learning_rate = float(training.get("learning_rate", -1.0))
    if is_hypernet_matched_direct:
        if run_kind == "hypernet_matched_direct_sft_v1" and learning_rate != 1e-4:
            raise ValueError("direct-SFT anchor learning rate must equal 1e-4")
        if run_kind == "hypernet_matched_direct_sft_lr_sweep_v1" and learning_rate not in {
            3e-5,
            1e-4,
            3e-4,
        }:
            raise ValueError("direct-SFT LR sweep uses 3e-5/1e-4/3e-4")
        packed = config.get("packed_data", {})
        if int(packed.get("prefix", -1)) != 1_200_000:
            raise ValueError("direct-SFT packed data must contain 1.2M events")
        if int(packed.get("n_ctx", -1)) != 128:
            raise ValueError("direct-SFT packed data n_ctx must equal 128")
        if int(packed.get("assignment_seed", -1)) != 42:
            raise ValueError("direct-SFT packed assignment seed must equal 42")
        if packed.get("order") != "balanced_epoch_order_v1":
            raise ValueError("direct-SFT packed order protocol changed")
        if not str(packed.get("cache_path", "")):
            raise ValueError("direct-SFT packed cache path is required")
        if not str(packed.get("hypernet_source_config_path", "")):
            raise ValueError("direct-SFT HyperNet source config path is required")
        source_sha = str(packed.get("expected_hypernet_config_sha256", ""))
        if len(source_sha) != 64 or any(
            char not in "0123456789abcdef" for char in source_sha
        ):
            raise ValueError("direct-SFT HyperNet source config SHA256 is invalid")
        execution = config.get("execution", {})
        expected_execution = {
            "auto_select_best": True,
            "auto_run_test": (
                run_kind == "hypernet_matched_direct_sft_v1"
            ),
        }
        if execution != expected_execution:
            raise ValueError(
                "direct-SFT anchor must select/run final once; "
                "LR-sweep cells must select on validation only"
            )
    elif is_alias_intermediate:
        if learning_rate != 1e-4:
            raise ValueError("alias continuation learning rate must equal 1e-4")
        if config.get("execution") != {
            "auto_select_best": True,
            "auto_run_test": True,
        }:
            raise ValueError("alias continuation must select once, then run final tests")
        alias_evaluation = config.get("evaluation", {})
        if alias_evaluation.get("primary_inference_dtype") != "torch.bfloat16":
            raise ValueError("alias continuation primary evaluation must use BF16")
        if (
            alias_evaluation.get("validation_diagnostic_inference_dtype")
            != "torch.float32"
        ):
            raise ValueError("alias continuation diagnostic evaluation must use FP32")
        if (
            alias_evaluation.get("alias_validation_suite_sha256")
            != ALIAS_INTERMEDIATE_VALIDATION_SUITE_SHA256
        ):
            raise ValueError("alias validation suite identity drifted")
        if (
            alias_evaluation.get("alias_test_suite_sha256")
            != ALIAS_INTERMEDIATE_TEST_SUITE_SHA256
        ):
            raise ValueError("alias test suite identity drifted")
        if alias_evaluation.get(
            "original_nine_test_suite_sha256"
        ) != "4938f7fbe8ac8cc1364139aa4f6a4f9fd7da303ca4e89b8b5a2cfda1c111f405":
            raise ValueError("alias continuation original-nine test bank drifted")
    elif is_stage_b:
        if run_kind == "formal_sft" and learning_rate != 3e-5:
            raise ValueError("formal Stage-B SFT LR must equal frozen 3e-5")
        if run_kind == "formal_sft_v2" and learning_rate != 1e-4:
            raise ValueError("formal Stage-B v2 LR must equal frozen 1e-4")
        approved_lrs = {3e-4, 1e-4, 3e-5}
        if run_kind in {"lr_tuning", "warmup_tuning", "schedule_tuning"}:
            approved_lrs = {3e-6, 1e-5, 3e-5, 1e-4}
        if run_kind in {"warmup_tuning", "schedule_tuning"}:
            approved_lrs = {1e-4}
        if run_kind not in {"formal_sft", "formal_sft_v2"} and learning_rate not in approved_lrs:
            raise ValueError("Stage-B SFT LR must be an approved pilot candidate")
        execution = config.get("execution", {})
        if bool(execution.get("auto_select_best", True)):
            raise ValueError("Stage-B must not auto-select a formal best model")
        if bool(execution.get("auto_run_test", True)):
            raise ValueError("Stage-B must not auto-run final test")
        expected_resume_interval = 20_000 if run_kind == "schedule_tuning" else 10_000
        if int(training.get("resume_checkpoint_interval", -1)) != expected_resume_interval:
            raise ValueError(
                "Stage-B training.resume_checkpoint_interval drifted"
            )
        depth3 = config.get("depth3_evaluation", {})
        depth3_enabled = bool(depth3.get("enabled", False))
        if depth3_enabled:
            if tuple(depth3.get("tasks", ())) != DEPTH3_TASKS:
                raise ValueError("depth3_evaluation.tasks must equal frozen three tasks")
            if int(depth3.get("validation_seed_base", -1)) != 888_889_000:
                raise ValueError("depth-3 validation seed base must equal 888889000")
            if int(depth3.get("validation_samples_per_task", -1)) != 50:
                raise ValueError("depth-3 validation must use 50 samples/task")
            if int(depth3.get("test_seed_base", -1)) != 999_998_000:
                raise ValueError("depth-3 test seed base must equal 999998000")
            if int(depth3.get("test_samples_per_task", -1)) != 100:
                raise ValueError("depth-3 test must use 100 samples/task")
        if run_kind == "formal_sft":
            if config.get("renderer_version") not in {
                STAGE_B_RENDERER_VERSION_V1,
                STAGE_B_RENDERER_VERSION,
            }:
                raise ValueError("formal Stage-B renderer version is unsupported")
            _validate_formal_stage_b_data_config(config)
            should_enable_depth3 = config.get("data_config_id") in {
                "atomic_inclusive_full",
                "composition_only_full",
            }
            if depth3_enabled != should_enable_depth3:
                raise ValueError(
                    "depth-3 evaluation must be enabled only for full-composition configs"
                )
        if run_kind == "formal_sft_v2":
            if config.get("renderer_version") != STAGE_B_RENDERER_VERSION:
                raise ValueError("formal Stage-B v2 requires corrected renderer v2")
            _validate_formal_stage_b_data_config(config)
            should_enable_depth3 = config.get("data_config_id") in {
                "atomic_inclusive_full",
                "composition_only_full",
            }
            if depth3_enabled != should_enable_depth3:
                raise ValueError(
                    "formal Stage-B v2 depth-3 must be enabled only for full configs"
                )
        if run_kind == "lr_tuning":
            if config.get("renderer_version") != STAGE_B_RENDERER_VERSION:
                raise ValueError("Stage-B LR tuning requires corrected renderer v2")
            if config.get("data_config_id") not in {
                "atomic_inclusive_full",
                "composition_only_full",
            }:
                raise ValueError("Stage-B LR tuning uses only two full configs")
            _validate_formal_stage_b_data_config(config)
            if depth3_enabled:
                raise ValueError("Stage-B LR tuning must not generate depth-3 data")
            if training.get("candidate_steps") != [
                250,
                500,
                1000,
                1500,
                2000,
                3000,
                4000,
                6000,
                7629,
            ]:
                raise ValueError("Stage-B LR tuning candidate steps drifted")
        if run_kind == "warmup_tuning":
            if config.get("renderer_version") != STAGE_B_RENDERER_VERSION:
                raise ValueError("warmup tuning requires corrected renderer v2")
            if config.get("data_config_id") not in {
                "atomic_inclusive_full",
                "composition_only_full",
            }:
                raise ValueError("warmup tuning uses only two full configs")
            _validate_formal_stage_b_data_config(config)
            if depth3_enabled:
                raise ValueError("warmup tuning must not generate depth-3 data")
            if training.get("candidate_steps") != [
                250,
                500,
                1000,
                1500,
                2000,
                3000,
                4000,
                6000,
                7629,
            ]:
                raise ValueError("warmup tuning candidate steps drifted")
        if run_kind == "schedule_tuning":
            if config.get("renderer_version") != STAGE_B_RENDERER_VERSION:
                raise ValueError("schedule tuning requires corrected renderer v2")
            if config.get("data_config_id") not in {
                "atomic_inclusive_full",
                "composition_only_full",
            }:
                raise ValueError("schedule tuning uses only two full configs")
            _validate_formal_stage_b_data_config(config)
            if depth3_enabled:
                raise ValueError("schedule tuning must not generate depth-3 data")
            if training.get("candidate_steps") != [
                250,
                500,
                1000,
                1500,
                2000,
                3000,
                4000,
                6000,
                8000,
                10000,
                12000,
                14000,
                15258,
            ]:
                raise ValueError("schedule tuning candidate steps drifted")
    elif learning_rate != 3e-4:
        raise ValueError("training.learning_rate must equal 3e-4")
    if training.get("precision") != "fp16":
        raise ValueError("training.precision must equal fp16")

    evaluation = config["evaluation"]
    if evaluation.get("metric") != "strict_first_tactic_exact_match":
        raise ValueError("only strict first-tactic exact match is allowed")
    if evaluation.get("validation_suite_seed_bases") != [888_888_000]:
        raise ValueError("use the single public validation seed base")
    if evaluation.get("test_suite_seed_bases") != [999_999_000]:
        raise ValueError("use the single public test seed base")
    if int(evaluation.get("test_samples_per_task", -1)) != 100:
        raise ValueError("evaluation.test_samples_per_task must equal 100")

    tokenizer = config["tokenizer"]
    if tokenizer.get("corpus_tasks") != "all_nine":
        raise ValueError("the public tokenizer corpus must cover all nine tasks")
    if int(tokenizer.get("corpus_samples", -1)) != 1_000_000:
        raise ValueError("tokenizer.corpus_samples must equal 1,000,000")
    if int(tokenizer.get("vocab_size", -1)) != 2048:
        raise ValueError("tokenizer.vocab_size must equal 2,048")
    if (
        is_alias_intermediate
        and tokenizer.get("expected_model_sha256")
        != ALIAS_INTERMEDIATE_TOKENIZER_SHA256
    ):
        raise ValueError("alias continuation tokenizer hash drifted")


def public_total_steps(config: dict[str, Any]) -> int:
    training = config["training"]
    tokens_per_step = int(training["global_batch_size"]) * int(
        config["model"]["block_size"]
    )
    return int(training["total_tokens"]) // tokens_per_step

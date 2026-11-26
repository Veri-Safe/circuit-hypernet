"""CPU-only checks for rejecting a wrong target, partial run or stale task record."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_base_step130_ttft_lr2e5_job.py"
spec = importlib.util.spec_from_file_location("base130_ttft_job_test", PATH)
job = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(job)


@pytest.fixture
def cfg():
    return {"format": job.FORMAT, "base_profile": job.BASE_PROFILE,
            "expected_model_aggregate_sha256": job.BASE_SHA,
            "learning_rates": [2e-5], "task_selection": "all", "rows_per_task": 128,
            "selection_views": 16, "max_new_tokens": 1024,
            "lora_rank": 256, "lora_alpha": 32, "lora_dropout": 0.0,
            "lora_use_rslora": True, "lora_target_modules": job.TARGETS,
            "weight_decay": 0.0, "warmup_ratio": 0.1, "max_grad_norm": 1.0,
            "world_size": 2, "output_root": "outputs/nvarc/base_ttft/step130_lr2e5_final128_test"}


def write_cfg(tmp_path, cfg):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.mark.parametrize("world", [1, 3, 5])
def test_accepts_only_fixed_budget_and_supported_scheduling(tmp_path, cfg, world):
    cfg["world_size"] = world
    assert job.settings(write_cfg(tmp_path, cfg)) == cfg


def test_relative_paths_in_manifests_use_forward_slashes():
    relative_path = "outputs/nvarc/base_ttft"
    assert job.relative(job.ROOT / "outputs" / "nvarc" / "base_ttft") == relative_path


def test_selected_ttft_profiles_share_launch_settings_and_pin_the_step65_receipt(monkeypatch):
    import hashlib
    monkeypatch.syspath_prepend(str(PATH.parent))
    import run_sft_ttft_lora_job as direct
    import run_s0_warmstart_ttft_job as warm

    monkeypatch.setenv("HC_ROOT", str(job.ROOT))
    monkeypatch.setenv("ARC_TTFT_WORKERS", "3")
    folder = job.ROOT / "s4_circuit/configs/ttft"
    base = job.settings(folder / "nvarc_base130_lr2e5_final128_20261117_v1.json")
    standard = direct.settings(folder / "nvarc_direct_sft_step65_ttft_final128_20261110_v1.json")[1]
    generated = warm.settings(folder / "nvarc_s0_warmstart_step65_final128_20261110_v1.json")[1]
    assert base["world_size"] == standard["world_size"] == generated["world_size"] == 3
    digest = hashlib.sha256((job.ROOT / standard["base_identity_manifest"]).read_bytes()).hexdigest()
    assert standard["base_identity_manifest_sha256"] == generated["base_identity_manifest_sha256"] == digest
    assert (job.ROOT / generated["warmstart"]["source_config"]).is_file()


def test_rejects_noncanonical_output_path(tmp_path, cfg):
    cfg["output_root"] = cfg["output_root"].replace("base_ttft/", "base_ttft/./")
    with pytest.raises(ValueError, match="normalized repository-relative path"):
        job.settings(write_cfg(tmp_path, cfg))


@pytest.mark.parametrize("field,value", [
    ("world_size", None), ("world_size", 0), ("learning_rates", [2.5e-5]),
    ("rows_per_task", 2), ("task_selection", "probe"), ("selection_views", 1),
    ("base_profile", "step65.yaml"), ("expected_model_aggregate_sha256", "0" * 64),
    ("lora_rank", 16), ("output_root", "outputs/nvarc/circuit_stage_c/ttft/t08_sft_ttft_final128_v2"),
])
def test_rejects_wrong_target_reduced_budget_or_old_output(tmp_path, cfg, field, value):
    cfg[field] = value
    with pytest.raises(ValueError):
        job.settings(write_cfg(tmp_path, cfg))


@pytest.fixture
def plan(cfg):
    tasks = [f"{i:08x}" for i in range(400)]
    return job.signed({"format": "base130_ttft_plan_v1", "settings": cfg,
        "task_ids": tasks, "task_assignments": [tasks[0::2], tasks[1::2]],
        "task_count": 400, "test_input_count": 419, "query_count": 7123,
        "learning_rates": [job.LR], "query_manifest_sha256": job.QUERY_SHA,
        "frozen_decode_protocol_sha256": job.PROTOCOL_SHA,
        "weights_verified_before_preparation": True,
        "checkpoint_identity": {"model_aggregate_sha256": job.BASE_SHA,
                                "source_step": 130, "base_checkpoint": job.BASE_PATH},
        "package_versions": {"torch": "test", "transformers": "test", "numpy": "test", "peft": "test", "tokenizers": "test"}})


def test_assignment_cannot_duplicate_or_drop_a_task(plan, cfg):
    job.validate_plan(plan, cfg)
    plan["task_assignments"][1][0] = plan["task_assignments"][0][0]
    with pytest.raises(ValueError, match="exactly once"):
        job.validate_plan(plan, cfg)


def test_plan_requires_real_preparation_weight_verification(plan, cfg):
    plan["weights_verified_before_preparation"] = False
    with pytest.raises(ValueError, match="identity or protocol"):
        job.validate_plan(plan, cfg)


@pytest.fixture
def task_record(plan):
    task = plan["task_ids"][0]
    views = []
    for i in range(17):
        record = {"query_index": i, "task_id": task, "test_index": 0,
                  "is_canonical": i == 0, "prompt_sha256": f"prompt{i}"}
        views.append(SimpleNamespace(query_index=i, manifest_record=lambda r=record: r))
    runtime = {"format": "nvarc_cuda_deterministic_runtime_v1", "mode": "strict_cuda_deterministic_v1",
               "deterministic_algorithms": True, "deterministic_debug_mode": 2,
               "cublas_workspace_config": ":4096:8", "cuda_matmul_allow_tf32": False,
               "cudnn_allow_tf32": False, "cudnn_benchmark": False, "cudnn_deterministic": True}
    rows_sha = "a" * 64
    value = {"format": "base130_ttft_result_v1", "status": "complete", "mode": job.MODE,
        "diagnostic_only": False, "task_id": task, "plan_sha256": plan["sha256"], "lr": job.LR,
        "initialization": "stock_peft_zero_delta", "identity": job.result_identity(plan, task, rows_sha),
        "worker_rank": 0, "support_rows_sha256": rows_sha,
        "checkpoint_identity_sha256": job.semantic_sha(plan["checkpoint_identity"]),
        "seed": 42, "runtime_settings": runtime, "attention_implementation": "sdpa",
        "package_versions": plan["package_versions"], "updates": 128, "lora_delta_l2": 0.1,
        "lora_trainable_parameter_count": 529801216,
        "update_trace_this_attempt": [{"step": i + 1, "steps": 128, "task_id": task,
             "lr": job.LR * job.cosine_schedule(i), "loss": 0.5, "grad_norm": 0.25} for i in range(128)],
        "predictions": [{**v.manifest_record(), "finished": True, "token_ids": [1, 12],
                         "new_tokens": 2, "normalized_logprob": -0.5, "grid": [[1]]} for v in views]}
    return task, views, value


def test_complete_task_record_accepts_warmup_first_call_zero(plan, task_record):
    task, views, value = task_record
    assert value["update_trace_this_attempt"][0]["lr"] == 0
    assert value["update_trace_this_attempt"][13]["lr"] == job.LR
    job.validate_result(job.signed(value), plan, task, views, rows_sha="a" * 64)


@pytest.mark.parametrize("mutation", ["partial_views", "duplicate_view", "wrong_finished_type", "wrong_prompt",
                                     "missing_update", "bad_schedule", "bad_seed", "non_strict", "wrong_target",
                                     "missing_decoder_field", "wrong_token_count", "wrong_token_type", "negative_delta"])
def test_rejects_signed_but_incomplete_or_mismatched_task(plan, task_record, mutation):
    task, views, original = task_record
    value = copy.deepcopy(original)
    if mutation == "partial_views":
        value["predictions"].pop()
    elif mutation == "duplicate_view":
        value["predictions"][-1] = copy.deepcopy(value["predictions"][0])
    elif mutation == "wrong_finished_type":
        value["predictions"][0]["finished"] = "false"
    elif mutation == "wrong_prompt":
        value["predictions"][0]["prompt_sha256"] = "wrong"
    elif mutation == "missing_update":
        value["update_trace_this_attempt"].pop()
    elif mutation == "bad_schedule":
        value["update_trace_this_attempt"][0]["lr"] = job.LR
    elif mutation == "bad_seed":
        value["seed"] = 24
    elif mutation == "non_strict":
        value["runtime_settings"]["cuda_matmul_allow_tf32"] = True
    elif mutation == "wrong_target":
        value["checkpoint_identity_sha256"] = "0" * 64
    elif mutation == "missing_decoder_field":
        del value["predictions"][0]["grid"]
    elif mutation == "wrong_token_count":
        value["predictions"][0]["new_tokens"] = 3
    elif mutation == "wrong_token_type":
        value["predictions"][0]["token_ids"][0] = True
    elif mutation == "negative_delta":
        value["lora_delta_l2"] = -0.01
    with pytest.raises(ValueError):
        job.validate_result(job.signed(value), plan, task, views)


def test_cached_task_cannot_reuse_different_support_rows(plan, task_record):
    task, views, value = task_record
    with pytest.raises(ValueError, match="identity changed"):
        job.validate_result(job.signed(value), plan, task, views, rows_sha="b" * 64)


def test_cached_task_checksum_is_required(plan, task_record):
    task, views, value = task_record
    value = job.signed(value)
    value["lora_delta_l2"] = 0.3
    with pytest.raises(ValueError, match="identity changed"):
        job.validate_result(value, plan, task, views)


def test_zero_lora_delta_is_a_valid_observation(plan, task_record):
    task, views, value = task_record
    value["lora_delta_l2"] = 0.0
    job.validate_result(job.signed(value), plan, task, views)


def test_truncated_query_is_complete_evidence_not_missing(plan, task_record):
    task, views, value = task_record
    value["predictions"][0].update(finished=False, grid=None, token_ids=[1] * 1024,
                                   new_tokens=1024, normalized_logprob=-0.5)
    job.validate_result(job.signed(value), plan, task, views)


def test_frozen_scorer_keeps_invalid_query_in_denominator():
    """Synthetic CPU unit fixture, not a benchmark subset evaluation."""
    from nvarc_qwen3.arc_eval_protocol import canonical_view_spec
    from nvarc_qwen3.nvarc_grid_grammar import decode_grid_tokens
    from s2_hypernet.nvarc_pg.eval_aggregation import group_query_predictions
    from s2_hypernet.nvarc_pg.eval_scoring import score_grouped_predictions

    # The real decoder returns None when its token sequence has no EOS.
    assert decode_grid_tokens([1], {1: 1}, 11, 12) is None
    assert decode_grid_tokens([1, 12], {1: 1}, 11, 12) == [[1]]
    views = [SimpleNamespace(query_index=i, task_id="synthetic", test_index=0,
                            is_canonical=i == 0, view_index=i,
                            view_descriptor="synthetic-identity", spec=canonical_view_spec(1))
             for i in range(17)]
    records = [{"query_index": i, "grid": None, "normalized_logprob": -0.5, "finished": False}
               for i in range(17)]
    challenges = {"synthetic": {"test": [{"input": [[0]]}]}}
    solutions = {"synthetic": [[[1]]]}
    grouped = group_query_predictions(views, records)
    metrics, _ = score_grouped_predictions(challenges, solutions, grouped)
    assert metrics["selected_exact@2"] == 0.0
    assert metrics["selection_view_count"] == 16
    assert metrics["valid_view_count"] == 0
    assert metrics["score_details"]["selected_exact@2"]["task_count"] == 1
    assert metrics["score_details"]["selected_exact@2"]["test_case_count"] == 1
    # finished is metadata; shared scoring uses grids, and never drops the task.
    changed_flags = [{**r, "finished": True} for r in records]
    unchanged, _ = score_grouped_predictions(challenges, solutions, group_query_predictions(views, changed_flags))
    assert unchanged == metrics

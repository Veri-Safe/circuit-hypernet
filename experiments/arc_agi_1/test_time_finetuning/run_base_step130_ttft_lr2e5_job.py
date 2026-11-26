"""One fixed-LR, full400 stock-LoRA TTFT comparison on original REARC8 step130.

Preparation and collection are CPU operations. A worker owns one visible GPU.
Existing Base plans and completed step65 experiments are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for package in ("s4_circuit", "s3_circuit", "s2_hypernet", "s1_lora_sft"):
    sys.path.insert(0, str(ROOT / package / "src"))

FORMAT = "s4_nvarc_base130_ttft_lr2e5_v1"
MODE = "base_step130_stock_lora_ttft"
LR = 2e-5
LABEL = "lr2e-5"
BASE_PROFILE = "s2_hypernet/configs/nvarc_pg/arc1_rearc8_s42_step130_v1.yaml"
BASE_PATH = "outputs/nvarc/qwen3_4b_sft/rearc_scale_coord/0804_rearc_scale_50ep_v2/hf/r8_s42/step_130-hf"
BASE_SHA = "b74bcb895fe162511ab8d047648a9d3cbba110fead5f3d58b9a24e9456d0084c"
QUERY_SHA = "72335c7c5aa4c8b3671393f6b85907a85382d4713a33bb3f0fdf6f070b747333"
PROTOCOL_SHA = "54037b1baac229754a7f372b1de887025482733f39941184ada0afa6877d235e"
CHALLENGE_SHA = "46b27a6330e9b2dde16e33c81f80f96a3d7553d1c44b14eb712a256349fd55f5"
FROZEN_UTIL = "s4_circuit/scripts/run_sft_ttft_lora_job.py"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
           "down_proj", "embed_tokens", "lm_head"]
SCORER_FILES = (
    "s4_circuit/scripts/run_base_step130_ttft_lr2e5_job.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/config.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/profiles.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_manifest.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_protocol.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_aggregation.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_scoring.py",
    "s1_lora_sft/src/nvarc_qwen3/arc_eval_protocol.py",
)
CODE_FILES = (*SCORER_FILES, FROZEN_UTIL,
    "s4_circuit/src/s4_circuit/ttft/data.py",
    "s4_circuit/src/s4_circuit/ttft/state.py",
    "s4_circuit/src/s4_circuit/contracts.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/runtime.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/checkpoint.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/target_layout.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/events.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/objective.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/decode.py",
    "s3_circuit/src/s3circuit/nvarc_circuit/deterministic_runtime.py",
    "s1_lora_sft/src/nvarc_qwen3/nvarc_grid_grammar.py",
    "ref_repo/NVARC/ARC-AGI1/002_ivan_arc1.ipynb",
)


def semantic_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed(value: dict) -> dict:
    return {**value, "sha256": semantic_sha(value)}


def read_signed(path: Path) -> dict:
    value = json.loads(path.read_text())
    body = {k: v for k, v in value.items() if k != "sha256"}
    if value.get("sha256") != semantic_sha(body):
        raise ValueError(f"Checksum changed: {path}")
    return value


def relative(path: str | Path) -> str:
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve().relative_to(ROOT.resolve()).as_posix()


def cosine_schedule(step: int, total: int = 128, warmup_ratio: float = 0.1) -> float:
    # Same schedule as the frozen TTFT utility; kept CPU-only for result audit.
    warmup = math.ceil(total * warmup_ratio)
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def settings(profile: Path) -> dict:
    cfg = json.loads(profile.read_text())
    cfg.setdefault("world_size", int(os.environ.get("ARC_TTFT_WORKERS", "1")))
    expected = {
        "format": FORMAT, "base_profile": BASE_PROFILE,
        "expected_model_aggregate_sha256": BASE_SHA,
        "learning_rates": [LR], "task_selection": "all", "rows_per_task": 128,
        "selection_views": 16, "max_new_tokens": 1024,
        "lora_rank": 256, "lora_alpha": 32, "lora_dropout": 0.0,
        "lora_use_rslora": True, "lora_target_modules": TARGETS,
        "weight_decay": 0.0, "warmup_ratio": 0.1, "max_grad_norm": 1.0,
    }
    if set(cfg) != set(expected) | {"world_size", "output_root"}:
        raise ValueError("Base130 profile has missing or unsupported fields")
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("Base130 target or fixed full400 TTFT settings changed")
    if type(cfg["world_size"]) is not int or not 0 < cfg["world_size"] <= 400:
        raise ValueError("Task workers must be positive and no greater than the task count")
    output = Path(relative(cfg["output_root"]))
    if output.parent != Path("outputs/nvarc/base_ttft") or not output.name.startswith("step130_lr2e5_final128_"):
        raise ValueError("Use a new step130_lr2e5_final128_* output under outputs/nvarc/base_ttft")
    if cfg["output_root"] != output.as_posix():
        raise ValueError("output_root must be a normalized repository-relative path")
    return cfg


def package_versions() -> dict:
    return {name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "numpy", "tokenizers")}


def native_runtime(cfg: dict, *, verify_weights: bool):
    from s2_hypernet.nvarc_pg.config import load_experiment_config
    from s2_hypernet.nvarc_pg.checkpoint import inspect_frozen_hf_checkpoint

    os.environ["HC_ROOT"] = str(ROOT)
    runtime = load_experiment_config(ROOT / cfg["base_profile"])
    if (runtime.profile_id != "arc1_rearc8_s42_step130"
            or runtime.base.source_step != 130
            or runtime.base.source_kind != "nvarc_rearc8_s42_fullft"
            or relative(runtime.base.hf_checkpoint) != BASE_PATH
            or runtime.profile.hf_aggregate_sha256 != BASE_SHA):
        raise ValueError("Target is not the pinned original REARC8 step130 Base")
    model_dir = Path(runtime.base.hf_checkpoint)
    files = {}
    for name, expected_sha in runtime.profile.hf_artifact_sha256:
        path = model_dir / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".json" and file_sha(path) != expected_sha:
            raise ValueError(f"Small Base artifact changed: {name}")
        files[name] = {"sha256": expected_sha, "size_bytes": path.stat().st_size}
    if semantic_sha({"files": files}) != BASE_SHA:
        raise ValueError("Base artifact sizes or pinned hashes changed")
    token_dir = Path(runtime.base.tokenizer)
    token_sha = file_sha(token_dir / "tokenizer_config.json")
    mapping_sha = file_sha(token_dir / "nvarc_16t_token_mapping.json")
    if (token_sha != runtime.profile.tokenizer_config_sha256
            or mapping_sha != runtime.profile.token_mapping_sha256):
        raise ValueError("Pinned Base tokenizer identity changed")
    if verify_weights:
        receipt = inspect_frozen_hf_checkpoint(
            hf_checkpoint=model_dir, tokenizer=token_dir,
            source_run_manifest=runtime.base.source_run_manifest or None,
            profile=runtime.profile, layout=runtime.target_layout)
        if receipt["hf_aggregate_sha256"] != BASE_SHA:
            raise ValueError("Verified Base weight aggregate changed")
    identity = {
        "source_kind": runtime.base.source_kind, "source_step": 130,
        "base_checkpoint": BASE_PATH, "model_aggregate_sha256": BASE_SHA,
        "model_artifact_files": files, "tokenizer": relative(token_dir),
        "tokenizer_config_sha256": token_sha, "token_mapping_sha256": mapping_sha,
        "base_profile": cfg["base_profile"],
        "base_profile_sha256": file_sha(ROOT / cfg["base_profile"]),
    }
    return runtime, identity


def frozen_views(runtime):
    from nvarc_qwen3.arc_eval_protocol import load_challenges
    from s2_hypernet.nvarc_pg.eval_manifest import build_query_views
    from s2_hypernet.nvarc_pg.eval_protocol import load_profile_eval_protocol, validate_profile_dataset_identity
    from s2_hypernet.nvarc_pg.runtime import load_tokenizer

    protocol, protocol_sha = load_profile_eval_protocol(runtime.evaluation)
    if protocol_sha != PROTOCOL_SHA:
        raise ValueError("Frozen full-view decoding protocol changed")
    challenges = load_challenges(runtime.evaluation.challenges)
    validate_profile_dataset_identity(runtime.evaluation.protocol_kind, challenges)
    if (len(challenges) != 400 or sum(len(x["test"]) for x in challenges.values()) != 419
            or semantic_sha(challenges) != CHALLENGE_SHA):
        raise ValueError("ARC-AGI-1 400/419 challenge identity changed")
    tokenizer = load_tokenizer(runtime)
    views = build_query_views(challenges, config=protocol, tokenizer=tokenizer)
    if (len(views) != 7123 or semantic_sha([v.manifest_record() for v in views]) != QUERY_SHA
            or set(Counter((v.task_id, v.test_index) for v in views).values()) != {17}
            or sum(v.is_canonical for v in views) != 419):
        raise ValueError("Full400 7123-query manifest changed")
    return challenges, tuple(views), tokenizer


def validate_plan(plan: dict, cfg: dict) -> None:
    tasks = plan.get("task_ids", [])
    assignments = plan.get("task_assignments", [])
    if (plan.get("format") != "base130_ttft_plan_v1" or plan.get("settings") != cfg
            or plan.get("task_count") != 400 or plan.get("test_input_count") != 419
            or plan.get("query_count") != 7123 or plan.get("query_manifest_sha256") != QUERY_SHA
            or plan.get("learning_rates") != [LR]
            or plan.get("frozen_decode_protocol_sha256") != PROTOCOL_SHA
            or plan.get("checkpoint_identity", {}).get("model_aggregate_sha256") != BASE_SHA
            or plan.get("checkpoint_identity", {}).get("source_step") != 130
            or plan.get("checkpoint_identity", {}).get("base_checkpoint") != BASE_PATH
            or not plan.get("weights_verified_before_preparation")):
        raise ValueError("Base130 plan identity or protocol changed")
    if (tasks != sorted(set(tasks)) or len(tasks) != 400
            or len(assignments) != cfg["world_size"]
            or Counter(t for lane in assignments for t in lane) != Counter(tasks)):
        raise ValueError("Worker assignment must cover all400 tasks exactly once")


def prepare(profile: Path, *, check_only: bool = False) -> dict:
    from s4_circuit.ttft.state import atomic_json, exclusive_lock

    cfg = settings(profile)
    runtime, identity = native_runtime(cfg, verify_weights=not check_only)
    challenges, _, _ = frozen_views(runtime)
    if check_only:
        return {"status": "checked", "weights_verified": False,
                "base_checkpoint": BASE_PATH, "task_count": 400,
                "test_input_count": 419, "query_count": 7123,
                "world_size": cfg["world_size"], "learning_rates": [LR]}
    tasks = sorted(challenges)
    body = {
        "format": "base130_ttft_plan_v1", "mode": MODE, "settings": cfg,
        "checkpoint_identity": identity, "weights_verified_before_preparation": True,
        "task_count": 400, "test_input_count": 419, "query_count": 7123,
        "task_ids": tasks, "task_assignments": [tasks[r::cfg["world_size"]] for r in range(cfg["world_size"])],
        "assignment_algorithm": "sorted_round_robin_v1", "learning_rates": [LR],
        "query_manifest_sha256": QUERY_SHA, "frozen_decode_protocol_sha256": PROTOCOL_SHA,
        "frozen_protocol_version": "arc_agi1_nvarc_public_validation_v1",
        "challenge_sha256": CHALLENGE_SHA, "package_versions": package_versions(),
        "code_files_sha256": {rel: file_sha(ROOT / rel) for rel in CODE_FILES},
    }
    plan = signed(body)
    validate_plan(plan, cfg)
    output = ROOT / cfg["output_root"]
    with exclusive_lock(output / "locks" / "prepare.lock"):
        path = output / "plan.json"
        if path.exists():
            if read_signed(path) != plan:
                raise ValueError("Existing plan differs; choose a new output_root")
        else:
            if any(p.exists() for p in (output / "results", output / "workers", output / "summary.json")):
                raise ValueError("Refusing to prepare over existing unbound experiment artifacts")
            atomic_json(path, plan)
    return {"status": "prepared", "plan_sha256": plan["sha256"],
            "output_root": str(output), "weights_verified": True}


def load_plan(profile: Path, *, training: bool):
    cfg = settings(profile)
    plan = read_signed(ROOT / cfg["output_root"] / "plan.json")
    validate_plan(plan, cfg)
    for rel in CODE_FILES if training else SCORER_FILES:
        if file_sha(ROOT / rel) != plan["code_files_sha256"].get(rel):
            raise ValueError(f"{'Training' if training else 'Scoring'} dependency changed: {rel}")
    if training and package_versions() != plan["package_versions"]:
        raise ValueError("Runtime packages changed since preparation")
    if not training:
        versions = package_versions()
        if any(versions[k] != plan["package_versions"][k] for k in ("numpy", "transformers", "tokenizers")):
            raise ValueError("Scoring/tokenizer packages changed since preparation")
    runtime, identity = native_runtime(cfg, verify_weights=False)
    if identity != plan["checkpoint_identity"]:
        raise ValueError("Native Base checkpoint/profile identity changed since preparation")
    return cfg, plan, runtime


def strict_runtime(value: dict) -> bool:
    expected = {"format": "nvarc_cuda_deterministic_runtime_v1",
        "mode": "strict_cuda_deterministic_v1", "deterministic_algorithms": True,
        "deterministic_debug_mode": 2, "cublas_workspace_config": ":4096:8",
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True}
    return isinstance(value, dict) and all(value.get(k) == v for k, v in expected.items())


def result_identity(plan: dict, task: str, rows_sha: str) -> str:
    return semantic_sha({"plan_sha256": plan["sha256"], "task_id": task, "lr": LR,
                         "support_rows_sha256": rows_sha, "initialization": "stock_peft_zero_delta"})


def validate_result(value: dict, plan: dict, task: str, task_views, *, rows_sha: str | None = None) -> None:
    body = {k: v for k, v in value.items() if k != "sha256"}
    stored_rows = value.get("support_rows_sha256")
    rank = value.get("worker_rank")
    if (value.get("sha256") != semantic_sha(body) or value.get("format") != "base130_ttft_result_v1"
            or value.get("status") != "complete" or value.get("mode") != MODE
            or value.get("diagnostic_only") is not False or value.get("task_id") != task
            or value.get("plan_sha256") != plan["sha256"] or value.get("lr") != LR
            or value.get("initialization") != "stock_peft_zero_delta"
            or value.get("seed") != 42
            or value.get("identity") != result_identity(plan, task, stored_rows)
            or type(rank) is not int or not 0 <= rank < len(plan["task_assignments"])
            or task not in plan["task_assignments"][rank]
            or value.get("checkpoint_identity_sha256") != semantic_sha(plan["checkpoint_identity"])
            or not isinstance(stored_rows, str) or len(stored_rows) != 64
            or (rows_sha is not None and stored_rows != rows_sha)):
        raise ValueError(f"Base130 result identity changed: {task}")
    if (not strict_runtime(value.get("runtime_settings"))
            or value.get("attention_implementation") != "sdpa"
            or value.get("package_versions") != plan["package_versions"]):
        raise ValueError(f"Base130 result runtime changed: {task}")
    trace = value.get("update_trace_this_attempt", [])
    delta = value.get("lora_delta_l2", float("nan"))
    if value.get("updates") != 128 or len(trace) != 128 or not math.isfinite(delta) or delta < 0:
        raise ValueError(f"Base130 result update budget/delta invalid: {task}")
    if value.get("lora_trainable_parameter_count") != 529801216:
        raise ValueError(f"Base130 trainable LoRA target count changed: {task}")
    for i, entry in enumerate(trace):
        if (entry.get("step") != i + 1 or entry.get("steps") != 128 or entry.get("task_id") != task
                or not math.isclose(entry.get("lr", -1), LR * cosine_schedule(i), rel_tol=1e-12, abs_tol=1e-20)
                or not math.isfinite(entry.get("loss", float("nan")))
                or not math.isfinite(entry.get("grad_norm", float("nan")))):
            raise ValueError(f"Base130 update trace invalid: {task}/{i}")
    expected = {v.query_index: v.manifest_record() for v in task_views}
    predictions = value.get("predictions", [])
    actual = {p["query_index"]: p for p in predictions}
    if len(predictions) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"Base130 full-view coverage incomplete: {task}")
    for query, manifest in expected.items():
        prediction = actual[query]
        if any(prediction.get(k) != v for k, v in manifest.items()):
            raise ValueError(f"Base130 query manifest changed: {task}/{query}")
        tokens = prediction.get("token_ids")
        count = prediction.get("new_tokens")
        logprob = prediction.get("normalized_logprob")
        if (type(prediction.get("finished")) is not bool
                or not isinstance(tokens, list) or any(type(t) is not int or not 0 <= t < 16 for t in tokens)
                or type(count) is not int or count != len(tokens) or not 1 <= count <= 1024
                or type(logprob) not in (int, float) or not math.isfinite(logprob)
                or "grid" not in prediction
                or (prediction["grid"] is not None and not isinstance(prediction["grid"], list))):
            raise ValueError(f"Base130 decoder record malformed: {task}/{query}")
        # A complete query record may have finished=False/grid=None after the
        # token budget. Keep it in the denominator; the frozen scorer assigns
        # no candidate to invalid output. Neither EOS nor positive LoRA delta
        # is an experiment-completion requirement.


def frozen_utility():
    spec = importlib.util.spec_from_file_location("base130_frozen_ttft_utility", ROOT / FROZEN_UTIL)
    if spec is None or spec.loader is None:
        raise RuntimeError("Frozen TTFT utility unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worker(profile: Path, rank: int) -> None:
    import time
    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from s2_hypernet.nvarc_pg.runtime import _configure_runtime_from_environment, load_frozen_qwen, set_reproducible_seed
    from s2_hypernet.nvarc_pg.decode import constrained_greedy_batch, verify_token_contract
    from s4_circuit.ttft.data import build_support_rows, rows_identity
    from s4_circuit.ttft.state import atomic_json, emit, exclusive_lock

    runtime_settings = _configure_runtime_from_environment()
    if not strict_runtime(runtime_settings) or os.environ.get("NVARC_ATTENTION_IMPLEMENTATION") != "sdpa":
        raise ValueError("Base130 requires strict_cuda_deterministic_v1 and sdpa")
    if torch.cuda.device_count() != 1:
        raise ValueError("Each Base130 worker requires exactly one visible GPU")
    cfg, plan, runtime = load_plan(profile, training=True)
    if not 0 <= rank < cfg["world_size"]:
        raise ValueError("Worker rank is outside configured world_size")
    challenges, views, tokenizer = frozen_views(runtime)
    if sorted(challenges) != plan["task_ids"]:
        raise ValueError("Plan task IDs differ from the frozen evaluation challenges")
    tasks = plan["task_assignments"][rank]
    by_task = {task: tuple(v for v in views if v.task_id == task) for task in tasks}
    output = ROOT / cfg["output_root"]
    util = frozen_utility()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    with exclusive_lock(output / "locks" / f"worker-{rank}.lock"):
        set_reproducible_seed(42)
        base = load_frozen_qwen(runtime, device=device, for_training=False)
        base.config.use_cache = False
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        model = get_peft_model(base, LoraConfig(
            r=256, lora_alpha=32, lora_dropout=0.0, bias="none",
            target_modules=TARGETS, use_rslora=True, task_type="CAUSAL_LM"))
        trainable = [p for p in model.parameters() if p.requires_grad]
        if any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
            raise ValueError("Base weights must remain frozen")
        if sum(p.numel() for p in trainable) != 529801216:
            raise ValueError("Stock rank256 LoRA parameter count differs from comparison")
        stock = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        zero_keys = [k for k in stock if "lora_B" in k or "lora_embedding_A" in k]
        if not zero_keys or any(torch.count_nonzero(stock[k]).item() for k in zero_keys):
            raise ValueError("Stock PEFT initialization must have zero effective delta")
        token_contract = verify_token_contract(tokenizer, model)
        for task in tasks:
            rows = build_support_rows(task, challenges[task], tokenizer, ROOT)
            if len(rows) != 128:
                raise ValueError(f"Expected128 augmented support rows: {task}")
            rows_sha = rows_identity(rows)
            path = output / "results" / LABEL / f"{task}.json"
            if path.exists():
                validate_result(read_signed(path), plan, task, by_task[task], rows_sha=rows_sha)
                emit("base130_ttft_puzzle_reused", task_id=task, rank=rank)
                continue
            tokenized = tuple(row.tokenize(tokenizer) for row in rows)
            util.reset_lora(model, stock)
            model.train()
            model.config.use_cache = False
            optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=0.0, foreach=False)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: util.cosine_schedule(step, 128, 0.1))
            order = torch.randperm(128, generator=torch.Generator().manual_seed(42)).tolist()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            began = time.monotonic()
            trace = []
            for position in range(128):
                optimizer.zero_grad(set_to_none=True)
                metrics = util.adapt_row(model, tokenized[order[position]], device=device)
                norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                active_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                report = {**metrics, "step": position + 1, "steps": 128, "lr": active_lr,
                          "grad_norm": float(norm.detach()), "seconds": time.monotonic() - began, "task_id": task}
                trace.append(report)
                emit("base130_ttft_update", rank=rank, **report)
            del optimizer, scheduler
            model.eval()
            model.config.use_cache = True
            predictions = []
            for test_index in range(len(challenges[task]["test"])):
                batch = sorted((v for v in by_task[task] if v.test_index == test_index), key=lambda v: v.query_index)
                decoded = constrained_greedy_batch(
                    type("System", (), {"base_model": model})(), tokenizer,
                    [v.prompt for v in batch], token_contract=token_contract,
                    max_new_tokens=1024, device=device, condition_prompts=None, precomputed_state=None)
                predictions.extend({**view.manifest_record(), **prediction, "base130_ttft_lora": True}
                                   for view, prediction in zip(batch, decoded, strict=True))
            torch.cuda.synchronize(device)
            value = signed({
                "format": "base130_ttft_result_v1", "status": "complete", "mode": MODE,
                "diagnostic_only": False, "initialization": "stock_peft_zero_delta",
                "identity": result_identity(plan, task, rows_sha), "plan_sha256": plan["sha256"],
                "task_id": task, "lr": LR, "worker_rank": rank, "support_rows_sha256": rows_sha,
                "checkpoint_identity_sha256": semantic_sha(plan["checkpoint_identity"]),
                "updates": 128, "update_trace_this_attempt": trace,
                "lora_delta_l2": util.lora_delta_l2(model, stock),
                "lora_trainable_parameter_count": sum(p.numel() for p in trainable),
                "runtime_settings": runtime_settings, "attention_implementation": "sdpa",
                "package_versions": package_versions(), "seed": 42,
                "seconds_this_attempt": time.monotonic() - began,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                "predictions": predictions,
            })
            validate_result(value, plan, task, by_task[task], rows_sha=rows_sha)
            atomic_json(path, value)
            emit("base130_ttft_puzzle_complete", rank=rank, task_id=task, seconds=value["seconds_this_attempt"])
        atomic_json(output / "workers" / f"{rank}.json", signed({
            "format": "base130_ttft_worker_v1", "status": "worker_complete",
            "plan_sha256": plan["sha256"], "worker_rank": rank, "task_ids": tasks,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }))
        emit("base130_ttft_worker_complete", rank=rank, task_count=len(tasks))


@contextmanager
def collection_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def collect(profile: Path) -> bool:
    from nvarc_qwen3.arc_eval_protocol import load_solutions, validate_arc_agi1_identity
    from s2_hypernet.nvarc_pg.eval_aggregation import group_query_predictions
    from s2_hypernet.nvarc_pg.eval_scoring import score_grouped_predictions
    from s4_circuit.ttft.state import atomic_json

    cfg, plan, runtime = load_plan(profile, training=False)
    output = ROOT / cfg["output_root"]
    for rank, tasks in enumerate(plan["task_assignments"]):
        marker = output / "workers" / f"{rank}.json"
        if not marker.is_file():
            return False
        receipt = read_signed(marker)
        if (receipt.get("format") != "base130_ttft_worker_v1" or receipt.get("status") != "worker_complete"
                or receipt.get("plan_sha256") != plan["sha256"] or receipt.get("worker_rank") != rank
                or receipt.get("task_ids") != tasks):
            raise ValueError(f"Worker completion receipt changed: {rank}")
    if {p.stem for p in (output / "results" / LABEL).glob("*.json")} != set(plan["task_ids"]):
        return False
    with collection_lock(output / "locks" / "collect.lock"):
        challenges, views, _ = frozen_views(runtime)
        if sorted(challenges) != plan["task_ids"]:
            raise ValueError("Plan task IDs differ from the frozen evaluation challenges")
        solutions = load_solutions(runtime.evaluation.solutions, challenges)
        validate_arc_agi1_identity(challenges, solutions)
        records, result_rows = [], []
        for task in plan["task_ids"]:
            value = read_signed(output / "results" / LABEL / f"{task}.json")
            validate_result(value, plan, task, tuple(v for v in views if v.task_id == task))
            records.extend(value["predictions"])
            result_rows.append({k: value[k] for k in ("task_id", "lr", "updates", "lora_delta_l2", "seconds_this_attempt")})
        if len(records) != 7123 or {p["query_index"] for p in records} != set(range(7123)):
            raise ValueError("Collected predictions do not cover all7123 query views exactly once")
        records.sort(key=lambda p: p["query_index"])
        metrics, submission = score_grouped_predictions(challenges, solutions, group_query_predictions(views, records))
        if metrics.get("primary_metric") != "selected_exact@2" or metrics.get("selection_view_count") != 6704:
            raise ValueError("Frozen selected_exact@2 evaluation protocol drifted")
        metrics["protocol_version"] = plan["frozen_protocol_version"]
        summary = signed({"format": "base130_ttft_summary_v1", "status": "complete", "mode": MODE,
            "diagnostic_only": False, "plan_sha256": plan["sha256"], "base_checkpoint": BASE_PATH,
            "checkpoint_identity": plan["checkpoint_identity"], "task_count": 400,
            "test_input_count": 419, "query_count": 7123, "optimizer_update_count": 51200,
            "metrics": {LABEL: metrics}, "results": result_rows})
        for path, value in ((output / LABEL / "metrics.json", metrics),
                            (output / LABEL / "submission.json", submission),
                            (output / "summary.json", summary)):
            if path.exists():
                if json.loads(path.read_text()) != value:
                    raise ValueError(f"Refusing to overwrite differing collected output: {path}")
            else:
                atomic_json(path, value)
        print(json.dumps({"status": "complete", "summary": str(output / "summary.json"),
                          "selected_exact@2": metrics["selected_exact@2"]}), flush=True)
    return True


def run_job(profile: Path) -> None:
    """Run only this job's worker processes and collect after every exit is zero."""
    from s4_circuit.ttft.state import atomic_json, exclusive_lock

    cfg = settings(profile)
    output = ROOT / cfg["output_root"]
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(cfg["world_size"])))).split(",")
    if (len(devices) != cfg["world_size"] or any(not d.strip() for d in devices)
            or len(set(devices)) != len(devices)):
        raise ValueError("Visible GPU list must contain exactly world_size unique devices")
    with exclusive_lock(output / "locks" / "job.lock"):
        prepared = prepare(profile)
        cfg, plan, _ = load_plan(profile, training=True)
        attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{os.getpid()}"
        receipt = {"format": "base130_ttft_job_v1", "attempt_id": attempt,
                   "plan_sha256": prepared["plan_sha256"], "status": "running",
                   "started_at_utc": datetime.now(timezone.utc).isoformat(),
                   "profile": relative(profile), "world_size": cfg["world_size"],
                   "log_root": os.environ.get("BASE_TTFT_LOG_ROOT"),
                   "worker_pids": [], "exit_codes": [None] * cfg["world_size"]}
        atomic_json(output / "job.json", signed(receipt))
        children = []
        previous_handlers = {}

        def interrupted(signum, _frame):
            raise SystemExit(128 + signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupted)
        try:
            for rank, device in enumerate(devices):
                child = subprocess.Popen(
                    [sys.executable, "-u", str(Path(__file__).resolve()),
                     "--profile", str(profile.resolve()), "--worker", str(rank)],
                    cwd=ROOT, start_new_session=True,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": device,
                         "OMP_NUM_THREADS": "4", "NUMBA_NUM_THREADS": "4"})
                children.append(child)
                receipt["worker_pids"].append(child.pid)
            atomic_json(output / "job.json", signed(receipt))
            while any(child.poll() is None for child in children):
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("A Base130 worker failed; completed task files are preserved")
                time.sleep(2)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError("A Base130 worker exited unsuccessfully")
            if not collect(profile):
                raise RuntimeError("All workers exited zero but collection is incomplete")
            receipt["status"] = "complete"
            receipt["summary"] = str(output / "summary.json")
        except BaseException as error:
            receipt["status"] = "failed"
            receipt["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            for child in children:
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for child in children:
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
            receipt["exit_codes"] = [child.returncode for child in children] + [None] * (cfg["world_size"] - len(children))
            receipt["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            final = signed(receipt)
            atomic_json(output / "attempts" / f"{attempt}.json", final)
            atomic_json(output / "job.json", final)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        print(json.dumps({"status": "complete", "plan_sha256": plan["sha256"],
                          "job_receipt": str(output / "job.json")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--worker", type=int)
    mode.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.check_only or args.prepare_only:
        print(json.dumps(prepare(args.profile, check_only=args.check_only), sort_keys=True), flush=True)
    elif args.worker is not None:
        worker(args.profile, args.worker)
    elif args.collect:
        if not collect(args.profile):
            print(json.dumps({"status": "pending", "reason": "missing_worker_receipts_or_results"}), flush=True)
            raise SystemExit(2)
    else:
        run_job(args.profile)


if __name__ == "__main__":
    main()

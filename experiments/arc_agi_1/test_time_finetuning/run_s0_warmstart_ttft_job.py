from __future__ import annotations

import argparse
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sft_ttft_lora_job as direct

import torch

from s4_circuit.contracts import semantic_sha256, sha256_file
from s4_circuit.ttft.data import build_support_rows, rows_identity
from s4_circuit.ttft.state import atomic_json, emit, exclusive_lock
from s4_circuit.ttft.warmstart import initialize_peft_from_generated


MODE = "s0_warmstart_lora_ttft"


def settings(profile: Path):
    root, cfg = direct.settings(profile)
    if cfg["format"] != "s4_nvarc_sft_ttft_v3":
        raise ValueError("S0 warmstart requires the step65 v3 protocol")
    warmstart = cfg.get("warmstart", {})
    if warmstart.get("anchor_rule") != "first_test_canonical_v1" or warmstart.get("gate") != 0.25:
        raise ValueError("S0 warmstart requires first_test_canonical_v1 and gate=0.25")
    for key in ("source_config", "source_checkpoint"):
        if not warmstart.get(key):
            raise ValueError(f"Missing warmstart {key}")
    if not cfg.get("cost_reference_summary"):
        raise ValueError("Missing completed direct-TTFT cost reference")
    return root, cfg


def code_identity(root: Path) -> str:
    paths = [
        Path(__file__).resolve(),
        root / "s4_circuit/src/s4_circuit/ttft/s0_anchor.py",
        root / "s4_circuit/src/s4_circuit/ttft/warmstart.py",
        root / "s4_circuit/src/s4_circuit/ttft/state.py",
        root / "s4_circuit/src/s4_circuit/config.py",
        root / "s4_circuit/src/s4_circuit/model.py",
        root / "s4_circuit/src/s4_circuit/lora_scale.py",
        root / "s4_circuit/src/s4_circuit/contracts.py",
        root / "s4_circuit/src/s4_circuit/runtime.py",
        root / "s4_circuit/src/s4_circuit/initialization.py",
        root / "s4_circuit/src/s4_circuit/checkpoint.py",
        root / "s4_circuit/src/s4_circuit/evaluation.py",
        root / "s4_circuit/src/s4_circuit/artifacts.py",
        root / "s4_circuit/src/s4_circuit/join.py",
        root / "s4_circuit/src/s4_circuit/pg_core.py",
        root / "s4_circuit/src/s4_circuit/fusion.py",
        root / "s4_circuit/src/s4_circuit/encoders.py",
        root / "s4_circuit/src/s4_circuit/representations.py",
        root / "s4_circuit/src/s4_circuit/arc_targeted_repair_v1/identity_compat.py",
        root / "s3_circuit/src/s3circuit/nvarc_circuit/events.py",
        root / "s3_circuit/src/s3circuit/nvarc_circuit/storage.py",
        root / "s3_circuit/src/s3circuit/nvarc_circuit/finalize.py",
        root / "s2_hypernet/src/s2_hypernet/config.py",
        root / "s2_hypernet/src/s2_hypernet/parameter_generator.py",
        root / "s2_hypernet/src/s2_hypernet/lora_tokenizer.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/config.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/profiles.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/checkpoint.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/model.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/generated_lora.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/parameter_generator.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/target_layout.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_protocol.py",
        root / "s1_lora_sft/src/nvarc_qwen3/eval_nvarc_internal.py",
        root / "s1_lora_sft/src/nvarc_qwen3/nvarc_grid_grammar.py",
    ]
    return semantic_sha256({
        "direct_runtime": direct.code_identity(root),
        "files": {str(path.relative_to(root)): sha256_file(path) for path in paths},
    })


def signed(value: dict) -> dict:
    return {**value, "sha256": semantic_sha256(value)}


def read_signed(path: Path) -> dict:
    value = direct.read_json(path)
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != semantic_sha256(body):
        raise ValueError(f"Artifact checksum changed: {path}")
    return value


def assign_tasks(costs: dict[str, float], workers: int):
    if workers <= 0 or not costs or any(not math.isfinite(c) or c <= 0 for c in costs.values()):
        raise ValueError("Task assignment requires positive finite costs and worker count")
    lanes = [[] for _ in range(workers)]
    totals = [0.0] * workers
    for task in sorted(costs, key=lambda item: (-costs[item], item)):
        lane = min(range(workers), key=lambda index: (totals[index], index))
        lanes[lane].append(task)
        totals[lane] += costs[task]
    return lanes, totals


def reference_costs(summary: dict, task_ids: list[str], learning_rates: list[float]):
    if summary.get("status") != "complete" or summary.get("task_count") != 400:
        raise ValueError("Cost reference must be the completed 400-task direct sweep")
    expected = {(task, lr) for task in task_ids for lr in learning_rates}
    observed = {}
    for row in summary["results"]:
        key = (row["task_id"], row["lr"])
        seconds = float(row["seconds_this_attempt"])
        if key in observed or key not in expected or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("Cost reference has duplicate, unexpected, or invalid task/LR records")
        observed[key] = seconds
    if set(observed) != expected:
        raise ValueError("Cost reference does not cover all 400 tasks and seven LRs")
    return {task: sum(observed[(task, lr)] for lr in learning_rates) for task in task_ids}


def validate_plan_shape(plan: dict):
    tasks = plan["task_ids"]
    if len(tasks) != 400 or tasks != sorted(set(tasks)):
        raise ValueError("Warmstart plan must contain 400 unique sorted tasks")
    if plan.get("test_input_count") != 419 or plan.get("query_count") != 7123:
        raise ValueError("Warmstart plan must contain 419 test inputs and 7123 views")
    lanes = plan["task_assignments"]
    flat = [task for lane in lanes for task in lane]
    if len(lanes) != plan["settings"]["world_size"] or len(flat) != 400 or Counter(flat) != Counter(tasks):
        raise ValueError("Warmstart task assignments must cover every task exactly once")
    if set(plan["anchors"]) != set(tasks):
        raise ValueError("Warmstart anchors must cover every task exactly once")
    for task, anchor in plan["anchors"].items():
        body = {key: value for key, value in anchor.items() if key != "anchor_sha256"}
        if anchor.get("anchor_sha256") != semantic_sha256(body):
            raise ValueError(f"Warmstart anchor checksum changed: {task}")
        if anchor.get("task_id") != task or anchor.get("test_index") != 0 or not anchor.get("is_canonical"):
            raise ValueError(f"Warmstart anchor is not first-test canonical: {task}")


def build_plan(profile: Path):
    from s4_circuit.ttft.s0_anchor import prepare_source

    root, cfg = settings(profile)
    runtime = direct.runtime_config(root, cfg, verify_weights=True)
    _, challenges, views, protocol_sha = direct.selected_views(root, cfg)
    task_ids = sorted(challenges)
    source_identity, anchors = prepare_source(root, cfg, views)
    cost_path = root / cfg["cost_reference_summary"]
    costs = reference_costs(direct.read_json(cost_path), task_ids, cfg["learning_rates"])
    lanes, totals = assign_tasks(costs, cfg["world_size"])
    plan = {
        "format": "s0_warmstart_ttft_plan_v1", "mode": MODE,
        "settings": cfg, "code_sha256": code_identity(root),
        "base_checkpoint": direct.relative_path(root, runtime.base.hf_checkpoint),
        "tokenizer": direct.relative_path(root, runtime.base.tokenizer),
        "checkpoint_identity": runtime.checkpoint_identity,
        "source_identity": source_identity, "anchors": anchors,
        "task_ids": task_ids, "task_assignments": lanes,
        "assignment_algorithm": "greedy_lpt_direct_seven_lr_seconds_v1",
        "estimated_lane_seconds": totals,
        "cost_reference": {"path": cfg["cost_reference_summary"], "sha256": sha256_file(cost_path)},
        "task_count": len(task_ids), "test_input_count": sum(len(v["test"]) for v in challenges.values()),
        "query_count": len(views),
        "query_manifest_sha256": semantic_sha256([view.manifest_record() for view in views]),
        "dataset_identity": {"challenge_count": len(challenges), "challenge_sha256": semantic_sha256(challenges)},
        "frozen_decode_protocol_sha256": protocol_sha,
        "frozen_protocol_version": direct.read_json(Path(runtime.evaluation.protocol_config))["protocol_version"],
    }
    validate_plan_shape(plan)
    plan = signed(plan)
    destination = root / cfg["output_root"] / "plan.json"
    if destination.exists():
        if read_signed(destination) != plan:
            raise ValueError("Existing warmstart plan differs; use a new output_root")
    else:
        atomic_json(destination, plan)
    return root, cfg, plan


def load_plan(profile: Path):
    root, cfg = settings(profile)
    plan = read_signed(root / cfg["output_root"] / "plan.json")
    if plan["settings"] != cfg or plan["code_sha256"] != code_identity(root):
        raise ValueError("Warmstart plan/config/code identity changed")
    actual = direct.runtime_config(root, cfg, verify_weights=True).checkpoint_identity
    if actual != plan["checkpoint_identity"]:
        raise ValueError("Warmstart target checkpoint identity changed")
    validate_plan_shape(plan)
    return root, cfg, plan


def initializer_paths(output: Path, task: str):
    return output / "initializers" / f"{task}.pt", output / "initializers" / f"{task}.json"


def initializer_meta(output: Path, plan: dict, task: str, *, verify_file: bool = True):
    tensor_path, meta_path = initializer_paths(output, task)
    if not tensor_path.exists() and not meta_path.exists():
        return None
    if not tensor_path.is_file() or not meta_path.is_file():
        raise ValueError(f"Incomplete warmstart initializer cache: {task}")
    meta = read_signed(meta_path)
    if (meta.get("plan_sha256") != plan["sha256"] or meta.get("task_id") != task
            or meta.get("anchor_sha256") != plan["anchors"][task]["anchor_sha256"]):
        raise ValueError(f"Warmstart initializer identity changed: {task}")
    if tensor_path.stat().st_size != meta["file_bytes"]:
        raise ValueError(f"Warmstart initializer size changed: {task}")
    if verify_file and sha256_file(tensor_path) != meta["file_sha256"]:
        raise ValueError(f"Warmstart initializer hash changed: {task}")
    return meta


def cache_initializer(output: Path, plan: dict, task: str, state: dict):
    tensor_path, meta_path = initializer_paths(output, task)
    if tensor_path.exists() or meta_path.exists():
        raise ValueError(f"Refusing to replace existing warmstart initializer: {task}")
    if not state or "scale" not in state:
        raise ValueError("Empty generated initializer")
    for key, tensor in state.items():
        if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                or tensor.requires_grad or tensor.is_inference()
                or not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all())):
            raise ValueError(f"Initializer must contain normal detached finite CPU tensors: {key}")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_name(tensor_path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, tensor_path)
    finally:
        temporary.unlink(missing_ok=True)
    meta = signed({
        "format": "s0_warmstart_initializer_v1", "plan_sha256": plan["sha256"],
        "task_id": task, "anchor_sha256": plan["anchors"][task]["anchor_sha256"],
        "file_sha256": sha256_file(tensor_path), "file_bytes": tensor_path.stat().st_size,
        "tensor_shapes": {key: list(value.shape) for key, value in state.items()},
        "tensor_dtypes": {key: str(value.dtype) for key, value in state.items()},
    })
    atomic_json(meta_path, meta)
    return meta


def ensure_initializers(root, cfg, plan, task_ids, views, device):
    from s4_circuit.ttft.s0_anchor import build_source, generate_anchor

    output = root / cfg["output_root"]
    missing = [task for task in task_ids if initializer_meta(output, plan, task) is None]
    if not missing:
        return
    by_index = {view.query_index: view for view in views}
    system = tokenizer = source_model = None
    try:
        system, tokenizer, source_model = build_source(root, cfg, plan["source_identity"], device)
        for task in missing:
            anchor = plan["anchors"][task]
            state = generate_anchor(system, tokenizer, source_model, by_index[anchor["query_index"]], anchor, device)
            meta = cache_initializer(output, plan, task, state)
            emit("s0_initializer_complete", task_id=task, file_sha256=meta["file_sha256"])
            del state
    finally:
        if system is not None:
            system.close()
        del system, tokenizer, source_model
        gc.collect()
        torch.cuda.empty_cache()


def result_identity(plan, cfg, task, lr, rows_sha, meta):
    return semantic_sha256({
        "plan_sha256": plan["sha256"], "task_id": task, "lr": lr,
        "support_rows_sha256": rows_sha, "lora_target_modules": cfg["lora_target_modules"],
        "initializer_file_sha256": meta["file_sha256"], "initializer_meta_sha256": meta["sha256"],
        "anchor_sha256": meta["anchor_sha256"],
    })


def validate_result(value, plan, cfg, task, lr, task_views, meta):
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value.get("sha256") != semantic_sha256(body):
        raise ValueError(f"Warmstart result checksum changed: {task}/{lr}")
    if (value.get("plan_sha256") != plan["sha256"] or value.get("task_id") != task
            or value.get("lr") != lr or value.get("status") != "complete" or value.get("mode") != MODE):
        raise ValueError(f"Warmstart result identity/status changed: {task}/{lr}")
    if value.get("identity") != result_identity(plan, cfg, task, lr, value["support_rows_sha256"], meta):
        raise ValueError(f"Warmstart result initializer/support identity changed: {task}/{lr}")
    if (value.get("initializer_file_sha256") != meta["file_sha256"]
            or value.get("initializer_meta_sha256") != meta["sha256"]
            or value.get("anchor_sha256") != meta["anchor_sha256"]):
        raise ValueError(f"Warmstart result initializer fields changed: {task}/{lr}")
    updates = 128 if lr > 0 else 0
    trace = value.get("update_trace_this_attempt", [])
    delta = value.get("update_delta_l2", float("nan"))
    if (value.get("updates") != updates or len(trace) != updates or not math.isfinite(delta)
            or (delta <= 0 if lr > 0 else delta != 0)):
        raise ValueError(f"Warmstart result update budget/delta invalid: {task}/{lr}")
    for index, item in enumerate(trace):
        expected_lr = lr * direct.cosine_schedule(index, 128, cfg["warmup_ratio"])
        if (item["step"] != index + 1 or item["steps"] != 128 or item["task_id"] != task
                or not math.isclose(item["lr"], expected_lr, rel_tol=1e-12, abs_tol=1e-20)):
            raise ValueError(f"Warmstart update trace changed: {task}/{lr}")
    predictions = value.get("predictions", [])
    expected = {view.query_index: view.manifest_record() for view in task_views}
    actual = {row["query_index"]: row for row in predictions}
    if len(predictions) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"Warmstart prediction coverage incomplete: {task}/{lr}")
    for query, manifest in expected.items():
        if not actual[query].get("finished") or any(actual[query].get(k) != v for k, v in manifest.items()):
            raise ValueError(f"Warmstart prediction unfinished or view changed: {task}/{lr}/{query}")


def worker(profile: Path, rank: int):
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from s2_hypernet.nvarc_pg.decode import constrained_greedy_batch, verify_token_contract
    from s2_hypernet.nvarc_pg.runtime import (
        _configure_runtime_from_environment, load_frozen_qwen, load_tokenizer, set_reproducible_seed,
    )

    if not 0 <= rank < direct.task_workers():
        raise ValueError("Warmstart worker rank outside the task plan")
    runtime_settings = _configure_runtime_from_environment()
    if not runtime_settings or not runtime_settings.get("deterministic_algorithms"):
        raise ValueError("Warmstart workers require the strict deterministic CUDA runtime")
    root, cfg, plan = load_plan(profile)
    runtime, challenges, views, _ = direct.selected_views(root, cfg)
    if semantic_sha256([view.manifest_record() for view in views]) != plan["query_manifest_sha256"]:
        raise ValueError("Warmstart query manifest changed")
    task_ids = plan["task_assignments"][rank]
    task_views = {task: tuple(view for view in views if view.task_id == task) for task in task_ids}
    output = root / cfg["output_root"]
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    with exclusive_lock(output / "locks" / f"worker-{rank}.lock"):
        set_reproducible_seed(42)
        ensure_initializers(root, cfg, plan, task_ids, views, device)
        # Source construction consumes RNG; reproduce the direct baseline's stock PEFT initialization.
        set_reproducible_seed(42)
        base_model = load_frozen_qwen(runtime, device=device, for_training=False)
        base_model.config.use_cache = False
        if hasattr(base_model, "gradient_checkpointing_disable"):
            base_model.gradient_checkpointing_disable()
        model = get_peft_model(base_model, LoraConfig(
            r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            bias="none", target_modules=list(cfg["lora_target_modules"]),
            use_rslora=cfg["lora_use_rslora"], task_type="CAUSAL_LM",
        ))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        stock = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
        tokenizer = load_tokenizer(runtime)
        token_contract = verify_token_contract(tokenizer, model)
        try:
            for task in task_ids:
                meta = initializer_meta(output, plan, task)
                if meta is None:
                    raise ValueError(f"Missing committed initializer: {task}")
                state_path, _ = initializer_paths(output, task)
                state = torch.load(state_path, map_location="cpu", weights_only=True)
                rows = build_support_rows(task, challenges[task], tokenizer, root)
                if len(rows) != 128:
                    raise ValueError(f"Warmstart task must have 128 support rows: {task}")
                rows_sha = rows_identity(rows)
                tokenized = tuple(row.tokenize(tokenizer) for row in rows)
                for lr in cfg["learning_rates"]:
                    label = direct.lr_label(lr)
                    path = output / "results" / label / f"{task}.json"
                    identity = result_identity(plan, cfg, task, lr, rows_sha, meta)
                    if path.exists():
                        existing = direct.read_json(path)
                        validate_result(existing, plan, cfg, task, lr, task_views[task], meta)
                        if existing["identity"] != identity:
                            raise ValueError(f"Support rows changed on resume: {task}/{label}")
                        emit("s0_ttft_puzzle_reused", rank=rank, task_id=task, lr=lr)
                        continue
                    direct.reset_lora(model, stock)
                    initialization = initialize_peft_from_generated(model, state, gate=cfg["warmstart"]["gate"])
                    warmstart = {key: value.detach().cpu().clone() for key, value in get_peft_model_state_dict(model).items()}
                    model.train()
                    model.config.use_cache = False
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    began = time.monotonic()
                    trace = []
                    if lr > 0:
                        optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                                      weight_decay=cfg["weight_decay"], foreach=False)
                        scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer, lambda step: direct.cosine_schedule(step, 128, cfg["warmup_ratio"]))
                        order = torch.randperm(128, generator=torch.Generator().manual_seed(42)).tolist()
                        for position in range(128):
                            optimizer.zero_grad(set_to_none=True)
                            metrics = direct.adapt_row(model, tokenized[order[position]], device=device)
                            norm = torch.nn.utils.clip_grad_norm_(trainable, cfg["max_grad_norm"], error_if_nonfinite=True)
                            active_lr = optimizer.param_groups[0]["lr"]
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            report = {**metrics, "step": position + 1, "steps": 128, "lr": active_lr,
                                      "grad_norm": float(norm.detach()), "seconds": time.monotonic() - began,
                                      "task_id": task}
                            trace.append(report)
                            emit("s0_ttft_update", rank=rank, **report)
                        del optimizer, scheduler
                    model.eval()
                    model.config.use_cache = True
                    predictions = []
                    for test_index in range(len(challenges[task]["test"])):
                        batch = sorted((v for v in task_views[task] if v.test_index == test_index), key=lambda v: v.query_index)
                        decoded = constrained_greedy_batch(
                            type("System", (), {"base_model": model})(), tokenizer, [view.prompt for view in batch],
                            token_contract=token_contract, max_new_tokens=cfg["max_new_tokens"], device=device,
                            condition_prompts=None, precomputed_state=None,
                        )
                        predictions.extend({**view.manifest_record(), **prediction, "s0_warmstart_ttft_lora": True}
                                           for view, prediction in zip(batch, decoded, strict=True))
                    torch.cuda.synchronize(device)
                    delta = direct.lora_delta_l2(model, warmstart)
                    value = signed({
                        "format": "s0_warmstart_ttft_puzzle_v1", "status": "complete", "mode": MODE,
                        "diagnostic_only": False, "identity": identity, "plan_sha256": plan["sha256"],
                        "task_id": task, "lr": lr, "worker_rank": rank, "support_rows_sha256": rows_sha,
                        "initializer_file_sha256": meta["file_sha256"], "initializer_meta_sha256": meta["sha256"],
                        "anchor_sha256": meta["anchor_sha256"], "initialization_report": initialization,
                        "updates": 128 if lr > 0 else 0, "update_trace_this_attempt": trace,
                        "update_delta_l2": delta, "lora_delta_l2": delta, "delta_reference": "warmstart_adapter",
                        "lora_trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
                        "runtime_settings": runtime_settings, "seconds_this_attempt": time.monotonic() - began,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                        "predictions": predictions,
                    })
                    validate_result(value, plan, cfg, task, lr, task_views[task], meta)
                    atomic_json(path, value)
                    emit("s0_ttft_puzzle_complete", rank=rank, task_id=task, lr=lr, seconds=value["seconds_this_attempt"])
                    del warmstart
                del state, rows, tokenized
        finally:
            del model, base_model, stock, trainable
            gc.collect()
            torch.cuda.empty_cache()


def all_results_exist(output, plan, cfg):
    return all((output / "results" / direct.lr_label(lr) / f"{task}.json").is_file()
               for lr in cfg["learning_rates"] for task in plan["task_ids"])


@contextmanager
def collection_lock(path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def collect(profile: Path) -> bool:
    root, cfg, plan = load_plan(profile)
    output = root / cfg["output_root"]
    if not all_results_exist(output, plan, cfg):
        emit("s0_ttft_collection_pending", reason="missing_task_lr_results")
        return False
    with collection_lock(output / "locks" / "collect.lock"):
        summary_path = output / "summary.json"
        if summary_path.exists():
            previous = read_signed(summary_path)
            if previous.get("plan_sha256") != plan["sha256"] or previous.get("status") != "complete":
                raise ValueError("Existing warmstart summary identity/status changed")
            return True
        from nvarc_qwen3.arc_eval_protocol import load_solutions, validate_arc_agi1_identity
        from s2_hypernet.nvarc_pg.eval_aggregation import group_query_predictions
        from s2_hypernet.nvarc_pg.eval_scoring import score_grouped_predictions

        runtime, challenges, views, _ = direct.selected_views(root, cfg)
        solutions = load_solutions(runtime.evaluation.solutions, challenges)
        validate_arc_agi1_identity(challenges, solutions)
        if (sorted(challenges) != plan["task_ids"] or len(views) != 7123
                or sum(len(task["test"]) for task in challenges.values()) != 419
                or semantic_sha256([view.manifest_record() for view in views]) != plan["query_manifest_sha256"]):
            raise ValueError("Warmstart final coverage/query identity changed")
        task_views = {task: tuple(view for view in views if view.task_id == task) for task in plan["task_ids"]}
        metas = {task: initializer_meta(output, plan, task, verify_file=False) for task in plan["task_ids"]}
        if any(meta is None for meta in metas.values()):
            raise ValueError("Warmstart collection requires all committed initializer identities")
        summary = {
            "format": "s0_warmstart_ttft_summary_v1", "status": "complete", "mode": MODE,
            "diagnostic_only": False, "plan_sha256": plan["sha256"],
            "task_count": 400, "test_input_count": 419, "query_count": 7123,
            "base_checkpoint": plan["base_checkpoint"], "source_identity": plan["source_identity"],
            "delta_reference": "warmstart_adapter", "metrics": {}, "results": [],
        }
        for lr in cfg["learning_rates"]:
            label = direct.lr_label(lr)
            result_dir = output / "results" / label
            if {path.stem for path in result_dir.glob("*.json")} != set(plan["task_ids"]):
                raise ValueError(f"Warmstart result task set changed: {label}")
            records = []
            for task in plan["task_ids"]:
                value = direct.read_json(result_dir / f"{task}.json")
                validate_result(value, plan, cfg, task, lr, task_views[task], metas[task])
                records.extend(value["predictions"])
                summary["results"].append({key: value[key] for key in (
                    "task_id", "lr", "updates", "update_delta_l2", "lora_delta_l2",
                    "peak_allocated_gib", "seconds_this_attempt")})
            if len(records) != 7123 or {r["query_index"] for r in records} != set(range(7123)):
                raise ValueError(f"Warmstart full unique-query coverage changed: {label}")
            records.sort(key=lambda value: value["query_index"])
            metrics, submission = score_grouped_predictions(challenges, solutions, group_query_predictions(views, records))
            metrics["protocol_version"] = plan["frozen_protocol_version"]
            summary["metrics"][label] = metrics
            atomic_json(output / label / "metrics.json", metrics)
            atomic_json(output / label / "submission.json", submission)
        atomic_json(summary_path, signed(summary))
        emit("s0_ttft_complete", summary=str(summary_path))
        return True


def run_shard(profile: Path, shard: int):
    root, cfg, plan = load_plan(profile)
    local_width = int(os.environ.get("ARC_TTFT_LOCAL_WORKERS", str(cfg["world_size"])))
    if local_width <= 0 or shard < 0 or shard * local_width >= cfg["world_size"]:
        raise ValueError("Warmstart shard is outside the worker plan")
    begin = shard * local_width
    end = min(begin + local_width, cfg["world_size"])
    devices = direct.visible_devices()[:end - begin]
    if len(devices) != end - begin:
        raise ValueError("Shard worker count exceeds the available devices")
    children = []
    began = time.monotonic()
    try:
        for local_rank, device in enumerate(devices):
            rank = begin + local_rank
            children.append(subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__)), "--profile", str(profile.resolve()), "--worker", str(rank)],
                cwd=root, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": device, "OMP_NUM_THREADS": "4", "NUMBA_NUM_THREADS": "4"},
            ))
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("Warmstart worker failed; stopping this shard, preserving committed results")
            time.sleep(2)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("Warmstart worker failed")
        body = signed({
            "format": "s0_warmstart_ttft_shard_v1", "status": "shard_complete",
            "plan_sha256": plan["sha256"], "shard": shard,
            "worker_ranks": list(range(begin, end)),
            "exit_codes": [child.returncode for child in children], "seconds": time.monotonic() - began,
        })
        atomic_json(root / cfg["output_root"] / "shards" / f"{shard}.json", body)
        complete = collect(profile)
        print(json.dumps({"status": "complete" if complete else "shard_complete", "shard": shard,
                          "plan_sha256": plan["sha256"], "output_root": cfg["output_root"]}), flush=True)
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


def main():
    parser = argparse.ArgumentParser(description="S0-initialized direct-LoRA TTFT on SFT step65")
    parser.add_argument("--profile", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--worker", type=int)
    mode.add_argument("--shard", type=int)
    mode.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        _, cfg, plan = build_plan(args.profile)
        print(json.dumps({"status": "prepared", "plan_sha256": plan["sha256"], "output_root": cfg["output_root"],
                          "task_count": 400, "test_input_count": 419, "query_count": 7123}), flush=True)
    elif args.worker is not None:
        worker(args.profile, args.worker)
    elif args.shard is not None:
        run_shard(args.profile, args.shard)
    else:
        print(json.dumps({"status": "complete" if collect(args.profile) else "pending"}), flush=True)


if __name__ == "__main__":
    main()

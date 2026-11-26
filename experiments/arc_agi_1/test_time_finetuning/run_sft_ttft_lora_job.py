from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for folder in ("s4_circuit", "s3_circuit", "s2_hypernet", "s1_lora_sft"):
    sys.path.insert(0, str(REPO_ROOT / folder / "src"))

import torch

from s4_circuit.contracts import semantic_sha256, sha256_file
from s4_circuit.ttft.data import build_support_rows, rows_identity
from s4_circuit.ttft.state import atomic_json, emit, exclusive_lock


def task_workers() -> int:
    workers = int(os.environ.get("ARC_TTFT_WORKERS", "1"))
    if not 0 < workers <= 400:
        raise ValueError("Task worker count must be between one and the task count")
    return workers


def visible_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = [str(i) for i in range(torch.cuda.device_count())] if raw is None else raw.split(",")
    if not devices or any(not d.strip() for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("A non-empty list of unique visible devices is required")
    return devices

VIEWS_PER_TEST = 17


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def lr_label(lr: float) -> str:
    if lr == 0:
        return "control"
    return f"lr{lr:.0e}".replace("-0", "-")


def cosine_schedule(step: int, total: int, warmup_ratio: float) -> float:
    warmup = math.ceil(total * warmup_ratio)
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def settings(profile: Path) -> tuple[Path, dict]:
    root = Path(os.environ.get("HC_ROOT", REPO_ROOT)).resolve()
    value = read_json(profile)
    value["world_size"] = task_workers()
    if value.get("format") not in {"s4_nvarc_sft_ttft_v1", "s4_nvarc_sft_ttft_v2", "s4_nvarc_sft_ttft_v3"}:
        raise ValueError("SFT TTFT protocol format changed; create an explicitly reviewed new version")
    expected = {
        "selection_views": 16,
        "max_new_tokens": 1024,
        "lora_rank": 256,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "lora_use_rslora": True,
        "weight_decay": 0.0,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0,
    }
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise ValueError("SFT TTFT protocol changed; create an explicitly reviewed new version")
    if value["format"] == "s4_nvarc_sft_ttft_v1":
        if value.get("probe_rows_per_task") != 2:
            raise ValueError("The v1 SFT TTFT probe must use two support rows per puzzle")
    else:
        if value.get("task_selection") != "all" or value.get("rows_per_task") != 128:
            raise ValueError("The v2 SFT TTFT final must use all tasks and 128 support rows per puzzle")
    if tuple(value["learning_rates"]) != (0.0, 1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5):
        raise ValueError("SFT TTFT LR matrix must match the S4 TTFT sweep")
    target_modules = tuple(value.get("lora_target_modules", ()))
    expected_targets = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head",
    )
    if target_modules != expected_targets:
        raise ValueError("SFT TTFT LoRA target modules changed; create a reviewed protocol version")
    return root, value


def code_identity(root: Path) -> str:
    paths = [
        Path(__file__).resolve(),
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/decode.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_manifest.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_aggregation.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/eval_scoring.py",
        root / "s2_hypernet/src/s2_hypernet/nvarc_pg/objective.py",
        root / "s1_lora_sft/src/nvarc_qwen3/arc_eval_protocol.py",
        root / "s4_circuit/src/s4_circuit/ttft/data.py",
        root / "s4_circuit/src/s4_circuit/ttft/direct_runtime.py",
    ]
    return semantic_sha256({str(path.relative_to(root)): sha256_file(path) for path in paths})


def source_config(root: Path, cfg: dict):
    from s4_circuit.config import load_config

    return load_config(root / cfg["source_config"])


def runtime_config(root: Path, cfg: dict, *, verify_weights: bool = False):
    if cfg["format"] == "s4_nvarc_sft_ttft_v3":
        from s4_circuit.ttft.direct_runtime import load_direct_runtime
        return load_direct_runtime(root, cfg, verify_weights=verify_weights)
    return source_config(root, cfg).load_s2_config()


def relative_path(root: Path, value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return str(path.relative_to(root))
    except ValueError as error:
        raise ValueError(f"Path is outside HC_ROOT: {path}") from error


def selected_views(root: Path, cfg: dict):
    from nvarc_qwen3.arc_eval_protocol import load_challenges
    from s2_hypernet.nvarc_pg.eval_manifest import build_query_views
    from s2_hypernet.nvarc_pg.eval_protocol import (
        load_profile_eval_protocol,
        validate_profile_dataset_identity,
    )
    from s2_hypernet.nvarc_pg.runtime import load_tokenizer

    s2 = runtime_config(root, cfg)
    protocol, protocol_sha256 = load_profile_eval_protocol(s2.evaluation)
    challenges_all = load_challenges(s2.evaluation.challenges)
    validate_profile_dataset_identity(s2.evaluation.protocol_kind, challenges_all)
    if cfg["format"] != "s4_nvarc_sft_ttft_v1":
        task_ids = sorted(challenges_all)
        if len(task_ids) != 400:
            raise ValueError("SFT TTFT final requires all 400 challenge tasks")
        challenges = challenges_all
    else:
        reference = read_json(root / cfg["reference_probe_plan"])
        task_ids = sorted(reference["task_ids"])
        if reference.get("mode") != "probe" or len(task_ids) != 8:
            raise ValueError("SFT TTFT baseline must use the completed eight-task TTFT probe")
        if reference["settings"].get("probe_rows_per_task") != cfg["probe_rows_per_task"]:
            raise ValueError("SFT TTFT baseline row budget differs from the S4 probe")
        challenges = {task_id: challenges_all[task_id] for task_id in task_ids}
    tokenizer = load_tokenizer(s2)
    local_views = build_query_views(challenges, config=protocol, tokenizer=tokenizer)
    expected_count = VIEWS_PER_TEST * sum(len(task["test"]) for task in challenges.values())
    if len(local_views) != expected_count:
        raise ValueError(f"Unexpected SFT TTFT query-view count: {len(local_views)}")

    global_test_offsets: dict[str, int] = {}
    test_offset = 0
    for task_id in sorted(challenges_all):
        global_test_offsets[task_id] = test_offset
        test_offset += len(challenges_all[task_id]["test"])
    adjusted = []
    for view in local_views:
        global_test_index = global_test_offsets[view.task_id] + view.test_index
        position = view.query_index % VIEWS_PER_TEST
        adjusted.append(
            replace(
                view,
                query_index=global_test_index * VIEWS_PER_TEST + position,
                global_test_index=global_test_index,
            )
        )
    return s2, challenges_all, tuple(adjusted), protocol_sha256


def build_plan(profile: Path) -> tuple[Path, dict, dict]:
    root, cfg = settings(profile)
    output = root / cfg["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    s2, challenges_all, views, protocol_sha256 = selected_views(root, cfg)
    reference_path = root / cfg["reference_probe_plan"] if cfg.get("reference_probe_plan") else None
    reference = read_json(reference_path) if reference_path is not None else None
    task_ids = sorted(reference["task_ids"]) if reference is not None else sorted(challenges_all)
    plan = {
        "format": "s4_sft_ttft_lora_plan_v2" if cfg["format"] != "s4_nvarc_sft_ttft_v1" else "s4_sft_ttft_lora_plan_v1",
        "settings": cfg,
        "code_sha256": code_identity(root),
        "base_checkpoint": relative_path(root, s2.base.hf_checkpoint),
        "tokenizer": relative_path(root, s2.base.tokenizer),
        "task_ids": task_ids,
        "query_count": len(views),
        "query_manifest_sha256": semantic_sha256([view.manifest_record() for view in views]),
        "frozen_decode_protocol_sha256": protocol_sha256,
        "dataset_identity": {
            "challenge_count": len(challenges_all),
            "challenge_sha256": semantic_sha256(challenges_all),
        },
    }
    if cfg["format"] == "s4_nvarc_sft_ttft_v3":
        plan["checkpoint_identity"] = runtime_config(root, cfg, verify_weights=True).checkpoint_identity
    else:
        plan["source_config"] = cfg["source_config"]
        plan["source_config_sha256"] = sha256_file(root / cfg["source_config"])
    if reference_path is not None:
        plan["reference_probe_plan"] = cfg["reference_probe_plan"]
        plan["reference_probe_plan_sha256"] = sha256_file(reference_path)
    plan["sha256"] = semantic_sha256(plan)
    destination = output / "plan.json"
    if destination.exists():
        existing = read_json(destination)
        checksum = existing.pop("sha256", None)
        expected = {key: value for key, value in plan.items() if key != "sha256"}
        if checksum != semantic_sha256(existing) or existing != expected:
            raise ValueError("Existing SFT TTFT plan differs; use a new output_root")
    else:
        atomic_json(destination, plan)
    return root, cfg, plan


def load_plan(profile: Path) -> tuple[Path, dict, dict]:
    root, cfg = settings(profile)
    plan = read_json(root / cfg["output_root"] / "plan.json")
    checksum = plan.pop("sha256", None)
    if checksum != semantic_sha256(plan):
        raise ValueError("SFT TTFT plan checksum changed")
    if plan["settings"] != cfg or plan["code_sha256"] != code_identity(root):
        raise ValueError("SFT TTFT plan/code identity changed")
    if cfg["format"] == "s4_nvarc_sft_ttft_v3" and runtime_config(root, cfg, verify_weights=True).checkpoint_identity != plan["checkpoint_identity"]:
        raise ValueError("Direct TTFT checkpoint identity changed")
    plan["sha256"] = checksum
    return root, cfg, plan


def event_tensors(row, event_index: int, *, device: torch.device):
    event = row.events[int(event_index)]
    input_ids = torch.tensor(row.input_ids[: event.response_end], device=device, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor(event.labels, device=device, dtype=torch.long).unsqueeze(0)
    return input_ids, attention_mask, labels


def adapt_row(model, row, *, device: torch.device) -> dict:
    from s2_hypernet.nvarc_pg.objective import causal_loss_sum

    total_tokens = row.supervised_token_count
    if total_tokens <= 0:
        raise ValueError("Empty TTFT supervision")
    total_loss = 0.0
    for event in row.events:
        input_ids, attention_mask, labels = event_tensors(row, event.event_index, device=device)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        loss_sum, count = causal_loss_sum(outputs.logits, labels)
        if int(count.detach()) != event.response_token_count:
            raise ValueError("SFT TTFT assistant mask/token count changed")
        scaled = loss_sum / total_tokens
        if not bool(torch.isfinite(scaled)):
            raise FloatingPointError("Non-finite SFT TTFT loss")
        scaled.backward()
        total_loss += float(scaled.detach())
    return {"loss": total_loss, "tokens": total_tokens, "events": len(row.events)}


def reset_lora(model, initial_state: dict[str, torch.Tensor]) -> None:
    from peft import set_peft_model_state_dict

    set_peft_model_state_dict(
        model,
        {key: value.clone() for key, value in initial_state.items()},
        adapter_name="default",
    )
    model.zero_grad(set_to_none=True)


def lora_delta_l2(model, initial_state: dict[str, torch.Tensor]) -> float:
    from peft import get_peft_model_state_dict

    current = get_peft_model_state_dict(model, adapter_name="default")
    square = 0.0
    for key, before in initial_state.items():
        after = current[key].detach().float().cpu()
        square += float((after - before.float()).square().sum())
    return square**0.5


def worker(profile: Path, rank: int) -> None:
    from nvarc_qwen3.arc_eval_protocol import load_challenges
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from s2_hypernet.nvarc_pg.decode import constrained_greedy_batch, verify_token_contract
    from s2_hypernet.nvarc_pg.runtime import load_frozen_qwen, load_tokenizer, set_reproducible_seed

    root, cfg, plan = load_plan(profile)
    if not 0 <= rank < cfg["world_size"]:
        raise ValueError("Worker rank exceeds the configured task count")
    task_ids = plan["task_ids"][rank::cfg["world_size"]]
    s2 = runtime_config(root, cfg)
    challenges = load_challenges(s2.evaluation.challenges)
    _, _, views, _ = selected_views(root, cfg)
    task_views = {task_id: tuple(view for view in views if view.task_id == task_id) for task_id in task_ids}
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    set_reproducible_seed(42)

    base_model = load_frozen_qwen(s2, device=device, for_training=False)
    base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_disable"):
        base_model.gradient_checkpointing_disable()
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        target_modules=list(cfg["lora_target_modules"]),
        use_rslora=cfg["lora_use_rslora"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("SFT TTFT LoRA has no trainable parameters")
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in get_peft_model_state_dict(model, adapter_name="default").items()
    }
    tokenizer = load_tokenizer(s2)
    token_contract = verify_token_contract(tokenizer, model)
    output = root / cfg["output_root"]
    try:
        with exclusive_lock(output / "locks" / f"worker-{rank}.lock"):
            for task_id in task_ids:
                views_for_task = task_views[task_id]
                expected_count = VIEWS_PER_TEST * len(challenges[task_id]["test"])
                if len(views_for_task) != expected_count:
                    raise ValueError(f"Unexpected view count for {task_id}: {len(views_for_task)}")
                rows = build_support_rows(task_id, challenges[task_id], tokenizer, root)
                row_limit = int(cfg.get("rows_per_task", cfg.get("probe_rows_per_task", 0)))
                if row_limit <= 0 or len(rows) < row_limit:
                    raise ValueError(f"SFT TTFT support rows incomplete for {task_id}: {len(rows)} < {row_limit}")
                rows = rows[:row_limit]
                tokenized_rows = tuple(row.tokenize(tokenizer) for row in rows)
                for lr in cfg["learning_rates"]:
                    label = lr_label(lr)
                    identity = semantic_sha256({
                        "plan": plan["sha256"],
                        "task_id": task_id,
                        "lr": lr,
                        "rows": rows_identity(rows),
                        "lora": cfg["lora_target_modules"],
                    })
                    result_path = output / "results" / label / f"{task_id}.json"
                    if result_path.is_file():
                        existing = read_json(result_path)
                        checksum = existing.pop("sha256", None)
                        if checksum != semantic_sha256(existing) or existing.get("identity") != identity:
                            raise ValueError(f"Completed SFT TTFT identity changed: {result_path}")
                        if existing.get("status") == "complete" and len(existing.get("predictions", [])) == expected_count:
                            emit("sft_ttft_puzzle_reused", rank=rank, task_id=task_id, lr=lr)
                            continue
                    reset_lora(model, initial_state)
                    model.train()
                    model.config.use_cache = False
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    began = time.monotonic()
                    trace = []
                    if lr > 0:
                        optimizer = torch.optim.AdamW(
                            trainable,
                            lr=lr,
                            betas=(0.9, 0.999),
                            eps=1e-8,
                            weight_decay=cfg["weight_decay"],
                            foreach=False,
                        )
                        scheduler = torch.optim.lr_scheduler.LambdaLR(
                            optimizer,
                            lambda step: cosine_schedule(step, len(tokenized_rows), cfg["warmup_ratio"]),
                        )
                        order = torch.randperm(
                            len(tokenized_rows),
                            generator=torch.Generator().manual_seed(42),
                        ).tolist()
                        for position in range(len(tokenized_rows)):
                            optimizer.zero_grad(set_to_none=True)
                            row = tokenized_rows[order[position]]
                            metrics = adapt_row(model, row, device=device)
                            norm = torch.nn.utils.clip_grad_norm_(
                                trainable,
                                cfg["max_grad_norm"],
                                error_if_nonfinite=True,
                            )
                            active_lr = optimizer.param_groups[0]["lr"]
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)
                            report = {
                                **metrics,
                                "step": position + 1,
                                "steps": len(tokenized_rows),
                                "lr": active_lr,
                                "grad_norm": float(norm.detach()),
                                "seconds": time.monotonic() - began,
                                "task_id": task_id,
                            }
                            trace.append(report)
                            emit("sft_ttft_update", **report)
                        del optimizer, scheduler
                    model.eval()
                    model.config.use_cache = True
                    predictions = []
                    for test_index in sorted({view.test_index for view in views_for_task}):
                        batch = [view for view in views_for_task if view.test_index == test_index]
                        batch.sort(key=lambda view: view.query_index)
                        decoded = constrained_greedy_batch(
                            type("System", (), {"base_model": model})(),
                            tokenizer,
                            [view.prompt for view in batch],
                            token_contract=token_contract,
                            max_new_tokens=cfg["max_new_tokens"],
                            device=device,
                            condition_prompts=None,
                            precomputed_state=None,
                        )
                        predictions.extend(
                            {
                                **view.manifest_record(),
                                **prediction,
                                "sft_ttft_lora": True,
                            }
                            for view, prediction in zip(batch, decoded, strict=True)
                        )
                    torch.cuda.synchronize(device)
                    body = {
                        "format": "s4_sft_ttft_lora_puzzle_v2" if cfg["format"] != "s4_nvarc_sft_ttft_v1" else "s4_sft_ttft_lora_puzzle_v1",
                        "status": "complete",
                        "identity": identity,
                        "plan_sha256": plan["sha256"],
                        "task_id": task_id,
                        "lr": lr,
                        "mode": "sft_lora_ttft",
                        "diagnostic_only": cfg["format"] == "s4_nvarc_sft_ttft_v1",
                        "updates": len(tokenized_rows) if lr > 0 else 0,
                        "lora_trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
                        "lora_delta_l2": lora_delta_l2(model, initial_state),
                        "seconds_this_attempt": time.monotonic() - began,
                        "update_trace_this_attempt": trace,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                        "predictions": predictions,
                    }
                    body["sha256"] = semantic_sha256(body)
                    atomic_json(result_path, body)
                    emit("sft_ttft_puzzle_complete", rank=rank, task_id=task_id, lr=lr, seconds=body["seconds_this_attempt"])
    finally:
        del model, base_model
        torch.cuda.empty_cache()


def collect(profile: Path) -> None:
    from nvarc_qwen3.arc_eval_protocol import load_challenges, load_solutions
    from s2_hypernet.nvarc_pg.eval_aggregation import group_query_predictions
    from s2_hypernet.nvarc_pg.eval_scoring import score_grouped_predictions

    root, cfg, plan = load_plan(profile)
    s2 = runtime_config(root, cfg)
    challenges_all = load_challenges(s2.evaluation.challenges)
    solutions_all = load_solutions(s2.evaluation.solutions, challenges_all)
    task_ids = plan["task_ids"]
    challenges = {task_id: challenges_all[task_id] for task_id in task_ids}
    solutions = {task_id: solutions_all[task_id] for task_id in task_ids}
    _, _, views, _ = selected_views(root, cfg)
    output = root / cfg["output_root"]
    diagnostic_only = cfg["format"] == "s4_nvarc_sft_ttft_v1"
    if not diagnostic_only and (task_ids != sorted(challenges_all) or len(task_ids) != 400 or len(views) != 7123):
        raise ValueError("SFT TTFT final coverage must contain all 400 tasks and 7,123 query views")
    summary = {
        "format": "s4_sft_ttft_lora_summary_v2" if not diagnostic_only else "s4_sft_ttft_lora_summary_v1",
        "status": "complete",
        "plan_sha256": plan["sha256"],
        "diagnostic_only": diagnostic_only,
        "task_count": len(task_ids),
        "mode": "sft_lora_ttft",
        "metrics": {},
        "results": [],
    }
    for lr in cfg["learning_rates"]:
        label = lr_label(lr)
        records = []
        for task_id in task_ids:
            value = read_json(output / "results" / label / f"{task_id}.json")
            checksum = value.pop("sha256", None)
            if checksum != semantic_sha256(value) or value.get("plan_sha256") != plan["sha256"]:
                raise ValueError(f"SFT TTFT result identity changed for {label}/{task_id}")
            if value.get("status") != "complete" or value.get("lr") != lr:
                raise ValueError(f"SFT TTFT result incomplete for {label}/{task_id}")
            if len(value.get("predictions", [])) != VIEWS_PER_TEST * len(challenges[task_id]["test"]):
                raise ValueError(f"SFT TTFT result coverage incomplete for {label}/{task_id}")
            records.extend(value["predictions"])
            summary["results"].append({
                key: value[key]
                for key in ("task_id", "lr", "updates", "lora_delta_l2", "peak_allocated_gib", "seconds_this_attempt")
            })
        records.sort(key=lambda value: int(value["query_index"]))
        grouped = group_query_predictions(views, records)
        metrics, submission = score_grouped_predictions(challenges, solutions, grouped)
        summary["metrics"][label] = metrics
        atomic_json(output / label / "metrics.json", metrics)
        atomic_json(output / label / "submission.json", submission)
    atomic_json(output / "summary.json", summary)
    emit("sft_ttft_complete", summary=str(output / "summary.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched direct-SFT LoRA TTFT sweep")
    parser.add_argument("--profile", default="s4_circuit/configs/ttft/nvarc_direct_sft_step65_ttft_final128_20261110_v1.json")
    parser.add_argument("--worker", type=int)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    profile = Path(args.profile)
    if args.worker is not None:
        worker(profile, args.worker)
        return
    if args.collect:
        collect(profile)
        return

    root, cfg, _ = build_plan(profile)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "profile": str(profile), "output_root": cfg["output_root"]}), flush=True)
        return
    devices = visible_devices()[:cfg["world_size"]]
    if len(devices) != cfg["world_size"]:
        raise ValueError("Worker count exceeds the available devices")
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"-{os.getpid()}"
    log = root / "logs/eval" / time.strftime("%Y-%m-%d") / f"sft-ttft-{stamp}"
    log.mkdir(parents=True, exist_ok=True)
    children: list[subprocess.Popen] = []
    result = {"status": "failed", "log_root": str(log), "profile": str(profile)}
    try:
        for rank, device in enumerate(devices):
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": device,
                "OMP_NUM_THREADS": "4",
                "NUMBA_NUM_THREADS": "4",
            }
            child = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__)), "--profile", str(profile), "--worker", str(rank)],
                cwd=root,
                env=env,
                start_new_session=True,
            )
            children.append(child)
        while any(child.poll() is None for child in children):
            failed = [rank for rank, child in enumerate(children) if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"SFT TTFT workers failed: {failed}; committed task/LR outputs retained")
            time.sleep(2)
        if any(child.returncode for child in children):
            raise RuntimeError("SFT TTFT worker failed")
        collect(profile)
        result["status"] = "complete"
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
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
        result["exit_codes"] = [child.returncode for child in children]
        (log / "job.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (log / "summary.md").write_text(
            f"# Direct SFT LoRA TTFT {result['status']}\n\n"
            + ("400-task, 128-row score-bearing full sweep with 17-view scoring.\n"
               if cfg["format"] != "s4_nvarc_sft_ttft_v1"
               else "Matched eight-task, two-row diagnostic sweep with 17-view scoring.\n"),
            encoding="utf-8",
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

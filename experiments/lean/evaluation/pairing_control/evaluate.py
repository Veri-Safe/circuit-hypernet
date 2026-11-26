#!/usr/bin/env python3
"""Evaluation-only formal-suite replay; there is no training/resume entrypoint."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import subprocess
import sys

from common import (ROOT, PACKAGE, LOG, TASKS, ATOMS, SUITES, load_manifest, sha, read,
                    write_new, verify_sources, config_for, check_file, selection_audit,
                    check_bank_metadata, verify_predictions, write_prediction_chunks, merge_family_results)

sys.path[:0] = [str(ROOT / "s5_lean_circuit/src"), str(ROOT / "s2_hypernet/src")]


def preflight(task, suite, manifest):
    """Read-only CPU preflight; no model construction or result directory writes."""
    verify_sources(manifest)
    from s5_lean_circuit.composition_loco import load_eval_bank

    import torch

    config = config_for(task, manifest)
    spec = manifest["tasks"][task]
    check_file(ROOT / spec["checkpoint_path"], spec["checkpoint_sha256"])
    check_file(ROOT / config["base"]["checkpoint_path"], config["base"]["checkpoint_sha256"])
    check_file(ROOT / manifest["tokenizer"]["path"], manifest["tokenizer"]["sha256"])
    # The checked weight file authenticates the embedded historical recipe ID.
    checkpoint = torch.load(ROOT / spec["checkpoint_path"], map_location="cpu", weights_only=False, mmap=True)
    archived_config_sha = checkpoint["config_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", archived_config_sha):
        raise ValueError("Invalid checkpoint recipe identity")
    del checkpoint
    audit = selection_audit(task, config, manifest, archived_config_sha)
    banks = {}
    for key, bank_spec, families in (
            ("validation", manifest["validation"], TASKS),
            ("atomic_validation", manifest["atomic_validation"], ATOMS),
            (suite, manifest["suites"][suite], ATOMS + TASKS)):
        path = ROOT / bank_spec["path"]
        check_file(path, bank_spec["sha256"])
        bank = load_eval_bank(path)
        check_bank_metadata(bank.metadata, bank_spec, families, manifest)
        if Counter(row.task_type for row in bank.records) != {t: 500 for t in families}:
            raise ValueError("bank record coverage disagrees with metadata")
        expected_base = bank.metadata.get("expected_source_base_identity", {})
        if any(expected_base.get(k) != config["base"][k]
               for k in ("checkpoint_sha256", "model_state_sha256", "config_sha256", "step")):
            raise ValueError("bank Circuit vectors are bound to a different Base")
        banks[key] = bank
    return config, audit, banks, archived_config_sha


def load_core(config, spec, device, archived_config_sha):
    """Build the same inference architecture and load the complete selected state.

    Unlike the training initializer, this does not need an atomic checkpoint:
    every trainable tensor is present in the selected stage-2 checkpoint.
    The frozen Base is independently checked by load_s5_base_model.
    """
    import torch
    from s5_lean_circuit.hypernet.config import S5BaseIdentity
    from s5_lean_circuit.hypernet.runtime import S5HyperNet, load_s5_base_model

    values = {key: config["base"][key] for key in (
        "checkpoint_sha256", "model_state_sha256", "config_sha256", "step")}
    values["checkpoint_path"] = str(ROOT / config["base"]["checkpoint_path"])
    values["architecture_config_path"] = str(ROOT / config["base"]["architecture_config_path"])
    identity = S5BaseIdentity(**values)
    base, _ = load_s5_base_model(identity.architecture_config_path, identity.checkpoint_path,
                                 device=device, dtype=torch.bfloat16, expected_identity=identity)
    core = S5HyperNet(base, candidate="A", seed=int(config["training"]["seed"]),
                      condition_source="circuit", fusion_mode="append",
                      raw_fusion_mode=config["model"].get("raw_fusion_mode", "query_cross"))
    core.validate_step0_initialization()
    payload = torch.load(ROOT / spec["checkpoint_path"], map_location="cpu", weights_only=False, mmap=True)
    if (payload.get("config_sha256") != archived_config_sha
            or payload.get("global_step") != spec["selected_step"]
            or payload.get("fold_id") != config["fold_id"]
            or payload.get("route") != "circuit_a"):
        raise ValueError("selected checkpoint envelope differs from frozen identity")
    state = payload["model_state_dict"]
    # Loading the complete state prevents random or atomic-init leftovers.
    expected_keys = set(core.trainable_state_dict())
    if set(state) != expected_keys:
        raise ValueError("selected checkpoint does not contain the complete trainable state")
    core.load_trainable_state_dict(state)
    core.to(device).eval()
    for parameter in core.parameters():
        parameter.requires_grad_(False)
    return core


def run(task, suite, attempt, manifest):
    import torch
    import torch.distributed as dist
    from s5_lean_circuit.circuit.model_io import load_tokenizer
    from s5_lean_circuit.composition_loco_eval import evaluate_bank, _result

    if not {"RANK", "LOCAL_RANK", "WORLD_SIZE"} <= os.environ.keys():
        raise ValueError("launch with torchrun and choose the process count for this deployment")
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    config, validation, banks, archived_config_sha = preflight(task, suite, manifest)
    directory = LOG / task / suite / attempt
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=False)
        write_new(directory / "started.json", {"task": task, "suite": suite, "attempt": attempt,
                  "time_utc": datetime.now(timezone.utc).isoformat(),
                  "manifest_sha256": sha(PACKAGE / "eval_manifest.json")})
        evidence = directory / "provenance"
        evidence.mkdir()
        originals = [ROOT / validation["selection_path"]]
        originals.extend(ROOT / config["output_dir"] / "evaluations" / f"visible_step{step:06d}.json"
                         for step in [0] + manifest["validation"]["eligible_steps"])
        for original in originals:
            with (evidence / original.name).open("xb") as stream:
                stream.write(original.read_bytes())
    dist.barrier()
    spec = manifest["tasks"][task]
    core = load_core(config, spec, device, archived_config_sha)
    tokenizer = load_tokenizer(ROOT / manifest["tokenizer"]["path"])
    options = dict(device=device, route="circuit_a", condition="matched", n_ctx=128,
                   max_new_tokens=50, gate_multiplier=float(config["model"]["gate_max"]))
    with torch.inference_mode():
        dev = evaluate_bank(core, tokenizer, banks["validation"], task_types=config["train_tasks"],
                            batch_size=manifest["inference"]["validation_batch_size"], **options)
    if rank == 0:
        write_new(directory / "validation.json", {"original_selection": validation, "reevaluation": dev})
        if dev["task_counts"] != validation["task_counts"]:
            raise ValueError("selected-checkpoint dev replay differs; preserve attempt and investigate, never reselect on test")
    dist.barrier()  # No formal inference until the original development score is reproduced.
    family_results = []
    # Match the main-table evaluator: nine independent task-family calls, so
    # same-length prompts from different families never share a decode batch.
    with torch.inference_mode():
        for family in ATOMS + TASKS:
            family_result = evaluate_bank(core, tokenizer, banks[suite], task_types=(family,),
                                          batch_size=manifest["inference"]["batch_size"],
                                          collect_predictions=True, collect_generated_text=True, **options)
            if rank == 0:
                family_results.append(family_result)
    if rank == 0:
        combined = merge_family_results(family_results, ATOMS + TASKS)
        result = _result(combined["task_counts"], combined["proof_groups"], condition="matched",
                         gate_multiplier=float(config["model"]["gate_max"]))
        result.update(combined, route="circuit_a")
        verify_predictions(result, ATOMS + TASKS)
        predictions = write_prediction_chunks(directory, result.pop("predictions"))
        verify_sources(manifest)
        check_file(ROOT / spec["checkpoint_path"], spec["checkpoint_sha256"])
        if selection_audit(task, config, manifest, archived_config_sha) != validation:
            raise ValueError("original selection artifacts changed during evaluation")
        write_new(directory / "result.json", result)
        summary = {
            "format": "stage2_permuted_formal_evaluation_v1", "status": "complete",
            "task": task, "suite": suite, "attempt": attempt,
            "manifest_sha256": sha(PACKAGE / "eval_manifest.json"),
            **{k: spec[k] for k in ("checkpoint_path", "checkpoint_sha256", "selected_step", "config_sha256")},
            "bank": {**manifest["suites"][suite], "metadata_identity": banks[suite].metadata["identity_sha256"]},
            "validation": {**validation, "replayed": True},
            "validation_sha256": sha(directory / "validation.json"),
            "result_sha256": sha(directory / "result.json"),
            "inference": manifest["inference"], "task_counts": result["task_counts"],
            "event_count": result["event_count"], "predictions": predictions,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "training_updates": 0, "circuit_present": True,
        }
        write_new(directory / "summary.json", summary)
        print(f"COMPLETE {task} {suite} {directory}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--attempt")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    manifest = load_manifest()
    if args.preflight_only:
        _, validation, _, _ = preflight(args.task, args.suite, manifest)
        print(__import__("json").dumps({"status": "passed", "task": args.task, "suite": args.suite,
              "manifest_sha256": sha(PACKAGE / "eval_manifest.json"), "validation": validation}, sort_keys=True))
        return
    if not args.attempt or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", args.attempt):
        parser.error("--attempt must be a unique safe run identifier, at most 96 characters")
    run(args.task, args.suite, args.attempt, manifest)


if __name__ == "__main__":
    main()

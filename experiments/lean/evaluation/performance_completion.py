"""Identity-preserving seed repetitions of the frozen 2026-11-03 matrix.

The historical validators still validate the source recipe. Only training RNG,
fresh initialization, output locations and this provenance tag may differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "4server/1103/lean_performance_completion_v1"
OUTPUT = "outputs/s5_lean_circuit/performance_completion_20261103_v1"
TAG = "performance_completion_run_id"
SEED_SCREEN_PACKAGE = ROOT / "4server/1104/lean_circuit_seed_screen_v1"


def read(path):
    return json.loads(Path(path).read_text())


def matrix():
    return read(PACKAGE / "run_matrix.json")


def study_matrix(run_id):
    if run_id.startswith("seed_screen1104_"):
        return read(SEED_SCREEN_PACKAGE / "run_matrix.json")
    return matrix()


def approved_training_seeds():
    seeds = {42, 43, 44}
    path = SEED_SCREEN_PACKAGE / "run_matrix.json"
    if path.exists():
        seeds.update(int(seed) for seed in read(path)["training_seeds"])
    return seeds


def evaluation_cases(spec, tasks):
    """Explicit condition dispatch; never silently score an unknown label as ON."""
    cases = []
    for condition in spec["conditions"]:
        if condition == "cross_task":
            from .protocol import COMPOSITION_TASKS
            target = spec["heldout"]
            donors = spec.get("donor_tasks", [])
            if (spec["route"] != "circuit_a" or spec["kind"] != "composition"
                    or target not in COMPOSITION_TASKS or target not in tasks
                    or len(donors) != 4 or set(donors) != set(COMPOSITION_TASKS) - {target}
                    or not set(donors) <= set(tasks)):
                raise ValueError("cross_task spec requires one heldout and the other four composition donors")
            for donor in donors:
                cases.append({"label": f"cross_task_from_{donor}", "condition": "cross_task",
                              "task": target, "donor_task": donor})
        elif condition in {"matched", "absent", "shuffled", "zero12_token_present"}:
            for task in tasks:
                cases.append({"label": condition, "condition": "matched" if condition == "zero12_token_present" else condition,
                              "task": task, "donor_task": None})
        else:
            raise ValueError(f"Unsupported evaluation condition: {condition}")
    if len({(c['label'], c['task']) for c in cases}) != len(cases):
        raise ValueError("Duplicate evaluation cases")
    return cases


def job(run_id):
    return next(j for j in study_matrix(run_id)["jobs"] if j["run_id"] == run_id)


def source(j):
    plan = study_matrix(j["run_id"])
    record = (plan["atomic_origins"][j["route"]] if j["stage"] == "atomic_hypernet"
              else next(a for a in plan["anchors"] if a["config"] == j["source_config"]))
    p = ROOT / record["config"]
    if hashlib.sha256(p.read_text(encoding="utf-8-sig").encode()).hexdigest() != record["config_lf_text_sha256"]:
        raise ValueError("Frozen source recipe changed: " + str(p))
    return p


def output(j):
    root = study_matrix(j["run_id"]).get("output_root", OUTPUT)
    return f"{root}/runs/{j['run_id']}"


def atomic_value(j):
    from .hypernet.config import load_run_config
    plan = study_matrix(j["run_id"])
    if j["stage"] != "atomic_hypernet" or j["training_seed"] not in plan["training_seeds"]:
        raise ValueError("Not an approved independent atomic run")
    value = copy.deepcopy(load_run_config(source(j)).to_dict())
    seed = j["training_seed"]
    init = f"{plan.get('output_root', OUTPUT)}/initialization/seed{seed}"
    value["model"]["seed"] = seed
    value["assignment_seed"] = seed  # batch order, not corpus identity
    value["common_init_manifest"] = init + "/common"
    if value["candidate_init_manifest"] is not None:
        value["candidate_init_manifest"] = init + "/candidate_A"
    value["output_dir"] = output(j)
    value["run_name"] = j["run_id"]
    value["extra"][TAG] = j["run_id"]
    return value


def validate_atomic(config):
    run_id = config.extra.get(TAG)
    if config.model.seed == 42 and run_id is None:
        return
    if run_id is None or config.to_dict() != atomic_value(job(run_id)):
        raise ValueError("Independent atomic run differs from its frozen recipe")


def composition_value(j):
    from .composition_loco_config import load_config
    if j["stage"] != "composition" or j["training_seed"] not in study_matrix(j["run_id"])["training_seeds"]:
        raise ValueError("Not an approved independent composition run")
    value = copy.deepcopy(load_config(source(j)).to_dict())
    value["training"]["seed"] = j["training_seed"]
    value["training"][TAG] = j["run_id"]
    value["output_dir"] = output(j)
    if j["route"] != "direct_sft":
        parent = job(j["depends_on"][0])
        # Published only after terminal success and exact checkpoint verification.
        ready = read(ROOT / output(parent) / "accepted.json")
        if ready["run_id"] != parent["run_id"] or ready["training_seed"] != j["training_seed"]:
            raise ValueError("Atomic predecessor seed/identity differs")
        expected_path = f"{output(parent)}/checkpoints/model_step{parent['export_step']:07d}.pt"
        if ready["checkpoint_path"] != expected_path or ready["step"] != parent["export_step"]:
            raise ValueError("Atomic predecessor checkpoint/step differs")
        value["initialization"]["checkpoint_path"] = expected_path
        value["initialization"]["checkpoint_sha256"] = ready["checkpoint_sha256"]
    return value


def validate_composition(config):
    if config.to_dict() != composition_value(job(config.training[TAG])):
        raise ValueError("Independent composition run differs from frozen recipe or predecessor")

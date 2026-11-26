"""Sealed LR-only extension of Circuit retention20; 16 learning rate variants."""
from pathlib import Path
import copy
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
NAME = "composition_loco_circuit_refine16_v1"
OUTPUT = "outputs/s5_lean_circuit/" + NAME
CONFIG = "s5_lean_circuit/configs/" + NAME
TASKS = ("T_nested_or", "T_and_or_and")
# Actual peak LR: PG, query, circuit. Ratios are derived, never inherited.
RATES = {
    "R01_query_half": (2.5e-5, 6.25e-6, 5e-5),
    "R02_query_quarter": (2.5e-5, 3.125e-6, 5e-5),
    "R03_circuit_half": (2.5e-5, 1.25e-5, 2.5e-5),
    "R04_circuit_quarter": (2.5e-5, 1.25e-5, 1.25e-5),
    "R05_circuit_double": (2.5e-5, 1.25e-5, 1e-4),
    "R06_pg_half": (1.25e-5, 1.25e-5, 5e-5),
    "R07_all_half": (1.25e-5, 6.25e-6, 2.5e-5),
    "R08_encoders_half": (2.5e-5, 6.25e-6, 2.5e-5),
}

def rows():
    return [
        {"task": task, "recipe": recipe,
         "config": f"{CONFIG}/{task}/{recipe}/circuit_a.json"}
        for i, (task, recipe) in enumerate((t, r) for t in TASKS for r in RATES)
    ]

def build(task, recipe):
    if task not in TASKS or recipe not in RATES:
        raise ValueError("Unknown Circuit refine16 cell")
    anchor_group = "M25" if task == "T_nested_or" else "M12"
    anchor = ROOT / "s5_lean_circuit/configs/composition_loco_circuit_retention20_v2" / task / anchor_group / "circuit_a.json"
    value = copy.deepcopy(json.loads(anchor.read_text(encoding="utf-8")))
    value.pop("config_sha256", None)
    pg, query, circuit = RATES[recipe]
    value["model"].update(pg_lr=pg, query_lr_multiplier=query / pg, circuit_lr_multiplier=circuit / pg)
    value["training"]["circuit_refine16_id"] = recipe
    # Original retention group fields remain replay/anchor identities, not scheduling.
    value["output_dir"] = f"{OUTPUT}/runs/{task}/{recipe}/circuit_a"
    return value

def validate(config):
    recipe = config.training.get("circuit_refine16_id")
    if config.to_dict() != build(config.heldout_task, recipe):
        raise ValueError("Circuit refine16 differs from the sealed LR-only configuration")

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()

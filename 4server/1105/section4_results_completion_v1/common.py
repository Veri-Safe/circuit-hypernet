"""Validate the two published repair recipes against their base training configurations."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parent
STORE = ROOT / "outputs/s5_lean_circuit/section4_results_completion_20261105_v1"
FOLDS = {'T_or_and': 'M25', 'T_nested_and': 'C25'}
sys.path[:0] = [str(ROOT/'s5_lean_circuit/src'), str(ROOT/'s2_hypernet/src'), str(ROOT/'temp_s')]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def repair_config(task):
    """The selected repair has the published fold settings and a separate output directory."""
    from s5_lean_circuit.composition_loco_config import load_config
    source = ROOT / "s5_lean_circuit/configs/composition_loco_circuit_retention20_v2" / task / FOLDS[task] / "circuit_a.json"
    baseline = load_config(source)
    config = replace(baseline, output_dir=(STORE / "runs" / task).relative_to(ROOT).as_posix(), config_sha256=None)
    recipe = read(PACKAGE / "configs" / f"{task}.json")
    if recipe.pop("config_sha256") != config.computed_sha256 or config.to_dict() != recipe:
        raise ValueError(f"Published repair recipe differs from the selected fold: {task}")
    if config.training["seed"] != 42 or config.training["total_steps"] != 12000 or config.training["scheduler_total_steps"] != 38000:
        raise ValueError("Repair training schedule changed")
    return config


def require_sources():
    for path, digest in read(PACKAGE/'runtime_sources.json').items():
        if sha(ROOT/path) != digest:
            raise ValueError('Frozen runtime source changed: '+path)

"""Task identity and output locations shared by the migrated M12 engines."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "s5_lean_circuit/src", ROOT / "s2_hypernet/src", ROOT / "s5_lean_circuit/scripts", ROOT / "temp_s"):
    if str(directory) not in sys.path: sys.path.insert(0, str(directory))
SPEC = ROOT / "4server/lean_nested_or_r03_mechanism_v1/spec.json"
SOURCE = ROOT / "logs/eval/2026-11-04/lean-lc026-nested-or-r03-mechanism-v1"
OUT = SOURCE / "circuit"
LOG = SOURCE / "sae"
DATA = ROOT / "outputs/pg_transcoder/lean-lc026-nested-or-r03-mechanism-v1"
CONFIG = ROOT / "s5_lean_circuit/configs/composition_loco_circuit_refine16_v1/T_nested_or/R03_circuit_half/circuit_a.json"
COMPOSITION_BANK = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/visible_validation.pt"
ATOMIC_BANK = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/atomic_retention.pt"
TASKS = ("T_nested_or", "S_or_l", "S_or_r")
COMPOSITION = TASKS[0]
BRANCHES = ("left_left", "left_right", "direct_right")
ZERO12 = tuple(range(9)) + (131, 132, 133)
RUNTIME_FILES = (
    "s5_lean_circuit/src/s5_lean_circuit/hypernet/runtime.py",
    "s5_lean_circuit/src/s5_lean_circuit/hypernet/parameter_generator.py",
    "s2_hypernet/src/s2_hypernet/parameter_generator.py",
    "s2_hypernet/src/s2_hypernet/nvarc_pg/generated_lora.py",
    "s5_lean_circuit/src/s5_lean_circuit/pg_transcoder_adapter.py",
    "s5_lean_circuit/src/s5_lean_circuit/modeling.py",
    "s5_lean_circuit/src/s5_lean_circuit/protocol.py",
    "s5_lean_circuit/src/s5_lean_circuit/circuit/transcoder.py",
    "s5_lean_circuit/src/s5_lean_circuit/circuit/model_io.py",
    "s5_lean_circuit/src/s5_lean_circuit/composition_loco.py",
)


def sha(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()


@lru_cache(maxsize=1)
def target_identity():
    runtime_sources = json.loads((SOURCE / "runtime_sources.json").read_text())
    if not runtime_sources["complete"] or any(sha(ROOT / item["path"]) != item["sha256"] for item in runtime_sources["files"]):
        raise ValueError("shared runtime code changed since the R03 source snapshot")
    target = json.loads((SOURCE / "target.json").read_text())
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    config = json.loads(CONFIG.read_text())
    if (config["heldout_task"] != COMPOSITION or config["fold_id"] != "holdout_T_nested_or"
            or config["training"]["circuit_retention20_group_id"] != "M25"
            or config["training"].get("circuit_refine16_id") != "R03_circuit_half"
            or sha(ROOT / target["checkpoint_path"]) != manifest["checkpoint_sha256"]
            or target["checkpoint_sha256"] != manifest["checkpoint_sha256"]
            or sha(CONFIG) != manifest["runtime_config_sha256"]
            or target["generation"] != {"max_new_tokens": 50, "n_ctx": 128}):
        raise ValueError("frozen R03 identity differs")
    return target


def baseline_counts():
    """Frozen counts for the three graph conditions; extra controls stay in the full summary."""
    value = json.loads((SOURCE / "baseline_summary.json").read_text())
    if not value["complete"] or value["question_count"] != 1500:
        raise ValueError("full R03 baseline is not complete")
    return {task: {condition: value['tasks'][task][condition]['correct']
                  for condition in ('on','off','zero12')} for task in TASKS}

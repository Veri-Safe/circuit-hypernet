"""Immutable identities and existing data authorization for the gradient round."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LOG = ROOT / "logs/eval/2026-11-14/lean-stage2-nested-or-groups-v1/gradient_contrast_r03_pareto_v1"
OUT = ROOT / "outputs/pg_transcoder/lean_stage2_nested_or_groups_v1/gradient_contrast_r03_pareto_v1"
PRIOR = ROOT / "logs/eval/2026-11-14/lean-stage2-nested-or-groups-v1/relative_nodes_r03_expand_v1"
PREVIOUS_CODE = HERE.parent / "relative_nodes_r03_expand_v1"
SOURCE = ROOT / "logs/eval/2026-11-04/lean-lc026-nested-or-r03-mechanism-v1"
TASKS = ("T_nested_or", "T_and_or", "T_or_and", "T_and_or_and", "T_nested_and",
         "S_and", "S_or_l", "S_or_r", "S_id")
BLOCKS = (0, 5, 10)
CHECKPOINT = "cd749589e530be32621c80f5c4cc278ce844f0d74cb90a7c9103ce08889c031e"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def freeze(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, f"Frozen artifact changed: {path}"
    else:
        write(path, value)


def previous_authorization():
    # Reuse the user's actual authorization of these exact banks, not a new
    # implicit substitute. The prior module has different names from this round.
    sys.path.append(str(PREVIOUS_CODE))
    from runtime import approved_plan
    return approved_plan()


def inputs():
    prior_plan, approval = previous_authorization()
    protocol = json.loads((HERE / "protocol.json").read_text())
    parent_code = HERE.parent / "gradient_contrast_r03_v1"
    parent_log = LOG.parent / "gradient_contrast_r03_v1"
    assert protocol["parent_groups_sha256"] == sha(parent_code / "groups.json")
    parent = json.loads((parent_code / "groups.json").read_text())
    assert parent["protocol_sha256"] == sha(parent_code / "protocol.json")
    assert protocol["parent_screen_summary_sha256"] == sha(parent_log / "screen_summary.json")
    assert protocol["parent_smoke_receipt_sha256"] == sha(parent_log / "receipts/smoke/lane0.json")
    for name, digest in protocol["release_code_sha256"].items():
        assert sha(HERE / name) == digest
    assert protocol["checkpoint_sha256"] == CHECKPOINT == prior_plan["checkpoint_sha256"]
    assert protocol["banks"] == prior_plan["banks"] and protocol["totals"] == prior_plan["totals"]
    assert protocol["data_authorization_sha256"] == sha(PRIOR / "data_approval.json")
    assert protocol["previous_summary_sha256"] == sha(PRIOR / "full_summary.json")
    assert protocol["questions_sha256"] == prior_plan["questions_sha256"]
    assert protocol["blocks"] == list(BLOCKS)
    assert protocol["source_manifest_sha256"] == sha(SOURCE / "manifest.json")
    assert protocol["source_target_sha256"] == sha(SOURCE / "target.json")
    for item in protocol["dictionary"].values():
        assert sha(ROOT / item["path"]) == item["sha256"]
    questions = json.loads((ROOT / prior_plan["questions_path"]).read_text())["rows"]
    assert len(questions) == len({(q["task"], q["index"]) for q in questions}) == 4500
    assert Counter(q["task"] for q in questions) == {t: 500 for t in TASKS}
    return protocol, questions


def references(questions):
    summary = json.loads((PRIOR / "full_summary.json").read_text())
    expected = {(q["task"], q["index"]): q for q in questions}
    result = {}
    for item in summary["receipts"]:
        path = ROOT / item["raw_path"]
        assert sha(path) == item["raw_sha256"]
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                q = row["question"]
                key = q["task"], q["index"]
                assert key not in result and q == expected[key]
                predictions = row["predictions"]
                result[key] = {"native": predictions["native"],
                    "anchor8": predictions["r03_expand_anchor14"],
                    "anchor11": predictions["r03_expand_union_14_07_a000"]}
    assert set(result) == set(expected)
    return result


def screen_questions(questions):
    rows = []
    for task in TASKS:
        rows.extend([q for q in questions if q["task"] == task][:200 if task == "T_nested_or" else 50])
    assert len(rows) == 600
    return rows


def identity(phase, lane, parts):
    return {"phase": phase, "lane": lane, "parts": parts,
            "checkpoint_sha256": CHECKPOINT, "protocol_sha256": sha(HERE / "protocol.json"),
            "code_sha256": {p.name: sha(p) for p in HERE.glob("*.py")}}

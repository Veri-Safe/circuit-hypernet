"""Frozen inputs and explicit user data approval for the R03 expansion round."""
from __future__ import annotations

import json
import os
from pathlib import Path

from prepare import ROOT, HERE, LOG, OUT, SOURCE, TASKS, CHECKPOINT, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def approved_plan():
    plan = json.loads((HERE / "plan.json").read_text())
    approval_path = LOG / "data_approval.json"
    if not approval_path.is_file():
        raise RuntimeError("Explicit user approval of the exact banks and 500/task denominator is required; see README.md")
    approval = json.loads(approval_path.read_text())
    assert approval["approved"] is True and approval["user_message"].strip()
    assert approval["plan_sha256"] == sha(HERE / "plan.json")
    assert approval["banks"] == plan["banks"] and approval["totals"] == plan["totals"]
    assert approval["scope"] == "visible500_discovery_only"
    assert plan["checkpoint_sha256"] == CHECKPOINT and plan["circuit_present"] is True
    assert plan["gate_multiplier"] == 0.5
    assert plan["totals"] == {t: 500 for t in TASKS}
    assert plan["question_count"] == 4500 and plan["formal_holdout"] is False
    for label, filename in (("source_target", "target.json"), ("source_manifest", "manifest.json")):
        assert sha(SOURCE / filename) == plan[label + "_sha256"]
    for label in ("source_groups", "source_summary", "questions"):
        assert sha(ROOT / plan[label + "_path"]) == plan[label + "_sha256"]
    assert sha(ROOT / plan["checkpoint_path"]) == CHECKPOINT
    for item in (*plan["banks"].values(), *plan["dictionary"].values()):
        assert sha(ROOT / item["path"]) == item["sha256"]
    assert plan["banks"]["visible"]["path"].endswith("/visible_validation.pt")
    assert plan["banks"]["atomic"]["path"].endswith("/atomic_retention.pt")
    assert len(plan["groups"]) == len({g["id"] for g in plan["groups"]}) == 19
    for group in plan["groups"]:
        assert group["alpha"] in (0.0, 0.25) and len(group["sites"]) == 1
        site = group["sites"][0]
        assert site["site"] == "block5.residual"
        features = site["feature_ids"]
        assert features and len(features) == len(set(features))
        assert all(type(f) is int and 0 <= f < 8192 for f in features)
        slots = site["slots"]
        assert slots is None or (slots and len(slots) == len(set(slots)) and
                                all(type(s) is int and 0 <= s < 272 for s in slots))
    return plan, approval


def questions_for(plan):
    rows = json.loads((ROOT / plan["questions_path"]).read_text())["rows"]
    assert len(rows) == len({(r["task"], r["index"]) for r in rows}) == 4500
    assert {t: sum(r["task"] == t for r in rows) for t in TASKS} == plan["totals"]
    assert all(r["bank"] == ("visible" if r["task"].startswith("T_") else "atomic") for r in rows)
    return rows


if __name__ == "__main__":
    plan, _ = approved_plan()
    print("LEAN_NOR_R03_EXPAND_APPROVAL_VALIDATED", len(questions_for(plan)), len(plan["groups"]))

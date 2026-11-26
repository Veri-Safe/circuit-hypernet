#!/usr/bin/env python3
"""Audit completed predictions and freeze an unevaluated expansion round on CPU."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LOG = ROOT / "logs/eval/2026-11-14/lean-stage2-nested-or-groups-v1/relative_nodes_r03_expand_v1"
OUT = ROOT / "outputs/pg_transcoder/lean_stage2_nested_or_groups_v1/relative_nodes_r03_expand_v1"
PRIOR = ROOT / "logs/eval/2026-11-13/lean-stage2-nested-or-groups-v1/relative_nodes_r03_controls_v1"
PRIOR_GROUPS = ROOT / "outputs/pg_transcoder/lean_stage2_nested_or_groups_v1/relative_nodes_r03_controls_v1/groups.json"
SOURCE = ROOT / "logs/eval/2026-11-04/lean-lc026-nested-or-r03-mechanism-v1"
QUESTIONS = ROOT / "outputs/pg_transcoder/lean_stage2_nested_or_groups_v1/relative_nodes_r03_v1/questions.json"
COMP = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/visible_validation.pt"
ATOM = ROOT / "outputs/s5_lean_circuit/composition_loco_sft_v1/eval/atomic_retention.pt"
TASKS = ("T_nested_or", "T_and_or", "T_or_and", "T_and_or_and", "T_nested_and",
         "S_and", "S_or_l", "S_or_r", "S_id")
CHECKPOINT = "cd749589e530be32621c80f5c4cc278ce844f0d74cb90a7c9103ce08889c031e"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text() != data:
            raise ValueError(f"Frozen artifact changed: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(data)
    os.replace(temporary, path)


def main():
    summary_path = PRIOR / "full_summary.json"
    old = json.loads(summary_path.read_text())
    manifest = json.loads((PRIOR / "manifest.json").read_text())
    groups = json.loads(PRIOR_GROUPS.read_text())
    target = json.loads((SOURCE / "target.json").read_text())
    questions = json.loads(QUESTIONS.read_text())["rows"]
    assert old["complete"] and old["checkpoint_sha256"] == CHECKPOINT
    assert manifest["scope"] == "visible500_discovery_only" and not manifest["retired_final_bank_used"]
    assert manifest["questions_sha256"] == old["questions_sha256"] == sha(QUESTIONS)
    assert manifest["groups_sha256"] == sha(PRIOR_GROUPS)
    assert sha(ROOT / target["checkpoint_path"]) == CHECKPOINT == target["checkpoint_sha256"]
    assert manifest["banks"] == {"visible": sha(COMP), "atomic": sha(ATOM)}
    for dictionary in groups["dictionary"].values():
        assert sha(ROOT / dictionary["path"]) == dictionary["sha256"]
    expected = {(q["task"], q["index"]): q for q in questions}
    assert len(questions) == len(expected) == 4500
    assert Counter(q["task"] for q in questions) == {t: 500 for t in TASKS}
    names = [g["id"] for g in groups["groups"]]
    counts = {name: Counter() for name in ("native", *names)}
    losses = {name: {task: set() for task in TASKS} for name in names}
    gains = {name: {task: set() for task in TASKS} for name in names}
    seen, receipts = set(), []
    receipts_dir = PRIOR / "receipts/full"
    partitions = {p.name for p in receipts_dir.glob("lane[0-9]*.json")}
    parts = len(partitions)
    assert parts and partitions == {f"lane{i}.json" for i in range(parts)}, "Incomplete prediction partitions"
    for lane in range(parts):
        receipt_path = PRIOR / f"receipts/full/lane{lane}.json"
        receipt = json.loads(receipt_path.read_text())
        assert receipt["complete"] and receipt["identity"]["phase"] == "full"
        assert receipt["identity"]["lane"] == lane
        assert receipt["identity"]["checkpoint_sha256"] == CHECKPOINT
        assert receipt["identity"]["questions_sha256"] == sha(QUESTIONS)
        assert receipt["identity"]["groups_sha256"] == sha(PRIOR_GROUPS)
        assert receipt["identity"]["candidates"] == names
        raw = ROOT / receipt["path"]
        assert sha(raw) == receipt["raw_sha256"]
        lane_counts = {name: Counter() for name in counts}
        lane_totals = Counter()
        lane_expected = {(q["task"], q["index"]) for q in questions[lane::parts]}
        with gzip.open(raw, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                q = row["question"]
                key, task = (q["task"], q["index"]), q["task"]
                assert key in lane_expected and key not in seen and q == expected[key]
                seen.add(key)
                predictions = row["predictions"]
                assert set(predictions) == set(counts)
                native = predictions["native"]["correct"]
                for name, prediction in predictions.items():
                    assert prediction["correct"] == (prediction["first_tactic"] == q["gold"])
                    correct = int(prediction["correct"])
                    counts[name][task] += correct
                    lane_counts[name][task] += correct
                    if name != "native":
                        if native and not correct:
                            losses[name][task].add(q["index"])
                        elif correct and not native:
                            gains[name][task].add(q["index"])
                lane_totals[task] += 1
        assert dict(lane_totals) == {k: v for k, v in receipt["totals"].items() if v}
        for name in counts:
            assert all(lane_counts[name][t] == receipt["counts"][name][t] for t in TASKS)
        receipts.append({"path": str(receipt_path.relative_to(ROOT)), "sha256": sha(receipt_path),
                         "raw_path": receipt["path"], "raw_sha256": receipt["raw_sha256"]})
    assert len(seen) == 4500
    assert {name: dict(c) for name, c in counts.items()} == old["counts"]
    qualified = [c for c in old["candidates"] if c["per_id_relative_criterion"]]
    qualified.sort(key=lambda c: (-c["ood_relative_decline"], -c["relative_margin"], c["group"]))
    selected = [c["group"] for c in qualified[:4]]
    assert selected == [f"r03_b05_random_matched_{i:02d}" for i in (14, 7, 11, 10)]
    by_name = {g["id"]: g for g in groups["groups"]}
    candidates = []

    def candidate(name, features, alpha, slots, origin):
        assert len(features) == len(set(features)) and all(f in groups["parent_feature_ids"] for f in features)
        candidates.append({"id": name, "sites": [{"site": "block5.residual", "feature_ids": features,
                                                  "slots": slots}], "alpha": alpha, "origin": origin})

    candidate("r03_expand_anchor14", by_name[selected[0]]["sites"][0]["feature_ids"], 0.0, None,
              {"kind": "reproduce_previous_best", "group": selected[0]})
    overlap = []
    for left, right in combinations(selected, 2):
        suffix = left.rsplit("_", 1)[1] + "_" + right.rsplit("_", 1)[1]
        features = sorted(set(by_name[left]["sites"][0]["feature_ids"]) |
                          set(by_name[right]["sites"][0]["feature_ids"]))
        for alpha in (0.0, 0.25):
            candidate(f"r03_expand_union_{suffix}_a{int(alpha*100):03d}", features, alpha, None,
                      {"kind": "union_of_discovery_selected_groups", "parents": [left, right]})
        overlap.append({"parents": [left, right], "feature_union_size": len(features),
                        "ood_losses_left": len(losses[left]["T_nested_or"]),
                        "ood_losses_right": len(losses[right]["T_nested_or"]),
                        "ood_loss_intersection": len(losses[left]["T_nested_or"] & losses[right]["T_nested_or"]),
                        "ood_loss_union": len(losses[left]["T_nested_or"] | losses[right]["T_nested_or"]),
                        "interpretation": "paired old predictions only; union intervention accuracy is unknown"})
    partitions = [("quarter0", 0, 68), ("quarter1", 68, 136),
                  ("quarter2", 136, 204), ("quarter3", 204, 272),
                  ("half0", 0, 136), ("half1", 136, 272)]
    for label, start, end in partitions:
        candidate(f"r03_expand_parent32_{label}", groups["parent_feature_ids"], 0.0,
                  list(range(start, end)), {"kind": "fixed_slot_partition_of_parent32",
                    "canonical_layout": "8 target-layer positions x 34 parameter slots; runtime mapping must be saved"})
    assert len(candidates) == len({c["id"] for c in candidates}) == 19
    plan = {"format": "lean_nested_or_r03_expansion_plan_v1", "complete": True,
            "execution_status": "awaiting_explicit_data_approval", "new_scores_available": False,
            "checkpoint_path": target["checkpoint_path"], "checkpoint_sha256": CHECKPOINT,
            "dictionary": groups["dictionary"], "groups": candidates,
            "source_summary_path": str(summary_path.relative_to(ROOT)), "source_summary_sha256": sha(summary_path),
            "source_groups_path": str(PRIOR_GROUPS.relative_to(ROOT)), "source_groups_sha256": sha(PRIOR_GROUPS),
            "source_target_sha256": sha(SOURCE / "target.json"),
            "source_manifest_sha256": sha(SOURCE / "manifest.json"),
            "questions_path": str(QUESTIONS.relative_to(ROOT)), "questions_sha256": sha(QUESTIONS),
            "banks": {"visible": {"path": str(COMP.relative_to(ROOT)), "sha256": sha(COMP)},
                      "atomic": {"path": str(ATOM.relative_to(ROOT)), "sha256": sha(ATOM)}},
            "totals": {task: 500 for task in TASKS}, "question_count": 4500,
            "scope": "proposed_visible500_discovery_only", "formal_holdout": False,
            "circuit_present": True, "gate_multiplier": 0.5,
            "scoring": "greedy max_new_tokens=50 strict_first_tactic_exact_match",
            "intervention": "subtract (1-alpha) native ReLU decoded contribution, preserving reconstruction residual",
            "same_intervention_for_every_task": True,
            "selection_rule": "OOD relative decline first among groups with every ID relative decline smaller; report all nine tasks and full Pareto tradeoffs",
            "random_control_claim": False,
            "approval_requirement": "Explicit approval of the exact banks and 500/task denominator before new scoring; these banks are not an independent final test"}
    save(HERE / "plan.json", plan)
    save(LOG / "prior_prediction_audit.json", {"complete": True, "new_inference_run": False,
         "checkpoint_sha256": CHECKPOINT, "questions_verified": len(seen), "conditions_verified": 21,
         "counts": old["counts"], "best": old["best"], "receipts": receipts,
         "paired_loss_overlap": overlap, "new_intervention_scores": None,
         "historical_scope": "visible500_discovery_only"})
    print(json.dumps({"verified_questions": len(seen), "verified_prediction_conditions": 21,
                      "unevaluated_candidates": len(candidates), "execution_status": plan["execution_status"],
                      "paired_loss_overlap": overlap}, ensure_ascii=False))


if __name__ == "__main__":
    main()

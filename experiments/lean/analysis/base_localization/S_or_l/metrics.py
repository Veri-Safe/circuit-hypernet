#!/usr/bin/env python3
"""Protocol metrics, development selection and paired confirmation statistics.

Prediction row: question_id, task_type, outputs[condition].correct. Outputs may
also be top-level condition keys for read-only adapters of existing records.
All decisions use rational arithmetic. Floats are presentation/statistics only.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Mapping

TASKS = ("S_and", "S_or_l", "S_or_r", "S_id")
TARGET = "S_or_l"


def _fraction(value: Fraction | None) -> dict | None:
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def _recover(value: dict | None) -> Fraction | None:
    return None if value is None else Fraction(value["numerator"], value["denominator"])


def correct(row: Mapping, condition: str) -> int:
    output = row.get("outputs", row)[condition]
    value = output["correct"] if isinstance(output, Mapping) else output
    if type(value) not in (bool, int) or value not in (0, 1):
        raise ValueError(f"strict correctness must be bool/0/1: {condition}")
    return int(value)


def from_counts(counts: Mapping[str, Mapping[str, int]], target: str = TARGET) -> dict:
    if set(counts) != set(TASKS) or target not in TASKS:
        raise ValueError("all four tasks are required")
    per_task = {}
    for task in TASKS:
        N, n0, n1 = (counts[task][key] for key in ("N", "n0", "n1"))
        if any(type(x) is not int for x in (N, n0, n1)) or N <= 0 or not (0 <= n0 <= N and 0 <= n1 <= N):
            raise ValueError(f"invalid counts for {task}")
        signed = Fraction(n0 - n1, n0) if n0 else None
        per_task[task] = {"N": N, "n0": n0, "n1": n1, "accuracy0": n0 / N,
                          "accuracy1": n1 / N, "net_correct_drop": n0 - n1,
                          "signed_relative_drop": float(signed) if signed is not None else None,
                          "signed_relative_drop_exact": _fraction(signed)}
    n0 = counts[target]["n0"]
    d = n0 - counts[target]["n1"]
    b = sum(counts[t]["n0"] for t in TASKS if t != target)
    a = sum(abs(counts[t]["n0"] - counts[t]["n1"]) for t in TASKS if t != target)
    L = Fraction(d, n0) if n0 else None
    C = Fraction(a, b) if b else None
    valid = L is not None and C is not None
    R = L / C if valid and C > 0 else None
    if not valid:
        status = "undefined_original_denominator"
    elif C == 0 and L > 0:
        status = "zero_control_net_change"
    elif L <= 0:
        status = "no_positive_target_effect"
    else:
        status = "finite"
    return {"target": target, "per_task": per_task, "target_net_correct_drop": d,
            "target_original_correct": n0, "control_abs_net_correct_change": a,
            "control_original_correct": b, "L": float(L) if L is not None else None,
            "C": float(C) if C is not None else None,
            "R": "+inf" if status == "zero_control_net_change" else (float(R) if R is not None else None),
            "L_exact": _fraction(L), "C_exact": _fraction(C), "R_exact": _fraction(R),
            "ratio_status": status, "metrics_defined": valid,
            "L_minus_C": float(L - C) if valid else None,
            "hn_group_pass": bool(valid and 5 * d >= 2 * n0 and 10 * (d * b - a * n0) > n0 * b),
            "joint_quality_goal_met": bool(valid and 5 * d >= 2 * n0)}


def score_rows(rows: Iterable[Mapping], baseline: str, edited: str, target: str = TARGET) -> dict:
    counts = {task: {"N": 0, "n0": 0, "n1": 0} for task in TASKS}
    seen = set()
    for row in rows:
        qid, task = row["question_id"], row["task_type"]
        if qid in seen or task not in counts:
            raise ValueError(f"duplicate question or unknown task: {qid}/{task}")
        seen.add(qid)
        counts[task]["N"] += 1
        counts[task]["n0"] += correct(row, baseline)
        counts[task]["n1"] += correct(row, edited)
    return {**from_counts(counts, target), "baseline_condition": baseline, "edited_condition": edited,
            "question_count": len(seen)}


def compare(direct: dict, joint: dict, hn: dict | None = None) -> dict:
    if direct["target"] != joint["target"]:
        raise ValueError("targets differ")
    for task in TASKS:
        if any(direct["per_task"][task][k] != joint["per_task"][task][k] for k in ("N", "n0")):
            raise ValueError("direct and joint must share identical baseline counts")
    ld, cd, lj, cj = (_recover(x[k]) for x, k in ((direct, "L_exact"), (direct, "C_exact"),
                                                (joint, "L_exact"), (joint, "C_exact")))
    valid = all(x is not None for x in (ld, cd, lj, cj))
    rd, rj = _recover(direct["R_exact"]), _recover(joint["R_exact"])
    ds, js = direct["ratio_status"], joint["ratio_status"]
    advantage = False
    if not valid:
        ratio_comparison = "undefined_original_denominator"
    elif cd == 0 and cj == 0:
        ratio_comparison = "both_zero_control_no_ratio_advantage"
    elif ds == "no_positive_target_effect":
        # SEARCH: an ineffective direct fallback does not establish ratio superiority.
        ratio_comparison = "direct_no_positive_target_effect"
    elif js == "no_positive_target_effect":
        ratio_comparison = "joint_no_positive_target_effect"
    elif js == "zero_control_net_change":
        advantage = True
        ratio_comparison = "joint_positive_infinity_direct_finite"
    elif ds == "zero_control_net_change":
        ratio_comparison = "direct_positive_infinity"
    else:
        advantage = rj > rd
        ratio_comparison = "finite"
    g = lj * cd - ld * cj if valid else None
    delta_r = rj - rd if rd is not None and rj is not None else None
    result = {"direct": direct, "joint": joint, "G": float(g) if g is not None else None,
              "G_exact": _fraction(g), "ratio_advantage": advantage, "ratio_comparison": ratio_comparison,
              "joint_quality_goal_met": joint["joint_quality_goal_met"],
              "delta_L": float(lj - ld) if valid else None,
              "delta_L_percentage_points": 100 * float(lj - ld) if valid else None,
              "control_reduction": float(cd - cj) if valid else None,
              "control_reduction_percentage_points": 100 * float(cd - cj) if valid else None,
              "delta_R": float(delta_r) if delta_r is not None else None,
              "ratio_multiplier": float(rj / rd) if rd is not None and rj is not None and rd > 0 else None}
    if hn is not None:
        result.update(hn=hn, hn_group_pass=hn["hn_group_pass"],
                      chain_positive=bool(hn["hn_group_pass"] and advantage and joint["joint_quality_goal_met"]))
    return result


def paired_bootstrap(rows: Iterable[Mapping], baseline: str = "base_baseline", direct: str = "base_direct",
                     joint: str = "base_joint", target: str = TARGET,
                     hn_baseline: str | None = None, hn_edited: str | None = None) -> dict:
    """5000 task-stratified resamples with shared indices across all conditions."""
    import numpy as np
    rows = list(rows)
    # Validate uniqueness and complete task denominators before resampling.
    score_rows(rows, baseline, direct, target)
    conditions = [baseline, direct, joint]
    if bool(hn_baseline) != bool(hn_edited):
        raise ValueError("supply both HN conditions")
    if hn_baseline:
        conditions += [hn_baseline, hn_edited]
    rng = np.random.default_rng(42)
    counts = []
    for task in TASKS:
        task_rows = sorted((r for r in rows if r["task_type"] == task), key=lambda r: r["question_id"])
        matrix = np.asarray([[correct(r, c) for c in conditions] for r in task_rows], dtype=np.int64)
        indices = rng.integers(0, len(task_rows), size=(5000, len(task_rows)))
        counts.append(matrix[indices].sum(axis=1))
    values = np.stack(counts, axis=1)
    ti = TASKS.index(target)
    ci = [i for i in range(4) if i != ti]

    def effects(before, after):
        n0 = values[:, ti, before]
        controls = values[:, ci, before].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            loss = (n0 - values[:, ti, after]) / n0
            collateral = np.abs(values[:, ci, before] - values[:, ci, after]).sum(axis=1) / controls
        loss[n0 == 0] = np.nan
        collateral[controls == 0] = np.nan
        return loss, collateral

    ld, cd = effects(0, 1)
    lj, cj = effects(0, 2)
    arrays = {"L_direct": ld, "C_direct": cd, "L_joint": lj, "C_joint": cj,
              "G": lj * cd - ld * cj}
    if hn_baseline:
        arrays["L_H"], arrays["C_H"] = effects(3, 4)
    intervals = {}
    for name, arr in arrays.items():
        finite = arr[np.isfinite(arr)]
        intervals[name] = {"lower": float(np.quantile(finite, 0.025)) if len(finite) else None,
                           "upper": float(np.quantile(finite, 0.975)) if len(finite) else None,
                           "defined_resamples": int(len(finite)), "undefined_resamples": 5000 - int(len(finite))}
    lower = intervals["G"]["lower"]
    return {"resamples": 5000, "seed": 42, "paired": True, "stratify": "task",
            "row_order": "question_id ascending within task", "confidence": 0.95,
            "method": "percentile", "intervals": intervals,
            "interval_support": lower is not None and lower > 0,
            "undefined_rule": "original-correct denominator zero is undefined; excluded and counted, no epsilon"}


def _normal_key(candidate: dict) -> tuple:
    m = candidate["metrics"]
    L, C = _recover(m["L_exact"]), _recover(m["C_exact"])
    if L is None or C is None:
        return (2, 0, 0, 0, str(candidate["id"]))
    if L <= 0:
        return (1, 0, -L, C, str(candidate["id"]))
    R = L / C if C else None
    return (0, 0 if R is None else 1, -R if R is not None else 0, -L, C, str(candidate["id"]))


def choose_best(candidates: Iterable[dict], route: str) -> dict:
    """Each candidate: id, K, metrics. Caller evaluates on complete development."""
    candidates = list(candidates)
    if not candidates or route not in ("direct", "joint"):
        raise ValueError("nonempty candidates and direct/joint route required")
    if len({c["K"] for c in candidates}) != 1:
        raise ValueError("within-route selection requires same K")
    if route == "joint":
        qualified = [c for c in candidates if c["metrics"]["joint_quality_goal_met"]]
        if qualified:
            candidates = qualified
        else:
            # Below 40%, prioritize real target effect, not tiny-loss high ratios.
            return min(candidates, key=lambda c: (
                not c["metrics"]["metrics_defined"],
                -(_recover(c["metrics"]["L_exact"]) or Fraction(0)),
                _recover(c["metrics"]["C_exact"]) if c["metrics"]["C_exact"] else math.inf,
                str(c["id"])))
    return min(candidates, key=_normal_key)


def select_pair(candidates: Iterable[dict]) -> dict:
    """Across K: quality, ratio advantage, ratio gap, joint L, joint C, K."""
    candidates = list(candidates)
    pairs = []
    for k in sorted({c["K"] for c in candidates}):
        branches = {route: [c for c in candidates if c["K"] == k and c["route"] == route]
                    for route in ("direct", "joint")}
        if not all(branches.values()):
            raise ValueError(f"both routes required for K={k}")
        d, j = (choose_best(branches[r], r) for r in ("direct", "joint"))
        comparison = compare(d["metrics"], j["metrics"])
        pairs.append({"K": k, "direct": d, "joint": j, "comparison": comparison})
    if not pairs:
        raise ValueError("no pairs")

    def key(pair):
        c, j = pair["comparison"], pair["joint"]["metrics"]
        infinity_advantage = c["ratio_comparison"] == "joint_positive_infinity_direct_finite"
        rd, rj = _recover(pair["direct"]["metrics"]["R_exact"]), _recover(j["R_exact"])
        gap = rj - rd if rd is not None and rj is not None else None
        loss, collateral = _recover(j["L_exact"]), _recover(j["C_exact"])
        return (not c["joint_quality_goal_met"], not c["ratio_advantage"], not infinity_advantage,
                -(gap if gap is not None else Fraction(-10**12)),
                -(loss if loss is not None else Fraction(-10**12)),
                collateral if collateral is not None else math.inf, pair["K"],
                str(pair["direct"]["id"]), str(pair["joint"]["id"]))
    winner = min(pairs, key=key)
    return {**winner, "all_pairs": pairs,
            "selection_rule": "quality,ratio_advantage,only_joint_positive_infinity,delta_R,L_joint,-C_joint,-K; exact rational"}


def report_markdown(summary: dict, K: int, title: str = "S_or_l frozen fresh confirmation") -> str:
    def pct(x):
        return "undefined" if x is None else f"{100 * x:.2f}%"
    def num(x):
        return "undefined" if x is None else (x if isinstance(x, str) else f"{x:.6g}")
    lines = [f"# {title}", "", f"Same K = {K}. Relative drops use original correct counts; negative control drops are improvements.", "",
             "| Route | Task | n0/N (accuracy) | n1/N (accuracy) | Signed relative drop |",
             "|---|---|---:|---:|---:|"]
    for route in ("direct", "joint"):
        for task in (TARGET,) + tuple(t for t in TASKS if t != TARGET):
            r = summary[route]["per_task"][task]
            lines.append(f"| {route} | {task} | {r['n0']}/{r['N']} ({pct(r['accuracy0'])}) | {r['n1']}/{r['N']} ({pct(r['accuracy1'])}) | {pct(r['signed_relative_drop'])} |")
    lines += ["", "| Route | L | C (sum absolute net changes / original control correct) | R |",
              "|---|---:|---:|---:|"]
    for route in ("direct", "joint"):
        m = summary[route]
        lines.append(f"| {route} | {pct(m['L'])} | {pct(m['C'])} | {num(m['R'])} |")
    lines += ["", f"delta_L: {num(summary['delta_L_percentage_points'])} percentage points; control_reduction: {num(summary['control_reduction_percentage_points'])} percentage points; delta_R: {num(summary['delta_R'])}; R_J/R_D: {num(summary['ratio_multiplier'])}.",
              f"G: {num(summary['G'])}; ratio status: {summary['ratio_comparison']}; ratio advantage: {summary['ratio_advantage']}; joint quality: {summary['joint_quality_goal_met']}."]
    if "hn" in summary:
        lines += ["", "| HN condition | Task | n0/N | n1/N |", "|---|---|---:|---:|"]
        for task in TASKS:
            r = summary["hn"]["per_task"][task]
            lines.append(f"| H_S_or_l suppressed | {task} | {r['n0']}/{r['N']} | {r['n1']}/{r['N']} |")
        h = summary["hn"]
        lines += ["", f"L_H: {pct(h['L'])}; C_H: {pct(h['C'])}; L_H-C_H: {pct(h['L_minus_C'])}; HN gate: {h['hn_group_pass']}; chain_positive: {summary['chain_positive']}."]
    if "bootstrap" in summary:
        b = summary["bootstrap"]
        lines += ["", "95% paired, task-stratified bootstrap intervals (5000, seed 42):", "",
                  "| Statistic | Lower | Upper | Undefined replicates |", "|---|---:|---:|---:|"]
        for name, value in b["intervals"].items():
            lines.append(f"| {name} | {num(value['lower'])} | {num(value['upper'])} | {value['undefined_resamples']} |")
        lines.append(f"\ninterval_support: {b['interval_support']}. Interval support is separate from point-estimate success.")
    return "\n".join(lines) + "\n"


def summarize_confirmation(rows: Iterable[Mapping], K: int, out_dir: Path | None = None) -> dict:
    rows = list(rows)
    if any(r.get("split") != "fresh_confirmation" for r in rows):
        raise ValueError("formal report requires frozen fresh_confirmation rows")
    if any(sum(r["task_type"] == task for r in rows) != 100 for task in TASKS):
        raise ValueError("formal confirmation must include exactly 100 questions per task")
    d = score_rows(rows, "base_baseline", "base_direct")
    j = score_rows(rows, "base_baseline", "base_joint")
    h = score_rows(rows, "hn_baseline", "hn_edited")
    summary = compare(d, j, h)
    summary.update(K=K, formal_results="fresh_confirmation_only",
                   bootstrap=paired_bootstrap(rows, hn_baseline="hn_baseline", hn_edited="hn_edited"))
    summary["interval_support"] = summary["bootstrap"]["interval_support"]
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "formal_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
        (out_dir / "formal_report.md").write_text(report_markdown(summary, K))
    return summary

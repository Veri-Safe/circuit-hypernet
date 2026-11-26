#!/usr/bin/env python3
"""Exact fusion-v1 metrics; bootstrap always keeps conditions paired within task.

Development API: ``measure(rows)`` accepts question_id/task_type/baseline/edited
records, with a bool or a native output containing ``correct`` per condition.
Formal API: ``summarize(rows)`` accepts base/direct/joint/hn_base/hn_edited.
No output is excluded, no epsilon is used, and no correctness-flip metric is used.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

TASKS = ("S_and", "S_or_l", "S_or_r", "S_id")
TARGET = "S_and"


def _correct(value):
    if isinstance(value, dict):
        value = value["correct"]
    if not isinstance(value, bool):
        raise TypeError("Correctness must be the native strict-scoring boolean")
    return int(value)


def count_rows(rows, baseline_key="baseline", edited_key="edited"):
    counts = {task: {"n0": 0, "n1": 0, "N": 0} for task in TASKS}
    seen = set()
    for row in rows:
        qid = row["question_id"]
        if qid in seen:
            raise ValueError(f"Duplicate question_id: {qid}")
        seen.add(qid)
        task = row.get("task_type", row.get("task"))
        count = counts[task]
        count["n0"] += _correct(row[baseline_key])
        count["n1"] += _correct(row[edited_key])
        count["N"] += 1
    if any(c["N"] == 0 for c in counts.values()):
        raise ValueError("All four tasks are required, without correctness filtering")
    return counts


def effect(counts, target=TARGET):
    """Compute L/C/R and all gates with exact integer/rational comparisons."""
    if set(counts) != set(TASKS) or target not in TASKS:
        raise ValueError("Counts must include exactly the four atomic tasks")
    clean, per_task = {}, {}
    for task in TASKS:
        row = counts[task]
        n0, n1, n = row["n0"], row["n1"], row["N"]
        if any(type(x) is not int for x in (n0, n1, n)):
            raise TypeError("Counts must be integers")
        if n <= 0 or not (0 <= n0 <= n and 0 <= n1 <= n):
            raise ValueError(f"Invalid counts for {task}: {row}")
        clean[task] = {"n0": n0, "n1": n1, "N": n}
        per_task[task] = {
            **clean[task], "accuracy0": n0 / n, "accuracy1": n1 / n,
            "net_correct_drop": n0 - n1,
            "accuracy_drop_pp": 100 * (n0 - n1) / n,
            "signed_relative_drop": (n0 - n1) / n0 if n0 else None,
            "direction": "drop" if n1 < n0 else "improvement" if n1 > n0 else "unchanged",
        }
    n0 = clean[target]["n0"]
    d = n0 - clean[target]["n1"]
    b = sum(clean[t]["n0"] for t in TASKS if t != target)
    a = sum(abs(clean[t]["n0"] - clean[t]["n1"]) for t in TASKS if t != target)
    out = {"target": target, "counts": clean, "per_task": per_task,
           "target_net_drop": d, "target_original_correct": n0,
           "control_abs_net_change": a, "control_original_correct": b,
           "L": None, "C": None, "R": None, "L_minus_C": None,
           "hn_gate": False, "hn_group_pass": False, "joint_quality_goal_met": False,
           "ratio_status": "undefined_original_denominator", "defined": bool(n0 and b)}
    if not out["defined"]:
        return out
    loss, control = Fraction(d, n0), Fraction(a, b)
    out.update(L=float(loss), C=float(control), L_minus_C=float(loss-control),
               L_fraction=str(loss), C_fraction=str(control),
               joint_quality_goal_met=5*d >= 2*n0,
               hn_gate=5*d >= 2*n0 and 10*(d*b-a*n0) > n0*b)
    out["hn_group_pass"] = out["hn_gate"]
    if a:
        out.update(R=float(loss/control), R_fraction=str(loss/control), ratio_status="finite")
    elif d > 0:
        out.update(R=math.inf, R_fraction=None, ratio_status="zero_control_net_change")
    else:
        out.update(ratio_status="no_positive_target_effect")
    return out


def measure(rows, target=TARGET, baseline_key="baseline", edited_key="edited"):
    return effect(count_rows(rows, baseline_key, edited_key), target)


def compare(direct, joint):
    """Compare exact ratios without treating an undefined R as zero."""
    if direct["target"] != joint["target"]:
        raise ValueError("Targets differ")
    for task in TASKS:
        d, j = direct["counts"][task], joint["counts"][task]
        if (d["n0"], d["N"]) != (j["n0"], j["N"]):
            raise ValueError("Direct/joint must share the same unadapted Base baseline")
    out = {"G": None, "G_fraction": None, "delta_L": None, "control_reduction": None,
           "delta_R": None, "ratio_multiple": None, "ratio_advantage": False,
           "comparison_status": "undefined_original_denominator"}
    if not (direct["defined"] and joint["defined"]):
        return out
    ld, cd = Fraction(direct["L_fraction"]), Fraction(direct["C_fraction"])
    lj, cj = Fraction(joint["L_fraction"]), Fraction(joint["C_fraction"])
    g = lj*cd - ld*cj
    out.update(G=float(g), G_fraction=str(g), delta_L=float(lj-ld),
               control_reduction=float(cd-cj), joint_quality_goal_met=joint["joint_quality_goal_met"])
    if not cd and not cj:
        out["comparison_status"] = "both_zero_control_no_ratio_advantage"
    elif direct["R"] is None or joint["R"] is None:
        out["comparison_status"] = "no_positive_target_effect_ratio_unavailable"
    elif not cj:
        out.update(ratio_advantage=lj > 0, comparison_status="joint_observed_infinity")
    elif not cd:
        out["comparison_status"] = "direct_observed_infinity"
    else:
        rd, rj = ld/cd, lj/cj
        out.update(ratio_advantage=rj > rd, delta_R=float(rj-rd),
                   ratio_multiple=float(rj/rd) if rd > 0 else None,
                   comparison_status="finite")
    return out


def paired_bootstrap(rows, target=TARGET, *, resamples=5000, seed=42):
    """Percentile 95% CIs, paired across all five outputs, stratified by task.

    Undefined draws are counted and omitted only from the corresponding interval;
    questions themselves are never removed from the resampling universe.
    """
    import numpy as np
    if resamples != 5000 or seed != 42:
        raise ValueError("Formal protocol fixes bootstrap resamples=5000 and seed=42")
    rows = list(rows)
    columns = ("base", "direct", "joint", "hn_base", "hn_edited")
    # Also validates IDs, task names, strict booleans and all-task coverage.
    count_rows(rows, "base", "direct")
    rng = np.random.default_rng(seed)
    sums = {}
    for task in TASKS:
        values = np.asarray([[_correct(r[c]) for c in columns]
                             for r in rows if r.get("task_type", r.get("task")) == task], dtype=np.int64)
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        sums[task] = values[indices].sum(axis=1)
    statistics = {}
    for label, baseline, edited in (("direct", 0, 1), ("joint", 0, 2), ("hn", 3, 4)):
        denom = sums[target][:, baseline]
        ctl_denom = sum(sums[t][:, baseline] for t in TASKS if t != target)
        ctl_num = sum(np.abs(sums[t][:, baseline]-sums[t][:, edited]) for t in TASKS if t != target)
        valid = (denom > 0) & (ctl_denom > 0)
        loss = np.full(resamples, np.nan)
        control = np.full(resamples, np.nan)
        loss[valid] = (denom[valid]-sums[target][valid, edited])/denom[valid]
        control[valid] = ctl_num[valid]/ctl_denom[valid]
        statistics[f"{label}_L"], statistics[f"{label}_C"] = loss, control
    statistics["G"] = statistics["joint_L"]*statistics["direct_C"] - statistics["direct_L"]*statistics["joint_C"]
    intervals = {}
    for key, values in statistics.items():
        valid = values[np.isfinite(values)]
        intervals[key] = {
            "low": float(np.quantile(valid, .025)) if len(valid) else None,
            "high": float(np.quantile(valid, .975)) if len(valid) else None,
            "valid_resamples": int(len(valid)), "undefined_resamples": resamples-int(len(valid)),
        }
    return {"resamples": resamples, "seed": seed, "paired": True, "stratified_by": "task",
            "confidence": .95, "method": "percentile", "intervals": intervals,
            "interval_support": intervals["G"]["low"] is not None and intervals["G"]["low"] > 0}


def summarize(rows, target=TARGET, *, formal=True):
    rows = list(rows)
    direct = measure(rows, target, "base", "direct")
    joint = measure(rows, target, "base", "joint")
    hn = measure(rows, target, "hn_base", "hn_edited")
    if formal and any(c["N"] != 100 for c in direct["counts"].values()):
        raise ValueError("Formal confirmation must include exactly 100 questions/task")
    comparison = compare(direct, joint)
    bootstrap = paired_bootstrap(rows, target)
    stages = {"hn_group_pass": hn["hn_gate"],
              "ratio_advantage": comparison["ratio_advantage"],
              "joint_quality_goal_met": joint["joint_quality_goal_met"],
              "interval_support": bootstrap["interval_support"]}
    stages["chain_positive"] = all(stages[k] for k in ("hn_group_pass", "ratio_advantage", "joint_quality_goal_met"))
    return {"protocol": "lean_atomic_hn_readout_fusion_v1", "target": target,
            "scope": "fresh_confirmation" if formal else "development_only",
            "direct": direct, "joint": joint, "hn": hn, "comparison": comparison,
            "bootstrap": bootstrap, "stages": stages}


def json_safe(value):
    """JSON does not have Infinity; preserve its protocol meaning as a string."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "+inf" if value == math.inf else "-inf" if value == -math.inf else None
    return value


def markdown_report(summary):
    """Main report: signed task changes, aggregate C, ratio gaps and HN gate."""
    def pct(x):
        return "undefined" if x is None else f"{100*x:.2f}%"
    def number(x):
        return "undefined" if x is None else "+inf" if x in (math.inf, "+inf") else f"{x:.4f}"
    lines = [f"S_and fusion v1 — {summary['scope']}", "",
             "| 组 | 任务 | 原正确/N | 干预正确/N | 原准确率 | 干预准确率 | 有符号相对下降 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for key, name in (("direct", "直接"), ("joint", "联合")):
        for task, row in summary[key]["per_task"].items():
            lines.append(f"| {name} | {task} | {row['n0']}/{row['N']} | {row['n1']}/{row['N']} | {pct(row['accuracy0'])} | {pct(row['accuracy1'])} | {pct(row['signed_relative_drop'])} |")
    lines += ["", "负下降表示提升；C 为各控制任务净正确数变化绝对值之和/控制原正确数之和。", "",
              "| 组 | 目标下降L | 控制合计变化C | 比例R |", "|---|---:|---:|---:|"]
    for key, name in (("direct", "直接"), ("joint", "联合")):
        v = summary[key]
        lines.append(f"| {name} | {pct(v['L'])} | {pct(v['C'])} | {number(v['R'])} |")
    c = summary["comparison"]
    lines += ["", f"目标下降差 L_J−L_D：{pct(c['delta_L'])}（百分点）；控制变化减少 C_D−C_J：{pct(c['control_reduction'])}（百分点）。",
              f"比例差：{number(c['delta_R'])}；比例倍数：{number(c['ratio_multiple'])}；状态：{c['comparison_status']}。", "",
              "| HN 功能组 | 原正确/N | 干预正确/N | 有符号相对下降 |", "|---|---:|---:|---:|"]
    for task, row in summary["hn"]["per_task"].items():
        lines.append(f"| {task} | {row['n0']}/{row['N']} | {row['n1']}/{row['N']} | {pct(row['signed_relative_drop'])} |")
    h = summary["hn"]
    lines += ["", f"HN：L={pct(h['L'])}，C={pct(h['C'])}，L−C={pct(h['L_minus_C'])}，门槛通过={h['hn_gate']}。", "",
              "| 配对分层bootstrap统计量 | 点估计 | 95%区间 | 未定义重采样数 |", "|---|---:|---|---:|"]
    points = {"direct_L": summary["direct"]["L"], "direct_C": summary["direct"]["C"],
              "joint_L": summary["joint"]["L"], "joint_C": summary["joint"]["C"],
              "hn_L": h["L"], "hn_C": h["C"], "G": c["G"]}
    for key, row in summary["bootstrap"]["intervals"].items():
        lines.append(f"| {key} | {number(points[key])} | [{number(row['low'])}, {number(row['high'])}] | {row['undefined_resamples']} |")
    lines += ["", "阶段状态："+json.dumps(summary["stages"], ensure_ascii=False, sort_keys=True), ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rows", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    result = summarize(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "formal_metrics.json").write_text(json.dumps(json_safe(result), indent=2, sort_keys=True, allow_nan=False)+"\n")
    (args.output / "formal_metrics.md").write_text(markdown_report(result))


if __name__ == "__main__":
    main()

"""Exact protocol metrics and task-stratified, question-paired uncertainty.

No model output is excluded.  ``C`` uses absolute *net count* changes, never
correctness flips. Fractions decide gates; floating point is presentation only.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Mapping

TASKS = ("S_and", "S_or_l", "S_or_r", "S_id")
TARGET = "S_or_r"
CONDITIONS = ("baseline_hn", "edited_hn", "baseline_base", "direct", "joint")


def _fraction(value: Fraction | None):
    return None if value is None else {"numerator": value.numerator, "denominator": value.denominator}


def _read_fraction(value):
    return None if value is None else Fraction(value["numerator"], value["denominator"])


def effect(counts: Mapping[str, Mapping], target: str = TARGET) -> dict:
    """Counts are ``{task: {n0: int, n1: int, N: int}}`` for all four tasks."""
    if set(counts) != set(TASKS) or target not in TASKS:
        raise ValueError("exactly the four protocol tasks are required")
    tasks = {}
    for task in TASKS:
        n0, n1, total = (counts[task][name] for name in ("n0", "n1", "N"))
        if any(type(v) is not int for v in (n0, n1, total)) or not 0 <= min(n0, n1) <= max(n0, n1) <= total or total < 1:
            raise ValueError(f"invalid correct counts: {task}")
        signed = Fraction(n0 - n1, n0) if n0 else None
        tasks[task] = {"n0": n0, "n1": n1, "N": total,
                       "accuracy0": n0 / total, "accuracy1": n1 / total,
                       "net_correct_drop": n0 - n1,
                       "signed_relative_drop": float(signed) if signed is not None else None,
                       "signed_relative_drop_exact": _fraction(signed)}
    n0 = tasks[target]["n0"]
    d = tasks[target]["net_correct_drop"]
    b = sum(tasks[t]["n0"] for t in TASKS if t != target)
    a = sum(abs(tasks[t]["net_correct_drop"]) for t in TASKS if t != target)
    L = Fraction(d, n0) if n0 else None
    C = Fraction(a, b) if b else None
    defined = L is not None and C is not None
    R, ratio_status = None, "undefined_original_denominator"
    if defined:
        if C:
            R, ratio_status = float(L / C), "finite"
        elif L > 0:
            R, ratio_status = "+inf", "zero_control_net_change"
        else:
            ratio_status = "no_positive_target_effect"
    return {"target": target, "tasks": tasks,
            "target_net_drop": d, "target_original_correct": n0,
            "control_absolute_net_change": a, "control_original_correct": b,
            "L": float(L) if L is not None else None,
            "C": float(C) if C is not None else None,
            "R": R, "ratio_status": ratio_status,
            "L_exact": _fraction(L), "C_exact": _fraction(C),
            "R_exact": _fraction(L / C) if defined and C else None,
            "L_minus_C": float(L - C) if defined else None,
            "quality_goal_met": bool(defined and 5 * d >= 2 * n0),
            "hn_group_pass": bool(defined and 5 * d >= 2 * n0
                                  and 10 * (d * b - a * n0) > n0 * b),
            "defined": defined}


def compare(direct: dict, joint: dict) -> dict:
    """Compare same-baseline effects without inf-inf or epsilon denominators."""
    if direct["target"] != joint["target"]:
        raise ValueError("target mismatch")
    for t in TASKS:
        if any(direct["tasks"][t][k] != joint["tasks"][t][k] for k in ("n0", "N")):
            raise ValueError("direct/joint must share the same original Base baseline")
    ld, cd = (_read_fraction(direct[k]) for k in ("L_exact", "C_exact"))
    lj, cj = (_read_fraction(joint[k]) for k in ("L_exact", "C_exact"))
    if None in (ld, cd, lj, cj):
        return {"ratio_advantage": False, "ratio_comparison_status": "undefined_original_denominator",
                "G": None, "G_exact": None, "delta_L": None, "control_reduction": None,
                "delta_R": None, "ratio_multiple": None}
    gain = lj * cd - ld * cj
    delta_r = multiple = None
    advantage = False
    if cd == 0 and cj == 0:
        status = "both_zero_control_net_change_no_ratio_advantage"
    elif ld <= 0 or lj <= 0:
        status = "no_positive_target_effect_not_ratio_selection_advantage"
    elif cj == 0:
        status, advantage = "joint_infinite_direct_finite", True
    elif cd == 0:
        status = "direct_infinite_joint_finite"
    else:
        status, advantage = "both_finite", gain > 0
    if cd > 0 and cj > 0:
        rd, rj = ld / cd, lj / cj
        delta_r = float(rj - rd)
        if rd > 0:
            multiple = float(rj / rd)
    return {"ratio_advantage": bool(advantage), "ratio_comparison_status": status,
            "G": float(gain), "G_exact": _fraction(gain),
            "delta_L": float(lj - ld), "control_reduction": float(cd - cj),
            "delta_R": delta_r, "ratio_multiple": multiple}


def align_results(results: Mapping[str, list[dict]], *, expected_per_task: int | None = None):
    if set(results) != set(CONDITIONS):
        raise ValueError(f"expected conditions {CONDITIONS}")
    lookup, ids = {}, None
    for condition in CONDITIONS:
        by_id = {}
        for row in results[condition]:
            if row["question_id"] in by_id:
                raise ValueError(f"duplicate question: {condition}/{row['question_id']}")
            if row["task_type"] not in TASKS or type(row["correct"]) is not bool:
                raise ValueError("task/correctness schema differs; failed outputs must be correct=false")
            by_id[row["question_id"]] = row
        if ids is None:
            ids = set(by_id)
        elif set(by_id) != ids:
            raise ValueError("all conditions must evaluate the same questions")
        lookup[condition] = by_id
    grouped = {task: [] for task in TASKS}
    for qid in sorted(ids):
        task = lookup[CONDITIONS[0]][qid]["task_type"]
        if any(lookup[c][qid]["task_type"] != task for c in CONDITIONS):
            raise ValueError("question task differs across conditions")
        grouped[task].append([lookup[c][qid]["correct"] for c in CONDITIONS])
    if any(not grouped[t] or (expected_per_task is not None and len(grouped[t]) != expected_per_task) for t in TASKS):
        raise ValueError("wrong per-task denominator")
    return grouped


def paired_summary(results: Mapping[str, list[dict]], target: str = TARGET,
                   bootstrap: int = 5000, seed: int = 42,
                   expected_per_task: int | None = None) -> dict:
    """Pair all five conditions by question ID; resample within every task."""
    grouped = align_results(results, expected_per_task=expected_per_task)
    import numpy as np
    arrays = {task: np.asarray(rows, dtype=np.int64) for task, rows in grouped.items()}
    pairs = {"hn": (0, 1), "direct": (2, 3), "joint": (2, 4)}
    reports = {}
    for name, (base, edited) in pairs.items():
        reports[name] = effect({t: {"n0": int(a[:, base].sum()),
                                    "n1": int(a[:, edited].sum()), "N": len(a)}
                                for t, a in arrays.items()}, target)
    comparison = compare(reports["direct"], reports["joint"])
    if bootstrap < 1:
        raise ValueError("bootstrap must be positive")
    rng = np.random.default_rng(seed)
    draws = {t: a[rng.integers(0, len(a), size=(bootstrap, len(a)))].sum(axis=1)
             for t, a in arrays.items()}
    samples = {}
    for name, (base, edited) in pairs.items():
        n0 = draws[target][:, base].astype(float)
        b = sum(draws[t][:, base] for t in TASKS if t != target).astype(float)
        d = draws[target][:, base] - draws[target][:, edited]
        a = sum(np.abs(draws[t][:, base] - draws[t][:, edited]) for t in TASKS if t != target)
        with np.errstate(divide="ignore", invalid="ignore"):
            samples[name + "_L"] = np.where(n0 > 0, d / n0, np.nan)
            samples[name + "_C"] = np.where(b > 0, a / b, np.nan)
    samples["G"] = samples["joint_L"] * samples["direct_C"] - samples["direct_L"] * samples["joint_C"]
    intervals = {}
    for name, sample in samples.items():
        valid = sample[np.isfinite(sample)]
        intervals[name] = {"lower": float(np.quantile(valid, .025)) if len(valid) else None,
                           "upper": float(np.quantile(valid, .975)) if len(valid) else None,
                           "valid_resamples": len(valid), "undefined_resamples": bootstrap - len(valid)}
    states = {"hn_group_pass": reports["hn"]["hn_group_pass"],
              "ratio_advantage": comparison["ratio_advantage"],
              "joint_quality_goal_met": reports["joint"]["quality_goal_met"],
              "interval_support": intervals["G"]["lower"] is not None and intervals["G"]["lower"] > 0}
    states["chain_positive"] = all(states[k] for k in ("hn_group_pass", "ratio_advantage", "joint_quality_goal_met"))
    return {"target": target, **reports, "comparison": comparison, "states": states,
            "bootstrap": {"resamples": bootstrap, "seed": seed, "paired": True,
                          "stratify": "task", "confidence": .95, "intervals": intervals}}


def markdown_tables(summary: dict, K: int) -> str:
    def pct(v):
        return "undefined" if v is None else f"{100 * v:.2f}%"
    def ratio(v):
        return "undefined" if v is None else (v if isinstance(v, str) else f"{v:.6g}")
    target = summary["target"]
    lines = ["正式结果仅来自冻结后新生成的同一批100题/任务。", "",
             f"| 方法 | K | {target}正确数/N | 准确率前→后 | 目标相对下降L | 控制合计变化C | R |",
             "|---|---:|---|---|---:|---:|---:|"]
    for name, label in (("direct", "直接"), ("joint", "联合")):
        row, t = summary[name], summary[name]["tasks"][target]
        lines.append(f"| {label} | {K} | {t['n0']}→{t['n1']}/{t['N']} | {pct(t['accuracy0'])}→{pct(t['accuracy1'])} | {pct(row['L'])} | {pct(row['C'])} | {ratio(row['R'])} |")
    lines += ["", "| 控制任务 | 原正确/N | 直接后正确/N | 直接相对净下降 | 联合后正确/N | 联合相对净下降 |",
              "|---|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        if task == target:
            continue
        d, j = summary["direct"]["tasks"][task], summary["joint"]["tasks"][task]
        lines.append(f"| {task} | {d['n0']}/{d['N']} | {d['n1']}/{d['N']} | {pct(d['signed_relative_drop'])} | {j['n1']}/{j['N']} | {pct(j['signed_relative_drop'])} |")
    gap = summary["comparison"]
    points = lambda v: "undefined" if v is None else f"{100*v:.2f}个百分点"
    lines += ["", "控制净下降为负表示提升；C对各控制任务的净正确数变化分别取绝对值后合计。",
              f"目标下降差：{points(gap['delta_L'])}；控制变化减少：{points(gap['control_reduction'])}；比例增益：{ratio(gap['delta_R'])}；倍数：{ratio(gap['ratio_multiple'])}。",
              f"比例状态：{gap['ratio_comparison_status']}。", "",
              "| H_S_or_r功能确认任务 | 原正确 | 干预正确 | N |",
              "|---|---:|---:|---:|"]
    for task in TASKS:
        t = summary["hn"]["tasks"][task]
        lines.append(f"| {task} | {t['n0']} | {t['n1']} | {t['N']} |")
    h, ci = summary["hn"], summary["bootstrap"]["intervals"]["G"]
    lines += ["", f"HN：L_H={pct(h['L'])}，C_H={pct(h['C'])}，差值={points(h['L_minus_C'])}；功能门槛={h['hn_group_pass']}。",
              f"G={gap['G']}，任务分层配对bootstrap 95%区间=[{ci['lower']}, {ci['upper']}]。",
              "状态：" + "; ".join(f"{k}={v}" for k, v in summary["states"].items()) + "。"]
    return "\n".join(lines) + "\n"

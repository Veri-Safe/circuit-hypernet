from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .identity import sha256_file
from .traces import write_json_atomic


def _resolve_repo_artifact_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate.resolve()
    root_value = os.environ.get("HC_ROOT")
    if root_value:
        parts = candidate.parts
        for marker in ("outputs", "logs"):
            if marker not in parts:
                continue
            rebased = Path(root_value).resolve().joinpath(*parts[parts.index(marker) :])
            if rebased.exists():
                return rebased
    return candidate.resolve()


def _retain_snapshot(source: Path, destination: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if not (source / "COMPLETE").is_file():
        raise ValueError(f"Selected raw snapshot is incomplete: {source}")
    if (destination / "COMPLETE").is_file():
        if sha256_file(source / "transcoder.safetensors") != sha256_file(
            destination / "transcoder.safetensors"
        ):
            raise ValueError(f"Retained snapshot does not match selected source: {destination}")
        return destination
    if destination.exists():
        raise FileExistsError(f"Incomplete retained snapshot already exists: {destination}")
    temporary = destination.parent / f".{destination.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError(f"Stale retained-snapshot temporary path: {temporary}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    def link_or_copy(source_file: str, destination_file: str) -> str:
        try:
            os.link(source_file, destination_file)
            return destination_file
        except OSError:
            return shutil.copy2(source_file, destination_file)

    try:
        shutil.copytree(source, temporary, copy_function=link_or_copy)
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _load_rows(path: str | Path) -> list[dict[str, Any]]:
    resolved = _resolve_repo_artifact_path(path)
    rows = [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Validation row file is empty: {resolved}")
    return rows


def _paired_bootstrap_se(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    samples: int,
) -> float:
    if left.shape != right.shape or left.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be aligned vectors")
    if left.size < 2:
        return 0.0
    differences = left - right
    rng = np.random.default_rng(seed)
    replicate_means = np.empty(samples, dtype=np.float64)
    batch = 256
    for start in range(0, samples, batch):
        count = min(batch, samples - start)
        indices = rng.integers(0, differences.size, size=(count, differences.size))
        replicate_means[start : start + count] = differences[indices].mean(axis=1)
    return float(replicate_means.std(ddof=1))


def _candidate_values(candidate: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in candidate["rows"]], dtype=np.float64)


def _progress(candidate: dict[str, Any]) -> int:
    return int(candidate["summary"]["snapshot_manifest"]["progress_percent"])


def select_candidate(
    validation_summaries: Iterable[str | Path],
    *,
    output_path: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 24,
    retention_root: str | Path | None = None,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for path_value in validation_summaries:
        path = Path(path_value).resolve()
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("format") != "nvarc_transcoder_validation_v1":
            raise ValueError(f"Unsupported validation summary: {path}")
        if summary.get("status") != "complete":
            raise ValueError(f"Validation candidate is incomplete: {path}")
        rows = _load_rows(summary["row_metrics"])
        candidates.append({"path": str(path), "summary": summary, "rows": rows})
    if len(candidates) != 11:
        raise ValueError(f"Selection requires p000 plus p010..p100; got {len(candidates)}")

    layer_values = {int(candidate["summary"]["layer"]) for candidate in candidates}
    if len(layer_values) != 1:
        raise ValueError(f"Validation summaries span multiple layers: {layer_values}")
    layer = next(iter(layer_values))
    identities = {
        (
            candidate["summary"]["config_sha256"],
            candidate["summary"]["base_identity_sha256"],
            candidate["summary"]["validation_trace_manifest_sha256"],
        )
        for candidate in candidates
    }
    if len(identities) != 1:
        raise ValueError("Validation candidates do not share one frozen evaluation identity")
    row_indices = [row["dataset_index"] for row in candidates[0]["rows"]]
    for candidate in candidates[1:]:
        if [row["dataset_index"] for row in candidate["rows"]] != row_indices:
            raise ValueError("Validation rows are not paired in identical dataset order")

    by_progress = {_progress(candidate): candidate for candidate in candidates}
    if set(by_progress) != set(range(0, 101, 10)):
        raise ValueError(f"Candidate progress coverage mismatch: {sorted(by_progress)}")
    public = by_progress[0]
    guardrail: dict[int, dict[str, Any]] = {}
    eligible: list[dict[str, Any]] = []
    for progress, candidate in sorted(by_progress.items()):
        checks: dict[str, Any] = {}
        passes = True
        for metric_offset, metric in enumerate(("replacement_kl", "fvu")):
            values = _candidate_values(candidate, metric)
            reference = _candidate_values(public, metric)
            standard_error = _paired_bootstrap_se(
                values,
                reference,
                seed=seed + layer * 1009 + progress * 17 + metric_offset,
                samples=bootstrap_samples,
            )
            regression = float(values.mean() - reference.mean())
            metric_passes = progress == 0 or regression <= standard_error
            checks[metric] = {
                "regression_from_p000": regression,
                "paired_bootstrap_se": standard_error,
                "passes": metric_passes,
            }
            passes &= metric_passes
        checks["passes"] = passes
        guardrail[progress] = checks
        if passes:
            eligible.append(candidate)
    if not eligible:
        raise RuntimeError("No candidate passes the public-step-0 validation guardrails")

    kl_best = min(eligible, key=lambda item: float(item["summary"]["replacement_kl"]))
    kl_best_values = _candidate_values(kl_best, "replacement_kl")
    kl_retained = []
    kl_rule: dict[int, Any] = {}
    for candidate in eligible:
        values = _candidate_values(candidate, "replacement_kl")
        standard_error = _paired_bootstrap_se(
            values,
            kl_best_values,
            seed=seed + layer * 2027 + _progress(candidate),
            samples=bootstrap_samples,
        )
        gap = float(values.mean() - kl_best_values.mean())
        retained = gap <= standard_error
        kl_rule[_progress(candidate)] = {
            "gap_from_best": gap,
            "paired_bootstrap_se": standard_error,
            "retained": retained,
        }
        if retained:
            kl_retained.append(candidate)

    fvu_best = min(kl_retained, key=lambda item: float(item["summary"]["fvu"]))
    fvu_best_values = _candidate_values(fvu_best, "fvu")
    final_retained = []
    fvu_rule: dict[int, Any] = {}
    for candidate in kl_retained:
        values = _candidate_values(candidate, "fvu")
        standard_error = _paired_bootstrap_se(
            values,
            fvu_best_values,
            seed=seed + layer * 4099 + _progress(candidate),
            samples=bootstrap_samples,
        )
        gap = float(values.mean() - fvu_best_values.mean())
        retained = gap <= standard_error
        fvu_rule[_progress(candidate)] = {
            "gap_from_best": gap,
            "paired_bootstrap_se": standard_error,
            "retained": retained,
        }
        if retained:
            final_retained.append(candidate)

    ranked = sorted(
        final_retained,
        key=lambda item: (float(item["summary"]["l0_median"]), _progress(item)),
    )
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None

    def reference(
        candidate: dict[str, Any] | None, *, role: str
    ) -> dict[str, Any] | None:
        if candidate is None:
            return None
        original = Path(candidate["summary"]["snapshot"]).resolve()
        retained = (
            _retain_snapshot(original, Path(retention_root) / role)
            if retention_root is not None
            else original
        )
        return {
            "progress_percent": _progress(candidate),
            "snapshot": str(retained),
            "original_candidate_snapshot": str(original),
            "validation_summary": candidate["path"],
            "replacement_kl": candidate["summary"]["replacement_kl"],
            "fvu": candidate["summary"]["fvu"],
            "l0_median": candidate["summary"]["l0_median"],
        }

    result = {
        "format": "nvarc_transcoder_selection_v1",
        "layer": layer,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "best": reference(best, role="raw_best"),
        "runner_up": reference(runner_up, role="raw_runner_up"),
        "guardrail": {str(key): value for key, value in guardrail.items()},
        "kl_one_se": {str(key): value for key, value in kl_rule.items()},
        "fvu_one_se": {str(key): value for key, value in fvu_rule.items()},
        "final_retained_progress": [_progress(item) for item in ranked],
        "status": "complete",
    }
    write_json_atomic(output_path, result)
    return result


def compare_selected_runs(
    selection_paths: Iterable[str | Path],
    *,
    output_path: str | Path,
    bootstrap_samples: int = 2000,
    seed: int = 24,
) -> dict[str, Any]:
    candidates = []
    for selection_path in selection_paths:
        selection_file = Path(selection_path).resolve()
        selection = json.loads(selection_file.read_text(encoding="utf-8"))
        if selection.get("format") != "nvarc_transcoder_selection_v1" or selection.get(
            "status"
        ) != "complete":
            raise ValueError(f"Audit selection is incomplete: {selection_file}")
        selected = selection.get("best")
        if not isinstance(selected, dict):
            raise ValueError(f"Audit selection has no best candidate: {selection_file}")
        summary_path = _resolve_repo_artifact_path(selected["validation_summary"])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "selection": str(selection_file),
                "summary_path": str(summary_path),
                "summary": summary,
                "rows": _load_rows(summary["row_metrics"]),
            }
        )
    if len(candidates) < 2:
        raise ValueError("Audit comparison requires at least two isolated selected runs")
    layers = {int(candidate["summary"]["layer"]) for candidate in candidates}
    if len(layers) != 1:
        raise ValueError("Audit selections span multiple layers")
    layer = next(iter(layers))
    identities = {
        (
            candidate["summary"].get("config_sha256"),
            candidate["summary"].get("base_identity_sha256"),
            candidate["summary"].get("validation_trace_manifest_sha256"),
        )
        for candidate in candidates
    }
    if len(identities) != 1 or any(value is None for value in next(iter(identities))):
        raise ValueError("Audit selections do not share one frozen evaluation identity")
    config_sha256, base_identity_sha256, validation_manifest_sha256 = next(
        iter(identities)
    )
    row_orders = [
        [int(row["dataset_index"]) for row in candidate["rows"]]
        for candidate in candidates
    ]
    if any(order != row_orders[0] for order in row_orders[1:]):
        raise ValueError("Audit selections are not evaluated on paired validation rows")

    retained = candidates
    rules: dict[str, Any] = {}
    for metric_index, metric in enumerate(("replacement_kl", "fvu")):
        best = min(retained, key=lambda item: float(item["summary"][metric]))
        best_values = _candidate_values(best, metric)
        next_retained = []
        metric_rule = {}
        for candidate_index, candidate in enumerate(retained):
            values = _candidate_values(candidate, metric)
            standard_error = _paired_bootstrap_se(
                values,
                best_values,
                seed=seed + layer * 8191 + metric_index * 1009 + candidate_index,
                samples=bootstrap_samples,
            )
            gap = float(values.mean() - best_values.mean())
            keep = gap <= standard_error
            run_id = str(candidate["summary"]["snapshot_manifest"]["run_id"])
            metric_rule[run_id] = {
                "gap_from_best": gap,
                "paired_bootstrap_se": standard_error,
                "retained": keep,
            }
            if keep:
                next_retained.append(candidate)
        rules[metric] = metric_rule
        retained = next_retained
    ranked = sorted(
        retained,
        key=lambda item: (
            float(item["summary"]["l0_median"]),
            int(item["summary"]["snapshot_manifest"]["progress_percent"]),
            str(item["summary"]["snapshot_manifest"]["run_id"]),
        ),
    )
    winner = ranked[0]
    winner_manifest = winner["summary"]["snapshot_manifest"]
    result = {
        "format": "nvarc_transcoder_audit_run_comparison_v1",
        "layer": layer,
        "config_sha256": config_sha256,
        "base_identity_sha256": base_identity_sha256,
        "validation_trace_manifest_sha256": validation_manifest_sha256,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "recommended": {
            "run_id": winner_manifest["run_id"],
            "learning_rate": winner_manifest["learning_rate"],
            "microbatch_tokens": winner_manifest["microbatch_tokens"],
            "matrix_compute": winner_manifest["matrix_compute"],
            "validation_summary": winner["summary_path"],
        },
        "one_se_rules": rules,
        "retained_run_ids": [
            item["summary"]["snapshot_manifest"]["run_id"] for item in ranked
        ],
        "status": "complete",
    }
    write_json_atomic(output_path, result)
    return result

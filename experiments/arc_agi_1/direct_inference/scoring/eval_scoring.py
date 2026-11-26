from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from nvarc_qwen3.arc_eval_protocol import (
    INTERNAL_PROTOCOL_VERSION,
    build_submission,
    score_attempts,
)


Grid = list[list[int]]


def _attempt_maps(
    records: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, int], list[Grid]],
    dict[tuple[str, int], list[Grid]],
    dict[tuple[str, int], list[Grid]],
]:
    canonical: dict[tuple[str, int], list[Grid]] = {}
    selected: dict[tuple[str, int], list[Grid]] = {}
    oracle_views: dict[tuple[str, int], list[Grid]] = {}
    for record in records:
        key = (str(record["task_id"]), int(record["test_index"]))
        prediction = record.get("canonical_prediction")
        canonical[key] = [prediction] if prediction is not None else []
        selected[key] = [
            candidate["grid"]
            for candidate in record.get("selected_candidates", [])
            if candidate.get("grid") is not None
        ]
        oracle_views[key] = [
            view["grid"]
            for view in record.get("selection_views", [])
            if view.get("grid") is not None
        ]
    return canonical, selected, oracle_views


def score_grouped_predictions(
    challenges: Mapping[str, Mapping[str, Sequence[Any]]],
    solutions: Mapping[str, Sequence[Grid]],
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Grid]]]]:
    """Score already-grouped predictions against the separately loaded gold file."""

    canonical, selected, oracle_views = _attempt_maps(records)
    canonical_score = score_attempts(challenges, solutions, canonical, top_k=1)
    selected_at_1 = score_attempts(challenges, solutions, selected, top_k=1)
    selected_at_2 = score_attempts(challenges, solutions, selected, top_k=2)
    oracle_score = score_attempts(challenges, solutions, oracle_views, top_k=16)

    total_views = sum(
        int(record.get("view_summary", {}).get("view_count", 0)) for record in records
    )
    valid_views = sum(
        int(record.get("view_summary", {}).get("valid_view_count", 0)) for record in records
    )
    unique_counts = [
        int(record.get("view_summary", {}).get("unique_candidates", 0)) for record in records
    ]
    metrics = {
        "protocol_version": INTERNAL_PROTOCOL_VERSION,
        "primary_metric": "selected_exact@2",
        "canonical_exact@1": canonical_score["accuracy"],
        "canonical_task_score@1": canonical_score["task_score"],
        "selected_exact@1": selected_at_1["accuracy"],
        "selected_task_score@1": selected_at_1["task_score"],
        "selected_exact@2": selected_at_2["accuracy"],
        "selected_task_score@2": selected_at_2["task_score"],
        "oracle_view@16": oracle_score["accuracy"],
        "oracle_task_score@16": oracle_score["task_score"],
        "valid_grid_rate": valid_views / total_views if total_views else 0.0,
        "valid_view_count": valid_views,
        "selection_view_count": total_views,
        "mean_unique_candidates": (
            sum(unique_counts) / len(unique_counts) if unique_counts else 0.0
        ),
        "score_details": {
            "canonical_exact@1": canonical_score,
            "selected_exact@1": selected_at_1,
            "selected_exact@2": selected_at_2,
            "oracle_view@16": oracle_score,
        },
    }
    return metrics, build_submission(challenges, selected)

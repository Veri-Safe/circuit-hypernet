from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from nvarc_qwen3.arc_eval_protocol import (
    ViewPrediction,
    invert_grid,
    rank_view_predictions,
    summarize_view_counts,
)

from .eval_manifest import QueryView


def _candidate_record(candidate: Any) -> dict[str, Any]:
    return {
        "grid": candidate.grid,
        "votes": candidate.votes,
        "mean_normalized_logprob": candidate.mean_normalized_logprob,
    }


def group_query_predictions(
    query_views: Sequence[QueryView],
    query_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_index = {int(record["query_index"]): record for record in query_records}
    if len(by_index) != len(query_views):
        raise ValueError(
            f"Expected {len(query_views)} unique query predictions, found {len(by_index)}"
        )
    grouped: dict[tuple[str, int], list[tuple[QueryView, Mapping[str, Any]]]] = defaultdict(list)
    for view in query_views:
        try:
            prediction = by_index[view.query_index]
        except KeyError as error:
            raise ValueError(f"Missing query prediction {view.query_index}") from error
        grouped[(view.task_id, view.test_index)].append((view, prediction))

    results: list[dict[str, Any]] = []
    for (task_id, test_index), items in sorted(grouped.items()):
        canonical_grid = None
        selection_records: list[dict[str, Any]] = []
        selection_predictions: list[ViewPrediction] = []
        for view, raw in sorted(items, key=lambda item: item[0].query_index):
            grid = raw.get("grid")
            if view.is_canonical:
                canonical_grid = grid
                continue
            inverted = invert_grid(grid, view.spec) if grid is not None else None
            normalized_logprob = float(raw.get("normalized_logprob", float("-inf")))
            prediction = ViewPrediction(view.view_index, inverted, normalized_logprob)
            selection_predictions.append(prediction)
            selection_records.append(
                {
                    "view_index": view.view_index,
                    "view_descriptor": view.view_descriptor,
                    "grid": inverted,
                    "normalized_logprob": normalized_logprob,
                }
            )
        ranked = rank_view_predictions(selection_predictions)
        results.append(
            {
                "task_id": task_id,
                "test_index": test_index,
                "canonical_prediction": canonical_grid,
                "selected_candidates": [_candidate_record(item) for item in ranked[:2]],
                "selection_views": selection_records,
                "view_summary": summarize_view_counts(selection_predictions),
            }
        )
    return results

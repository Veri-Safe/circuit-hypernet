"""One fixed PG-slot ablation and paired answer accounting on Lean banks."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import Tensor


PG_LAYER = 11
FLAT_SLOT = 72
GRID_LAYERS = 8
GRID_SLOTS = 34


def zero_slot(output: Tensor, *, slot: int = FLAT_SLOT) -> Tensor:
    """Zero one generated-parameter token in a batched PG FF output."""
    if (
        not isinstance(output, Tensor)
        or output.ndim != 4
        or tuple(output.shape[1:3]) != (GRID_LAYERS, GRID_SLOTS)
        or not 0 <= slot < GRID_LAYERS * GRID_SLOTS
    ):
        raise ValueError("unexpected PG feed-forward output or slot")
    edited = output.clone()
    edited[:, slot // GRID_SLOTS, slot % GRID_SLOTS, :] = 0
    return edited


@contextmanager
def ablate_slot(core: Any, *, layer: int = PG_LAYER, slot: int = FLAT_SLOT) -> Iterator[dict[str, int]]:
    """Apply the edit only while generating LoRA, then remove the hook."""
    blocks = core.parameter_generator.model.blocks
    if len(blocks) != 12 or not 0 <= layer < len(blocks):
        raise ValueError("unexpected PG depth or layer")
    calls = {"count": 0}

    def hook(_module: Any, _inputs: tuple[Any, ...], output: Tensor) -> Tensor:
        calls["count"] += 1
        return zero_slot(output, slot=slot)

    handle = blocks[layer].feed_forward.register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()


def pair_predictions(
    native: Sequence[Mapping[str, Any]], edited: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, int]], list[dict[str, Any]]]:
    """Count right-to-wrong and wrong-to-right by identical test event."""
    def keyed(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            key = str(row["target_event_key"])
            if key in result:
                raise ValueError(f"duplicate event {key}")
            result[key] = row
        return result

    before, after = keyed(native), keyed(edited)
    if before.keys() != after.keys():
        raise ValueError("native and edited prediction sets differ")
    counts: dict[str, dict[str, int]] = {}
    changes: list[dict[str, Any]] = []
    for key, source in before.items():
        target = after[key]
        for field in (
            "canonical_global_index", "task_type", "expected_tactic", "seed",
            "condition", "source_event_key",
        ):
            if source[field] != target[field]:
                raise ValueError(f"paired event differs in {field}: {key}")
        task = str(source["task_type"])
        count = counts.setdefault(task, {
            "total": 0, "native_correct": 0, "edited_correct": 0,
            "right_to_wrong": 0, "wrong_to_right": 0, "tactic_changed": 0,
        })
        was, now = bool(source["correct"]), bool(target["correct"])
        count["total"] += 1
        count["native_correct"] += int(was)
        count["edited_correct"] += int(now)
        count["right_to_wrong"] += int(was and not now)
        count["wrong_to_right"] += int(not was and now)
        changed = source["predicted_tactic"] != target["predicted_tactic"]
        count["tactic_changed"] += int(changed)
        if changed or was != now:
            changes.append({
                "target_event_key": key,
                "canonical_global_index": int(source["canonical_global_index"]),
                "task_type": task,
                "seed": int(source["seed"]),
                "expected_tactic": source["expected_tactic"],
                "native_tactic": source["predicted_tactic"],
                "edited_tactic": target["predicted_tactic"],
                "native_correct": was,
                "edited_correct": now,
            })
    changes.sort(key=lambda row: (row["task_type"], row["canonical_global_index"]))
    return counts, changes

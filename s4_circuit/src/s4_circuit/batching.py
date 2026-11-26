from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from s2_hypernet.nvarc_pg.events import TokenizedRow


FIXED_ROWS = "fixed_rows_v1"
TOKEN_BUDGET = "token_budget_v1"


def _event_padded_tokens(
    rows: Sequence[TokenizedRow],
    *,
    supervised_assistant_start_index: int = 0,
) -> int:
    if not rows:
        return 0
    peak = 0
    for event_index in range(
        supervised_assistant_start_index,
        max(len(row.events) for row in rows),
    ):
        active = tuple(row for row in rows if event_index < len(row.events))
        peak = max(
            peak,
            len(active) * max(row.events[event_index].response_end for row in active),
        )
    return peak


def _event_token_totals(
    rows: Sequence[TokenizedRow],
    *,
    supervised_assistant_start_index: int = 0,
) -> tuple[int, int]:
    padded = 0
    unpadded = 0
    if not rows:
        return padded, unpadded
    for event_index in range(
        supervised_assistant_start_index,
        max(len(row.events) for row in rows),
    ):
        active = tuple(row for row in rows if event_index < len(row.events))
        response_ends = tuple(row.events[event_index].response_end for row in active)
        padded += len(active) * max(response_ends)
        unpadded += sum(response_ends)
    return padded, unpadded


def _response_token_totals(
    rows: Sequence[TokenizedRow],
    *,
    supervised_assistant_start_index: int = 0,
) -> tuple[int, int]:
    padded = 0
    unpadded = 0
    for event_index in range(
        supervised_assistant_start_index,
        max(len(row.events) for row in rows),
    ):
        active = tuple(row for row in rows if event_index < len(row.events))
        counts = tuple(row.events[event_index].response_token_count for row in active)
        padded += len(active) * max(counts)
        unpadded += sum(counts)
    return padded, unpadded


def _row_sort_key(
    row: TokenizedRow,
    *,
    supervised_assistant_start_index: int = 0,
) -> tuple[int, int, int, int, str]:
    response_ends = tuple(
        event.response_end
        for event in row.events[supervised_assistant_start_index:]
    )
    return (
        max(response_ends),
        sum(response_ends),
        len(response_ends),
        int(row.dataset_index),
        row.puzzle_name or "",
    )


@dataclass(frozen=True)
class MicrobatchPlan:
    mode: str
    microbatches: tuple[tuple[TokenizedRow, ...], ...]
    max_micro_rows: int
    padded_token_budget: int
    peak_padded_event_tokens: int
    peak_padded_condition_tokens: int
    padded_event_tokens: int
    unpadded_event_tokens: int
    padded_response_tokens: int
    unpadded_response_tokens: int
    padded_condition_tokens: int
    unpadded_condition_tokens: int

    @property
    def event_padding_ratio(self) -> float:
        return self.padded_event_tokens / max(self.unpadded_event_tokens, 1)

    @property
    def condition_padding_ratio(self) -> float:
        return self.padded_condition_tokens / max(self.unpadded_condition_tokens, 1)


def build_microbatch_plan(
    rows: Sequence[TokenizedRow],
    *,
    mode: str,
    fixed_micro_rows: int,
    max_micro_rows: int,
    padded_token_budget: int,
    supervised_assistant_start_index: int = 0,
) -> MicrobatchPlan:
    """Group one rank's fixed rows without changing membership or event semantics."""

    row_tuple = tuple(rows)
    if not row_tuple:
        raise ValueError("S4 microbatch planner requires at least one row")
    if supervised_assistant_start_index < 0:
        raise ValueError("S4 supervised assistant start index must be non-negative")
    if any(
        len(row.events) <= supervised_assistant_start_index for row in row_tuple
    ):
        raise ValueError("S4 loss protocol leaves a row with no supervised event")
    if mode == FIXED_ROWS:
        if fixed_micro_rows <= 0 or len(row_tuple) % fixed_micro_rows:
            raise ValueError("Fixed S4 microbatches must divide the local rows")
        microbatches = tuple(
            row_tuple[start : start + fixed_micro_rows]
            for start in range(0, len(row_tuple), fixed_micro_rows)
        )
        configured_max = fixed_micro_rows
        configured_budget = 0
    elif mode == TOKEN_BUDGET:
        if max_micro_rows <= 1 or padded_token_budget <= 0:
            raise ValueError(
                "Token-budget batching requires a row cap and token budget"
            )
        ordered = tuple(
            sorted(
                row_tuple,
                key=lambda row: _row_sort_key(
                    row,
                    supervised_assistant_start_index=(
                        supervised_assistant_start_index
                    ),
                ),
            )
        )
        groups: list[tuple[TokenizedRow, ...]] = []
        current: list[TokenizedRow] = []
        for row in ordered:
            candidate = (*current, row)
            exceeds = (
                len(candidate) > max_micro_rows
                or _event_padded_tokens(
                    candidate,
                    supervised_assistant_start_index=(
                        supervised_assistant_start_index
                    ),
                )
                > padded_token_budget
                or len(candidate) * max(value.length for value in candidate)
                > padded_token_budget
            )
            if current and exceeds:
                groups.append(tuple(current))
                current = [row]
            else:
                current.append(row)
        if current:
            groups.append(tuple(current))
        microbatches = tuple(groups)
        configured_max = max_micro_rows
        configured_budget = padded_token_budget
    else:
        raise ValueError(f"Unsupported S4 batching mode: {mode}")

    flattened = tuple(row for group in microbatches for row in group)
    if Counter(map(id, flattened)) != Counter(map(id, row_tuple)):
        raise RuntimeError("S4 microbatch planning changed local row membership")

    padded_event_tokens = 0
    unpadded_event_tokens = 0
    padded_response_tokens = 0
    unpadded_response_tokens = 0
    for group in microbatches:
        padded, unpadded = _event_token_totals(
            group,
            supervised_assistant_start_index=supervised_assistant_start_index,
        )
        padded_event_tokens += padded
        unpadded_event_tokens += unpadded
        padded, unpadded = _response_token_totals(
            group,
            supervised_assistant_start_index=supervised_assistant_start_index,
        )
        padded_response_tokens += padded
        unpadded_response_tokens += unpadded
    padded_condition_tokens = sum(
        len(group) * max(row.length for row in group) for group in microbatches
    )
    unpadded_condition_tokens = sum(row.length for row in row_tuple)
    return MicrobatchPlan(
        mode=mode,
        microbatches=microbatches,
        max_micro_rows=configured_max,
        padded_token_budget=configured_budget,
        peak_padded_event_tokens=max(
            _event_padded_tokens(
                group,
                supervised_assistant_start_index=(
                    supervised_assistant_start_index
                ),
            )
            for group in microbatches
        ),
        peak_padded_condition_tokens=max(
            len(group) * max(row.length for row in group) for group in microbatches
        ),
        padded_event_tokens=padded_event_tokens,
        unpadded_event_tokens=unpadded_event_tokens,
        padded_response_tokens=padded_response_tokens,
        unpadded_response_tokens=unpadded_response_tokens,
        padded_condition_tokens=padded_condition_tokens,
        unpadded_condition_tokens=unpadded_condition_tokens,
    )

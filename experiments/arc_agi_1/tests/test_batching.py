from __future__ import annotations

from s2_hypernet.nvarc_pg.events import PredictionEvent, TokenizedRow
from s4_circuit.batching import FIXED_ROWS, TOKEN_BUDGET, build_microbatch_plan


def _row(dataset_index: int, response_ends: tuple[int, ...]) -> TokenizedRow:
    length = max(response_ends)
    events = tuple(
        PredictionEvent(
            event_index=index,
            message_index=2 * index + 1,
            condition_end=response_end - 1,
            response_start=response_end - 1,
            response_end=response_end,
            labels=(-100,) * (response_end - 1) + (dataset_index + 1,),
        )
        for index, response_end in enumerate(response_ends)
    )
    return TokenizedRow(
        input_ids=(dataset_index + 1,) * length,
        events=events,
        dataset_index=dataset_index,
        puzzle_name=f"p{dataset_index}",
    )


def _indices(plan: object) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(row.dataset_index for row in group) for group in plan.microbatches
    )


def test_fixed_row_plan_preserves_original_order() -> None:
    rows = tuple(_row(index, (20 + index,)) for index in range(4))
    plan = build_microbatch_plan(
        rows,
        mode=FIXED_ROWS,
        fixed_micro_rows=2,
        max_micro_rows=1,
        padded_token_budget=0,
    )
    assert _indices(plan) == ((0, 1), (2, 3))
    assert plan.max_micro_rows == 2
    assert plan.peak_padded_event_tokens == 46


def test_token_budget_plan_is_deterministic_and_isolates_long_rows() -> None:
    rows = (
        _row(0, (100,)),
        _row(1, (110,)),
        _row(2, (120,)),
        _row(3, (900,)),
    )
    kwargs = {
        "mode": TOKEN_BUDGET,
        "fixed_micro_rows": 1,
        "max_micro_rows": 4,
        "padded_token_budget": 400,
    }
    first = build_microbatch_plan(rows, **kwargs)
    second = build_microbatch_plan(tuple(reversed(rows)), **kwargs)
    assert _indices(first) == _indices(second) == ((0, 1, 2), (3,))
    assert sorted(index for group in _indices(first) for index in group) == [0, 1, 2, 3]
    assert first.peak_padded_event_tokens == 900
    assert all(
        len(group) == 1
        or len(group) * max(row.events[0].response_end for row in group) <= 400
        for group in first.microbatches
    )
    assert first.event_padding_ratio >= 1.0
    assert first.condition_padding_ratio >= 1.0


def test_token_budget_uses_all_event_prefixes_not_only_final_length() -> None:
    rows = (
        _row(0, (50, 200)),
        _row(1, (50, 210)),
        _row(2, (60,)),
    )
    plan = build_microbatch_plan(
        rows,
        mode=TOKEN_BUDGET,
        fixed_micro_rows=1,
        max_micro_rows=3,
        padded_token_budget=430,
    )
    assert _indices(plan) == ((2, 0), (1,))
    assert plan.padded_event_tokens >= plan.unpadded_event_tokens


def test_first_assistant_can_remain_in_rows_but_be_excluded_from_plan() -> None:
    rows = (
        _row(0, (50, 200)),
        _row(1, (60, 210)),
    )
    plan = build_microbatch_plan(
        rows,
        mode=FIXED_ROWS,
        fixed_micro_rows=2,
        max_micro_rows=1,
        padded_token_budget=0,
        supervised_assistant_start_index=1,
    )
    assert _indices(plan) == ((0, 1),)
    assert plan.padded_event_tokens == 420
    assert plan.unpadded_event_tokens == 410
    assert plan.padded_response_tokens == 2
    assert plan.unpadded_response_tokens == 2

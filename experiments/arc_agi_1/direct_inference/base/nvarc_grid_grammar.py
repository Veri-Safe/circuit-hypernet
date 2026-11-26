#!/usr/bin/env python3
"""Rectangular ARC grid grammar used by the frozen internal evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class GridGrammarState:
    completed_rows: int = 0
    current_width: int = 0
    target_width: int | None = None
    finished: bool = False

    def allowed_token_ids(
        self,
        digit_token_ids: Sequence[int],
        newline_token_id: int,
        eos_token_id: int,
    ) -> tuple[int, ...]:
        if self.finished:
            return ()
        allowed: list[int] = []
        row_limit = self.target_width if self.target_width is not None else 30
        if self.current_width < row_limit:
            allowed.extend(digit_token_ids)
        row_complete = self.current_width > 0 and (
            self.target_width is None or self.current_width == self.target_width
        )
        if row_complete:
            allowed.append(eos_token_id)
            if self.completed_rows + 1 < 30:
                allowed.append(newline_token_id)
        return tuple(allowed)

    def consume(
        self,
        token_id: int,
        digit_token_ids: Sequence[int],
        newline_token_id: int,
        eos_token_id: int,
    ) -> None:
        allowed = self.allowed_token_ids(digit_token_ids, newline_token_id, eos_token_id)
        if token_id not in allowed:
            raise ValueError(f"Token {token_id} is not allowed in grammar state {self}")
        if token_id in digit_token_ids:
            self.current_width += 1
        elif token_id == newline_token_id:
            if self.target_width is None:
                self.target_width = self.current_width
            self.completed_rows += 1
            self.current_width = 0
        elif token_id == eos_token_id:
            if self.target_width is None:
                self.target_width = self.current_width
            self.completed_rows += 1
            self.finished = True


def decode_grid_tokens(
    token_ids: Sequence[int],
    digit_by_token_id: Mapping[int, int],
    newline_token_id: int,
    eos_token_id: int,
) -> list[list[int]] | None:
    state = GridGrammarState()
    rows: list[list[int]] = [[]]
    digit_ids = tuple(digit_by_token_id)
    try:
        for token_id in token_ids:
            state.consume(token_id, digit_ids, newline_token_id, eos_token_id)
            if token_id in digit_by_token_id:
                rows[-1].append(digit_by_token_id[token_id])
            elif token_id == newline_token_id:
                rows.append([])
            elif token_id == eos_token_id:
                return rows
    except ValueError:
        return None
    return None

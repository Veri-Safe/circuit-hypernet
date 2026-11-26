from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class RecoveryCheckpointSchedule:
    """Runtime-only checkpoint cadence; it does not affect model updates."""

    dense_until_step: int
    dense_every: int
    steady_every: int

    @classmethod
    def from_environment(
        cls,
        *,
        max_steps: int,
        default_steady_every: int,
    ) -> "RecoveryCheckpointSchedule":
        schedule = cls(
            dense_until_step=int(
                os.environ.get("S4_CHECKPOINT_DENSE_UNTIL_STEP", max_steps // 2)
            ),
            dense_every=int(os.environ.get("S4_CHECKPOINT_DENSE_EVERY", 25)),
            steady_every=int(
                os.environ.get(
                    "S4_CHECKPOINT_STEADY_EVERY", default_steady_every
                )
            ),
        )
        schedule.validate(max_steps=max_steps)
        return schedule

    def validate(self, *, max_steps: int) -> None:
        if max_steps <= 0:
            raise ValueError("S4 max_steps must be positive")
        if not 0 <= self.dense_until_step <= max_steps:
            raise ValueError("S4 dense checkpoint boundary is outside training")
        if self.dense_every <= 0 or self.steady_every <= 0:
            raise ValueError("S4 checkpoint intervals must be positive")

    def should_save(
        self,
        step: int,
        *,
        target_step: int,
        milestone_steps: Iterable[int],
    ) -> bool:
        if step <= 0 or step > target_step:
            raise ValueError("S4 checkpoint step is outside the active run")
        if step == target_step or step in milestone_steps:
            return True
        interval = (
            self.dense_every
            if step <= self.dense_until_step
            else self.steady_every
        )
        return step % interval == 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

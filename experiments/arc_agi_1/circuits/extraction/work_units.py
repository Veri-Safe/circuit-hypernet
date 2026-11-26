from __future__ import annotations


def resolve_work_units(
    *,
    total_work_units: int,
    work_unit: int | None,
    worker_index: int | None,
    num_workers: int | None,
) -> tuple[int, ...]:
    if total_work_units <= 0:
        raise ValueError("total_work_units must be positive")
    if work_unit is not None:
        if worker_index is not None or num_workers is not None:
            raise ValueError(
                "Use either --work_unit or --worker_index/--num_workers, not both"
            )
        if not 0 <= work_unit < total_work_units:
            raise ValueError("work_unit is outside the frozen range")
        return (int(work_unit),)
    if worker_index is None or num_workers is None:
        raise ValueError("Provide --work_unit or both --worker_index and --num_workers")
    if num_workers <= 0 or not 0 <= worker_index < num_workers:
        raise ValueError("Worker partition is invalid")
    result = tuple(range(worker_index, total_work_units, num_workers))
    if not result:
        raise ValueError("Worker partition received no work units")
    return result

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .identity import semantic_sha256


REPAIR_SCHEDULE_FORMAT = "nvarc_rearc_balanced_repair_schedule_v1"


def _schedule_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": value["format"],
        "config_sha256": value["config_sha256"],
        "production_policy_sha256": value["production_policy_sha256"],
        "catalog_manifest_sha256": value["catalog_manifest_sha256"],
        "state_sha256": value["state_sha256"],
        "num_workers": value["num_workers"],
        "assignments": value["assignments"],
    }


def repair_schedule_sha256(value: Mapping[str, Any]) -> str:
    return semantic_sha256(_schedule_identity(value))


def build_balanced_repair_schedule(
    work_units: Sequence[Mapping[str, Any]],
    *,
    num_workers: int,
    config_sha256: str,
    production_policy_sha256: str,
    catalog_manifest_sha256: str,
    state_sha256: str,
) -> dict[str, Any]:
    if num_workers <= 0:
        raise ValueError("Repair num_workers must be positive")
    for label, value in (
        ("config_sha256", config_sha256),
        ("production_policy_sha256", production_policy_sha256),
        ("catalog_manifest_sha256", catalog_manifest_sha256),
        ("state_sha256", state_sha256),
    ):
        if len(value) != 64:
            raise ValueError(f"Repair {label} is not a SHA256")

    normalized = []
    seen: set[int] = set()
    for raw in work_units:
        work_unit = int(raw["work_unit"])
        if work_unit < 0 or work_unit in seen:
            raise ValueError(f"Invalid or duplicate repair work unit: {work_unit}")
        seen.add(work_unit)
        pending_events = int(raw["pending_events"])
        expected_events = int(raw["expected_events"])
        predicted_seconds = float(raw["predicted_seconds"])
        if not 0 <= pending_events <= expected_events or expected_events <= 0:
            raise ValueError(f"Invalid repair event counts for work unit {work_unit}")
        if predicted_seconds < 0:
            raise ValueError(f"Negative repair cost for work unit {work_unit}")
        normalized.append(
            {
                "work_unit": work_unit,
                "expected_events": expected_events,
                "committed_events": int(raw["committed_events"]),
                "pending_events": pending_events,
                "pending_replay_tokens": int(raw["pending_replay_tokens"]),
                "predicted_seconds": round(predicted_seconds, 6),
                "state_sha256": str(raw["state_sha256"]),
            }
        )

    if not normalized:
        raise ValueError("Repair schedule has no unfinished work units")

    assignments: list[dict[str, Any]] = [
        {
            "worker_index": worker_index,
            "predicted_seconds": 0.0,
            "pending_events": 0,
            "pending_replay_tokens": 0,
            "work_units": [],
        }
        for worker_index in range(num_workers)
    ]
    ordered = sorted(
        normalized,
        key=lambda value: (
            -float(value["predicted_seconds"]),
            -int(value["pending_events"]),
            int(value["work_unit"]),
        ),
    )
    for item in ordered:
        lane = min(
            assignments,
            key=lambda value: (
                float(value["predicted_seconds"]),
                int(value["pending_events"]),
                int(value["worker_index"]),
            ),
        )
        lane["work_units"].append(item)
        lane["predicted_seconds"] = round(
            float(lane["predicted_seconds"]) + float(item["predicted_seconds"]), 6
        )
        lane["pending_events"] += int(item["pending_events"])
        lane["pending_replay_tokens"] += int(item["pending_replay_tokens"])

    identity = {
        "format": REPAIR_SCHEDULE_FORMAT,
        "config_sha256": config_sha256,
        "production_policy_sha256": production_policy_sha256,
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "state_sha256": state_sha256,
        "num_workers": num_workers,
        "assignments": assignments,
    }
    result = {
        **identity,
        "status": "complete",
        "schedule_sha256": semantic_sha256(identity),
        "unfinished_work_units": len(normalized),
        "pending_events": sum(int(value["pending_events"]) for value in normalized),
        "predicted_wall_seconds": max(
            float(value["predicted_seconds"]) for value in assignments
        ),
    }
    validate_repair_schedule(result)
    return result


def validate_repair_schedule(
    value: Mapping[str, Any],
    *,
    config_sha256: str | None = None,
    production_policy_sha256: str | None = None,
    catalog_manifest_sha256: str | None = None,
) -> None:
    if value.get("format") != REPAIR_SCHEDULE_FORMAT:
        raise ValueError("Repair schedule format changed")
    if value.get("status") != "complete":
        raise ValueError("Repair schedule is not complete")
    if value.get("schedule_sha256") != repair_schedule_sha256(value):
        raise ValueError("Repair schedule checksum changed")
    expected = {
        "config_sha256": config_sha256,
        "production_policy_sha256": production_policy_sha256,
        "catalog_manifest_sha256": catalog_manifest_sha256,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if expected_value is not None and value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(f"Repair schedule identity changed: {mismatches}")

    num_workers = int(value["num_workers"])
    assignments = list(value["assignments"])
    if len(assignments) != num_workers:
        raise ValueError("Repair schedule worker coverage changed")
    work_units = []
    for expected_worker, assignment in enumerate(assignments):
        if int(assignment["worker_index"]) != expected_worker:
            raise ValueError("Repair schedule worker order changed")
        work_units.extend(
            int(item["work_unit"]) for item in assignment.get("work_units", [])
        )
    if len(work_units) != len(set(work_units)):
        raise ValueError("Repair schedule assigns a work unit more than once")


def scheduled_work_units(
    value: Mapping[str, Any], *, worker_index: int, num_workers: int
) -> tuple[int, ...]:
    validate_repair_schedule(value)
    if int(value["num_workers"]) != num_workers:
        raise ValueError("Repair schedule num_workers changed")
    if not 0 <= worker_index < num_workers:
        raise ValueError("Repair worker index is outside the schedule")
    assignment = value["assignments"][worker_index]
    if int(assignment["worker_index"]) != worker_index:
        raise ValueError("Repair worker assignment identity changed")
    return tuple(int(item["work_unit"]) for item in assignment["work_units"])

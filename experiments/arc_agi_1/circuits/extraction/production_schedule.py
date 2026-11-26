from __future__ import annotations

from typing import Any, Mapping, Sequence

from .identity import semantic_sha256


FORMAT = "nvarc_rearc128_outside_cost_schedule_v1"


def schedule_sha256(value: Mapping[str, Any]) -> str:
    return semantic_sha256(
        {
            "format": value["format"],
            "config_sha256": value["config_sha256"],
            "production_policy_sha256": value["production_policy_sha256"],
            "catalog_manifest_sha256": value["catalog_manifest_sha256"],
            "num_workers": value["num_workers"],
            "long_replay_threshold": value["long_replay_threshold"],
            "default_event_lanes_per_gpu": value["default_event_lanes_per_gpu"],
            "assignments": value["assignments"],
        }
    )


def build_production_schedule(
    work_units: Sequence[Mapping[str, Any]],
    *,
    num_workers: int,
    config_sha256: str,
    production_policy_sha256: str,
    catalog_manifest_sha256: str,
    long_replay_threshold: int,
) -> dict[str, Any]:
    if num_workers <= 0:
        raise ValueError("Worker count must be positive")
    if long_replay_threshold <= 0:
        raise ValueError("Long replay threshold must be positive")
    normalized = []
    seen = set()
    for raw in work_units:
        work_unit = int(raw["work_unit"])
        if work_unit < 0 or work_unit in seen:
            raise ValueError(f"Invalid or duplicate work unit: {work_unit}")
        seen.add(work_unit)
        events = int(raw["events"])
        predicted_cost = float(raw["predicted_cost"])
        max_predicted_replay = int(raw["max_predicted_replay"])
        if events <= 0 or predicted_cost <= 0 or max_predicted_replay <= 0:
            raise ValueError(f"Invalid work-unit prediction: {raw}")
        normalized.append(
            {
                "work_unit": work_unit,
                "events": events,
                "condition_tokens": int(raw["condition_tokens"]),
                "predicted_target_tokens": int(raw["predicted_target_tokens"]),
                "max_predicted_replay": max_predicted_replay,
                "predicted_cost": round(predicted_cost, 6),
                "single_lane_required": max_predicted_replay > long_replay_threshold,
            }
        )
    assignments = [
        {
            "worker_index": index,
            "predicted_cost": 0.0,
            "events": 0,
            "single_lane_required": False,
            "work_units": [],
        }
        for index in range(num_workers)
    ]
    for item in sorted(
        normalized,
        key=lambda value: (-float(value["predicted_cost"]), int(value["work_unit"])),
    ):
        lane = min(
            assignments,
            key=lambda value: (
                float(value["predicted_cost"]),
                int(value["events"]),
                int(value["worker_index"]),
            ),
        )
        lane["work_units"].append(item)
        lane["predicted_cost"] = round(
            float(lane["predicted_cost"]) + float(item["predicted_cost"]), 6
        )
        lane["events"] += int(item["events"])
        lane["single_lane_required"] = bool(
            lane["single_lane_required"] or item["single_lane_required"]
        )
    if any(not value["work_units"] for value in assignments):
        raise ValueError("Cost-balanced schedule left an empty logical worker")
    identity = {
        "format": FORMAT,
        "config_sha256": config_sha256,
        "production_policy_sha256": production_policy_sha256,
        "catalog_manifest_sha256": catalog_manifest_sha256,
        "num_workers": num_workers,
        "long_replay_threshold": long_replay_threshold,
        "default_event_lanes_per_gpu": 2,
        "assignments": assignments,
    }
    result = {
        **identity,
        "status": "complete",
        "schedule_sha256": semantic_sha256(identity),
        "work_units": len(normalized),
        "events": sum(int(value["events"]) for value in normalized),
        "single_lane_workers": sum(
            bool(value["single_lane_required"]) for value in assignments
        ),
        "physical_gpu_workers": num_workers,
        "predicted_max_worker_cost": max(
            float(value["predicted_cost"]) for value in assignments
        ),
    }
    validate_production_schedule(result)
    return result


def validate_production_schedule(
    value: Mapping[str, Any],
    *,
    config_sha256: str | None = None,
    production_policy_sha256: str | None = None,
    catalog_manifest_sha256: str | None = None,
    expected_events: int | None = None,
) -> None:
    if value.get("format") != FORMAT or value.get("status") != "complete":
        raise ValueError("Production schedule format/status changed")
    if (
        int(value.get("long_replay_threshold", 0)) <= 0
        or int(value.get("default_event_lanes_per_gpu", 0)) != 2
        or int(value.get("physical_gpu_workers", 0)) != int(value["num_workers"])
        or int(value["num_workers"]) <= 0
    ):
        raise ValueError("Production schedule execution topology changed")
    if value.get("schedule_sha256") != schedule_sha256(value):
        raise ValueError("Production schedule checksum changed")
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
        raise ValueError(f"Production schedule identity changed: {mismatches}")
    assignments = list(value.get("assignments", []))
    if len(assignments) != int(value["num_workers"]):
        raise ValueError("Production schedule worker coverage changed")
    work_units = []
    for worker_index, assignment in enumerate(assignments):
        if int(assignment.get("worker_index", -1)) != worker_index:
            raise ValueError("Production schedule worker order changed")
        items = list(assignment.get("work_units", []))
        if sum(int(item["events"]) for item in items) != int(
            assignment.get("events", -1)
        ):
            raise ValueError("Production schedule worker event accounting changed")
        if bool(assignment.get("single_lane_required")) != any(
            bool(item.get("single_lane_required")) for item in items
        ):
            raise ValueError("Production schedule long-event accounting changed")
        work_units.extend(int(item["work_unit"]) for item in items)
    if len(work_units) != len(set(work_units)) or len(work_units) != int(
        value.get("work_units", -1)
    ):
        raise ValueError("Production schedule work-unit coverage changed")
    if int(value.get("work_units", -1)) != 1024 or set(work_units) != set(range(1024)):
        raise ValueError("Production schedule must cover work units 0..1023 exactly")
    total_events = sum(int(value.get("events", 0)) for value in assignments)
    if total_events != int(value.get("events", -1)):
        raise ValueError("Production schedule event accounting changed")
    if expected_events is not None and total_events != expected_events:
        raise ValueError(
            f"Production schedule has {total_events} events, expected {expected_events}"
        )


def scheduled_work_units(
    value: Mapping[str, Any], *, worker_index: int, num_workers: int
) -> tuple[int, ...]:
    validate_production_schedule(value)
    if num_workers != int(value["num_workers"]) or not 0 <= worker_index < num_workers:
        raise ValueError("Production schedule worker topology changed")
    assignment = value["assignments"][worker_index]
    if int(assignment["worker_index"]) != worker_index:
        raise ValueError("Production schedule assignment changed")
    return tuple(int(item["work_unit"]) for item in assignment["work_units"])


def scheduled_worker_assignment(
    value: Mapping[str, Any], *, worker_index: int, num_workers: int
) -> Mapping[str, Any]:
    """Return one validated physical-GPU assignment."""

    validate_production_schedule(value)
    if num_workers != int(value["num_workers"]) or not 0 <= worker_index < num_workers:
        raise ValueError("Production schedule worker topology changed")
    assignment = value["assignments"][worker_index]
    if int(assignment["worker_index"]) != worker_index:
        raise ValueError("Production schedule assignment changed")
    return assignment

from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


S3_ROOT = Path(__file__).resolve().parents[1]
HC_ROOT = S3_ROOT.parent
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
sys.path.insert(0, str(S3_ROOT / "src"))

from s3circuit.nvarc_circuit.config import load_config
from s3circuit.nvarc_circuit.events import read_catalog_chunk
from s3circuit.nvarc_circuit.finalize import validate_final_shard
from s3circuit.nvarc_circuit.identity import sha256_file
from s3circuit.nvarc_circuit.policy import load_production_policy
from s3circuit.nvarc_circuit.storage import write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify every Stage-B final shard and publish exact global coverage."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--final_root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_expected_events(
    catalog_root: Path,
    *,
    config_sha256: str,
    expected_events: int,
    num_work_units: int,
) -> tuple[dict, dict[int, set[str]]]:
    manifest = json.loads((catalog_root / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest = {
        "format": "nvarc_rearc_event_catalog_v1",
        "status": "complete",
        "config_sha256": config_sha256,
        "events": expected_events,
        "unique_event_keys": expected_events,
        "num_work_units": num_work_units,
    }
    mismatches = {
        key: {"expected": expected, "actual": manifest.get(key)}
        for key, expected in expected_manifest.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Finalization catalog identity changed: {mismatches}")
    by_work_unit: dict[int, set[str]] = defaultdict(set)
    all_keys: set[str] = set()
    for chunk in manifest["chunks"]:
        path = catalog_root / chunk["path"]
        if sha256_file(path) != chunk["tensor_sha256"]:
            raise ValueError(f"Catalog checksum changed during finalization: {path}")
        for event in read_catalog_chunk(path):
            if event.event_key in all_keys:
                raise ValueError(f"Catalog repeats event key: {event.event_key}")
            all_keys.add(event.event_key)
            by_work_unit[event.work_unit].add(event.event_key)
    if len(all_keys) != expected_events or len(by_work_unit) != num_work_units:
        raise ValueError("Catalog finalization coverage is incomplete")
    return manifest, dict(by_work_unit)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    policy = load_production_policy(args.policy, config=config)
    catalog_root = args.catalog.resolve()
    final_root = (
        args.final_root.resolve()
        if args.final_root
        else Path(config.artifacts.output_root).resolve() / "final"
    )
    output = args.output.resolve() if args.output else final_root / "root_manifest.json"
    catalog_manifest, expected_by_unit = _load_expected_events(
        catalog_root,
        config_sha256=config.semantic_sha256,
        expected_events=config.data.expected_train_events,
        num_work_units=config.artifacts.num_work_units,
    )
    seen_events: set[str] = set()
    seen_artifacts: set[str] = set()
    work_units = []
    total_targets = total_trails = total_bytes = 0
    for work_unit in range(config.artifacts.num_work_units):
        root = final_root / f"work_unit_{work_unit:04d}"
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing final work-unit manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest = {
            "format": "nvarc_rearc_circuit_final_work_unit_v1",
            "status": "complete",
            "config_sha256": config.semantic_sha256,
            "production_policy_sha256": policy.semantic_sha256,
            "work_unit": work_unit,
            "events": len(expected_by_unit[work_unit]),
            "unique_event_keys": len(expected_by_unit[work_unit]),
        }
        mismatches = {
            key: {"expected": expected, "actual": manifest.get(key)}
            for key, expected in expected_manifest.items()
            if manifest.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Final work-unit {work_unit} identity changed: {mismatches}"
            )
        local_events: set[str] = set()
        local_artifacts: set[str] = set()
        shard_rows = []
        for shard in manifest["shards"]:
            path = root / shard["path"]
            actual_sha = sha256_file(path)
            if actual_sha != shard["tensor_sha256"]:
                raise ValueError(f"Final shard checksum changed: {path}")
            validation = validate_final_shard(
                path,
                expected_config_sha256=config.semantic_sha256,
                expected_policy_sha256=policy.semantic_sha256,
            )
            if validation["events"] != int(shard["events"]):
                raise ValueError(f"Final shard event count changed: {path}")
            event_keys = set(validation["event_keys"])
            artifact_ids = set(validation["artifact_ids"])
            if (
                len(event_keys) != validation["events"]
                or len(artifact_ids) != validation["events"]
                or local_events.intersection(event_keys)
                or local_artifacts.intersection(artifact_ids)
            ):
                raise ValueError(
                    f"Final work unit contains duplicate identities: {path}"
                )
            local_events.update(event_keys)
            local_artifacts.update(artifact_ids)
            shard_bytes = path.stat().st_size
            total_bytes += shard_bytes
            total_targets += int(validation["targets"])
            total_trails += int(validation["trails"])
            shard_rows.append(
                {
                    "path": f"work_unit_{work_unit:04d}/{path.name}",
                    "events": validation["events"],
                    "targets": validation["targets"],
                    "trails": validation["trails"],
                    "bytes": shard_bytes,
                    "tensor_sha256": actual_sha,
                }
            )
        if local_events != expected_by_unit[work_unit]:
            missing = sorted(expected_by_unit[work_unit] - local_events)[:5]
            unexpected = sorted(local_events - expected_by_unit[work_unit])[:5]
            raise ValueError(
                f"Final work-unit {work_unit} coverage changed: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if seen_events.intersection(local_events) or seen_artifacts.intersection(
            local_artifacts
        ):
            raise ValueError("Final identities are duplicated across work units")
        seen_events.update(local_events)
        seen_artifacts.update(local_artifacts)
        work_units.append(
            {
                "work_unit": work_unit,
                "events": len(local_events),
                "payload_deleted": bool(manifest.get("payload_deleted", False)),
                "manifest_sha256": sha256_file(manifest_path),
                "shards": shard_rows,
            }
        )
        if (
            work_unit + 1
        ) % 64 == 0 or work_unit + 1 == config.artifacts.num_work_units:
            print(
                json.dumps(
                    {
                        "phase": "global_finalization",
                        "work_units_complete": work_unit + 1,
                        "work_units_total": config.artifacts.num_work_units,
                        "events_complete": len(seen_events),
                        "events_total": config.data.expected_train_events,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if len(seen_events) != config.data.expected_train_events:
        raise ValueError("Global final event coverage is incomplete")
    if len(seen_artifacts) != len(seen_events):
        raise ValueError("Global final artifact IDs are not one-to-one with events")
    result = {
        "format": "nvarc_rearc_circuit_final_root_v1",
        "status": "complete",
        "config_sha256": config.semantic_sha256,
        "production_policy_sha256": policy.semantic_sha256,
        "dataset_snapshot_sha256": catalog_manifest["dataset_snapshot_sha256"],
        "events": len(seen_events),
        "unique_event_keys": len(seen_events),
        "unique_artifact_ids": len(seen_artifacts),
        "targets": total_targets,
        "trails": total_trails,
        "tensor_bytes": total_bytes,
        "work_units": work_units,
    }
    write_json_atomic(output, result)
    print(
        json.dumps({**result, "work_units": len(work_units)}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()

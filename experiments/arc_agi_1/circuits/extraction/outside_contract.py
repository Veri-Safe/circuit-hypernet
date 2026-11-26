from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .identity import sha256_file
from s3circuit.nvarc_transcoder.identity import config_semantic_sha256_for_root


DATASET_FORMAT = "nvarc_rearc128_outside_b0_b1_union_v1"
SOURCE_DATASET_FORMAT = "rearc128_outside_delta4_s42_v1"
POLICY_FORMAT = "nvarc_rearc128_outside_b0_b1_lr_policy_v1"
SOURCES = ("outside_r128_b0", "outside_r128_b1")
EXPECTED_STATS = {
    "rows": 3_200,
    "puzzles": 400,
    "events": 20_810,
    "source_rows": {"outside_r128_b0": 1_600, "outside_r128_b1": 1_600},
    "source_events": {"outside_r128_b0": 10_430, "outside_r128_b1": 10_380},
    "generation_mode_rows": {"cumulative_generated": 3_200},
    "trace_shards": 240,
}
EXPECTED_SOURCE_IDENTITIES = {
    "b0": {
        "dataset_id": "outside_r128_b0",
        "rows": 1_600,
        "puzzles": 400,
        "events": 10_430,
        "dataset_fingerprint": "99cc13629b61b468",
        "storage_fingerprint": "4b8da2571362f8a7",
        "indices_sha256": "1fa5fe172f11f87aa38e4b891f9f01c11e2bd65e7c2cc389aff6c91a2b2eea70",
        "semantic_sha256": "db4371cd2e2ed670af39488df8ad608122ffff3b4dc43a2e1bd65c766f24e4f0",
    },
    "b1": {
        "dataset_id": "outside_r128_b1",
        "rows": 1_600,
        "puzzles": 400,
        "events": 10_380,
        "dataset_fingerprint": "3778a540152fdc46",
        "storage_fingerprint": "07c1282b0de509dc",
        "indices_sha256": "88907229724bf0c7bcc543857039344e24591426accbc1af97d228d8b5eb7fb8",
        "semantic_sha256": "e58f604fb0bf3e9ec158218b18a5912a0dfc41b5ed07f01f757ee5bd244d81bc",
    },
}


def load_and_validate_dataset_manifest(
    path: str | Path,
    *,
    selected_sources: tuple[str, ...] = (),
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("format_version") == SOURCE_DATASET_FORMAT:
        if len(selected_sources) != 1 or selected_sources[0] not in SOURCES:
            raise ValueError(
                "Direct outside manifest requires exactly one B0/B1 source"
            )
        source_name = selected_sources[0]
        source_key = "b0" if source_name == SOURCES[0] else "b1"
        expected = EXPECTED_SOURCE_IDENTITIES[source_key]
        block = value.get("blocks", {}).get(source_key, {})
        direct_expected = {
            "dataset_id": expected["dataset_id"],
            "rows": expected["rows"],
            "puzzles": expected["puzzles"],
            "assistant_events": expected["events"],
            "dataset_fingerprint": expected["dataset_fingerprint"],
            "storage_fingerprint": expected["storage_fingerprint"],
            "indices_sha256": expected["indices_sha256"],
            "semantic_sha256": expected["semantic_sha256"],
            "rows_per_puzzle": 4,
        }
        mismatches = {
            f"blocks.{source_key}.{key}": {
                "expected": expected_value,
                "actual": block.get(key),
            }
            for key, expected_value in direct_expected.items()
            if block.get(key) != expected_value
        }
        top_level_expected = {
            "selection_algorithm": "rearc128_per_puzzle_sha256_v1",
            "selection_seed": 42,
            "parent_rearc128_indices_sha256": (
                "49f0342ee4b4eeca7d0362db9e65a93447641c7f4199d8d0b2f69e4cbc5e3ddc"
            ),
            "deterministic_rebuild_verified": True,
        }
        for key, expected_value in top_level_expected.items():
            if value.get(key) != expected_value:
                mismatches[key] = {
                    "expected": expected_value,
                    "actual": value.get(key),
                }
        invariants = value.get("invariants", {})
        expected_invariants = {
            "groups_disjoint_from_rearc128": True,
            "groups_pairwise_disjoint": True,
            "rows_per_puzzle_per_group": 4,
            "unique_rows_per_group": 1_600,
        }
        if invariants != expected_invariants:
            mismatches["invariants"] = {
                "expected": expected_invariants,
                "actual": invariants,
            }
        indices_path = manifest_path.parent / "indices" / f"{source_key}.npy"
        if not indices_path.is_file():
            raise FileNotFoundError(indices_path)
        if sha256_file(indices_path) != expected["indices_sha256"]:
            mismatches[f"blocks.{source_key}.indices_file_sha256"] = {
                "expected": expected["indices_sha256"],
                "actual": sha256_file(indices_path),
            }
        if mismatches:
            raise ValueError(f"RE-ARC128 direct outside manifest mismatch: {mismatches}")
        return {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "dataset_fingerprint": str(block["dataset_fingerprint"]),
            "storage_fingerprint": str(block["storage_fingerprint"]),
            "source_name": source_name,
            "source_indices_path": str(indices_path),
            "source_indices_sha256": expected["indices_sha256"],
            "value": value,
        }
    mismatches: dict[str, Any] = {}
    if value.get("format_version") != DATASET_FORMAT:
        mismatches["format_version"] = value.get("format_version")
    stats = value.get("stats")
    if not isinstance(stats, dict):
        mismatches["stats"] = stats
    else:
        for key, expected in EXPECTED_STATS.items():
            if stats.get(key) != expected:
                mismatches[f"stats.{key}"] = {
                    "expected": expected,
                    "actual": stats.get(key),
                }
    sources = value.get("sources")
    if not isinstance(sources, dict):
        mismatches["sources"] = sources
    else:
        for key, expected in EXPECTED_SOURCE_IDENTITIES.items():
            observed = sources.get(key)
            if not isinstance(observed, dict):
                mismatches[f"sources.{key}"] = observed
                continue
            for field, expected_value in expected.items():
                if observed.get(field) != expected_value:
                    mismatches[f"sources.{key}.{field}"] = {
                        "expected": expected_value,
                        "actual": observed.get(field),
                    }
    exact = {
        "experiment_path": (
            "4server/0813/nvarc_rearc128_outside_delta4_fullft/experiment.json"
        ),
        "pairwise_overlap_rows": 0,
        "parent_rearc128_indices_sha256": (
            "49f0342ee4b4eeca7d0362db9e65a93447641c7f4199d8d0b2f69e4cbc5e3ddc"
        ),
        "parent_overlap_rows": {"b0": 0, "b1": 0},
        "source_order": list(SOURCES),
        "experiment_semantic_sha256": (
            "f5014a2710afaa3c0074aea00ea144d5c946ac3874c766807ce48a2d97fe39f8"
        ),
    }
    for key, expected in exact.items():
        if value.get(key) != expected:
            mismatches[key] = {"expected": expected, "actual": value.get(key)}
    fingerprint = value.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        mismatches["dataset_fingerprint"] = fingerprint
    if mismatches:
        raise ValueError(f"RE-ARC128 outside union manifest mismatch: {mismatches}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "dataset_fingerprint": fingerprint,
        "value": value,
    }


def normalize_outside_event_coordinates(
    *,
    source_name: str,
    trace_generation_mode: str,
    view_index: int,
    test_index: int,
) -> tuple[None, None]:
    if source_name not in SOURCES:
        raise ValueError(f"Unknown outside source: {source_name}")
    if trace_generation_mode != "cumulative_generated":
        raise ValueError(f"Outside source {source_name} changed generation mode")
    if (view_index, test_index) != (-1, -1):
        raise ValueError(
            f"Outside source {source_name} must use (-1, -1) view/test sentinels"
        )
    return None, None


def validate_catalog_source_counts(
    *,
    selected_sources: tuple[str, ...],
    source_rows: Mapping[str, int],
    source_events: Mapping[str, int],
    generation_mode_rows: Mapping[str, int],
) -> None:
    sources = SOURCES if not selected_sources else selected_sources
    expected_rows = {
        source: EXPECTED_STATS["source_rows"][source] for source in sources
    }
    expected_events = {
        source: EXPECTED_STATS["source_events"][source] for source in sources
    }
    expected = {
        "source_rows": expected_rows,
        "source_events": expected_events,
        "generation_mode_rows": {
            "cumulative_generated": sum(expected_rows.values())
        },
    }
    actual = {
        "source_rows": dict(sorted(source_rows.items())),
        "source_events": dict(sorted(source_events.items())),
        "generation_mode_rows": dict(sorted(generation_mode_rows.items())),
    }
    if actual != expected:
        raise ValueError(f"Outside catalog source contract changed: {actual} != {expected}")


def load_and_validate_lr_policy(
    path: str | Path,
    *,
    expected_config_sha256: str,
    stage_a_config: Any | None = None,
) -> dict[str, Any]:
    policy_path = Path(path).resolve()
    value = json.loads(policy_path.read_text(encoding="utf-8"))
    if value.get("format") != POLICY_FORMAT or value.get("status") != "complete":
        raise ValueError("Outside production LR policy format/status changed")
    accepted_config_hashes = {expected_config_sha256}
    if stage_a_config is not None:
        root_value = os.environ.get("HC_ROOT")
        layers = value.get("layers")
        if root_value and isinstance(layers, list) and layers:
            current_root = Path(root_value).resolve()
            current_output = Path(stage_a_config.model.output_root).resolve()
            try:
                output_relative = current_output.relative_to(current_root)
            except ValueError:
                output_relative = None
            declared = Path(str(layers[0].get("selection", "")))
            if output_relative is not None and declared.is_absolute():
                declared_parts = declared.parts
                suffix_parts = output_relative.parts
                for index in range(len(declared_parts) - len(suffix_parts) + 1):
                    if declared_parts[index : index + len(suffix_parts)] == suffix_parts:
                        declared_root = Path(*declared_parts[:index])
                        accepted_config_hashes.add(
                            config_semantic_sha256_for_root(
                                stage_a_config,
                                current_root=current_root,
                                identity_root=declared_root,
                            )
                        )
                        break
    if value.get("config_sha256") not in accepted_config_hashes:
        raise ValueError("Outside production LR policy config identity changed")
    run_ids = value.get("candidate_run_ids")
    layers = value.get("layers")
    if not isinstance(run_ids, list) or len(set(run_ids)) != 3:
        raise ValueError("Outside LR policy must contain three unique candidate runs")
    if not isinstance(layers, list) or len(layers) != 36:
        raise ValueError("Outside LR policy must contain exactly 36 layers")
    if [int(item.get("layer", -1)) for item in layers] != list(range(36)):
        raise ValueError("Outside LR policy layers are not ordered 0..35")
    for item in layers:
        if item.get("run_id") not in run_ids or float(item.get("learning_rate", 0)) <= 0:
            raise ValueError("Outside LR policy has an invalid layer selection")
        if not item.get("selection") or not item.get("comparison"):
            raise ValueError("Outside LR policy lacks selection provenance")
    return {
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "format": POLICY_FORMAT,
        "layers": 36,
        "accepted_config_sha256": sorted(accepted_config_hashes),
        "value": value,
    }


def count_values(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))

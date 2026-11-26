from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .identity import sha256_file
from s3circuit.nvarc_transcoder.identity import config_semantic_sha256_for_root


DATASET_FORMAT = "nvarc_rearc8_union_fast128_v1"
POLICY_FORMAT = "nvarc_rearc8_union_fast128_lr_policy_v1"
REARC8_PARITY_SELECTION_MODE = "rearc8_source_quartile_frozen_generation_v1"
REARC8_SINGLE_SOURCE_CANDIDATE_SELECTION_MODE = (
    "rearc8_single_source_candidate_v1"
)
REARC8_SINGLE_SOURCE_MASTER_SELECTION_MODE = (
    "rearc8_single_source_frozen_generation_v1"
)
EXPECTED_STATS = {
    "rows": 13_523,
    "puzzles": 800,
    "events": 48_692,
    "source_rows": {
        "arc1_public_author_view": 6_704,
        "arc1_public_canonical": 419,
        "rearc8_pg_s24_excl_base_s42_v1": 3_200,
        "rearc8_s42": 3_200,
    },
    "generation_mode_rows": {
        "cumulative_generated": 6_400,
        "final_only_gold_prefix": 7_123,
    },
    "trace_shards": 240,
}
EXPECTED_SOURCE_EVENTS = {
    "arc1_public_author_view": 6_704,
    "arc1_public_canonical": 419,
    "rearc8_pg_s24_excl_base_s42_v1": 20_826,
    "rearc8_s42": 20_743,
}
SEED24_EXCLUSIVE_SOURCE = "rearc8_pg_s24_excl_base_s42_v1"
SEED24_EXCLUSIVE_ROWS = 3_200
SEED24_EXCLUSIVE_PUZZLES = 400
SEED24_EXCLUSIVE_EVENTS = 20_826
REARC_SOURCES = {
    "rearc8_pg_s24_excl_base_s42_v1",
    "rearc8_s42",
}
PUBLIC_CANONICAL_SOURCE = "arc1_public_canonical"
PUBLIC_AUTHOR_VIEW_SOURCE = "arc1_public_author_view"
PUBLIC_EVAL_SOURCES = (PUBLIC_CANONICAL_SOURCE, PUBLIC_AUTHOR_VIEW_SOURCE)
PUBLIC_EVAL_ROWS = 7_123
PUBLIC_EVAL_PUZZLES = 400
PUBLIC_EVAL_EVENTS = 7_123
EXPECTED_BASE_INDEX_SHA256 = (
    "17669730a429768a3a3a97e626086f132c0c82b7c12fc7daa1e9ba5e76037f7b"
)
EXPECTED_PG_INDEX_SHA256 = (
    "af81733dff510c1f48592b3b8d540b21474c2cf3437d7f96e9904c253c46bfa6"
)


def load_and_validate_dataset_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    exact_fields = {
        "base_indices_sha256": EXPECTED_BASE_INDEX_SHA256,
        "pg_indices_sha256": EXPECTED_PG_INDEX_SHA256,
        "pg_algorithm": "rearc8_pg_s24_excl_base_s42_v1",
        "base_pg_overlap_rows": 0,
    }
    for key, expected in exact_fields.items():
        if value.get(key) != expected:
            mismatches[key] = {"expected": expected, "actual": value.get(key)}
    public = value.get("arc1_public")
    if not isinstance(public, dict) or public.get("solutions_loaded") is not False:
        mismatches["arc1_public.solutions_loaded"] = (
            public.get("solutions_loaded") if isinstance(public, dict) else None
        )
    fingerprint = value.get("dataset_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        mismatches["dataset_fingerprint"] = fingerprint
    if mismatches:
        raise ValueError(f"RE-ARC8 union manifest mismatch: {mismatches}")
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "dataset_fingerprint": fingerprint,
        "value": value,
    }


def validate_catalog_source_counts(
    *,
    source_rows: Mapping[str, int],
    source_events: Mapping[str, int],
    generation_mode_rows: Mapping[str, int],
) -> None:
    expected_rows = EXPECTED_STATS["source_rows"]
    expected_modes = EXPECTED_STATS["generation_mode_rows"]
    actual = {
        "source_rows": dict(sorted(source_rows.items())),
        "source_events": dict(sorted(source_events.items())),
        "generation_mode_rows": dict(sorted(generation_mode_rows.items())),
    }
    expected = {
        "source_rows": expected_rows,
        "source_events": EXPECTED_SOURCE_EVENTS,
        "generation_mode_rows": expected_modes,
    }
    if actual != expected:
        raise ValueError(f"RE-ARC8 catalog source contract changed: {actual} != {expected}")


def validate_seed24_exclusive_catalog_source_counts(
    *,
    source_rows: Mapping[str, int],
    source_events: Mapping[str, int],
    generation_mode_rows: Mapping[str, int],
) -> None:
    expected = {
        "source_rows": {SEED24_EXCLUSIVE_SOURCE: SEED24_EXCLUSIVE_ROWS},
        "source_events": {SEED24_EXCLUSIVE_SOURCE: SEED24_EXCLUSIVE_EVENTS},
        "generation_mode_rows": {"cumulative_generated": SEED24_EXCLUSIVE_ROWS},
    }
    actual = {
        "source_rows": dict(sorted(source_rows.items())),
        "source_events": dict(sorted(source_events.items())),
        "generation_mode_rows": dict(sorted(generation_mode_rows.items())),
    }
    if actual != expected:
        raise ValueError(
            f"RE-ARC8 seed24-exclusive catalog changed: {actual} != {expected}"
        )


def validate_public_eval_catalog_source_counts(
    *,
    source_rows: Mapping[str, int],
    source_events: Mapping[str, int],
    generation_mode_rows: Mapping[str, int],
) -> None:
    expected = {
        "source_rows": {
            PUBLIC_AUTHOR_VIEW_SOURCE: 6_704,
            PUBLIC_CANONICAL_SOURCE: 419,
        },
        "source_events": {
            PUBLIC_AUTHOR_VIEW_SOURCE: 6_704,
            PUBLIC_CANONICAL_SOURCE: 419,
        },
        "generation_mode_rows": {"final_only_gold_prefix": PUBLIC_EVAL_ROWS},
    }
    actual = {
        "source_rows": dict(sorted(source_rows.items())),
        "source_events": dict(sorted(source_events.items())),
        "generation_mode_rows": dict(sorted(generation_mode_rows.items())),
    }
    if actual != expected:
        raise ValueError(f"RE-ARC8 public-eval catalog changed: {actual} != {expected}")


def normalize_rearc8_event_coordinates(
    *,
    source_name: str,
    trace_generation_mode: str,
    view_index: int,
    test_index: int,
) -> tuple[int | None, int | None]:
    """Translate frozen dataset sentinels into optional artifact coordinates."""

    if source_name in REARC_SOURCES:
        if trace_generation_mode != "cumulative_generated":
            raise ValueError(f"RE-ARC8 source {source_name} changed generation mode")
        if (view_index, test_index) != (-1, -1):
            raise ValueError(
                f"RE-ARC8 source {source_name} must use (-1, -1) view/test sentinels"
            )
        return None, None

    if trace_generation_mode != "final_only_gold_prefix":
        raise ValueError(f"Public ARC source {source_name} changed generation mode")
    if test_index < 0:
        raise ValueError(f"Public ARC source {source_name} has a negative test_index")
    if source_name == PUBLIC_CANONICAL_SOURCE:
        if view_index != 0:
            raise ValueError("Canonical ARC rows must use view_index=0")
    elif source_name == PUBLIC_AUTHOR_VIEW_SOURCE:
        if view_index <= 0:
            raise ValueError("Author-view ARC rows must use a positive view_index")
    else:
        raise ValueError(f"Unknown RE-ARC8 union source: {source_name}")
    return view_index, test_index


def load_and_validate_lr_policy(
    path: str | Path,
    *,
    expected_config_sha256: str,
    stage_a_config: Any | None = None,
) -> dict[str, Any]:
    policy_path = Path(path).resolve()
    value = json.loads(policy_path.read_text(encoding="utf-8"))
    if value.get("format") != POLICY_FORMAT or value.get("status") != "complete":
        raise ValueError("RE-ARC8 production LR policy format/status changed")
    accepted_config_hashes = {expected_config_sha256}
    if stage_a_config is not None:
        root_value = os.environ.get("HC_ROOT")
        layers_for_alias = value.get("layers")
        if root_value and isinstance(layers_for_alias, list) and layers_for_alias:
            current_root = Path(root_value).resolve()
            current_output = Path(stage_a_config.model.output_root).resolve()
            try:
                output_relative = current_output.relative_to(current_root)
            except ValueError:
                output_relative = None
            declared = Path(str(layers_for_alias[0].get("selection", "")))
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
        raise ValueError(
            "RE-ARC8 production LR policy config identity changed: "
            f"{value.get('config_sha256')} not in {sorted(accepted_config_hashes)}"
        )
    run_ids = value.get("candidate_run_ids")
    if not isinstance(run_ids, list) or len(set(run_ids)) != len(run_ids) or not run_ids:
        raise ValueError("RE-ARC8 LR policy candidate run IDs are invalid")
    layers = value.get("layers")
    if not isinstance(layers, list) or len(layers) != 36:
        raise ValueError("RE-ARC8 LR policy must contain exactly 36 layers")
    observed = []
    for item in layers:
        if not isinstance(item, dict):
            raise ValueError("RE-ARC8 LR policy layer entry is not a mapping")
        layer = int(item.get("layer", -1))
        observed.append(layer)
        if item.get("run_id") not in run_ids:
            raise ValueError(f"RE-ARC8 LR policy layer {layer} has an unknown run")
        if float(item.get("learning_rate", 0.0)) <= 0.0:
            raise ValueError(f"RE-ARC8 LR policy layer {layer} has an invalid LR")
        for key in ("selection", "comparison"):
            if not isinstance(item.get(key), str) or not item[key]:
                raise ValueError(f"RE-ARC8 LR policy layer {layer} lacks {key}")
    if observed != list(range(36)):
        raise ValueError(f"RE-ARC8 LR policy layers are not ordered 0..35: {observed}")
    return {
        "path": str(policy_path),
        "sha256": sha256_file(policy_path),
        "format": POLICY_FORMAT,
        "layers": 36,
        "accepted_config_sha256": sorted(accepted_config_hashes),
        "value": value,
    }


def count_values(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))

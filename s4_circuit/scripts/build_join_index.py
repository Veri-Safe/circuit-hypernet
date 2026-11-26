from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
from pathlib import Path


HC_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
for source_root in (
    HC_ROOT / "s4_circuit" / "src",
    HC_ROOT / "s3_circuit" / "src",
    HC_ROOT / "s2_hypernet" / "src",
    HC_ROOT / "s1_lora_sft" / "src",
):
    sys.path.insert(0, str(source_root))

from s2_hypernet.nvarc_pg.data import load_training_sources
from s2_hypernet.nvarc_pg.runtime import load_tokenizer

from s4_circuit.artifacts import load_catalog_events, load_vector_bank
from s4_circuit.config import load_config
from s4_circuit.contracts import sha256_file
from s4_circuit.join import build_join_records, load_join_bundle, write_join_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the immutable S4 event-circuit join.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output.exists():
        records, metadata = load_join_bundle(args.output)
        if metadata.get("artifact_contract_sha256") != config.artifact_contract_sha256:
            raise ValueError("Existing join belongs to another S4 artifact contract")
        if len(records) != config.train_artifacts.expected_events:
            raise ValueError("Existing join coverage changed")
        print(json.dumps({"status": "already_complete", "events": len(records), "output": str(args.output)}, indent=2))
        return

    from s2_hypernet.nvarc_pg.config import load_experiment_config

    s2 = load_experiment_config(config.s2_config)
    tokenizer = load_tokenizer(s2)
    dataset = load_training_sources(s2.data.train_paths).combined
    catalog, catalog_manifest = load_catalog_events(
        config.train_artifacts.catalog_root,
        expected_events=config.train_artifacts.expected_events,
    )
    vector_bank = load_vector_bank(config.train_artifacts)
    final_manifest = vector_bank.root_manifest
    identity_mismatches = {
        "dataset_snapshot_sha256": (
            catalog_manifest.get("dataset_snapshot_sha256"),
            final_manifest.get("dataset_snapshot_sha256"),
        ),
        "config_sha256": (
            catalog_manifest.get("config_sha256"),
            final_manifest.get("config_sha256"),
        ),
    }
    changed = {
        key: {"catalog": values[0], "final": values[1]}
        for key, values in identity_mismatches.items()
        if values[0] != values[1]
    }
    if changed:
        raise ValueError(f"Catalog/final circuit identity mismatch: {changed}")
    records = build_join_records(
        dataset,
        tokenizer=tokenizer,
        max_sequence_length=s2.data.max_sequence_length,
        catalog_events=catalog,
        vector_bank=vector_bank,
    )
    metadata = write_join_bundle(
        args.output,
        records,
        artifact_contract_sha256=config.artifact_contract_sha256,
        catalog_manifest_sha256=sha256_file(
            Path(config.train_artifacts.catalog_root) / "manifest.json"
        ),
        final_root_manifest_sha256=config.train_artifacts.root_manifest_sha256,
    )
    print(json.dumps({"status": "complete", "events": len(records), "join_sha256": metadata["join_sha256"], "output": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

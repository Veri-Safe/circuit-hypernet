from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import sys
import time
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

from nvarc_qwen3.arc_eval_protocol import load_challenges
from s2_hypernet.nvarc_pg.eval_manifest import build_query_views, query_manifest_sha256
from s2_hypernet.nvarc_pg.eval_protocol import load_profile_eval_protocol
from s2_hypernet.nvarc_pg.runtime import load_tokenizer

from s4_circuit.artifacts import load_catalog_events, load_vector_bank
from s4_circuit.config import load_config
from s4_circuit.contracts import sha256_file
from s4_circuit.join import (
    build_eval_join_records,
    load_eval_join_bundle,
    validate_matching_eval_join_bundle,
    write_eval_join_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the immutable S4 ARC eval join.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output.exists():
        records, metadata = load_eval_join_bundle(args.output)
        if metadata.get("eval_artifact_contract_sha256") != config.eval_artifact_contract_sha256:
            raise ValueError("Existing evaluation join belongs to another artifact contract")
        if len(records) != config.eval_artifacts.expected_events:
            raise ValueError("Existing evaluation join coverage changed")
        print(json.dumps({"status": "already_complete", "queries": len(records), "output": str(args.output)}, indent=2))
        return

    from s2_hypernet.nvarc_pg.config import load_experiment_config

    s2 = load_experiment_config(config.s2_config)
    tokenizer = load_tokenizer(s2)
    protocol, _ = load_profile_eval_protocol(s2.evaluation)
    challenges = load_challenges(s2.evaluation.challenges)
    query_views = build_query_views(challenges, config=protocol, tokenizer=tokenizer)
    if len(query_views) != config.eval_artifacts.expected_events:
        raise ValueError("ARC evaluation query count changed")
    catalog, catalog_manifest = load_catalog_events(
        config.eval_artifacts.catalog_root,
        expected_events=config.eval_artifacts.expected_events,
    )
    vector_bank = load_vector_bank(config.eval_artifacts)
    final_manifest = vector_bank.root_manifest
    for key in ("dataset_snapshot_sha256", "config_sha256"):
        if catalog_manifest.get(key) != final_manifest.get(key):
            raise ValueError(f"Evaluation catalog/final identity mismatch: {key}")
    records = build_eval_join_records(
        query_views,
        tokenizer=tokenizer,
        catalog_events=catalog,
        vector_bank=vector_bank,
    )
    identity = {
        "query_manifest_sha256": query_manifest_sha256(query_views),
        "catalog_manifest_sha256": sha256_file(
            Path(config.eval_artifacts.catalog_root) / "manifest.json"
        ),
        "final_root_manifest_sha256": config.eval_artifacts.root_manifest_sha256,
    }
    status = "complete"
    try:
        metadata = write_eval_join_bundle(
            args.output,
            records,
            eval_artifact_contract_sha256=config.eval_artifact_contract_sha256,
            **identity,
        )
    except FileExistsError:
        # Siflow jobs can run on different hosts, where flock on the shared
        # mount does not reliably serialize the final commit. A concurrent
        # winner is reusable only after its full bundle and records validate.
        deadline = time.monotonic() + 60.0
        while True:
            try:
                metadata = validate_matching_eval_join_bundle(
                    args.output,
                    records,
                    expected_metadata={
                        "status": "complete",
                        "queries": config.eval_artifacts.expected_events,
                        "eval_artifact_contract_sha256": (
                            config.eval_artifact_contract_sha256
                        ),
                        **identity,
                    },
                )
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)
        status = "already_complete_concurrent"
    print(
        json.dumps(
            {
                "status": status,
                "queries": len(records),
                "join_sha256": metadata["join_sha256"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

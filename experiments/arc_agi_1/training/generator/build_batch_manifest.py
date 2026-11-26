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

from s2_hypernet.nvarc_pg.config import load_experiment_config
from s2_hypernet.nvarc_pg.data import load_training_sources

from s4_circuit.batches import build_logical_batch_manifest, validate_logical_batch_manifest
from s4_circuit.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the paired S4 logical batch manifest.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = Path(config.batch_manifest)
    if output.exists():
        rows, metadata = validate_logical_batch_manifest(output, config=config)
        print(json.dumps({"status": "already_complete", "shape": list(rows.shape), "schedule_sha256": metadata["schedule_sha256"], "output": str(output)}, indent=2))
        return
    s2 = load_experiment_config(config.s2_config)
    dataset = load_training_sources(s2.data.train_paths).combined
    metadata = build_logical_batch_manifest(
        config,
        puzzle_names=[str(item) for item in dataset["puzzle_name"]],
    )
    print(json.dumps({"status": "complete", "schedule_sha256": metadata["schedule_sha256"], "output": str(output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

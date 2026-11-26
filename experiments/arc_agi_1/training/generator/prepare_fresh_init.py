from __future__ import annotations

# ruff: noqa: E402

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


HC_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
for source_root in (HC_ROOT / "s4_circuit" / "src", HC_ROOT / "s2_hypernet" / "src"):
    sys.path.insert(0, str(source_root))

from s4_circuit.config import load_config
from s4_circuit.initialization import prepare_fresh_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the shared fresh S4 PG trunk.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HC_ROOT, text=True
    ).strip()
    manifest = prepare_fresh_init(config, source_commit=commit)
    print(json.dumps({"status": "complete", "root": config.fresh_init_root, "aggregate_sha256": manifest["aggregate_sha256"], "source_commit": commit}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

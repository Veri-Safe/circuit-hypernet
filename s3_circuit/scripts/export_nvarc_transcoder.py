from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


S3_ROOT = Path(__file__).resolve().parents[1]
HC_ROOT = S3_ROOT.parent
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
sys.path.insert(0, str(S3_ROOT / "src"))

from s3circuit.nvarc_transcoder.config import load_config
from s3circuit.nvarc_transcoder.export import export_selected_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonicalize one selected transcoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_selected_candidate(
        load_config(args.config),
        selection_path=args.selection,
        destination=args.destination,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

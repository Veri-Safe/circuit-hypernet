from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

import torch


S3_ROOT = Path(__file__).resolve().parents[1]
HC_ROOT = S3_ROOT.parent
os.environ.setdefault("HC_ROOT", str(HC_ROOT))
sys.path.insert(0, str(S3_ROOT / "src"))

from s3circuit.nvarc_transcoder.config import load_config
from s3circuit.nvarc_transcoder.trainer import train_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continue one NVARC MLP transcoder on frozen RE-ARC step_399."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics_path", type=Path, default=None)
    parser.add_argument("--run_id", default="production")
    parser.add_argument(
        "--row_indices_path",
        type=Path,
        default=None,
        help="Fixed 5% row-stratified LR-audit manifest; omit for full production.",
    )
    parser.add_argument(
        "--microbatch_tokens", type=int, choices=(2048, 4096, 8192), default=8192
    )
    parser.add_argument(
        "--matrix_compute",
        choices=("fp32_tf32", "bf16_autocast"),
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = (
            HC_ROOT
            / "logs"
            / "train"
            / date.today().isoformat()
            / "nvarc_rearc_transcoder_step399"
            / args.run_id
            / f"layer_{args.layer:02d}.jsonl"
        )
    result = train_layer(
        config,
        layer=args.layer,
        learning_rate=args.learning_rate,
        device=torch.device(args.device),
        resume=args.resume,
        metrics_path=metrics_path,
        run_id=args.run_id,
        row_indices_path=args.row_indices_path,
        microbatch_tokens=args.microbatch_tokens,
        matrix_compute=args.matrix_compute,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

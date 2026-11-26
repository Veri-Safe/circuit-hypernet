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
from s3circuit.nvarc_transcoder.snapshot import (
    load_snapshot_manifest,
    load_snapshot_profile,
)
from s3circuit.nvarc_transcoder.trainer import train_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one NVARC transcoder from an immutable trace snapshot."
    )
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--metrics_path", type=Path, default=None)
    parser.add_argument("--run_id", default="production")
    parser.add_argument("--row_indices_path", type=Path, default=None)
    parser.add_argument("--trace_snapshot_path", type=Path, default=None)
    parser.add_argument(
        "--resumable_percent",
        type=int,
        choices=(5, 10, 20, 25, 50, 100),
        default=None,
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
    profile = load_snapshot_profile(args.profile)
    config = load_config(profile.base_config)
    trace_snapshot_path = args.trace_snapshot_path or Path(profile.snapshot_root)
    manifest = load_snapshot_manifest(trace_snapshot_path)
    if manifest.get("profile_sha256") != profile.semantic_sha256:
        raise ValueError("Frozen snapshot does not belong to this training profile")
    metrics_path = args.metrics_path
    if metrics_path is None:
        metrics_path = (
            HC_ROOT
            / "logs"
            / "train"
            / date.today().isoformat()
            / profile.run_name
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
        trace_snapshot_path=trace_snapshot_path,
        source_trace_snapshot_path=Path(profile.snapshot_root),
        output_root_override=Path(profile.output_root),
        resumable_percent_override=args.resumable_percent,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

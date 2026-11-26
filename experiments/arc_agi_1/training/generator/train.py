from __future__ import annotations

# ruff: noqa: E402

import argparse
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

from s4_circuit.train_runner import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one S4 R1 fusion variant.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--join", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    run_training(
        config_path=args.config,
        join_path=args.join,
        output_dir=args.output,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        initialize_from=args.initialize_from,
        auto_resume=args.auto_resume,
        stop_after_step=args.stop_after_step,
    )


if __name__ == "__main__":
    main()

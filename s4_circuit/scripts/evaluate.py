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

from s4_circuit.evaluation import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one S4 R1 checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-join", type=Path, required=True)
    parser.add_argument("--eval-join", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("matched", "absent", "shuffled"), required=True
    )
    parser.add_argument(
        "--validation-scope",
        choices=("full", "canonical", "author4"),
        default="full",
    )
    parser.add_argument(
        "--max-eval-tasks",
        type=int,
        help=(
            "Deprecated and rejected: score-bearing ARC/NVARC evaluation must "
            "use all 400 tasks."
        ),
    )
    parser.add_argument("--lora-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--lora-norm-diagnostics", action="store_true")
    parser.add_argument("--lora-norm-max-queries-per-rank", type=int)
    parser.add_argument("--experiment-label")
    parser.add_argument(
        "--circuit-intervention", type=Path,
        help="Evaluation-only ARC59 JSON intervention (F3 independent, matched/full, multiplier 1).",
    )
    args = parser.parse_args()
    run_evaluation(
        config_path=args.config,
        checkpoint=args.checkpoint,
        train_join_path=args.train_join,
        eval_join_path=args.eval_join,
        output_dir=args.output,
        mode=args.mode,
        max_eval_tasks=args.max_eval_tasks,
        lora_scale_multiplier=args.lora_scale_multiplier,
        lora_norm_diagnostics=args.lora_norm_diagnostics,
        lora_norm_max_queries_per_rank=args.lora_norm_max_queries_per_rank,
        experiment_label=args.experiment_label,
        validation_scope=args.validation_scope,
        circuit_intervention=args.circuit_intervention,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train one approved composition LOCO fold/route."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


S5_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = S5_ROOT.parent
for source in (S5_ROOT / "src", REPO_ROOT / "s2_hypernet" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from s5_lean_circuit.composition_loco_config import load_config  # noqa: E402
from s5_lean_circuit.composition_loco_train import train  # noqa: E402


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    train(
        load_config(repo_path(args.config)),
        max_steps=args.max_steps,
        resume_path=None if args.resume is None else repo_path(args.resume),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

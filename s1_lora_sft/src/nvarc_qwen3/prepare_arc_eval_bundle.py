#!/usr/bin/env python3
"""Prepare the frozen public ARC-AGI-2 evaluation challenge/solution bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arc_eval_protocol import prepare_arc_eval_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_arc_eval_bundle(args.task_dir, args.output_dir)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()

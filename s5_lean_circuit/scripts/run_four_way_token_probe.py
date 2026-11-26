#!/usr/bin/env python3
"""Entry point for the additive frozen four-way full-token probe experiment."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 's5_lean_circuit/scripts'), str(ROOT / 's5_lean_circuit/src'),
               str(ROOT / 's2_hypernet/src'), str(ROOT / 's3_circuit/src'), str(ROOT / 'temp_s')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    commands = parser.add_subparsers(dest='phase', required=True)
    commands.add_parser('preflight')
    worker = commands.add_parser('run')
    worker.add_argument('--route', required=True, choices=['base_query2048', 'circuit1024', 'hn_query1024', 'query_only_hn1024'])
    commands.add_parser('report')
    args = parser.parse_args()
    from four_way_token_probe.experiment import load_config, preflight, run, report
    config = load_config(args.config)
    if args.phase == 'run':
        run(args.config, config, args.route)
    else:
        {'preflight': preflight, 'report': report}[args.phase](args.config, config)


if __name__ == '__main__':
    main()

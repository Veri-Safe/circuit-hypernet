#!/usr/bin/env python3
"""Reuse the validated historical trainer, with only a fresh output directory."""
import argparse
from common import ROOT, require_sources, repair_config


def main():
    from s5_lean_circuit.composition_loco_train import train
    p = argparse.ArgumentParser(); p.add_argument('--task', required=True); args = p.parse_args()
    require_sources(); config = repair_config(args.task)
    resumes = sorted((ROOT/config.output_dir/'checkpoints').glob('resume_step*.pt'))
    train(config, resume_path=resumes[-1] if resumes else None)
    require_sources()


if __name__ == '__main__':
    main()

# Reproducing the blog results

Use the experiment guides to find the code, evaluation protocol, checkpoints required for a full run and records behind the published numbers.

## Lean first-tactic prediction

[Run the Lean scripts](../experiments/lean/README.md#running-the-scripts) · [Formal suites](../experiments/lean/README.md#evaluation-suites) · [Results and counts](../experiments/lean/README.md#results)

## Probing, ablation and localization

[Probes, interventions, target-specific counts and uncertainty intervals](../experiments/lean/README.md#probing-ablation-and-localization)

## ARC-AGI-1

[Direct inference and TTFT](../experiments/arc_agi_1/README.md) · [Results on 400 public validation tasks](../experiments/arc_agi_1/README.md#results)

## Running original scripts

After [installing the dependencies](../README.md#cpu-checks), run commands from the checkout root. For example, inspect the Lean evaluator’s arguments:

```shell
python experiments/run.py experiments/lean/evaluation/evaluate_performance_completion.py --help
```

The launcher runs experiments in `.circuit-workspace/` by default. A data path such as `datasets/arc_agi_1/` and an output path such as `outputs/nvarc/` are relative to that workspace. To use a sibling directory, pass `--workspace ../circuit-run` before the script name and restore the assets listed in the experiment guide under `../circuit-run/`. Use a separate workspace for each independent run that writes to the same outputs.

The [path map](../experiments/paths.tsv) lists the original and current locations of the released experiment files. The [release manifest](source-files.tsv) lists the SHA-256 of every tracked file except the manifest itself.

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

The launcher prepares a local `.circuit-workspace/` with the paths recorded by the original scripts and runs the selected script there. Generated outputs and relative data paths resolve inside that directory. For a full run, use `--workspace /path/to/workspace` before the script name and restore the assets listed in the relevant experiment guide there. Use a separate workspace for each independent run that writes to the same output paths.

The [path map](../experiments/paths.tsv) lists the original and current locations of the released experiment files. The [release manifest](source-files.tsv) lists the SHA-256 of every tracked file except the manifest itself.

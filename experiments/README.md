# Experiments

- [Lean first-tactic prediction](lean/README.md) follows a problem from circuit extraction through LoRA generation to the second attempt. The guide also links to the [probe and intervention studies](lean/README.md#probing-ablation-and-localization).
- [ARC-AGI-1 grid reasoning](arc_agi_1/README.md) follows the generated adapter through direct inference and test-time fine-tuning. [All six result files](arc_agi_1/README.md#results) contain per-task scores.

To run an original script, give its path in this checkout to [run.py](run.py). It prepares the relative paths used by the original code and configurations. The [path map](paths.tsv) connects filenames in archived records to their current locations. [Running the scripts](../docs/reproduction.md)

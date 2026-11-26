# Circuit-HyperNet

Circuit-HyperNet (C-HyperNet) uses a frozen model’s first attempt to generate a temporary LoRA update for its second attempt. The hypernetwork receives both the original query and a summary of its attribution circuit. The Lean experiments use a 397-dimensional summary. The ARC-AGI-1 implementation uses 59 dimensions.

This repository packages code from the research repository, selected configurations and the evaluation records for the [research blog](paper/blog/20261115_circuit_hypernet/hf_blog/article_en.md). For the formatted article, open `paper/blog/20261115_circuit_hypernet/hf_blog/index.html` locally after cloning.

## Find the implementation

| Blog section | Question | Original code and records |
| --- | --- | --- |
| §2 · Method | How does the first attempt condition the update? | [Lean attribution](s5_lean_circuit/src/s5_lean_circuit/circuit/attribution.py), [397-dimensional descriptor](s5_lean_circuit/src/s5_lean_circuit/circuit/descriptor_v2_397.py), [parameter generator](s5_lean_circuit/src/s5_lean_circuit/hypernet/parameter_generator.py), [LoRA runtime](s5_lean_circuit/src/s5_lean_circuit/hypernet/runtime.py) |
| §§3.1–3.4 · Lean | Does adapting to a circuit help on an unseen composition family? | [Atomic training](s5_lean_circuit/src/s5_lean_circuit/hypernet/formal_training.py), [leave-one-family-out training](s5_lean_circuit/scripts/train_composition_loco.py), [formal evaluation](s5_lean_circuit/scripts/evaluate_performance_completion.py), [results by task and checkpoint](logs/analysis/2026-11-15/lean-suite23-results-review/sources/per_task.csv) |
| §3.5 · ARC-AGI-1 | Can the generated LoRA be applied directly or used to initialize test-time fine-tuning? | [59-dimensional descriptor](s3_circuit/src/s3circuit/nvarc_circuit/vector59.py), [S0 configuration](s4_circuit/configs/selected/s0.yaml), [direct inference](s4_circuit/scripts/evaluate.py), [TTFT initialization](s4_circuit/src/s4_circuit/ttft/warmstart.py), [three-way TTFT comparison](paper/blog/20261115_circuit_hypernet/hf_blog/ttft_results_en.md) |
| §4 · Compositional analysis | What can be read from the representations, and what happens when selected features are ablated? | [Matched attention probe and ranking metric](s5_lean_circuit/scripts/four_way_token_probe/core.py), [audited probe results](logs/analysis/2026-11-22/probe-id-and-correction-v3/local/REPORT.md), [matched ablations](paper/blog/20261115_circuit_hypernet/hf_blog/composition_nodes_en.md) |
| §5 · Functional localization | Do HyperNet responses help select task-relevant features in the base model? | [OR-R graph perturbation and readout](4server/lean_atomic_localize_v1/hn_readout_fusion_v1/targets/S_or_r/graph_readout.py), [selectivity metric](4server/lean_atomic_localize_v1/hn_readout_fusion_v1/targets/S_or_r/metrics.py), [four-target counts and uncertainty](paper/blog/20261115_circuit_hypernet/hf_blog/localization_results_en.md) |

## Main results

On Lean, the metric is greedy **first-tactic exact match**. Each formal suite contains 500 questions per family. The four atomic families and five composition families are scored separately. The last row holds out one composition family per training fold and averages across both formal suites.

| Lean accuracy (%) | Base / Direct SFT | Query-only | Raw-weight | C-HyperNet |
| --- | ---: | ---: | ---: | ---: |
| Atomic families after atomic training | 43.98 | 69.00 | 65.25 | 71.00 |
| Zero-shot composition after atomic training | 0.06 | 2.38 | 4.26 | 6.22 |
| Held-out composition after stage-2 training | 6.34 | 4.60 | 10.58 | 13.14 |

The first two rows use the atomic-trained base model in the first column. The last row uses Direct SFT continued on the seen composition families. C-HyperNet outscored Raw-weight on And-Or (43.00% versus 27.00%) and scored lower on Nested OR (22.70% versus 25.90%). C-HyperNet scored zero on the other three held-out families. [Atomic-stage counts](logs/analysis/2026-11-15/blog-evidence/sources/remote/logs/eval/2026-11-14/lean-suite2-suite3-completion/results/atomic_stage_summary.csv) · [Five-fold counts](logs/analysis/2026-11-15/lean-suite23-results-review/sources/four_route_fivefold_summary.csv)

On ARC-AGI-1 public validation, each of the 400 tasks contributes equally to `selected_exact@2`. The denominator includes 419 test inputs. Evaluating the public validation set also informed checkpoint selection.

| Direct inference | Accuracy (%) | Test-time fine-tuning | Accuracy (%) |
| --- | ---: | --- | ---: |
| Frozen base | 16.375 | Original base, standard initialization | 42.625 |
| Direct SFT | 17.125 | SFT base, standard initialization | 45.875 |
| C-HyperNet | 17.250 | SFT base, generated initialization | 47.875 |

For direct inference, the generator uses the problem representation and its 59-dimensional circuit summary. For TTFT, one generated adapter from the first test input’s canonical view initializes each task. Holding the SFT base and adaptation settings fixed gives a 2.000 percentage-point gain. [ARC protocols and original per-task records](docs/reproduction.md#arc-agi-1)

## Run the CPU checks

From the repository root in a Python environment, run:

```shell
python -m pip install -r requirements-review.txt
python -m pytest -q
```

The tests exercise the original graph descriptor, parameter generator, probe, scoring contracts and TTFT logic. Published model scores come from the linked evaluation records. Full training and inference require separate checkpoints, tokenizers, data and circuit caches. The [reproduction guide](docs/reproduction.md) lists the entry points, resource requirements and evaluation splits.

## Release contents

The [release manifest](docs/source-files.tsv) lists the packaged files and their SHA-256 digests. The [reproduction guide](docs/reproduction.md) maps paper results to the measurements used to compute them. Result records retain the scientific counts. Setup paths and launch settings are portable.

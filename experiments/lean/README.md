# Lean first-tactic prediction

Given a goal such as `(p ∧ q) ∨ r` and a list of premises, the base model first predicts a proof tactic. C-HyperNet uses that attempt to condition a second prediction for the same problem. The evaluation scores an exact match to the reference **first tactic**. [Task families and example proof](../../blog/experiments_en.md)

1. **Extract the circuit.** Trace the frozen model's first attempt and summarize the attribution graph in 397 dimensions. [Attribution](circuits/attribution.py) · [Circuit descriptor](circuits/descriptor_v2_397.py)
2. **Generate and apply an adapter.** The hypernetwork takes the original query and this descriptor, generates a LoRA update and applies it to the same base model for the second attempt. [Parameter generator](adapters/parameter_generator.py) · [LoRA application](adapters/runtime.py)
3. **Score the first tactic.** The formal evaluator runs greedy decoding and records correct counts by route, task, checkpoint and bank. The records call the circuit-conditioned route `circuit_a`. [Evaluator](evaluation/evaluate_performance_completion.py) · [Per-task counts](results/first_tactic/per_task.csv)

## Results

| Method | Atomic tasks after atomic training (%) | Zero-shot composition after atomic training (%) | Held-out composition after stage 2 (%) |
| --- | ---: | ---: | ---: |
| Base model (atomic-task SFT) | 43.98 | 0.06 | — |
| Direct SFT (continued on seen compositions) | — | — | 6.34 |
| Query-only | 69.00 | 2.38 | 4.60 |
| Raw-weight | 65.25 | 4.26 | 10.58 |
| C-HyperNet | 71.00 | 6.22 | 13.14 |

The Base model row uses the backbone fine-tuned on the four atomic tasks. Direct SFT continues training that backbone on seen composition families. The three hypernetworks keep the backbone frozen. For held-out compositions, C-HyperNet's two-suite mean of 13.14% includes 43.00% on And-Or, 22.70% on Nested OR and zero on the other three families. [Results by family](results/first_tactic/per_task.csv)

## Evaluation suites

The two formal suites have 500 examples per family. Historical result files use older suite labels. Use this mapping to reproduce the current article. The [evaluation manifest](evaluation/pairing_control/eval_manifest.json) gives the full paths and content hashes.

| Blog label | `paper_suite` in the results CSV | Bank filename | Generation seed |
| --- | --- | --- | ---: |
| Suite 1 | Suite 2 | `suite2.pt` | 730000000000 |
| Suite 2 | Suite 3 | `suite1.pt` | 720000000000 |

For results after atomic training, the four atomic families contribute 4,000 predictions across the two suites. The five composition families contribute 5,000. During compositional training, each fold reserves one family, selects a checkpoint using seen-family validation, and evaluates the held-out family on both formal suites. There are 2,500 held-out predictions per suite. [Atomic-stage counts](results/first_tactic/atomic_stage_summary.csv) · [Five-fold counts](results/first_tactic/four_route_fivefold_summary.csv)

## Training and pairing controls

During the first stage, train the hypernetwork on the four atomic tasks. In stage two each fold trains on four composition families and holds out the fifth. For the circuit route on Or-And and Nested AND, the blog uses the checkpoints from the repair runs. [Atomic training](training/atomic/formal_training.py) · [Atomic configuration](training/atomic/configs/a_p03.json) · [Fold training](training/composition/train_composition_loco.py) · [Repair training](training/composition/checkpoint_repair/train.py) · [Repair configurations](training/composition/checkpoint_repair/configs)

With ordinary training, C-HyperNet gets 657/5,000 held-out predictions correct. Shuffling the student query–circuit pairings during stage-2 training and evaluating with the correct pairing gives 351/5,000. Swapping summaries within a proof family only at inference, after ordinary training, gives 658/5,000. The inference swap preserves the family identity. Its result leaves open whether the model identifies a mistake specific to an individual problem. [Training permutation](training/pairing_control/composition_stage2_permuted.py) · [Shuffled-training counts](results/pairing_controls/ood_five_rows.csv) · [Inference controls](results/first_tactic/circuit_fivefold_controls.csv)

## Probing, ablation and localization

### Atomic probes

The three main probe columns can be read from the [original summary](results/atomic_probes/summary.json) by selecting `domain=composition`, `condition=clean_no_identity`, `method=token_attention` and `task_type=ALL`. The task is to rank all relevant atomic operations above all irrelevant ones on 2,500 questions. The three counts are 599 for the base query, 1,032 for the circuit encoder and 1,270 for the HyperNet query. [Metric and probe](analysis/atomic_probes/core.py) · [Audited report](results/atomic_probes/REPORT.md)

### Composition ablations

The And-Or and Nested OR interventions compare C-HyperNet and Query-only on a separate development bank with 500 questions per family. Within each comparison the two models use the same node count and intervention strength. Their starting accuracies and node counts differ between the two tasks. [And-Or circuit route](analysis/composition_ablation/and_or/circuit/evaluate_block.py) · [And-Or query route](analysis/composition_ablation/and_or/query_only/evaluate_block.py) · [Nested OR circuit route](analysis/composition_ablation/nested_or/circuit/evaluation/evaluate.py) · [Nested OR query route](analysis/composition_ablation/nested_or/query_only/evaluate.py) · [Counts and selection details](../../blog/composition_nodes_en.md)

| Held-out family | C-HyperNet group and correct counts | Query-only group and correct counts |
| --- | --- | --- |
| And-Or | [c25-b00-cross_attn-selective-128, 221 → 78](results/composition_ablation/and_or/circuit/block0_full500_summary.json) | [qm12-b00-residual-selective-128, 106 → 62](results/composition_ablation/and_or/query_only/candidate_b00_summary.json) |
| Nested OR | [grad_signed_all_k008_a000, 112 → 73](results/composition_ablation/nested_or/circuit/continuation_summary.json) | [q12_signed_all_k008_a000, 95 → 84](results/composition_ablation/nested_or/query_only/combined_summary.json) |

### Base localization

Each target has its own fresh bank of 100 examples per atomic task. Paired direct and HyperNet-assisted interventions act on the same base model and share the same bank for that target. The mean target-to-other-task selectivity ratio increases from 3.62 to 5.04. All four reported paired 95% intervals for the advantage statistic include zero. The [localization note](../../blog/localization_results_en.md) explains the intervention and uncertainty estimates. The [four-target evidence table](results/base_localization/recomputed_metrics.json) aggregates the formal records below.

| Target and confirmation | Implementation | Counts from the formal run |
| --- | --- | --- |
| AND · attempt 001 | [Scoring](analysis/base_localization/S_and/metrics.py) | [Formal metrics](results/base_localization/S_and/attempt_001/formal_metrics.json) |
| OR-L · attempt 008 | [Scoring](analysis/base_localization/S_or_l/metrics.py) | [Formal metrics](results/base_localization/S_or_l/attempt_008/formal_metrics.json) |
| OR-R · attempt 001 | [Graph perturbation and readout](analysis/base_localization/S_or_r/graph_readout.py) and [scoring](analysis/base_localization/S_or_r/metrics.py) | [Completed formal evaluation](results/base_localization/S_or_r/attempt_001/formal_summary.json) |
| ID · attempt 001 | [Confirmation and scoring](analysis/base_localization/S_id/confirmation.py) | [Formal result](results/base_localization/S_id/attempt_001/formal_result.json) |

## Running the scripts

From the repository root, inspect an entry point without loading model weights or a question bank.

```shell
python experiments/run.py experiments/lean/circuits/run_circuit_graph_pilot.py --help
python experiments/run.py experiments/lean/circuits/vectorize_payload_v2_397.py --help
python experiments/run.py experiments/lean/training/composition/train_composition_loco.py --help
python experiments/run.py experiments/lean/evaluation/evaluate_performance_completion.py --help
```

For a full run, pass `--workspace PATH` before the script name and supply the original base weights and tokenizer, transcoders, packed training and replay caches, fold indices, hypernetwork checkpoints, and both formal banks there. The evaluator checks the recorded bank and checkpoint identities. Some archived configs include fields for a withdrawn 2,000-per-family bank. Use the two 500-per-family banks listed above for formal results. Circuit extraction can regenerate traces from the pinned base and transcoders. Exact replay of historical traces with different profile identifiers requires their verified manifest. [Evaluation manifest](evaluation/pairing_control/eval_manifest.json) · [CPU checks](../../README.md#cpu-checks)

Attribution for the Lean runtime and task generator is available in the [third-party notice](THIRD_PARTY_NOTICE.md) and [upstream license](licenses/circuit-extraction-stability/LICENSE).

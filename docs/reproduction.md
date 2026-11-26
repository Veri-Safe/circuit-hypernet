# Reproducing the blog results

The [blog](../paper/blog/20261115_circuit_hypernet/hf_blog/article_en.md) identifies each experiment by its scientific question. The original code uses names such as `circuit_a` for the Lean circuit-conditioned model and `S0` for the selected ARC generator. These names are also the keys in the result files.

## Lean first-tactic prediction

Each example supplies a goal and premises. Scoring uses exact match against the first reference tactic under greedy decoding. The two formal suites have 500 examples per family. Historical result files use older suite labels. Use this mapping to reproduce the current article. The [evaluation manifest](../4server/lean_stage2_permuted_formal_eval_v1/eval_manifest.json) gives the full paths and content hashes.

| Blog label | `paper_suite` in the results CSV | Bank filename | Generation seed |
| --- | --- | --- | ---: |
| Suite 1 | Suite 2 | `suite2.pt` | 730000000000 |
| Suite 2 | Suite 3 | `suite1.pt` | 720000000000 |

For results after atomic training, the four atomic families contribute 4,000 predictions across the two suites. The five composition families contribute 5,000. During compositional training, each fold reserves one family, selects a checkpoint using seen-family validation, and evaluates the held-out family on both formal suites. There are 2,500 held-out predictions per suite. [Atomic-stage counts](../logs/analysis/2026-11-15/blog-evidence/sources/remote/logs/eval/2026-11-14/lean-suite2-suite3-completion/results/atomic_stage_summary.csv) · [Five-fold counts](../logs/analysis/2026-11-15/lean-suite23-results-review/sources/four_route_fivefold_summary.csv)

| Operation | Where to look |
| --- | --- |
| Extract an attribution graph and compute its descriptor | [Graph extraction](../s5_lean_circuit/scripts/run_circuit_graph_pilot.py) and [397-dimensional vectorization](../s5_lean_circuit/scripts/vectorize_payload_v2_397.py) |
| Generate an adapter from the query and circuit | [Parameter generator](../s5_lean_circuit/src/s5_lean_circuit/hypernet/parameter_generator.py) and [adapter application](../s5_lean_circuit/src/s5_lean_circuit/hypernet/runtime.py) |
| Fit the atomic-stage hypernetwork | [Training module](../s5_lean_circuit/src/s5_lean_circuit/hypernet/formal_training.py) and [selected circuit configuration](../s5_lean_circuit/configs/hypernet_atomic_1p2m_seed42/a_p03.json) |
| Continue training on four composition families | [Fold training entry point](../s5_lean_circuit/scripts/train_composition_loco.py). The [per-task evaluation ledger](../logs/analysis/2026-11-15/lean-suite23-results-review/sources/per_task.csv) identifies the config, checkpoint, and hashes for each route and fold. For the circuit route on Or-And and Nested AND, use the [repair training entry point](../4server/1105/section4_results_completion_v1/train.py) with the [adopted repair configurations](../4server/1105/section4_results_completion_v1/configs). |
| Recompute after loading the original assets | [Formal evaluator](../s5_lean_circuit/scripts/evaluate_performance_completion.py) |

To inspect the argument names without loading a model or question bank, run these from the repository root.

```shell
python s5_lean_circuit/scripts/run_circuit_graph_pilot.py --help
python s5_lean_circuit/scripts/vectorize_payload_v2_397.py --help
python s5_lean_circuit/scripts/train_composition_loco.py --help
python s5_lean_circuit/scripts/evaluate_performance_completion.py --help
```

**Two distinct pairing controls.** With ordinary training, C-HyperNet gets 657/5,000 held-out predictions correct. Shuffling the student query–circuit pairings during stage-2 training and evaluating with the correct pairing gives 351/5,000. Swapping summaries within a proof family only at inference, after ordinary training, gives 658/5,000. The inference swap preserves the family identity. Its result leaves open whether the model identifies a mistake specific to an individual problem. [Training permutation](../s5_lean_circuit/src/s5_lean_circuit/composition_stage2_permuted.py) · [Shuffled-training counts](../logs/analysis/2026-11-23/lean-stage2-permuted-formal-eval-v1/results/ood_five_rows.csv) · [Inference controls](../logs/analysis/2026-11-15/lean-suite23-results-review/sources/circuit_fivefold_controls.csv)

Full execution requires the original base weights and tokenizer, transcoders, packed training and replay caches, fold indices, hypernetwork checkpoints, and the two formal banks. Supply the versions identified by the configs and result records. Historical configs retain legacy fields for an earlier 2,000-per-family protocol as part of their recorded identities. That bank is withdrawn. Use only the two 500-per-family banks above for the formal results. The shuffled-evaluation preflight checks the packaged sources and the original bank and checkpoint identities. The released extraction profile can regenerate circuit traces from the pinned base and transcoders. Historical trace metadata may use different profile identifiers and requires a separately verified manifest for exact replay. The checkout alone supports component testing and inspection of the published counts.

## Probing, ablation and localization

**§4.1.** The three main probe columns can be read from the [original summary](../logs/analysis/2026-11-22/probe-id-and-correction-v3/local/summary.json) by selecting `domain=composition`, `condition=clean_no_identity`, `method=token_attention` and `task_type=ALL`. The task is to rank all relevant atomic operations above all irrelevant ones on 2,500 questions. The three counts are 599 for the base query, 1,032 for the circuit encoder and 1,270 for the HyperNet query. [Metric and probe](../s5_lean_circuit/scripts/four_way_token_probe/core.py) · [Audited report](../logs/analysis/2026-11-22/probe-id-and-correction-v3/local/REPORT.md)

**§4.2.** The And-Or and Nested OR interventions compare C-HyperNet and Query-only on a separate development bank with 500 questions per family. Within each comparison the two models use the same node count and intervention strength. Their starting accuracies and node counts differ between the two tasks. [And-Or circuit route](../4server/lean_stage2_and_or_groups_c25_dev500_9task_v1/evaluate_block.py) · [And-Or query route](../4server/lean_stage2_and_or_groups_query_m12_dev500_9task_v1/evaluate_block.py) · [Nested OR circuit route](../4server/lean_stage2_nested_or_groups_v1/gradient_contrast_r03_pareto_v1/evaluate.py) · [Nested OR query route](../4server/lean_stage2_nested_or_groups_query_q12_v1/evaluate.py) · [Counts and selection details](../paper/blog/20261115_circuit_hypernet/hf_blog/composition_nodes_en.md)

| Held-out family | C-HyperNet group and correct counts | Query-only group and correct counts |
| --- | --- | --- |
| And-Or | [c25-b00-cross_attn-selective-128, 221 → 78](../logs/eval/2026-11-11/lean-stage2-and-or-groups-c25-dev500-9task-v1/block0_full500_summary.json) | [qm12-b00-residual-selective-128, 106 → 62](../logs/eval/2026-11-12/lean-stage2-and-or-groups-query-m12-dev500-9task-v1/candidate_b00_summary.json) |
| Nested OR | [grad_signed_all_k008_a000, 112 → 73](../logs/eval/2026-11-14/lean-stage2-nested-or-groups-v1/continuation_summary.json) | [q12_signed_all_k008_a000, 95 → 84](../logs/eval/2026-11-14/lean-stage2-nested-or-query-q12-v1/site_followup/combined_summary.json) |

**§5.** Each target has its own fresh bank of 100 examples per atomic task. Paired direct and HyperNet-assisted interventions act on the same base model and share the same bank for that target. The mean target-to-other-task selectivity ratio increases from 3.62 to 5.04. All four reported paired 95% intervals for the advantage statistic include zero. The [four-target evidence table](../paper/blog/20261115_circuit_hypernet/revisions/20261117_17_analysis2_atomic_table/evidence/recomputed_metrics.json) aggregates the formal records below.

| Target and confirmation | Implementation | Counts from the formal run |
| --- | --- | --- |
| AND · attempt 001 | [Scoring](../4server/lean_atomic_localize_v1/hn_readout_fusion_v1/targets/S_and/metrics.py) | [Formal metrics](../logs/analysis/2026-11-14/lean-atomic-hn-readout-fusion-v1/S_and/attempt_001/formal_metrics.json) |
| OR-L · attempt 008 | [Scoring](../4server/lean_atomic_localize_v1/hn_readout_fusion_v1/targets/S_or_l/metrics.py) | [Formal metrics](../logs/analysis/2026-11-14/lean-atomic-hn-readout-fusion-v1/S_or_l/attempt_008/formal_metrics.json) |
| OR-R · attempt 001 | [Graph perturbation and readout](../4server/lean_atomic_localize_v1/hn_readout_fusion_v1/targets/S_or_r/graph_readout.py) | [Completed formal evaluation](../logs/analysis/2026-11-14/lean-atomic-hn-readout-fusion-v1/S_or_r/attempt_001/formal/formal_summary.json) |
| ID · attempt 001 | [Confirmation and scoring](../4server/lean_atomic_localize_v1/hn_readout_fusion_v1/targets/S_id/confirmation.py) | [Formal result](../logs/analysis/2026-11-14/lean-atomic-hn-readout-fusion-v1/S_id/attempt_001/formal/formal_result.json) |

## ARC-AGI-1

ARC circuit extraction defaults to **one attribution row per backward pass**. This limits the replay activations held at once. Feature selection still uses the logical row batch and update interval in the Stage-B configuration. To trade memory for throughput, pass an explicit row limit such as `--compute-microbatch-cap 4` to either extraction command. The input policy supplies the upper bound. [Single-work-unit extractor](../s3_circuit/scripts/extract_nvarc_circuit_payloads.py) · [Worker extractor](../s3_circuit/scripts/extract_nvarc_circuit_payload_worker.py)

Use a fresh output directory when changing the effective cap. Both commands verify the input policy and save the effective policy at `policies/<policy_sha256>.json` inside their output directory. Use that file as `--policy` for subsequent materialization and finalization. The worker passes it automatically when a compatible `--materializer` is supplied and automatic materialization is enabled. The scores below come from the archived evaluations.

The direct experiment uses a Qwen3-4B backbone trained on RE-ARC, a 59-dimensional descriptor and a rank-16 generated adapter. TTFT uses the generated adapter from the first test input’s canonical view to initialize one adapter per task. The [original warm-start implementation](../s4_circuit/src/s4_circuit/ttft/warmstart.py) embeds the rank-16 update into the rank-256 rsLoRA adapter used for fine-tuning. The three TTFT rows below share the learning rate of 2e-5 and 128 optimizer calls per task.

| Experiment | Original per-task result |
| --- | --- |
| Frozen base, direct | [16.375%](../logs/eval/2026-08-04/0804_rearc_scale_50ep_v2_r8_s42_step130/metrics.json) |
| Direct SFT, direct | [17.125%](../logs/eval/2026-08-19/0818_rearc8_s42_step130_to_seed24_exclusive_skip_b1_lr1e5_6ep_v1_step65/metrics.json) |
| C-HyperNet, direct | [17.250%](../logs/eval/2026-11-05/section4-results-completion-v1/arc/metrics.json) |
| Original base, standard TTFT | [42.625%](../logs/analysis/2026-11-18/blog-ttft-base-result/receipts/lr2e-5/metrics.json) |
| SFT base, standard TTFT | [45.875%](../outputs/nvarc/direct_sft_ttft/step65_final128_20261110_v1/lr2e-5/metrics.json) |
| SFT base, generated TTFT initialization | [47.875%](../outputs/nvarc/direct_sft_ttft/s0_step300_warmstart_step65_20261110_v1/lr2e-5/metrics.json) |

For each JSON, average the 400 values in `score_details["selected_exact@2"]["per_task_score"]` and multiply by 100. Each task has equal weight even when it has multiple test inputs. For the matched SFT comparison the task-score sums are 183.5 and 191.5. All six records cover the same 400 public validation tasks and 419 test inputs. With another learning rate the original study observed 49.000% under standard initialization. The 2.000-point difference above quantifies the effect of changing initialization at the matched setting. [Original TTFT comparison](../paper/blog/20261115_circuit_hypernet/hf_blog/ttft_results_en.md)

The [direct evaluation entry point](../s4_circuit/scripts/evaluate.py) needs the trained weights, tokenizer, transcoders and matching join indices. The [TTFT runner](../s4_circuit/scripts/run_s0_warmstart_ttft_job.py) also requires precomputed circuits, the task data and the upstream notebook at `ref_repo/NVARC/ARC-AGI1/002_ivan_arc1.ipynb`. Restore these assets separately for full replay. Map the packaged `training_circuits`, `eval_circuits` and `selected_generator` directories to the original training circuits, evaluation circuits and generator checkpoint. The S0 loader verifies the mapped artifacts. For distributed training, the launcher supplies `WORLD_SIZE` and `LOCAL_WORLD_SIZE`. For TTFT, set `ARC_TTFT_WORKERS` to the desired concurrency. `ARC_TTFT_LOCAL_WORKERS` optionally sets local concurrency for the generated initialization. The [S0 identity records](../logs/eval/2026-11-05/section4-results-completion-v1/arc) identify the selected checkpoint and the configuration used by the released loader. [Pinned upstream NVARC notebook](https://github.com/1ytic/NVARC/blob/846d0198efa752534594e321fc3289fc0a06c657/ARC-AGI1/002_ivan_arc1.ipynb)

## Release contents

The implementation was adapted from the research repository for standalone review. The [file manifest](source-files.tsv) gives the SHA-256 of each packaged file except the manifest itself. Counts, task IDs and metric definitions in the linked evidence identify the results in the article. Runtime settings are supplied separately when running the code in another environment.

The Lean runtime and task-generator attribution are documented in the [original notice](../s5_lean_circuit/THIRD_PARTY_NOTICE.md) and [upstream license](../s5_lean_circuit/upstream/circuit-extraction-stability/LICENSE). Components derived from HY-WU carry the [Tencent Hunyuan Community License Agreement](../ref_repos/HY-WU/LICENSE) and [notice](../ref_repos/HY-WU/Notice.txt).

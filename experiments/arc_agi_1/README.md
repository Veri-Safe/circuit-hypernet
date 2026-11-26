# ARC-AGI-1 grid reasoning

Starting with Qwen3-4B trained on RE-ARC, C-HyperNet combines a problem representation with a 59-dimensional circuit summary to generate a rank-16 LoRA. The experiments use that adapter in two ways. [Circuit descriptor](circuits/vector59.py) · [Adapter model](adapters/model.py) · [Selected generator configuration](training/generator/configs/s0.yaml)

| How the adapter is used | Follow the implementation |
| --- | --- |
| **Direct inference.** Apply the generated LoRA to the frozen base model and decode candidate grids for the test input. | [Direct evaluator](direct_inference/evaluate.py) |
| **Test-time fine-tuning (TTFT).** Take the adapter from the canonical view of the first test input, embed its rank-16 update in a rank-256 rsLoRA adapter, then fine-tune that adapter on the task's examples. The task shares one adapter across its test inputs. | [Warm-start implementation](test_time_finetuning/warmstart.py) · [Per-task runner](test_time_finetuning/run_s0_warmstart_ttft_job.py) |

## Results

Each row covers the same 400 public validation tasks and 419 test inputs. Under `selected_exact@2`, an input is correct if either selected grid exactly matches the target. The score averages across inputs within a task, then equally across tasks. The public validation set also informed checkpoint selection.

| Experiment | Original per-task result |
| --- | --- |
| Frozen base model, direct | [16.375%](results/direct_inference/frozen_base/metrics.json) |
| Direct SFT, direct | [17.125%](results/direct_inference/direct_sft/metrics.json) |
| C-HyperNet, direct | [17.250%](results/direct_inference/circuit_hypernet/metrics.json) |
| Original base, standard TTFT | [42.625%](results/test_time_finetuning/original_base/metrics.json) |
| Supervised fine-tuned base model, standard TTFT | [45.875%](results/test_time_finetuning/standard_initialization/metrics.json) |
| Supervised fine-tuned base model, generated TTFT initialization | [47.875%](results/test_time_finetuning/generated_initialization/metrics.json) |

For each linked JSON, average the 400 values in `score_details["selected_exact@2"]["per_task_score"]` and multiply by 100. In direct inference, C-HyperNet and Direct SFT used different training budgets. The 0.125-point difference is descriptive. An ARC query-only control would be needed to isolate the circuit vector's contribution. [Blog results and protocol](../../blog/experiments_en.md)

All three TTFT runs use 128 augmented training examples, 128 optimizer updates per task and a learning rate of 2e-5. With the same SFT base and fine-tuning settings, the task-score sums rise from 183.5 to 191.5, giving the **2.000-point** gain from generated initialization. Standard initialization reached 49.000% at a different learning rate. [TTFT settings and comparison](../../blog/ttft_results_en.md)

## Extracting circuits

ARC circuit extraction defaults to **one attribution row per backward pass**. This limits the replay activations held at once. Feature selection still uses the logical row batch and update interval in the Stage-B configuration. To trade memory for throughput, pass an explicit row limit such as `--compute-microbatch-cap 4` to either extraction command. The input policy supplies the upper bound. [Single-work-unit extractor](circuits/extract_nvarc_circuit_payloads.py) · [Worker extractor](circuits/extract_nvarc_circuit_payload_worker.py)

Use a fresh output directory when changing the effective cap. Both commands verify the input policy and save the effective policy at `policies/<policy_sha256>.json` inside their output directory. Use that file as `--policy` for subsequent materialization and finalization. The worker passes it automatically when a compatible `--materializer` is supplied and automatic materialization is enabled.

## Running the scripts

From the repository root, inspect the direct evaluator and the circuit extraction entry point.

```shell
python experiments/run.py experiments/arc_agi_1/direct_inference/evaluate.py --help
python experiments/run.py experiments/arc_agi_1/circuits/extract_nvarc_circuit_payloads.py --help
```

For a full run, pass `--workspace PATH` before the script name. Restore the trained weights, tokenizer, transcoders and matching join indices there. TTFT also requires the task data, precomputed circuits and the upstream notebook at `ref_repo/NVARC/ARC-AGI1/002_ivan_arc1.ipynb`. Supply the original circuit artifacts and generator checkpoint under the paths for `training_circuits`, `eval_circuits` and `selected_generator`. The S0 loader verifies these artifacts. [S0 identity records](results/direct_inference/circuit_hypernet) · [Pinned NVARC notebook](https://github.com/1ytic/NVARC/blob/846d0198efa752534594e321fc3289fc0a06c657/ARC-AGI1/002_ivan_arc1.ipynb)

For an archived S0 checkpoint whose identity includes absolute paths, restore the trainer’s original base checkpoint manifest alongside its checkpoint directory or set `ARC_ARCHIVE_ROOT` to its original mount alias for the workspace.

For distributed training, the launcher supplies `WORLD_SIZE` and `LOCAL_WORLD_SIZE`. For TTFT, set `ARC_TTFT_WORKERS` to the desired concurrency. `ARC_TTFT_LOCAL_WORKERS` optionally sets local concurrency for generated initialization.

Components derived from HY-WU carry the [Tencent Hunyuan Community License Agreement](licenses/HY-WU/LICENSE) and [notice](licenses/HY-WU/Notice.txt).

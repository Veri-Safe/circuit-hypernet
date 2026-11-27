# Circuit-HyperNet

**Research blog**  <https://hitzjq.github.io/Circuit-HyperNet-Blog/>

[Report (PDF)](https://hitzjq.github.io/Circuit-HyperNet-Blog/Circuit-HyperNet.pdf) · [Cite this work (Zenodo)](https://doi.org/10.5281/zenodo.24566714)

Circuit-HyperNet uses the computation behind a model’s first answer to generate a temporary LoRA update for its second attempt. The hypernetwork reads the query and a summary of the first attempt’s attribution graph.

[Article in Markdown](blog/article_en.md) · [Lean experiments](experiments/lean/README.md) · [ARC-AGI-1 experiments](experiments/arc_agi_1/README.md)

| Question | Where to start |
| --- | --- |
| How does a circuit become an update? | Follow a Lean problem from the [397-dimensional circuit summary](experiments/lean/circuits/descriptor_v2_397.py) through the [LoRA generator](experiments/lean/adapters/parameter_generator.py) to the [second attempt](experiments/lean/adapters/runtime.py). |
| Does it transfer to unseen logical compositions? | On Lean, C-HyperNet predicts the exact first tactic on **13.14%** of held-out problems, compared with **10.58%** for Raw-weight and **6.34%** for Direct SFT. Correct predictions come from two of the five held-out families. [Results and per-family counts](experiments/lean/README.md#results) |
| How does the generated LoRA help on ARC-AGI-1? | The adapter supports [direct inference](experiments/arc_agi_1/direct_inference/evaluate.py) and [initialization for test-time fine-tuning](experiments/arc_agi_1/test_time_finetuning/warmstart.py). With the same SFT base and tuning settings, generated initialization raises accuracy from **45.875% to 47.875%**. [All six evaluations](experiments/arc_agi_1/README.md#results) |
| Which internal features matter? | Explore the Lean [atomic probes, composition ablations and Base localization](experiments/lean/README.md#probing-ablation-and-localization). |

## CPU checks

From the repository root, install the review dependencies and run the component tests.

```shell
python -m pip install -r requirements-review.txt
python -m pytest -q
```

The evaluation records are included. Full model runs require separate weights, data and circuit caches. [Running experiment scripts and restoring assets](docs/reproduction.md)

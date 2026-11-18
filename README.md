# Functional memory

Browse the result collections below for figures, comparisons, and findings. `experiments/` stores the
runs behind them: one permanent ID per execution, with its recipe, measurements,
provenance, checkpoint, and log together. A study can vary several settings at once;
a result collection can combine runs from several studies.

| Study | Results | Run data and reproduction |
| --- | --- | --- |
| Attention scaling: softmax, Taylor, Hermite, learned polynomials, and Gated DeltaNet | [Heatmaps and degree curves](results/attention_scaling/README.md) | [10B attention runs](experiments/attention_scaling/README.md) |
| Input-dependent (tanh) polynomial coefficients | [Matched-degree comparison](results/input_dependent_coefficients/README.md) | [10B attention runs](experiments/attention_scaling/README.md#input-dependent-coefficient-comparison) |
| Polynomial Gated DeltaNet: independent erase, polynomial memory, key width | [Heatmap and validation gap](results/gdn_polynomial/README.md) | [10B attention runs](experiments/attention_scaling/README.md#polynomial-gated-deltanet) |
| Rational polynomial attention: separate learned denominator | [Comparison by denominator](results/rational_polynomial/README.md) | [10B attention runs](experiments/attention_scaling/README.md) |
| Transformer++: attention variants under its recipe, with model shape and token budget per run | [Results and exploratory diagnostics](experiments/transformer_pp/results/README.md) | [Recipe and runs](experiments/transformer_pp/README.md) |
| Transformer++ degree scaling: fixed Taylor, fixed Hermite, learned static, and query coefficients at `m = 1-8` and 12 | [Perplexity by degree](experiments/transformer_pp/results/degree_scaling/README.md) | [Recipe and runs](experiments/transformer_pp/README.md#degree-scaling) |
| Transformer++ at the 12-layer width-640 shape: eight attention variants against their reference-shape partners, and the variants at 1×, 2× and 4× the shape's budget | [Shape index](experiments/transformer_pp/results/l12_d640/README.md): [shape pairs](experiments/transformer_pp/results/l12_d640/shape_pairs/README.md) and [budgets](experiments/transformer_pp/results/l12_d640/budgets/README.md) | [Recipe and runs](experiments/transformer_pp/README.md#model-scale) |
| Transformer++ score MLP: a conventional SiLU MLP of the score and query token in place of exp, at `J = 4` and 8 | [Learning curves and comparison](experiments/transformer_pp/results/score_mlp/README.md) | [Recipe and runs](experiments/transformer_pp/README.md#score-mlp) |
| Transformer++ spline attention: a learned quartic B-spline of the score, static or query-dependent, from four initial functions | [Initial functions and controls](experiments/transformer_pp/results/spline_attention/README.md) | [Recipe and runs](experiments/transformer_pp/README.md#spline-attention) |
| State rank | [Figures and findings](results/state_rank/README.md) | [Runs and execution](experiments/state_rank/README.md) |

Exploratory results awaiting a decision (promote or discard) are in
[`results/exploratory/`](results/exploratory/README.md). Closed diagnostic records of
training behavior, such as gradient spikes, are in
[`results/diagnostics/`](results/diagnostics/README.md).

The established 10B attention settings vary within `experiments/attention_scaling/runs/<run_id>/`.
The separately requested Transformer++ study owns its recipe, runs, results, and exploratory
diagnostics under `experiments/transformer_pp/`; the folder keeps its original name. Mechanisms and sweeps remain
runs within their applicable study. The [10B run index](experiments/attention_scaling/runs.csv)
and [Transformer++ run index](experiments/transformer_pp/runs.csv) expose the settings,
home cluster, training budget, status, and outcome. Each result collection selects
its inputs explicitly, so the same run can participate in several comparisons.

The maintained 10B attention results are the combined heatmap and degree plot in
`results/attention_scaling/`, the input-dependent coefficient comparison in
`results/input_dependent_coefficients/`, the polynomial GDN comparison in
`results/gdn_polynomial/`, and the rational polynomial comparison in
`results/rational_polynomial/`. Run inventories, learning-curve data, coefficient
diagnostics, and execution metadata remain under their studies; result directories
organize the comparisons we want to present.

```text
results/<analysis>/
  README.md, selection.yaml, *.pdf, *.png, summary.csv
  plot.py, ...
experiments/<study>/
  README.md, sweep.yaml, runs.csv
  runs/<run_id>/
    config.yaml, record.json, metrics.jsonl
    checkpoints/, training.log                 # ignored artifacts
  acceptance.py, ...                           # optional execution code
scaffolds/<capability>/                        # shared implementation and tests
```

Settings such as key dimension, Taylor degree, normalization, coefficient constraints,
sharing, and seed are fields in the recipe and run index. A new execution gets a new
ID; a resume keeps its ID. Figures select explicit IDs, so adding another run cannot
silently change a published comparison. The canonical measurements stay under the run.

See [AGENTS.md](AGENTS.md) for layout and retention rules, [CLUSTERS.md](CLUSTERS.md)
for operations, and [run bookkeeping](scaffolds/records/README.md) for allocating IDs,
validating records, and regenerating indices. Existing active worktrees retain their
frozen layout until their jobs finish; their records migrate when integrated.

Each scaffold defines its environment. Sync each worktree independently with
`uv sync --project scaffolds/<capability> --locked --link-mode hardlink` (add
`--extra cuda` for [Transformer](scaffolds/transformer/README.md)). See also
[state-spectrum analysis](scaffolds/state_rank/README.md) and the
[Transformer qualification record](scaffolds/transformer/validation/README.md).
Pulling Git does not transfer datasets, numerical archives, checkpoints, logs, or
environments; the records identify their home cluster, locations, and hashes.

The main research notes are `overleaf/iclr/iclr2027_conference.tex`. The Overleaf project is cloned directly into the local, Git-ignored `overleaf/` directory; its own `iclr/` subdirectory holds the main notes and template. These are research notes using the ICLR LaTeX format, not an ICLR submission. Supporting notes live in `overleaf/notes/`, figures in `overleaf/media/`, and older drafts in `overleaf/archive/`; `notation.tex` and `notes.cls` are shared project files.

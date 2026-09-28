# Functional memory

Browse `results/` for figures, comparisons, and findings. `experiments/` stores the
runs behind them: one permanent ID per execution, with its recipe, measurements,
provenance, checkpoint, and log together. A study can vary several settings at once;
a result collection can combine runs from several studies.

| Study | Results | Run data and reproduction |
| --- | --- | --- |
| Attention scaling: softmax, Taylor, Hermite, learned polynomials, and GLA | [Heatmaps and degree curves](results/attention_scaling/README.md) | [All attention runs](experiments/attention_scaling/README.md) |
| State rank | [Figures and findings](results/state_rank/README.md) | [Runs and execution](experiments/state_rank/README.md) |

All attention settings vary within `experiments/attention_scaling/runs/<run_id>/`.
There are no separate experiment folders for individual mechanisms or sweeps.
The [run index](experiments/attention_scaling/runs.csv) exposes the settings,
home cluster, training budget, status, and outcome. Each result collection selects
its inputs explicitly, so the same run can participate in several comparisons.

Additional attention views:
[GLA width](results/gla_key_dimension/README.md),
[key width and degree](results/key_dimension_taylor/README.md),
[learned quadratic](results/learned_quadratic_attention/README.md),
[learned unnormalized Taylor](results/learned_unnormalized_taylor/README.md),
[nonnegative learned Taylor](results/nonnegative_learned_taylor/README.md),
[odd Taylor](results/odd_taylor_attention/README.md),
[QK-normalized softmax](results/qk_norm_attention/README.md),
[QK-normalized learned Taylor](results/qk_norm_learned_taylor/README.md),
[unnormalized exponential](results/unnormalized_attention/README.md), and
[unnormalized Taylor](results/unnormalized_taylor_attention/README.md).

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

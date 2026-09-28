# Functional memory

Browse `results/` for figures, comparisons, and findings. `experiments/` stores the
runs behind them: one permanent ID per execution, with its recipe, measurements,
provenance, checkpoint, and log together. A study can vary several settings at once;
a result collection can combine runs from several studies.

| Study or comparison | Results | Run data and reproduction |
| --- | --- | --- |
| State rank | [Figures and findings](results/state_rank/README.md) | [Runs and execution](experiments/state_rank/README.md) |
| Attention scaling across studies | [Combined heatmaps and degree curves](results/attention_scaling/README.md) | Selected from the studies below |
| GLA key dimension | [Figures and findings](results/gla_key_dimension/README.md) | [Runs and execution](experiments/gla_key_dimension/README.md) |
| Hermite and Taylor attention with QK-norm | [Hermite–Taylor figure and findings](results/attention_scaling/README.md) | [Runs and execution](experiments/hermite_attention/README.md) |
| Query/key dimension and Taylor degree | [Figures and findings](results/key_dimension_taylor/README.md) | [Runs and execution](experiments/key_dimension_taylor/README.md) |
| Learned quadratic attention | [Figures and findings](results/learned_quadratic_attention/README.md) | [Runs and execution](experiments/learned_quadratic_attention/README.md) |
| Learned unnormalized Taylor attention | [Figures and findings](results/learned_unnormalized_taylor/README.md) | [Runs and execution](experiments/learned_unnormalized_taylor/README.md) |
| Nonnegative learned Taylor attention | [Figures and findings](results/nonnegative_learned_taylor/README.md) | [Runs and execution](experiments/nonnegative_learned_taylor/README.md) |
| Odd Taylor attention | [Figures and findings](results/odd_taylor_attention/README.md) | [Runs and execution](experiments/odd_taylor_attention/README.md) |
| QK-normalized attention | [Figures and findings](results/qk_norm_attention/README.md) | [Runs and execution](experiments/qk_norm_attention/README.md) |
| QK-normalized learned Taylor attention | [Figures and findings](results/qk_norm_learned_taylor/README.md) | [Runs and execution](experiments/qk_norm_learned_taylor/README.md) |
| Unnormalized exponential attention | [Figures and findings](results/unnormalized_attention/README.md) | [Runs and execution](experiments/unnormalized_attention/README.md) |
| Unnormalized Taylor attention | [Figures and findings](results/unnormalized_taylor_attention/README.md) | [Runs and execution](experiments/unnormalized_taylor_attention/README.md) |

The older raw-score [Hermite sweep](worktrees/hermite_attention/experiments/hermite_attention/README.md)
and [factored degree-two run](worktrees/hermite_factored_m2/experiments/hermite_factored_m2/README.md)
use `qk_norm: false`; they are legacy results, kept in their worktrees until
integration and excluded from the matched QK-norm comparison.

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

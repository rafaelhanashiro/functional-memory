# Unified state-rank comparison

The same analysis is applied to pretrained and randomly initialized Llama and Gated DeltaNet. All curves reuse the retained measurements; no inference or training was rerun.

## How to read the figures

- **Per head, across tokens:** fix a layer; each colored curve preserves a head index and averages over sequences. The random panel additionally averages the five seeds at each head index. Heads remain separate.
- **Per layer, across tokens:** each colored curve is one layer's rank averaged over sequences and heads, and additionally over seeds for random models, at each token position.
- **Across depth:** at the final token, average over sequences and heads within each model. Random means and ranges then summarize the five independently initialized models.
- Each architecture has one collection with pretrained and random panels side by side. No individual-sequence or individual-seed figures are retained in the collection.
- Random per-head curves average the same numerical head index across independently initialized models. This describes an initialization ensemble, not matched learned features across seeds or with the pretrained model.
- In per-head panels, pretrained shading is the sequence min/max; random shading is the min/max of the five sequence-averaged seed curves. Layer panels show means only. The retained depth overview also shows faint seed curves.
- Compute rank separately for each state before averaging. No rank is computed from an averaged state or spectrum. Bands are descriptive ranges, not confidence intervals.

All layer and head indices are zero-based. Token positions count the initial BOS token.

| Model | Layers | Heads | State rank ceiling | Selected layer |
|---|---:|---:|---:|---:|
| Llama | 16 | 8 | 64 | 8 |
| Gated DeltaNet | 24 | 8 | 256 | 12 |

[Terminal ranks versus layer](depth_counts.png) · [PDF](depth_counts.pdf)

Llama measures its implicit softmax numerator (eight KV heads, width 64); Gated DeltaNet measures its native 256 × 256 recurrent state before output processing.

The same four source articles are used in both architectures, with model-native 512-token prefixes. The text spans and token IDs differ across architectures; pretrained and random conditions within an architecture use exactly the same tokens.

## Model collections

Each figure has pretrained and random columns, with four rank measures and shared vertical scales. PNG links below also have sibling PDF files.

| Model collection | Per head | Per layer |
|---|---|---|
| [Llama](llama/README.md) | [Layer 8](llama/heads.png) | [All layers](llama/layers.png) |
| [Gated DeltaNet](gated-deltanet/README.md) | [Layer 12](gated-deltanet/heads.png) | [All layers](gated-deltanet/layers.png) |

## Terminal comparisons

The table compares pretrained ranks with the mean of the five random models, after averaging over sequences and heads at token 512. It describes aggregate layers, not every individual head.

| Model | Metric | Pretrained lower | Equal | Pretrained higher |
|---|---|---:|---:|---:|
| Llama | Participation ratio | 16 | 0 | 0 |
| Llama | Threshold rank | 16 | 0 | 0 |
| Llama | Modes for 90% energy | 16 | 0 | 0 |
| Llama | Modes for 99% energy | 16 | 0 | 0 |
| Gated DeltaNet | Participation ratio | 24 | 0 | 0 |
| Gated DeltaNet | Threshold rank | 15 | 0 | 9 |
| Gated DeltaNet | Modes for 90% energy | 21 | 0 | 3 |
| Gated DeltaNet | Modes for 99% energy | 18 | 0 | 6 |

Llama's pretrained layer averages are lower for all four rank measures at every layer. DeltaNet's participation ratio is lower at every layer, but its threshold and energy-rank comparisons are mixed. Thus the result depends on the rank definition; it does not support a blanket claim that pretraining reduces all ranks.

These are cross-sectional comparisons with fresh reference initializations, not the actual initial checkpoints of the pretrained models or a measured trajectory through training.

Participation ratio is `(sum(sigma^2))^2 / sum(sigma^4)`. Threshold rank counts singular values strictly above `1e-4 * sigma_max`. Energy ranks count modes needed for 90% and 99% of squared singular-value energy.

## Data and provenance

- [Per-head terminal summaries](per_head_terminal.csv): sequence mean/min/max at the selected layer; each seed remains separate.
- [Per-layer terminal summaries](depth_summary.csv): sequence/head means and fractions of state width for each model and seed.
- [Detailed comparisons](comparison_statistics.json), [resolved configuration](config.json), [provenance](provenance.json), and [postprocessing validation](validation.json).
- Full token curves remain in the original immutable `metrics.npz` archives; no duplicate archives or checkpoints were created.
- The original successful Slurm logs are recorded in provenance. This CPU postprocessing required no Slurm job.

Reproduce from the repository root with a fresh output directory:

```bash
uv run --frozen python -m experiments.state_rank.unify_analysis --output /path/to/new-results
```

# Pretrained versus random-weight state rank

Complete conditions: 6/6; completed random seeds: 5.

| Condition | Weights | Seed | Status |
|---|---|---:|---|
| pretrained | pretrained | — | complete |
| random-seed0 | random | 0 | complete |
| random-seed1 | random | 1 | complete |
| random-seed2 | random | 2 | complete |
| random-seed3 | random | 3 | complete |
| random-seed4 | random | 4 | complete |

Figures and `summary.csv` include complete conditions only. All included conditions use identical token IDs, layer indices, head indices, and saved spectrum prefixes.
There are 4 fixed inputs, 512 tokens per input, 8 KV heads, and 16 measured layers. Layers use zero-based indices. Token and spectrum panels show layer index 8.

- `token_metrics.{png,pdf}`: four rank metrics at every prefix.
- `terminal_rank_depth.{png,pdf}`: terminal rank metrics across layers.
- `normalized_spectra.{png,pdf}`: relative singular spectra at saved prefixes.
- `norm_energy_depth.{png,pdf}`: terminal-token key/value norms and terminal cumulative state energy.
- `summary.csv`: one row per complete condition and layer; terminal metrics retain full numerical precision.

Threshold rank counts singular values strictly greater than 0.0001 times the largest singular value.

## Interpretation and aggregation

Participation ratio is (sum(sigma²))² / sum(sigma⁴). The 90% and 99% energy ranks count the minimum number of descending modes whose squared singular values reach the respective energy fraction.

Rank metrics are computed for each input/head state first, then averaged equally over inputs and heads. We never compute a rank from an averaged spectrum. Each faint random curve is one seed's input/head aggregate; the solid random curve is the equal-weight mean over completed seeds, and shading is the minimum–maximum range of those seed aggregates. It is a descriptive range, not a confidence interval. Inputs and heads are not treated as independent experimental replications, and corresponding pretrained/random heads are not assumed to be matched.

For each saved input/head state, relative singular values are computed as sqrt(lambda_i / lambda_1) from the normalized eigenvalues, and only then averaged over inputs and heads. A zero state contributes zeros. Spectral values below 10⁻⁶ are clipped only for logarithmic display; saved measurements and CSV values are unchanged.

Key/value norm panels show sqrt(mean(||vector||₂²)) across inputs and heads at the terminal token. These are RMS vector norms, not per-component RMS and not averages across all token positions. The energy panel averages log(trace(B)) over input/head states, then converts to base 10; it is not log of the arithmetic mean energy. Zero state energy is represented by negative infinity.

This is a descriptive diagnostic on a fixed, small set of inputs. The pretrained model is one checkpoint, while random weights are independent initialization seeds of the same architecture; seed ranges do not measure pretrained uncertainty. Matching the inputs and architecture isolates this comparison but does not identify a causal contribution of an individual head or establish generalization to other texts, checkpoints, model sizes, or training recipes. No language-model quality claim follows from state rank alone.

Model: `meta-llama/Llama-3.2-1B`; revision: `4e20de362430cd3b72f300e6b0f18e50e7166e08`.

Resolved configuration and planned conditions are in `manifest.json`; condition-specific seeds, source commits, and execution metadata are in each condition's `metadata.json`. Incomplete conditions remain visible in the status table and do not enter aggregates.

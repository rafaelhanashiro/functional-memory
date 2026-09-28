# Attention mechanisms across key width and Taylor degree

This record puts the completed 200M-reference attention experiments on one
terminal held-out perplexity scale, in two figures. The **heatmap** covers the
mechanisms trained across per-head query/key widths `d_K = 8, 16, 32, 64, 128`:
normalized fixed Taylor kernels by degree, softmax, and GLA. The **degree
chart** covers the polynomial families trained only at `d_K = 64`, as perplexity
against degree `m = 1, 2, 3, 4`. The comparison is a derived analysis of existing
runs; it trains no new models.

Both figures describe the target state of the repository's experiments: every
mechanism trained under the scaffold's current defaults (selective weight decay
and QK-norm scores), plus one softmax run without QK-norm as the plain-score
reference. Runs that predate those defaults are marked as **legacy**, outlined in
the heatmap and drawn hollow with dashed connectors in the chart; each is
replaced, and its mark disappears, when its experiment is rerun. Legacy status is
decided from each run's resolved configuration, not by hand.

Each plotted value is a model evaluated after **19,074 optimizer updates** and
**10,000,269,312 prediction tokens**. All plotted runs use seed 0 and the same
packed FineWeb-Edu order and held-out evaluation set. Terminal validation loss
and perplexity come from each source run's committed `metrics.jsonl` and
`completion.json`. The two figure configurations,
[`selection.yaml`](selection.yaml) and
[`degree_chart.yaml`](degree_chart.yaml), pin their source
manifests by SHA256 and name every plotted run. The
[comparison provenance](manifest.json) records the source commits,
run-file hashes, and generated artifact hashes.

**Home cluster:** Kennel. **Artifact root:**
`/data/cl/u/rafah/home/projects/functional-memory`. This analysis has no new
checkpoint or Slurm log. The source experiments retain their own training
provenance and artifacts.

## Reading the figures

**Heatmap.** The panels share one linear perplexity color scale; lower is better.

| Panel | Columns | Meaning |
| --- | --- | --- |
| Normalized Taylor, fixed | `m = 1, 2, 3, 4, 8, 16` | Fixed coefficients `1/r!`; causal row normalization |
| References | Softmax, softmax without QK-norm, GLA | Softmax with QK-norm (the default), the plain-score softmax reference, and GLA, which has no Taylor degree and no QK-norm |

**FAIL** marks a recorded scientific numerical failure at the annotated update; it
has no terminal perplexity. **NOT RUN** means no run was trained at that
coordinate. The figure never fills a gap with a partial validation or an estimated
perplexity; other execution states are labeled **INCOMPLETE** if encountered.

**Degree chart.** One line per family on a shared linear perplexity axis, with
horizontal reference levels for QK-norm softmax, softmax without QK-norm, and the
unnormalized exponential. Numerical failures have no perplexity, so they appear
in the strip below the axis at their degree, labeled with the failing update.
Every learned family learns its coefficients per layer and head.

| Family | Degrees | Meaning |
| --- | --- | --- |
| Normalized, learned | 1–4 | All `m + 1` coefficients learned from `1/r!`; signed kernel and row sum |
| Normalized, learned nonnegative | 2, 4 | The same normalized kernel written as a sum of squares `q1^2 + q2^2`, nonnegative by construction; odd degrees cannot be nonnegative on the real line |
| Unnormalized, learned | 1–4 | All coefficients learned from `1/r!`; no row division |
| Unnormalized, fixed Taylor | 1–4 | Fixed coefficients `1/r!`; no row division |

A legacy run was trained with weight decay on all parameters and, for mechanisms
that take it, without QK-norm. Its perplexity is comparable with current runs only
up to the size of those two effects (about 0.03 PPL for the decay policy and 0.21
for QK-norm at the `d_K = 64` softmax).

<!-- BEGIN GENERATED RESULTS -->
45 displayed coordinates: 31 complete, 2 numerical failures, 12 not run, and 0 incomplete. Only completed runs at the common 19,074-update endpoint receive a color and a terminal PPL. A failed run is labeled with its failure update; an untrained coordinate is marked NOT RUN. 25 of the 33 trained cells are outlined as legacy runs: they trained with weight decay on all parameters and, where it applies, without QK-norm, before those became the scaffold's defaults; each is replaced when its experiment is rerun. 8 cells already use the current defaults.

![Combined terminal perplexity heatmap](combined_terminal_ppl.png)

[PDF](combined_terminal_ppl.pdf)

### Polynomial kernels at d_K = 64

17 trained runs: 13 complete and 4 numerical failures; 5 are legacy runs, drawn hollow with dashed connectors. Numerical failures appear in the strip below the axis at their degree, labeled with the failing update. The nonnegative family exists only for even degrees, which are the only degrees at which a polynomial can be nonnegative on the real line.

![Terminal perplexity against polynomial degree](degree_chart.png)

[PDF](degree_chart.pdf) · [Cell-level data and source checksums](summary.csv) · [Provenance](manifest.json)
<!-- END GENERATED RESULTS -->

## Snapshot under the original weight-decay policy

Every run in the original heatmap trained under the scaffold's original optimizer policy,
in which AdamW's decay applied to all parameters, including the token embedding
and RMSNorm scales. Commit `cb31558` made the policy that excludes embeddings,
normalization scales, and biases canonical for new runs. As the source records are
retrained under it, the figures and `summary.csv` above change with them.
[`combined_terminal_ppl-original-decay.png`](combined_terminal_ppl-original-decay.png)
([PDF](combined_terminal_ppl-original-decay.pdf)) and
[`summary-original-decay.csv`](summary-original-decay.csv) are frozen copies of the
2026-09-25 state under the original policy; regeneration does not rewrite them.
The two old-policy `d_K = 64` softmax runs retired on 2026-09-26 ended at PPL 14.6898
(`200m-dk064-seed0`, the cell shown in the frozen copy) and 14.6942
(`200m-softmax-seed0`, the control the other records quoted); the current figure's
`d_K = 64` softmax is the selective-policy baseline.

## Findings

Under the current defaults, eighteen runs exist so far, all at `d_K = 64`:

- **Softmax references:** QK-norm 14.4574, plain 14.6655.
- **Normalized fixed Taylor kernels, m = 1, 2, 3, 4, 8, 16:** the odd degrees failed
  numerically at updates 31 and 419; odd polynomials are negative for low scores,
  so their row sums can cross zero. The even degrees reached 16.511, 14.896,
  14.498, and 14.454. Degree 16 ties QK-norm softmax (0.0035 lower, within
  single-seed noise), and degree 8 is 0.04 above it.
- **Unnormalized learned Taylor kernels, m = 1–4:** 20.279, 17.599, 16.330, 16.384,
  from the [QK-norm learned Taylor sweep](../qk_norm_learned_taylor/README.md). Each
  is better than the legacy fixed unnormalized kernel of the same degree, and
  degrees 3 and 4 are within 0.05 of each other.
- **Normalized learned kernels, m = 1–4:** all failed numerically (updates 27, 1,857,
  318, and 1,456) once a kernel turned negative inside the bounded score range and a
  row sum crossed zero.
- **Nonnegative learned kernels, m = 2 and 4:** the
  [same normalized kernels written as a sum of squares](../nonnegative_learned_taylor/README.md)
  trained without failure, reaching 16.176 and 14.742, 0.33 and 0.15 below the
  fixed Taylor kernels of the same degree. The `m = 4` result is 0.28 above QK-norm
  softmax and the lowest in the degree chart; in the heatmap, fixed `m = 8` and
  `m = 16` at the same width are lower. Both nonnegative kernels learned a touching
  root inside the score range: the zero the signed kernels reached only by crossing
  it.

Every other plotted value is a legacy run, so the comparisons below among them share
one training setup.

At the other widths, the legacy fixed normalized grid shows that increasing degree
narrows the softmax gap. At `d_K=8`, fixed `m=2` has PPL 20.5775, versus 16.8479 at
`m=16` and 16.7053 for softmax. At `d_K=128`, fixed `m=8` reaches 14.3586, close to
softmax's 14.3776. The difference of 0.0190 is a single-seed observation, not
evidence of a reliable improvement.

GLA has no degree axis. Its PPL decreases from 17.2239 at `d_K=8` to 15.4366 at
`d_K=128`; it beats fixed normalized `m=2` at each width but trails softmax at each
width. GLA adds gates and head normalization, and parameter counts change with
`d_K`, so the heatmap compares the observed models rather than isolating a single
architectural effect.

The unnormalized exponential reference at `d_K=64` finishes at 17.4583; it does not
divide by an attention row sum and is a different baseline from softmax.

The source records also include a fixed normalized `m=0` run at `d_K=64`
(PPL 47.0653), softmax runs at `d_K=2, 4, 256`, and the learned quadratic variants:

- the per-head positive quadratic `0.5 (1 + (a s + b)^2)`, 16.2125, a legacy run
  that the sum-of-squares `m = 2` run replaced in the chart;
- the globally shared positive quadratic, 16.1890;
- the globally shared unrestricted quadratic, which failed at update 2,324.

They remain in their [source](../key_dimension_taylor/README.md)
[records](../learned_quadratic_attention/README.md) but lie outside these figures.
The heatmap has a declared width and degree range, and the chart shows only per-head
coefficients under a single parameterization per family.

## Limits and reproduction

This is one seed per trained coordinate. Width changes model parameter count;
GLA also changes the attention architecture. The source runs used different GPU
types and execution revisions, as their manifests record. Small PPL differences
should not be read as statistically established rankings. Sparse panels show
which runs were actually performed, not evidence that untested coordinates
would fail or match the plotted values.

From a checkout with the committed source records, sync its own locked
environment and regenerate the summary, figures, and provenance:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m \
  results.attention_scaling.plot
```

The renderer checks that each cell's run still has the scientific inputs pinned
in the two selection files, then writes both figures and the [cell-level summary](summary.csv), whose `figure`
column says which figure each row belongs to. No GPU, dataset, or source
checkpoint is required.

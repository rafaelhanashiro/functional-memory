# Attention mechanisms across key width and degree

Terminal held-out perplexity of the 200M attention runs: 19,074 updates (10.0B
tokens), seed 0, same data order and evaluation set. No training happens here; the
figures are built from existing runs named in [`selection.yaml`](selection.yaml)
(heatmap) and [`degree_chart.yaml`](degree_chart.yaml). Provenance:
[`manifest.json`](manifest.json). Home cluster: Kennel.

The two maintained attention figures are the combined heatmap and the degree plot.
Their selections are subsets of the [complete run inventory](../../experiments/attention_scaling/runs.csv).
Each run retains its recipe, measurements (including training curves and recorded
coefficient diagnostics), and execution metadata under
[`experiments/attention_scaling/runs/`](../../experiments/attention_scaling/runs/).

Every plotted run uses selective weight decay and QK-norm, except "softmax no QK-norm"
and Gated DeltaNet, which have no QK-norm, and its reference initialization:
GPT-2/Megatron (`transformer_reference`) for softmax, unnormalized exp, and every
polynomial kernel; the paper's (`gated_deltanet_reference`) for Gated DeltaNet.

<!-- BEGIN GENERATED RESULTS -->
### Heatmap

45 cells: 31 complete, 2 failed, 12 not run, 0 incomplete.

![Combined terminal perplexity heatmap](combined_terminal_ppl.png)

[PDF](combined_terminal_ppl.pdf)

### Degree chart (d_K = 64)

30 runs: 20 complete, 10 failed. Failures sit below the axis at their update.

![Terminal perplexity against polynomial degree](degree_chart.png)

[PDF](degree_chart.pdf) · [Cell data and checksums](summary.csv) · [Provenance](manifest.json)
<!-- END GENERATED RESULTS -->

## Figure key

- **Heatmap:** normalized fixed Taylor (`1/r!`) at `m = 1, 2, 3, 4, 8, 16` and
  `d_K = 8–128`, plus softmax with and without QK-norm and Gated DeltaNet. FAIL = numerical
  failure at the shown update; NOT RUN = not trained.
- **Degree chart:** families at `d_K = 64`; failures sit in the strip below the axis.
  Learned coefficients start from `1/r!` and are learned per layer and head.
  "Nonnegative" writes the kernel as `q1^2 + q2^2`, so it exists only at even degrees.
  Fixed Hermite uses the standard-normal projection
  `p_m(s) = exp(1/2) sum_(r<=m) He_r(s)/r!` (normalized `m = 2` evaluated as
  `exp(1/2) (s+1)^2 / 2`).

## Results at d_K = 64

References: softmax **13.514**, Gated DeltaNet **13.705**, softmax no QK-norm
**13.932**, unnormalized exp with QK-norm **15.885**.

Normalized kernels (failures show the update):

| m | Fixed Taylor | Fixed Hermite | Learned | Learned nonnegative |
| ---: | ---: | ---: | ---: | ---: |
| 1 | fail @35 | fail @374 | fail @60 | — |
| 2 | 14.911 | fail @176 | fail @1,923 | 14.666 |
| 3 | fail @555 | fail @549 | fail @336 | — |
| 4 | 13.774 | fail @680 | fail @1,143 | **13.689** |
| 8 | 13.543 | | | |
| 16 | **13.519** | | | |

Unnormalized kernels:

| m | Fixed Taylor | Fixed Hermite | Learned Taylor |
| ---: | ---: | ---: | ---: |
| 1 | 17.903 | 18.196 | 17.883 |
| 2 | 18.166 | 15.673 | 15.507 |
| 3 | 15.621 | 15.747 | **14.657** |
| 4 | 15.888 | 15.278 | 14.720 |

- **Normalized kernels need a nonnegative kernel.** Signed kernels fail when a row sum
  crosses zero: every odd degree, the signed learned kernels, and Hermite `m = 4`
  (negative for `s` in about (−2.8, −1.6)). Even-degree Taylor is strictly positive and
  finishes. Hermite `m = 2` is nonnegative but zero at `s = -1` and fails (at update 176);
  the learned sum-of-squares kernels also learned a zero and finished.
- **Normalized Taylor `m = 16` ties softmax** (13.519 vs 13.514); `m = 8` is 0.03 above.
  Nonnegative learned `m = 4` (13.689) is 0.18 above softmax and 0.08 below fixed Taylor
  `m = 4`; at `m = 2` it is 0.24 below fixed Taylor.
- **Hermite vs Taylor, unnormalized:** Hermite is better at even degrees (−2.49 at
  `m = 2`, −0.61 at `m = 4`) and worse at odd degrees (+0.29, +0.13).
- **Learned beats fixed Taylor, unnormalized,** at every degree (by 0.02 at `m = 1` to
  2.66 at `m = 2`). Learned `m = 3` and `4`, Hermite `m = 4`, and Taylor `m = 3` beat
  unnormalized exp (15.885).
- **QK-norm improves softmax by 0.42** (13.514 vs 13.932), the only comparison of a
  kernel with and without QK-norm.

## Other widths

- **Normalized fixed Taylor:** higher degree closes the gap to softmax. `m = 16` is
  within 0.03 of softmax at `d_K >= 16` and 0.11 above at `d_K = 8` (15.75 vs 15.64);
  `m = 8` is 0.44 above at `d_K = 8` and within 0.03 from `d_K = 64` (13.29 vs 13.30 at
  128). `m = 4` trails softmax by 1.42 at `d_K = 8`, narrowing to 0.15 at 128; `m = 2`
  by 3.85, narrowing to 1.03.
- **Softmax vs Gated DeltaNet:** Gated DeltaNet is
  lower at `d_K = 8` (14.85 vs 15.64) and level at 16 (14.29 vs 14.34); softmax is lower
  at 32, 64, and 128 (by 0.19, 0.19, 0.08). Both improve with every doubling of `d_K`.

## Limits

- One seed per cell; differences of a few tenths are not established.
- Parameter count changes with `d_K`. Gated DeltaNet also changes the architecture and
  has 10.9–11.0M (5–6%) more parameters than softmax at the same `d_K`.
- Some runs resumed on another GPU type after preemption: Gated DeltaNet `d_K = 128`
  (H100, then B300 from update 13,500); normalized Taylor `d_K = 16`, `m = 16` and
  `d_K = 64, 128`, `m = 8`; unnormalized fixed Taylor `m = 2`, `4` and fixed Hermite
  `m = 1`, `4` (attempts on both H100 and B300; see their records).
- Unnormalized fixed Taylor and Hermite `m = 1` train with recurring loss spikes (up to
  +3.8 and +6.1 nats at updates 2,900–8,200) and recover. No other plotted run with a
  Kennel log spikes.
- Runs on B300 use cuBLAS 12.9, which JAX warns can silently corrupt data when a cuBLAS
  kernel runs concurrently with another kernel. No plotted run shows a B300-specific loss
  spike or numerical failure: the Taylor `m = 1` spikes occur on H100 too, and the
  normalized failures repeat on H100.

## Reproduce

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.attention_scaling.plot
```

Checks every pinned input, then writes the figures, [`summary.csv`](summary.csv) (its
`figure` column names the figure), and `manifest.json`. No GPU or dataset needed.

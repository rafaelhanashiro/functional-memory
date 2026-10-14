# Attention mechanisms across key width and degree

Terminal held-out perplexity of the 200M attention runs: 19,074 updates (10.0B
tokens) with 400 warmup updates, seed 0, same data order and evaluation set. No training happens here; the
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

References: Gated DeltaNet **13.594**, softmax **13.731**, softmax no QK-norm
**14.184**, unnormalized exp with QK-norm **15.663**.

Normalized kernels (failures show the update):

| m | Fixed Taylor | Fixed Hermite | Learned | Learned nonnegative |
| ---: | ---: | ---: | ---: | ---: |
| 1 | fail @67 | fail @192 | fail @73 | — |
| 2 | 15.014 | fail @2,875 | fail @937 | 14.771 |
| 3 | fail @218 | fail @254 | fail @203 | — |
| 4 | 13.927 | fail @1,122 | fail @1,395 | **13.815** |
| 8 | 13.741 | | | |
| 16 | **13.740** | | | |

Unnormalized kernels:

| m | Fixed Taylor | Fixed Hermite | Learned Taylor |
| ---: | ---: | ---: | ---: |
| 1 | 17.930 | 18.233 | 17.874 |
| 2 | 18.099 | 15.624 | 15.461 |
| 3 | 15.604 | 15.794 | **14.630** |
| 4 | 15.862 | 15.238 | 14.742 |

- **Normalized kernels need a nonnegative kernel.** Signed kernels fail when a row sum
  crosses zero: every odd degree, the signed learned kernels, and Hermite `m = 4`
  (negative for `s` in about (−2.8, −1.6)). Even-degree Taylor is strictly positive and
  finishes. Hermite `m = 2` is nonnegative but zero at `s = -1` and fails (at update
  2,875); the learned sum-of-squares kernels also learned a zero and finished.
- **Gated DeltaNet leads every `d_K = 64` cell,** softmax by 0.14.
- **Normalized Taylor `m = 8` and `16` tie softmax** (13.741 and 13.740 vs 13.731).
  Nonnegative learned `m = 4` (13.815) is 0.08 above softmax and 0.11 below fixed Taylor
  `m = 4`; at `m = 2` it is 0.24 below fixed Taylor.
- **Hermite vs Taylor, unnormalized:** Hermite is better at even degrees (−2.48 at
  `m = 2`, −0.62 at `m = 4`) and worse at odd degrees (+0.30, +0.19).
- **Learned beats fixed Taylor, unnormalized,** at every degree (by 0.06 at `m = 1` to
  2.64 at `m = 2`). `m = 3` beats `m = 4` for learned and fixed Taylor; Hermite `m = 4`
  beats `m = 3`. Learned `m = 2`–`4`, Hermite `m = 2` and `4`, and Taylor `m = 3` beat
  unnormalized exp (15.663).
- **QK-norm improves softmax by 0.45** (13.731 vs 14.184), the only comparison of a
  kernel with and without QK-norm.

## Other widths

- **Normalized fixed Taylor:** higher degree closes the gap to softmax. `m = 16` is
  within 0.01 of softmax from `d_K = 32` (14.028 vs 14.028 at 32), 0.06 above at 16, and
  0.10 above at `d_K = 8` (15.84 vs 15.73); `m = 8` is 0.44 above at `d_K = 8` and within
  0.03 from `d_K = 64` (13.50 vs 13.52 at 128). `m = 4` trails softmax by 1.42 at
  `d_K = 8`, narrowing to 0.09 at 128; `m = 2` by 3.89, narrowing to 0.91.
- **Gated DeltaNet is lower than softmax at every `d_K`:** by 0.92 at `d_K = 8`, 0.25 at
  16, and 0.12, 0.14, and 0.19 at 32, 64, and 128. Both improve with every doubling of
  `d_K`.

## Limits

- One seed per cell; differences of a few tenths are not established. Rerunning the
  same recipe and seed on other hardware moved terminal perplexity by about 0.01
  (normalized fixed Taylor `m = 2` at `d_K = 16`, 32, and 128 on Kennel and ORCD).
- Parameter count changes with `d_K`. Gated DeltaNet also changes the architecture and
  has 10.9–11.0M (5–6%) more parameters than softmax at the same `d_K`.
- Several runs resumed on another GPU type (B300, H100, H200) after preemption, and some
  trained on two GPUs with the same global batch; each run's record lists its attempts.
- Unnormalized `m = 1` (fixed Taylor, fixed Hermite, and learned) trains with loss
  spikes of up to +7.3 nats between updates 3,100 and 6,100 and recovers within 100
  updates. No other plotted run spikes (training-loss jumps of more than 0.5 nats over
  the running median after update 1,000).
- Runs on B300 use cuBLAS 12.9, which JAX warns can silently corrupt data when a cuBLAS
  kernel runs concurrently with another kernel. Softmax at `d_K = 64` stopped on B300 at
  update 18,408 with an infinite gradient norm at a finite loss, then resumed on H100 from
  its update-18,000 checkpoint, reproduced the loss trajectory, and completed. The
  normalized failures also occur on H100.

## Reproduce

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.attention_scaling.plot
```

Checks every pinned input, then writes the figures, [`summary.csv`](summary.csv) (its
`figure` column names the figure), and `manifest.json`. No GPU or dataset needed.

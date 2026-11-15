# Coefficient-input normalization

Opened **2026-11-14**. This Transformer++ exploration asks whether limiting the
input to the tanh coefficient generator reduces saturation and improves perplexity.
It contains four completed runs, at degrees eight and twelve, at the 15-layer
width-896 shape, each matched to its own existing control. These runs
remain exploratory and enter no canonical comparison.

| Coefficient form | Unit-L2 input run | Original-input control |
| --- | --- | --- |
| Query, degree 8 | `r25a5f78841b2` | `r25fffc16c2f3` |
| SoS query, degree 8 | `r717267afcb78` | `reeefbcc19abf` |
| Query, degree 12 | `r7e34dec9dbd5` | `r0fceea0627c8` |
| SoS query, degree 12 | `r6b17c3463a60` | `r3d52b7ab43d1` |

For the current gain-applied pre-attention RMSNorm activation `x`, only the
coefficient generator uses `x_hat = x / max(||x||_2, 1e-6)` in FP32. Direct query
uses `(b_r + a_r*tanh(W_r^T x_hat))/r!`; SoS uses
`F_j = f_j*(1 + b_j + a_j*tanh(W_j^T x_hat))/2`, then expands `U^2 + V^2`.
The Q/K/V inputs and normalization, RoPE, scores, output scaling, and residual
path retain the control recipe. W is unnormalized and trainable with weight
decay 0.1; a and b have no weight decay. W initialization, all learning rates,
seed 42, global batch 256, 2,048 prediction tokens per sequence, 7,570 updates,
and the controls' reference initialization are retained. All use native Mosaic.

The [explicit selection](selection.yaml) pins all four matched run/control pairs.
The [study README](../../../README.md#coefficient-input-l2-experiment) gives
execution commands and acceptance requirements. Configurations and compact
measurements live only in the selected runs.

## Completed degree-eight comparison

Both runs completed all 7,570 updates and retained the validation-best checkpoint
at update 7,500. Lower perplexity is better.

| Coefficient form | Best PPL, original / L2 | Terminal PPL, original / L2 | Best-PPL change, L2 − original |
| --- | ---: | ---: | ---: |
| Query | 14.3228 / **14.2604** | 14.3278 / **14.2636** | −0.0624 |
| SoS query | **14.1955** / 14.2707 | **14.1996** / 14.2748 | +0.0752 |

At the terminal diagnostic, saturation averaged over every layer, head, token,
and generator coordinate is **0% for query and 0.0250% for SoS**, using
`abs(tanh(z)) >= 0.99`. The corresponding mean tanh derivatives are 0.9490 and
0.8836. These are all-model averages on one fixed sequence; the original-run
probes measured only layer 8, head 1, so their saturation percentages are not a
matched all-model comparison. The most saturated SoS coordinate reaches 15.38%
on this sequence despite its small aggregate average.

All-update gradient telemetry shows one clipped update after update 700 in each
run, at update 6,376: norms 1.3333 (query) and 1.4469 (SoS), against threshold 1.
Neither run encountered a numerical failure. Low coefficient-input saturation
therefore accompanies a small query improvement but a small SoS regression in
these single-seed runs; removing saturation is not sufficient to improve both.

## Completed degree-twelve comparison

Both degree-twelve runs completed all 7,570 updates and retained their best
checkpoint at update 7,500. Each has 200,905,126 parameters and matches its own
control except for coefficient-input normalization.

| Coefficient form | Best PPL, original / L2 | Terminal PPL, original / L2 | Best-PPL change, L2 − original |
| --- | ---: | ---: | ---: |
| Query | **14.2256** / 14.2592 | **14.2296** / 14.2636 | +0.0336 |
| SoS query | **14.1649** / 14.2815 | **14.1684** / 14.2857 | +0.1166 |

At update 7,500, the new diagnostics use the same first held-out 2,048-token
sequence as the original probes: record SHA256
`26d5803c7dc95fe7f5dd7015ae4195205246bda089792b2e0f478a29398df8a6`.
The matched layer-8/head-1 comparison therefore controls the checkpoint step,
head, sequence, and saturation threshold `abs(tanh(z)) >= 0.99`.

| Coefficient form | Selected-head saturation, original / L2 | Selected-head mean tanh derivative, L2 |
| --- | ---: | ---: |
| Query | 53.5382% / **0%** | 0.9934 |
| SoS query | 61.9028% / **0%** | 0.9679 |

The original probe values and explicit parents are recorded in the
[SoS saturation diagnostic](../sum_of_squares/README.md#matched-head-tanh-saturation).
Across all layers/heads/coordinates, terminal sampled saturation is **0% for
query and 0.0113% for SoS**. Mean tanh derivatives are 0.9514 and 0.9202. These
all-model averages have no matched all-model original-input probe. The most
saturated SoS coordinate reaches 8.45% of sampled tokens. Mean learned W-vector
norms reach 2.05 for query and 2.52 for SoS; W remains trainable and unnormalized.

All-update gradient telemetry after update 500 covers 7,070 updates per run.

| Coefficient form | Clipped updates, original / L2 | Maximum pre-clip norm, original / L2 |
| --- | ---: | ---: |
| Query | 2 / 5 | 11.6393 / 17.9544 |
| SoS query | 1 / 3 | 1.5031 / 1.5540 |

The query L2 maximum occurred at update 1,249; its other recorded spikes were
2.4136 at 1,410, 2.8980 at 5,858, and 1.7640 at 6,376. The remaining clipped
update lies in 901–1,000, with maximum 1.4232. SoS L2 had two clipped updates
within 501–600 (maximum 1.1059) and one at 6,376 (1.5540). Both runs completed
without numerical failure. The cutoff is 500 because the reporting window
401–500 crosses the end of warmup at 455; exact post-warmup counts cannot be
reconstructed from that window's aggregate.

Thus this L2 intervention removes saturation in the matched head and keeps it
very low in the full sampled model, but does not improve m=12 perplexity or
clipping frequency. This does not isolate saturation from changes to coefficient
variation and optimization caused by changing the input scale.

## Measurements and limits

At scheduled evaluations, the existing fixed diagnostic sequence records all
layers/heads/coordinates: raw and normalized input norms, projection mean/std/min/max,
`abs(tanh(z)) >= 0.99` fractions, and mean `1-tanh(z)^2`, alongside a/b/W norms,
effective coefficient moments, and ordinary gradient-norm telemetry. The native
model forward supplies these generator activations. Attention-weight diagnostics
use the existing blocked reference alongside the native forward. SoS coordinate
labels refer to U/V factor coefficients, not expanded monomial orders.

A drop in saturation alone is not success: coefficient variation and model
quality matter.
The original controls have only the retained-checkpoint saturation probes, so
full saturation trajectories are available for the new runs, not retrospectively
for every control checkpoint. One seed per cell cannot establish small effects.

## Open decision

The degree-eight result is mixed, and both degree-twelve models regress slightly
in this single-seed comparison. Keep coefficient-input L2 as an exploration; the
results do not establish a general improvement or justify replacing the original
input normalization by default. The open decision is whether a less aggressive
input scale or another targeted optimization experiment warrants further study.
No additional runs are part of this selection.

Refresh the selected scientific inputs after new measurements:

```bash
python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/coefficient_input_norm
```

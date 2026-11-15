# Coefficient-input normalization

Opened **2026-11-14**. This Transformer++ exploration asks whether limiting the
input to the tanh coefficient generator reduces saturation and improves perplexity.
It contains two completed degree-eight runs and two degree-twelve runs at the
15-layer width-896 shape, each matched to its own existing control. These runs
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

## Degree-twelve extension

The two degree-twelve runs each have 200,905,126 parameters and retain their
respective controls' initialization, optimizer, data, and reporting recipe.
Only `attention_coefficient_input_normalization: l2` changes scientifically.
Normalization remains after the learned pre-attention RMSNorm channel gains;
W stays unnormalized and trainable. Degree twelve has 13 query coefficient
coordinates, or seven U and six V factor coordinates for SoS. No degree-twelve
outcome is available at launch.

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

The degree-eight result is mixed. The requested degree-twelve extension tests
whether the same coefficient-only normalization helps the more saturated
higher-degree generators. The earlier selected-head comparison found less
saturation in SoS at degree eight but more at degree twelve despite improved
perplexity, so saturation has not been established as the general cause of the
performance gap.

Refresh the selected scientific inputs after new measurements:

```bash
python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/coefficient_input_norm
```

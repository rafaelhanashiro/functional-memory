# Coefficient-input normalization

Opened **2026-11-14**. This Transformer++ exploration asks whether limiting the
input to the tanh coefficient generator reduces saturation and improves perplexity.
It contains two fresh runs at the 15-layer width-896 shape, matched to their own
existing controls; neither enters a canonical comparison.

| Coefficient form | Unit-L2 input run | Original-input control |
| --- | --- | --- |
| Query, degree 8 | `r25a5f78841b2` | `r25fffc16c2f3` |
| SoS query, degree 8 | `r717267afcb78` | `reeefbcc19abf` |

For the current gain-applied pre-attention RMSNorm activation `x`, only the
coefficient generator uses `x_hat = x / max(||x||_2, 1e-6)` in FP32. Direct query
uses `(b_r + a_r*tanh(W_r^T x_hat))/r!`; SoS uses
`F_j = f_j*(1 + b_j + a_j*tanh(W_j^T x_hat))/2`, then expands `U^2 + V^2`.
The Q/K/V inputs and normalization, RoPE, scores, output scaling, and residual
path retain the control recipe. W is unnormalized and trainable with weight
decay 0.1; a and b have no weight decay. W initialization, all learning rates,
seed 42, global batch 256, 2,048 prediction tokens per sequence, 7,570 updates,
and the controls' reference initialization are retained. Both use native Mosaic.

The [explicit selection](selection.yaml) pins the two controls and two new runs.
The [study README](../../../README.md#coefficient-input-l2-experiment) gives
execution commands and acceptance requirements. Configurations and compact
measurements live only in the selected runs.

## Measurements and limits

At scheduled evaluations, the existing fixed diagnostic sequence records all
layers/heads/coordinates: raw and normalized input norms, projection mean/std/min/max,
`abs(tanh(z)) >= 0.99` fractions, and mean `1-tanh(z)^2`, alongside a/b/W norms,
effective coefficient moments, and ordinary gradient-norm telemetry. The native
model forward supplies these generator activations. Attention-weight diagnostics
use the existing blocked reference alongside the native forward. SoS coordinate
labels refer to U/V factor coefficients, not expanded monomial orders.

No outcome is available at launch. Compare terminal and validation-best PPL
separately, then inspect whether saturation stays low as W learns. A drop in
saturation alone is not success: coefficient variation and model quality matter.
The original controls have only the retained-checkpoint saturation probes, so
full saturation trajectories are available for the new runs, not retrospectively
for every control checkpoint. One seed per cell cannot establish small effects.

## Open decision

After both runs complete, decide whether coefficient-input normalization merits
further study. The earlier selected-head comparison found less saturation in SoS
at degree eight but more at degree twelve despite improved perplexity, so saturation
has not been established as the general cause of the performance gap.

Refresh the selected scientific inputs after new measurements:

```bash
python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/coefficient_input_norm
```

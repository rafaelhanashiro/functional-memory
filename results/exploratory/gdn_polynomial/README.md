# Polynomial GDN: independent erasure and memory degree

Opened 2026-10-09. Home cluster: **Kennel**.

This topic studies separate erase and write gates, polynomial memory extensions,
and write-coefficient activations in Gated DeltaNet under the canonical
400-step-warmup recipe. The selected runs below cover the independent-erase
comparison: does giving GDN separate sigmoid erase and write gates improve
language-model performance?

## Runs and method

| Role | Run ID | Erase gate | Status |
| --- | --- | --- | --- |
| Primary | `rf7341031e3d5` | Independent learned projection | Complete, 19,074 updates |
| Control | `rf640fc13f3a8` | Shared with the write gate | Complete, 19,074 updates |

The control is explicitly selected for its canonical GDN initialization, degree-one
key space, width, data, training recipe, and current telemetry on Kennel. Its terminal
validation perplexity is 13.582856. Selection does not depend on its performance.
[selection.yaml](selection.yaml) names both roles and pins their scientific inputs.

Both models use the existing GDN block: linear projections, width-four causal
depthwise convolutions and SiLU for Q/K/V, per-head L2 normalization of Q/K, Mamba-2
scalar decay, gated output RMSNorm, and no RoPE. The new update, in keys-first layout,
is `S_t = alpha_t S_{t-1} + k_t (beta_t v_t - lambda_t alpha_t S_{t-1}^T k_t)^T`.
Both `beta_t = sigmoid(W_beta x_t)` and `lambda_t = sigmoid(W_lambda x_t)` are scalar
per token, layer, and head. The bias-free erase projection starts as a copy of the
write projection in a separate device buffer and is optimized independently. The
copy preserves canonical values and the original RNG draws. All original model
parameters have
the same initialization as tied GDN. The key feature space is unchanged; this run
does not introduce polynomial features.

The recipe has 10 layers, width 1024, 16 heads, `d_K = d_V = 64`, MLP width 3008,
sequence length 2,048, BF16 training, effective batch 256, global microbatch 64,
seed 0, and 19,074 updates (10.0B prediction tokens) on the same packed FineWeb data.
AdamW uses learning rates 3e-4 to 3e-5, 400 warmup updates, betas 0.9/0.95,
weight decay 0.1 with the canonical exclusions, and gradient clipping at 1.
The extra projection adds 163,840 parameters, bringing the total to 211,007,424.
GDN's internal QK normalization and gated output RMSNorm are active even though the
dot-product-only recipe fields `qk_norm` and `attention_output_scaling` are disabled.

The new run records standard every-update activation, gradient, and actual-update
telemetry, plus erase/write gate values and their difference. The original control
has the earlier standard telemetry; newly separated gate groups cannot be compared
directly against a combined historical group.

## Completed findings

The primary completed all 19,074 updates (10,000,269,312 prediction tokens) on
2026-10-12 at 18:32 EDT. Its final checkpoint is also its validation-best checkpoint.
Both runs select update 19,074 by lowest validation NLL.

| Run | Terminal validation NLL | Terminal validation perplexity | Validation-best step |
| --- | ---: | ---: | ---: |
| Independent erase, `rf7341031e3d5` | 2.608554593 | 13.579408889 | 19,074 |
| Tied erase/write, `rf640fc13f3a8` | 2.608808373 | 13.582855503 | 19,074 |

The observed difference is -0.000253780 NLL or -0.003446614 perplexity
(-0.02537% relative perplexity): a numerical near-tie. It does not establish an
improvement from separating the gates or statistical equivalence. This is one seed,
and the executions differ in arithmetic and hardware: the primary uses FP32 chunk
intermediates and TF32x3 products on four H100 SXM GPUs; the control uses the earlier
reduced-precision kernel on two B300 GPUs. A comparison with matched arithmetic and
runtime would be needed to isolate the effect of gate separation.

Validation perplexity decreased at every one of the primary's 20 evaluations,
from 34.225204 at update 1,000 to 13.579409 at update 19,074. Every logged numeric
measurement is finite, all attention diagnostic passes are valid, and all recorded
invalid counts are zero. Gradient and activation summaries cover all 19,074 updates.
The gradient-spike detector (five times the recent median) flagged three isolated
updates: 6,376 (norm 1.4312), 7,176 (1.1688), and 11,692 (1.0777), with none afterward.
Gradient clipping at norm 1 applied to 80 updates, including the initial training
period. The final 74-update window has mean gradient norm 0.2011 and maximum 0.2580.

The erase and write gates did separate: on the fixed held-out diagnostic sequence,
their final per-layer RMS difference ranges from 0.1463 to 0.3947, despite identical
initial values. The final training window independently records per-layer mean RMS
differences of 0.1574 to 0.3930. Thus the added projection is being used, without a
clear validation advantage in this comparison.

Run recipes, measurements, provenance, and launch instructions live in
[the attention study](../../../experiments/attention_scaling/README.md#independent-gdn-erase-gate).
The [shared scaffold](../../../scaffolds/transformer/README.md#gated-deltanet) defines
the implementation, and its
[validation record](../../../scaffolds/transformer/validation/README.md#fp32-chunk-precision-repair)
documents the arithmetic checks. The monitoring mirror is
[W&B run `200m-dk064-gdn-independent-erase-warmup400-seed0`](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/3113f4ce156c46c78fcc7c87efb60273).

Refresh the source pins from the repository root with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  results/exploratory/gdn_polynomial
```

The table reads each selected run's `completion.json` and `best_validation.json`;
the curve, health, and gate summaries read its `metrics.jsonl`. These canonical
measurements remain under the runs.

## Open decision

Decide whether the near-tie with separated gates warrants a matched-arithmetic
control and further seeds, and whether polynomial memory or alternative write
activations improve on this independent-erase baseline. Promotion of these GDN
extensions into the maintained attention comparison remains open. This topic
remains exploratory until promoted or discarded.

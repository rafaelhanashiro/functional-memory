# Per-degree score biases

Opened: 2026-11-11. Status: **allocated, not yet trained**; the trials at the 12-layer width-640
shape go first.

Does giving every order `r = 1..8` of the tanh query polynomial its own signed shift of the score,
`(s_ti + beta_r)^r` in place of `s_ti^r`, improve validation perplexity, with RoPE and with
per-order query–key distance gates without positional encoding? Which shift form matters: a
constant per order, one that depends on the query token, or one that depends on both tokens?

First, two trials at the 12-layer width-640 shape: the static and the query shift on the RoPE
control there. The six reference-shape runs follow only if a trial beats its control
([open decision](#open-decision)).

## Runs

| Role | Run | Label | Shift | Home cluster |
| --- | --- | --- | --- | --- |
| Primary | `rb70f9e57190f` | `query-m8-dk64-w-decay-score-bias-static` | static | Kennel |
| Primary | `r891870ef80e9` | `query-m8-dk64-w-decay-score-bias-query` | query | Kennel |
| Primary | `r3b1619f8a901` | `query-m8-dk64-w-decay-score-bias-query-key` | query–key | Kennel |
| Control | `r25fffc16c2f3` | `query-m8-dk64-w-decay` | – | Kennel |
| Primary | `ra2edcdc5ab60` | `query-m8-per-order-query-key-gate-nope-wd-score-bias-static` | static | Kennel |
| Primary | `rbf5d3bd6059c` | `query-m8-per-order-query-key-gate-nope-wd-score-bias-query` | query | Kennel |
| Primary | `r75dccfcf072e` | `query-m8-per-order-query-key-gate-nope-wd-score-bias-query-key` | query–key | Kennel |
| Control | `r227b0ed7c605` | `query-m8-per-order-query-key-gate-nope-wd` | – | Kennel |
| Trial | `r8dc6e147601c` | `query-m8-w-decay-l12-d640-score-bias-query` | query | Kennel |
| Trial | `rb4e5457ef2a0` | `query-m8-w-decay-l12-d640-score-bias-static` | static | Kennel |
| Control | `r95a5691d293f` | `query-m8-w-decay-l12-d640` | – | Kennel |

The [selection](selection.yaml) names each role and pins the runs' scientific files; the primaries
and the trials pin only their recipes until they train. The controls' validation-best perplexities:

| Control | Backbone | Shape | GPU | Update | Validation-best PPL |
| --- | --- | --- | --- | ---: | ---: |
| `r25fffc16c2f3` | tanh query `m = 8`, RoPE, projection weight decay | reference | B300 | 7,500 | 14.323 |
| `r227b0ed7c605` | tanh query `m = 8`, per-order query–key gates, no positional encoding, projection weight decay | reference | H100 SXM | 7,500 | 13.712 |
| `r95a5691d293f` | tanh query `m = 8`, RoPE, projection weight decay | 12-layer width-640 | B300 | 7,551 | 17.592 |

## Method

With `s_ti = Q_t K_i^T / sqrt(d_K)`, each run computes

    W(t,i) = c_0(x_t) G_0(t,i) + sum_(r=1..8) c_r(x_t) G_r(t,i) (s_ti + beta_r(t,i))^r,

with its control's coefficients `c_r(x_t)` (factorial scales unchanged) and gates (`G_r = 1` with
RoPE, or the per-order endpoint gates). Per layer, head and order: static `beta_r = b_r`; query
`beta_r = b_r + a_r tanh(u_r . x_t)`; query–key `beta_r = b_r + a_r tanh(u_r . x_t + v_r . x_i)`.
They start at `b = 0`, `a = 1`, `u = v = 0`, so every run starts as its control's function, and
`u`, `v` receive gradients from the first update; nothing else in the initialization changes.
AdamW decays `u` and `v` (0.1), never `b` or `a`. The shifts are unconstrained in sign. Each run is
its control's recipe with only the shift added and gradient-event capture on, at its shape's 1×
budget. The six reference-shape runs (7,570 updates) train on the blocked backend's causal strips,
which evaluate each shifted power directly rather than expanding it in powers of `s`; both of their
controls trained on the Mosaic kernels. Each trains on one B200, B300 or H100 SXM, the gated
query–key run on a B200 or B300 only (its update needs 147 GiB); on a B200 the six project 9.5–79
hours. The trials (7,551 updates of 128 × 2,048 tokens) keep their control's attention settings and
train on one H100 on the Mosaic score-bias kernels, which also form each shifted power directly.
Recipe, parameter counts, acceptance checks, measured update times and launch commands:
[study README](../../../README.md#per-degree-score-biases).

**Gradient-event capture.** After warmup (update 455; 454 for the trials), an update whose finite
pre-clip gradient norm exceeds `max(1, 5 × median)` of the previous 100 norms (once 20 exist) saves
the exact state it started from, with its batch and RNG coordinates; at most three states are kept
per run (the first event and the two largest later ones), and every event is logged in the run's
`gradient_events.jsonl`. Details: [study README](../../../README.md#per-degree-score-biases).

**Diagnostics.** At each validation the forward-only JAX reference of the same shifted kernel, on the
first held-out sequence and without gradients, records the bias parameters; per layer, head and
order, the shifts' and shifted scores' moments and extremes, the fractions of shifted scores below
zero and within `1e-3` of zero, and each order's term `T_r = c_r G_r (s + beta_r)^r`; per layer and
order, shifted-score histograms on [-16, 16] pooled over heads; and per layer and head, how much the
orders cancel, per pair (`sum_r |T_r| / |sum_r T_r|`) and per output (`sum_r ||o_r|| / ||o||`).
The reference-shape runs train, and their acceptance pilots run, on the causal strips; the trials on
the Mosaic kernels. Positive shifted scores do not imply positive weights: the coefficients `c_r` are
signed, so an order's term can be negative with a positive shifted score.

## Limits

- One seed per configuration.
- Both reference-shape controls trained on the Mosaic kernels (the RoPE control on a B300, the gated
  control on an H100 SXM), and the primaries train on the blocked backend's causal strips (on a B200,
  B300 or H100 SXM; the gated query–key run on a B200 or B300), so every reference-shape comparison
  also crosses attention implementations and possibly GPU types: the Mosaic kernels round the weights and score gradients to
  BF16 (7 stored mantissa bits) for their products, while the strips keep them as FP32 values whose
  products run as TF32 (10 stored mantissa bits) under JAX's default precision. No control is
  retrained on the strips, so this difference is not measured here.
- The trials and their control all round the weights to BF16 on Mosaic kernels, but on different
  kernels and GPUs: the trials' H100 row-shift score-bias kernels against the control's B300
  query-polynomial kernels.

## Open decision

The trials decide whether the six reference-shape runs are submitted: only if a trial's
validation-best perplexity beats its control's 17.592 by clearly more than seed noise, taken as about
0.25% (below about 17.548). Once the submitted runs complete: promote the topic to its own canonical collection,
or discard it with its runs and their artifacts.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/score_biases
```

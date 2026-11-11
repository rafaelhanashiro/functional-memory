# Per-degree score biases

Opened: 2026-11-11. Status: **allocated, not yet trained**.

Does giving every order `r = 1..8` of the tanh query polynomial its own signed shift of the score,
`(s_ti + beta_r)^r` in place of `s_ti^r`, improve validation perplexity, with RoPE and with
per-order query–key distance gates without positional encoding? Which shift form matters: a
constant per order, one that depends on the query token, or one that depends on both tokens?

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

The [selection](selection.yaml) names each role and pins the runs' scientific files; the primaries
pin only their recipes until they train. The controls' validation-best perplexities (update 7,500):

| Control | Backbone | GPU | Validation-best PPL |
| --- | --- | --- | ---: |
| `r25fffc16c2f3` | tanh query `m = 8`, RoPE, projection weight decay | B300 | 14.323 |
| `r227b0ed7c605` | tanh query `m = 8`, per-order query–key gates, no positional encoding, projection weight decay | H100 SXM | 13.712 |

## Method

With `s_ti = Q_t K_i^T / sqrt(d_K)`, each run computes

    W(t,i) = c_0(x_t) G_0(t,i) + sum_(r=1..8) c_r(x_t) G_r(t,i) (s_ti + beta_r(t,i))^r,

with its control's coefficients `c_r(x_t)` (factorial scales unchanged) and gates (`G_r = 1` with
RoPE, or the per-order endpoint gates). Per layer, head and order: static `beta_r = b_r`; query
`beta_r = b_r + a_r tanh(u_r . x_t)`; query–key `beta_r = b_r + a_r tanh(u_r . x_t + v_r . x_i)`.
They start at `b = 0`, `a = 1`, `u = v = 0`, so every run starts as its control's function, and
`u`, `v` receive gradients from the first update; nothing else in the initialization changes.
AdamW decays `u` and `v` (0.1), never `b` or `a`. The shifts are unconstrained in sign. Each run is
its control's recipe with only the shift added and gradient-event capture on, at the 1× budget
(7,570 updates), on the blocked backend's causal strips, which evaluate each shifted power directly
rather than expanding it in powers of `s`; both controls trained on the Mosaic kernels. Recipe,
parameter counts, acceptance checks and launch commands:
[study README](../../../README.md#per-degree-score-biases).

**Gradient-event capture.** After warmup (update 455), an update whose finite pre-clip gradient norm
exceeds `max(1, 5 × median)` of the previous 100 norms (once 20 exist) saves the exact state it
started from, with its batch and RNG coordinates; at most three states are kept per run (the first
event and the two largest later ones), and every event is logged in the run's
`gradient_events.jsonl`. Details: [study README](../../../README.md#per-degree-score-biases).

**Diagnostics.** At each validation the forward-only JAX reference of the same shifted kernel, on the
first held-out sequence and without gradients, records the bias parameters; per layer, head and
order, the shifts' and shifted scores' moments and extremes, the fractions of shifted scores below
zero and within `1e-3` of zero, and each order's term `T_r = c_r G_r (s + beta_r)^r`; per layer and
order, shifted-score histograms on [-16, 16] pooled over heads; and per layer and head, how much the
orders cancel, per pair (`sum_r |T_r| / |sum_r T_r|`) and per output (`sum_r ||o_r|| / ||o||`).
Training and the acceptance pilots use the causal strips. Positive shifted scores do not imply
positive weights: the coefficients `c_r` are signed, so an order's term can be negative with a
positive shifted score.

## Limits

- One seed per configuration.
- Both controls trained on the Mosaic kernels (the RoPE control on a B300, the gated control on an
  H100 SXM), and the primaries train on the blocked backend's causal strips (on a B200, B300 or H100),
  so every comparison also crosses attention implementations and possibly GPU types: the Mosaic
  kernels round the weights and score gradients to BF16 for their products, and the strips keep both
  in FP32. No control was retrained on the strips, so this difference is not measured here.

## Open decision

Once the six runs complete: promote the topic to its own canonical collection, or discard it with
its six runs and their artifacts.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/score_biases
```

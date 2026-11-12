# Score transforms

Opened: 2026-11-12. Status: **allocated, not yet trained**.

With a very negative score `s`, the degree-eight query polynomial gives a large weight: odd orders a
large negative one (the value pushed in the opposite direction), even orders a large positive one, where
`exp(s)` gives about zero and removes the key. Does evaluating the polynomial at a transformed score fix
this and improve validation perplexity at the 12-layer width-640 shape?

- **softplus input**: `z = log(1 + exp(s))`, which goes to 0 as `s` falls, with the constant coefficient
  started at 0, so that the kernel starts at `sum_(r=1..8) z^r/r!`: the Taylor polynomial of
  `e^z - 1 = e^s`, about `e^s` for negative scores and about `T_8(s) - 1` for large positive ones.
- **abs input**: `z = |s|`, from the control's Taylor start.

## Runs

| Role | Run | Label | Input | Home cluster |
| --- | --- | --- | --- | --- |
| Primary | `r1d342ca6e58c` | `query-m8-w-decay-l12-d640-softplus-input` | softplus, `c_0` from 0 | Kennel |
| Primary | `r17a1290cb4ac` | `query-m8-w-decay-l12-d640-abs-input` | abs | Kennel |
| Control | `r95a5691d293f` | `query-m8-w-decay-l12-d640` | `s` | Kennel |
| Reference | `r3e2fa2532c23` | `exp-l12-d640` | – (exp) | Kennel |

The [selection](selection.yaml) names each role and pins the runs' scientific files; the primaries pin only
their recipes until they train.

| Run | Kernel | GPU | Update | Validation-best PPL |
| --- | --- | --- | ---: | ---: |
| `r95a5691d293f` | tanh query `m = 8`, RoPE, projection weight decay | B300 | 7,551 | 17.592 |
| `r3e2fa2532c23` | exp, RoPE | B300 | 7,551 | 17.550 |

## Method

Each primary computes, with `s_ti = Q_t K_i^T / sqrt(d_K)` after QK RMSNorm, the head scale and RoPE,

    W(t,i) = sum_(r=0..8) c_r(x_t) z_ti^r,    o_t = head_RMSNorm(sum_(i<=t) W(t,i) v_i),

with the control's coefficients `c_r(x_t) = b_r/r! + (a_r/r!) tanh(x_t w_r)` (`a = 0` and `b_r = 1` stored
at the start). The softplus run starts `b_0` at 0 instead of 1; no parameter is added (99,678,056) and every
other initial array is the control's. Each recipe is the control's with only the transform, the softplus
run's constant start, and gradient-event capture changed, at the shape's 1x budget (7,551 updates of
128 × 2,048 tokens). The runs train on the blocked backend's causal strips on one B200, B300 or H100, the
only computation that evaluates the transform; the control trained on B300's Mosaic kernels. Recipe,
acceptance checks and launch commands: [study README](../../../README.md#score-transforms).

**Diagnostics.** At each validation the forward-only JAX scan of the same kernel, on the first held-out
sequence, records under `coefficients.score_transform`, per layer and head, the raw scores' and the inputs'
moments and extrema, the fraction of raw scores below zero, and the fraction of causal pairs at which the
polynomial decreases in `z`; and per layer, histograms of `s` and `z` on [-16, 16] pooled over heads. With
the coefficient statistics (each order's mean, spread and extremes per head) they show the coefficients'
signs and whether the kernel is monotone where the scores fall.

## Limits

- One seed per configuration.
- Every comparison also crosses attention implementations: the control's Mosaic kernels round the weights
  and score gradients to BF16 for their products, while the strips keep FP32 values whose products run as
  TF32 under JAX's default precision. No control is retrained on the strips.

## Open decision

A primary counts as a win only if its validation-best perplexity beats the control's 17.592 by at least
0.5% (17.504 or below); seed noise at this budget is about 0.1–0.2%. Each is also compared with exp at this
shape (17.550). Once both complete: promote the topic to its own canonical collection, or discard it with its
runs and their artifacts.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/score_transforms
```

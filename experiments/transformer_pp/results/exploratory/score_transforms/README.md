# Score transforms

Opened: 2026-11-12. Status: **open**; each run's status is in its record and the study's `runs.csv`.

With a very negative score `s`, the degree-eight query polynomial gives a large weight: odd orders a
large negative one (the value pushed in the opposite direction), even orders a large positive one, where
`exp(s)` gives about zero and removes the key. Does evaluating the polynomial at a transformed score fix
this and improve validation perplexity at the 12-layer width-640 shape, and, for softplus, at the
reference shape ([below](#reference-shape))?

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
128 × 2,048 tokens). The runs train on the blocked backend's causal strips on one B200, B300 or H100 (on an
H100 PCIe the pilots project 7.3–7.5 hours); the control trained on B300's Mosaic kernels. Recipe, acceptance
checks and launch commands: [study README](../../../README.md#score-transforms).

**Diagnostics.** At each validation the forward-only JAX scan of the same kernel, on the first held-out
sequence, records under `coefficients.score_transform`, per layer and head, the raw scores' and the inputs'
moments and extrema, the fraction of raw scores below zero, and the fraction of causal pairs at which the
polynomial decreases in `z`; and per layer, histograms of `s` and `z` on [-16, 16] pooled over heads. With
the coefficient statistics (each order's mean, spread and extremes per head) they show the coefficients'
signs and whether the kernel is monotone where the scores fall.

## Reference shape

The softplus variant also runs at the reference shape (15 layers, 14 heads, width 896, `d_K = 64`, 7,570
updates of 256 × 2,048 tokens), against the RoPE query control there:

| Role | Run | Label | Input | Home cluster |
| --- | --- | --- | --- | --- |
| Primary | `ra3b54e2489f8` | `query-m8-dk64-w-decay-softplus-input` | softplus, `c_0` from 0 | Kennel |
| Control | `r25fffc16c2f3` | `query-m8-dk64-w-decay` | `s` | Kennel |
| Reference | `r6a76e7fbde45` | `exp-head-scale` | – (exp) | Kennel |
| Reference | `ra7ee8a4b2544` | `fixed-m8-head-scale` | `s`, fixed Taylor `m = 8` | Kennel |

| Run | Kernel | GPU | Update | Validation-best PPL |
| --- | --- | --- | ---: | ---: |
| `r25fffc16c2f3` | tanh query `m = 8`, RoPE, projection weight decay | B300 | 7,500 | 14.323 |
| `r6a76e7fbde45` | exp, RoPE | B300 | 7,500 | 14.257 |
| `ra7ee8a4b2544` | fixed Taylor `m = 8`, RoPE | H100 | 7,500 | 14.218 |

The primary is its control's recipe with the transform, the zero constant start and gradient-event capture
(200,150,806 parameters, every initial array the control's but `b_0 = 0`). Unlike the 12-layer width-640
runs it keeps its control's attention implementation: the Mosaic query-polynomial kernels, here their
transformed family, which evaluates `softplus(s)` inside the same causal tiles
([validation](../../../../../scaffolds/transformer/validation/README.md#score-transforms-on-the-mosaic-query-polynomial-kernels-2026-11-12)),
on H100 alone. On an H100 PCIe its pilot projects 8.3 hours (an H100 SXM about 5.8), where the blocked strips
would take 23–30.
Recipe, acceptance checks and launch command:
[study README](../../../README.md#reference-shape-softplus-run).

## Limits

- One seed per configuration.
- The 12-layer width-640 comparisons also cross attention implementations: the control's Mosaic kernels
  round the weights and score gradients to BF16 for their products, while the strips keep FP32 values whose
  products run as TF32 under JAX's default precision. No control is retrained on the strips. The
  reference-shape primary and its control both use the Mosaic kernels' BF16 products, the control on B300.

## Open decision

A primary counts as a win only if its validation-best perplexity beats its control's by at least 0.5%
(17.504 or below against 17.592 at the 12-layer width-640 shape; 14.251 or below against 14.323 at the
reference shape); seed noise at this budget is about 0.1–0.2%. Each is also compared with exp at its shape
(17.550; 14.257), and the reference-shape primary with fixed Taylor at degree eight (14.218). Once the runs
complete: promote the topic to its own canonical collection, or discard it with its runs and their
artifacts.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/score_transforms
```

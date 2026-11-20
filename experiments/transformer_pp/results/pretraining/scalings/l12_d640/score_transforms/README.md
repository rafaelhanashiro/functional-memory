# Score transforms

Opened: 2026-11-12. Both runs are complete: softplus cleared the bar, abs did not. Each run's status is in its
record and the study's `runs.csv`.

With a very negative score `s`, the degree-eight query polynomial gives a large weight: odd orders a
large negative one (the value pushed in the opposite direction), even orders a large positive one, where
`exp(s)` gives about zero and removes the key. Does evaluating the polynomial at a transformed score fix
this and improve validation perplexity at the 12-layer width-640 shape? The softplus input also ran at the
reference shape, at degrees eight and twelve: [softplus_inputs](../../../parametrizations/sos/softplus_inputs/README.md).

- **softplus input**: `z = log(1 + exp(s))`, which goes to 0 as `s` falls, with the constant coefficient
  started at 0, so that the kernel starts at `sum_(r=1..8) z^r/r!`: the Taylor polynomial of
  `e^z - 1 = e^s`, about `e^s` for negative scores and about `T_8(s) - 1` for large positive ones.
- **abs input**: `z = |s|`, from the control's Taylor start.

## Runs

| Role | Run | Label | Input | Attention, GPU | Home cluster |
| --- | --- | --- | --- | --- | --- |
| Primary | `r1d342ca6e58c` | `query-m8-w-decay-l12-d640-softplus-input` | softplus, `c_0` from 0 | causal strips, H100 PCIe | Kennel |
| Primary | `r17a1290cb4ac` | `query-m8-w-decay-l12-d640-abs-input` | abs | causal strips, H100 PCIe | Kennel |
| Control | `r95a5691d293f` | `query-m8-w-decay-l12-d640` | `s` | Mosaic query polynomial, B300 | Kennel |
| Reference | `r3e2fa2532c23` | `exp-l12-d640` | – (exp) | fused Triton, B300 | Kennel |

The [selection](selection.yaml) names each role and pins the runs' scientific files.

## Method

Each primary computes, with `s_ti = Q_t K_i^T / sqrt(d_K)` after QK RMSNorm, the head scale and RoPE,

    W(t,i) = sum_(r=0..8) c_r(x_t) z_ti^r,    o_t = head_RMSNorm(sum_(i<=t) W(t,i) v_i),

with the control's coefficients `c_r(x_t) = b_r/r! + (a_r/r!) tanh(x_t w_r)` (`a = 0` and `b_r = 1` stored
at the start). The softplus run starts `b_0` at 0 instead of 1; no parameter is added (99,678,056) and every
other initial array is the control's. Each recipe is the control's with only the transform, the softplus
run's constant start, gradient-event capture and the attention backend (`blocked` with `strips`, where the
control has `mosaic` and `scan`) changed, at the shape's 1x budget (7,551 updates of 128 × 2,048 tokens). Both runs trained on the blocked backend's causal strips, each on one H100 PCIe; the
control trained on B300's Mosaic kernels. Recipe, acceptance checks and launch commands:
[study README](../../../../../README.md#score-transforms).

**Diagnostics.** At each validation the forward-only JAX scan of the same kernel, on the first held-out
sequence, records under `coefficients.score_transform`, per layer and head, the raw scores' and the inputs'
moments and extrema, the fraction of raw scores below zero, and the fraction of causal pairs at which the
polynomial decreases in `z`; and per layer, histograms of `s` and `z` on [-16, 16] pooled over heads. With
the coefficient statistics (each order's mean, spread and extremes per head) they show the coefficients'
signs and whether the kernel is monotone where the scores fall. The values below are from the final
validation, update 7,551 (2,098,176 causal pairs per head).

## Results

| Run | Input | Validation-best PPL (update) | vs control | vs exp | Terminal PPL (update 7,551) | vs control | vs exp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `r1d342ca6e58c` | softplus | 17.488 (7,551) | −0.59% | −0.36% | 17.488 | −0.59% | −0.36% |
| `r17a1290cb4ac` | abs | 17.941 (7,551) | +1.98% | +2.22% | 17.941 | +1.98% | +2.22% |
| `r95a5691d293f` | `s` (control) | 17.592 (7,551) | – | +0.24% | 17.592 | – | +0.24% |
| `r3e2fa2532c23` | – (exp) | 17.550 (7,551) | −0.24% | – | 17.550 | −0.24% | – |

Validation perplexity against the control's at the same update (`metrics.jsonl`; warmup ends at 454):

| Update | 200 | 300 | 500 | 700 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 5,000 | 7,551 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| softplus | −1.6% | −1.2% | +0.4% | +0.6% | −0.8% | −0.7% | −0.4% | −0.7% | −0.6% | −0.6% | −0.59% |
| abs | +4.7% | +10.0% | +18.3% | +13.2% | +6.8% | +4.6% | +3.9% | +3.6% | +3.0% | +2.3% | +1.98% |

The softplus kernel and its coefficients (`metadata.json`):

| Quantity | Value |
| --- | --- |
| Causal pairs where the kernel increases with the score (mean over 120 heads) | 95.2% |
| Heads with some decreasing pairs | 56 of 120 (47%); the largest share in one head 94% |
| Raw scores below zero (mean over heads; per-head range) | 55% (9–87%) |
| `c_0` below zero (token mean per head) | 89 of 120 heads |

| Order `r` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean `c_r r!` | −0.12 | 0.68 | 0.56 | 0.43 | 0.33 | 0.18 | 0.34 | 0.89 | 1.49 |

`c_r r!` is 1 for every order of the Taylor polynomial; the softplus run starts with `c_0 = 0`. Each entry
averages, over the 12 layers and 10 heads, the head's mean coefficient over the diagnostic sequence's tokens.

Gradient events (after warmup, a pre-clip norm above `max(1, 5 × median)` of the previous 100): softplus 8,
the largest 12.38 at update 2,845; abs 5, the largest 7.83 at update 5,707. Each run keeps three pre-update
states (softplus 2,845, 5,741, 6,238; abs 1,765, 5,707, 6,823), listed in its record.

## Findings

- **Softplus cleared the bar.** Validation-best and terminal 17.488 at update 7,551: 0.59% below the
  control (the bar is 0.5%, 17.504) and 0.36% below exp. It led the control at every validation except
  updates 500–700 (up to 0.6% behind), by 0.57–0.74% from update 2,500 on (0.34–0.40% ahead of exp).
- **The softplus kernel is close to monotone.** It increases with the score at 95.2% of causal pairs on
  average; 47% of heads have some decreasing pairs, one head 94%. 55% of raw scores are negative on
  average.
- **Very negative scores get a small negative weight.** As `s` falls the kernel tends to `c_0`, which
  drifted from 0 to −0.12 on average.
- **The learned coefficients sit below Taylor in orders 1–7 and above it in order 8**: mean `c_r r!` of
  0.18–0.68 for orders 1–6, 0.89 for order 7 and 1.49 for order 8.
- **Abs lost about 2%.** Validation-best and terminal 17.941 at update 7,551, 1.98% above the control. It
  trailed by 10–18% around the learning-rate peak (updates 300–700) and recovered to about 2% behind.
  `|s|` gives a score and its negation the same weight, like the even-only kernels of the
  [even_powers topic](../../../other/even_powers/README.md), which trail full degree eight by about 2% at the
  reference shape.

## Limits

- One seed per configuration; seed noise at this budget is about 0.1–0.2%.
- The comparisons also cross attention implementations and GPUs: the primaries trained on the causal strips
  on H100 PCIe, the control on B300's Mosaic kernels and exp on B300's fused Triton kernels. The Mosaic
  kernels round the weights and score gradients to BF16 for their products, while the strips keep FP32
  values whose products run as TF32 under JAX's default precision. No control is retrained on the strips.
- The kernel and coefficient statistics come from one held-out sequence at the final validation.

## Open question

A primary counts as a win only if its validation-best perplexity beats its control's by at least 0.5%
(17.504 or below against 17.592). Softplus (17.488) cleared it; abs (17.941) did not and gets no further
runs. Softplus then fell short of the bar at the reference shape at degrees eight and twelve
([softplus_inputs](../../../parametrizations/sos/softplus_inputs/README.md)); at twice this shape's budget, on
the Mosaic transformed kernels, it is 0.25% (degree eight) and 0.11% (degree twelve) below the query runs
([budgets](../budgets/README.md)). All runs are complete. Keep the folder, or delete it with its runs and
their artifacts.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/scalings/l12_d640/score_transforms
```

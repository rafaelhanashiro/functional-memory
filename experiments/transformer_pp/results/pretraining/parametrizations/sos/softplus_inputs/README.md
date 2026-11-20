# Softplus inputs

Opened: 2026-11-12. Both runs are complete and fell short of the bar, each against the control at its degree:
0.42% below it at degree eight, 0.02% at degree twelve. Three direct-coefficient runs on `z`, static and times `z`,
are [planned](#direct-polynomials-of-z-static-and-times-z). Each run's status is in its record and the study's
`runs.csv`.

With a very negative score `s`, the query polynomial gives a large weight: odd orders a large negative one
(the value pushed in the opposite direction), even orders a large positive one, where `exp(s)` gives about
zero and removes the key. Does evaluating the tanh query polynomial at `z = softplus(s) = log(1 + exp(s))`,
which goes to 0 as `s` falls, improve validation perplexity at the reference shape at degrees eight and
twelve? With the constant coefficient started at 0 the kernel starts at `sum_(r=1..m) z^r/r!`, the Taylor
polynomial of `e^z - 1 = e^s`: about `e^s` for negative scores and about `T_m(s) - 1` for large positive ones.
The variant first ran at the 12-layer width-640 shape, where it cleared the bar beside an abs input that did
not: [score_transforms](../../../scalings/l12_d640/score_transforms/README.md).

## Runs

At the reference shape (15 layers, 14 heads, width 896, `d_K = 64`, 7,570 updates of 256 × 2,048 tokens),
against the RoPE query control at each degree:

| Role | Run | Label | Input | Attention, GPU | Home cluster |
| --- | --- | --- | --- | --- | --- |
| Primary | `ra3b54e2489f8` | `query-m8-dk64-w-decay-softplus-input` | softplus, `c_0` from 0 | Mosaic transformed query polynomial, H100 PCIe | Kennel |
| Control | `r25fffc16c2f3` | `query-m8-dk64-w-decay` | `s` | Mosaic query polynomial, B300 | Kennel |
| Primary, `m = 12` | `r9eb845aae5bb` | `query-m12-dk64-w-decay-softplus-input` | softplus, `c_0` from 0 | Mosaic transformed query polynomial, H100 PCIe | Kennel |
| Control, `m = 12` | `r0fceea0627c8` | `query-m12-dk64-w-decay` | `s` | Mosaic query polynomial, B300 | Kennel |
| Reference | `r6a76e7fbde45` | `exp-head-scale` | – (exp) | fused Triton, B300 | Kennel |
| Reference | `ra7ee8a4b2544` | `fixed-m8-head-scale` | `s`, fixed Taylor `m = 8` | Mosaic query polynomial, H100 PCIe | Kennel |
| Reference, `m = 12` | `rb2b70dae4d21` | `fixed-m12-head-scale` | `s`, fixed Taylor `m = 12` | Mosaic query polynomial, B300 to update 1,500, then H100 PCIe | Kennel |

The [selection](selection.yaml) names each role and pins the runs' scientific files.

## Method

Each primary computes, with `s_ti = Q_t K_i^T / sqrt(d_K)` after QK RMSNorm, the head scale and RoPE, and
`z_ti = softplus(s_ti)`,

    W(t,i) = sum_(r=0..m) c_r(x_t) z_ti^r,    o_t = head_RMSNorm(sum_(i<=t) W(t,i) v_i),

with the control's coefficients `c_r(x_t) = b_r/r! + (a_r/r!) tanh(x_t w_r)` (`a = 0` and `b_r = 1` stored
at the start, but `b_0 = 0`). Each primary is its control's recipe with the transform, the zero constant start
and gradient-event capture (200,150,806 parameters at degree eight and 200,905,126 at degree twelve, every
initial array the control's but `b_0 = 0`). Both keep their controls' attention implementation: the Mosaic
query-polynomial kernels, here their transformed family, which evaluates `softplus(s)` inside the same causal
tiles ([validation](../../../../../../../scaffolds/transformer/validation/README.md#score-transforms-on-the-mosaic-query-polynomial-kernels-2026-11-12)),
on H100 alone. Each trained on one H100 PCIe, where the blocked strips would take 23–30 hours at degree eight.
Recipes, acceptance checks and launch commands: the study README at
[degree eight](../../../../../README.md#reference-shape-softplus-run) and
[degree twelve](../../../../../README.md#reference-shape-softplus-at-degree-twelve).

**Diagnostics.** At each validation the forward-only JAX scan of the same kernel, on the first held-out
sequence, records under `coefficients.score_transform`, per layer and head, the raw scores' and the inputs'
moments and extrema, the fraction of raw scores below zero, and the fraction of causal pairs at which the
polynomial decreases in `z`; and per layer, histograms of `s` and `z` on [-16, 16] pooled over heads. The
values below are the degree-eight run's, from its final validation, update 7,570 (2,098,176 causal pairs per
head).

## Results

| Run | Input | Validation-best PPL (update) | vs control | vs exp | vs fixed Taylor | Terminal PPL (update 7,570) | vs control | vs exp | vs fixed Taylor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ra3b54e2489f8` | softplus | 14.262 (7,500) | −0.42% | +0.04% | +0.31% | 14.265 | −0.44% | +0.03% | +0.30% |
| `r25fffc16c2f3` | `s` (control) | 14.323 (7,500) | – | +0.46% | +0.73% | 14.328 | – | +0.46% | +0.74% |
| `r6a76e7fbde45` | – (exp) | 14.257 (7,500) | −0.46% | – | +0.27% | 14.262 | −0.46% | – | +0.27% |
| `ra7ee8a4b2544` | `s`, fixed Taylor `m = 8` | 14.218 (7,500) | −0.73% | −0.27% | – | 14.223 | −0.73% | −0.27% | – |

At degree twelve, against its control `r0fceea0627c8` (exp and softplus at degree eight are in the table above):

| Run | Input | Validation-best PPL (update) | vs control | vs exp | vs softplus `m = 8` | vs fixed Taylor `m = 12` | Terminal PPL (update 7,570) | vs control | vs exp | vs softplus `m = 8` | vs fixed Taylor `m = 12` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `r9eb845aae5bb` | softplus, `m = 12` | 14.222 (7,500) | −0.02% | −0.24% | −0.28% | −0.60% | 14.226 | −0.02% | −0.25% | −0.28% | −0.59% |
| `r0fceea0627c8` | `s` (control), `m = 12` | 14.226 (7,500) | – | −0.22% | −0.26% | −0.58% | 14.230 | – | −0.23% | −0.25% | −0.57% |
| `rb2b70dae4d21` | `s`, fixed Taylor `m = 12` | 14.309 (7,500) | +0.58% | +0.36% | +0.32% | – | 14.311 | +0.57% | +0.34% | +0.32% | – |

Validation perplexity against each primary's control at the same update (`metrics.jsonl`; warmup ends at 455):

| Update | 200 | 300 | 400 | 500 | 700 | 1,000 | 1,500 | 2,000 | 2,500 | 3,000 | 5,000 | 7,500 | 7,570 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| softplus `m = 8` | −4.7% | +3.7% | +4.9% | +1.0% | +2.0% | +0.9% | −0.1% | −0.5% | −0.5% | −0.4% | −0.4% | −0.42% | −0.44% |
| softplus `m = 12` | +1.5% | +1.1% | +4.6% | +4.9% | +2.0% | +1.5% | +0.6% | +0.3% | +0.3% | +0.1% | +0.1% | −0.02% | −0.02% |

The degree-eight softplus kernel and its coefficients (`metadata.json`):

| Quantity | Value |
| --- | --- |
| Causal pairs where the kernel increases with the score (mean over 210 heads) | 96.5% |
| Heads with some decreasing pairs | 92 of 210 (44%); the largest share in one head 97% |
| Raw scores below zero (mean over heads; per-head range) | 53% (13–88%) |
| `c_0` below zero (token mean per head) | 136 of 210 heads |

| Order `r` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean `c_r r!` | −0.11 | 0.69 | 0.57 | 0.48 | 0.36 | 0.24 | 0.41 | 0.88 | 1.43 |

Each entry averages, over the 15 layers and 14 heads, the head's mean coefficient over the diagnostic
sequence's tokens.

Gradient events (after warmup, a pre-clip norm above `max(1, 5 × median)` of the previous 100): at degree eight
one, at update 6,376 (pre-clip norm 1.91); at degree twelve eight, the largest 11.93 at update 2,612. The
degree-eight run keeps its one pre-update state and the degree-twelve run three (updates 2,612, 3,028 and 3,885),
listed in their records.

## Findings

- **Softplus at degree eight fell short of the bar at the reference shape.** Validation-best 14.262 at update
  7,500 (terminal 14.265): 0.42% below the control, 0.08 percentage points (0.011 perplexity) short of the 0.5%
  bar (14.251), where the 12-layer width-640 run gained 0.59%. It ties exp (+0.04%; 0.36% ahead at the 12-layer
  shape) and trails fixed Taylor at degree eight by 0.31%.
- **It trailed the control around the learning-rate peak and led it from update 2,000 on.** Ahead by 4.7% at
  update 200, behind at every validation from 300 to 1,000 (by up to 4.9%, at 400), within 0.22% at
  1,100–1,500, then 0.40–0.55% ahead; within 0.07% of exp from update 3,000 on.
- **The degree-eight kernel and its coefficients look like the
  [12-layer width-640 run's](../../../scalings/l12_d640/score_transforms/README.md).** The kernel increases with the
  score at 96.5% of causal pairs on average; 44% of heads have some decreasing pairs, one head 97%; 53% of raw
  scores are negative on average. `c_0`, the weight a very negative score tends to, drifted from 0 to −0.11 on
  average, and the mean `c_r r!` again sits below Taylor in orders 1–7 (0.24–0.69 for orders 1–6, 0.88 for
  order 7) and above it in order 8 (1.43).
- **Softplus at degree twelve fell short too.** Validation-best 14.222 at update 7,500 (terminal 14.226): 0.02%
  below its control (14.226), 0.48 percentage points (0.068 perplexity) short of the bar (14.155). It trailed the
  control at every validation through update 7,000, by up to 4.9% (at update 500) and by at most 0.12% from
  update 3,000 on, and led it by 0.02% at 7,500 and 7,570. It is 0.24% below exp, 0.28% below softplus at degree
  eight and 0.60% below fixed Taylor at degree twelve. From degree eight to twelve the control gains 0.68%,
  softplus 0.28%.

## Limits

- One seed per configuration; seed noise at this budget is about 0.1–0.2%.
- The primaries and their controls all use the Mosaic kernels' BF16 products, the primaries on H100 PCIe and
  the controls on B300; exp trained on B300's fused Triton kernels, fixed Taylor at degree eight on H100's
  Mosaic query-polynomial kernels and at degree twelve on B300's to update 1,500, then H100's.
- The kernel and coefficient statistics come from one held-out sequence at the final validation.

## Direct polynomials of z: static, and times z

Planned; no results. Added 2026-11-20. At degree eight, do static coefficients or a factor `z` help the direct
(not sum-of-squares) polynomial of `z = softplus(s)`? Three runs, each its partner's recipe with the listed score
functions and gradient-event capture, from the partner's exact initial parameters. Validation-best perplexity of the
existing runs:

| Run | Weight | Start | Partner | Comparisons |
| --- | --- | --- | --- | --- |
| `r2026eaa99f9a` | static `p(z)` | `T_8(z)`, about `1 + e^s` | static `p(s)`, `r9c69dd7cfc8d`, 14.294 | static `p(z)` with `c_0 = 0`, `rc28ef4dbe26d`, 14.192; query `p_t(z)`, `ra3b54e2489f8`, 14.262 |
| `rd00848e00f1d` | static `z p(z)` | `z T_8(z)` | `r9c69dd7cfc8d` | static `p(z)`, `r2026eaa99f9a`; `rc28ef4dbe26d`, which also vanishes at very negative scores |
| `r7f80334aafbd` | `z p_t(z)` | `z (T_8(z) − 1)`, about `softplus(s) e^s` | `ra3b54e2489f8` | query `p_t(z)` with `c_0 = 0`, `rb5b604ae829a`, 14.250; `z h_t(z)`, `r841f35a24241`, 14.095; softplus(s) exp(s), `rbd24b948a240`, 14.132; exp, 14.257 |

- The factor `z` is the parameter-free score prefactor `f = softplus` on the Mosaic transformed kernels
  (`query_softplus` for `p(z)`, `query_softplus_times_softplus` for `z p(z)`; static coefficients are broadcast
  over queries), on one B200 or H100
  ([methods and launch](../../../../../README.md#score-prefactors-swish-and-narrow-keys)).
- The static constant `c_0` stays learnable from its Taylor start of 1: no scaffold option starts a learnable
  static constant at zero (`rc28ef4dbe26d` removes it), so `p(z)` starts near `1 + e^s`, where the query runs and
  `rc28ef4dbe26d` start near `e^s`.

## Open question

A primary counts as a win only if its validation-best perplexity beats its control's by at least 0.5%
(14.251 or below against 14.323 at degree eight, 14.155 or below against 14.226 at degree twelve). Neither
did: softplus at degree eight (14.262) is 0.42% below its control, level with exp (14.257) and 0.31% behind
fixed Taylor at degree eight (14.218); at degree twelve (validation-best 14.222, terminal 14.226) it is 0.02%
below its control, 0.24% below exp and 0.60% below fixed Taylor at degree twelve (14.309). Softplus inputs
composed with query sum of squares are in [softplus_sos](../softplus_sos/README.md), and with structurally
zero constants in [zero_constant](../../../other/zero_constant/README.md). Keep the folder, or delete it with
its runs and their artifacts.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/parametrizations/sos/softplus_inputs
```

# Sum-of-squares query coefficients

Opened 2026-11-11.

**Question.** Learned tanh query coefficients trail fixed Taylor at `m = 8` under the RoPE
Transformer++ recipe. In their first 500 updates the coefficients lower the polynomial at negative
scores and make the kernel signed. Does a parameterization that cannot go negative, or a start
closer to where the coefficients move, let learned query coefficients beat fixed Taylor and exp as
the degree grows?

## Runs

All runs use the reference shape, 1× budget, seed 42, RoPE and per-head output RMSNorm. The
explicit selection is [`selection.yaml`](selection.yaml).

| Role | Run | Recipe | Kernel, GPU |
| --- | --- | --- | --- |
| Sum of squares, m = 8 | `reeefbcc19abf` | query m = 8 (`r25fffc16c2f3`) with sum-of-squares coefficients | Mosaic, H100 |
| Sum of squares, m = 12 | `r3d52b7ab43d1` | query m = 12 (`r0fceea0627c8`) with sum-of-squares coefficients | Mosaic, H100 |
| Sum of squares, m = 16 | `rdfc61c988309` | the m = 12 run at degree 16 | blocked strips, H100 |
| Hermite start, m = 8 | `r6adefb42f443` | query m = 8 with `b_r` starting at the fixed Hermite coefficients | Mosaic, B300 then H100 |
| Controls | `r25fffc16c2f3`, `r0fceea0627c8` | query coefficients, m = 8 and 12 | Mosaic, B300 |
| | `ra7ee8a4b2544`, `rb2b70dae4d21` | fixed Taylor, m = 8 and 12 | Mosaic, H100; B300 then H100 |
| | `ra040de24096c` | fixed Hermite, m = 8 | Mosaic, H100 |
| | `r6a76e7fbde45` | exp | fused |

**Sum of squares.** Each query's polynomial is `p(s) = q_1(s)^2 + q_2(s)^2`, with `q_1` of degree
`m/2` and `q_2` of degree `m/2 - 1`: the same `m + 1` coefficients as the direct form. Factor `k`
is `f_k (1 + g_k(x_t)) / 2` with `g_k = b_k + a_k tanh(w_k . x_t)`, where `f` are the fixed Taylor
square factors (`attention/nonnegative_polynomial.py`), so `b = 1`, `a = 0` gives exactly `T_m`,
and a step in `g` has the direct form's first-order effect on the polynomial. The factors are
expanded per query to `c_r(x_t)` for the unchanged query kernel. Degree 16 exceeds the Mosaic
kernels and runs on the blocked backend's causal strips.

## Findings

Validation-best PPL, at update 7,500 in every run; terminal (update 7,570) in parentheses.

| Degree | Sum of squares | Query coefficients | Fixed Taylor | vs exp (14.257) | vs query | vs Taylor |
| ---: | --- | --- | --- | ---: | ---: | ---: |
| 8 | 14.196 (14.200) | 14.323 | 14.218 | −0.43% | −0.89% | −0.16% |
| 12 | **14.165** (14.168) | 14.226 | 14.309 | **−0.65%** | −0.43% | −1.00% |
| 16 | 14.168 (14.172) | not run | not run | −0.63% | — | — |

- **Sum of squares beats exp at every degree, and degree helps from 8 to 12.** m = 12 and m = 16
  are below exp at every evaluation from update 1,000 on (by 0.30–1.35% and 0.54–1.38%); m = 8
  from update 1,500 on (by 0.33–0.48%). From update 4,000 on m = 12 is 0.14–0.29% below m = 8.
- **Degree 16 adds nothing over 12** (+0.02%); it also changes the kernel (blocked strips).
- **It beats the direct form by a margin larger than any recorded rerun gap** (−0.89% at m = 8,
  −0.43% at m = 12). It gives up the direct form's early lead (+2.32% vs exp at update 500,
  against +0.96%) and stays ahead of it from update 1,500 on.
- **Training is calm.** After update 1,500 each sum-of-squares run clips one update, at update
  6,376, where fixed Taylor m = 12 also spikes (a batch, not the mechanism); largest gradient norms
  1.68, 1.50 and 1.18, against 3.84 for the direct form at m = 8.
- **The polynomials change shape, not just scale.** At m = 8 the token-mean polynomial, with its
  overall scale removed (the head RMSNorm cancels it), weights negative scores 25–30% below Taylor;
  orders 6–8 fall 12–26% relative to the lower orders. The per-query variation sits in `q_1`'s three
  lowest coefficients (token CV 9–17%) and mostly changes how deep each query's floor is.
- **Heads can switch off.** After update 1,500 the smallest pre-RMSNorm head-output RMS reaches
  about `4e-7` (m = 8, 12) and `2e-7` (m = 16), where the head RMSNorm's gain sits at its
  `1/sqrt(eps)` cap of 1,000.
  The direct form does this too (exactly 0 in the Hermite-start run); fixed Taylor stays above 0.01.
  The epsilon (`1e-6`) has never been varied.
- **The Hermite start is not the lever.** Starting the query coefficients at the fixed Hermite
  polynomial ends 14.371: 0.34% above the Taylor start and 1.03% above fixed Hermite.

The [SoS training-dynamics diagnostic](../../diagnostics/sos_training_dynamics/README.md) records
the coefficient trajectories of the m = 8 and m = 12 runs.

## Matched-head tanh saturation

The forward-only probe `rc3dfb9569345` measures the SoS factor generators, compared
with regular-query probe `r1ac137c4f1ce`. Both use layer 8, head 1 (indices 7, 0),
checkpoint update 7,500, and the same first held-out 2,048-token input. The native
Mosaic forward is preserved. Saturation means `abs(tanh(W_j^T x)) >= 0.99`;
fractions average all query tokens and generator coordinates equally.

| Degree | Regular query saturated | SoS saturated | Mean tanh derivative, regular / SoS | Coordinates saturated on >90% of tokens, regular / SoS |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 48.46% | 17.42% | 0.1994 / 0.3565 | 2/9 / 0/9 |
| 12 | 53.54% | 61.90% | 0.1836 / 0.1498 | 4/13 / 5/13 |

**SoS has less saturation at m = 8 in this head, but more at m = 12.** Both SoS
models have better validation perplexity than their regular-query controls.
These observations do not support uniformly reduced tanh saturation as an
explanation of SoS's advantage. They do not rule out an effect at other heads,
sequences, checkpoints, or output-weighted generator sensitivities.

At m = 8, the norm discrepancy remains: mean post-RMSNorm activation norm is
28.72 (regular: 28.17), and projection-vector norms are 1.42–1.58 (regular:
1.26–1.64). Less saturation therefore occurs without matching the two norms.
SoS coordinate mean projections range from -1.52 to 1.79, while regular query
has two means above 4.7; projection direction and activation alignment matter.
At m = 12, three SoS factor generators are saturated on every sampled token,
and two more on over 96%. A nearly saturated generator contributes little local
tanh derivative but may still supply a useful near-constant factor coefficient;
its overall influence also depends on amplitude, the square factors, and downstream
attention. The displayed tanh derivatives are not full model gradient estimates.

Regular-query coordinates are polynomial orders; SoS coordinates are the factors
`U0..U(m/2), V0..V(m/2-1)` in `p = U^2 + V^2`. Coordinate averages are descriptive,
not weighted by amplitude, polynomial contribution, or model-output sensitivity.
The probes verify the selected raw parameters against training measurements and
reconstruct native generator outputs. Historical SoS expanded-coefficient moments
are not compared with the pre-expansion generator moments.

The explicit [saturation selection](saturation_selection.yaml),
[figure](saturation.png), and [summary](saturation_summary.csv) remain part of this
open exploration. Regenerate using committed compact metrics, without a GPU,
checkpoint, dataset or numerical archive:

```bash
python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/sum_of_squares/saturation_selection.yaml
python experiments/transformer_pp/results/exploratory/sum_of_squares/saturation_plot.py
```

## Limits

- One seed per cell; differences below about 0.2% are not resolved.
- GPUs differ within comparisons: the sum-of-squares runs trained on H100 PCIe, their query
  controls on B300. Fixed Taylor m = 12 and the Hermite start moved from B300 to H100 PCIe at
  update 1,500 after the B300 node failed.
- The m = 16 point differs in kernel as well as degree and has no direct-form or Taylor partner.

## Open decision

Sum of squares is the first configuration on this recipe whose learned polynomial beats exp
(−0.65% at m = 12) and improves with degree up to 12. Decide whether to confirm m = 12 (and its
direct-form and Taylor partners) with a second seed and promote the topic as its own collection,
or discard it.

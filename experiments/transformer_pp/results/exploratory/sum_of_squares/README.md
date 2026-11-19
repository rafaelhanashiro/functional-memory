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
| Sum of squares, m = 4 | `r016b914d927a` | query m = 4 (`rca97cc9db8f2`) with sum-of-squares coefficients | Mosaic, H100 SXM |
| Sum of squares, m = 8 | `reeefbcc19abf` | query m = 8 (`r25fffc16c2f3`) with sum-of-squares coefficients | Mosaic, H100 PCIe |
| Sum of squares, m = 12 | `r3d52b7ab43d1` | query m = 12 (`r0fceea0627c8`) with sum-of-squares coefficients | Mosaic, H100 PCIe |
| Sum of squares, m = 16 | `rdfc61c988309` | the m = 12 run at degree 16 | blocked strips, H100 PCIe |
| Hermite start, m = 8 | `r6adefb42f443` | query m = 8 with `b_r` starting at the fixed Hermite coefficients | Mosaic, B300 then H100 PCIe |
| Controls | `rca97cc9db8f2` | query coefficients, m = 4 | Mosaic, H100 SXM |
| | `r25fffc16c2f3`, `r0fceea0627c8` | query coefficients, m = 8 and 12 | Mosaic, B300 |
| | `r816ab0fbe119` | fixed Taylor, m = 4 | Mosaic, B300 |
| | `ra7ee8a4b2544`, `rb2b70dae4d21` | fixed Taylor, m = 8 and 12 | Mosaic, H100 PCIe; B300 then H100 PCIe |
| | `ra040de24096c` | fixed Hermite, m = 8 | Mosaic, H100 PCIe then SXM |
| | `r6a76e7fbde45` | exp | fused, B300 |

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
| 4 | 14.642 (14.648) | 14.709 | 14.722 | +2.70% | −0.45% | −0.54% |
| 8 | 14.196 (14.200) | 14.323 | 14.218 | −0.43% | −0.89% | −0.16% |
| 12 | **14.165** (14.168) | 14.226 | 14.309 | **−0.65%** | −0.43% | −1.00% |
| 16 | 14.168 (14.172) | not run | not run | −0.63% | — | — |

- **Sum of squares beats exp at m = 8, 12 and 16, not at m = 4; degree helps from 4 to 12.**
  m = 12 and m = 16 are below exp at every evaluation from update 1,000 on (by 0.30–1.35% and
  0.54–1.38%); m = 8 from update 1,500 on (by 0.33–0.48%). m = 4 is above exp at every evaluation
  (by 2.70–3.71% from update 2,000 on), as are its direct form (+3.17%) and fixed Taylor (+3.26%);
  m = 8 is 3.05% below it. From update 4,000 on m = 12 is 0.14–0.29% below m = 8.
- **Degree 16 adds nothing over 12** (+0.02%); it also changes the kernel (blocked strips).
- **It beats the direct form by a margin larger than any recorded rerun gap** (−0.45% at m = 4,
  −0.89% at m = 8, −0.43% at m = 12). At m = 8 it gives up the direct form's early lead (+2.32% vs
  exp at update 500, against +0.96%) and stays ahead of it from update 1,500 on; at m = 4 it trails
  the direct form through update 3,000 (by up to 10.6%, at update 500) and leads from update 3,500
  on, by 0.07–0.45%.
- **At m = 4 the gain misses a 0.5% bar.** Beating the direct form (14.709) by 0.5% needs 14.635
  or below; 14.642 misses it by 0.05 percentage points (0.007 perplexity). It is below fixed Taylor
  m = 4 at every evaluation from update 500 on (by 0.50–3.18%). Against the same bar m = 8 clears
  it (14.251) and m = 12 misses it (14.155).
- **Training is calm.** After update 1,500 each sum-of-squares run clips one update, at update
  6,376, where fixed Taylor m = 12 also spikes (a batch, not the mechanism); largest gradient norms
  2.05, 1.68, 1.50 and 1.18 (m = 4, 8, 12 and 16), against 26.3 for the direct form at m = 4 (eight
  clipped updates) and 3.84 at m = 8.
- **The polynomials change shape, not just scale.** At m = 8 the token-mean polynomial, with its
  overall scale removed (the head RMSNorm cancels it), weights negative scores 25–30% below Taylor;
  orders 6–8 fall 12–26% relative to the lower orders. The per-query variation sits in `q_1`'s three
  lowest coefficients (token CV 9–17%) and mostly changes how deep each query's floor is.
- **Heads can switch off.** After update 1,500 the smallest pre-RMSNorm head-output RMS reaches
  about `5e-8` (m = 4), `4e-7` (m = 8, 12) and `2e-7` (m = 16), where the head RMSNorm's gain sits
  at its `1/sqrt(eps)` cap of 1,000.
  The direct form does this too (exactly 0 in the Hermite-start run); fixed Taylor m = 12 stays
  above 0.01. The comparison rests on these two runs: the direct-form controls (m = 4, 8 and 12),
  fixed Taylor m = 4 and 8 and fixed Hermite predate per-update telemetry. The epsilon (`1e-6`)
  has never been varied.
- **The Hermite start is not the lever.** Starting the query coefficients at the fixed Hermite
  polynomial reaches validation-best 14.371 (terminal 14.374): 0.34% above the Taylor start and
  1.03% above fixed Hermite.

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

## Layerwise polynomial swap

Does a head's polynomial fit its own query better than the one it would generate for another
sequence at the same position? The direct Query diagnostic `r2d8dd18744ea` and SoS diagnostic
`re6a52dd6fc33` use their respective degree-eight checkpoints `r25fffc16c2f3` and `reeefbcc19abf`
at update 7,500. Both evaluate the same packed validation rows at the training context of 2,048.

Each intervention replaces all 14 heads' polynomials in one of 15 layers. Each head receives
the same head's nine effective monomial coefficients from the other sequence at the same position.
The recipient supplies its own queries, keys and values. Subsequent layers generate coefficients
from the resulting hidden states. For SoS, the vector contains the coefficients of the expanded
polynomial `U² + V²`.

Validation rows 0–63 supply one mean coefficient vector per layer and head, averaged over
calibration positions and sequences after expansion. Rows 64–319 form 128 reciprocal pairs from
a seed-42 permutation. Every condition and both models use those pairs. Each run's `metrics.jsonl`
records native, swapped and calibration-mean next-token NLL. A positive `delta_nll` is a loss
increase relative to native on the same sequences. The standard error uses the 128 pair means.
The order of the per-sequence losses and the donor row IDs are in `metadata.json`.

[Launch instructions](../../../README.md#layerwise-polynomial-swap)

## Limits

- One seed per cell; differences below about 0.2% are not resolved.
- GPUs differ within comparisons: the sum-of-squares runs at m = 8 to 16 trained on H100 PCIe,
  their query controls on B300; at m = 4 the run and its query control trained on H100 SXM, fixed
  Taylor m = 4 on B300. Fixed Taylor m = 12 and the Hermite start moved from B300 to H100 PCIe at
  update 1,500 after the B300 node failed.
- The m = 16 point differs in kernel as well as degree and has no direct-form or Taylor partner.

## Open decision

Sum of squares is the first configuration on this recipe whose learned polynomial beats exp
(−0.65% at m = 12) and improves with degree from 4 to 12. Decide whether to confirm m = 12 (and its
direct-form and Taylor partners) with a second seed and promote the topic as its own collection,
or discard it.

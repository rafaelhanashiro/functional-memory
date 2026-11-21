# Softplus and sum-of-squares composition

Opened 2026-11-13. Does composing the softplus-input and query sum-of-squares
(SoS) mechanisms improve Transformer++ validation perplexity, and does allowing
a learned baseline help compared with forcing the negative-score tail to vanish?

The [study methods and launch instructions](../../../../../README.md#softplus-and-sum-of-squares-composition)
define the execution. All four primaries use the 15-layer width-896 shape, query
conditioning, RoPE, coefficient-projection W decay, and the reference token
budget; the two degree-twelve compositions also ran at the 12-layer width-640 shape at
1x, 2x and 4x its budget ([softplus_sos there](../../../scalings/l12_d640/softplus_sos/README.md)).
The [selection](selection.yaml) pins all inputs explicitly.

| Role | Run | Weight, z = softplus(s) | Core degree | Total degree |
| --- | --- | --- | ---: | ---: |
| Learned baseline | `r41a4c74c5f58` | h_t(z) | 8 | 8 |
| Vanishing negative tail | `r841f35a24241` | z h_t(z) | 8 | 9 |
| SoS control | `reeefbcc19abf` | h_t(s) | 8 | 8 |
| Direct softplus control | `ra3b54e2489f8` | p_t(softplus(s)) | — | 8 |
| Degree-12 learned baseline | `rc5aac8f37a2e` | h_t(z) | 12 | 12 |
| Degree-12 vanishing negative tail | `rf637de9b65b1` | z h_t(z) | 12 | 13 |
| Degree-12 SoS control | `r3d52b7ab43d1` | h_t(s) | 12 | 12 |
| Degree-12 direct softplus control | `r9eb845aae5bb` | p_t(softplus(s)) | — | 12 |
| Degree-12 direct polynomial control | `r0fceea0627c8` | p_t(s) | — | 12 |
| Direct polynomial control | `r25fffc16c2f3` | p_t(s) | — | 8 |
| Exponential reference | `r6a76e7fbde45` | exp(s) | — | — |

The two degree-eight primaries have the same 200,150,806 learned parameters and start from the
SoS control's exact parameter arrays. The core starts at T_8, so the initial
kernels are T_8(z) and z T_8(z). The direct-softplus control instead starts at
T_8(z)-1; its constant coefficient remains learnable. These are not identical
function starts. The prefactor variant retains the degree-eight core and
increases the total degree to nine.

The first primary tends to its query-dependent h_t(0) for very negative scores.
The second has a structurally zero constant coefficient and tends to zero. Both
are nonnegative in exact arithmetic on z>0; neither imposes monotonicity. Mosaic
evaluates expanded FP32 coefficients, so small negative values from cancellation
remain possible. W decays at 0.1; generator amplitudes and biases do not decay.

The [open question](#open-question) is whether the composition improves
validation perplexity and whether the baseline or vanishing-tail variant is
preferable. Terminal and validation-best metrics are reported separately.
Each run's `metrics.jsonl` keeps every gradient window and, for the learned
polynomials, expanded coefficient summaries at every evaluation. One seed does
not establish statistical significance or the cause of an improvement. Compare
hardware and resolved kernels from each execution record.

The degree-eight runs completed: h(softplus(s)) reached validation-best 14.3388
and terminal 14.3435; the prefactor variant reached 14.0953 and 14.0992. The latter
is 1.14% below exp in validation-best PPL (14.2572). Best values are at update
7,500 and terminal values at update 7,570. These are single-seed comparisons.

All four runs completed 7,570 updates and selected update 7,500 by validation
NLL. Their retained artifacts are on Kennel. The degree-twelve pair has
200,905,126 parameters and starts with h = T_12, matching the raw-score
degree-twelve SoS control's initial arrays.

| SoS core degree | Weight, z = softplus(s) | Best validation PPL (7,500) | Terminal PPL (7,570) |
| ---: | --- | ---: | ---: |
| 8 | h_t(z) | 14.3388 | 14.3435 |
| 8 | z h_t(z) | 14.0953 | 14.0992 |
| 12 | h_t(z) | 14.3270 | 14.3315 |
| 12 | z h_t(z) | 14.0934 | 14.0972 |

The prefactor variant improves on the learned-baseline composition at both core
degrees. Its degree-twelve best PPL is 0.014% below its degree-eight partner,
a small difference in this single-seed comparison. It is below the raw-score
SoS degree-twelve control (14.1649), direct softplus (14.2223), direct query
(14.2256), and shared exp control (14.2572). These comparisons do not identify
which mechanism caused the improvement.

## Prefactor factor-generator histories

The [combined parameter figure](head_parameters_combined.png)
([PDF](head_parameters_combined.pdf)) shows raw learned a, b, and W-vector L2
norms for the two original-input prefactor runs, `r841f35a24241` and
`rf637de9b65b1`. It uses layer 8, head 1, matching the existing SoS diagnostic.
The explicit roles, fields, and head are recorded in `selection.yaml`.

Each section plots the U/V factor powers for core m=8 or m=12; the corresponding
weights are `z [U_t(z)^2 + V_t(z)^2]`, with `z=softplus(s)`, giving total degrees
9 and 13. These are the original coefficient inputs without L2 normalization.
No factorial or fixed Taylor-factor scaling is applied to the displayed generator
parameters. W norms span 896 input channels. The 28 saved snapshots cover updates
100–7,570; step-zero a=0 and b=1 diamonds are analytic, and no step-zero W norm is
imputed. All panels within a section's row share a linear y-axis. The figure
describes this fixed head, not a whole-model parameter average or saturation rate.

Regenerate both formats from the pinned, committed measurements:

```bash
scaffolds/transformer/.venv/bin/python experiments/transformer_pp/results/pretraining/parametrizations/sos/softplus_sos/plot_head_parameters.py
```

## Score prefactors on query SoS

Added 2026-11-20. Is `z = softplus(s)` the best factor in front of `h_t`? Seven runs weight
`W = f(s) h_t(g(s))`, with `f` a parameter-free function of the raw score, `g` the identity or softplus, and a query
SoS core of degree eight with input power zero. Each is `reeefbcc19abf`'s recipe (`h_t(s)`, 14.196) with `f`, `g`
and gradient-event capture, from its exact initial parameters, so `h_t` starts at `T_8`. Validation-best perplexity:

| `f` | `g(s) = s` | `g(s) = z` |
| --- | --- | --- |
| softplus | `r41662adb22fa`, 14.097 | `r841f35a24241`, 14.095 (the control) |
| exp | `r6525664e6696`, 14.089 | `raeaa47865867`, **14.067** |
| sigmoid | `r8f56a5b7e525`, 14.148 | `r5d578a29f080`, 14.271 |
| swish, `s sigmoid(s)` | `re6846615ffe2`, 16.494 | `r50c893fe159c`, 16.525 |

- **No prefactor improves on the control beyond one seed's noise.** `e^s h_t(z)` is 0.20% below it (1.33% below
  exp, `r6a76e7fbde45`, 14.257); `e^s h_t(s)` and `softplus(s) h_t(s)` are level with it (−0.04% and +0.01%).
- **With a growing positive prefactor the polynomial's input stops mattering**: the softplus and exp rows span
  0.21%, where without a prefactor `h_t(s)` leads `h_t(z)` (`r41a4c74c5f58`, 14.339) by 1.00%. The prefactor is
  worth 0.69% on `h_t(s)` and 1.70% on `h_t(z)`.
- **A bounded prefactor is worse**: sigmoid gives 14.148 on the raw score (+0.37% against the control) and 14.271
  on `z` (+1.24%), level with exp.
- **Swish fails**: 16.494 on the raw score and 16.525 on `z`, 17% behind the control and 16% behind exp. Its
  weights are signed and it trains unstably: gradient norms reach `1e5` and `1e4` (the control's largest is 13),
  with an update clipped in 26 and 21 of the 28 evaluation windows (the control: 6). Swish alone
  (`r8a9fd53dacfd`, 18.393 in
  [pointwise_attention](../../../alternative_attention/fixed_kernels/pointwise_attention/README.md#swish)) is
  worse still.
- The learned core is worth 0.25–0.27% over the fixed kernels its starts approximate: `softplus(s) h_t(s)` against
  softplus(s) exp(s) (`rbd24b948a240`, 14.132) and `sigmoid(s) h_t(s)` against sigmoid(s) exp(s)
  (`rd7c9f610e9dc`, 14.186) of
  [pointwise_attention](../../../alternative_attention/fixed_kernels/pointwise_attention/README.md).
- The control computes `softplus(s) h_t(softplus(s))` through the input power (degree nine with a structural
  zero); the new runs use the general prefactor on the Mosaic transformed kernels (`query_times_<f>`,
  `query_softplus_times_<f>`, on one B200 or H100). The two routes are equal in exact arithmetic and differ by FP32
  rounding, which the GPU qualification compares.
- Limits: one seed each. The exp prefactor stayed far from FP32's range: its scores reached 11 and 14 at the final
  validation (`e^s h_t(s)` passes the FP32 maximum at `s = 65.7` at `h_t = T_8`), and every run's attention row sums
  stayed below `3e11`, where head RMSNorm's plain arithmetic is exact
  ([methods and launch](../../../../../README.md#score-prefactors-swish-and-narrow-keys)).

## Open question

Every run is complete: the four compositions (core degrees eight and twelve), their controls and the seven
prefactor runs. Whether to extend the comparison across seeds or recipes, or delete it with its runs and their
artifacts, is the user's decision. On one seed each, the vanishing-tail prefactor improves on the learned-baseline composition by 1.70%
at core degree eight and 1.63% at core degree twelve, where it leads exp by 1.15%; at the 12-layer width-640
shape that lead narrows from 0.88% at 1x to 0.27% at 4x
([softplus_sos there](../../../scalings/l12_d640/softplus_sos/README.md)). Softplus and exp prefactors on either
input are within 0.21% of one another, so the runs do not identify which mechanism causes the improvement beyond
a growing positive factor of the raw score.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/parametrizations/sos/softplus_sos
```

The same polynomials with structurally zero constants, on softplus inputs and on raw scores:
[zero_constant](../../../other/zero_constant/README.md).

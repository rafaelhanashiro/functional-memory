# Softplus and sum-of-squares composition

Opened 2026-11-13. Does composing the softplus-input and query sum-of-squares
(SoS) mechanisms improve Transformer++ validation perplexity, and does allowing
a learned baseline help compared with forcing the negative-score tail to vanish?

The [study methods and launch instructions](../../../README.md#softplus-and-sum-of-squares-composition)
define the execution. All four primaries use the 15-layer width-896 shape, query
conditioning, RoPE, coefficient-projection W decay, and the reference token
budget. The [selection](selection.yaml) pins all inputs explicitly; none enters
a canonical comparison.

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

The open decision is whether the composition improves validation perplexity and
whether the baseline or vanishing-tail variant is preferable. Report terminal
and validation-best metrics separately, with all-update gradient maxima,
clipping frequency, and actual expanded coefficient histories. One seed does
not establish statistical significance or the cause of an improvement. Compare
hardware and resolved kernels from each execution record when results arrive.

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
which mechanism caused the improvement. The open decision remains the value
of the composition and its vanishing negative tail across seeds and recipes.

## The 12-layer width-640 shape at 1x, 2x and 4x

Planned; no results. Six runs train the two degree-twelve compositions at the 12-layer width-640 shape with
1, 2 and 4 times its 1x budget: the prefactor composition z h_t(z) of `rf637de9b65b1` and h_t(z) without it,
of `rc5aac8f37a2e`
([methods and launch](../../../README.md#the-12-layer-width-640-shape-at-1x-2x-and-4x)). Each is compared
with the other composition and the runs at its shape and budget:

| Budget (updates) | z h_t(z) | h_t(z) | exp | Polynomial comparisons |
| --- | --- | --- | --- | --- |
| 1x (7,551) | `r3b9c8cc6fab8` | `r5a18841b21b5` | `r3e2fa2532c23` | degree-eight query `r95a5691d293f`, degree-eight softplus input `r1d342ca6e58c` |
| 2x (15,103) | `re7593f1d0a4c` | `r83fa09df3931` | `rd920368a922e` | degree-twelve raw-score SoS `r08be7093202a`, degree-twelve softplus input `r53d5bbcac16f` |
| 4x (30,205) | `r4589172dcd8b` | `r0623d3020268` | `r333444bd4304` | degree-twelve raw-score SoS `r1f935f18fe21`, degree-twelve softplus input `rb74e8b994b74` |

No raw-score SoS or softplus-input run of degree twelve exists at 1x at this shape, so the 1x
polynomial comparisons differ from the run in degree as well. Report validation-best and terminal PPL
separately; each comparison is single-seed.

Refresh input pins after collecting run measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/softplus_sos
```

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
scaffolds/transformer/.venv/bin/python experiments/transformer_pp/results/exploratory/softplus_sos/plot_head_parameters.py
```

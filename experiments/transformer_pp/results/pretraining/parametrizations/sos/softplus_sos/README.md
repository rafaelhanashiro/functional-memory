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

## Open question

Every run is complete: the four compositions (core degrees eight and twelve) with their controls. Whether to
extend the comparison across seeds or recipes, or delete it with its runs and their artifacts, is the user's
decision. On one seed each, the vanishing-tail prefactor improves on the learned-baseline composition by 1.70%
at core degree eight and 1.63% at core degree twelve, where it leads exp by 1.15%; at the 12-layer width-640
shape that lead narrows from 0.88% at 1x to 0.27% at 4x
([softplus_sos there](../../../scalings/l12_d640/softplus_sos/README.md)). The runs do not identify which
mechanism causes the improvement.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/parametrizations/sos/softplus_sos
```

The same polynomials with structurally zero constants, on softplus inputs and on raw scores:
[zero_constant](../../../other/zero_constant/README.md).

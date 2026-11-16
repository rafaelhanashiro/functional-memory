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

The degree-twelve pair is allocated for fresh training. Both have 200,905,126
parameters and start with h = T_12, matching the raw-score degree-twelve SoS
control's initial arrays. Compare each with its degree-eight partner as well as
the degree-twelve raw-score SoS (best 14.1649), direct softplus (14.2223), direct
query (14.2256), and shared exp controls. Current execution status lives in the
run records and generated study index. Full acceptance gates production training;
queued jobs are not training results.

Refresh input pins after collecting run measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/softplus_sos
```

# Static versus query-conditioned SoS

Opened 2026-11-17.

**Question.** Do the SoS score function and its softplus prefactor improve Transformer++
without input-dependent polynomial coefficients? Six static runs compare directly
with their completed, degree-matched query counterparts and the shared exp reference.
The [study methods](../../../README.md#static-sum-of-squares-comparison) define the
recipe and launch commands. Home cluster: Kennel.

| Static run | Attention weight | Core degree | Total degree | Query control |
| --- | --- | ---: | ---: | --- |
| `r0e32dd2bd3f7` | h(s) | 8 | 8 | `reeefbcc19abf` |
| `rdc8b04b223af` | h(s) | 12 | 12 | `r3d52b7ab43d1` |
| `r54f771e54722` | h(softplus(s)) | 8 | 8 | `r41a4c74c5f58` |
| `r1b9e2e69f7f0` | h(softplus(s)) | 12 | 12 | `rc5aac8f37a2e` |
| `ra3bffee9a528` | softplus(s) h(softplus(s)) | 8 | 9 | `r841f35a24241` |
| `r28a991faed4f` | softplus(s) h(softplus(s)) | 12 | 13 | `rf637de9b65b1` |

All use the 15-layer width-896 shape, RoPE, canonical QK RMSNorm with head scales,
attention-output head RMSNorm, seed 42, 2,048-token context, global batch 256,
7,570 updates, and the query controls' data, initialization and AdamW recipe.
The independent exp reference is `r6a76e7fbde45`.

For each layer/head, h(z)=A(z)^2+B(z)^2 with degrees m/2 and m/2-1.
The static model stores m+1 dimensionless parameters b, initialized to one,
and maps them to factors F_j=f_j(1+b_j)/2, with fixed Taylor factors f.
This is the query generator's bias branch with W and a removed, preserving
its factor scaling and initial kernel h=T_m. No coefficient input normalization,
forget gate, score bias or coefficient LR multiplier is introduced. Factors are
exempt from decay; the backbone retains its usual exemptions and WD=0.1.
The prefactor adds no parameter. Parameter counts are 198,455,476 (core 8) and
198,456,316 (core 12). Removing input dependence also reduces parameter count,
so the comparison is not parameter-matched.

Mosaic evaluates FP32-expanded coefficients, broadcast across queries; its backward
sums gradients into the shared factor coordinates. Algebraic nonnegativity does not
exclude small negative rounding near roots. The prefactor fixes the constant term
to zero and makes the negative-score tail vanish. Each allocation must pass CPU and
GPU forward/gradient checks, actual model initialization and decay-mask checks,
a full-shape pilot, and fresh-process checkpoint resume before training.

**Status.** Six recipes allocated; acceptance and training are tracked in their run records.

**Open decision.** Does input dependence improve each of these three score functions,
and does the prefactor's benefit persist for static factors? Report validation-best
and terminal PPL separately against each query partner and exp. These are single-seed
comparisons; the prefactor changes the total polynomial degree and initial function.
No run enters a canonical selection.

Refresh the explicit input pins after collecting measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/static_sos
```

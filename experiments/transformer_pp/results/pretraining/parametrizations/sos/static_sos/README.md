# Static versus query-conditioned SoS

Opened 2026-11-17.

**Question.** Do the SoS score function and its softplus prefactor improve Transformer++
without input-dependent polynomial coefficients? Six static core-eight/twelve runs compare directly
with their completed, degree-matched query counterparts and the shared exp reference.
The [study methods](../../../../../README.md#static-sum-of-squares-comparison) define the
recipe and launch commands. Home cluster: Kennel.

| Static run | Attention weight | Core degree | Total degree | Query control |
| --- | --- | ---: | ---: | --- |
| `r0e32dd2bd3f7` | h(s) | 8 | 8 | `reeefbcc19abf` |
| `rdc8b04b223af` | h(s) | 12 | 12 | `r3d52b7ab43d1` |
| `r54f771e54722` | h(softplus(s)) | 8 | 8 | `r41a4c74c5f58` |
| `r1b9e2e69f7f0` | h(softplus(s)) | 12 | 12 | `rc5aac8f37a2e` |
| `ra3bffee9a528` | softplus(s) h(softplus(s)) | 8 | 9 | `r841f35a24241` |
| `r28a991faed4f` | softplus(s) h(softplus(s)) | 12 | 13 | `rf637de9b65b1` |

Two core-degree-four extensions use the same static recipe with only degree and
run identity changed. No matched query-conditioned softplus SoS run exists at
degree four.

| Static run | Attention weight | Core degree | Total degree | Degree-eight counterpart |
| --- | --- | ---: | ---: | --- |
| `r6d4217061453` | h(softplus(s)) | 4 | 4 | `r54f771e54722` |
| `r91e0c1bb8458` | softplus(s) h(softplus(s)) | 4 | 5 | `ra3bffee9a528` |

The ordinary static degree-four reference is `r4339a245820e`. This comparison
changes the score transform and coefficient parameterization, not just degree.

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
The prefactor adds no parameter. Parameter counts are 198,454,636 (core 4), 198,455,476 (core 8) and
198,456,316 (core 12). Removing input dependence also reduces parameter count,
so the comparison is not parameter-matched.

Mosaic evaluates FP32-expanded coefficients, broadcast across queries; its backward
sums gradients into the shared factor coordinates. Algebraic nonnegativity does not
exclude small negative rounding near roots. The prefactor fixes the constant term
to zero and makes the negative-score tail vanish. Each allocation must pass CPU and
GPU forward/gradient checks, actual model initialization and decay-mask checks,
a full-shape pilot, and fresh-process checkpoint resume before training.

## Results

Validation perplexity, best with terminal in parentheses. The counterpart is the degree-matched query run, or for
core degree four the static degree-eight run; exp `r6a76e7fbde45` reaches 14.257.

| Static run | Attention weight | Core degree | Best (terminal) | Counterpart, best | vs counterpart | vs exp |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| `r0e32dd2bd3f7` | h(s) | 8 | 14.304 (14.306) | `reeefbcc19abf` 14.196 | +0.77% | +0.33% |
| `rdc8b04b223af` | h(s) | 12 | 14.167 (14.172) | `r3d52b7ab43d1` 14.165 | +0.02% | −0.63% |
| `r54f771e54722` | h(softplus(s)) | 8 | 14.354 (14.360) | `r41a4c74c5f58` 14.339 | +0.11% | +0.68% |
| `r1b9e2e69f7f0` | h(softplus(s)) | 12 | 14.367 (14.367) | `rc5aac8f37a2e` 14.327 | +0.28% | +0.77% |
| `ra3bffee9a528` | softplus(s) h(softplus(s)) | 8 | 14.148 (14.152) | `r841f35a24241` 14.095 | +0.37% | −0.77% |
| `r28a991faed4f` | softplus(s) h(softplus(s)) | 12 | 14.121 (14.125) | `rf637de9b65b1` 14.093 | +0.20% | −0.95% |
| `r6d4217061453` | h(softplus(s)) | 4 | 14.757 (14.760) | `r54f771e54722` 14.354 | +2.81% | +3.51% |
| `r91e0c1bb8458` | softplus(s) h(softplus(s)) | 4 | 14.297 (14.301) | `ra3bffee9a528` 14.148 | +1.06% | +0.28% |

## Findings

- **Static SoS nearly matches query SoS.** It ties at core degree twelve with raw scores (14.167 against 14.165)
  and trails by 0.11–0.77% everywhere else: input-dependent coefficients add little to the SoS kernel.
- **The softplus prefactor helps the static kernel too:** 14.148 and 14.121 against 14.354 and 14.367 for the
  softplus input alone at core degrees eight and twelve, 1.4–1.7% lower. With it, static SoS is 0.77–0.95% below exp.
- **Core degree four is clearly worse:** 2.81% (softplus input) and 1.06% (prefactor) above core degree eight.

## Limits

- One seed per run. Removing input dependence also removes parameters, so the comparison is not parameter-matched.
- Hardware: the six degree-eight and degree-twelve runs trained on an H100, except `r1b9e2e69f7f0`, which trained
  its first 300 updates on an H100 and the rest on a B200; the two degree-four runs trained on a B200.

## Open question

Keep the topic, or delete it with its runs and their artifacts.

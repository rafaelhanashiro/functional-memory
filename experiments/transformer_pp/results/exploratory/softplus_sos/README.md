# Softplus and sum-of-squares composition

Opened 2026-11-13. Does composing the softplus-input and query sum-of-squares
(SoS) mechanisms improve Transformer++ validation perplexity, and does allowing
a learned baseline help compared with forcing the negative-score tail to vanish?

The [study methods and launch instructions](../../../README.md#softplus-and-sum-of-squares-composition)
define the execution. All four primaries use the 15-layer width-896 shape, query
conditioning, RoPE, coefficient-projection W decay, and the reference token
budget; six degree-twelve runs at the 12-layer width-640 shape follow
[below](#the-12-layer-width-640-shape-at-1x-2x-and-4x). The [selection](selection.yaml)
pins all inputs explicitly; none enters a canonical comparison.

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

The [open decision](#open-decision) turns on whether the composition improves
validation perplexity and whether the baseline or vanishing-tail variant is
preferable. Terminal and validation-best metrics are reported separately; the
12-layer width-640 results add gradient maxima, clipping frequency and spikes.
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

## The 12-layer width-640 shape at 1x, 2x and 4x

All six runs are complete. They train the two degree-twelve compositions at the 12-layer width-640 shape with
1, 2 and 4 times its 1x budget: the prefactor composition z h_t(z) of `rf637de9b65b1` and h_t(z) without it,
of `rc5aac8f37a2e`
([methods and launch](../../../README.md#the-12-layer-width-640-shape-at-1x-2x-and-4x)). Each is compared
with the other composition and the runs at its shape and budget; the [selection](selection.yaml) names the
fifteen runs' roles. Batch 128 × 2,048 tokens, seed 42, one GPU each, home cluster Kennel; every evaluation
uses the same 38 validation batches.

| Budget (updates) | z h_t(z) | h_t(z) | exp | Polynomial comparisons |
| --- | --- | --- | --- | --- |
| 1x (7,551) | `r3b9c8cc6fab8` | `r5a18841b21b5` | `r3e2fa2532c23` | degree-eight query `r95a5691d293f`, degree-eight softplus input `r1d342ca6e58c` |
| 2x (15,103) | `re7593f1d0a4c` | `r83fa09df3931` | `rd920368a922e` | degree-twelve raw-score SoS `r08be7093202a`, degree-twelve softplus input `r53d5bbcac16f` |
| 4x (30,205) | `r4589172dcd8b` | `r0623d3020268` | `r333444bd4304` | degree-twelve raw-score SoS `r1f935f18fe21`, degree-twelve softplus input `rb74e8b994b74` |

Softplus input is the direct form p_t(softplus(s)) of the controls above, raw-score SoS is h_t(s). No
raw-score SoS or softplus-input run of degree twelve exists at 1x at this shape, so the 1x polynomial
comparisons differ from the runs in degree as well.

### Results

Validation perplexity from each run's `metrics.jsonl`; `best_validation.json` and `completion.json` agree. The
1x and 4x runs are validation-best at their last update, the 2x runs at update 15,000 (3,932,160,000 tokens,
the reference-shape runs' count at their validation-best update, 7,500). "vs exp" compares validation-best
values with exp at the same budget; terminal values give the same gaps within 0.02 points.

1x, 7,551 updates (1.98B tokens); z h_t(z) is 0.72% below h_t(z):

| Run | Weight | Validation-best PPL (update) | Terminal PPL (update) | vs exp |
| --- | --- | ---: | ---: | ---: |
| `r3b9c8cc6fab8` | z h_t(z), core degree 12 | 17.3967 (7,551) | 17.3967 (7,551) | −0.88% |
| `r5a18841b21b5` | h_t(z), degree 12 | 17.5235 (7,551) | 17.5235 (7,551) | −0.15% |
| `r1d342ca6e58c` | p_t(z), degree 8 (softplus input) | 17.4878 (7,551) | 17.4878 (7,551) | −0.36% |
| `r95a5691d293f` | p_t(s), degree 8 (query) | 17.5922 (7,551) | 17.5922 (7,551) | +0.24% |
| `r3e2fa2532c23` | exp(s) | 17.5504 (7,551) | 17.5504 (7,551) | – |

2x, 15,103 updates (3.96B tokens); z h_t(z) is 0.35% below h_t(z):

| Run | Weight | Validation-best PPL (update) | Terminal PPL (update) | vs exp |
| --- | --- | ---: | ---: | ---: |
| `re7593f1d0a4c` | z h_t(z), core degree 12 | 15.6715 (15,000) | 15.6792 (15,103) | −0.63% |
| `r83fa09df3931` | h_t(z), degree 12 | 15.7260 (15,000) | 15.7323 (15,103) | −0.29% |
| `r08be7093202a` | h_t(s), degree 12 (raw-score SoS) | 15.7154 (15,000) | 15.7208 (15,103) | −0.36% |
| `r53d5bbcac16f` | p_t(z), degree 12 (softplus input) | 15.7242 (15,000) | 15.7307 (15,103) | −0.30% |
| `rd920368a922e` | exp(s) | 15.7716 (15,000) | 15.7772 (15,103) | – |

4x, 30,205 updates (7.92B tokens); z h_t(z) is 0.20% below h_t(z):

| Run | Weight | Validation-best PPL (update) | Terminal PPL (update) | vs exp |
| --- | --- | ---: | ---: | ---: |
| `r4589172dcd8b` | z h_t(z), core degree 12 | 14.5649 (30,205) | 14.5649 (30,205) | −0.27% |
| `r0623d3020268` | h_t(z), degree 12 | 14.5937 (30,205) | 14.5937 (30,205) | −0.08% |
| `r1f935f18fe21` | h_t(s), degree 12 (raw-score SoS) | 14.5645 (30,205) | 14.5645 (30,205) | −0.28% |
| `rb74e8b994b74` | p_t(z), degree 12 (softplus input) | 14.5815 (30,205) | 14.5815 (30,205) | −0.16% |
| `r333444bd4304` | exp(s) | 14.6050 (30,205) | 14.6050 (30,205) | – |

The same compositions at the reference shape, from the first table (7,570 updates of 256 × 2,048 tokens);
z h_t(z) is 1.63% below h_t(z):

| Run | Weight | Validation-best PPL (update) | Terminal PPL (update) | vs exp |
| --- | --- | ---: | ---: | ---: |
| `rf637de9b65b1` | z h_t(z), core degree 12 | 14.0934 (7,500) | 14.0972 (7,570) | −1.15% |
| `rc5aac8f37a2e` | h_t(z), degree 12 | 14.3270 (7,500) | 14.3315 (7,570) | +0.49% |
| `r6a76e7fbde45` | exp(s) | 14.2572 (7,500) | 14.2617 (7,570) | – |

Gap in percent at the same fraction of training, as 1x / 2x / 4x: the 1x run at update s, the 2x run at 2s and
the 4x run at 4s, against exp at its budget or, in the last row, z h_t(z) against h_t(z) at its budget. The last
column compares the last updates:

| Gap | 1,500 / 3,000 / 6,000 | 3,000 / 6,000 / 12,000 | 4,500 / 9,000 / 18,000 | 6,000 / 12,000 / 24,000 | 7,551 / 15,103 / 30,205 |
| --- | ---: | ---: | ---: | ---: | ---: |
| z h_t(z) vs exp | −0.99 / −0.76 / −0.31 | −0.87 / −0.72 / −0.23 | −0.95 / −0.64 / −0.28 | −0.85 / −0.64 / −0.29 | −0.88 / −0.62 / −0.27 |
| h_t(z) vs exp | +0.57 / −0.04 / +0.38 | −0.05 / −0.20 / +0.26 | −0.18 / −0.29 / +0.04 | −0.16 / −0.26 / −0.06 | −0.15 / −0.28 / −0.08 |
| z h_t(z) vs h_t(z) | −1.55 / −0.73 / −0.69 | −0.82 / −0.52 / −0.48 | −0.77 / −0.36 / −0.32 | −0.69 / −0.38 / −0.23 | −0.72 / −0.34 / −0.20 |

Gradient statistics from each run's `metrics.jsonl`: global gradient norms before clipping at 1.0, and spikes,
norms above five times the median of the preceding 100 updates. "After warmup" counts from the first evaluation
after warmup, update 500, 1,000 or 2,000:

| Run | Weight, budget | Largest norm after warmup (update) | Clipped updates | Clipped after warmup | Spikes (updates) |
| --- | --- | ---: | ---: | ---: | --- |
| `r3b9c8cc6fab8` | z h_t(z), 1x | 1.72 (7,461) | 4.99% | 14 (0.20%) | 2 (6,823, 7,461) |
| `re7593f1d0a4c` | z h_t(z), 2x | 3.21 (12,752) | 4.95% | 4 (0.03%) | 4 (6,823, 7,461, 12,752, 14,352) |
| `r4589172dcd8b` | z h_t(z), 4x | 3.98 (12,752) | 4.19% | 6 (0.02%) | 6 (6,823, 7,461, 12,352, 12,752, 14,352, 23,384) |
| `r5a18841b21b5` | h_t(z), 1x | 1.82 (7,461) | 4.44% | 13 (0.18%) | 1 (7,461) |
| `r83fa09df3931` | h_t(z), 2x | 2.68 (12,752) | 5.10% | 3 (0.02%) | 3 (7,461, 12,752, 14,352) |
| `r0623d3020268` | h_t(z), 4x | 3.43 (12,752) | 4.28% | 4 (0.01%) | 5 (6,823, 7,461, 12,752, 14,352, 23,384) |

Over all updates the largest norm falls in the first evaluation window, in warmup: 10.68 with the prefactor at
every budget, 11.02–11.03 without it. Each run's largest norm after warmup is a spike; the prefactor runs have
one more spike at each budget, and every spike but one (update 12,352) falls at an update where another run
also spikes.

### Findings

- **z h_t(z) is the lowest of the study's runs at this shape without a forget gate at 1x and 2x, and ties for
  lowest at 4x.** Of 12, 14 and 9 such runs at 1x, 2x and 4x (`runs.csv`), it is 0.52% below the next at 1x
  (degree-eight softplus input) and 0.28% below the next at 2x (raw-score SoS). At 4x raw-score SoS is 0.0003
  lower (14.5645 against 14.5649, 0.002%). The forget-gate runs at 1x and 2x are lower still
  ([model_scale](../../model_scale/README.md), [matched_tokens](../matched_tokens/README.md)); none trained at
  4x.
- **Its lead over exp narrows with the budget**: 0.88% at 1x, 0.63% at 2x and 0.27% at 4x, against 1.15% at
  the reference shape at the 2x runs' validation-best token count. It narrows with the budget at each fifth of
  training as well (table above), where it varies within a budget by at most 0.14 points; over every evaluation
  from a fifth of training on it spans 0.77–1.49% at 1x, 0.61–0.91% at 2x and 0.23–0.37% at 4x. At updates
  4,000 and 6,000, the same tokens at every budget, it is 0.21% and 0.31% at 4x against 0.77% and 0.85% at 1x
  and 0.91% and 0.72% at 2x, so the narrowing does not come from the tokens seen alone.
- **The prefactor's gain over h_t(z) narrows too**: 0.72%, 0.35% and 0.20% at 1x, 2x and 4x, against 1.63% at
  the reference shape. Within each budget it also narrows over training, from 1.55%, 0.73% and 0.69% at a
  fifth of training.
- **h_t(z) ends within 0.3% of exp at every budget**: validation-best 0.15%, 0.29% and 0.08% below it at 1x, 2x
  and 4x. At a fifth of training it is 0.57% above exp at 1x and 0.38% at 4x (table above). At the reference
  shape it is 0.49% above exp.
- **At 4x z h_t(z) ties raw-score SoS** (+0.002%), which it led by 0.28% at 2x. Its lead over degree-twelve
  softplus input narrows from 0.34% at 2x to 0.11% at 4x.
- **Exp gains more from each doubling than z h_t(z)**: from 1x to 2x and from 2x to 4x, exp's validation-best
  perplexity falls 10.14% and 7.40%, z h_t(z)'s 9.92% and 7.06%, and h_t(z)'s 10.26% and 7.20%. From 2x to 4x
  raw-score SoS falls 7.32% and softplus input 7.27%.

### Limits

- One seed (42) and data order, shared by every run at this shape. In the study's
  [noise check](../../../README.md#noise) a second seed and data order moved single runs by 0.6–0.7% and a gap
  between two variants by 0.1 percentage point; the 4x gaps, 0.28% or less, are of that order.
- GPUs of the six runs, from each record's attempts: every update trained on `dogo`'s H100 SXM, so every
  validation-best checkpoint did. Attempts granted `shiba`'s B200 (1x z h_t(z), 2x and 4x h_t(z)) or `dogo`
  (2x z h_t(z)) were preempted during acceptance, before any update, and requeued; the 4x h_t(z) run was
  preempted again at update 215, before its first checkpoint, and restarted from update 0. No run resumed from
  a checkpoint.
- GPUs of the comparison runs differ: exp trained on `akita`'s B300 at 1x, on `mastiff`'s H100 PCIe at 2x and
  on `dogo` at 4x; the 2x raw-score SoS and softplus-input runs on `mastiff` until updates 5,500 and 8,500,
  then on `dogo`; the 4x raw-score SoS run on `akita`'s B300 until update 1,000, then on `dogo`; the 4x
  softplus-input run on `dogo` throughout; the 1x query run on `akita`'s B300 and the 1x softplus-input run on
  `mastiff`. Only the 4x exp and softplus-input runs thus trained entirely on `dogo`, like the six runs; every
  other comparison's validation-best checkpoint was trained wholly or partly on another GPU. At the reference
  shape z h_t(z) trained on `shiba`'s B200, h_t(z) on `dogo` and exp on `akita`'s B300.
- Kernels, from each run's measurements: the six runs and degree-twelve softplus input on the Mosaic transformed
  query-polynomial kernels, raw-score SoS and the 1x query run on the Mosaic query-polynomial kernels, the 1x
  softplus-input run on the blocked backend's causal strips, exp on fused Triton.

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

## Open decision

Every run of the topic is complete: the four compositions at the reference shape (core degrees eight and
twelve) with their controls, and the six degree-twelve runs at the 12-layer width-640 shape. Whether to promote
the topic to its own canonical collection, discard it with its runs and their artifacts, or extend it across
seeds or recipes is the user's decision. On one seed each, the vanishing-tail prefactor improves on the
learned-baseline composition at both core degrees at the reference shape and at every budget at the 12-layer
width-640 shape, by 0.20–1.70%. At degree twelve its lead over exp, 1.15% at the reference shape, narrows from
0.88% at 1x to 0.27% at 4x at the 12-layer width-640 shape, where at 4x it ties raw-score SoS. The runs do not
identify which mechanism causes the improvement.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/softplus_sos
```

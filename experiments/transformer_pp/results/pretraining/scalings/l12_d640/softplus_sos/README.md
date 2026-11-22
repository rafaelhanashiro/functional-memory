# Softplus and sum-of-squares composition at 1x, 2x and 4x

Opened 2026-11-16. How do the two degree-twelve compositions of softplus inputs with query sum of squares
(SoS), h_t(z) and its prefactor form z h_t(z), with z = softplus(s) and h_t the query SoS polynomial, compare
with exp and the other polynomials at the 12-layer width-640 shape, at 1, 2 and 4 times its 1x budget? The
compositions' definitions, controls and reference-shape results are in
[softplus_sos](../../../parametrizations/sos/softplus_sos/README.md).

## Runs

All six runs are complete. They train the two degree-twelve compositions at the 12-layer width-640 shape with
1, 2 and 4 times its 1x budget: the prefactor composition z h_t(z) of `rf637de9b65b1` and h_t(z) without it,
of `rc5aac8f37a2e`
([methods and launch](../../../../../README.md#the-12-layer-width-640-shape-at-1x-2x-and-4x)). The 4x z h_t(z) run and
the 4x raw-score SoS comparison evaluate head RMSNorm with its max-abs prescale (`attention_head_rms_prescale:
max_abs`), the same function, because their outputs grow past the RMS beyond which the plain FP32 backward is
inexact ([launch](../../../../../README.md#max-abs-alibi-exp-and-12-layer-width-640-4x-runs)). Each is compared
with the other composition and the runs at its shape and budget; the [selection](selection.yaml) names the
fifteen runs' roles and the three reference-shape runs they are contrasted with. Batch 128 × 2,048
tokens, seed 42, one GPU each, home cluster Kennel; every evaluation uses the same 38 validation batches.

| Budget (updates) | z h_t(z) | h_t(z) | exp | Polynomial comparisons |
| --- | --- | --- | --- | --- |
| 1x (7,551) | `r3b9c8cc6fab8` | `r5a18841b21b5` | `r3e2fa2532c23` | degree-eight query `r95a5691d293f`, degree-eight softplus input `r1d342ca6e58c` |
| 2x (15,103) | `re7593f1d0a4c` | `r83fa09df3931` | `rd920368a922e` | degree-twelve raw-score SoS `r08be7093202a`, degree-twelve softplus input `r53d5bbcac16f` |
| 4x (30,205) | `r3474c2559dd7` | `r0623d3020268` | `r333444bd4304` | degree-twelve raw-score SoS `r775f6499974f`, degree-twelve softplus input `rb74e8b994b74` |

Softplus input is the direct query polynomial p_t(softplus(s)), raw-score SoS is h_t(s). No
raw-score SoS or softplus-input run of degree twelve exists at 1x at this shape, so the 1x polynomial
comparisons differ from the runs in degree as well.

## Results

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

4x, 30,205 updates (7.92B tokens); z h_t(z) is 0.29% below h_t(z):

| Run | Weight | Validation-best PPL (update) | Terminal PPL (update) | vs exp |
| --- | --- | ---: | ---: | ---: |
| `r3474c2559dd7` | z h_t(z), core degree 12 | 14.5509 (30,205) | 14.5509 (30,205) | −0.37% |
| `r0623d3020268` | h_t(z), degree 12 | 14.5937 (30,205) | 14.5937 (30,205) | −0.08% |
| `r775f6499974f` | h_t(s), degree 12 (raw-score SoS) | 14.5730 (30,205) | 14.5730 (30,205) | −0.22% |
| `rb74e8b994b74` | p_t(z), degree 12 (softplus input) | 14.5815 (30,205) | 14.5815 (30,205) | −0.16% |
| `r333444bd4304` | exp(s) | 14.6050 (30,205) | 14.6050 (30,205) | – |

The same compositions at the reference shape ([softplus_sos](../../../parametrizations/sos/softplus_sos/README.md);
7,570 updates of 256 × 2,048 tokens);
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
| z h_t(z) vs exp | −0.99 / −0.76 / −0.39 | −0.87 / −0.72 / −0.34 | −0.95 / −0.64 / −0.32 | −0.85 / −0.64 / −0.38 | −0.88 / −0.62 / −0.37 |
| h_t(z) vs exp | +0.57 / −0.04 / +0.38 | −0.05 / −0.20 / +0.26 | −0.18 / −0.29 / +0.04 | −0.16 / −0.26 / −0.06 | −0.15 / −0.28 / −0.08 |
| z h_t(z) vs h_t(z) | −1.55 / −0.73 / −0.77 | −0.82 / −0.52 / −0.60 | −0.77 / −0.36 / −0.36 | −0.69 / −0.38 / −0.32 | −0.72 / −0.34 / −0.29 |

Gradient statistics from each run's `metrics.jsonl`: global gradient norms before clipping at 1.0, and spikes,
norms above five times the median of the preceding 100 updates. "After warmup" counts from the first evaluation
after warmup, update 500, 1,000 or 2,000:

| Run | Weight, budget | Largest norm after warmup (update) | Clipped updates | Clipped after warmup | Spikes (updates) |
| --- | --- | ---: | ---: | ---: | --- |
| `r3b9c8cc6fab8` | z h_t(z), 1x | 1.72 (7,461) | 4.99% | 14 (0.20%) | 2 (6,823, 7,461) |
| `re7593f1d0a4c` | z h_t(z), 2x | 3.21 (12,752) | 4.95% | 4 (0.03%) | 4 (6,823, 7,461, 12,752, 14,352) |
| `r3474c2559dd7` | z h_t(z), 4x | 3.89 (12,752) | 4.17% | 6 (0.02%) | 7 (6,823, 7,461, 12,352, 12,752, 12,753, 14,352, 23,384) |
| `r5a18841b21b5` | h_t(z), 1x | 1.82 (7,461) | 4.44% | 13 (0.18%) | 1 (7,461) |
| `r83fa09df3931` | h_t(z), 2x | 2.68 (12,752) | 5.10% | 3 (0.02%) | 3 (7,461, 12,752, 14,352) |
| `r0623d3020268` | h_t(z), 4x | 3.43 (12,752) | 4.28% | 4 (0.01%) | 5 (6,823, 7,461, 12,752, 14,352, 23,384) |

Over all updates the largest norm falls in the first evaluation window, in warmup: 10.68 with the prefactor at
every budget (10.69 at 4x), 11.02–11.03 without it. Each run's largest norm after warmup is a spike; the prefactor
runs have one more spike than h_t(z) at 1x and 2x and two more at 4x, and every spike but two (updates 12,352 and
12,753, at 4x) falls at an update where another run also spikes.

## Findings

- **z h_t(z) is the lowest of the study's runs at this shape without a forget gate at every budget.** Of 12, 14
  and 9 such runs at 1x, 2x and 4x (`runs.csv`), it is 0.52% below the next at 1x (degree-eight softplus input),
  0.28% at 2x and 0.15% at 4x (raw-score SoS both). The forget-gate runs at 1x and 2x are lower still
  ([shape pairs](../shape_pairs/README.md), [budgets](../budgets/README.md)); none trained at
  4x.
- **Its lead over exp narrows with the budget**: 0.88% at 1x, 0.63% at 2x and 0.37% at 4x, against 1.15% at
  the reference shape at the 2x runs' validation-best token count. It narrows with the budget at each fifth of
  training as well (table above), where it varies within a budget by at most 0.14 points; over every evaluation
  from a fifth of training on it spans 0.77–1.49% at 1x, 0.61–0.91% at 2x and 0.32–0.40% at 4x. At updates
  4,000 and 6,000, the same tokens at every budget, it is 0.37% and 0.39% at 4x against 0.77% and 0.85% at 1x
  and 0.91% and 0.72% at 2x, so the narrowing does not come from the tokens seen alone.
- **The prefactor's gain over h_t(z) narrows too**: 0.72%, 0.35% and 0.29% at 1x, 2x and 4x, against 1.63% at
  the reference shape. Within each budget it also narrows over training, from 1.55%, 0.73% and 0.77% at a
  fifth of training.
- **h_t(z) ends within 0.3% of exp at every budget**: validation-best 0.15%, 0.29% and 0.08% below it at 1x, 2x
  and 4x. At a fifth of training it is 0.57% above exp at 1x and 0.38% at 4x (table above). At the reference
  shape it is 0.49% above exp.
- **Its leads over raw-score SoS and degree-twelve softplus input narrow** from 0.28% and 0.34% at 2x to 0.15%
  and 0.21% at 4x.
- **Exp gains more from each doubling than z h_t(z)**: from 1x to 2x and from 2x to 4x, exp's validation-best
  perplexity falls 10.14% and 7.40%, z h_t(z)'s 9.92% and 7.15%, and h_t(z)'s 10.26% and 7.20%. From 2x to 4x
  raw-score SoS and softplus input fall 7.27%.

## Limits

- One seed (42) and data order, shared by every run at this shape. In the study's
  [noise check](../../../../../README.md#noise) a second seed and data order moved single runs by 0.6–0.7% and a gap
  between two variants by 0.1 percentage point; the 4x gaps, 0.37% or less, are a few times that.
- GPUs of the six runs, from each record's attempts: the 4x z h_t(z) run trained on `shiba`'s B200 throughout,
  the other five on `dogo`'s H100 SXM. Their earlier attempts granted `shiba`'s B200 (1x z h_t(z), 2x and 4x
  h_t(z)) or `dogo` (2x z h_t(z)) were preempted during acceptance, before any update, and requeued; the 4x h_t(z)
  run was preempted again at update 215, before its first checkpoint, and restarted from update 0. No run resumed
  from a checkpoint.
- GPUs of the comparison runs differ: exp trained on `akita`'s B300 at 1x, on `mastiff`'s H100 PCIe at 2x and
  on `dogo` at 4x; the 2x raw-score SoS and softplus-input runs on `mastiff` until updates 5,500 and 8,500,
  then on `dogo`; the 4x raw-score SoS run on `akita`'s B300 throughout; the 4x
  softplus-input run on `dogo` throughout; the 1x query run on `akita`'s B300 and the 1x softplus-input run on
  `mastiff`. Only the 4x exp and softplus-input runs thus trained entirely on `dogo`, like five of the six runs; every
  other comparison's validation-best checkpoint was trained wholly or partly on another GPU. At the reference
  shape z h_t(z) trained on `shiba`'s B200, h_t(z) on `dogo` and exp on `akita`'s B300.
- Kernels, from each run's measurements: the six runs and degree-twelve softplus input on the Mosaic transformed
  query-polynomial kernels, raw-score SoS and the 1x query run on the Mosaic query-polynomial kernels, the 1x
  softplus-input run on the blocked backend's causal strips, exp on fused Triton.

## Open question

Every run is complete. Whether to extend the comparison across seeds or recipes, or delete it with its runs
and their artifacts, is the user's decision. On one seed each, the vanishing-tail prefactor improves on the
learned-baseline composition at every budget, by 0.72%, 0.35% and 0.29% at 1x, 2x and 4x, and its lead over
exp narrows from 0.88% at 1x to 0.37% at 4x, where it leads raw-score SoS by 0.15%. The runs do not identify which
mechanism causes the improvement.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/scalings/l12_d640/softplus_sos
```

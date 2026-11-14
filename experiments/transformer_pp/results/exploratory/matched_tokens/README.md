# Matched tokens

Opened: 2026-11-12. Status: **open**. All fifteen runs are complete; promoting or discarding the topic is the
user's decision ([below](#open-decision)). Each run's status is in its record and the study's `runs.csv`.

Trained on the reference shape's token count, twice its own 1x budget, how do the attention variants compare
at the 12-layer width-640 shape at degrees eight and twelve? How close does each come to its reference-shape
counterpart at the same tokens?

## Runs

Each run is a 12-layer width-640 1x run derived with `derive_recipe.py --budget 2`: 15,103 updates of
128 × 2,048 tokens (3.96B prediction tokens, against the reference shape's 3.97B), warmup 908, validation
every 200 updates through 3,000 and every 1,000 after, seed 42. Where the 1x partner has another degree, only
`attention_degree` changes. One GPU each; home cluster Kennel.

| Run | Label | Variant | `m` | Recipe source | Attention |
| --- | --- | --- | ---: | --- | --- |
| `r3759b2a57a54` | `query-m8-w-decay-l12-d640-2x` | tanh query, RoPE, W decay | 8 | `r95a5691d293f` | Mosaic |
| `r0f6d929e5992` | `query-m12-w-decay-l12-d640-2x` | tanh query, RoPE, W decay | 12 | `r95a5691d293f`, degree 12 | Mosaic |
| `r3cca4c13edab` | `static-m8-l12-d640-2x` | learned static | 8 | `r94e0f2b8cff9`, degree 8 | Mosaic |
| `r84f325593209` | `static-m12-l12-d640-2x` | learned static | 12 | `r94e0f2b8cff9`, degree 12 | Mosaic |
| `rd77db79142e6` | `fixed-m8-l12-d640-2x` | fixed Taylor | 8 | `r2c432bd3a2a4`, degree 8 | Mosaic |
| `rffab7c4eeac5` | `fixed-m12-l12-d640-2x` | fixed Taylor | 12 | `r2c432bd3a2a4`, degree 12 | Mosaic |
| `rc6a7e50d27e1` | `query-m8-w-decay-l12-d640-softplus-input-2x` | softplus input, `c_0` from 0 | 8 | `r3759b2a57a54` with the transform | Mosaic transformed, H100 |
| `r53d5bbcac16f` | `query-m12-w-decay-l12-d640-softplus-input-2x` | softplus input, `c_0` from 0 | 12 | `r0f6d929e5992` with the transform | Mosaic transformed, H100 |
| `rccd2cb5e8bab` | `query-m8-sum-of-squares-w-decay-l12-d640-2x` | sum-of-squares query | 8 | `r3759b2a57a54` with sum of squares | Mosaic |
| `r08be7093202a` | `query-m12-sum-of-squares-w-decay-l12-d640-2x` | sum-of-squares query | 12 | `r0f6d929e5992` with sum of squares | Mosaic |
| `r656e8e652b72` | `query-m8-forget-gate-no-rope-w-decay-l12-d640-2x` | FoX query: cumulative gate, no RoPE, W decay | 8 | `r3d14b598d88f`, degree 8 | causal strips |
| `rbf05f7baab5e` | `query-m12-forget-gate-no-rope-w-decay-l12-d640-2x` | FoX query | 12 | `r3d14b598d88f`, degree 12 | causal strips |
| `rd920368a922e` | `exp-l12-d640-2x` | exp | – | `r3e2fa2532c23` | fused Triton |
| `r3bb5b8bd5c6a` | `softmax-l12-d640-2x` | softmax | – | `rf81c96f76a1d` | cuDNN |
| `re634d82ee479` | `exp-forget-gate-no-rope-l12-d640-2x` | FoX exp | – | `ra5dfd978d3bd` | causal strips |

- The softplus runs are the 2x query runs with the reference-shape softplus run's changes: the transform, `c_0`
  started at 0 and gradient-event capture. Their 1x counterpart `r1d342ca6e58c` trained on the causal strips.
- The sum-of-squares runs are the 2x query runs with the reference-shape sum-of-squares runs' changes.
- The forget-gate runs train on the causal strips, where their 1x partners used the blocked scan: the same
  arithmetic in another summation order.

Recipes, acceptance and launch commands:
[study README](../../../README.md#the-12-layer-width-640-shape-at-the-reference-token-count).

## Comparisons

Each 2x run is compared with:

- exp at 2x, `rd920368a922e`;
- its 1x run at the 12-layer width-640 shape, where one exists at its degree: query 17.592, softplus 17.488,
  exp 17.550, softmax 17.964 and FoX exp 16.753, all at degree eight or without a degree;
- the reference shape's 1x run of the same variant and degree, at almost the same tokens: query 14.323 (`m = 8`)
  and 14.226 (`m = 12`), fixed Taylor 14.218 and 14.309, static 14.294 (`m = 8`), softplus input 14.262 and
  14.222, sum of squares 14.196 and 14.165, exp 14.257, softmax 14.543, and FoX query 13.907 (`m = 8`) and FoX
  exp 13.628.

Validation-best perplexity is reported, with terminal values beside it. Every 2x run is validation-best at update
15,000 (3,932,160,000 tokens), every reference-shape run at update 7,500 (the same tokens), and every 1x run at
this shape at its last update, 7,551 (1,979,449,344 tokens). The [selection](selection.yaml) names each run's
role and pins its scientific files.

## Results

At 2x, against exp at 2x, in order of validation-best perplexity (`metrics.jsonl`; `best_validation.json` and
`completion.json` agree):

| Run | Variant | `m` | Validation-best PPL (update) | vs exp | Terminal PPL (update 15,103) | vs exp |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `r656e8e652b72` | FoX query | 8 | 15.314 (15,000) | −2.90% | 15.325 | −2.87% |
| `re634d82ee479` | FoX exp | – | 15.320 (15,000) | −2.87% | 15.330 | −2.84% |
| `rbf05f7baab5e` | FoX query | 12 | 15.335 (15,000) | −2.77% | 15.341 | −2.76% |
| `r08be7093202a` | sum of squares | 12 | 15.715 (15,000) | −0.36% | 15.721 | −0.36% |
| `r53d5bbcac16f` | softplus input | 12 | 15.724 (15,000) | −0.30% | 15.731 | −0.29% |
| `rccd2cb5e8bab` | sum of squares | 8 | 15.740 (15,000) | −0.20% | 15.748 | −0.19% |
| `rc6a7e50d27e1` | softplus input | 8 | 15.740 (15,000) | −0.20% | 15.748 | −0.19% |
| `r0f6d929e5992` | query | 12 | 15.742 (15,000) | −0.19% | 15.750 | −0.17% |
| `rffab7c4eeac5` | fixed Taylor | 12 | 15.763 (15,000) | −0.06% | 15.769 | −0.05% |
| `rd920368a922e` | exp | – | 15.772 (15,000) | – | 15.777 | – |
| `r84f325593209` | static | 12 | 15.778 (15,000) | +0.04% | 15.783 | +0.04% |
| `r3cca4c13edab` | static | 8 | 15.778 (15,000) | +0.04% | 15.787 | +0.06% |
| `r3759b2a57a54` | query | 8 | 15.780 (15,000) | +0.05% | 15.787 | +0.06% |
| `rd77db79142e6` | fixed Taylor | 8 | 15.783 (15,000) | +0.07% | 15.788 | +0.07% |
| `r3bb5b8bd5c6a` | softmax | – | 16.192 (15,000) | +2.66% | 16.203 | +2.70% |

Against the 1x runs, by validation-best perplexity:

| Run | Variant | `m` | 2x | 1x at 12-layer width-640 | 2x vs 1x | Reference shape, 1x | 2x vs reference shape |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: |
| `r3759b2a57a54` | query | 8 | 15.780 | `r95a5691d293f` 17.592 | −10.30% | `r25fffc16c2f3` 14.323 | +10.17% |
| `r0f6d929e5992` | query | 12 | 15.742 | – | – | `r0fceea0627c8` 14.226 | +10.66% |
| `r3cca4c13edab` | static | 8 | 15.778 | – | – | `r9c69dd7cfc8d` 14.294 | +10.38% |
| `r84f325593209` | static | 12 | 15.778 | – | – | – | – |
| `rd77db79142e6` | fixed Taylor | 8 | 15.783 | – | – | `ra7ee8a4b2544` 14.218 | +11.00% |
| `rffab7c4eeac5` | fixed Taylor | 12 | 15.763 | – | – | `rb2b70dae4d21` 14.309 | +10.16% |
| `rc6a7e50d27e1` | softplus input | 8 | 15.740 | `r1d342ca6e58c` 17.488, causal strips | −9.99% | `ra3b54e2489f8` 14.262 | +10.36% |
| `r53d5bbcac16f` | softplus input | 12 | 15.724 | – | – | `r9eb845aae5bb` 14.222 | +10.56% |
| `rccd2cb5e8bab` | sum of squares | 8 | 15.740 | – | – | `reeefbcc19abf` 14.196 | +10.88% |
| `r08be7093202a` | sum of squares | 12 | 15.715 | – | – | `r3d52b7ab43d1` 14.165 | +10.95% |
| `r656e8e652b72` | FoX query | 8 | 15.314 | – | – | `r9353f554b917` 13.907 | +10.12% |
| `rbf05f7baab5e` | FoX query | 12 | 15.335 | – | – | – | – |
| `rd920368a922e` | exp | – | 15.772 | `r3e2fa2532c23` 17.550 | −10.14% | `r6a76e7fbde45` 14.257 | +10.62% |
| `r3bb5b8bd5c6a` | softmax | – | 16.192 | `rf81c96f76a1d` 17.964 | −9.86% | `r9040f12f3d80` 14.543 | +11.34% |
| `re634d82ee479` | FoX exp | – | 15.320 | `ra5dfd978d3bd` 16.753 | −8.56% | `r183f08346ddc` 13.628 | +12.41% |

–: no run at that shape and degree. At the 12-layer width-640 shape static, fixed Taylor and FoX query have 1x
runs only at `m = 4`, query and softplus only at `m = 8`, sum of squares none; the reference shape has no static
run at `m = 12` and no FoX query run with the shared cumulative gate at `m = 12`.

## Findings

- **The forget gate leads, and FoX query catches FoX exp.** FoX query at `m = 8` (15.314) and `m = 12` (15.335)
  and FoX exp (15.320) are 2.77–2.90% below exp. FoX query at `m = 8` is 0.03% below FoX exp, which it trails by
  2.05% at the reference shape; at `m = 12` it is 0.10% above.
- **The ten RoPE polynomials lie within 0.43% of one another, around exp**: from 15.715 (sum of squares,
  `m = 12`) to 15.783 (fixed Taylor, `m = 8`), −0.36% to +0.07% against exp. The nine of them with
  reference-shape runs span 1.11% there (−0.65% to +0.46% against exp).
- **Softplus input and sum of squares lead query by 0.25% at `m = 8`, and by 0.11% and 0.17% at `m = 12`**, short
  of the score_transforms topic's 0.5% bar. At the reference shape softplus leads it by 0.42% and 0.02%, sum of
  squares by 0.89% and 0.43%.
- **Degree twelve moves the RoPE polynomials by at most 0.24%**: query −0.24%, sum of squares −0.15%, fixed
  Taylor −0.13%, softplus −0.10%, static 0.00%; FoX query +0.14%. At the reference shape query gains 0.68% and
  fixed Taylor loses 0.63%.
- **Softmax trails exp by 2.66%** (2.35% at 1x, 2.00% at the reference shape).
- **Twice the tokens lowers validation-best perplexity by 8.6–10.3%**: query −10.30%, exp −10.14%, softplus
  −9.99% (against its 1x run on the causal strips), softmax −9.86%, FoX exp −8.56%. Against exp, query moves
  from +0.24% to +0.05%, softplus from −0.36% to −0.20%, FoX exp from −4.54% to −2.87% and softmax from
  +2.35% to +2.66%.
- **At the same tokens the 12-layer width-640 shape stays 10.1–12.4% above the reference shape.** FoX query at
  `m = 8` (+10.12%) and fixed Taylor at `m = 12` (+10.16%) come closest, softmax (+11.34%) and FoX exp
  (+12.41%) least close. For the five variants with 1x runs at this shape, the gap at 1x, with half the tokens,
  is 22.6–23.5%.

## Limits

- One seed (42) and data order per run. In the study's [noise check](../../../README.md#noise) a second seed and
  data order moved single runs by 0.6–0.7% and a gap between two variants by about 0.1 percentage point.
- GPUs, from each record's attempts; no update ran on a B300:
  - one H100 PCIe (`mastiff`) throughout: query at `m = 8` and `m = 12`, exp, softmax and FoX exp;
  - one H100 SXM (`dogo`) throughout: static and fixed Taylor at `m = 8` and `m = 12`, whose first attempts,
    on `akita`'s B300, were preempted during acceptance, before any update, and requeued;
  - an H100 PCIe (`mastiff`) until Kennel's Slurm reset on 2026-11-13, then an H100 SXM (`dogo`) from the last
    checkpoint, after acceptance there: softplus input from update 13,500 (`m = 8`) and 8,500 (`m = 12`), sum of
    squares from 6,500 and 5,500, FoX query from 13,500 and 12,500. The sum-of-squares and FoX query jobs were
    first preempted on `akita` during acceptance. These six runs' validation-best checkpoints were trained on
    `dogo`.
- The comparisons with 1x runs also cross GPUs and implementations: the 1x query, exp, softmax and FoX exp runs
  at this shape trained on B300, the 1x softplus run on an H100 PCIe on the causal strips (the 2x softplus runs
  use the Mosaic transformed kernels), and the reference-shape runs on B300, H100 PCIe or H100 SXM (FoX query on
  ORCD).

## Open decision

All fifteen runs are complete. Whether to promote the topic to its own canonical collection or discard it with
its runs and their artifacts is the user's decision.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/matched_tokens
```

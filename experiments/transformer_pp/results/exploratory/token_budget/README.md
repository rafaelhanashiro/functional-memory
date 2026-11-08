# Token budget: 1x against 2.5x Chinchilla

Opened 2026-10-27.

**Question.** Do the comparisons between attention variants change when the Transformer++
recipe trains 2.5x longer, with warmup and validation cadence scaled with the budget?

## Runs

Five runs at 2.5x Chinchilla (18,925 updates, 9.92B tokens, warmup 1,138), each the
data-scaled copy of a 1x run with the canonical per-head QK scale, one GPU each. The
[study README](../../../README.md#model-shapes-and-token-budgets) gives the scaling rule and launch
commands. Softmax has no canonical-QK 1x partner; the 1x exp partner `r6a76e7fbde45`
is cited here but not pinned.

Five further 2.5x runs on ORCD H100s add key–query attention at degrees 4 and 8 and the
degree-8 fixed Taylor, static, and query variants. The key–query runs pair with their
canonical-QK 1x runs `r501f5cca0d4c` (m = 4) and `r536dee0929d5` (m = 8); the degree-8
fixed, static, and query variants have no canonical-QK 1x run. Key–query uses the fused
query–key kernel; the other polynomial runs use the Mosaic kernels.

## Findings

Terminal validation perplexity. For every 2.5x run the validation-best checkpoint is
the terminal one; each pinned 1x run's best (update 7,500) is at most 0.006 lower than
its terminal value.

| Variant | 1x | 2.5x | 2.5x vs 1x | vs fixed Taylor at 1x | vs fixed Taylor at 2.5x |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp | 14.262 (`r6a76e7fbde45`) | 12.449 (`rfe12e5c36e94`) | -12.7% | -3.1% | -2.1% |
| Fixed Taylor m = 4 | 14.724 (`r816ab0fbe119`) | 12.710 (`r4f1682c1b54d`) | -13.7% | +0.0% | +0.0% |
| Static m = 4 | 14.773 (`r4339a245820e`) | 12.718 (`raa81f92b6aaf`) | -13.9% | +0.3% | +0.1% |
| Query m = 4 | 14.779 (`r2b22f2ff6461`) | 12.724 (`reb9b5be4ff1b`) | -13.9% | +0.4% | +0.1% |
| Key–query m = 4 | 14.726 (`r501f5cca0d4c`) | 12.664 (`r751984d1ca00`) | -14.0% | +0.0% | -0.4% |
| Softmax | — | 12.769 (`r8ccda981be2b`) | — | — | +0.5% |

- Training 2.5x longer lowers perplexity by 12.7–14.0% for every variant.
- Among exp, fixed Taylor, static and query m = 4 the ordering does not change: exp stays
  best, and the three polynomials end within 0.1% of one another at 2.5x (0.4% at 1x);
  their gap to exp narrows from 3.1–3.5% to 2.1–2.2%.
- Softmax trails fixed Taylor, static and query m = 4 by 0.4–0.5% and key–query m = 4 by
  0.8% at 2.5x. At the same update the query m = 4 run is below softmax from update
  11,250 on, and key–query m = 4 from update 6,250.
- Apart from key–query m = 4 at 2.5x, these runs used the pre-7e5c962f single-device
  training loop; their throughput is not comparable with runs launched after that commit.

Degree 8, against fixed Taylor m = 8 and exp at 2.5x:

| Variant | 1x | 2.5x | 2.5x vs 1x | vs fixed Taylor m = 8 | vs exp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed Taylor m = 8 | — | 12.475 (`r0ea4c732c5ac`) | — | +0.0% | +0.2% |
| Static m = 8 | — | 12.471 (`r6c9bcd661078`) | — | -0.0% | +0.2% |
| Query m = 8 | — | 12.475 (`r77bca86ef253`) | — | -0.0% | +0.2% |
| Key–query m = 8 | 14.366 (`r536dee0929d5`) | 12.449 (`rb7e83a4b653f`) | -13.3% | -0.2% | +0.0% |

- Key–query m = 4 ties fixed Taylor at 1x and ends 0.4% below it at 2.5x, below every
  other m = 4 polynomial and softmax.
- Degree 8 lowers 2.5x perplexity by 1.7–2.0% relative to degree 4. Fixed Taylor,
  static, and query m = 8 finish within 0.04% of one another; key–query m = 8 is 0.2%
  lower, level with exp.
- Each cell is a single seed (42), and the study has no seed repeats, so differences of
  a few tenths of a percent are not resolved. These five runs used the single
  compiled-step training loop (7e5c962f); hardware and kernels differ from the
  degree-4 Kennel runs.

## Open decision

Add further budgets (for example 5x) and promote a budget-scaling collection, or
discard the topic.

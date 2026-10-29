# Token budget: 1x against 2.5x Chinchilla

Opened 2026-10-27.

**Question.** Do the comparisons between attention variants change when the Transformer++
recipe trains 2.5x longer, with warmup and validation cadence scaled with the budget?

## Runs

Five runs at 2.5x Chinchilla (18,925 updates, 9.92B tokens, warmup 1,138), each the
data-scaled copy of a 1x run with the canonical per-head QK scale, one GPU each. The
[study README](../../../README.md#token-budgets) gives the scaling rule and launch
commands. Softmax has no canonical-QK 1x partner; the 1x exp partner `r6a76e7fbde45`
is on `codex/qk-head-controls-4b` and is cited here but not pinned.

## Findings

Terminal validation perplexity (the validation-best checkpoint is the terminal one for
every run):

| Variant | 1x | 2.5x | 2.5x vs 1x | vs fixed Taylor at 1x | vs fixed Taylor at 2.5x |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp | 14.262 (`r6a76e7fbde45`) | 12.449 (`rfe12e5c36e94`) | -12.7% | -3.1% | -2.1% |
| Fixed Taylor m = 4 | 14.724 (`r816ab0fbe119`) | 12.710 (`r4f1682c1b54d`) | -13.7% | +0.0% | +0.0% |
| Static m = 4 | 14.773 (`r4339a245820e`) | 12.718 (`raa81f92b6aaf`) | -13.9% | +0.3% | +0.1% |
| Query m = 4 | 14.779 (`r2b22f2ff6461`) | 12.724 (`reb9b5be4ff1b`) | -13.9% | +0.4% | +0.1% |
| Softmax | — | 12.769 (`r8ccda981be2b`) | — | — | +0.5% |

- Training 2.5x longer lowers perplexity by 12.7–13.9% for every variant.
- The ordering does not change: exp stays best, and fixed Taylor, static and query m = 4
  stay within 0.1% of one another; their gap to exp narrows from 3.1% to 2.1%.
- Softmax trails the m = 4 polynomials by 0.4–0.5% at 2.5x. At the same update the
  query run overtakes softmax from update 12,500 on.
- These runs used the pre-7e5c962f single-device training loop; their throughput is
  not comparable with runs launched after that commit.

## Open decision

Add further budgets (for example 5x) and promote a budget-scaling collection, or
discard the topic.

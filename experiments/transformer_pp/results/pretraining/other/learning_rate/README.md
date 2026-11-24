# Learning rate at Q/K width 4: exp and `z h_t(z)` (Transformer++)

Opened 2026-11-22.

**Question.** The recipe's learning rate was tuned at `d_K` 64. At `d_K` 4, is it the best one for exp and for
`z h_t(z)`, `z = softplus(s)`, with a degree-sixteen SoS core, the two kernels that tie there, and does the tie hold
at other rates? Each trains again with its whole schedule at half and at double: peak and floor scale together
(1e-3 and 1e-4 at the recipe's rate), and the 455-update warmup, horizon, clipping at 1.0, weight decay, seed 42 and
every model setting stay. The [study methods](../../../../README.md#learning-rate-at-key-width-four)
define the recipes and launch commands. Home cluster: Kennel.

## Runs

Validation perplexity, best with terminal in parentheses.

| Kernel | Half rate | Recipe's rate | Double rate |
| --- | ---: | ---: | ---: |
| exp | 20.860 (20.860) `r03217591054b` | 19.190 (19.201) `r2739c80cc032` | 18.667 (18.679) `rcc51ed9dbca8` |
| `z h_t(z)`, core 16 | 20.633 (20.635) `r1934bca84bd4` | 19.192 (19.197) `r2e7f5617fdf2` | 18.637 (18.647) `rd5cc0aec9712` |
| `z h_t(z)` against exp | −1.08% | +0.01% | −0.16% |
| Against the recipe's rate: exp | +8.70% | | −2.73% |
| Against the recipe's rate: `z h_t(z)` | +7.51% | | −2.90% |

## Findings

- **The recipe's rate is too low at `d_K` 4 for both kernels.** Doubling it lowers perplexity by 2.7% (exp) and
  2.9% (`z h_t(z)`), halving it raises it by 8.7% and 7.5%. Neither kernel shows its best rate inside the range.
- **The tie holds at the recipe's rate and at double:** `z h_t(z)` is 0.01% above exp and 0.16% below. At half
  rate it is 1.1% below.
- **Updates are clipped for longer at lower rates.** More than 2% of a validation window's updates exceed the clip
  until update 1,400 at half rate (59–96% through update 1,000), until update 500 at the recipe's rate and until
  update 300 at double. After update 1,500 no window has more than 1.0% clipped at any rate, 0.2% at double.
- **`z h_t(z)` leads exp early at every rate,** by 8.7%, 5.8% and 4.6% at update 1,000 from half to double. By
  update 3,000 the lead is 3.0% at half rate, 0.5% at the recipe's, and gone at double.

## Limits

- One seed per run, so the kernels' differences of 0.01% and 0.16% at the recipe's rate and at double are not
  resolved.
- Three rates only, a factor of two apart; the best rate lies at or above double.
- The exp runs at half and double rate take head RMSNorm's max-abs prescale and the run at the recipe's rate its
  plain arithmetic, the same function.
- Hardware: the exp runs trained on a B300 (the recipe's rate also on an H100); each `z h_t(z)` run on a B200 and a
  B300 across preemptions.

## Open question

Where each kernel's best rate lies: a run of each at four times the rate would close the range.

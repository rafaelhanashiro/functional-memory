# Learning rate at Q/K width 4: exp and `z h_t(z)` (Transformer++)

Opened 2026-11-22.

**Question.** The recipe's learning rate was tuned at `d_K` 64. At `d_K` 4, is it the best one for exp and for
`z h_t(z)`, `z = softplus(s)`, with a degree-sixteen SoS core, the two kernels that tie there, and does the tie hold
at each kernel's best rate? Each trains again with its whole schedule at half, double and four times: peak and floor
scale together (1e-3 and 1e-4 at the recipe's rate), and the 455-update warmup, horizon, clipping at 1.0, weight
decay, seed 42 and every model setting stay. The [study methods](../../../../README.md#learning-rate-at-key-width-four)
define the recipes and launch commands. Home cluster: Kennel.

## Runs

Validation perplexity, best with terminal in parentheses.

| Kernel | Half rate | Recipe's rate | Double rate | Four times |
| --- | ---: | ---: | ---: | ---: |
| exp | 20.860 (20.860) `r03217591054b` | 19.190 (19.201) `r2739c80cc032` | 18.667 (18.679) `rcc51ed9dbca8` | 18.904 (18.925) `r1ff405fe0926` |
| `z h_t(z)`, core 16 | 20.633 (20.635) `r1934bca84bd4` | 19.192 (19.197) `r2e7f5617fdf2` | 18.637 (18.647) `rd5cc0aec9712` | 18.203 (18.210) `rdc43b5c9d2d0` |
| `z h_t(z)` against exp | −1.08% | +0.01% | −0.16% | −3.71% |
| Against the recipe's rate: exp | +8.70% | | −2.73% | −1.49% |
| Against the recipe's rate: `z h_t(z)` | +7.51% | | −2.90% | −5.15% |

## Findings

- **exp's best rate is double, and `z h_t(z)`'s is at least four times.** exp gains 2.7% at double and loses 1.3% of
  it again at four times; `z h_t(z)` gains at every step up, 2.3% more from double to four times.
- **At each kernel's best rate here, `z h_t(z)` beats exp by 2.5%:** 18.203 against 18.667. The kernels tie at the
  recipe's rate and at double (+0.01% and −0.16%).
- **At four times exp trains unstably and `z h_t(z)` does not.** After update 1,500 exp has 133 gradient spikes,
  up to 34% of a validation window's updates clipped, a largest norm of 23,496, and more than 2% of updates clipped
  in some window until update 5,500; its validation perplexity reaches 101.5 at update 5,000 and recovers to 20.8 by
  update 5,500. `z h_t(z)` has 14 spikes, at most 0.8% clipped and a largest norm of 12.4 after update 1,500.
- **Updates are clipped for longer at lower rates.** More than 2% of a validation window's updates exceed the clip
  until update 1,400 at half rate (59–96% through update 1,000), until update 500 at the recipe's rate, 300 at double
  and 200 for `z h_t(z)` at four times.
- **`z h_t(z)` leads exp early at every rate,** by 8.7%, 5.8%, 4.6% and 18.4% at update 1,000 from half to four
  times. By update 3,000 the lead is 3.0% at half rate, 0.5% at the recipe's, gone at double and 3.6% at four
  times, where it stays.

## Limits

- One seed per run, so the kernels' differences of 0.01% and 0.16% at the recipe's rate and at double are not
  resolved.
- Rates a factor of two apart; `z h_t(z)`'s best rate lies at or above four times and is not bracketed.
- The exp runs at half and double rate take head RMSNorm's max-abs prescale and the run at the recipe's rate its
  plain arithmetic, the same function.
- Hardware: the exp runs trained on a B300 (the recipe's rate also on an H100); each `z h_t(z)` run on a B200 and a
  B300 across preemptions, the four-times run on a B200 alone.

## Open question

Where `z h_t(z)`'s best rate lies: a run at eight times the rate would bracket it.

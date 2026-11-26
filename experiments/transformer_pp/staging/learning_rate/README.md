# Learning rate at Q/K widths 4 and 64 (Transformer++)

Opened 2026-11-22.

**Question.** The recipe's learning rate was tuned at `d_K` 64. At `d_K` 4, is it the best one for exp and for
`z h_t(z)`, `z = softplus(s)`, with a degree-sixteen or degree-eight SoS core, and does their tie at the recipe's rate
hold at each kernel's best rate? At `d_K` 64, does it stay the best one for exp, `z h_t(z)` and the degree-eight SoS
core under a prefactor of the raw score? Each run is its partner's recipe with the whole schedule's peak and floor
scaled together (1e-3 and 1e-4 at the recipe's rate); the 455-update warmup, horizon, clipping at 1.0, weight decay,
seed 42 and every model setting stay. The [study methods](../../README.md#learning-rate-at-key-width-four)
define the recipes and launch commands. Home cluster: Kennel.

## Runs at `d_K` 4

Validation perplexity, best with terminal in parentheses.

| Kernel | Half rate | Recipe's rate | Double rate | Four times | Eight times |
| --- | ---: | ---: | ---: | ---: | ---: |
| exp | 20.860 (20.860) `r03217591054b` | 19.190 (19.201) `r2739c80cc032` | 18.667 (18.679) `rcc51ed9dbca8` | 18.904 (18.925) `r1ff405fe0926` | |
| `z h_t(z)`, core 16 | 20.633 (20.635) `r1934bca84bd4` | 19.192 (19.197) `r2e7f5617fdf2` | 18.637 (18.647) `rd5cc0aec9712` | **18.203** (18.210) `rdc43b5c9d2d0` | 18.417 (18.451) `rda43ee639e0a` |
| `z h_t(z)`, core 8 | | 19.463 (19.471) `rabfbab32a594` | 19.075 (19.083) `rfc577a90377d` | 18.569 (18.579) `r9814496cc661` | 21.109 (21.109) `r7cbb2c7ca07d` |
| Against the recipe's rate: exp | +8.70% | | −2.73% | −1.49% | |
| Against the recipe's rate: core 16 | +7.51% | | −2.90% | −5.15% | −4.04% |
| Against the recipe's rate: core 8 | | | −1.99% | −4.59% | +8.46% |

## Runs at `d_K` 64

| Kernel | Half rate | Recipe's rate | Double rate | Four times |
| --- | ---: | ---: | ---: | ---: |
| exp | 14.573 (14.578) `rc44dc19a9972` | 14.257 (14.262) `r6a76e7fbde45` | | |
| `z h_t(z)`, core 8 | 14.535 (14.536) `rd1baa331563c` | 14.095 (14.099) `r841f35a24241` | | |
| `exp(s) h_t(s)`, core 8 | | 14.089 (14.092) `r6525664e6696` | **14.063** (14.066) `r76b9aa49fe23` | 14.210 (14.222) `rbf850552cbba` |
| `softplus(s) h_t(s)`, core 8 | | 14.097 (14.101) `r41662adb22fa` | 14.166 (14.170) `rf93d2b71f731` | 14.268 (14.277) `rb418110c763c` |
| `sigmoid(s) h_t(s)`, core 8 | | 14.148 (14.151) `r8f56a5b7e525` | 14.281 (14.282) `r7d7504df667b` | 14.378 (14.386) `rb6eeed222062` |

Against the recipe's rate: exp at half +2.22%; `z h_t(z)` at half +3.12%; `exp(s) h_t(s)` −0.18% at double and
+0.86% at four times; `softplus(s) h_t(s)` +0.49% and +1.21%; `sigmoid(s) h_t(s)` +0.94% and +1.63%. ORCD trained exp,
the degree-eight query polynomial and the softplus-input SoS kernels at double and four times on its branch; they join
these tables when it merges.

## Findings

- **At `d_K` 4 a larger rate helps, and the SoS cores most.** exp's best is double (−2.73%); both SoS cores are best
  at four times (core 16 −5.15%, core 8 −4.59%), where they beat exp's best by 2.5% (18.203) and 0.5% (18.569).
- **The higher degree tolerates the larger rate.** At eight times core 16 has one gradient spike (update 900) and ends
  1.2% behind four times; core 8 trains unstably (267 gradient events, validation 182 at update 3,000 and 359 at
  7,000) and ends 8.5% above its recipe's rate.
- **At `d_K` 64 the recipe's rate is about right.** No kernel gains more than 0.2% from double, four times costs
  0.9–1.6%, and half costs 2.2% (exp) and 3.1% (`z h_t(z)`): the rate tuned at `d_K` 64 stays its best.
- **At four times exp trains unstably at `d_K` 4 and `z h_t(z)` does not.** After update 1,500 exp has 133 gradient
  spikes, up to 34% of a validation window's updates clipped, a largest norm of 23,496, and more than 2% of updates
  clipped in some window until update 5,500; its validation perplexity reaches 101.5 at update 5,000 and recovers to
  20.8 by update 5,500. `z h_t(z)` has 14 spikes, at most 0.8% clipped and a largest norm of 12.4 after update 1,500.
- **Mid-run gaps overstate a larger rate's cost.** The higher rate gains 1–3 points on the lower one during the
  decay: `exp(s) h_t(s)` at four times is 4.1% behind its recipe's rate at update 3,500 and ends 0.9% behind.
- **Updates are clipped for longer at lower rates.** More than 2% of a validation window's updates exceed the clip
  until update 1,400 at half rate (59–96% through update 1,000), until update 500 at the recipe's rate, 300 at double
  and 200 for `z h_t(z)` at four times.

## Limits

- One seed per run, so differences under 0.6–0.7% (seed noise at `d_K` 64) are not resolved, among them every
  `d_K` 64 kernel's double rate against its recipe's.
- Rates a factor of two apart.
- The exp runs at half and double rate take head RMSNorm's max-abs prescale and the run at the recipe's rate its
  plain arithmetic, the same function; the exp run at half rate at `d_K` 64 also takes the row-maximum shift.
- Hardware: the `d_K` 4 exp runs trained on a B300 (the recipe's rate also on an H100); the SoS runs on a B200, some
  also on a B300 across preemptions; the `d_K` 64 runs on a B200, the exp run at half rate on a B300.

## Open question

Whether a larger rate keeps the SoS's lead over exp at `d_K` 4 with more data: runs at twice the Chinchilla budget
(training) take both at their best rates and half of them.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/learning_rate
```

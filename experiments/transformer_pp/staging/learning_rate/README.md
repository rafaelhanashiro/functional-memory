# Learning rate at Q/K widths 4 and 64 (Transformer++)

Opened 2026-11-22.

**Question.** The recipe's learning rate was tuned at `d_K` 64. At `d_K` 4, is it the best one for exp and for
`z h_t(z)`, `z = softplus(s)`, with a degree-sixteen or degree-eight SoS core, and does their tie at the recipe's rate
hold at each kernel's best rate? At `d_K` 64, does it stay the best one for the tanh query polynomial, exp,
`softplus(s) exp(s)`, `z h_t(z)` and the degree-eight SoS core under a prefactor of the raw score, on the raw score or
on `z`? Each run is its partner's recipe with the whole schedule's peak and floor
scaled together (1e-3 and 1e-4 at the recipe's rate); the 455-update warmup, horizon, clipping at 1.0, weight decay,
seed 42 and every model setting stay. The study methods define the recipes and launch commands: Kennel's
[runs](../../README.md#learning-rate-at-key-width-four), and ORCD's
[runs at `d_K` 64](../../README.md#two-and-four-times-the-learning-rate-at-key-width-64) at double and four times for
the query polynomial, exp, `softplus(s) exp(s)` and the SoS core on `z`. Home clusters: Kennel and ORCD.

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
| tanh query polynomial, m = 8 | | 14.323 (14.328) `r25fffc16c2f3` | 14.742 (14.748) `r90776a5d4175` | 14.783 (14.790) `r9e36ee87aa7e` |
| exp | 14.573 (14.578) `rc44dc19a9972` | 14.257 (14.262) `r6a76e7fbde45` | 14.212 (14.215) `r19dfa87cb7ed` | 14.342 (14.348) `r530a1bce3597` |
| `softplus(s) exp(s)` | | 14.132 (14.136) `rbd24b948a240` | 14.118 (14.122) `r2118e3531d89` | 14.359 (14.371) `re7d0d4e6a5b7` |
| `z h_t(z)`, core 8 | 14.535 (14.536) `rd1baa331563c` | 14.095 (14.099) `r841f35a24241` | 14.168 (14.173) `r752878a0a030` | 14.305 (14.313) `r7924001c42f1` |
| `exp(s) h_t(z)`, core 8 | | 14.067 (14.069) `raeaa47865867` | **14.001** (14.009) `r39d0469cc122` | 14.179 (14.188) `r365f0c30f632` |
| `sigmoid(s) h_t(z)`, core 8 | | 14.271 (14.275) `r5d578a29f080` | 14.381 (14.387) `rf0053d2b3b45` | 14.366 (14.377) `r101b6c157a6b` |
| `exp(s) h_t(s)`, core 8 | | 14.089 (14.092) `r6525664e6696` | 14.063 (14.066) `r76b9aa49fe23` | 14.210 (14.222) `rbf850552cbba` |
| `softplus(s) h_t(s)`, core 8 | | 14.097 (14.101) `r41662adb22fa` | 14.166 (14.170) `rf93d2b71f731` | 14.268 (14.277) `rb418110c763c` |
| `sigmoid(s) h_t(s)`, core 8 | | 14.148 (14.151) `r8f56a5b7e525` | 14.281 (14.282) `r7d7504df667b` | 14.378 (14.386) `rb6eeed222062` |

Against the recipe's rate, at double and four times: the query polynomial +2.93% and +3.21%; exp −0.32% and +0.59%
(+2.22% at half); `softplus(s) exp(s)` −0.10% and +1.60%; `z h_t(z)` +0.52% and +1.49% (+3.12% at half);
`exp(s) h_t(z)` −0.47% and +0.80%; `sigmoid(s) h_t(z)` +0.77% and +0.67%; `exp(s) h_t(s)` −0.19% and +0.86%;
`softplus(s) h_t(s)` +0.49% and +1.21%; `sigmoid(s) h_t(s)` +0.94% and +1.63%.

## Findings

- **At `d_K` 4 a larger rate helps, and the SoS cores most.** exp's best is double (−2.73%); both SoS cores are best
  at four times (core 16 −5.15%, core 8 −4.59%), where they beat exp's best by 2.5% (18.203) and 0.5% (18.569).
- **The higher degree tolerates the larger rate.** At eight times core 16 has one gradient spike (update 900) and ends
  1.2% behind four times; core 8 trains unstably (267 gradient events, validation 182 at update 3,000 and 359 at
  7,000) and ends 8.5% above its recipe's rate.
- **At `d_K` 64 the recipe's rate is about right.** No kernel gains more than 0.5% from double, within seed noise
  (exp −0.32%, `exp(s) h_t(z)` −0.47%); four times costs 0.6–1.6% (3.2% for the query polynomial), and half costs
  2.2% (exp) and 3.1% (`z h_t(z)`): no other rate is resolvably better than the recipe's at `d_K` 64. The best
  `d_K` 64 run here is `exp(s) h_t(z)` at double, 14.001, 0.47% below its recipe's rate, within seed noise.
- **The tanh query polynomial loses most from a larger rate.** It ends 2.93% behind its recipe's rate at double and
  3.21% at four times, with 21 and 53 gradient spikes, the most of any `d_K` 64 run that counts them: no exp or SoS
  run counts more than 5. Its clipping stays at or below 1% of a window after update 500. Its partner and the exp
  partner predate the spike counter.
- **With one seed, the response to double follows the prefactor on both SoS inputs.** Under exp, double helps on the
  raw score (−0.19%) and on `z` (−0.47%); under softplus it costs 0.49% on the raw score and 0.52% on `z`
  (`z h_t(z)`, softplus on `z` through the input power); under sigmoid 0.94% and 0.77%. The exp and softplus
  differences are within seed noise and the sigmoid costs at or above it; the two inputs agree to 0.3 points for each
  prefactor.
- **At four times exp trains unstably at `d_K` 4 and `z h_t(z)` does not.** After update 1,500 exp has 133 gradient
  spikes, up to 34% of a validation window's updates clipped, a largest norm of 23,496, and more than 2% of updates
  clipped in some window until update 5,500; its validation perplexity reaches 101.5 at update 5,000 and recovers to
  20.8 by update 5,500. `z h_t(z)` has 14 spikes, at most 0.8% clipped and a largest norm of 12.4 after update 1,500.
- **Mid-run gaps overstate a larger rate's cost.** The higher rate gains 1–3 points on the lower one during the
  decay: `exp(s) h_t(s)` at four times is 4.1% behind its recipe's rate at update 3,500 and ends 0.9% behind. At four
  times every `d_K` 64 kernel is 3.5–4.8% behind its recipe's rate at updates 3,000–4,000 (the query polynomial
  6.2–6.5%) and regains about 3 points by the end; at double it regains about 1.
- **Updates are clipped for longer at lower rates.** More than 2% of a validation window's updates exceed the clip
  until update 1,400 at half rate (59–96% through update 1,000), until update 500 at the recipe's rate, 300 at double
  and 200 for `z h_t(z)` at four times.

## Limits

- One seed per run, so differences under 0.6–0.7% (seed noise at `d_K` 64) are not resolved, among them the double
  rate against the recipe's for exp, `softplus(s) exp(s)`, `z h_t(z)` and the exp and softplus prefactors (−0.47% to
  +0.52%); the query polynomial (+2.93%) and the sigmoid prefactor (+0.77% on `z`, +0.94% on the raw score) exceed
  it.
- Rates a factor of two apart.
- The exp runs at half and double rate take head RMSNorm's max-abs prescale and the run at the recipe's rate its
  plain arithmetic, the same function; at `d_K` 64 every run off the recipe's rate takes the prescale, which its
  partner predates, and the exp runs also the row-maximum shift.
- Hardware: the `d_K` 4 exp runs trained on a B300 (the recipe's rate also on an H100); the SoS runs on a B200, some
  also on a B300 across preemptions; Kennel's `d_K` 64 runs on a B200, the exp run at half rate on a B300. ORCD's
  `d_K` 64 runs trained on H100s, their partners on Kennel (the query polynomial and `softplus(s) exp(s)` on a B300,
  exp on a B300, `z h_t(z)` on a B200, `exp(s) h_t(z)` on an H100, `sigmoid(s) h_t(z)` on a B200 and an H100), so
  each of those comparisons but `exp(s) h_t(z)`'s (H100s on both clusters) also changes the GPU.
- ORCD's query polynomial and exp runs launched at `db4b0a35`, its SoS and `softplus(s) exp(s)` runs at `a189117e`;
  `z h_t(z)` and `sigmoid(s) h_t(z)` at four times resumed from checkpoints after three preemptions each and the query
  polynomial at double after one; `exp(s) h_t(z)` at four times restarted from the beginning after a preemption
  during acceptance.

## Open question

Whether a larger rate keeps the SoS's lead over exp at `d_K` 4 with more data: runs at twice the Chinchilla budget
(training) take both at their best rates and half of them.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/learning_rate
```

# The softplus prefactor on the raw score at Q/K width 4

Opened 2026-11-25. At `d_K` 4, does `softplus(s) h_t(s)`, the degree-eight query SoS core on the raw score under the
softplus prefactor, serve as well as `z h_t(z)`, `z = softplus(s)`, as it does at `d_K` 64 (14.097 against 14.095)?
Each run is the raw-score core's recipe (`reea3d6436271`, `h_t(s)`, 19.762) with the prefactor, from its exact
initial parameters, at one, two and four times the schedule's learning rate
([methods and launch](../../README.md#the-softplus-prefactor-on-the-raw-score-at-key-width-four)). Home cluster:
Kennel.

## Runs

Validation perplexity, best with terminal in parentheses.

| Kernel, core 8 | Recipe's rate | Double rate | Four times |
| --- | ---: | ---: | ---: |
| `softplus(s) h_t(s)` | 19.841 (19.859) `r382ea631f6f4` | 18.806 (18.815) `r0df2e1783a49` | **18.541** (18.560) `rbb13b7964401` |
| `z h_t(z)` | 19.463 (19.471) `rabfbab32a594` | 19.075 (19.083) `rfc577a90377d` | 18.569 (18.579) `r9814496cc661` |
| exp | 19.190 (19.201) `r2739c80cc032` | 18.667 (18.679) `rcc51ed9dbca8` | 18.904 (18.925) `r1ff405fe0926` |
| `softplus(s) h_t(s)` against `z h_t(z)` | +1.94% | −1.41% | −0.15% |
| `softplus(s) h_t(s)` against exp | +3.39% | +0.74% | −1.92% |
| Gradient spikes, `softplus(s) h_t(s)` | 358 | 3 | 62 |

## Findings

- **At its best rate the raw-score form ties `z h_t(z)` and edges past exp's best.** At four times the rate it
  reaches 18.541, level with `z h_t(z)` (18.569) and 0.68% below exp's best (18.667, at double the rate).
- **At the recipe's rate it is the worst of the three and trains unstably**: 19.841 with 358 gradient spikes, 1.9%
  behind `z h_t(z)`, 3.4% behind exp, and 0.4% behind its own core without the prefactor (`h_t(s)`, 19.762).
- **A larger rate helps it more than it helps the others**: −5.2% from the recipe's rate to double and −6.6% to four
  times, against −2.0% and −4.6% for `z h_t(z)` and −2.7% for exp at double.
- At double the rate it is 1.4% ahead of `z h_t(z)`, whose run there has a validation spike at update 3,500.

## Limits

- One seed per run; the 0.68% lead over exp's best and the tie with `z h_t(z)` are within one seed's noise.
- Rates a factor of two apart; eight times was not run for this form (`z h_t(z)` at eight times: 21.109, unstable).
- exp at four times the rate trains unstably, so the comparison there is with exp's best rate.
- Hardware: all three runs on a B200, the run at double the rate across ten attempts.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/narrow_softplus_prefactor_sos
```

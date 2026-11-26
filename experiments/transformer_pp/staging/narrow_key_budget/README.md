# Twice the Chinchilla budget at Q/K width 4

Opened 2026-11-24. With twice the training tokens, does `z h_t(z)`, `z = softplus(s)`, with a degree-sixteen SoS core
keep its lead over exp at `d_K` 4? At 1x budget each kernel's best rate leaves it 2.5% ahead: 18.203 at four times
the recipe's rate against exp's 18.667 at double ([learning_rate](../learning_rate/README.md)). Each kernel trains
at that rate and at half of it for 15,140 updates (2x Chinchilla at the 15-layer width-896 shape): its 1x recipe
with the horizon and warmup doubled (910 updates) and the batch, seed 42 and every model setting kept
([methods and launch](../../README.md#twice-the-chinchilla-budget-at-key-width-four)). Home cluster: Kennel.

## Runs

Validation perplexity, best. Every 2x run's best is its last update, 15,140.

| Kernel | Rate | 2x budget | 1x budget | Change | Gradient spikes at 2x |
| --- | --- | ---: | ---: | ---: | ---: |
| exp | recipe's | 16.792 `r0303c5fcd9b6` | 19.190 `r2739c80cc032` | −12.5% | 3 |
| exp | double | 16.614 `r41562d2eac42` | 18.667 `rcc51ed9dbca8` | −11.0% | 317 |
| `z h_t(z)`, core 16 | double | **16.505** `r0ee4e14367df` | 18.637 `rd5cc0aec9712` | −11.4% | 3 |
| `z h_t(z)`, core 16 | four times | 16.741 `rc81683587538` | 18.203 `rdc43b5c9d2d0` | −8.0% | 652 |

## Findings

- **`z h_t(z)` still beats exp at twice the budget, by less.** Best against best it is 0.66% ahead (16.505 against
  16.614), where it is 2.5% ahead at 1x. Both bests are at double the rate, and there `z h_t(z)` leads at all 28
  evaluations, by 0.6–0.7% from update 7,000 on (0.16% at 1x).
- **Its best rate moves down with the longer schedule.** At four times the rate it has 652 gradient spikes, in bursts
  from update 6,000 to 14,000 (validation perplexity 59.4 at update 7,000), and ends 1.4% behind its double-rate run
  and 0.76% behind exp's best. At 1x that rate was its best, with 14 spikes after update 1,500.
- **exp's best rate stays double,** 1.1% ahead of the recipe's rate (2.7% at 1x), with 317 gradient spikes: 163
  before update 3,000 and bursts near updates 7,000 and 9,000.
- **Twice the data lowers every run's perplexity by 8–12.5%.**

## Limits

- One seed per run; the 0.66% lead is at the 0.6–0.7% seed spread measured at `d_K` 64.
- Rates a factor of two apart: `z h_t(z)` between double and four times, and exp at four times, were not run at 2x.
- Hardware: the exp runs and `z h_t(z)` at four times trained on a B300 (exp across two and three attempts),
  `z h_t(z)` at double on a B200.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/narrow_key_budget
```

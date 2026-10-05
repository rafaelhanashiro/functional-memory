# Rational learned polynomial attention

Opened 2026-10-04. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) (a learned
normalized column of the degree chart) or discard.

Every normalized learned polynomial with one signed polynomial failed when a row sum
crossed zero (degree chart, learned column: fail at updates 27 to 1,857). Does a
separate denominator polynomial let it train through? The kernel is

```text
attn_ij = p_a(s_ij) / sum_(k<=i) p_b(s_ik),   p_a(s) = sum_(r<=8) a_r s^r
```

with cosine (QK-norm) scores and coefficients per layer and head. The denominator `p_b`
has constant 1 and is one of:

| Denominator | `p_b(s)` | Sign |
| --- | --- | --- |
| degree 8 | `1 + sum_(1<=r<=8) b_r s^r` | signed |
| linear | `1 + b_1 s` | signed |
| nonnegative quadratic | `(1 + u s)^2 + (w s)^2`, every nonnegative quadratic with constant 1 | `>= 0` |
| nonnegative degree 8 | `(1 + sum_(r<=4) u_r s^r)^2 + (sum_(r<=4) w_r s^r)^2`, every nonnegative degree-8 polynomial with constant 1 | `>= 0` |

The training loss is

```text
NLL + lambda * sum over layers and heads [ a_0^2 + sum_(r>=2) r^2 a_r^2 (+ sum_(r>=1) r^2 b_r^2) ]
```

with `a_1` unpenalized; the `b_r` term applies to the degree-8 denominator only. An
exactly zero row sum is a numerical failure, recorded as such. The validation loss and
the recorded training NLL exclude the penalty; `metrics.jsonl` records the penalty term
separately ([scaffold](../../../scaffolds/transformer/README.md#attention-and-parameter-ownership)).

Settings shared by these runs, each a candidate for tuning later:

| Setting | Value | Recipe field |
| --- | --- | --- |
| Initialization | linear attention: `a_1 = 1`, other numerator coefficients 0, denominator 1, so attention starts at `sum_(j<=i) s_ij v_j / (i + 1)` | `model.attention_coefficient_initialization: linear` |
| Nonnegative denominator start | `u_r = 0`, `w_1 = 10^-3`, other `w_r = 0` (all-zero `w` gets no gradient), so `p_b = 1 + 10^-6 s^2` | fixed in `attention/rational_polynomial.py` |
| Penalty weight `lambda` | 0.01 | `optim.coefficient_penalty` |
| AdamW weight decay (other parameters) | 0.1, the study's canonical value; polynomial coefficients are never decayed | `optim.weight_decay` |

All runs: 200M, `d_K = 64`, QK-norm, transformer reference initialization, seed 0,
19,074 updates (10.0B tokens) budget. Runs and roles: [`selection.yaml`](selection.yaml).

## Runs

| Run | Denominator | Penalized denominator | Outcome |
| --- | --- | --- | ---: |
| `r9c52f40605fb` | degree 8 | yes | fail @131 |
| `reb1c7f8a3f53` | linear | no | 15.157 |
| `rb5f4cb9e0d85` | nonnegative quadratic | no | 15.162 |
| `rbde7fc4c2135` | softmax (control) | | 13.514 |
| `r7d911aee60ed` | normalized fixed Taylor, degree 8 (control) | | 13.543 |
| `r4923c6939e74` | normalized fixed Taylor, degree 8, Flax-default initialization (control) | | 14.498 |

Outcomes are terminal validation perplexity.

## Low-degree denominators

Both runs trained to the end without a numerical failure. They finished 1.6 above
softmax and fixed Taylor, and within 0.005 of each other. Validation-best perplexity is
15.157 at update 19,074 for the linear denominator and 15.155 at 19,000 for the
quadratic.

| Validation PPL at update | 1,000 | 2,000 | 5,000 | 10,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear denominator | 54.33 | 29.93 | 20.16 | 16.93 | 15.157 |
| Nonnegative quadratic denominator | 54.44 | 30.04 | 20.09 | 16.90 | 15.162 |
| Softmax | 54.08 | 26.48 | 17.83 | 15.08 | 13.514 |
| Fixed Taylor, degree 8 | 54.85 | 26.85 | 17.89 | 15.12 | 13.543 |

- **The gap opens between updates 1,000 and 2,000**, reaching 3.5 against softmax, then
  narrows to 1.6 by the end.
- **The numerator is not linear.** Coefficients beyond `a_1` stay below 0.01, but
  within `|s| <= 3` they outweigh `a_1 s` by a median factor of 7 across heads. The
  penalty term was 5e-4 nats at the end.
- **The linear denominator turned negative for some keys.** `b_1` ranges from −0.73 to
  0.50, and per-key values reach −2.3 in layers 4 to 9. Row sums on the diagnostic
  sample stayed at least 0.84 in magnitude.
- **The quadratic denominator left its start.** `w` has median 0.23 against its start
  of 1e-3, and the expanded `b_2` reaches 1.56. Four of 160 heads have `|w| < 1e-3`, a
  near-double root.

## Degree-8 denominator

The run failed at update 131 (67M tokens, 0.7% of the budget) with a NaN forward loss
on finite parameters. The training NLL fell from 10.58 to 8.57 at update 70, then rose
to 10.18 by update 130, while softmax fell to 7.23. The penalty term was 1.1e-4 nats.

- **Not overflow.** QK-norm bounds scores by 8, and Adam moves each coefficient by at
  most 1.3e-3 by update 131, so `|p_b| <= 2.5e4`, far below FP32 or BF16 overflow.
- **A zero denominator sum, inferred.** With bounded scores and coefficients, the only
  forward NaN the kernel produces is an exactly zero row sum. The failure state is
  NaN throughout, so the sum itself was not measured.
- **High-order terms set the sign.** At `|s| = 5`, `s^8 = 3.9e5`: at update 20,
  coefficients within 2.9e-5 of their start already gave denominator values from
  −9.8 to 10.6 within one layer.
- **`lambda = 0.01` cannot act.** Under Adam the update direction is the task
  gradient's; the penalty's gradient on `b_8` was about 1e-3.
- **Weight decay is irrelevant**: coefficients are never decayed.

## Limits

- One seed, one numerator degree, one penalty weight, one coefficient initialization.
- The linear-denominator run trained on B300, the quadratic one on H100.
- No checkpoint precedes the degree-8 failure, so its failing update cannot be replayed.
- The Flax-default Taylor control is not an equal-initialization comparison; the
  softmax and reference-initialization Taylor controls are.

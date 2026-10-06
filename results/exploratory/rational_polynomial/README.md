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
is one of:

| Denominator | `p_b(s)` | Sign |
| --- | --- | --- |
| signed degree 8 | `1 + sum_(1<=r<=8) b_r s^r` | signed |
| linear, learned constant | `b_0 + b_1 s` | signed |
| nonnegative quadratic, learned constant | `(c_0 + c_1 s)^2 + d_0^2`, every nonnegative quadratic | `>= 0` |
| nonnegative degree 8 | `(1 + sum_(r<=4) u_r s^r)^2 + (sum_(r<=4) w_r s^r)^2`, every nonnegative degree-8 polynomial with constant 1 | `>= 0` |

With a learned constant, numerator and denominator can scale together without changing
the attention weights. The training loss is

```text
NLL + lambda * sum over layers and heads [ a_0^2 + sum_(r>=2) r^2 a_r^2 (+ sum_(r>=1) r^2 b_r^2) ]
```

with `a_1` unpenalized. The `b_r` term, on the expanded denominator for the nonnegative
form, applies only where the runs table says so. An exactly zero row sum is a numerical
failure, recorded as such. The validation loss and the recorded training NLL exclude the
penalty; `metrics.jsonl` records the penalty term separately
([scaffold](../../../scaffolds/transformer/README.md#attention-and-parameter-ownership)).

Settings shared by these runs, each a candidate for tuning later:

| Setting | Value | Recipe field |
| --- | --- | --- |
| Initialization | linear attention: `a_1 = 1`, other numerator coefficients 0, denominator 1, so attention starts at `sum_(j<=i) s_ij v_j / (i + 1)` | `model.attention_coefficient_initialization: linear` |
| Nonnegative denominator start, constant 1 | `u_r = 0`, `w_1 = 10^-3`, other `w_r = 0` (all-zero `w` gets no gradient), so `p_b = 1 + 10^-6 s^2` | fixed in `attention/rational_polynomial.py` |
| Nonnegative denominator start, learned constant | `c_0 = d_0 = 1/sqrt(2)`, `c_1 = 0`, so `p_b = 1` exactly | fixed in `attention/rational_polynomial.py` |
| Penalty weight `lambda` | 0.01 | `optim.coefficient_penalty` |
| AdamW weight decay (other parameters) | 0.1, the study's canonical value, or 0.01 as in the original specification; polynomial coefficients are never decayed | `optim.weight_decay` |

All runs: 200M, `d_K = 64`, QK-norm, transformer reference initialization, seed 0,
19,074 updates (10.0B tokens) budget. Runs and roles: [`selection.yaml`](selection.yaml).

## Runs

| Run | Denominator | Penalized denominator | Weight decay | Outcome |
| --- | --- | --- | ---: | ---: |
| `r9c52f40605fb` | signed degree 8 | yes | 0.1 | fail @131 |
| `re21d093cf4ac` | linear, learned constant | no | 0.1 | 15.206 |
| `r45c9067c8d64` | nonnegative quadratic, learned constant | no | 0.1 | 15.157 |
| `r817963a69c2c` | nonnegative degree 8 | yes | 0.01 | 15.288 |
| `rbde7fc4c2135` | softmax (control) | | 0.1 | 13.514 |
| `r7d911aee60ed` | normalized fixed Taylor, degree 8 (control) | | 0.1 | 13.543 |
| `r4923c6939e74` | normalized fixed Taylor, degree 8, Flax-default initialization (control) | | 0.1 | 14.498 |

Outcomes are terminal validation perplexity; for the three completed runs it is also the
validation best.

## Completed runs

The linear, nonnegative quadratic, and nonnegative degree-8 denominators all trained to
the end without a numerical failure, 1.6 to 1.8 above softmax.

| Validation PPL at update | 1,000 | 2,000 | 5,000 | 10,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear, learned constant | 53.93 | 30.32 | 20.28 | 17.03 | 15.206 |
| Nonnegative quadratic, learned constant | 54.30 | 30.20 | 20.27 | 16.94 | 15.157 |
| Nonnegative degree 8 | 51.41 | 28.83 | 20.00 | 17.08 | 15.288 |
| Softmax | 54.08 | 26.48 | 17.83 | 15.08 | 13.514 |
| Fixed Taylor, degree 8 | 54.85 | 26.85 | 17.89 | 15.12 | 13.543 |

- **The degree-8 denominator led early and finished last.** It was 2.7 below softmax at
  update 1,000, 2.4 above it at 2,000, and 0.08 to 0.13 above the low-degree runs at the end.
- **Learned constants grew rather than shrinking the common scale:** median 1.19
  (linear) and 1.51 (quadratic), never below 0.69.
- **Low-degree denominators stayed close to their constant.** Over `|s| <= 3` they vary
  by a median 38% (linear) and 71% (quadratic) of it. The linear one is negative for
  some keys (down to −2.1), but row sums on the diagnostic sample stayed at least 0.88.
- **The degree-8 denominator's coefficients stayed below 0.003 but still matter.** Over
  `|s| <= 3` it varies by a median 2.9 times its constant (up to 16); its values come
  within 0.008 of zero, and row sums stayed at least 1.0.
- **No numerator is linear.** Within `|s| <= 3`, the terms beyond `a_1 s` outweigh it by
  a median factor of 6.2 (linear), 5.9 (quadratic), and 10 (degree 8). The penalty term
  ended at 5e-4 to 7e-4 nats.

## Signed degree-8 denominator

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
- `r817963a69c2c` changes both the denominator penalty and the weight decay against the
  other runs, so neither effect is isolated.
- No checkpoint precedes the degree-8 failure, so its failing update cannot be replayed.
- The Flax-default Taylor control is not an equal-initialization comparison; the
  softmax and reference-initialization Taylor controls are.

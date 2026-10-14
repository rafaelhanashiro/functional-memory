# Rational learned polynomial attention

Opened 2026-10-04. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) (a learned
normalized column of the degree chart) or discard.

Every normalized learned polynomial with one signed polynomial failed when a row sum
crossed zero (degree chart, learned column: fail at updates 73 to 1,395). Does a
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
19,074 updates (10.0B tokens) budget with 400 warmup updates. Runs and roles:
[`selection.yaml`](selection.yaml).

## Runs

| Run | Denominator | Penalized denominator | Weight decay | Outcome |
| --- | --- | --- | ---: | ---: |
| `r90f6452bba82` | signed degree 8 | yes | 0.1 | fail @73 |
| `rc8aeb26cf795` | signed degree 8 | yes | 0.01 | fail @44 |
| `rd5ac0e3013e3` | linear, learned constant | no | 0.1 | 15.159 |
| `r05cd33157d94` | nonnegative quadratic, learned constant | no | 0.1 | 15.132 |
| `rbafa372b4e8b` | nonnegative degree 8 | yes | 0.01 | **15.006** |
| `rdf00da0866a8` | softmax (control) | | 0.1 | 13.731 |
| `r10baef5d8428` | normalized fixed Taylor, degree 8 (control) | | 0.1 | 13.741 |

Outcomes are terminal validation perplexity; for the three completed runs it is also the
validation best.

## Completed runs

The linear, nonnegative quadratic, and nonnegative degree-8 denominators all trained to
the end without a numerical failure, 1.3 to 1.4 above softmax.

| Validation PPL at update | 1,000 | 2,000 | 5,000 | 10,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Linear, learned constant | 32.66 | 24.60 | 19.35 | 16.71 | 15.159 |
| Nonnegative quadratic, learned constant | 32.50 | 24.75 | 19.16 | 16.68 | 15.132 |
| Nonnegative degree 8 | 31.31 | 23.63 | 18.94 | 16.50 | 15.006 |
| Softmax | 29.79 | 22.12 | 17.43 | 15.12 | 13.731 |
| Fixed Taylor, degree 8 | 30.16 | 22.28 | 17.47 | 15.14 | 13.741 |

- **The degree-8 denominator leads the rational runs throughout** and finishes 0.13 and
  0.15 below the quadratic and linear ones, which end within 0.03 of each other. All
  three trail softmax at every update, by 1.5 to 2.9 at update 1,000.
- **Learned constants grew rather than shrinking the common scale:** median 1.19
  (linear) and 1.48 (quadratic), never below 0.64.
- **Low-degree denominators vary moderately.** Over `|s| <= 3` they vary by a median 0.72
  (linear) and 1.1 (quadratic) times their constant. The linear one is negative for some
  keys (down to −2.1 on the diagnostic sample), but row sums there stayed at least 0.74;
  the quadratic one stays at least 0.33, with row sums at least 0.56.
- **The degree-8 denominator's coefficients stay below 0.0031 but still matter.** Over
  `|s| <= 3` it varies by a median 3.4 times its constant (up to 13); on the diagnostic
  sample its values come within 0.001 of zero, and row sums stayed at least 0.97.
- **No numerator is linear.** Within `|s| <= 3`, the terms beyond `a_1 s` outweigh it by
  a median factor of 5.8 (linear), 4.8 (quadratic), and 11 (degree 8). The penalty term
  ended at 5.0e-4 to 6.6e-4 nats.

## Signed degree-8 denominator

Both signed degree-8 runs fail within the first 75 updates, at update 73 (weight decay
0.1) and 44 (weight decay 0.01), with the forward loss, gradient norm, and updated
parameters all NaN in the same update. The training NLL fell from 10.58 to 9.00 by update
30 in both (within 0.002 of each other), then rose to 9.4–9.5 at update 40 and to 9.70 at
update 70 in the run that got that far, while softmax fell to 8.20 and 7.02.

- **A zero denominator sum, inferred.** QK-norm bounds the scores and Adam bounds how far
  each coefficient moves, so the denominator cannot overflow; with bounded scores and
  coefficients, the only forward NaN the kernel produces is an exactly zero row sum. The
  failure state is NaN throughout, so the sum itself was not measured.
- **High-order terms can set the sign.** At `|s| = 5`, `s^8 = 3.9e5`: a degree-8
  coefficient of 1e-5 already adds ±3.9 to a denominator whose constant is 1.
- **`lambda = 0.01` does not prevent it,** in either run.

## Limits

- One seed, one numerator degree, one penalty weight, one coefficient initialization.
- `rbafa372b4e8b` changes both the denominator penalty and the weight decay against the
  other completed runs, so neither effect is isolated.
- The two signed runs ran on 4 H100s, the degree-8 run on 4 H100s; the linear run moved
  from 4 to 2 H100s and the quadratic run from 4 H100s to 2 B300s and then 2 H100s after
  preemptions (same global batch).
- No checkpoint precedes either signed degree-8 failure, so the failing updates cannot be
  replayed.

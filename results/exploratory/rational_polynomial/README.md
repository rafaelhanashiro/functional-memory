# Rational learned polynomial attention

Opened 2026-10-04. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) (a learned
normalized column of the degree chart) or discard.

Every normalized learned polynomial with one signed polynomial failed when a row sum
crossed zero (degree chart, learned column: fail at updates 27 to 1,857). Does a
separate denominator polynomial, regularized toward low order, train through?
The kernel is

```text
attn_ij = p_a(s_ij) / sum_(k<=i) p_b(s_ik),   p_a(s) = sum_(r<=8) a_r s^r,   p_b(s) = 1 + sum_(1<=r<=8) b_r s^r
```

with cosine (QK-norm) scores, coefficients per layer and head, and the training loss

```text
NLL + lambda * sum over layers and heads [ a_0^2 + sum_(r>=2) r^2 a_r^2 + sum_(r>=1) r^2 b_r^2 ]
```

(`a_1` unpenalized). The denominator stays signed: an exactly zero row sum is a
numerical failure, recorded as such. The validation loss and the recorded training NLL
exclude the penalty; `metrics.jsonl` records the penalty term separately
([scaffold](../../../scaffolds/transformer/README.md#attention-and-parameter-ownership)).

Settings chosen for this first run, each a candidate for tuning later:

| Setting | Value | Recipe field |
| --- | --- | --- |
| Initialization | linear attention: `a_1 = 1`, every other coefficient 0, so attention starts at `sum_(j<=i) s_ij v_j / (i + 1)` | `model.attention_coefficient_initialization: linear` |
| Penalty weight `lambda` | 0.01 | `optim.coefficient_penalty` |
| AdamW weight decay (other parameters) | 0.1, the study's canonical value; polynomial coefficients are never decayed | `optim.weight_decay` |

All runs: 200M, `d_K = 64`, QK-norm, transformer reference initialization, seed 0,
19,074 updates (10.0B tokens) budget. Runs and roles: [`selection.yaml`](selection.yaml).

## Result

The run failed at update 131 (67M tokens, 0.7% of the budget) with a nonfinite loss:
a numerical failure, as the shared signed polynomials did at updates 27 to 1,857.
The loss had already turned: the training NLL fell from 10.58 to 8.57 at update 70,
then rose to 10.18 by update 130. The penalty term was 1.1e-4 nats at update 130,
negligible next to the NLL.

| Run | Kernel | Outcome |
| --- | --- | ---: |
| `r9c52f40605fb` | rational, degree 8, `lambda = 0.01`, linear-attention start | fail @131 |
| `rbde7fc4c2135` | softmax (control) | 13.514 |
| `r7d911aee60ed` | normalized fixed Taylor, degree 8 (control) | 13.543 |
| `r4923c6939e74` | normalized fixed Taylor, degree 8, Flax-default initialization (control) | 14.498 |

Evidence from the acceptance pilot at update 20 (`numerical_failure_step_00000131.json`
holds only the poisoned post-update state):

- Every coefficient had moved by at most 2.9e-5 from its start (Adam moves each
  coefficient by about the learning rate per update, whatever its gradient).
- Cosine scores already spanned about ±5 (QK-norm scale 8), so `s^8` reaches 4e5 and
  those 1e-5-sized high-order terms dominated both polynomials: denominators ranged
  from −9.8 to 11.9 and numerators from −23 to 16 within one layer, while the row sums
  stayed near their initial values (1 to 2,050) only because most scores are small.
- By update 130 the coefficients were of order 1e-3 (penalty / lambda = 0.011 summed
  over 160 polynomials), a thousand times 1/8! = 2.5e-5, and the signed denominator
  sums crossed zero.

What this says about the three settings chosen for tuning:

- **The penalty at `lambda = 0.01` cannot act.** Its gradient on `b_8` is
  `2 * lambda * 64 * b_8`, about 1e-3 at failure, against a task gradient scaled by
  `sum s^8`; under Adam the update direction is the task's. Raising `lambda` by several
  orders of magnitude, or normalizing the penalty to the score scale, would be needed
  for it to matter.
- **The linear-attention start does not keep the denominator positive.** Nothing
  constrains `p_b`, and the monomial basis in `s` with `|s|` up to 8 is so
  ill-conditioned at degree 8 that the first Adam updates already make the top terms
  the largest. Taylor's `1/r!` scaling is what keeps fixed degree-8 attention
  (13.543 with the same reference initialization) well-behaved.
- **Weight decay 0.1 is irrelevant here**: coefficients are never decayed.

Possible next steps, not started: scale learned coefficients as `c_r = c_r' / r!` (or
by the QK-norm scale to the power `r`) so an update of size `lr` moves every term
comparably; a positive denominator (sum-of-squares factors with the constant pinned);
a smaller learning rate or a lower initial QK-norm scale for the coefficients; a lower
degree first.

## Limits

- One seed, one degree, one penalty weight, one coefficient initialization.
- No checkpoint precedes the failure (the first resume checkpoint is at update 500),
  so the failing update cannot be replayed; the pilot's update-20 diagnostics are the
  only pre-failure attention measurements.
- The Flax-default Taylor control is not an equal-initialization comparison; the
  softmax and reference-initialization Taylor controls are.

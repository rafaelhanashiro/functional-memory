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
19,074 updates (10.0B tokens). Runs and roles: [`selection.yaml`](selection.yaml).

## Runs

| Run | Kernel | Terminal PPL |
| --- | --- | ---: |
| `r9c52f40605fb` | rational, degree 8, `lambda = 0.01`, linear-attention start | planned |
| `rbde7fc4c2135` | softmax (control) | 13.514 |
| `r4923c6939e74` | normalized fixed Taylor, degree 8, Flax-default initialization (control) | 14.498 |

The reference-initialization normalized Taylor degree-8 run (13.543) belongs to the
in-flight Taylor reference-initialization worktree and is not on `main` yet.

## Limits

- One seed, one degree, one penalty weight.
- The Taylor control uses Flax-default weights, so it is not an equal-initialization
  comparison; the softmax control is.

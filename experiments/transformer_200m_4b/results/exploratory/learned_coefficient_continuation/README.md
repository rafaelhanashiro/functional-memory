# Learned coefficients continued from Taylor and exp checkpoints (m = 4)

Opened: **2026-10-29**. Home cluster: **Kennel**.

Do learned degree-four coefficients, static or query-dependent, improve a pretrained model
when they are introduced after fixed-kernel pretraining, and does the answer depend on the
kernel the backbone grew up with? Two parents: the canonical fixed-Taylor model, whose
backbone is adapted to the degree-four Taylor kernel, and the unnormalized exp model, whose
backbone is adapted to the kernel the polynomial approximates. All runs use the canonical
QK RMSNorm with per-head scaling. The query branches decay their projection `W`
(`attention_coefficient_kernel_weight_decay: true`), which in pretraining brought query
coefficients from 14.78 to 14.71 against fixed Taylor's 14.72 (`query-m4-w-decay`,
`rca97cc9db8f2`). The runs live in the [200M / 4B study](../../../README.md) and enter no
canonical comparison.

## Runs and method

| Role | Run | Parent | Change at update 7,500 |
| --- | --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | Taylor `r816ab0fbe119` (14.7242) | None (reused from the query-scale topic) |
| Learned static | `r2e081e09c19a` | Taylor `r816ab0fbe119` | static coefficients per layer/head/order, Taylor at start |
| Learned query, decayed W | `rae91f24b06b4` | Taylor `r816ab0fbe119` | tanh query coefficients at zero amplitude, W decayed |
| Exp → learned static | `r7e6a68545d4f` | exp `r6a76e7fbde45` (14.2572) | degree-four kernel with static coefficients, Taylor at start |
| Exp → learned query, decayed W | `r673e14eab803` | exp `r6a76e7fbde45` | degree-four kernel with tanh query coefficients, W decayed |

Every branch resumes its parent's validation-best checkpoint at update **7,500** (weights,
Adam moments and clock, data cursor, RNG) and trains all parameters for **2,000** updates
on the same fresh data, seed, and 256-sequence batches as the control: backbone learning
rate **0.0001**, introduced parameters at **0.001** with zero Adam moments and their own
clock. Introduced parameters decay only where their declaration says so: the query
projection `W` here (`optim.weight_decay` 0.1), not biases, amplitudes, or static
coefficients. Evaluation uses all 19 held-out batches at the restored baseline and every
100 updates; terminal and validation-best metrics are reported separately.

The Taylor-parent branches start as exactly the parent's function. The exp-parent
branches swap the exp kernel for the degree-four Taylor polynomial on Mosaic at the start
(the parent ran the fused exp kernel): with scores of order 4, `p_4(s)` is a poor
approximation of `exp(s)`, so their restored baselines are far above 14.26 and the runs test
whether a backbone adapted to exp recovers to a better polynomial model than one adapted to
Taylor. They have no fixed-Taylor control of their own, so they are read against each other,
against the Taylor-parent pair, and against the parents' perplexities. The earlier degree-four
query continuation from an older QK-norm parent (500 updates at the backbone's learning rate,
tie at 14.686 against 14.689) was deleted with its runs on 2026-10-29.

Launch commands are in the [study README](../../../README.md#matched-taylor-continuations).

## Findings and open decision

All four branches completed their 2,000 updates on one GPU each (static-Taylor 84 min on a
`dogo` H100, query-Taylor 57 min on an `akita` B300, the exp branches 124 and 147 min on the
`mastiff` H100 PCIe; the acceptance pilot's slowest-step projection needed 20–48 h limits,
as in the earlier continuations). Every validation-best is the terminal checkpoint.

| Continuation | Restored PPL | Terminal PPL @ 9,500 | Δ vs control (nats/token) |
| --- | ---: | ---: | ---: |
| Fixed Taylor control | 14.7216 | 14.5064 | — |
| Taylor → learned static | 14.7217 | 14.4964 | −0.0007 |
| Taylor → learned query, decayed W | 14.7216 | 14.4869 | −0.0014 |
| Exp → learned static | 286.14 | 14.4483 | −0.0040 |
| Exp → learned query, decayed W | 286.14 | **14.4026** | **−0.0072** |

| Δ vs control (nats) at update | 100 | 500 | 1,000 | 1,500 | 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Taylor → static | −0.00004 | −0.0001 | −0.0004 | −0.0004 | −0.0007 |
| Taylor → query, decayed W | −0.00007 | −0.0004 | −0.0008 | −0.0011 | −0.0014 |
| Exp → static | +0.103 | +0.018 | +0.005 | −0.0007 | −0.0040 |
| Exp → query, decayed W | +0.095 | +0.014 | +0.002 | −0.0040 | −0.0072 |

**From the Taylor checkpoint**, learned coefficients now help, a little, with the canonical
QK norm and their own learning rate: static ends **0.07%** below the control and the
decayed-W query generator **0.13%**, both gaps growing monotonically through the budget.
These are about 15% and 30% of the forget gate's gain in the same protocol (−0.0045 nats) and
far above the protocol's numerical resolution (6e-6 nats between identical starts). The
static solution moves to the shape pretraining finds, `u_3` heavy and `u_1`, `u_2` light
(mean `u_r = r! c_r` over heads 0.94 / 0.87 / 0.56 / 1.31 / 0.87); the query generator's
biases take a similar shape (1.01 / 0.77 / 0.64 / 1.20 / 0.91) with amplitudes of mean
`|a_r|` 0.09–0.35 (maximum 1.79) and `||W||` grown to 1.24 despite the decay.

**From the exp checkpoint**, the kernel swap costs about one nat (PPL 286) at the start; the
models are within 10% of the control after 100 updates, cross it at updates 1,400–1,500,
and end **0.40%** (static) and **0.72%** (query, decayed W) below it: 14.4026 is the lowest
perplexity any continuation in this study has reached, below the forget-gate branch
(14.4417). Both exp-start solutions also take the `u_3`-heavy shape (static mean `u_r`
0.81 / 0.59 / 0.54 / 1.38 / 1.00). Without a fixed-Taylor continuation of the exp parent, the
share of this gain that belongs to the exp-adapted backbone (its parent is 3% better than
the Taylor parent) rather than to the learned coefficients is not identified; the difference
between the two exp branches, −0.0032 nats, does isolate query dependence with decayed `W`
in that regime, five times its effect from the Taylor start.

**Open decision:** allocate the fixed-Taylor continuation of the exp parent (one run) to
attribute the exp-start gain, or close the topic with these four. Discard or promote after
that decision.

Measurements live only in each selected run; `summary.csv` derives every selected
evaluation with the mean factorial-coordinate coefficients and the query generator's
amplitude and projection statistics, distinguishing terminal from validation-best.
Regenerate after intentionally incorporating new measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/learned_coefficient_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.results.exploratory.learned_coefficient_continuation.collect
```

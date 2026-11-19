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
`rca97cc9db8f2`). The runs live in the [Transformer++ study](../../../../README.md).

## Runs and method

| Role | Run | Parent | Change at update 7,500 |
| --- | --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | Taylor `r816ab0fbe119` (14.7242) | None (reused from the query-scale topic) |
| Learned static | `r2e081e09c19a` | Taylor `r816ab0fbe119` | static coefficients per layer/head/order, Taylor at start |
| Learned query, decayed W | `rae91f24b06b4` | Taylor `r816ab0fbe119` | tanh query coefficients at zero amplitude, W decayed |
| Exp → learned static | `r7e6a68545d4f` | exp `r6a76e7fbde45` (14.2572) | degree-four kernel with static coefficients, Taylor at start |
| Exp → learned query, decayed W | `r673e14eab803` | exp `r6a76e7fbde45` | degree-four kernel with tanh query coefficients, W decayed |
| Exp → fixed Taylor (control) | `rddcbe6dfb1eb` | exp `r6a76e7fbde45` | degree-four fixed Taylor kernel, no learned coefficients |

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
Taylor. Their own control, fixed Taylor continued from the exp checkpoint (allocated after the four
branches, at the user's request), starts as the same function and keeps its coefficients
fixed, so it attributes the exp-start gains between the exp-adapted backbone and the learned
coefficients. The earlier degree-four
query continuation from an older QK-norm parent (500 updates at the backbone's learning rate,
tie at 14.686 against 14.689) was deleted with its runs on 2026-10-29.

Launch commands are in the [study README](../../../../README.md#matched-taylor-continuations).

## Continuation of the pretrained learned models

Two controls extend models whose coefficients were learned from the start, using
their step-7,500 checkpoints and the same 2,000-update tail through step 9,500:

| Role | Run | Parent | Parent PPL at 7,500 |
| --- | --- | --- | ---: |
| Static → static | `r1522cd56901a` | `r4339a245820e` | 14.7667 |
| Query + W decay → query + W decay | `rdfc42659424f` | `rca97cc9db8f2` | 14.7085 |

They preserve every trained array and its Adam moments/count, including the
coefficients and query projection. No parameters are introduced or reinitialized.
Both use a constant **0.0001 for all parameters**, no warmup, the same seed and
data order, global batch 256, 2,048-token context, Mosaic, and 19 validation batches
every 100 updates. Projection decay and its exemptions are inherited unchanged;
query `W` decays at 0.1 and coefficient amplitudes/biases remain exempt. The input
cursor is sequence 1,920,000; 2,000 updates consume 1,048,576,000 prediction tokens.
These runs belong to Kennel and use one GPU each.

This compares equal-budget continuation recipes with different pretraining histories.
The earlier Taylor/exp branches gave **newly introduced** coefficients a 0.001 LR
and fresh Adam state; these controls retain their trained coefficients' history and
use 0.0001. Consequently this is not an isolated test of coefficient-learning onset
at an identical coefficient LR. It also differs from training with a cosine schedule
stretched to 9,500 steps.

The [study launch instructions](../../../../README.md#continuing-pretrained-static-and-query-models)
pin preparation and submission commands. Both use
`continuation.preserve_all_parameters: true`; the shared acceptance checks verify
exact restoration, data/RNG continuity, finite GPU training, and checkpoint resume.

## Findings and open question

The five branches of the first table completed their 2,000 updates on one GPU each
(static-Taylor 84 min on a `dogo` H100, query-Taylor 57 min on an `akita` B300, the exp
branches 124 and 147 min on the `mastiff` H100 PCIe, the exp control 83 min on `dogo`; the
acceptance pilot's slowest-step projection needed 20–48 h limits, as in the earlier
continuations). Every validation-best is the terminal checkpoint.

| Continuation | Restored PPL | Terminal PPL @ 9,500 | Δ vs control (nats/token) |
| --- | ---: | ---: | ---: |
| Fixed Taylor control | 14.7216 | 14.5064 | — |
| Taylor → learned static | 14.7217 | 14.4964 | −0.0007 |
| Taylor → learned query, decayed W | 14.7216 | 14.4869 | −0.0014 |
| Exp → learned static | 286.14 | 14.4483 | −0.0040 |
| Exp → learned query, decayed W | 286.14 | **14.4026** | **−0.0072** |
| Exp → fixed Taylor (control) | 286.14 | 14.5606 | +0.0037 |

| Δ vs control (nats) at update | 100 | 500 | 1,000 | 1,500 | 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Taylor → static | −0.00004 | −0.0001 | −0.0004 | −0.0004 | −0.0007 |
| Taylor → query, decayed W | −0.00007 | −0.0004 | −0.0008 | −0.0011 | −0.0014 |
| Exp → static | +0.103 | +0.018 | +0.005 | −0.0007 | −0.0040 |
| Exp → query, decayed W | +0.095 | +0.014 | +0.002 | −0.0040 | −0.0072 |
| Exp → fixed Taylor | +0.111 | +0.028 | +0.014 | +0.007 | +0.0037 |

| Δ vs the exp parent's own fixed-Taylor control (nats) | 100 | 500 | 1,000 | 1,500 | 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp → static | −0.0084 | −0.0103 | −0.0090 | −0.0080 | **−0.0077** |
| Exp → query, decayed W | −0.0168 | −0.0140 | −0.0116 | −0.0114 | **−0.0109** |

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
0.81 / 0.59 / 0.54 / 1.38 / 1.00).

**Attribution.** The exp parent's own fixed-Taylor continuation, the same start with the
coefficients held at Taylor, ends at **14.5606**: 0.37% *above* the Taylor-backbone control.
Two thousand updates do not carry an exp-adapted backbone with the fixed quartic past a
Taylor-adapted one, so none of the exp-start gain is the backbone's. All of it, and more, is
the learned coefficients: **−0.0077 nats** (static) and **−0.0109 nats** (query, decayed W)
against their own control, five to eight times their effect from the Taylor start. The
advantage is present at the first evaluation (−0.008 and −0.017 nats at update 100) and then
roughly constant: on a backbone whose Q/K statistics were shaped by exp, the fixed Taylor
quartic is the wrong kernel, and learned coefficients repair that mismatch at once rather than
accumulating a slow gain. Query dependence with decayed `W` adds −0.0032 nats over static
coefficients in that regime.

**Coefficients learned from the start.** The static and decayed-W query continuations (59 min
on an `akita` B300, 152 min on the `mastiff` H100 PCIe) match their parents' step-7,500
validation to within 3e-6 nats at update 0, and each validation-best is its terminal checkpoint.

| Best PPL @ 9,500 | Static | Query, decayed W |
| --- | ---: | ---: |
| Learned from the start, continued | 14.5559 (`r1522cd56901a`) | 14.4986 (`rdfc42659424f`) |
| Introduced at 7,500 on Taylor | 14.4964 (`r2e081e09c19a`) | 14.4869 (`rae91f24b06b4`) |
| Introduced at 7,500 on exp | 14.4483 (`r7e6a68545d4f`) | **14.4026** (`r673e14eab803`) |
| Fixed Taylor control (`r82203a85ef8e`) | 14.5064 | 14.5064 |

- Against the fixed-Taylor control, the from-start static model stays 0.0031–0.0036 nats behind
  through the tail (+0.0034 at the end); the from-start query model stays 0.0003–0.0010 nats
  ahead (−0.0005 at the end), a constant 0.0040 nats below static.
- At the same budget, coefficients introduced at 7,500 end below the from-start ones: by 0.0041
  (static) and 0.0008 nats (query, overtaking at update 1,200) from the Taylor start, and by
  0.0074 and 0.0066 nats from the exp start.
- At 0.0001 the trained coefficients barely move (mean `u_r` within 0.025, query `||W||`
  1.34 → 1.32), and both models gain 1.43% PPL over the tail, no more than the fixed control
  (1.46%). Their learning rate and Adam state differ from the introduced branches' (above), so
  this does not isolate when coefficients start learning.

**Open question:** whether to combine it with the forget-gate and query-scale continuations into
one account of what continued pretraining can and cannot add, or delete the topic.

Measurements live only in each selected run; `summary.csv` derives every selected
evaluation with the mean factorial-coordinate coefficients and the query generator's
amplitude and projection statistics, distinguishing terminal from validation-best.
Regenerate after intentionally incorporating new measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/diagnostics/continued_pretraining/learned_coefficient_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.diagnostics.continued_pretraining.learned_coefficient_continuation.collect
```

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

The four branches are allocated and not yet complete.

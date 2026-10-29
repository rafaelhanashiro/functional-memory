# Constant ALiBi-style decay after Taylor pretraining

Opened: **2026-10-29**. Home cluster: **Kennel**.

The [forget-gate continuation](../forget_gate_continuation/README.md) adds a span-dependent
factor to every attention weight, `prod_(j=i+1..t) alpha_j` with `log alpha_j` read from
token `j`, and the gate learned a sparse reset: `log alpha_j` near zero on most tokens and
strongly negative on a few. Is the token dependence the ingredient, or does any learnable
decay with distance do the same? This topic replaces the gate by a constant per layer and
head, `log alpha_t = -lambda_h`, so every weight is multiplied by `exp(-lambda_h (t - i))`:
ALiBi's linear bias in log space, with `lambda_h` learned. Everything else matches the
forget-gate branch. The run lives in the [200M / 4B study](../../../README.md) and enters
no canonical comparison.

## Runs and method

| Role | Run | Change |
| --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | None (reused from the query-scale topic) |
| Query coefficients + constant decay | `rfe7957b8f530` | `attention_forget_gate_form: constant` |
| Query coefficients + forget gate (reference) | `rd5ea49c1fde6` | token-dependent gate, in its own topic |

Both continue the canonical fixed-Taylor degree-four parent `r816ab0fbe119` from its
validation-best checkpoint at update **7,500** (held-out PPL **14.7242**) for **2,000**
updates with the same seed, data order, and 256-sequence batch. The branch adds the
zero-amplitude tanh query coefficients of the forget-gate run (the polynomial starts as
Taylor) and one learned log rate `A_h` per layer and head (**210 parameters**), with
`lambda_h = exp(A_h)` starting at ALiBi's geometric slopes `2^(-8 h / 14)` for heads
`h = 1..14`, the same in every layer: 0.67 down to 0.0039, half-lives from about one token to
177. This brackets the pretraining forget gate's median half-life of 2 tokens and, like the
gate's open initialization, perturbs the Taylor-adapted model from the first step. The
blocked reference backend evaluates the decay, as for the gate.

The backbone keeps its pretrained Adam moments and clock at learning rate **0.0001**; the
introduced parameters have zero Adam moments, their own clock, a constant learning rate of
**0.001**, and no weight decay. Evaluation uses all 19 held-out batches at the restored
baseline and every 100 updates; terminal and validation-best metrics are reported
separately. Launch commands are in the
[study README](../../../README.md#matched-taylor-continuations).

## Findings and open decision

The branch is allocated and not yet complete.

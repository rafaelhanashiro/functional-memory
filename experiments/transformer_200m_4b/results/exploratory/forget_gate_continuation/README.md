# Does the matched continuation detect the forget gate?

Opened: **2026-10-28**. Home cluster: **Kennel**.

The matched-continuation protocol (continue a pretrained fixed-Taylor checkpoint with and
without a new mechanism, same data and seed) has returned ties for learned static
coefficients, query-dependent coefficients, and the
[per-token query scale](../token_query_scale_continuation/README.md). Is the protocol
sensitive at all? The [forget gate](../forget_gate/README.md) is the one degree-four
mechanism known to beat fixed Taylor in full 4B pretraining (14.320 against 14.724, 2.7%).
If its continuation shows no gain either, the protocol, or the Taylor-adapted backbone
it starts from, cannot be trusted to rank mechanisms; if it shows a clear gain, the earlier
ties are informative. This topic lives in the [200M / 4B study](../../../README.md) and
enters no canonical comparison.

## Runs and method

| Role | Run | Change |
| --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | None (reused from the query-scale topic) |
| Query coefficients + forget gate | `rd5ea49c1fde6` | The pretraining winner's attention |

Both continue the canonical fixed-Taylor degree-four parent `r816ab0fbe119` from its
validation-best checkpoint at update **7,500** (held-out PPL **14.7242**) for **2,000**
updates with the same seed, data order, and 256-sequence batch; the control is the
completed run from the per-token query scale comparison, which has the same parent, budget,
and learning rate. The branch adds exactly the attention of the pretraining winner
`r111843b4c7cc`: tanh query coefficients `c_r(x) = (b_r + a_r tanh(W_r x)) / r!` at zero
amplitude (the polynomial starts as Taylor) and Gated DeltaNet's forget gate, which
multiplies the weight of key `i` for query `t` by `prod_(j=i+1..t) alpha_j` with
`log alpha_t = -exp(A_log) softplus(w_alpha^T x_t + b_dt)` per layer and head. The gate keeps
Gated DeltaNet's initialization, `exp(A_log) ~ U(0, 16)`, `softplus(b_dt)` log-uniform on
`[0.001, 0.1]`, `w_alpha ~ U(+-1/sqrt(896))`: it starts **open**, with random short-range
decay, not at the identity. A closed start has no gradient, and Adam could move the
log-rate by at most about two over the budget, so the gate could never reach the rates the
pretrained model uses. The restored model therefore starts above the parent's loss, and
the question is whether it recovers and overtakes the control. Mosaic has no forget gate,
so the branch runs on the blocked reference backend, like the pretraining winner.

The backbone keeps its pretrained Adam moments and clock at learning rate **0.0001**. The
introduced parameters (942,900 for the coefficients, 188,580 for the gate) have zero Adam
moments, their own clock, a constant learning rate of **0.001**, and no weight decay (the
pretraining run decayed `w_alpha`). Evaluation uses all 19 held-out batches at the restored
baseline and every 100 updates; terminal and validation-best metrics are reported
separately. Launch commands are in the
[study README](../../../README.md#matched-taylor-continuations).

## Findings and open decision

The branch is allocated and not yet complete.

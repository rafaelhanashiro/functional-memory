# Does the matched continuation detect the forget gate?

Opened: **2026-10-28**. Home cluster: **Kennel**.

The matched-continuation protocol (continue a pretrained fixed-Taylor checkpoint with and
without a new mechanism, same data and seed) has returned ties for learned static
coefficients, query-dependent coefficients, and the
[per-token query scale](../token_query_scale_continuation/README.md). Is the protocol
sensitive at all? The [forget gate](../forget_gate/README.md) improved on fixed Taylor
in full Transformer++ pretraining (terminal PPL 14.320 against 14.724, 2.7%).
If its continuation shows no gain either, the protocol, or the Taylor-adapted backbone
it starts from, cannot be trusted to rank mechanisms; if it shows a clear gain, the earlier
ties are informative. This topic lives in the [Transformer++ study](../../../README.md) and
enters no canonical comparison.

## Runs and method

| Role | Run | Change |
| --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | None (reused from the query-scale topic) |
| FoX with RoPE | `rd5ea49c1fde6` | Query coefficients + forget gate; no decay on introduced parameters |
| FoX without RoPE + projection decay | `r554ce7da6a8a` | Remove RoPE; decay coefficient `W` and gate `w_alpha` at 0.1 |

All three continue the canonical fixed-Taylor degree-four parent `r816ab0fbe119` from its
validation-best checkpoint at update **7,500** (held-out PPL **14.7218**) for **2,000**
updates with the same seed, data order, and 256-sequence batch; the control is the
completed run from the per-token query scale comparison, which has the same parent, budget,
and learning rate. The original RoPE branch adds the attention of the pretraining run
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

The backbone keeps its pretrained Adam moments and clock at learning rate **0.0001**.
In the original RoPE branch, the introduced parameters (942,900 for the coefficients,
188,580 for the gate) have zero Adam
moments, their own clock, a constant learning rate of **0.001**, and no weight decay (the
pretraining run decayed `w_alpha`). Evaluation uses all 19 held-out batches at the restored
baseline and every 100 updates; terminal and validation-best metrics are reported
separately. Launch commands are in the
[study README](../../../README.md#matched-taylor-continuations).

## Findings and open decision

The no-RoPE, decayed-projection branch **`r554ce7da6a8a`** tests the FoX configuration
that achieved the best degree-four polynomial pretraining PPL. It uses the same Taylor
parent at update 7,500 and the same 2,000-update budget, data order, seed, evaluation,
and learning rates as the control: 0.0001 for the restored backbone, 0.001 for all new
parameters. It removes RoPE, keeps full factorial scaling of the query coefficients,
and applies weight decay **0.1** to both the coefficient projection `W` and the forget
gate projection `w_alpha`. Coefficient `a,b`, gate `A_log,b_dt`, and normalization gains
remain exempt. Both the backbone parameters and Adam history are restored exactly;
removing RoPE and introducing an open gate intentionally changes the initial function.
This is a comparison of complete FoX recipes, not an isolated weight-decay ablation
against the earlier RoPE branch. The existing fixed-Taylor control is reused.

The no-RoPE branch completed all **2,000** updates on one H100 PCIe on Kennel. The frozen
configuration is in [its run directory](../../../runs/r554ce7da6a8a/config.yaml); preparation and submission
commands are in the [study README](../../../README.md#cpt-with-projection-weight-decay).

The original RoPE branch completed all **2,000** updates on one H100 (Kennel `mastiff`,
`long` partition, job **228802**). The blocked backend trained at **16.7 s per update**;
the trainer ran **9 h 57 min**.

| Continuation | Restored PPL | Terminal PPL | Validation-best PPL / step |
| --- | ---: | ---: | --- |
| Fixed Taylor control | 14.7216 | 14.5064 | 14.5064 / 9,500 |
| FoX with RoPE, no decay on introduced parameters | 40.9385 | **14.4417** | 14.4417 / 9,500 |
| FoX without RoPE, coefficient/gate projection decay | 158.7425 | 14.7630 | 14.7630 / 9,500 |

| Update | 0 | 100 | 200 | 500 | 1,000 | 1,500 | 2,000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FoX with RoPE − control (nats/token) | +1.0228 | +0.0034 | −0.0005 | −0.0025 | −0.0034 | −0.0039 | **−0.0045** |
| FoX without RoPE + decay − control (nats/token) | +2.3780 | +0.0656 | +0.0470 | +0.0327 | +0.0245 | +0.0199 | **+0.0175** |

For the original RoPE branch, the open initialization costs about one nat at the restored
baseline; the model recovers within 100 updates, overtakes the control by update 200, and its lead grows monotonically
to **0.0647 PPL (0.45%)**, or **0.00447 nats/token**, at the end: roughly 16% of the full
pretraining gap after a quarter of the pretraining tokens at the schedule's final learning
rate. This RoPE comparison detects a gain, but does not establish that continuation
will rank other recipes as full pretraining does.

The no-RoPE recipe starts at **158.7425 PPL**, falls to **15.7048** after 100 updates,
and continues recovering through update 2,000. It remains above the control at every
evaluation and ends **0.2566 PPL** (**0.017534 nats/token**) worse than fixed Taylor,
and **0.3213 PPL** worse than the original RoPE FoX branch. Its pretraining advantage
therefore did not transfer within this continuation budget. The larger initial loss is
measured before any decay update: removing RoPE from the restored backbone changes its
function immediately. The final comparison changes RoPE, coefficient-projection decay,
and forget-gate-projection decay together, so it cannot identify which change explains
the worse result or establish that weight decay itself hurts.

In the original RoPE branch, what the gate learned differs from the pretraining winner's
short-range forgetting. At update 9,500 the per-head mean `log alpha` has median **−0.0041** per token (half-lives of
46 / 168 / 367 tokens at the 10th / 50th / 90th percentiles, against a median of 2.2 tokens
in pretraining), but the distribution over tokens is very skewed: the most negative token
per head has median `log alpha = −0.97` and reaches **−83** in some heads. The gate is
near the identity on most tokens and fires hard on a few, segmenting the context at
content-selected boundaries. Against this, the ungated Taylor kernel on the same parent
puts more than half of its attention mass beyond 256 tokens for late queries (the Taylor
quartic never drops below 0.27, so every distant key keeps a floor weight); the resets remove
that mass. The query coefficients also opened further than in the earlier query
continuation (mean `|a_r|` **0.18**, maximum **1.58**, at ten times its learning rate).

Whether the token dependence of these resets matters, or a learned constant decay
would suffice, remains untested by these continuations.

**Open decision:** retain or discard this topic, or isolate the positional-encoding
and projection-decay effects before drawing a mechanistic conclusion. These results
alone do not establish how either recipe would behave with a longer continuation.

Measurements live only in each selected run; `summary.csv` derives every selected
evaluation with the gate's half-life and per-token statistics, distinguishing terminal from
validation-best. Regenerate after intentionally incorporating new measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/exploratory/forget_gate_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.forget_gate_continuation.collect
```

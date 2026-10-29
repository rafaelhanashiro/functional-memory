# ALiBi-style constant decay without RoPE (m = 4)

Opened: **2026-10-29**. Home cluster: **Kennel**.

The [forget gate](../forget_gate/README.md) multiplies every attention weight by a factor
that depends on the tokens between key and query, `prod_(j=i+1..t) alpha_j`, and is the one
degree-four mechanism that beats fixed Taylor in pretraining; its
[continuation](../forget_gate_continuation/README.md) learned sparse, content-selected
resets. Is the token dependence the ingredient, or does any learnable decay with distance
do the same? This topic replaces the gate by a constant per layer and head,
`log alpha_t = -lambda_h`, so every weight is multiplied by `exp(-lambda_h (t - i))`: ALiBi's
linear bias in log space, with `lambda_h` learned. Like the forget-gate run it compares
with, it uses **no positional encoding**, so the decay is the only source of position.
The runs live in the [200M / 4B study](../../../README.md) and enter no canonical
comparison.

## Runs

| Role | Run | Label | Position |
| --- | --- | --- | --- |
| Primary: constant decay | `r8d4fc24ef537` | `query-m4-alibi-no-rope` | decay only |
| Control: forget gate | `r82a80d3091e2` | `query-m4-forget-gate-no-rope` | gate only |
| Reference: forget gate | `r111843b4c7cc` | `query-m4-forget-gate` | RoPE + gate |
| Reference: fixed Taylor | `r816ab0fbe119` | `fixed-m4-head-scale` | RoPE |

The primary copies the control's recipe and changes one setting,
`attention_forget_gate_form: constant`: degree-four tanh query coefficients
`c_r(x) = (b_r + a_r tanh(W_r x)) / r!` at zero amplitude (Taylor at initialization),
canonical QK RMSNorm with per-head scaling, head RMSNorm output scaling, the blocked
reference backend (Mosaic has no decay), and the study's 4B recipe: 7,570 updates of 256
sequences, learning rate 0.001 with cosine decay to 0.0001, warmup 455, weight decay 0.1.
The decay adds one learned log rate `A_h` per layer and head (**210 parameters**, no weight
decay), with `lambda_h = exp(A_h)` starting at ALiBi's geometric slopes `2^(-8 h / 14)` for
heads `h = 1..14`, the same in every layer: 0.67 down to 0.0039, half-lives from about one
token to 177. The gate control instead has Gated DeltaNet's token-dependent gate with its
standard initialization (188,580 parameters).

Launch commands are in the [study README](../../../README.md#alibi-style-constant-decay-without-rope).

## Findings and open decision

The primary is allocated and not yet complete; the control is training.

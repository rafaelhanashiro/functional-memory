# ALiBi-style distance decay without RoPE

Opened: **2026-10-29**. Home cluster: **Kennel**.

The [forget gate](../forget_gate/README.md) multiplies every attention weight by a factor
that depends on the tokens between key and query, `prod_(j=i+1..t) alpha_j`, and gives the
largest degree-four gain over fixed Taylor in 1× pretraining; its
[continuation](../forget_gate_continuation/README.md) learned sparse, content-selected
resets. Is the token dependence the ingredient, or does any learnable decay with distance
do the same? This topic replaces the gate by a constant per layer and head,
`log alpha_t = -lambda_h`, so every weight is multiplied by `exp(-lambda_h (t - i))`: ALiBi's
linear bias in log space, with `lambda_h` learned. Like the forget-gate run it compares
with, it uses **no positional encoding**, so the decay is the only source of position.
A second comparison gives learned **static** coefficients a learned decay that reads no token
([input-free gates](#learned-input-free-gates-for-static-coefficients)), shared or per
polynomial order, at degrees four and eight. The runs live in the
[Transformer++ study](../../../README.md) and enter no canonical comparison.

## Constant decay for query coefficients

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
reference backend (Mosaic has no decay), and the study's 1× recipe: 7,570 updates of 256
sequences, learning rate 0.001 with cosine decay to 0.0001, warmup 455, weight decay 0.1.
The decay adds one learned log rate `A_h` per layer and head (**210 parameters**, no weight
decay), with `lambda_h = exp(A_h)` starting at ALiBi's geometric slopes `2^(-8 h / 14)` for
heads `h = 1..14`, the same in every layer: 0.67 down to 0.0039, half-lives from about one
token to 177. The gate control instead has Gated DeltaNet's token-dependent gate with its
standard initialization (188,580 parameters).

Launch commands are in the [study README](../../../README.md#alibi-style-constant-decay-without-rope).

### Findings

The primary completed its 7,570 updates on one B300 (Kennel `akita`, `long`; job 229720,
11 h 26 min). The control completed on the same node.

| Run | Position | Terminal PPL | Validation-best PPL / step | vs fixed Taylor |
| --- | --- | ---: | --- | ---: |
| Fixed Taylor `r816ab0fbe119` | RoPE | 14.724 | 14.722 / 7,500 | — |
| Forget gate `r111843b4c7cc` | RoPE + gate | 14.320 | 14.318 / 7,500 | −2.7% |
| **Constant decay** `r8d4fc24ef537` | decay only | **14.485** | 14.482 / 7,500 | **−1.6%** |
| **Forget gate** `r82a80d3091e2` | gate only | **13.908** | 13.906 / 7,500 | **−5.5%** |

| Validation PPL at update | 1,000 | 2,500 | 5,000 | 7,500 |
| --- | ---: | ---: | ---: | ---: |
| Constant decay | 25.61 | 18.85 | 15.77 | 14.48 |
| Forget gate, no RoPE | 23.90 | 17.89 | 15.12 | 13.91 |

- **A learnable decay with distance helps on its own**: the constant decay beats fixed Taylor
  with RoPE by 1.6% while carrying no positional encoding at all.
- **But it recovers only about 30% of the gate's gain.** The token-dependent gate without RoPE
  reaches 13.91, 4.0% below the constant decay, with the gap present from update 1,000 and
  roughly constant afterwards. The span dependence, not the decay per se, is the main
  ingredient, consistent with the sparse content-selected resets the gate learned in the
  continuation.
- **Removing RoPE helps the gate**: 13.91 without against 14.32 with, below unnormalized exp
  with RoPE (`r6a76e7fbde45`, 14.26). At the 1× budget and 2,048-token context, only the same
  recipe with projection weight decay (`r4a80e28ce0a7`, validation-best 13.873, terminal
  13.875) is lower among degree-four models. RoPE and the gate both carry position; the gate
  alone carries it better.
- **What the decay learned.** At the end, `lambda_h` has per-head medians over layers close to
  the ALiBi initialization for the fast heads (1.15 against 0.67, 0.62 against 0.45, 0.44
  against 0.30) and below it for the slow heads (0.002 against 0.004–0.006), with half-lives
  of **1.0 / 19 / 303** tokens at the 10th / 50th / 90th percentiles: 21% of heads forget within
  two tokens, 38% retain beyond 64, 4% beyond 512. Per-layer median half-lives are 12–26
  tokens except layers 3 (51), 4 (31) and 14 (86). The query coefficients opened to mean
  `|a_r|` 0.18 (maximum 0.97).

## Learned input-free gates for static coefficients

Added 2026-11-09. Does a learned distance decay help learned static coefficients without RoPE,
as the query-distance gates help query coefficients (degree-four query + W decay: validation-best
PPL 14.709 with RoPE, `rca97cc9db8f2`, against 14.198 with the gate and no positional encoding,
`r3e26412b2811`)? Four runs replace RoPE in the static controls by the query-distance gate of
`r3e26412b2811` (shared) and `r507193d47450` (per order) with its input projection removed. The
rate reads no token:

`lambda = exp(A_log) softplus(dt_bias)` per layer and head (shared gate), or per layer, head
and order (per-order gates), and the weight from key `i` to query `t` is
`sum_r c_r s_ti^r exp(-(t-i) lambda)` (per order: `exp(-(t-i) lambda_r)`), with
`s_ti = q_t·k_i / sqrt(d_K)` and the static coefficients `c_r` per layer and head.

For a rate that reads no token, the cumulative product of FoX and the query's rate to the
power `t - i` coincide. `A_log` and `dt_bias` keep the query gate's Gated DeltaNet
initialization (`exp(A_log)` uniform on (0, 16), `softplus(dt_bias)` log-uniform on
[0.001, 0.1]) and its seed-42 draws, and are exempt from weight decay. At the start, half-lives
`ln 2 / lambda` over the 210 layer-heads are 1.3 / 9.2 / 62 tokens (10th / 50th / 90th
percentiles). Per-order gates start as copies of the shared draw, so each per-order run starts
equal to its shared-gate run. Static coefficients have no input projection, so no coefficient
weight decay applies.

| Role | Run | Label | m | Gate | Position |
| --- | --- | --- | ---: | --- | --- |
| Primary: shared gate | `r0e76396bdee5` | `static-m4-learned-alibi-gate-no-rope` | 4 | input-free | gate only |
| Primary: per-order gates | `r527a30ff7ed1` | `static-m4-per-order-learned-alibi-gate-no-rope` | 4 | input-free, per order | gate only |
| Primary: shared gate | `rc60345dfcff7` | `static-m8-learned-alibi-gate-no-rope` | 8 | input-free | gate only |
| Primary: per-order gates | `re253a5f1b03e` | `static-m8-per-order-learned-alibi-gate-no-rope` | 8 | input-free, per order | gate only |
| Control: static | `r4339a245820e` | `static-m4-head-scale` | 4 | none | RoPE |
| Control: static | `r9c69dd7cfc8d` | `static-m8-head-scale` | 8 | none | RoPE |
| Reference: query, no gate | `rca97cc9db8f2` | `query-m4-w-decay` | 4 | none | RoPE |
| Reference: query gate | `r3e26412b2811` | `query-m4-query-distance-no-rope-w-decay` | 4 | reads `x_t` | gate only |
| Reference: query gates | `r507193d47450` | `query-m4-per-order-query-gate-nope-wd` | 4 | reads `x_t`, per order | gate only |
| Reference: query gate | `r9320e5cd3071` | `query-m8-query-distance-no-rope-w-decay` | 8 | reads `x_t` | gate only |
| Reference: query gates | `r20fe6fe864bc` | `query-m8-per-order-query-gate-nope-wd` | 8 | reads `x_t`, per order | gate only |

Each shared-gate run is its static control's recipe with `positional_encoding: none` and the
gate (`attention_forget_gate: true`, `attention_forget_gate_mode: query_distance`,
`attention_forget_gate_conditioning: none`); each per-order run adds
`attention_forget_gate_per_order: true` to its shared-gate run. Everything else is the
control's: learned static per-head coefficients stored factorial-scaled (`c_r = c~_r / r!`,
Taylor start), canonical QK RMSNorm with per-head scaling, head RMSNorm output scaling, the
Mosaic kernels, seed 42, 2,048 tokens, batch 256, 7,570 updates, learning rate 0.001 with 455
warmup updates and cosine decay to 0.0001, and weight decay 0.1. Every parameter other than
the gate starts at its control's start, and the gate at the query-distance gate's draws.

| Run | Parameters |
| --- | ---: |
| Static control, m = 4 / 8 | 198,454,636 / 198,455,476 |
| Shared input-free gate, m = 4 / 8 | 198,455,056 / 198,455,896 |
| Per-order input-free gates, m = 4 / 8 | 198,456,736 / 198,459,256 |

The runs train on the Mosaic decay kernels (`query_decay`, `per_order_query_decay`) with every
query's coefficients and rates equal, on one H100 each: B300 cannot hold the degree-eight
per-order gates, and the other three runs use H100 too, so that all four train on one GPU
type. Qualification (the scaffold's `tests/test_input_free_gate.py` and the study adapter
`static_distance_gate_acceptance.py`) checks the output and every gradient against FP64 on the
blocked scan and strips and, on the allocated GPU, on the Mosaic kernels at 2,048 tokens; that
duplicated orders reproduce the shared gate; the initialization against the pinned controls;
the weight-decay masks; and the shared full-recipe pilot and resume. Launch commands are in the
[study README](../../../README.md#learned-input-free-gates-for-static-coefficients).

Each shared gate compares with its RoPE static control, each per-order run with its shared-gate
run, and each static gate with its query-distance counterpart at the same degree and gate
sharing. Best PPL is the lowest validation PPL over all evaluations; terminal PPL is at update
7,570.

| Run | Status | Validation-best PPL / update | Terminal PPL |
| --- | --- | --- | ---: |
| Static m = 4, RoPE `r4339a245820e` | complete | 14.767 / 7,500 | 14.773 |
| Static m = 8, RoPE `r9c69dd7cfc8d` | complete | 14.294 / 7,500 | 14.299 |
| Query m = 4, RoPE `rca97cc9db8f2` | complete | 14.709 / 7,500 | 14.714 |
| Query m = 4, shared gate `r3e26412b2811` | complete | 14.198 / 7,500 | 14.203 |
| Query m = 4, per-order gates `r507193d47450` | complete | 13.892 / 7,500 | 13.894 |
| Query m = 8, shared gate `r9320e5cd3071` | planned | — | — |
| Query m = 8, per-order gates `r20fe6fe864bc` | planned | — | — |
| Static input-free gates (all four) | planned | — | — |

## Open decision

Discard the topic, or promote the comparison with the forget gate into a positional-structure
collection together with the no-RoPE and continuation results. Evaluate the input-free gates
against their static controls and the query-distance gates first.

## Regeneration

Measurements live only in each selected run. Refresh the selection's pins after
intentionally incorporating new measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/exploratory/alibi_decay
```

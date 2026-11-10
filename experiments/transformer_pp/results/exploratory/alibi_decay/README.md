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
A [second part](#softmax-and-exp-with-alibis-fixed-slopes) asks the same of softmax and exp
with ALiBi itself: the slopes fixed, as in the original. A
[third part](#softmax-and-exp-with-query-distance-gates) makes that slope input-dependent: the
query token's own forget-gate rate, or on exp a rate that reads both tokens of each pair. A
[fourth](#learned-input-free-gates-for-static-coefficients) gives learned static coefficients a
learned decay that reads no token, shared or per polynomial order, at degrees four and eight. The
runs live in the [Transformer++ study](../../../README.md) and enter no canonical comparison.

## Runs (degree-four query polynomial)

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

## Softmax and exp with ALiBi's fixed slopes

Added 2026-11-08. Does ALiBi, with its fixed slopes and no positional encoding, match the
token-dependent forget gate on the exponential kernels? Each primary copies the no-RoPE
forget-gate run on its kernel ([forget gate topic](../forget_gate/README.md#softmax-and-exp-with-the-forget-gate))
and changes only the gate: `attention_forget_gate_form: constant` with
`attention_forget_gate_constant_rate: fixed`, so every score gains `-m_h (t - i)`,
`m_h = 2^(-8 h / 14)` for heads `h = 1..14` in every layer, with **no parameters** (the gate
controls carry 188,580). Softmax with this bias is ALiBi (Press et al., 2022); exp keeps its
per-head output RMSNorm. These are the paper's geometric slopes, which the learned-slope run
starts from, 0.67 down to 1/256; ALiBi's code would give 14 heads `2^-1..2^-8` and
`2^-0.5, 2^-1.5, ..., 2^-5.5` instead (its rule for a head count that is not a power of two),
0.71 down to 1/256. The primaries train on the blocked backend's causal strips and the controls
trained on its scan: the same arithmetic, with FP32 rather than BF16 sums of the K and V gradients
across blocks.

| Role | Run | Label | Kernel | Position | Validation-best PPL (terminal) |
| --- | --- | --- | --- | --- | --- |
| Primary: softmax ALiBi | `ref652c257797` | `softmax-alibi-fixed-no-rope` | softmax | ALiBi only | 13.899 (13.904) |
| Primary: exp ALiBi | `r7820049184ef` | `exp-alibi-fixed-no-rope` | exp | ALiBi only | 13.901 (13.904) |
| Control: softmax forget gate | `r5adc29a91e3b` | `softmax-forget-gate-no-rope` | softmax | gate only | 13.680 (13.680) |
| Control: exp forget gate | `r183f08346ddc` | `exp-forget-gate-no-rope` | exp | gate only | 13.628 (13.630) |
| Reference: softmax | `r9040f12f3d80` | `softmax-head-scale` | softmax | RoPE | 14.543 (14.546) |
| Reference: exp | `r6a76e7fbde45` | `exp-head-scale` | exp | RoPE | 14.257 (14.262) |

Both primaries completed 7,570 updates in one attempt each, after acceptance (including ALiBi's
GPU tests on the strips) passed: softmax on one B300 (`akita`, `standard`, job 239928,
8 h 55 min), exp on one H100 SXM (`dogo`, `long`, job 239929, 11 h 16 min). Launch commands are
in the [study README](../../../README.md#alibi-on-softmax-and-exp).

## Softmax and exp with query-distance gates

Added 2026-11-09. How much is the input dependence of the slope worth on the exponential kernels?
Fixed-slope ALiBi trails the forget gate by 1.6% (softmax) and 2.0% (exp). Each primary copies the
no-RoPE forget-gate run on its kernel and changes only `attention_forget_gate_mode: query_distance`:
every score gains `(t - i) log alpha_t`, the query token's own rate times the distance, in place of
FoX's `L_t - L_i`, the sum of `log alpha_j` over the tokens `j = i+1..t`. This is ALiBi with a learned
slope per query token and head, `lambda_t = exp(A_log) softplus(w_alpha . x_t + b_dt)`. The gate, its
188,580 parameters and their initialization are FoX's, so each primary starts from its control's
initial parameters, though not as its control's model: `w_alpha` starts nonzero, so `log alpha_t`
varies by token and the two decays differ from the first step. Softmax normalizes the decayed
scores; exp keeps its per-head output RMSNorm. The primaries train on the causal strips and the
controls trained on the scan.

Added 2026-11-10: a query–key gate on exp. `rd3501a40e020` copies the exp query-distance primary and
changes only `attention_forget_gate_conditioning: query_key`: each pair's rate reads both of its
tokens, `lambda_ti = exp(A_log) softplus(w_alpha . x_t + w_k . x_i + b_dt)`, and every score gains
`(t - i) log alpha_ti`. This is the gate of the per-order query–key polynomials (`r03be769d2069`,
13.708 at degree four; degrees eight and twelve in training) with one gate per layer and head, since exp
has no orders: it is their matched exp baseline. On the degree-four polynomial, the shared query–key
gate lowers validation-best PPL by 1.2% against the query gate, and per-order query–key gates by a
further 2.3% ([query_distance_decay](../query_distance_decay/README.md#results)). The key projection
`w_k` (188,160 parameters) starts at zero, so the run starts exactly as the exp query-distance
primary; `w_alpha`, `A_log` and `b_dt` keep FoX's initialization. It compares with exp FoX
(`r183f08346ddc`, 13.628), the exp query-distance primary (`r17444ad0f60e`) and exp ALiBi
(`r7820049184ef`, 13.901). No softmax counterpart is planned.

| Role | Run | Label | Kernel | Score decay | Validation-best PPL |
| --- | --- | --- | --- | --- | ---: |
| Primary: softmax query distance | `r208110a25271` | `softmax-query-distance-no-rope` | softmax | `(t - i) log alpha_t` | 13.701 |
| Primary: exp query distance | `r17444ad0f60e` | `exp-query-distance-no-rope` | exp | `(t - i) log alpha_t` | 13.638 |
| Primary: exp query–key distance | `rd3501a40e020` | `exp-query-key-distance-no-rope` | exp | `(t - i) log alpha_ti` | 13.604 |
| Control: softmax forget gate | `r5adc29a91e3b` | `softmax-forget-gate-no-rope` | softmax | `L_t - L_i` | 13.680 |
| Control: exp forget gate | `r183f08346ddc` | `exp-forget-gate-no-rope` | exp | `L_t - L_i` | 13.628 |
| Reference: softmax ALiBi | `ref652c257797` | `softmax-alibi-fixed-no-rope` | softmax | `-m_h (t - i)` | 13.899 |
| Reference: exp ALiBi | `r7820049184ef` | `exp-alibi-fixed-no-rope` | exp | `-m_h (t - i)` | 13.901 |
| Reference: query m = 4 | `r3e26412b2811` | `query-m4-query-distance-no-rope-w-decay` | query polynomial | `(t - i) log alpha_t` | 14.198 |
| Reference: query m = 8 | `r9320e5cd3071` | `query-m8-query-distance-no-rope-w-decay` | query polynomial | `(t - i) log alpha_t` | 13.939 |
| Reference: query m = 8, per order | `r20fe6fe864bc` | `query-m8-per-order-query-gate-nope-wd` | query polynomial | `(t - i) log alpha_tr` for order `r` | 13.780 |
| Reference: query m = 4, per order, query–key | `r03be769d2069` | `query-m4-per-order-query-key-gate-nope-wd` | query polynomial | `(t - i) log alpha_tir` for order `r` | 13.708 |

All three primaries completed 7,570 updates: softmax on one B300 (`akita`, job 240747, 8 h 35 min),
exp on one H100 SXM (`dogo`, job 240749, 10 h 56 min) and the exp query–key run on one B300
(`akita`, job 241447, 8 h 52 min). The polynomial references come from
[query_distance_decay](../query_distance_decay/README.md), where the same gate in query-distance
mode trails cumulative FoX by 2.3% at degree four (14.198 against 13.873). Launch commands are in
the study README for the
[query-distance pair](../../../README.md#query-distance-gates-on-softmax-and-exp) and the
[query–key gate](../../../README.md#query-key-gate-on-exp).

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

## Findings and open decision

### Softmax and exp with query-distance gates

Validation-best PPL; each comparison covers all 28 evaluations.

- **An input-dependent slope recovers nearly all of the forget gate's gain over ALiBi.** Softmax
  (13.701) and exp (13.638) end 1.43% and 1.89% below fixed ALiBi and 0.15% and 0.07% above the
  forget gate: 90% and 96% of the ALiBi-to-FoX gap. From update 3,000 on, exp stays within −0.02%
  to +0.27% of FoX and softmax 0.15–0.29% above it. Against RoPE they are 5.8% and 4.3% lower.
- **For the exponential kernels the decay rule barely matters; for the query polynomial it does.**
  Query-distance decay costs exp 0.07% against FoX, but the degree-four query polynomial 2.3%
  (14.198 against 13.873).
- **The polynomial query-distance runs trail exp under the same gate** (pinned in
  [query_distance_decay](../query_distance_decay/README.md)): per-order query–key gates at m = 4
  (13.708) by 0.52%, per-order query gates at m = 12 (13.750) by 0.82% and at m = 8 by 1.04%, and
  the shared m = 12 gate by 1.56%; none is lower at more than 5 of 28 evaluations.

### Exp with a query-key gate

Validation-best PPL; each comparison covers all 28 evaluations.

- **Reading the key token helps exp slightly.** The query–key gate reaches 13.604 at update 7,500
  (terminal 13.608): −0.25% against the query-only gate (13.638), lower at 22 of 28 evaluations and
  by 0.14–0.27% from update 3,000 on, and −0.18% against FoX (13.628), lower at 11 of 28 and between
  −0.18% and +0.07% from update 3,000 on. It is 2.14% below exp ALiBi and 4.58% below exp RoPE, at all
  28 evaluations, and the lowest exp run without positional encoding.
- **The key projection is used.** Starting at zero, its per-head norm reaches a median of 0.36
  against the query projection's 0.64. Half-lives barely change: a median of 3.0 tokens, as with
  the query-only gate (3.1), with 19% of heads above 100 tokens against 21%.
- **It is the bar for the per-order query–key polynomials.** At degree four (13.708) they trail
  it by 0.77%, lower at 4 of 28 evaluations; degrees eight and twelve are in training. The
  per-order FoX polynomial at degree eight (`rf8fc9ab2cecc`, 13.633, pinned in
  [forget_gate](../forget_gate/README.md#per-order-gates)) trails it by 0.21%.

### Softmax and exp with ALiBi's fixed slopes

Validation-best PPL is at update 7,500 for all six runs; each has 28 evaluations.

- **Fixed-slope ALiBi beats RoPE on both kernels**: softmax 13.899 vs 14.543 (−4.43%) and exp
  13.901 vs 14.257 (−2.50%), lower at all 28 evaluations and by 4.3–4.6% and 2.5–2.7% from
  update 3,000 on.
- **It trails the forget gate on both**: softmax +1.60% against 13.680 and exp +2.00% against
  13.628, behind at all 28 evaluations and by 1.6–2.0% and 2.0–2.6% from update 3,000 on. As
  with the degree-four polynomial below, a decay with distance alone does not reach the
  token-dependent gate.
- **With ALiBi, softmax and exp tie**: 13.899 vs 13.901 (−0.01%). Softmax is lower at all 28
  evaluations, by at most 0.5% from update 3,000 on. Exp leads by 2.0% with RoPE and by 0.4%
  with the forget gate.
- **The degree-eight per-order query-distance polynomial beats both** (`r20fe6fe864bc`, 13.780,
  pinned in [query_distance_decay](../query_distance_decay/README.md#degree-eight)): −0.86%
  against softmax ALiBi, lower from update 3,000 on (11 of 28 evaluations) by 0.4–0.9%, and
  −0.87% against exp ALiBi, lower at 26 of 28. Its decay is learned and query-dependent, so
  this is not a matched comparison.

### Learned input-free gates for static coefficients

All four completed 7,570 updates on one H100 SXM each (`dogo`, jobs 240497–240500) without
preemption. Validation-best PPL; each comparison covers all 28 evaluations.

| Run | m | Gate | Validation-best PPL / update | Terminal | vs RoPE static | vs query-coefficient gate run |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `r0e76396bdee5` | 4 | shared | 14.304 / 7,570 | 14.304 | −3.14% (14.767) | +0.75% (14.198) |
| `r527a30ff7ed1` | 4 | per order | 14.285 / 7,500 | 14.288 | −3.26% (14.767) | +2.83% (13.892) |
| `rc60345dfcff7` | 8 | shared | 14.135 / 7,500 | 14.138 | −1.11% (14.294) | +1.41% (13.939) |
| `re253a5f1b03e` | 8 | per order | 14.261 / 7,500 | 14.266 | −0.23% (14.294) | +3.49% (13.780) |

- **A learned decay replaces RoPE for static coefficients.** Both shared gates are below their
  RoPE controls at all 28 evaluations: by 3.14% at degree four and 1.11% at degree eight at the
  best update, and by 2.6–3.2% and 0.8–1.1% from update 3,000 on.
- **Per-order gates do not help static coefficients.** At degree four they tie the shared gate
  (−0.13%; within 0.3% from update 3,000 on). At degree eight they trail it by 0.89% and are
  above it at 27 of 28 evaluations; the gap narrows from 4.4% at update 1,000 to 0.9% without
  closing, and the run is above its RoPE control from update 1,300 to 5,000. Its early gradient
  spikes (pre-clip norms up to 103 before update 1,500) subside: from update 3,000 on its
  largest norm is 44 and 0.8% of updates clip (36 of 4,570), against 53 and 1.6% (71 of 4,570)
  for the shared gate.
- **Static runs trail the query-coefficient gate runs at every evaluation**: shared gates by
  0.75% (m = 4) and 1.41% (m = 8), per-order gates by 2.83% and 3.49%. These pairs differ in
  the coefficients' and the gate's input dependence together. With query coefficients and query
  gates, per-order gating gained 2.2% (m = 4) and 1.1% (m = 8); here it gains nothing.
- The degree-eight static shared gate trails fixed ALiBi on softmax (13.899) and exp (13.901)
  by 1.7%.
- **What the gates learned.** At update 7,570 the median half-life over the 210 layer-heads is
  9.8 tokens for both shared gates and 6.7–9.1 (m = 4) and 6.9–10.7 (m = 8) for the orders of
  the per-order gates, longer than the query gates' 0.9–5.0. As with the query gates, the top
  orders hold the most long-range heads: above 100 tokens, 41 and 51 of 210 for orders 3 and 4
  at m = 4 and 50 and 45 for orders 7 and 8 at m = 8, against 12–27 for the other orders.

### Degree-four query polynomial

The primary
completed its 7,570 updates on one B300 (Kennel `akita`, `long`; job 229720, 11 h 26 min). The
control completed on the same node.

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

**Open decision:** discard the topic, or promote the comparison with the forget gate into a
positional-structure collection together with the no-RoPE and continuation results.

Measurements live only in each selected run. Refresh the selection's pins after
intentionally incorporating new measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/exploratory/alibi_decay
```

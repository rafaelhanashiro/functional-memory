# ALiBi-style constant decay without RoPE

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
query token's own forget-gate rate, or on exp a rate that reads both tokens of each pair. The runs
live in the [Transformer++ study](../../../README.md) and enter no canonical comparison.

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
13.708 at degree four; degrees eight and twelve planned) with one gate per layer and head, since exp
has no orders: it is their matched exp baseline. On the degree-four polynomial, the shared query–key
gate lowers validation-best PPL by 1.2% against the query gate, and per-order query–key gates by a
further 2.3% ([query_distance_decay](../query_distance_decay/README.md#results)). The key projection
`w_k` (188,160 parameters) starts at zero, so the run starts exactly as the exp query-distance
primary; `w_alpha`, `A_log` and `b_dt` keep FoX's initialization. It compares with exp FoX
(`r183f08346ddc`, 13.628), the exp query-distance primary (`r17444ad0f60e`) and exp ALiBi
(`r7820049184ef`, 13.901). No softmax counterpart is planned.

| Role | Run | Label | Kernel | Score decay | Validation-best PPL |
| --- | --- | --- | --- | --- | ---: |
| Primary: softmax query distance | `r208110a25271` | `softmax-query-distance-no-rope` | softmax | `(t - i) log alpha_t` | |
| Primary: exp query distance | `r17444ad0f60e` | `exp-query-distance-no-rope` | exp | `(t - i) log alpha_t` | |
| Primary: exp query–key distance | `rd3501a40e020` | `exp-query-key-distance-no-rope` | exp | `(t - i) log alpha_ti` | |
| Control: softmax forget gate | `r5adc29a91e3b` | `softmax-forget-gate-no-rope` | softmax | `L_t - L_i` | 13.680 |
| Control: exp forget gate | `r183f08346ddc` | `exp-forget-gate-no-rope` | exp | `L_t - L_i` | 13.628 |
| Reference: softmax ALiBi | `ref652c257797` | `softmax-alibi-fixed-no-rope` | softmax | `-m_h (t - i)` | 13.899 |
| Reference: exp ALiBi | `r7820049184ef` | `exp-alibi-fixed-no-rope` | exp | `-m_h (t - i)` | 13.901 |
| Reference: query m = 4 | `r3e26412b2811` | `query-m4-query-distance-no-rope-w-decay` | query polynomial | `(t - i) log alpha_t` | 14.198 |
| Reference: query m = 8 | `r9320e5cd3071` | `query-m8-query-distance-no-rope-w-decay` | query polynomial | `(t - i) log alpha_t` | 13.939 |
| Reference: query m = 8, per order | `r20fe6fe864bc` | `query-m8-per-order-query-gate-nope-wd` | query polynomial | `(t - i) log alpha_tr` for order `r` | 13.780 |
| Reference: query m = 4, per order, query–key | `r03be769d2069` | `query-m4-per-order-query-key-gate-nope-wd` | query polynomial | `(t - i) log alpha_tir` for order `r` | 13.708 |

The softmax query-distance primary has completed and the exp one is training; their results are
pending. The query–key primary is allocated and not yet trained. The polynomial references come
from [query_distance_decay](../query_distance_decay/README.md), where the same gate in
query-distance mode trails cumulative FoX by 2.3% at degree four (14.198 against 13.873). Launch
commands are in the study README for the
[query-distance pair](../../../README.md#query-distance-gates-on-softmax-and-exp) and the
[query–key gate](../../../README.md#query-key-gate-on-exp).

## Findings and open decision

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

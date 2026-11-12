# Forget gate on query coefficients (m = 4 and 8)

Opened 2026-10-27.

**Question.** Does a Forgetting Transformer decay, defined as in Gated DeltaNet, improve
degree-four polynomial attention with tanh query coefficients? Later sections add the
exponential kernels (softmax, exp), compare the paper's original sigmoid gate, and give each
polynomial order its own gate.

## Method

The run copies `query-m4-head-scale` (`r2b22f2ff6461`) and multiplies every attention
weight by the product of per-token gates between key and query:
$$w_{ti} = \Big(\prod_{j=i+1}^{t}\alpha_j\Big)\sum_{r\le 4} c_r(x_t)\,s_{ti}^r,\qquad
\log\alpha_t = -e^{A_{\log}}\,\mathrm{softplus}(w_\alpha^\top x_t + b_{dt})$$
per layer and head, with Gated DeltaNet's (Mamba-2) initialization:
$e^{A_{\log}} \sim U(0, 16)$, $\mathrm{softplus}(b_{dt})$ log-uniform on $[0.001, 0.1]$,
and $w_\alpha$ at PyTorch's default $U(\pm 1/\sqrt{d})$. As in Gated DeltaNet, $w_\alpha$
takes weight decay and $A_{\log}$, $b_{dt}$ do not. The gate reads the same pre-attention
activation as the coefficients.

| Run | Label | Backend |
| --- | --- | --- |
| `r111843b4c7cc` | query-m4-forget-gate | blocked JAX reference |
| `r82a80d3091e2` | query-m4-forget-gate-no-rope | blocked JAX reference, `positional_encoding: none` |

Mosaic has no kernel for this cumulative forget gate, so this run trains on the blocked
scan (FP32 weights, BF16 sums of the K and V gradients across 128-query blocks), while its
control ran on Mosaic (BF16 weights). Same-recipe tanh runs on the blocked scan and on the
fused Triton kernels agreed within 0.1 PPL at m = 1-4 (input-dependent coefficients collection).

Control in the [selection](selection.yaml): `query-m4-head-scale` (best 14.774), with
`fixed-m4-head-scale` (best 14.722) for reference.

## Findings

Validation perplexity (terminal, update 7,570; best at 7,500 in parentheses):

| Run | 1,000 | 2,500 | 5,000 | Terminal |
| --- | ---: | ---: | ---: | ---: |
| Forget gate `r111843b4c7cc` | 25.00 | 18.72 | 15.58 | **14.320** (14.318) |
| Query `r2b22f2ff6461` | 27.08 | 19.05 | 16.05 | 14.779 (14.774) |
| Fixed Taylor `r816ab0fbe119` | 28.21 | 19.16 | 16.01 | 14.724 (14.722) |

- The gate improves query attention by **3.1%** and beats fixed Taylor by **2.7%**, the
  largest gain of any m = 4 variant in this study. The lead over both controls holds at
  2.3–2.9% from update 2,500 onward; it is not an early-training effect.
- It closes most of the gap to unnormalized exp with the same QK setup
  (`r6a76e7fbde45`, 14.262): +0.4% behind at the end,
  down from +1.7% at update 2,500.
- The learned decays are mostly short-range. Over the 210 heads, the mean half-life of
  $\prod\alpha$ on the diagnostic sample has percentiles
  10/25/50/75/90% = 0.3/0.7/2.2/20.2/143 tokens;
  48% of heads forget within two tokens and 21% retain beyond 64. The gate
  adds 188,580 parameters (0.1%).
- Cost: the blocked reference backend ran at 6.0 s per update on a B300 against about
  2.1 s for the Mosaic query control. A production kernel is needed before wider use.

### Without RoPE

`r82a80d3091e2` repeats the forget-gate run with no positional encoding
(`positional_encoding: none`, added 2026-10-29): positions enter only through the causal
mask and the gate's decay, as in the Forgetting Transformer's RoPE-free variant.

| Run | 2,500 | 5,000 | 6,500 | Best | Terminal |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forget gate, no RoPE `r82a80d3091e2` | 17.89 | 15.12 | 14.23 | **13.906** | 13.908 |
| Forget gate with RoPE `r111843b4c7cc` | 18.72 | 15.58 | 14.66 | 14.318 | 14.320 |
| Query `r2b22f2ff6461` | 19.05 | 16.05 | 15.11 | 14.774 | 14.779 |
| Fixed Taylor `r816ab0fbe119` | 19.16 | 16.01 | 15.07 | 14.722 | 14.724 |

- Removing RoPE improves the gated model by **2.9%** (best 13.906 against 14.318),
  2.5% below unnormalized exp
  (`r6a76e7fbde45`, best 14.257) and 4.4% below softmax (`r9040f12f3d80`, best 14.543).
  The gap to the RoPE gate is steady at about 3% from update 2,500 onward.
- The learned decays are longer than with RoPE: mean half-life percentiles 10/25/50/75/90% =
  0.3/0.7/2.7/64.5/127 tokens (45% of heads under two tokens, 25% over 64).
  With RoPE gone the gate carries all positional information, and a learned recency decay
  appears to be a better positional signal for this attention than RoPE's rotation.
- Same cost caveat as above: 5.3 s per update on a B300 on the blocked backend.

### FoX with PaTH and projection decay

`r74495b47d226` tests the combination at degree four, with 7,570 updates (1×),
seed 42, and the canonical Transformer++ recipe. PaTH replaces RoPE in the scores;
the FoX product multiplies the resulting polynomial attention weights. Both the
coefficient projection and forget-gate projection receive weight decay 0.1, as do
PaTH's projection/convolution weights. Coefficient `a,b`, gate rates/biases, embeddings,
and normalization parameters remain exempt. Both coefficient `a,b` retain the
factorial order scaling. The peak/minimum learning rates are 0.001/0.0001, with
455 warmup updates and the canonical QK RMSNorm plus learned head scale.

The PaTH projection recipe matches `r1f2c24c109f9`; the FoX projection-decay recipe
matches `r4a80e28ce0a7`. This run uses the shared fused PaTH kernels with cumulative
log decay and explicit gate gradients; its GPU tests, full-recipe pilot, and resume
check passed on one H100 before training. The home cluster is Kennel; see the
[study launch instructions](../../../README.md#fox-with-path).

Validation perplexity; best is the validation-best checkpoint (update 7,500 for every
run), terminal is update 7,570:

| Run | 2,500 | 5,000 | 6,500 | Best | Terminal |
| --- | ---: | ---: | ---: | ---: | ---: |
| FoX + PaTH, projection decay `r74495b47d226` | 18.39 | 15.33 | 14.44 | **14.094** | 14.094 |
| Forget gate, no RoPE `r82a80d3091e2` | 17.89 | 15.12 | 14.23 | 13.906 | 13.908 |
| Forget gate with RoPE `r111843b4c7cc` | 18.72 | 15.58 | 14.66 | 14.318 | 14.320 |
| Query `r2b22f2ff6461` | 19.05 | 16.05 | 15.11 | 14.774 | 14.779 |
| Fixed Taylor `r816ab0fbe119` | 19.16 | 16.01 | 15.07 | 14.722 | 14.724 |

- Under the gate PaTH beats RoPE but trails no positional encoding: best 14.094 is 1.6%
  below the RoPE gate and 1.4% above the no-RoPE gate, a gap that narrows from 2.8% at
  update 2,500 to 1.4% by 5,000 and then holds. The matched no-RoPE gate with projection
  decay (`r4a80e28ce0a7`, best 13.873), whose recipe
  differs only in positional encoding and backend, is 1.6% better.
- Gate and projection decay together improve ungated PaTH without projection decay
  (`r74a50aca921e`, best 14.528) by 3.0%; the combination is 1.1% below unnormalized exp
  (`r6a76e7fbde45`, best 14.257). The matched ungated PaTH run with projection decay,
  `r1f2c24c109f9`, has no recorded result, so the gate's share at equal decay is unmeasured.
- With PaTH the gate learns longer decays than with RoPE or none: mean half-life
  percentiles 10/25/50/75/90% = 0.4/1.0/5.7/106/198 tokens (38% of heads under two
  tokens, 32% over 64). The fused kernels ran at 5.2 s per update on one H100.

### Softmax and exp with the forget gate

Four runs (added 2026-11-03) put the same gate on the exponential kernels of softmax
(`r9040f12f3d80`) and unnormalized exp (`r6a76e7fbde45`), without RoPE: the decay joins
every score, $\mathrm{softmax}_i(s_{ti} + L_t - L_i)$ or $\exp(s_{ti} + L_t - L_i)$ with
$L_t = \sum_{j\le t}\log\alpha_j$. Positions come from the gate alone (blocked reference) or
from PaTH as well (PaTH-FoX, FLA's gated kernels). Exp keeps its per-head output RMSNorm and
softmax has none, as in their sources. All are 1× (7,570 updates), seed 42; launch commands
are in the [study README](../../../README.md#forget-gate-on-softmax-and-exp).

Validation perplexity; best is the validation-best checkpoint (update 7,500 for every
run), terminal is update 7,570:

| Run | Kernel | Positions | 2,500 | 5,000 | Best | Terminal | Training GPU |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PaTH-FoX exp `rafcf22b799ed` | unnormalized exp, per-head RMSNorm | PaTH | 16.98 | 14.44 | **13.321** | 13.326 | H100 SXM (`dogo`) |
| PaTH-FoX softmax `r14db6f0e8592` | softmax | PaTH | 17.00 | 14.47 | 13.354 | 13.357 | H100 PCIe (`mastiff`) |
| FoX exp `r183f08346ddc` | unnormalized exp, per-head RMSNorm | none | 17.40 | 14.77 | 13.628 | 13.630 | H100 SXM (`dogo`) |
| FoX softmax `r5adc29a91e3b` | softmax | none | 17.39 | 14.81 | 13.680 | 13.680 | B300 (`akita`) |
| Forget gate, no RoPE, projection decay `r4a80e28ce0a7` | query m = 4 | none | 17.84 | 15.08 | 13.873 | 13.875 | B300 |
| FoX + PaTH, projection decay `r74495b47d226` | query m = 4 | PaTH | 18.39 | 15.33 | 14.094 | 14.094 | H100 SXM |
| Exp `r6a76e7fbde45` | unnormalized exp, per-head RMSNorm | RoPE | 18.40 | 15.47 | 14.257 | 14.262 | B300 |
| Softmax `r9040f12f3d80` | softmax | RoPE | 18.64 | 15.76 | 14.543 | 14.546 | B300 |

- The gate without RoPE improves both exponential kernels over their RoPE sources: exp
  14.257 to 13.628 (−4.4%), softmax 14.543 to 13.680 (−5.9%).
- PaTH adds 2.3% (exp) and 2.4% (softmax) under the gate, steady from update 2,500 on.
  PaTH-FoX exp (13.321) and softmax (13.354) are the two lowest validation-best PPLs of any
  Transformer++ 1× run. PaTH-FoX exp is 4.0% below the best gated polynomial
  (`r4a80e28ce0a7`, 13.873) and 6.6% below exp.
- Under the gate the exponential kernels lead the degree-four query polynomial with
  projection decay: exp by 1.8% without PaTH (13.628 against 13.873) and 5.5% with PaTH
  (13.321 against 14.094); softmax by 1.4% and 5.2%.
- Softmax and exp nearly tie under the gate: softmax is 0.4% (none) and 0.2% (PaTH) above
  exp. The pairs also differ in per-head output RMSNorm, so this does not isolate
  normalization.
- Cost: the blocked reference ran FoX at 8.0 s per update on an H100 SXM (exp) and 6.2 s on
  a B300 (softmax); FLA's gated PaTH kernels ran at 4.2 s (exp, H100 SXM) and 6.7 s
  (softmax, H100 PCIe).
- Limits: one seed each, on the mixed hardware above. FoX exp resumed from its update-6,500
  checkpoint after a preemption and a CUDA-graph runtime error.

### At m = 8 with projection decay

`r9353f554b917` (`query-m8-forget-gate-no-rope-wdecay`, added 2026-11-03) asks whether the
no-RoPE gate with projection decay still helps at degree eight. It copies
`query-m8-dk64-w-decay` (`r25fffc16c2f3`, from the
[query width-by-degree topic](../query_width_degree/README.md)) and changes only
`attention_forget_gate: true`, `positional_encoding: none`, and the blocked backend the gate
requires; that run is its control. Home cluster ORCD: it trained on H100s in three
allocations, the last resuming from update 3,500.

Validation perplexity; best is update 7,500 for every run, terminal is update 7,570:

| Run | 2,500 | 5,000 | Best | Terminal |
| --- | ---: | ---: | ---: | ---: |
| Forget gate, no RoPE, projection decay, m = 8 `r9353f554b917` | 17.86 | 15.10 | 13.907 | 13.910 |
| Forget gate, no RoPE, projection decay, m = 4 `r4a80e28ce0a7` | 17.84 | 15.08 | **13.873** | 13.875 |
| Query, projection decay, m = 8 `r25fffc16c2f3` (control) | 18.43 | 15.54 | 14.323 | 14.328 |
| Query, projection decay, m = 4 `rca97cc9db8f2` | 19.01 | 15.96 | 14.709 | 14.714 |

- The gate improves degree-eight query attention by 2.9% (13.910 against 14.328), half its
  5.7% gain at degree four, because the control improves with degree and the gated runs do
  not.
- Under the gate, degree eight matches degree four: 13.910 against 13.875 (+0.25%), within the
  0.2% gap between the degree-scaling study's numerically equivalent `m = 1` runs. The two
  curves are within 0.2% at updates 2,500 and 5,000 as well.
- Cost: 9.8 s per update on one H100 on the blocked reference.
- Limits: one seed. The control ran on a Kennel B300 at an earlier commit on Mosaic, and
  `r4a80e28ce0a7` on a B300; the gated degree-eight run trained on ORCD H100s.

### Original FoX sigmoid gate

Three runs (added 2026-11-05) replace the Gated DeltaNet gate of the no-RoPE runs with the
Forgetting Transformer's own gate, $\log\alpha_t = \log\sigma(w_f^\top x_t + b_f)$ per layer
and head, initialized as in its code: $w_f \sim \mathcal N(0, 0.02^2)$ and $b_f = 0$, so every
gate starts near $\sigma(0) = 1/2$; $w_f$ takes weight decay and $b_f$ does not. The Gated
DeltaNet gate is the same sigmoid raised to a learned per-head power,
$\alpha_t = \sigma(-z_t)^{e^{A_{\log}}}$ with $z_t = w_\alpha^\top x_t + b_{dt}$, under
Mamba-2's initialization. Exp and query change only the gate; softmax also drops QK
normalization (plain scores), as in FoX (LLaMA). All are 1× (7,570 updates), seed 42, on the
blocked backend; launch commands are in the
[study README](../../../README.md#original-fox-sigmoid-gate).

Validation perplexity; best is the validation-best checkpoint (update 7,500 for every run),
terminal is update 7,570:

| Run | Gate | 2,500 | 5,000 | Best | Terminal | Training GPU |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Exp, sigmoid gate `rdd3e9cb3bb80` | sigmoid | 17.98 | 15.20 | 14.005 | 14.010 | H100 SXM (`dogo`) |
| FoX exp `r183f08346ddc` | Gated DeltaNet | 17.40 | 14.77 | **13.628** | 13.630 | H100 SXM (`dogo`) |
| Query + W decay, sigmoid gate `rfa9636279199` | sigmoid | 18.58 | 15.64 | 14.372 | 14.377 | H100 SXM (`dogo`) |
| Forget gate, no RoPE, projection decay `r4a80e28ce0a7` | Gated DeltaNet | 17.84 | 15.08 | **13.873** | 13.875 | B300 |
| Softmax, sigmoid gate, plain scores `reeb92d46a95d` | sigmoid | 18.01 | 15.23 | 14.014 | 14.014 | H100 PCIe (`mastiff`) to update 200, then B300 (`akita`) |
| FoX softmax `r5adc29a91e3b` | Gated DeltaNet | 17.39 | 14.81 | **13.680** | 13.680 | B300 (`akita`) |

- Where the gate is the only change it costs 2.8% on exp (14.005 against 13.628) and 3.6% on
  query (14.372 against 13.873). Both gaps are steady through the second half of training
  (exp 2.8–3.0% from update 4,500, query 3.6–3.8% from 5,500).
- Softmax with plain scores, as in FoX (LLaMA), trails its control by 2.4% (14.014 against
  13.680). That gap narrowed from 4.2% at update 1,500, but the run changes the QK
  normalization as well as the gate, so it does not isolate the gate.
- The sigmoid gate stays short-range. At update 100 its heads' mean half-lives on the
  diagnostic sample have 10/50/90th percentiles of 0.38/0.52/1.4 tokens (exp) with none over
  64 tokens, against 0.31/3.2/79 tokens and 17% over 64 for the Gated DeltaNet gate. It
  gains long-range heads slowly: at the end 0.25/0.79/17.8 tokens, 68% of heads under two
  tokens and 7% over 64, against 0.28/2.8/98 tokens, 46% and 20%. Query matches (0.28/0.84/16
  tokens, 71% and 7%, against 0.32/3.0/117 tokens, 45% and 23%). Softmax's control starts
  longer-range (median 8.5 tokens and 22% over 64 at update 100, against 0.67 and none) and
  ends at 14.9 tokens and 38% over 64, against 1.7 and 10%. The learned biases stay between
  −1.1 and 0.55 (per-layer mean $\sigma(b_f)$ 0.27–0.53), so long memory comes only from the
  projection $w_f^\top x_t$.
- Mamba-2's initialization spreads the decay rates from the start, while the paper's starts
  every head at $\alpha \approx 1/2$; within 1× the sigmoid gate does not build the same
  long memory. These runs do not separate the gate's form from its initialization.
- Limits: one seed each. On an H100 SXM the blocked backend ran exp at 8.0 s per update, as
  with the Gated DeltaNet gate, and query at 9.8 s. Softmax trained its first 200 updates on
  an H100 PCIe and the rest on a B300 (6.1 s per update) after that H100 stopped responding.

### Per-order gates

Added 2026-11-09. Do independent per-order gates improve cumulative FoX, as they improved
query-distance decay by 2.2% (per-order query gates 13.892 against the shared gate's 14.198 at
m = 4, in [query_distance_decay](../query_distance_decay/README.md))? Each primary copies its
no-RoPE FoX + W decay control and changes only `attention_forget_gate_per_order: true`, so every
order $r = 0..m$ of the query polynomial decays with its own gate:
$$w_{ti} = \sum_{r\le m} c_r(x_t)\,s_{ti}^r\,e^{L_{t,r} - L_{i,r}},\qquad
L_{t,r} = \sum_{j\le t}\log\alpha_{j,r},\qquad
\log\alpha_{t,r} = -e^{A_{\log,r}}\,\mathrm{softplus}(w_{\alpha,r}^\top x_t + b_{dt,r})$$
per layer and head. Each order's gate starts as a copy of the control's shared draw, so each
primary starts exactly at its control; the gates add $m \times 188{,}580$ parameters. Both train on
the blocked backend's causal strips (the controls on its scan). The target is the best exponential
kernel under the same gate without positional encoding.

| Role | Run | Label | m | Validation-best PPL |
| --- | --- | --- | ---: | ---: |
| Primary: per-order gates | `r319b83140040` | `query-m4-per-order-forget-gate-no-rope-w-decay` | 4 | 13.767 |
| Primary: per-order gates | `rf8fc9ab2cecc` | `query-m8-per-order-forget-gate-no-rope-wdecay` | 8 | 13.633 |
| Control: shared gate | `r4a80e28ce0a7` | `query-m4-forget-gate-no-rope-w-decay` | 4 | 13.873 |
| Control: shared gate | `r9353f554b917` | `query-m8-forget-gate-no-rope-wdecay` | 8 | 13.907 |
| Target: FoX exp | `r183f08346ddc` | `exp-forget-gate-no-rope` | — | 13.628 |

Both completed 7,570 updates on one B300 each (`akita`; m = 4 on `standard`, job 240745, and m = 8 on
`long`, job 240746); each was preempted once during acceptance, before training began. Launch
commands are in the [study README](../../../README.md#per-order-cumulative-fox-gates).

- **Per-order gates improve cumulative FoX at both degrees.** m = 4 ends at 13.767 against the
  shared gate's 13.873 (−0.77%, lower at all 28 evaluations), m = 8 at 13.633 against 13.907
  (−1.97%, lower at 27 of 28). With per-order gates m = 8 improves on m = 4 by 0.97% (lower at 26 of
  28), where the shared gate got worse (+0.24%).
- **The degree-eight run ties the best exponential kernel.** It ends 0.04% above FoX exp (13.628;
  lower at 16 of 28 evaluations, within ±0.14% from update 3,000 on) and 0.04% below exp with the
  query-distance gate (13.638, lower at 27 of 28; pinned in [alibi_decay](../alibi_decay/README.md)),
  0.35% below FoX softmax (13.680), and 0.55% below the best per-order query-distance run
  (query–key gates at m = 4, 13.708, in [query_distance_decay](../query_distance_decay/README.md)).
- **Cumulative per-order gates beat query-distance ones** at both degrees: by 0.90% at m = 4 and
  1.07% at m = 8 against the per-order query gates (13.892, 13.780).
- **Low orders forget within about a token; the two highest hold the long ranges.** At update 7,570
  the median half-life over the 210 layer-heads is 0.62, 1.18 and 0.82 tokens for orders 0–2 at
  m = 4 and 4.75 and 3.33 for orders 3 and 4 (40 and 49 heads above 100 tokens); at m = 8 it is
  0.50–1.50 for orders 0–6 and 4.89 and 4.58 for orders 7 and 8 (36 and 43 heads above 100). The
  shared gates' medians are 3.0 (m = 4) and 3.2 (m = 8); all orders start at 4.5–5.9 tokens.

#### Degree twelve

Added 2026-11-11. Does raising the degree from eight to twelve take per-order cumulative gates past
FoX exp? `r95bfff096cd4` (`query-m12-per-order-forget-gate-no-rope-wdecay`) is `rf8fc9ab2cecc` at
degree twelve, trained on the Mosaic per-order cumulative kernels on one H100. Its thirteen gate
orders add 2,262,960 parameters (203,356,666 in all). No shared-gate run exists at degree twelve, so
acceptance checks the run's start against the degree-eight control's recipe at degree twelve, built
at seed 42: every order's gate is that recipe's shared draw and every other parameter is its own.
The launch command is in the [study README](../../../README.md#degree-twelve-on-the-mosaic-kernels).

| Role | Run | m | Implementation | Validation-best PPL |
| --- | --- | ---: | --- | ---: |
| Primary: per-order gates | `r95bfff096cd4` | 12 | Mosaic kernels, H100 | — |
| Comparison: per-order gates | `rf8fc9ab2cecc` | 8 | blocked strips, B300 | 13.633 |
| Comparison: per-order gates | `r319b83140040` | 4 | blocked strips, B300 | 13.767 |
| Target: FoX exp | `r183f08346ddc` | — | blocked scan, H100 | 13.628 |

**Limitation: the step to degree twelve also changes implementation and GPU.** The strips keep FP32
attention weights; the Mosaic kernels round weights and score gradients once to BF16 for the tensor
cores. No matched pair measures that change for this gate. Same-recipe tanh runs on the blocked
reference and on the fused Triton kernels, on different GPUs, differed by up to 0.1 PPL, 0.4–0.5% at
m = 3 and 4 (input-dependent coefficients collection). A difference from m = 8 or FoX exp below 0.5%
therefore does not separate the degree from the implementation; a best PPL at least 0.5% below both
(13.56 or lower) exceeds that variation.

## Open decision

Determine whether combining FoX and PaTH improves on either alone with projection
weight decay. Per-order gates bring the degree-eight query polynomial level with FoX exp; read the
degree-twelve run by the 0.5% rule above and decide whether to repeat it at a second seed, then
promote this question or discard it.

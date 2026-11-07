# Forget gate on query coefficients (m = 4 and 8)

Opened 2026-10-27.

**Question.** Does a Forgetting Transformer decay, defined as in Gated DeltaNet, improve
degree-four polynomial attention with tanh query coefficients? Later sections add the
exponential kernels (softmax, exp) and compare the paper's original sigmoid gate.

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

Mosaic has no forget gate, so this run evaluates attention on the exact blocked
reference, while its control ran on Mosaic. Earlier blocked and Mosaic tanh runs agreed.

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

## Open decision

Determine whether combining FoX and PaTH improves on either alone with projection
weight decay, then decide whether to promote this question or discard it.

# Forget gate on query coefficients (m = 4)

Opened 2026-10-27.

**Question.** Does a Forgetting Transformer decay, defined as in Gated DeltaNet, improve
degree-four polynomial attention with tanh query coefficients?

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

Control in the [selection](selection.yaml): `query-m4-head-scale` (14.779), with
`fixed-m4-head-scale` (14.724) for reference.

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
  (`r6a76e7fbde45`, 14.262, on `codex/qk-head-controls-4b`): +0.4% behind at the end,
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
  the lowest perplexity of any 1x run in the study: 2.5% below unnormalized exp
  (`r6a76e7fbde45`, best 14.257) and 4.4% below softmax (`r9040f12f3d80`, best 14.543).
  The gap to the RoPE gate is steady at about 3% from update 2,500 onward.
- The learned decays are longer than with RoPE: mean half-life percentiles 10/25/50/75/90% =
  0.3/0.7/2.7/64.5/127 tokens (45% of heads under two tokens, 25% over 64).
  With RoPE gone the gate carries all positional information, and a learned recency decay
  appears to be a better positional signal for this attention than RoPE's rotation.
- Same cost caveat as above: 5.3 s per update on a B300 on the blocked backend.

## Open decision

Extend the forget gate to fixed Taylor and other degrees (with a Mosaic kernel), or discard the topic.

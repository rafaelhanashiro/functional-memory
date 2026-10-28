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

## Open decision

Extend the forget gate to fixed Taylor and other degrees (with a Mosaic kernel), or discard the topic.

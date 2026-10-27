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

Pending.

## Open decision

Extend the forget gate to fixed Taylor and other degrees (with a Mosaic kernel), or discard the topic.

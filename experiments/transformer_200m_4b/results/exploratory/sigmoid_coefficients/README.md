# Sigmoid query coefficients (m = 4)

Opened 2026-10-26.

**Question.** Does a pure sigmoid generator, as used for the linear term in polynomial
GDN, train better than tanh query coefficients $c_r = b_r + a_r\tanh(w_r^\top x_t)$
at degree four?

## Method

Every run copies `query-m4` (`r49b7943aee1b`): 4B tokens, unnormalized degree-four
attention, per-head output RMSNorm, Mosaic kernels, one GPU. It changes only the
generator and adds per-head QK scaling (`model.qk_norm_head_scale: true`).

| Generator | Coefficients | Start |
| --- | --- | --- |
| Diagonal | $c_r = \sigma(w_r^\top x_t)\,[/\,r!]$ | $W = 0$ |
| Dense, $K = 5$ | $c_r = [\tfrac{1}{\sqrt5}]\sum_j V_{rj}\,[/\,r!]\;\sigma(w_j^\top x_t)$ | $W = 0$, $V = I$ |

There is no bias $b$ or amplitude $a$. Bracketed factors are present only with order
scaling (`attention_coefficient_amplitude_scale: factorial`; dense adds
`attention_coefficient_readout_scale` $= 1/\sqrt5$). At $W = 0$, every $c_r = 1/2$, or
$1/(2\,r!)$ with order scaling. Head RMSNorm removes this common factor, so order-scaled
runs start exactly at fixed Taylor; with no bias, the $1/\sqrt5$ factor has no effect
beyond RMSNorm's $\epsilon$. Unscaled runs start at $\tfrac12\sum_r s^r$.

"W decay" applies the optimizer's weight decay of 0.1 to $W$ alone
(`attention_coefficient_kernel_weight_decay: true`); $V$, norms, and gains stay excluded.

| Run | Label | Generator | Order scaling | W decay |
| --- | --- | --- | --- | --- |
| `r8b4e2a0603aa` | sigmoid-m4-scaled | diagonal | yes | no |
| `rd5550f8db5e4` | sigmoid-m4-unscaled | diagonal | no | no |
| `r343fff99fa9e` | sigmoid-m4-scaled-wdecay | diagonal | yes | yes |
| `re6d4c820a132` | sigmoid-m4-unscaled-wdecay | diagonal | no | yes |
| `r239ea241c978` | dense-sigmoid-m4-k5-scaled | dense | yes | no |
| `ra6ae537a5e37` | dense-sigmoid-m4-k5-unscaled | dense | no | no |
| `rf7e1a83e2e23` | dense-sigmoid-m4-k5-scaled-wdecay | dense | yes | yes |
| `r331b5fd5bebc` | dense-sigmoid-m4-k5-unscaled-wdecay | dense | no | yes |

Controls in the [selection](selection.yaml): `fixed-m4` (`r79da929de109`, 14.760) and
`query-m4` (`r49b7943aee1b`, 14.840), both without head scaling. Head-scale controls
`query-m4-head-scale` (`r2b22f2ff6461`) and `static-m4-head-scale` (`r4339a245820e`) run on
their own branches; join them when they are merged.

## Findings

Pending.

## Open decision

Promote the best sigmoid regime into `results/input_dependent_coefficients/`, or discard the topic.

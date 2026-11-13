# Dense readout scale

Opened 2026-10-29.

**Question.** With the canonical per-head QK scale, does the dense query generator
$c_r = \big[b_r + s\sum_{j=1}^{K} V_{rj}\tanh(w_j^\top x_t)\big]/r!$ need its readout scaled by
$s = 1/\sqrt{K}$, as a fan-in-$K$ linear layer would be?

## Runs

Both copy `query-m4-head-scale` (`r2b22f2ff6461`) with a dense readout of $K = 5$ tanh
features and $V = 0$ at the start, one GPU each; only the readout scale differs.

| Run | Label | Readout scale $s$ |
| --- | --- | --- |
| `r0c0f3634685c` | dense-query-m4-k5-head-scale | 1 |
| `rcaff0dfb095f` | dense-query-m4-k5-head-scale-vdivsqrt5 | $1/\sqrt{5}$ |

Two more runs use $K = 9$ features with $s = 1/3$, at m = 4 (`r04a5b2a3e8dd`) and m = 8
(`rfccffdd239dd`); their recipes differ from the scaled $K = 5$ run only in $K$, $s$, and degree.

## Findings

| Run | Best PPL | Terminal |
| --- | ---: | ---: |
| Dense K = 5, $s = 1/\sqrt{5}$ | **14.710** | 14.716 |
| Dense K = 5, $s = 1$ | 14.818 | 14.822 |
| Fixed Taylor `r816ab0fbe119` | 14.722 | 14.724 |
| Query (affine tanh) `r2b22f2ff6461` | 14.774 | 14.779 |
| Dense K = 9, $s = 1/3$ | 14.782 | 14.785 |
| Dense K = 9, $s = 1/3$, m = 8 | 14.400 | 14.406 |
| Fixed Taylor m = 8 `ra7ee8a4b2544` | 14.218 | 14.223 |

- The $1/\sqrt{K}$ scale improves dense K = 5 by 0.7% and brings it level
  with fixed Taylor (within 0.1%); unscaled, it was the weakest m = 4 generator.
- Independent features add incoherently, so the dense term's scale grows with $\sqrt{K}$;
  under Adam the fixed multiplier also sets the dense term's effective step size.
- With the same convention, K = 9 trails K = 5 by 0.5% at m = 4. At m = 8 it trails fixed
  Taylor by 1.3%.
- No dense variant beats fixed Taylor at m = 4.

## Open decision

Promote into its own canonical collection, or discard.

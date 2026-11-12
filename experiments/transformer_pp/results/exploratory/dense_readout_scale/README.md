# Dense readout scale (m = 4, K = 5)

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

## Findings

| Run | Best PPL | Terminal |
| --- | ---: | ---: |
| Dense K = 5, $s = 1/\sqrt{5}$ | **14.710** | 14.716 |
| Dense K = 5, $s = 1$ | 14.818 | 14.822 |
| Fixed Taylor `r816ab0fbe119` | 14.722 | 14.724 |
| Query (affine tanh) `r2b22f2ff6461` | 14.774 | 14.779 |

- The $1/\sqrt{K}$ scale improves dense K = 5 by 0.7% and brings it level
  with fixed Taylor (within 0.1%); unscaled, it was the weakest m = 4 generator.
- Independent features add incoherently, so the dense term's scale grows with $\sqrt{K}$;
  under Adam the fixed multiplier also sets the dense term's effective step size. The
  degree-four K = 9 run on `codex/dense-qk-scale-4b` (`r04a5b2a3e8dd`, scale 1/3, best 14.782)
  used the same convention and sits 0.5% behind K = 5.
- No dense variant beats fixed Taylor at m = 4.

## Open decision

Fold the scaled dense readout into the dense query topic when that branch merges, or
discard.

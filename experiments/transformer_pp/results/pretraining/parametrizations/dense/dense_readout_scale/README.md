# Dense readout scale and sigmoid coefficients

Opened 2026-10-26.

Two questions about the degree-four query coefficient generator under the canonical
per-head QK scale:

1. Does the dense generator
   $c_r = \big[b_r + s\sum_{j=1}^{K} V_{rj}\tanh(w_j^\top x_t)\big]/r!$ need its readout
   scaled by $s = 1/\sqrt{K}$, as a fan-in-$K$ linear layer would be?
2. Does a pure sigmoid generator, per degree or dense, as used for the linear term in
   polynomial GDN, train better than affine tanh?

Every run matches `query-m4-head-scale` (`r2b22f2ff6461`) except for the coefficient
generator, and the degree where noted: 1× Chinchilla, unnormalized attention with
per-head output RMSNorm, one GPU each. Runs and roles: [`selection.yaml`](selection.yaml).

## Dense readout scale

### Runs

Both copy `query-m4-head-scale` (`r2b22f2ff6461`) with a dense readout of $K = 5$ tanh
features and $V = 0$ at the start, one GPU each; only the readout scale differs.

| Run | Label | Readout scale $s$ |
| --- | --- | --- |
| `r0c0f3634685c` | dense-query-m4-k5-head-scale | 1 |
| `rcaff0dfb095f` | dense-query-m4-k5-head-scale-vdivsqrt5 | $1/\sqrt{5}$ |

Two more runs use $K = 9$ features with $s = 1/3$, at m = 4 (`r04a5b2a3e8dd`) and m = 8
(`rfccffdd239dd`); their recipes differ from the scaled $K = 5$ run only in $K$, $s$, and degree.

### Findings

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
- W decay on the feature projection leaves dense level or slightly worse: 14.739 (K = 5)
  and 14.798 (K = 9), in [query_optimization](../../../other/query_optimization/README.md#projection-decay-across-generators).

## Sigmoid coefficients

### Method

| Generator | Coefficients | Start |
| --- | --- | --- |
| Per degree | $c_r = \sigma(w_r^\top x_t)\,[/\,r!]$ | $W = 0$ |
| Dense, $K = 5$ | $c_r = [\tfrac{1}{\sqrt5\,r!}]\sum_j V_{rj}\,\sigma(w_j^\top x_t)$ | $W = 0$, $V = I$ |

There is no bias or amplitude; bracketed factors apply only with order scaling. At
$W = 0$ every $c_r$ starts at $1/2$, or $1/(2\,r!)$ with order scaling; head RMSNorm
removes the common factor, so order-scaled runs start at fixed Taylor. "W decay" applies
weight decay 0.1 to $W$ alone; $V$ stays exempt.

### Results

Best validation perplexity; terminal values are within 0.015 of it.

| Generator | Order scaling | No W decay | W decay 0.1 |
| --- | --- | ---: | ---: |
| Per degree | none | 16.103 (`rd5550f8db5e4`) | 15.919 (`re6d4c820a132`) |
| Per degree | $1/r!$ | 16.605 (`r8b4e2a0603aa`) | 16.406 (`r343fff99fa9e`) |
| Dense, $K = 5$ | none | 16.325 (`ra6ae537a5e37`) | 16.043 (`r331b5fd5bebc`) |
| Dense, $K = 5$ | $1/r!$ and $1/\sqrt5$ | 16.346 (`r239ea241c978`) | 16.467 (`rf7e1a83e2e23`) |

Controls: affine tanh query `r2b22f2ff6461`, 14.774; fixed Taylor `r816ab0fbe119`, 14.722.

- Every sigmoid generator trails the affine tanh query generator, by 7.8–12.4%.
- Order scaling hurts the per-degree generator (16.605 against 16.103) and leaves dense
  unchanged (16.346 against 16.325).
- W decay helps three of the four by 1.1–1.7% and hurts scaled dense by 0.7%.
- A pure sigmoid is affine tanh with $b_r = a_r = 1/2$ fixed, since
  $\sigma(z) = \tfrac12 + \tfrac12\tanh(z/2)$. The affine sigmoid in
  [coefficient_forms](../../../../exploratory/coefficient_forms/README.md) frees both and ties affine tanh.

## Open question

Keep the topic, or delete it with its runs and their artifacts.

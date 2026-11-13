# Sigmoid query coefficients (m = 4)

Opened 2026-10-26.

**Question.** Does a pure sigmoid coefficient generator, as used for the linear term in
polynomial GDN, train better than the affine tanh generator
$c_r = [b_r + a_r\tanh(w_r^\top x_t)]/r!$ at degree four?

## Method

Every run matches `query-m4-head-scale` (`r2b22f2ff6461`) except for the coefficient
generator: 1× Chinchilla, unnormalized degree-four attention with per-head output
RMSNorm, the canonical QK setup, Mosaic kernels, one GPU.

| Generator | Coefficients | Start |
| --- | --- | --- |
| Per degree | $c_r = \sigma(w_r^\top x_t)\,[/\,r!]$ | $W = 0$ |
| Dense, $K = 5$ | $c_r = [\tfrac{1}{\sqrt5\,r!}]\sum_j V_{rj}\,\sigma(w_j^\top x_t)$ | $W = 0$, $V = I$ |

There is no bias or amplitude; bracketed factors apply only with order scaling. At
$W = 0$ every $c_r$ starts at $1/2$, or $1/(2\,r!)$ with order scaling; head RMSNorm
removes the common factor, so order-scaled runs start at fixed Taylor. "W decay" applies
weight decay 0.1 to $W$ alone; $V$ stays exempt.

## Results

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
  [coefficient_forms](../coefficient_forms/README.md) frees both and ties affine tanh.

Runs and roles: [`selection.yaml`](selection.yaml).

## Open decision

Promote into its own canonical collection, or discard.

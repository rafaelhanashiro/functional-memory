# Coefficient forms (m = 4)

Opened 2026-10-27.

**Question.** At degree four, does the form of the query coefficient generator matter:
the affine generator $c_r = b_r + a_r f(w_r^\top x_t)$ against a pure one
$c_r = f(w_r^\top x_t)$, and tanh against sigmoid features $f$?

## Method

Both runs copy `query-m4-head-scale` (`r2b22f2ff6461`): 4B tokens, unnormalized
degree-four attention, per-head output RMSNorm, canonical QK RMSNorm with per-head
scaling, Mosaic kernels, one GPU. Each changes only the generator.

| Run | Label | Coefficients | Start |
| --- | --- | --- | --- |
| `r1d578d107115` | pure-tanh-m4 | $c_r = \tanh(w_r^\top x_t)$, no $b$, $a$ or $1/r!$ | $W \sim N(0, 1/d)$ |
| `r917f3650296e` | affine-sigmoid-m4 | $c_r = \big(\tilde b_r + a_r\,\sigma(w_r^\top x_t)\big)/r!$ | fixed Taylor: $a = 0$, $W \sim N(0, 1/d)$ |

The pure tanh run cannot start at $W = 0$: $\tanh(0) = 0$ zeroes every coefficient.
Its start is a random signed polynomial. The affine sigmoid generator equals the tanh
one up to reparameterization, $b + a\,\sigma(z) = (b + a/2) + (a/2)\tanh(z/2)$; only the
optimization differs.

Controls in the [selection](selection.yaml), all with per-head QK scaling:
`query-m4-head-scale` (14.779), `static-m4-head-scale` (14.773) and
`fixed-m4-head-scale` (14.724). Pure sigmoid coefficients are in the
`sigmoid_coefficients` topic on `claude/sigmoid-coefficients-4b`.

## Findings

Validation perplexity (terminal, update 7,570; best at 7,500 in parentheses):

| Run | 1,000 | 2,500 | 5,000 | Terminal |
| --- | ---: | ---: | ---: | ---: |
| Affine sigmoid `r917f3650296e` | 27.04 | 19.03 | 16.02 | **14.762** (14.756) |
| Pure tanh `r1d578d107115` | 77.73 | 33.32 | 25.14 | 22.638 (22.628) |
| Query (affine tanh) `r2b22f2ff6461` | 27.08 | 19.05 | 16.05 | 14.779 (14.774) |
| Static `r4339a245820e` | 27.51 | 19.11 | 16.04 | 14.773 (14.767) |
| Fixed Taylor `r816ab0fbe119` | 28.21 | 19.16 | 16.01 | 14.724 (14.722) |

- Affine sigmoid ties affine tanh (−0.1%) at every checkpoint, as the reparameterization
  predicts; the feature function does not matter once $b$ and $a$ are free.
- Pure tanh is **53%** worse than every affine form. Its start, a random signed
  polynomial with unit-scale coefficients, loses the first thousand updates (PPL 1,516 at
  update 100 against 27 for the affine runs) and never recovers; it was still improving
  at the end, so part of the gap is the start rather than the form.
- With the pure sigmoid runs 9–12% worse in the `sigmoid_coefficients` topic, every
  generator that removes the affine terms loses; none of the affine variants beats
  fixed Taylor.

## Open decision

Promote a generator form into its own canonical collection, or discard the topic.

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

Pending.

## Open decision

Promote a generator form into its own canonical collection, or discard the topic.

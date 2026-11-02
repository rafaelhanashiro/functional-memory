# Degree scaling at 1×

**Question.** How does held-out perplexity change with the polynomial degree
`m = 1, 2, 3, 4` for fixed Taylor, fixed Hermite, learned static, and query
coefficients with projection weight decay, under the Transformer++ recipe at 1×
Chinchilla?

## Runs

Every run uses the Transformer++ reference recipe at 1× (7,570 updates, warmup 455,
seed 42), the canonical QK setup (RMSNorm with a learned per-head score scale),
reference initialization, and unnormalized polynomial attention
`sum_(i<=t) sum_(r<=m) c_r s_ti^r v_i` followed by per-head output RMSNorm, on the
Mosaic kernels.

- **Fixed Taylor:** `c_r = 1/r!`.
- **Fixed Hermite:** the degree-`m` truncation of `exp(s) = exp(1/2) sum_r He_r(s)/r!`
  in monomial form, with the attention_scaling study's coefficients; for example
  `exp(1/2) (1/2 + s + s^2/2)` at `m = 2`.
- **Learned static:** one learned coefficient per order, layer and head, starting at
  Taylor.
- **Query, W decay:** `c_r(x_t) = [b_r + a_r tanh(w_r^T x_t)]/r!` per layer and head,
  starting at Taylor, with AdamW weight decay 0.1 on the projection `w_r`.

| Family | m = 1 | m = 2 | m = 3 | m = 4 |
| --- | --- | --- | --- | --- |
| Fixed Taylor | `ra7f2fda1c792` | `rdb1c706a5b85` | `r81fad1755b11` | `r816ab0fbe119` |
| Fixed Hermite | `rddf0b0173f42` | `rdf6363a4b83c` | `rf378c0e67607` | `rbd0cd771c676` |
| Learned static | `r5c691128699f` | `r99c3ab7d8a77` | `r1f880c4ee659` | `r4339a245820e` |
| Query, W decay | `rf9099d70ab3a` | `rd6b4d16d81bb` | `r00f1f91a7803` | `rca97cc9db8f2` |

Four cells reuse completed Kennel runs: Taylor at `m = 4` (B300) and static at `m = 4`
(H100 PCIe) trained with the earlier host-loop trainer, and query at `m = 2` (B300) and
`m = 4` (H100) with the current compiled loop. The twelve new runs differ from their family's `m = 4` recipe
only in degree and, for Hermite, the fixed coefficients. The `m = 1` column trains on
Kennel and the other new cells on ORCD H100s; the
[study README](../../README.md#degree-scaling) gives the launch commands.

Hermite at `m = 1` is `exp(1/2)` times Taylor at `m = 1`. The output RMSNorm removes
that factor, so the two runs train the same function up to the norm's epsilon and
floating point; their difference measures run-to-run variation, which the study has
not otherwise measured.

## Status

The twelve new runs are allocated and training. The figure, with degree on the x-axis
and terminal validation perplexity on the y-axis, one line per family, follows once
they finish, with an explicit selection, a summary table that also lists
validation-best perplexity, and regeneration code.

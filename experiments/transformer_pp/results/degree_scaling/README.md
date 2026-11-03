# Degree scaling at 1×

**Question.** How does held-out perplexity change with the polynomial degree
`m = 1, 2, 3, 4` for fixed Taylor, fixed Hermite, learned static, and query
coefficients with projection weight decay, under the Transformer++ recipe at 1×
Chinchilla?

![Terminal validation perplexity against polynomial degree for the four coefficient families](degree_scaling.png)

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
only in degree and, for Hermite, the fixed coefficients. The `m = 1` column trained on
Kennel B300s and the other new cells on ORCD H100s; the
[study README](../../README.md#degree-scaling) gives the launch commands. Softmax
(`r9040f12f3d80`, the same recipe with normalized softmax attention) is the reference
level in the figure.

Hermite at `m = 1` is `exp(1/2)` times Taylor at `m = 1`. The output RMSNorm removes
that factor, so the two runs train the same function up to the norm's epsilon and
floating point; their difference measures run-to-run variation, which the study has
not otherwise measured.

## Findings

<!-- BEGIN GENERATED RESULTS -->

**Terminal validation perplexity (update 7,570); lowest per degree in bold**

| Family | m = 1 | m = 2 | m = 3 | m = 4 |
| --- | ---: | ---: | ---: | ---: |
| Fixed Taylor | 19.556 | 15.912 | 15.974 | 14.724 |
| Fixed Hermite | 19.513 | 16.168 | **15.732** | 14.987 |
| Learned static | 19.531 | **15.822** | 16.649 | 14.773 |
| Query, W decay | **19.219** | 15.832 | 16.443 | **14.714** |

**Validation-best perplexity (update 7,500 in every run)**

| Family | m = 1 | m = 2 | m = 3 | m = 4 |
| --- | ---: | ---: | ---: | ---: |
| Fixed Taylor | 19.541 | 15.907 | 15.968 | 14.722 |
| Fixed Hermite | 19.495 | 16.164 | 15.722 | 14.984 |
| Learned static | 19.522 | 15.820 | 16.640 | 14.767 |
| Query, W decay | 19.205 | 15.827 | 16.436 | 14.709 |

**Change in terminal perplexity from the previous degree**

| Family | m = 1 | m = 2 | m = 3 | m = 4 |
| --- | ---: | ---: | ---: | ---: |
| Fixed Taylor | – | -18.6% | +0.4% | -7.8% |
| Fixed Hermite | – | -17.1% | -2.7% | -4.7% |
| Learned static | – | -19.0% | +5.2% | -11.3% |
| Query, W decay | – | -17.6% | +3.9% | -10.5% |

Softmax reference (`r9040f12f3d80`): terminal 14.546, validation-best 14.543 (update 7,500).

<!-- END GENERATED RESULTS -->

- **All sixteen runs finished the 7,570 updates** with no numerical failure. Every run's
  validation-best checkpoint is update 7,500, and its terminal perplexity is at most 0.1%
  higher; both metrics order the cells the same way.
- **Going from `m = 1` to 2 is the largest step in every family**, 17–19% lower
  perplexity, and every family is best at `m = 4`.
- **At `m = 4`, query (14.714), Taylor (14.724) and static (14.773) lie within 0.4% of
  one another**, inside the resolution below; Hermite (14.987) is 1.5–1.9% higher. All
  four stay 1.2–3.0% above softmax (14.546).
- **`m = 3` breaks the trend for the learned families.** Static (+5.2% over `m = 2`) and
  query (+3.9%) end worse at `m = 3` than at `m = 2`; Taylor is flat (+0.4%: its `m = 3`
  run led `m = 2` until update 6,000), and Hermite keeps improving (−2.7%). Both learned
  `m = 3` runs fall behind fixed Taylor at `m = 3` early, static from update 400 and
  query from the first evaluation at update 100, and from update 2,000 on stay
  0.040–0.045 nats (static) and 0.027–0.033 nats (query) behind it. At `m = 2` and 4 the
  learned runs end within 0.6% of fixed Taylor. All four `m = 3` runs passed acceptance,
  including the GPU kernel tests, on the H100s they trained on. The rises are 18–24 times
  the twin difference below in validation NLL, so they are not rounding; whether they are
  systematic or one unlucky seed is untested.
- **Run-to-run variation is about 0.2%.** Taylor and Hermite at `m = 1` end 0.042 apart
  (19.556 and 19.513). Learned static at `m = 1` adds nothing the per-head score scale
  cannot express while its constant coefficient stays positive, which it does in all 210
  heads, and it ends between them (19.531). Differences of a few tenths of a percent
  within a degree are not resolved.

## Limits

- One run (seed 42) per cell. The Taylor and Hermite `m = 1` twins measure numerical
  divergence between equivalent configurations, not variation across seeds or data
  orders, so the `m = 3` rises need a second seed of static at `m = 2` and 3 and query at
  `m = 3` before they are read as an effect of degree.
- Recipes are identical within each family, but execution differs: the `m = 1` column,
  Taylor at `m = 4` and query at `m = 2` trained on Kennel B300s; the other ORCD cells on
  H100 80GB HBM3s; query at `m = 4` on a Kennel H100 80GB HBM3 and static at `m = 4` on a
  Kennel H100 PCIe. Taylor and static at `m = 4` accumulated gradients with the earlier
  host loop, which computes the same sums with different rounding. Query's step from
  `m = 2` to 3 therefore also changes GPU and source commit; static's does not.
- The comparison is one 1× budget and schedule. The lower degrees were still closing the
  gap when training ended (Taylor at `m = 3` led `m = 2` until update 6,000), so the
  ranking may change with more tokens.
- The closest 10B-token analog in [attention_scaling](../../../../results/attention_scaling/README.md),
  learned coefficients with per-head output RMSNorm, improves at every degree (14.630 at
  `m = 2`, 13.988 at `m = 3`), so a penalty at `m = 3` does not carry across recipes.

## Regenerate

From the repository or worktree root:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/degree_scaling
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.degree_scaling.plot
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest -q experiments/transformer_pp/results/degree_scaling/tests
```

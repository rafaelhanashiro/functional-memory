# Input-dependent polynomial coefficients

Opened 2026-10-07. Updated 2026-10-19. Home cluster: **Kennel**.

Does letting the query token choose the polynomial coefficients improve unnormalized
attention with per-head output RMSNorm? The first tanh runs lost to their static
controls at every degree, so three checks followed: the same recipe on the plain-JAX
blocked reference (no custom kernel), a start with zero tanh amplitude, two
changes to the gate's initialization from that start, and an additional per-key factor.

## Runs

Coefficients are `c_r(x_t) = b_r + a_r tanh(w_r^T x_t)` for degrees `r = 0, ..., m`,
per layer and head, from the activation after the block input RMSNorm; `x_t` is the
query token. Static controls learn a constant `c_r = b_r`. Every run starts at
`b_r = 1/r!`.

| Role | Runs (m = 1, 2, 3, 4) | Start | Attention path |
| --- | --- | --- | --- |
| Static reference | `r03bfa6fd7b2e`, `rfb43d69f60dc`, `r5ae483d40d62`, `rc2b555b905dd`; m = 8: `raa043e978a3d` | – | fused kernel |
| Affine reference (m = 1) | `r9cf9589b9855` | `c = b + W x_t`, `W = 0` | fused kernel |
| Tanh, original start | `r80300a4cd5fa`, `rbfe1750cc24c`, `r0bc57959ef6c`, `rfee72428a45e` | `a_r = 1`, `w_r = 0` | fused kernel |
| Tanh, original start | `r6fdf6d484fd2`, `r722fc3dd263e`, `radd66df6371a`, `rc836509ca119` | `a_r = 1`, `w_r = 0` | blocked reference |
| Tanh, zero-amplitude start | `r75b81509b10b`, `r8c9b076f9635`, `rb306d49a5abe`, `rf1160aa95e46` | `a_r = 0`, `w_r ~ N(0, 0.002^2)` | blocked reference |
| Tanh, zero-amplitude start, wide draw | `r5be9a97f06e2`, `rec520e9303ac`, `r215d3d91dd02`, `rd4f66b1959d2`; m = 8: `rccf19c31669d` | `a_r = 0`, `w_r ~ N(0, 1/1024)` | Mosaic kernel (m = 8: Triton) |
| Tanh, zero-amplitude start, bias inside tanh | `rbd4eb6ccc16f`, `r6bba734352f0`, `rde534cbb87a1`, `r59490b6e0818` | `a_r = 0`, `w_r ~ N(0, 0.002^2)`, learned `beta_r` in `tanh(w_r^T x_t + beta_r)` from 0 | Mosaic kernel |
| Tanh, zero-amplitude start, saturating draws | `rac317a500871`, `r4cee946a1a4b` (m = 2, 4; `3/sqrt(d)`); `r49ac1e58fba3`, `r362b2df6edbd` (`10/sqrt(d)`) | `a_r = 0`, `w_r ~ N(0, 9/d)` or `N(0, 100/d)` | Mosaic kernel |
| Tanh, zero-amplitude start, amplitude `a_r/r!` | m = 8: `ra39c41604704` | as the wide draw, with `c_r = b_r + (a_r/r!) tanh(w_r^T x_t)` | Mosaic kernel |
| Tanh query-key | `rba839e6386b7`, `r0aab2b77017f`, `r4a7c601b4b1f`, `r4e3f4097a60e`; m = 8: `r66db587096c4` (one-pass Triton kernel) | as the wide draw, times `f_r(x_i) = 1 + a^k_r tanh(w_k_r^T x_i)` with `a^k_r = 0` | blocked reference (m = 1), Mosaic kernel once per order |

All 37 runs are complete at 19,074 updates (10.0B tokens) with the shared recipe:
`transformer_reference` initialization, QK normalization, unnormalized attention,
per-head output RMSNorm, `d_K = d_V = 64`, 10 layers, width 1024, 16 heads, seed 0,
BF16, effective batch 256, global microbatch 64, learning rate 3e-4 to 3e-5 with 400
warmup updates, FP32 coefficient parameters without weight decay. The original fused
runs used four GPUs; the blocked-reference runs one B300 each; the Mosaic-kernel runs
one B300 or H100 each with global microbatch 32 instead of 64. [selection.yaml](selection.yaml)
names the roles and controls and pins each run's scientific inputs; recipes, records
and launch commands are in the
[study README](../../../experiments/attention_scaling/README.md#tanh-coefficients-degrees-14).

## Findings

**Original start loses to static; the kernel is not the cause.** On the fused kernel
the tanh runs end 7.5%, 4.7%, 1.7% and 4.2% above static at degrees one to four. The
blocked reference reproduces them: 18.77 vs 18.87, 15.28 vs 15.32, 14.28 vs 14.22 and
14.14 vs 14.21 (blocked vs fused). Differences of at most 0.1 in both directions are
within run-to-run variation.

**A zero-amplitude start removes the deficit.** With `a_r = 0` and a small random
projection the runs end at 17.35, 14.56, 13.96 and 13.63 against static's 17.55,
14.63, 13.99 and 13.64: a tie within noise at degrees two to four (reruns of one
recipe differ by about 0.01) and 0.2 better at degree one.

**How the gate starts does not matter.** From the zero-amplitude start, drawing the
projection 15× wider (`w_r^T x_t` of unit scale, so tanh starts in its active range) or
adding a learned bias inside the tanh ends at 17.54 / 14.56 / 13.96 / 13.62 and
17.54 / 14.55 / 13.98 / 13.63: the same as static and as the small draw, except that
the small draw's 0.2 lead at degree one does not repeat, so it was run-to-run variation.
Both variants learn the same `b`, `|a_r|` and token variation as the small draw; the wide
draw keeps `||w_r|| ~ 1.25` but pointed along the constant direction of `x_t`, and the
inner bias stays within ±0.1. The optimizer removes token dependence rather than
being stuck near the start. Saturating draws (`w_r^T x_t` of standard deviation 3 or 10 at
the start, so the tanh begins as a near-binary gate whose input weights get little
gradient) end at 14.58 / 13.63 (`3/sqrt(d)`) and 14.59 / 13.65 (`10/sqrt(d)`) at degrees two
and four, no better than the 1/d draw, with smaller amplitudes and less token variation the
larger the draw (at degree four the median `c_0` variation falls from 14% to 8% and 3%).

**A per-key factor adds at most a few hundredths.** Multiplying each key's degree-`r`
term by `f_r(x_i)` (rank-one query–key coefficients, no recurrence) ends at 17.56 / 14.52 /
14.00 / 13.58 against 17.35–17.54 / 14.55–14.56 / 13.96–13.98 / 13.62–13.63 for the three
query-only starts: 0.03–0.04 better at degrees two and four, level with static at degree
one, and 0.02 worse at degree three. The key factors vary more across tokens than the query
coefficients (1–10% against 0.3–1%) but much of their amplitude again acts as a per-head
constant. At degree one the per-order kernel passes round each partial output to BF16, which
corrupts the gradient after head RMSNorm, so that degree runs on the blocked reference.

**At degree 8 input dependence costs perplexity.** Static m = 8 ends at 13.46, better than
every m = 4 variant; query–key ends at 13.49 and query-only at 13.56 (+0.8%). Unlike lower
degrees, query-only m = 8 modulates its coefficients strongly from the start: by update
1,000 the coefficients of orders 4–8 vary 25–72% across tokens with amplitudes as large as
their static parts, and at the end every order varies 19–82%. Adam moves every amplitude by
about the learning rate per update, which is large relative to the small high-order
coefficients (`1/r!` at the start); query-only m = 8 also clips its gradient more often
throughout (52% against 30% of updates in the sixth thousand), which slows the whole model.

**Order scaling removes most of the degree-8 deficit but does not beat static.** Dividing the
amplitude by `r!` (`attention_coefficient_amplitude_scale: factorial`), so that under Adam
order `r`'s modulation moves at `lr/r!`, ends at 13.48 (+0.11% against static's 13.46;
unscaled query-only +0.76%). Orders 5–8 stay static (median effective amplitude 0.1–6% of
`b_r`, token variation at most 4%), while orders 0–4 vary 17–39% across tokens, more than
unscaled query-only at orders 0–1. It clips 34% of logged updates against 26% for static
and 42% for unscaled query-only. Its high-order static parts move as far from Taylor as
static's (`b_8` about 300× `1/8!`).

**The learned coefficients are essentially static.** From the zero-amplitude start,
`b` stays near its Taylor values, the amplitudes stay small (median `|a_r|` 0.06–0.11)
and the coefficients barely vary across tokens: their standard deviation across the
pinned held-out sequence, relative to their mean, has median 0.4% at degree one and
5.6% at degree four (`median_token_variation` in [summary.csv](summary.csv)). The
original start keeps `|a_r|` near 1 and varies 14–35% across tokens, with worse
perplexity. Head RMSNorm removes any common per-token scale of all `c_r`, so only
their ratios can act.

<!-- BEGIN GENERATED RESULTS -->

| Degree | Variant | Terminal validation NLL | Terminal PPL | vs static | Best validation PPL | Best step |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | static | 2.865257 | 17.553569 |  | 17.553569 | 19,074 |
| 1 | affine | 2.941899 | 18.951802 | +7.97% | 18.951802 | 19,074 |
| 1 | tanh, original start, fused kernel | 2.937520 | 18.869000 | +7.49% | 18.869000 | 19,074 |
| 1 | tanh, original start, blocked reference | 2.932368 | 18.772035 | +6.94% | 18.772035 | 19,074 |
| 1 | tanh, zero-amplitude start, blocked reference | 2.853468 | 17.347837 | -1.17% | 17.347837 | 19,074 |
| 1 | tanh, zero-amplitude start, W ~ N(0, 1/d), fused kernel | 2.864399 | 17.538504 | -0.09% | 17.538504 | 19,074 |
| 1 | tanh, zero-amplitude start, bias inside tanh, Mosaic kernel | 2.864280 | 17.536430 | -0.10% | 17.536430 | 19,074 |
| 1 | tanh query-key, zero-amplitude start, W ~ N(0, 1/d) | 2.865312 | 17.554538 | +0.01% | 17.554538 | 19,074 |
| 2 | static | 2.683093 | 14.630271 |  | 14.630271 | 19,074 |
| 2 | tanh, original start, fused kernel | 2.728845 | 15.315193 | +4.68% | 15.315193 | 19,074 |
| 2 | tanh, original start, blocked reference | 2.726419 | 15.278076 | +4.43% | 15.278076 | 19,074 |
| 2 | tanh, zero-amplitude start, blocked reference | 2.678077 | 14.557070 | -0.50% | 14.557070 | 19,074 |
| 2 | tanh, zero-amplitude start, W ~ N(0, 1/d), fused kernel | 2.678387 | 14.561579 | -0.47% | 14.561579 | 19,074 |
| 2 | tanh, zero-amplitude start, bias inside tanh, Mosaic kernel | 2.677429 | 14.547647 | -0.56% | 14.547647 | 19,074 |
| 2 | tanh, zero-amplitude start, W ~ N(0, 9/d), Mosaic kernel | 2.679921 | 14.583944 | -0.32% | 14.583944 | 19,074 |
| 2 | tanh, zero-amplitude start, W ~ N(0, 100/d), Mosaic kernel | 2.680354 | 14.590263 | -0.27% | 14.590263 | 19,074 |
| 2 | tanh query-key, zero-amplitude start, W ~ N(0, 1/d) | 2.675586 | 14.520851 | -0.75% | 14.520851 | 19,074 |
| 3 | static | 2.638175 | 13.987648 |  | 13.987648 | 19,074 |
| 3 | tanh, original start, fused kernel | 2.654816 | 14.222367 | +1.68% | 14.222367 | 19,074 |
| 3 | tanh, original start, blocked reference | 2.659099 | 14.283418 | +2.11% | 14.283418 | 19,074 |
| 3 | tanh, zero-amplitude start, blocked reference | 2.636361 | 13.962307 | -0.18% | 13.962307 | 19,074 |
| 3 | tanh, zero-amplitude start, W ~ N(0, 1/d), fused kernel | 2.636479 | 13.963953 | -0.17% | 13.963953 | 19,074 |
| 3 | tanh, zero-amplitude start, bias inside tanh, Mosaic kernel | 2.637757 | 13.981805 | -0.04% | 13.981805 | 19,074 |
| 3 | tanh query-key, zero-amplitude start, W ~ N(0, 1/d) | 2.638958 | 13.998613 | +0.08% | 13.998613 | 19,074 |
| 4 | static | 2.612946 | 13.639175 |  | 13.639175 | 19,074 |
| 4 | tanh, original start, fused kernel | 2.654150 | 14.212903 | +4.21% | 14.212903 | 19,074 |
| 4 | tanh, original start, blocked reference | 2.649159 | 14.142133 | +3.69% | 14.142133 | 19,074 |
| 4 | tanh, zero-amplitude start, blocked reference | 2.611894 | 13.624832 | -0.11% | 13.624832 | 19,074 |
| 4 | tanh, zero-amplitude start, W ~ N(0, 1/d), fused kernel | 2.611755 | 13.622937 | -0.12% | 13.622937 | 19,074 |
| 4 | tanh, zero-amplitude start, bias inside tanh, Mosaic kernel | 2.612069 | 13.627211 | -0.09% | 13.627211 | 19,074 |
| 4 | tanh, zero-amplitude start, W ~ N(0, 9/d), Mosaic kernel | 2.612470 | 13.632688 | -0.05% | 13.632688 | 19,074 |
| 4 | tanh, zero-amplitude start, W ~ N(0, 100/d), Mosaic kernel | 2.613763 | 13.650314 | +0.08% | 13.650314 | 19,074 |
| 4 | tanh query-key, zero-amplitude start, W ~ N(0, 1/d) | 2.608653 | 13.580743 | -0.43% | 13.580743 | 19,074 |
| 8 | static | 2.599805 | 13.461116 |  | 13.461116 | 19,074 |
| 8 | tanh, zero-amplitude start, W ~ N(0, 1/d), fused kernel | 2.607381 | 13.563477 | +0.76% | 13.563477 | 19,074 |
| 8 | tanh, zero-amplitude start, W ~ N(0, 1/d), amplitude a_r/r!, Mosaic kernel | 2.600935 | 13.476329 | +0.11% | 13.476329 | 19,074 |
| 8 | tanh query-key, zero-amplitude start, W ~ N(0, 1/d) | 2.601888 | 13.489182 | +0.21% | 13.489182 | 19,074 |

<!-- END GENERATED RESULTS -->

![Degree-matched validation learning curves](learning_curves.png)

[PDF](learning_curves.pdf) · [Derived measurements and input checksums](summary.csv)

Limitations: one seed, one bounded nonlinear parameterization and one shared optimizer
recipe; the affine and original-start comparisons mix hardware and attention
implementations, while the zero-amplitude runs differ from their controls in the
attention path. Query-side coefficients reshape only the query's kernel; none of these
runs conditions on the key (writer) as GDN's write gate does.

## Open decision

Query-dependent coefficients do not beat static coefficients with this recipe, with or
without a kernel, from any of three starts, and a per-key factor changes the outcome by
at most a few hundredths, and at degree 8 query-dependent coefficients are worse than
static. Order scaling (`c_r = b_r + a_r tanh(.)/r!`) removes most of the degree-8 deficit
but only ties static. Open: whether order-scaled query–key or degree 12 changes this, and
then whether to promote the comparison as a negative result or discard the topic and its runs.

## Regeneration

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/exploratory/input_dependent_coefficients
scaffolds/transformer/.venv/bin/python -m results.exploratory.input_dependent_coefficients.plot
```

The plot verifies the pinned inputs, then writes the figure, `summary.csv`
(terminal and validation-best metrics, matched-control differences, coefficient
statistics; the inferred tanh moments `saturated_mean_branch_fraction` and
`mean_tanh_derivative_from_moments` divide by the amplitude and are not informative
where it is near zero) and the generated table above. Learning curves use every recorded
validation without smoothing. Measurements and full coefficient diagnostics stay in
each run's `metrics.jsonl`.

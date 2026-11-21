# Query coefficients across key width and degree (Transformer++)

Opened: **2026-10-31**. Home cluster: **Kennel**.

How do the key width `d_K` and the polynomial degree `m` shape validation perplexity for
tanh query-conditioned coefficients with a decayed projection `W`, the best query
parameterization at `d_K = 64`, `m = 4` (`query-m4-w-decay`, `rca97cc9db8f2`, 14.71)? This is
the Transformer++ counterpart of the key-width-by-degree heatmap in
[attention_scaling](../../../../../../results/attention_scaling/README.md), restricted to this one
mechanism. The runs live in the [Transformer++ study](../../../../README.md).

## Runs

| `d_K` \ `m` | 2 | 4 | 8 | 12 |
| --- | --- | --- | --- | --- |
| 8 | `r7bede0f75198` | `r1c80ab3a466b` | `r16b23893884d` | `r1b1123bf96e0` |
| 16 | `rc90fb11d4772` | `rb5fe0aef572a` | `rd10c2ebe1181` | `rcdfee1e67fd4` |
| 32 | `r18d67098c780` | `rcb1f56f31981` | `rf5b51bd384bc` | `r4f3aa6be7108` |
| 64 | `rd6b4d16d81bb` | `rca97cc9db8f2` | `r25fffc16c2f3` | `r0fceea0627c8` |

Every run is `query-m4-w-decay`'s recipe with `qk_head_dim` and `attention_degree` changed:
degree-`m` unnormalized polynomial attention with tanh query coefficients
`c_r(x) = (b_r + a_r tanh(W_r x)) / r!` (zero amplitude at initialization, so the kernel starts
as Taylor; `W` drawn `N(0, 1/896)` and decayed with `optim.weight_decay` 0.1; `b` and `a` in
factorial coordinates, undecayed), 14 heads with value width 64 (only the query and key width
changes, so narrower keys shrink the Q/K projections), canonical QK RMSNorm with per-head
scaling, head RMSNorm output scaling, Mosaic kernels, reference initialization, and the study's
recipe: 7,570 updates of 256 sequences, learning rate 0.001 with cosine decay to 0.0001,
warmup 455, weight decay 0.1. The `d_K = 64`, `m = 4` cell reuses `rca97cc9db8f2`, whose recipe
is identical up to run names. The `m = 12` column uses the largest degree the Mosaic query
kernels are qualified for.

Launch commands are in the [study README](../../../../README.md#query-coefficients-across-key-width-and-degree).

## Findings and open question

All fifteen new runs completed their 7,570 updates on one GPU each (B300 or H100; 2.4–4.7 h
at `m <= 8`, 3.9–5.4 h at `m = 12`). Validation-best perplexity; the best checkpoint is update
7,500 except at `m = 2`, `d_K = 8` and `16` (terminal, 7,570); terminal values are in
`summary.csv`.

![Validation-best perplexity across key width and degree](heatmap.png)

| `d_K` \ `m` | 2 | 4 | 8 | 12 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 19.396 | 17.430 | 16.854 | 16.681 |
| 16 | 17.205 | 15.768 | **15.413** | 15.450 |
| 32 | 16.146 | 15.225 | 15.046 | 14.803 |
| 64 | 15.827 | 14.709 | 14.323 | **14.226** |

- **Wider keys always help.** Every column falls monotonically from `d_K = 8` to 64.
- **Degree helps with diminishing returns.** Going from `m = 2` to 4 lowers perplexity by
  10.1% at `d_K = 8` and 7.1% at `d_K = 64`; from 4 to 8, by 3.3% and 2.6%; from 8 to 12, by
  1.0% (`d_K = 8`), 1.6% (32) and 0.7% (64), while `d_K = 16` is the one cell that gets worse
  (+0.2%).
- **Width keeps paying at high degree.** At `m = 8` and 12, doubling `d_K` from 32 to 64 still
  gains 4.8% and 3.9%; at `m = 2` the same step gains 2.0%.
- **The best cell, `d_K = 64`, `m = 12` (14.226), is 3.4% below fixed Taylor at `m = 4`**
  (`r816ab0fbe119`, 14.722) and slightly below unnormalized exp (`r6a76e7fbde45`, 14.257), the
  forget gate at `m = 4` with RoPE (`r111843b4c7cc`, 14.318), and query-key coefficients at
  `m = 12` (`r2459b843ca81`, 14.232, on ORCD). Outside the degree-eight column, which the
  [ladder below](#narrow-keys-for-exp-softmax-softplussos-and-a-degree-eight-ladder) matches with
  fixed-Taylor and static runs, the grid does not separate the contribution of query dependence
  from that of degree and width.

**Open question:** whether to add matched fixed-Taylor (or static) cells at the other degrees, rerun
query `p_t(s)` at `d_K` 4 with head RMSNorm's max-abs prescale
([limits below](#narrow-keys-for-exp-softmax-softplussos-and-a-degree-eight-ladder)), or delete the topic.

Regenerate the heatmap and `summary.csv` from the pinned selection:

```bash
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.pretraining.polynomial.query_width_degree.plot
```

## Narrow keys for exp, softmax, softplus–SoS and a degree-eight ladder

Added 2026-11-20. Does the key-width penalty above hold for the exponential, for softmax and for the best
sum-of-squares kernel, and which ingredient matters at narrow keys: the polynomial form, learned coefficients or
their input dependence? Every run changes only `qk_head_dim` (value width 64) of a partner: exp `r6a76e7fbde45`;
softmax `r9040f12f3d80`; `r841f35a24241`, `z h_t(z)` with `z = softplus(s)` and a query sum-of-squares core of
degree eight (total degree nine); and a degree-eight ladder on the raw score that adds one ingredient per step:

- fixed Taylor `T_8(s) = sum_(r<=8) s^r / r!`, exp's truncation with nothing learned (`ra7ee8a4b2544`);
- static learned `p(s)`, from that start (`r9c69dd7cfc8d`);
- tanh query coefficients `p_t(s)` with W decay, the grid's degree-eight column, which gains `d_K` 4 (from
  `r16b23893884d` at `d_K` 8).

Validation-best perplexity, every one at update 7,500 (terminal values are in each run's `completion.json`):

| `d_K` | `z h_t(z)` | exp | softmax | `T_8(s)` | static `p(s)` | query `p_t(s)` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 19.463 | **19.190** | 19.330 | 20.378 | 19.994 | 19.925 |
| 8 | **16.309** | 16.397 | 16.754 | 16.834 | 16.850 | 16.854 |
| 16 | **15.054** | 15.129 | 15.658 | 15.313 | 15.385 | 15.413 |
| 64 | **14.095** | 14.257 | 14.543 | 14.218 | 14.294 | 14.323 |

| `d_K` | `z h_t(z)` | exp | softmax | `T_8(s)` | static `p(s)` | query `p_t(s)` |
| ---: | --- | --- | --- | --- | --- | --- |
| 4 | `rabfbab32a594` | `r2739c80cc032` | `reaca8ac9420c` | `r1eaa4a4c3c77` | `r99b1eebb5254` | `rd5e3ec5a4f02` |
| 8 | `r8d6683db38a8` | `r2a45c9e3d452` | `r6f65dc73550a` | `r5e540d8620b1` | `r161f07127f20` | `r16b23893884d` |
| 16 | `r79abb4d7292c` | `rfdc104924c7c` | `r328466965c2b` | `rcc9e0b03e692` | `r576bb8758634` | `rd10c2ebe1181` |
| 64 | `r841f35a24241` | `r6a76e7fbde45` | `r9040f12f3d80` | `ra7ee8a4b2544` | `r9c69dd7cfc8d` | `r25fffc16c2f3` |

Exp at `d_K` 4, 8 and 16 subtracts each row's maximum score (`attention_exp_shift: row_max`), and the `d_K` 4
`z h_t(z)` and static `p(s)` and the `d_K` 4 and 8 `T_8(s)` cells use head RMSNorm's max-abs prescale
(`attention_head_rms_prescale: max_abs`); both compute the partner's function ([numerics](#numerics-at-narrow-keys)).

- **Exp holds up at narrow keys.** It is the best kernel at `d_K` 4 and 0.5% behind the best, `z h_t(z)`, at 8
  and 16. Narrowing costs it 6.1% (64 to 16), 15.0% (to 8) and 34.6% (to 4).
- **`z h_t(z)` leads exp by 0.5% at `d_K` 8 and 16**, as it does by 1.1% at 64, **and trails it by 1.4% at 4**,
  where softmax (+0.7% against exp) is also ahead of it.
- **The raw-score polynomials fall behind as keys narrow.** Level with exp at `d_K` 64 (−0.3% to +0.5%), they
  trail it by 1.2–1.9% at 16, 2.7–2.8% at 8 and 3.8–6.2% at 4.
- **Learned coefficients pay at `d_K` 4, and input dependence adds little to them**: static `p(s)` is 1.9% below
  `T_8(s)` there and query coefficients 2.2% below it; query is 0.3% below static, within one seed's noise.
- **At wider keys nothing learned beats fixed Taylor**: static `p(s)` is 0.1–0.5% above `T_8(s)` at `d_K` 8, 16
  and 64, and query coefficients 0.1–0.7% above it.
- **Softmax trails exp at every width**: by 0.7% at `d_K` 4, 2.2% at 8, 3.5% at 16 and 2.0% at 64.
- Limits: one seed each, so margins near 0.5–1% are within one seed's noise. Query `p_t(s)` at `d_K` 4 trained
  with the plain head RMSNorm arithmetic. Its recorded attention row sums stayed below `2e14` (largest `1e13`) and
  its gradient norm spiked twice (27 and 62), but recorded row sums did not predict what the plain arithmetic cost
  the static run ([below](#numerics-at-narrow-keys)), so 19.925 is not known to be exact.

### Numerics at narrow keys

Narrow keys train much larger scores: the largest score at the final validation grows from 21–90 at `d_K` 64 to
87–815 at `d_K` 4, and the attention output before head RMSNorm with it. Two FP32 evaluations fail there, and
each has an exact replacement:

- `e^s` overflows FP32 near `s = 88.7`. `attention_exp_shift: row_max` evaluates `e^(s − m)` with the row's
  maximum score `m`, which head RMSNorm divides out.
- Plain head RMSNorm's backward is inexact from an output RMS of about `2e14` and loses its radial term from
  `9e14`, which adds a spurious gradient to the score-scale parameters
  ([measurements](../../../../../../scaffolds/transformer/validation/README.md#head-rmsnorm-backward-at-run-checkpoints-2026-11-20)).
  `attention_head_rms_prescale: max_abs` divides each head's output by its largest magnitude first.

| Kernel, `d_K` | Plain evaluation | Exact evaluation | Change |
| --- | --- | --- | ---: |
| exp, 4 | `ra28b295d81e6`, numerical failure at update 3,954 | `r2739c80cc032`, 19.190 | – |
| exp, 8 | `r535bc84a7772`, 20.571 | `r2a45c9e3d452`, 16.397 | −20.3% |
| exp, 16 | `r6f84b1db6de6`, 15.836 | `rfdc104924c7c`, 15.129 | −4.5% |
| `z h_t(z)`, 4 | `r5a25f500bfa8`, 19.413 | `rabfbab32a594`, 19.463 | +0.3% |
| `T_8(s)`, 4 | `re3515eed3576`, 20.584 | `r1eaa4a4c3c77`, 20.378 | −1.0% |
| `T_8(s)`, 8 | `r858c094cb85f`, 16.852 | `r5e540d8620b1`, 16.834 | −0.1% |
| static `p(s)`, 4 | `r0f5e38b0d934`, 20.439 | `r99b1eebb5254`, 19.994 | −2.2% |

- The unshifted exp runs are not valid controls at `d_K` 4–16: the first overflowed (scores reached 74 by update
  3,500), and the other two trained with attention row sums up to `2e29` and `2e23`, far inside the inexact
  range. The 20% and 5% leads `z h_t(z)` showed over them were this defect.
- The prescale removed the gradient spikes of plain `T_8(s)` and plain static `p(s)` at `d_K` 4 (largest gradient
  norms 733 and 1,035, against 13.6 at the start of both reruns) and 1.0% and 2.2% of their perplexity; `z h_t(z)`
  at `d_K` 4 did not improve, although its row sums grew to `7e17`, past the `5e15` at which the plain run's had
  stalled.
- The plain static run spiked from update 1,400, when its recorded row sums were near `1e8`, and passed `2e14`
  only at update 4,500. The recorded row sums come from one held-out sequence per validation, so they do not
  bound what training rows reach, and a plain run with gradient spikes is suspect whatever they show.
- Narrower keys remove 18–23 M Q/K parameters and run QK RMSNorm over 4–16 channels; every other initial array is
  the partner's. `z h_t(z)` trains on the Mosaic transformed kernels (B200 or H100), the ladder on the Mosaic
  query-coefficient kernel (`d_K` 4 on B200 or H100, 8 and 16 on B300 or H100), exp on the fused kernel and softmax
  on cuDNN, which zero-pads Q and K to width 64 (both B300 or H100); the exp runs add gradient-event capture to
  their October control (methods and launch:
  [B1-B6](../../../../README.md#score-prefactors-swish-and-narrow-keys),
  [shifted exp and softmax](../../../../README.md#shifted-exp-and-softmax-at-narrow-keys),
  [the ladder and shifted exp at `d_K` 8 and 16](../../../../README.md#degree-eight-ladder-and-shifted-exp-at-narrow-keys),
  [max-abs reruns](../../../../README.md#max-abs-head-rmsnorm-at-qk-widths-4-and-8)).
- `plot.py` reads the `query_decay_*` roles only, so the heatmap and `summary.csv` above keep to the grid.

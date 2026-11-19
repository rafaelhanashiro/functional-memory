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
  `m = 12` (`r2459b843ca81`, 14.232, on ORCD). Without fixed-Taylor or static runs at the other
  cells, the grid does not separate the contribution of query dependence from that of degree
  and width.

**Open question:** whether to add matched fixed-Taylor (or static) cells to isolate query dependence,
or delete the topic.

Regenerate the heatmap and `summary.csv` from the pinned selection:

```bash
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.pretraining.polynomial.query_width_degree.plot
```

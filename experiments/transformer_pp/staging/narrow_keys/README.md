# Narrow keys for exp, softmax, softplus–SoS and a degree-eight ladder

Opened 2026-11-20. Does the key-width penalty of the query grid
([query_width_degree](../../results/pretraining/polynomial/query_width_degree/README.md)) hold for the exponential, for softmax and for the best
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
  with the plain head RMSNorm arithmetic. Its recorded attention row sums stayed below `2e14` (largest `1e13`), but
  its gradient norm spiked twice (27 and 62), so 19.925 is not known to be exact.

## The parts of `z h_t(z)` at `d_K` 4

Opened 2026-11-21. Which part of `z h_t(z)` matters at `d_K` 4, where it trails exp by 1.4%: the prefactor `z`
or the softplus input? Two runs drop them in turn from `rabfbab32a594`. Each is its own `d_K` 64 partner at width
4 with the max-abs prescale, and differs from `rabfbab32a594` only in the parts named:

| Kernel | `d_K` 4 run | `d_K` 64 partner, best PPL | Change from `z h_t(z)` |
| --- | --- | --- | --- |
| `h_t(z)`, `z = softplus(s)` | `r8dca728e7c99` | `r41a4c74c5f58`, 14.339 | no prefactor (degree eight) |
| `h_t(s)` | `reea3d6436271` | `reeefbcc19abf`, 14.196 | no prefactor, raw score |

At `d_K` 64 they rank `z h_t(z)` (14.095) ahead of `h_t(s)` and `h_t(z)`. `h_t(z)` trains on the Mosaic
transformed kernels and `h_t(s)` on the query-coefficient kernel, which takes the expanded squares' coefficients;
both on B200 or H100 ([launch](../../README.md#parts-of-the-softplus-sos-kernel-at-key-width-four)).

Both completed 7,570 updates without numerical failure. Validation-best PPL (terminal in each run's
`completion.json`):

| Kernel at `d_K` 4 | Best PPL | Against `z h_t(z)` | Against exp |
| --- | ---: | ---: | ---: |
| `z h_t(z)` (`rabfbab32a594`) | 19.463 | – | +1.4% |
| `h_t(z)` (`r8dca728e7c99`) | 19.753 | +1.5% | +2.9% |
| `h_t(s)` (`reea3d6436271`) | 19.762 | +1.5% | +3.0% |

- **The prefactor `z` carries the gain.** Without it, the softplus input and the raw score tie (19.753 against
  19.762); with it, `z h_t(z)` is 1.5% better than either.
- **At `d_K` 64 the prefactor also leads, by less**: `z h_t(z)` 14.095, `h_t(s)` 14.196 (+0.7%), `h_t(z)` 14.339
  (+1.7%); there the softplus input without the prefactor trails the raw score.
- **Both reductions stay ahead of the raw-score ladder** at `d_K` 4 (query `p_t(s)` 19.925, static `p(s)` 19.994).
- One seed each, so differences below about 0.5% are within one seed's noise.

`r2e7f5617fdf2` keeps `z h_t(z)` and doubles its core's degree to sixteen (total seventeen). Its only other change
is the evaluation: in units of each row's largest power (`attention_polynomial_shift: row_power`), on its fused
kernel, because the Mosaic kernels stop at degree twelve and `z^17` overflows FP32 at the scores width 4 reaches
([kernel](../../../../scaffolds/transformer/validation/README.md#row-power-shifted-query-polynomials-2026-11-21)).
It completed 7,570 updates without numerical failure:

| Kernel at `d_K` 4 | Best PPL | Against `z h_t(z)` | Against exp |
| --- | ---: | ---: | ---: |
| `z h_t(z)`, core sixteen (`r2e7f5617fdf2`) | 19.192 | −1.4% | +0.0% |

- **Doubling the core's degree ties exp at `d_K` 4** (19.192 against 19.190), the first kernel here to do so;
  softmax trails it by 0.7%.
- **The gain is steady**: 1.5–1.7% below the degree-eight core at every validation from update 2,000, where the
  degree-eight core sat 1.4–1.6% above exp.
- One seed; the tie is within one seed's noise.

## Degree sixteen at `d_K` 4

The tanh query polynomial at degree sixteen (`raec1c513a51b`) reaches 19.777, 0.7% below its degree-eight run
(19.925), and fixed Taylor `T_16` (`rca649583af62`) 19.829, 2.7% below `T_8` (20.378); exp stays ahead at 19.190.
Both train on the row-power shifted kernel and are pinned, with the pure tanh polynomials at the same degrees, in
[pure_tanh_prefactors](../pure_tanh_prefactors/README.md).

## Numerics at narrow keys

Narrow keys train much larger scores (the largest score at the final validation grows from 21–90 at `d_K` 64 to
87–815 at `d_K` 4), and with them the attention output before head RMSNorm. Two FP32 evaluations fail there, so the
cells named above use exact evaluations of the same function:

- `e^s` overflows FP32 near `s = 88.7`, and before that its outputs reach the inexact range of the next bullet.
  `attention_exp_shift: row_max` evaluates `e^(s − m)` with the row's largest score `m`, which head RMSNorm divides
  out. Without it, exp at `d_K` 4 overflowed at update 3,954; with it, exp's perplexity is 20% and 4.5% lower at
  `d_K` 8 and 16.
- Plain head RMSNorm's FP32 backward is inexact from an output RMS of about `2e14` and loses its radial term from
  `9e14` ([measurements](../../../../scaffolds/transformer/validation/README.md#head-rmsnorm-backward-at-run-checkpoints-2026-11-20)).
  `attention_head_rms_prescale: max_abs` divides each head's output by its largest magnitude first; it lowers the
  perplexity of `T_8(s)` and static `p(s)` at `d_K` 4 by 1.0% and 2.2%.
- Narrower keys remove 18–23 M Q/K parameters and run QK RMSNorm over 4–16 channels; every other initial array is
  the partner's. `z h_t(z)` trains on the Mosaic transformed kernels (B200 or H100), the ladder on the Mosaic
  query-coefficient kernel (`d_K` 4 on B200 or H100, 8 and 16 on B300 or H100), exp on the fused kernel and softmax
  on cuDNN, which zero-pads Q and K to width 64 (both B300 or H100); the exp runs add gradient-event capture to
  their October control (methods and launch:
  [`z h_t(z)` at `d_K` 8 and 16](../../README.md#score-prefactors-swish-and-narrow-keys),
  [shifted exp and softmax](../../README.md#shifted-exp-and-softmax-at-narrow-keys),
  [the ladder and shifted exp at `d_K` 8 and 16](../../README.md#degree-eight-ladder-and-shifted-exp-at-narrow-keys),
  [max-abs runs](../../README.md#max-abs-head-rmsnorm-at-qk-widths-4-and-8)).
- `plot.py` reads the `query_decay_*` roles only, so the heatmap and `summary.csv` above keep to the grid.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/narrow_keys
```

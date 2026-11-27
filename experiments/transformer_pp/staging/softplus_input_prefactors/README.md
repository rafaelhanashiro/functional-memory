# Direct polynomials of z: static, and times z

Opened 2026-11-20. At degree eight, do static coefficients or a factor `z` help the direct (not sum-of-squares)
polynomial of `z = softplus(s)`? Three runs, each its partner's recipe with the listed score functions and
gradient-event capture, from the partner's exact initial parameters. Validation-best perplexity:

| Run | Weight | Start | Validation-best PPL | Partner |
| --- | --- | --- | ---: | --- |
| `r2026eaa99f9a` | static `p(z)` | `T_8(z)`, about `1 + e^s` | 14.351 | static `p(s)`, `r9c69dd7cfc8d`, 14.294 |
| `rd00848e00f1d` | static `z p(z)` | `z T_8(z)` | **14.114** | `r9c69dd7cfc8d` |
| `r7f80334aafbd` | `z p_t(z)` | `z (T_8(z) − 1)`, about `softplus(s) e^s` | 14.159 | query `p_t(z)`, `ra3b54e2489f8`, 14.262 |

- **The factor `z` is what helps.** Static `z p(z)` is 1.65% below static `p(z)` and 1.26% below static `p(s)`;
  `z p_t(z)` is 0.73% below `p_t(z)`. Both beat exp (14.257), by 1.00% and 0.69%.
- **Static `p(z)` alone is worse than static `p(s)`** (+0.40%) and than exp (+0.66%). Removing its constant
  (`rc28ef4dbe26d`, 14.192) recovers 1.11%: the constant starts the kernel near `1 + e^s`, which does not vanish at
  very negative scores.
- **Static coefficients match query ones here**: static `z p(z)` is 0.31% below `z p_t(z)`.
- **Both tie the sum-of-squares form**: `z h_t(z)` (`r841f35a24241`, 14.095) is 0.14% below static `z p(z)` and
  0.45% below `z p_t(z)`; softplus(s) exp(s) (`rbd24b948a240`, 14.132) is 0.13% above static `z p(z)`.
- The factor `z` is the parameter-free score prefactor `f = softplus` on the Mosaic transformed kernels
  (`query_softplus` for `p(z)`, `query_softplus_times_softplus` for `z p(z)`; static coefficients are broadcast
  over queries), on one B200 or H100
  ([methods and launch](../../README.md#score-prefactors-swish-and-narrow-keys)).
- The static constant `c_0` stays learnable from its Taylor start of 1: no scaffold option starts a learnable
  static constant at zero (`rc28ef4dbe26d` removes it), so `p(z)` starts near `1 + e^s`, where the query runs and
  `rc28ef4dbe26d` start near `e^s`.
- Limits: one seed each; the margins among the three `z`-prefactor kernels (0.1–0.5%) are within one seed's noise.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/softplus_input_prefactors
```

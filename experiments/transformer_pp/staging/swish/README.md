# Swish

Opened 2026-11-20. Does swish, `phi(s) = s sigmoid(s)`, beat sigmoid and softplus? Like softplus it grows as `s`
for large scores, but its weights are signed: they dip to −0.2785 at `s = −1.28` before returning to 0.
`r8a9fd53dacfd` (`swish-head-rmsnorm-rope`) is the softplus run's recipe with the swish transform: fixed `[0, 1]`
coefficients, exp's 198,453,586 initial parameters and decay mask, FP32 embedding-gradient sums, and the Mosaic
`query_swish` kernel on one B200 or H100
([methods and launch](../../README.md#score-prefactors-swish-and-narrow-keys)).

| Run | Weight | Validation-best PPL |
| --- | --- | ---: |
| `r8a9fd53dacfd` | swish(s) | 18.393 |
| `r266c5dd1f935` | softplus(s) | 15.744 |
| `r88d8654a8a1d` | sigmoid(s) | 16.135 |
| `r6a76e7fbde45` | exp(s) | 14.257 |

- **Swish is the worst of the three functions**: 29.0% behind exp, 16.8% behind softplus and 14.0% behind sigmoid.
- **It trains unstably.** Its gradient norm reaches `3e5` (exp's largest is 13), and at least one update is clipped
  in 27 of its 28 evaluation windows (exp: 7).
- As a prefactor on a query sum-of-squares polynomial it fails the same way: `swish(s) h_t(s)` 16.494 and
  `swish(s) h_t(z)` 16.525, 16% behind exp
  ([softplus_sos](../sos_score_prefactors/README.md)), where
  softplus, exp and sigmoid prefactors reach 14.07–14.27, beside this topic's softplus(s) exp(s) and
  sigmoid(s) exp(s).
- Limits: one seed.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/swish
```

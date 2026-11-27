# Prefactors of the raw score

Opened 2026-11-25. Does the direct query polynomial need the input `z` for its prefactor to help, or does a prefactor of
the raw score do as well? Two runs, each the control's recipe (`r25fffc16c2f3`) with the prefactor, max-abs head RMSNorm
and gradient-event capture, from its exact initial parameters, on the Mosaic `query_times_<f>` kernels on a B200
([methods and launch](../../README.md#prefactors-of-the-raw-score-on-the-query-polynomial-at-key-width-64)).
Validation-best perplexity, all at update 7,500:

| Run | Weight | Validation-best PPL | vs control `p_t(s)` | vs `z p_t(z)` |
| --- | --- | ---: | ---: | ---: |
| `rdb8deaab4c91` | `exp(s) p_t(s)` | **14.118** | −1.43% | −0.29% |
| `r7aa6c853dafd` | `softplus(s) p_t(s)` | **14.118** | −1.43% | −0.29% |
| `r7f80334aafbd` | `z p_t(z)` | 14.159 | −1.14% | — |

- **The prefactor is what helps, and on the raw score it helps more.** Both prefactors of `s` end 1.43% below the
  control and 0.29% below `z p_t(z)`: the factor on `p_t(s)` beats the factor on `p_t(z)`, and exp and softplus tie
  (14.1180 and 14.1175).
- **Both beat exp** (14.257) by 0.98% and fixed Taylor (14.218) by 0.70%, and every split-score query run, which applies
  `e^s` to negative scores alone (best 14.201): the gain comes from the whole score range.
- **They come within 0.2% of the sum-of-squares core under the same prefactors**: `exp(s) h_t(s)` 14.089 and
  `softplus(s) h_t(s)` 14.097 ([sos_score_prefactors](../sos_score_prefactors/README.md)).
- Limits: one seed each; margins of 0.2–0.3% are within one seed's noise, the 1.4% to the control is not.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/raw_score_prefactors
```

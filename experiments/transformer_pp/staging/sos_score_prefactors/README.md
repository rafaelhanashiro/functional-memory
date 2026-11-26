# Score prefactors on query SoS

Opened 2026-11-20. Is `z = softplus(s)` the best factor in front of `h_t`? Seven runs weight
`W = f(s) h_t(g(s))`, with `f` a parameter-free function of the raw score, `g` the identity or softplus, and a query
SoS core of degree eight with input power zero. Each is `reeefbcc19abf`'s recipe (`h_t(s)`, 14.196) with `f`, `g`
and gradient-event capture, from its exact initial parameters, so `h_t` starts at `T_8`. Validation-best perplexity:

| `f` | `g(s) = s` | `g(s) = z` |
| --- | --- | --- |
| softplus | `r41662adb22fa`, 14.097 | `r841f35a24241`, 14.095 (the control) |
| exp | `r6525664e6696`, 14.089 | `raeaa47865867`, **14.067** |
| sigmoid | `r8f56a5b7e525`, 14.148 | `r5d578a29f080`, 14.271 |
| swish, `s sigmoid(s)` | `re6846615ffe2`, 16.494 | `r50c893fe159c`, 16.525 |

- **No prefactor improves on the control beyond one seed's noise.** `e^s h_t(z)` is 0.20% below it (1.33% below
  exp, `r6a76e7fbde45`, 14.257); `e^s h_t(s)` and `softplus(s) h_t(s)` are level with it (−0.04% and +0.01%).
- **With a growing positive prefactor the polynomial's input stops mattering**: the softplus and exp rows span
  0.21%, where without a prefactor `h_t(s)` leads `h_t(z)` (`r41a4c74c5f58`, 14.339) by 1.00%. The prefactor is
  worth 0.69% on `h_t(s)` and 1.70% on `h_t(z)`.
- **A bounded prefactor is worse**: sigmoid gives 14.148 on the raw score (+0.37% against the control) and 14.271
  on `z` (+1.24%), level with exp.
- **Swish fails**: 16.494 on the raw score and 16.525 on `z`, 17% behind the control and 16% behind exp. Its
  weights are signed and it trains unstably: gradient norms reach `1e5` and `1e4` (the control's largest is 13),
  with an update clipped in 26 and 21 of the 28 evaluation windows (the control: 6). Swish alone
  (`r8a9fd53dacfd`, 18.393 in
  [pointwise_attention](../swish/README.md)) is
  worse still.
- The learned core is worth 0.25–0.27% over the fixed kernels its starts approximate: `softplus(s) h_t(s)` against
  softplus(s) exp(s) (`rbd24b948a240`, 14.132) and `sigmoid(s) h_t(s)` against sigmoid(s) exp(s)
  (`rd7c9f610e9dc`, 14.186) of
  [pointwise_attention](../../results/pretraining/alternative_attention/fixed_kernels/pointwise_attention/README.md).
- The control computes `softplus(s) h_t(softplus(s))` through the input power (degree nine with a structural
  zero); the new runs use the general prefactor on the Mosaic transformed kernels (`query_times_<f>`,
  `query_softplus_times_<f>`, on one B200 or H100). The two routes are equal in exact arithmetic and differ by FP32
  rounding, which the GPU qualification compares.
- Limits: one seed each. The exp prefactor stayed far from FP32's range: its scores reached 11 and 14 at the final
  validation (`e^s h_t(s)` passes the FP32 maximum at `s = 65.7` at `h_t = T_8`), and every run's attention row sums
  stayed below `3e11`, where head RMSNorm's plain arithmetic is exact
  ([methods and launch](../../README.md#score-prefactors-swish-and-narrow-keys)).

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/sos_score_prefactors
```

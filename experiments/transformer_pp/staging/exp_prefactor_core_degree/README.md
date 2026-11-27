# A degree-twelve SoS core under exp of the raw score

Opened 2026-11-25. Does `e^s h_t(z)`, `z = softplus(s)`, the best RoPE polynomial at `d_K` 64, gain from a
degree-twelve SoS core? With the factor `z` in exp's place the two cores tie (`z h_t(z)`: 14.095 at core eight,
`r841f35a24241`, and 14.093 at core twelve, `rf637de9b65b1`). Each run is the degree-twelve `z h_t(z)` recipe with
exp of the raw score in place of `z`, at the recipe's learning rate and at double it, from that run's exact initial
parameters ([methods and launch](../../README.md#a-degree-twelve-sos-core-under-exp-of-the-raw-score-at-key-width-64)).
Home cluster: Kennel; the degree-eight run at double the rate and exp at double are ORCD's.

## Runs

Validation perplexity, best with terminal in parentheses.

| Kernel | Recipe's rate | Double rate |
| --- | ---: | ---: |
| `e^s h_t(z)`, core 8 | 14.067 (14.069) `raeaa47865867` | **14.001** (14.009) `r39d0469cc122` |
| `e^s h_t(z)`, core 12 | 14.086 (14.089) `r99fe75016666` | 14.036 (14.041) `rafbba8d1794f` |
| Core 12 against core 8 | +0.13% | +0.24% |
| exp | 14.257 (14.262) `r6a76e7fbde45` | 14.212 (14.215) `r19dfa87cb7ed` |
| Core 12 against exp | −1.20% | −1.24% |

## Findings

- **The degree-twelve core does not improve on the degree-eight one**: 0.13% behind at the recipe's rate and 0.24%
  at double, inside one seed's noise, as with the factor `z`.
- **Both cores beat exp at either rate**, core twelve by 1.2% and core eight by 1.3–1.5%.
- **Double the rate helps both cores a little**: 0.36% at core twelve and 0.47% at core eight.
- Each degree-twelve run records one gradient spike, at update 6,376.

## Limits

- One seed per run; every difference between the cores is below the 0.6–0.7% seed spread at this width.
- The Mosaic prefactor kernels take degree twelve for this family at `d_K` 64 alone
  (`mosaic.api.PREFACTOR_DEGREE_LIMITS`); each run's acceptance ran their degree-twelve GPU cases on its B200.
- Hardware: both degree-twelve runs on a B200, the run at the recipe's rate across fourteen preemptions.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/exp_prefactor_core_degree
```

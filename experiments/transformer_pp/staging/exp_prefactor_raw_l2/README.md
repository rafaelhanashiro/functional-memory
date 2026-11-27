# Raw-input L2 for e^s h_t(z)

Opened 2026-11-26. Does the raw-input L2 coefficient normalization help `e^s h_t(z)`, `z = softplus(s)`, with the
degree-eight affine tanh SoS core, the best RoPE polynomial at `d_K` 64, at the recipe's learning rate or a larger
one? With the factor `z` it did not at the recipe's rate (`z h_t(z)`: 14.131 with it, `r35805940a380`, against 14.095,
`r841f35a24241`; [coefficient_input_norm](../../results/pretraining/other/coefficient_input_norm/README.md)). Each run
is the plain-input run at its rate with the coefficient generator reading the raw attention input at unit L2 norm
in place of the gain-applied pre-attention RMSNorm activation, from the same initial parameters
([methods and launch](../../README.md#raw-input-l2-for-es-h_tz)). Home cluster: Kennel; the plain-input runs at two
and four times the rate, and exp at those rates, are ORCD's.

## Runs

Validation perplexity, best with terminal in parentheses.

| Rate | Plain input | Raw-input L2 | L2 against plain | exp |
| --- | ---: | ---: | ---: | ---: |
| Recipe's | 14.067 (14.069) `raeaa47865867` | 14.088 (14.091) `rd5daec72b8e7` | +0.15% | 14.257 (14.262) `r6a76e7fbde45` |
| Double | **14.001** (14.009) `r39d0469cc122` | 14.050 (14.056) `r2c608f03d08c` | +0.35% | 14.212 (14.215) `r19dfa87cb7ed` |
| Four times | 14.179 (14.188) `r365f0c30f632` | 14.188 (14.194) `rdd827b2c2b7d` | +0.06% | 14.342 (14.348) `r530a1bce3597` |

## Findings

- **Raw-input L2 does not help `e^s h_t(z)` at any rate**: 0.06–0.35% behind the plain input, inside one seed's noise,
  as with the factor `z` (+0.26%). The best run stays the plain input at double the rate.
- **It changes the rates' order no more than their values**: double is best and four times worst with either
  input.
- Every L2 run is 1.1–1.2% below exp at its rate, as the plain-input runs are (1.1–1.5%).
- Each L2 run records one gradient spike, at update 6,376.

## Limits

- One seed per run; no difference between the inputs exceeds the 0.6–0.7% seed spread at this width.
- The plain-input runs at two and four times the rate trained on ORCD, the L2 runs on a Kennel B200 (the run at
  four times the rate across two attempts).

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/exp_prefactor_raw_l2
```

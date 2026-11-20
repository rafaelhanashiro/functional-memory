# Input dependence of polynomial coefficients

Opened 2026-11-18. Does a trained Transformer++ checkpoint predict better when
its polynomial coefficients vary with the query? Fixing every head to its
calibration-mean polynomial raised held-out perplexity by **1.21–1.37%** across
the three evaluated checkpoints.

| Model | Evaluation run | Parent checkpoint | Native PPL | Calibration-mean PPL | Change |
| --- | --- | --- | ---: | ---: | ---: |
| Query | `r50200ac83292` | `r25fffc16c2f3` | 15.524 | 15.736 | +1.37% |
| SoS | `r850466844aff` | `reeefbcc19abf` | 15.410 | 15.602 | +1.25% |
| SoS + softplus input + prefactor | `r2384d21a3134` | `r841f35a24241` | 15.278 | 15.463 | +1.21% |

All three checkpoints are from update 7,500 of the 15-layer, width-896 models
with 14 heads and a degree-eight polynomial core. The prefactor gives the third
model total degree nine in the softplus score. For each checkpoint we averaged
the effective monomial coefficients over positions in the first 64 validation
sequences, giving one polynomial per layer and head. In the SoS variants, these
are the coefficients after expanding the squares. We substituted the means in
all layers simultaneously while each input continued to supply its own queries,
keys and values.

The native and calibration-mean passes scored the same next 256 validation
sequences, each with 2,048 prediction targets. PPL is the exponential of the
mean next-token NLL on this subset. The intervention measures the contribution
of query-conditioned coefficients to these frozen checkpoints. Models trained
with static coefficients can adapt their other parameters.

The [selection](selection.yaml) pins the evaluation inputs. The full-precision
[summary](summary.csv) also supplies the website table. Regenerate it from the
repository root with:

```sh
python -m experiments.transformer_pp.results.diagnostics.input_dependence.calibration_mean.summarize
```

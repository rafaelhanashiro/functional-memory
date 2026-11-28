# Frozen static coefficients and fixed Taylor

Opened 2026-11-27.

Do the learned static coefficients improve held-out predictions on the backbone that learned them? When both backbones use the same Taylor coefficients, which backbone predicts better?

## Checkpoints and evaluation

The [degree comparison](../../results/pretraining/polynomial/degree_scaling/README.md) supplies the original learned-static checkpoint (`r9c69dd7cfc8d`) and fixed-Taylor checkpoint (`ra7ee8a4b2544`). Both have 15 layers, width 896, degree eight and key width 64. Both are the validation-best checkpoint at update 7,500. The evaluation run is `ra4d05272d53c`. Each condition reads the original validation sequences in order. The first 19 full batches cover 4,864 packed sequences of 2,048 target tokens each, or 9,961,472 tokens.

For the static reset, every stored coefficient in every layer and head is set to one. With the training recipe's factorial scaling, this gives the represented coefficients $c_r=1/r!$ for all nine orders. The native static condition uses the original stored coefficients. All three evaluations keep Q, K, V, per-head score scales, output RMSNorm gains, and all other parameters at the values from their respective checkpoints.

## Findings

<!-- BEGIN GENERATED RESULTS -->

| Frozen checkpoint and coefficients | NLL (nats/token) | PPL |
| --- | ---: | ---: |
| Learned static, trained coefficients | 2.659845 | 14.2941 |
| Learned static, Taylor coefficients | 2.742872 | 15.5315 |
| Fixed Taylor, trained with Taylor coefficients | 2.654530 | 14.2183 |

| Difference | Δ NLL (nats/token) | Paired 95% interval | Δ PPL | Packed rows with higher loss |
| --- | ---: | ---: | ---: | ---: |
| Reset minus trained coefficients on the static backbone | +0.083027 | [+0.082411, +0.083643] | +1.2375 | 4,864/4,864 |
| Static-trained minus Taylor-trained backbone, both using Taylor | +0.088342 | [+0.087613, +0.089072] | +1.3132 | 4,863/4,864 |

<!-- END GENERATED RESULTS -->

Restoring Taylor coefficients raised loss on every packed validation sequence and increased PPL by 1.24 on the static backbone. The trained backbone relied on its learned coefficients. With both backbones evaluated using Taylor, the Taylor-trained backbone leads by 1.31 PPL. With each using its own coefficients, that advantage is 0.08 PPL.

Both native evaluations reproduced their checkpoint records. The absolute differences were 0.00000952 nats/token for static and 0.00000055 for Taylor, within the prespecified tolerance of 0.002 nats/token. The intervals use the mean and standard error of within-sequence NLL differences across the 4,864 packed rows, with a normal 95% interval. Perplexity is the exponential of the mean token loss.

## Limits

The paired intervals describe variation across packed validation sequences. Training variability across seeds remains unknown. The common-Taylor comparison measures how each trained backbone works with this specific kernel. The question of which coefficient changes account for the reset effect remains open.

## Regeneration

From the repository root, run `python -m experiments.transformer_pp.staging.static_taylor_reset.report`. The evaluation's per-sequence measurements live under its run. The selection pins those measurements and the source checkpoints' training results.

# Polynomial attention gradient dynamics

Opened 2026-11-26.

Why do degree-eight attention polynomials move away from their initial coefficients, and how does that movement affect the backbone they train with? These Transformer++ diagnostics use the 15-layer width-896 shape and key width 64.

## Experiments

- **Completed · regular query and static.** Two instrumented runs repeated the original query-conditioned and learned-static recipes from initialization. Both recorded signed coefficient gradients before clipping and the actual AdamW updates at every one of 7,570 steps. For the query model this includes each tanh amplitude and offset, and gradient and update norms for its input-projection vectors. Selected batches also have per-query gradient decompositions. [Query run](../../runs/r1422e8000207/) · [Static run](../../runs/r6f55fe172f41/)
- **Completed · Taylor reset.** We evaluated the original learned-static checkpoint with its own coefficients and again with every coefficient restored to Taylor. A separately trained fixed-Taylor checkpoint provided the common-kernel comparison. [Evaluation](../../runs/ra4d05272d53c/) · [Original static checkpoint](../../runs/r9c69dd7cfc8d/) · [Fixed-Taylor checkpoint](../../runs/ra7ee8a4b2544/)
- **Completed · exp-prefactored query sum of squares.** Diagnostic `r11215998f5f6` repeated the original recipe with a degree-eight sum of squares applied to the softplus score, an exponential prefactor and peak learning rate 0.002. It recorded training gradients and updates of the two square factors’ generator parameters. It finished all 7,570 updates, with best validation PPL 14.0456 at step 7,500 and terminal PPL 14.0469. [Diagnostic on W&B](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/b1ecbc8cc042446fbcb540a4357b38a1) · [Original training run](../../runs/r39d0469cc122/)
- **In progress · freeze or continue learning the static coefficients.** The first full checkpoint retained from the instrumented static run was at update 1,675, when fixed Taylor already led on validation. A replay of the static recipe (`r14145c364f8c`) produced a checkpoint at update 500 with validation NLL 3.902374 and PPL 49.5198. From this shared checkpoint, Kennel job 16393 (`r3c1d740f28d1`) trains the entire model. Job 16397 (`r8111c7eda7d3`) holds the 1,890 polynomial coefficients at their learned values and trains every other parameter. Both resume from the same AdamW state and data position and follow the original 7,570-step learning-rate schedule through update 2,000. We compare validation loss at the fork, after one update and every 100 updates thereafter. [Shared checkpoint on W&B](https://wandb.ai/rafah-massachusetts-institute-of-technology/functional-memory/runs/eb6e2da7cfb5469db8c7a5ebbcec548c)

## Measurements and results

<!-- BEGIN GENERATED RESULTS -->

### Training perplexity

| Completed training run | Run ID | Validation-best PPL (step) | Terminal PPL (step) |
| --- | --- | ---: | ---: |
| Static, original | [r9c69dd7cfc8d](../../runs/r9c69dd7cfc8d/) | 14.2942 (7,500) | 14.2985 (7,570) |
| Static, gradient diagnostic | [r6f55fe172f41](../../runs/r6f55fe172f41/) | 14.3516 (7,500) | 14.3551 (7,570) |
| Query-conditioned, original | [r25fffc16c2f3](../../runs/r25fffc16c2f3/) | 14.3228 (7,500) | 14.3278 (7,570) |
| Query-conditioned, gradient diagnostic | [r1422e8000207](../../runs/r1422e8000207/) | 14.3134 (7,500) | 14.3160 (7,570) |
| Fixed Taylor control | [ra7ee8a4b2544](../../runs/ra7ee8a4b2544/) | 14.2183 (7,500) | 14.2231 (7,570) |
| Exp-prefactored query SoS, original | [r39d0469cc122](../../runs/r39d0469cc122/) | 14.0014 (7,500) | 14.0088 (7,570) |

### How the coefficients moved

For each static checkpoint, $u_r=r!c_r$ initially equals one. The table averages each selected order over all 210 layer/head pairs. For the query diagnostic, each layer, head and order has a mean and standard deviation of $r!c_r(x)$ across the 2,048 positions in the same held-out sequence. The last two columns are the median absolute departure of those means from one and the median of those standard deviations.

| Update | Static $u_4$, original / repeat | Static $u_7$, original / repeat | Query median mean shift | Query median within-sequence SD |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 0.969 / 0.970 | 1.065 / 1.064 | 0.050 | 0.012 |
| 1,500 | 0.879 / 0.879 | 1.228 / 1.228 | 0.092 | 0.048 |
| 7,500 | 0.812 / 0.809 | 1.402 / 1.407 | 0.159 | 0.097 |

### Frozen Taylor reset

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

The query and static repeats reversed their original PPL ordering at the same seed. Across both static runs, the fourth-order coefficient fell and the seventh-order coefficient rose as training progressed. In the query repeat, the typical mean coefficient had moved roughly four times as far from Taylor at update 400 as its standard deviation across queries. The across-query spread grew relative to this shift over the course of training.

Both frozen checkpoints came from update 7,500, their validation best. All three reset conditions used the same 4,864 packed validation sequences of 2,048 targets each, or 9,961,472 tokens. Setting each of the learned static checkpoint’s 1,890 stored coefficients to one restored $c_r=1/r!$. Q, K, V, per-head score scales, output RMSNorm gains and every other parameter kept their checkpoint values. The paired intervals use the standard error of within-sequence NLL differences across the 4,864 rows, with a normal 95% interval. PPL is the exponential of mean token NLL.

The native losses in this replay differed from the original measurements by 0.00000952 nats/token for static and 0.00000055 for Taylor, within the prespecified tolerance of 0.002. Restoring Taylor worsened loss on every packed sequence and cost the static backbone 1.24 PPL. With both backbones using Taylor, the Taylor-trained backbone led by 1.31 PPL. With each checkpoint’s native coefficients, its lead was 0.08 PPL. [Paired evaluation](../../runs/ra4d05272d53c/)

## Open question

Does continuing to update the static coefficients after step 500 help held-out loss on the first update, and does that advantage persist as the backbone adapts? A persistent gap between the two arms measures the effect of allowing the coefficients to keep learning from this particular checkpoint.

The reset establishes how the fitted coefficients support a particular backbone. The training trajectories alone cannot establish which updates caused the final performance difference. Each recipe used one seed. The paired intervals quantify variation across packed validation rows at fixed checkpoints.

## Regeneration

From the repository root, run `python -m experiments.transformer_pp.staging.polynomial_gradient_dynamics.report`. The [selection](selection.yaml) pins completed measurements and their source runs. The tables use the compact metrics, best-validation and completion records. The detailed per-step arrays are retained with the corresponding runs on Kennel.

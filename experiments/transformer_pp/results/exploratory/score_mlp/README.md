# Score MLP

Opened 2026-11-04.

**Question.** Can a learned, input-dependent score function `f(x_t, s)` replace exp in
unnormalized attention with attention-output RMSNorm? Method and launch commands are in
the [study README](../../../README.md#score-mlp).

## Runs

Each copies `query-m4-head-scale` (`r2b22f2ff6461`) and changes only `f`.

| Run | Label | Form | `J` | Status |
| --- | --- | --- | ---: | --- |
| `re65b12334c5e` | `score-mlp-exp-residual-j4` | `exp(s) (1 + sum_j a_j SiLU(w_j s + b_j))` | 4 | queued |
| `r98e7f3e60a59` | `score-mlp-exp-residual-j8` | `exp(s) (1 + sum_j a_j SiLU(w_j s + b_j))` | 8 | queued |

Both run on the fused score-MLP kernel ([qualification and throughput](../../../../../scaffolds/transformer/validation/README.md)).
The pure kernel form `f = sum_j a_j SiLU(w_j s + b_j)` (`J = 4, 8`) is not allocated:
its starting point is undecided.

## Controls

Validation-best perplexity, 1x Chinchilla:

| Run | Label | Best PPL |
| --- | --- | ---: |
| `r2b22f2ff6461` | `query-m4-head-scale` | 14.774 |
| `r816ab0fbe119` | `fixed-m4-head-scale` | 14.722 |
| `r6a76e7fbde45` | `exp-head-scale` | 14.257 |

## Open decision

Promote the score MLP to its own collection or discard it, once the runs finish.

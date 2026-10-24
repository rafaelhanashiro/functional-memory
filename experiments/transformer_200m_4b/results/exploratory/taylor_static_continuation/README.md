# Learning static coefficients after Taylor pretraining

Opened: **2026-10-24**. Home cluster: **Kennel**.

Does enabling learned static coefficients after the backbone has trained with
fixed Taylor improve a matched continuation? This tests whether coefficient
learning benefits an established representation, given that jointly trained
static coefficients slightly trail fixed Taylor in the 4B recipe.

## Runs and method

Both runs start from the validation-best checkpoint at update **7,500** of
fixed-Taylor degree-eight parent `r019ca96d6533`. Source recipe and checkpoint
hashes are frozen in each child's recipe and execution record.

| Role | Run | Coefficients |
| --- | --- | --- |
| Fixed Taylor control | `r28fb40a1d0e9` | Fixed at `1/r!` |
| Learned static | `r0332cff0b0c0` | Separate per layer/head/order, stored as `a_r`, with `c_r = a_r/r!` and `a_r = 1` initially |

Both backbones continue training for **500 additional updates**, ending at global
update **8,000**. Both preserve the source backbone parameters, Adam moments and
clock, training-data cursor, and RNG step. Newly introduced coefficient parameters
start with zero Adam moments and their own clock at zero. The shared trainer clips
the complete gradient once, then applies the appropriate optimizer state to each
parameter group. Coefficients receive no weight decay.

The learning rate is constant **0.0001** for all trainable parameters. Both runs
use the same 256-sequence global batch, data order, and seed; each receives one GPU
with an eight-sequence microbatch and 32 accumulation steps. The additional budget
is **262,144,000 prediction tokens per run**. Validation uses the unchanged held-out
split and all 19 batches every 100 updates. The initial restored model is evaluated
before the first continuation update. Both use unnormalized polynomial attention,
QK RMSNorm, per-head output RMSNorm, and the Mosaic kernel.

The shared acceptance procedure checks initialization, full-recipe training,
checkpoint continuation, data position, RNG, and the selected GPU implementation
before production. Execution commands are in the [study README](../../../README.md).

## Findings and open decision

Both runs completed all 500 updates on one B300 each after passing GPU acceptance.
Their independently evaluated starting checkpoints have exactly matching held-out
PPL **14.2561500256**. The terminal checkpoint is also validation best in both runs.

| Continuation | Initial PPL | Terminal PPL | Validation-best PPL / step |
| --- | ---: | ---: | --- |
| Fixed Taylor | 14.256150 | 14.195665 | 14.195665 / 8,000 |
| Learned static | 14.256150 | 14.194798 | 14.194798 / 8,000 |

Learned static is slightly better at every post-update evaluation, but its terminal
advantage is only **0.000867 PPL (0.0061%)**, or **0.0000611 nats/token**. The two
continuations are effectively tied. Both improve with the additional training;
this improvement alone cannot be attributed to learning coefficients.

All 1,890 stored coefficients moved from their Taylor initialization `a_r = 1`.
Their mean absolute change is **0.008436**, with a maximum change of **0.04032**;
terminal values span approximately **[0.95968, 1.03931]**. Thus the learned branch
performed real coefficient updates, but remained close to Taylor. Every validation
passed the numerical diagnostics, and neither continuation clipped gradients.

This single-seed, 500-update test shows that enabling static coefficients late
preserves Taylor's quality, with negligible observed benefit at learning rate
0.0001. It does not identify the cause of the original from-scratch static deficit
or establish delayed unfreezing as a general fix. Other coefficient learning rates,
longer continuations, and seed variation remain untested here.

**Open decision:** whether to investigate coefficient update scale or a different
unfreezing point next. The topic remains open, and these runs enter no canonical
pretraining comparison. `summary.csv` reports every matched evaluation, terminal
status, and validation-best selection separately.

## Regeneration

Measurements live only in the selected runs. After intentionally incorporating new
measurements, refresh their pins and regenerate the compact comparison from the
worktree root:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/taylor_static_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.results.exploratory.taylor_static_continuation.collect
```

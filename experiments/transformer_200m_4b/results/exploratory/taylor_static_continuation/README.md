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

Both runs are allocated one B300 each in `standard` and are performing GPU
acceptance before production. No continuation measurements are available yet.
Their job IDs and execution settings are in their run records. Compare
held-out NLL/PPL at matched continuation updates, particularly the terminal update
500. Validation-best measurements are reported separately. A run that fails or is
incomplete remains visible in `summary.csv`.

If learned coefficients improve over the continued Taylor control, delayed
coefficient learning becomes a candidate training change. If they fall behind,
coefficient optimization and regularization remain the next questions. This single
seed and short continuation cannot establish that any parameterization is optimal.
The runs enter no canonical pretraining comparison while this topic is open.

## Regeneration

Measurements live only in the selected runs. After intentionally incorporating new
measurements, refresh their pins and regenerate the compact comparison from the
worktree root:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/taylor_static_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.results.exploratory.taylor_static_continuation.collect
```

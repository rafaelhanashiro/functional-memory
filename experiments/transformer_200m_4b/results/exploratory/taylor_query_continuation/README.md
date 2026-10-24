# Query-dependent coefficients after Taylor pretraining

Opened: **2026-10-24**. Home cluster: **Kennel**.

Does enabling query-dependent polynomial coefficients after fixed-Taylor
pretraining improve held-out performance beyond matched additional Taylor
training? This topic tests degree four in the [200M / 4B study](../../../README.md).
The degree-eight static continuation is a separate comparison in
[taylor_static_continuation](../taylor_static_continuation/README.md).

## Runs and method

Both runs start from the validation-best checkpoint at update **7,500** of
fixed-Taylor degree-four parent `r79da929de109` (held-out PPL **14.755660**).
The parent's configuration, checkpoint, and packed-data identity are pinned by
SHA256 in each child's frozen recipe and execution record.

| Role | Run | Coefficients |
| --- | --- | --- |
| Fixed Taylor control | `rf311afb9339e` | Fixed at `1/r!` |
| Query-dependent | `r6186c797b4db` | Per layer/head/order, `c_r(x) = (b_r + a_r tanh(W_r x))/r!` |

Here `x` is the normalized hidden state at the query position. The query branch
uses the same parameterization and initialization as the study's original
degree-four query run: `b=1`, `a=0`, and projection entries sampled with standard
deviation `1/sqrt(896)`. There is no inner tanh bias. Zero amplitude makes the
initial polynomial exactly Taylor despite the random projection. Bias and
amplitude both use factorial scaling; the generator adds **942,900 parameters**.

**All model parameters continue training in both runs.** The fixed control keeps
its polynomial fixed; the query branch additionally trains its coefficient
generator. Both preserve pretrained backbone parameters, Adam moments and clock,
the next training-data position, and the RNG step. New generator parameters have
zero Adam moments and their own clock at zero, with no weight decay. The shared
trainer clips the combined gradient once before applying the two optimizer states.

Both train for **500 additional updates**, from 7,500 to **8,000**, with constant
learning rate **0.0001**, the same seed and data order, and a 256-sequence global
batch. Each uses one GPU, microbatch eight, and 32 accumulation steps. This adds
**262,144,000 prediction tokens per run** relative to the source checkpoint.
Data starts at sequence 1,920,000 without restarting the stream. The first 70
updates overlap the original parent's final 70 updates after this checkpoint;
the remaining 430 extend beyond its completed training budget.

Both use unnormalized polynomial attention, QK RMSNorm, per-head output RMSNorm,
and Mosaic. Evaluation uses all 19 unchanged held-out batches at the restored
baseline and every 100 updates; checkpoints are saved every 100 updates. Terminal
and validation-best metrics are reported separately. The shared acceptance checks
restoration, query gradients, the GPU implementation, a finite pilot, and exact
checkpoint continuation before production. Launch commands are in the study README.

## Status and open decision

The pair is prepared for submission. `selection.yaml` explicitly selects only
these two degree-four continuations. No findings are available yet, and these runs
enter no canonical pretraining comparison.

**Open decision:** whether query-dependent coefficients improve the pretrained
model beyond the fixed-Taylor continuation, and whether any observed difference
warrants longer training or seed replication. A before/after improvement alone
cannot separate coefficient learning from the benefit of additional training.

Measurements live only in each selected run. Refresh selection pins after
intentionally incorporating new measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/taylor_query_continuation
```

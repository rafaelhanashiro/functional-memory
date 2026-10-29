# Query-dependent coefficients after Taylor pretraining

Opened: **2026-10-24**. Home cluster: **Kennel**.

Does enabling query-dependent polynomial coefficients after fixed-Taylor
pretraining improve held-out performance beyond matched additional Taylor
training? This topic tests degree four in the [Transformer++ study](../../../README.md).
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

## Findings and open decision

Both jobs passed acceptance on one H100 each on Kennel's `mastiff`, in the
`standard` partition: query job **224947** and fixed-Taylor job **224948**.
Both completed all **500 updates**, ending at global step **8,000**. Their
independently evaluated starting PPLs are exactly equal at
**14.7556604308**. `selection.yaml` explicitly selects only these two degree-four
continuations. These runs enter no canonical pretraining comparison.

| Continuation | Initial PPL | Terminal PPL | Validation-best PPL / step |
| --- | ---: | ---: | --- |
| Fixed Taylor | 14.755660 | 14.688601 | 14.688601 / 8,000 |
| Query-dependent | 14.755660 | 14.685864 | 14.685864 / 8,000 |

The query branch is slightly better at every post-update evaluation, but its
terminal advantage is only **0.002737 PPL (0.0186%)**, or **0.0001863 nats/token**.
Both improve with additional training; the observed difference between branches
is small. Every recorded evaluation has valid numerical diagnostics, and neither
run clipped gradients. Maximum gradient norms were **0.3060** for fixed Taylor
and **0.3055** for query-dependent coefficients.

All 1,050 query biases and all 1,050 modulation amplitudes moved from initialization.
Terminal mean absolute bias displacement is **0.010606**; mean absolute amplitude
is **0.007447**, with maximum absolute amplitude **0.043300**. All 1,050
projection-vector norms changed too (mean absolute change **0.001491**). The
generator therefore learned nonzero input dependence, while remaining close to
Taylor. `summary.csv` includes these diagnostics at each evaluation.

The fixed and query jobs took **1 hour 6 minutes 23 seconds** and **1 hour
9 minutes 53 seconds**, respectively, including qualification. The trainer-reported
runtimes, including evaluation and checkpointing, were **43 minutes 57 seconds**
and **46 minutes 5 seconds**. Each run retains its validation-best checkpoint at
step 8,000; terminal measurements are preserved independently.

This single-seed, 500-update continuation preserves Taylor's quality and yields
only a small observed benefit at learning rate 0.0001. It does not establish the
best achievable query-conditioned fit or isolate why the original jointly
pretrained query model trailed Taylor. Longer continuations, other learning rates,
and seed replication remain untested in this comparison.

**Open decision:** whether the small query-dependent advantage warrants longer
training or seed replication. A before/after improvement alone cannot separate
coefficient learning from the benefit of additional training. The topic remains
open despite both runs being complete.

Measurements live only in each selected run. `summary.csv` derives every selected
evaluation and distinguishes terminal from validation-best measurements. Refresh
selection pins after intentionally incorporating new measurements and regenerate:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/taylor_query_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.results.exploratory.taylor_query_continuation.collect
```

# Per-token query scale after Taylor pretraining

Opened: **2026-10-28**. Home cluster: **Kennel**.

Does giving each query its own bounded score scale, read from the attention input,
improve held-out perplexity beyond matched additional fixed-Taylor training? QK RMSNorm
with a per-head scale `exp(gamma_h)` removes the query-norm freedom that plain scores
have. The gate multiplies Q by `exp(gamma_h + a_h cos(w_h, x_t))` instead: one bounded
number per token and head, where `x_t` is the activation after the block's pre-attention
RMSNorm. For the degree-four Taylor kernel this is exactly a per-query temperature. The
fitting-batch analyses in [coefficient_oracle](../coefficient_oracle/README.md) motivated
the question but say nothing about held-out text. This topic tests degree four in the
[Transformer++ study](../../../README.md) and enters no canonical comparison.

## Runs and method

Both runs start from the validation-best checkpoint at update **7,500** of the canonical
fixed-Taylor degree-four parent `r816ab0fbe119` (held-out PPL **14.7218**; QK RMSNorm
with per-head scaling, head RMSNorm output scaling, Mosaic). The parent's configuration,
checkpoint, and packed-data identity are pinned by SHA256 in each child's frozen recipe
and execution record.

| Role | Run | Change |
| --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | None |
| Per-token query scale + direction decay | `re563830cf150` | `qk_norm_token_scale: cosine`; decay only the gate direction `w_h` at 0.1 |

The gate adds, per layer and head, a direction `w_h` (896 entries drawn `N(0, 1/896)`;
only its direction matters) and an amplitude `a_h` starting at zero: **188,370
parameters**. Zero amplitude reproduces the parent's function. Both vectors are
normalized with `eps = 1e-6`, so `|delta| <= |a_h|` without saturation.

All backbone parameters continue training in each run, preserving pretrained Adam
moments and clock, the next training-data position, and the RNG step. The gate's
parameters have zero Adam moments, their own clock, and a constant learning rate of
**0.001**; the backbone keeps **0.0001**. Weight decay **0.1** applies to the gate's
`qk_norm_token_scale_kernel` (`w_h`), enabled by
`model.qk_norm_token_scale_kernel_weight_decay: true`; its amplitude `a_h` remains
exempt. Both runs train **2,000** additional updates, from
7,500 to **9,500**, with the same seed and data order, a 256-sequence global batch, one
GPU, microbatch eight, and 32 accumulation steps: **1,048,576,000 prediction tokens per
run** beyond the source checkpoint. Data continues at sequence 1,920,000 without
restarting the stream; the first 70 updates overlap the parent's final 70. Evaluation
uses all 19 held-out batches at the restored baseline and every 100 updates; checkpoints
are saved every 100 updates. Terminal and validation-best metrics are reported
separately.

The shared acceptance procedure checks restoration, the GPU implementation, a finite
pilot, and exact checkpoint continuation before production. The gate's closed start,
its gradients, and its equivalence to a shifted head scale when its cosine is constant
are unit-tested in `scaffolds/transformer/tests/test_token_query_scale.py` and
`test_token_scale_continuation.py`. Launch commands are in the
[study README](../../../README.md#cpt-with-projection-weight-decay). The gate's frozen
configuration is in [its run directory](../../../runs/re563830cf150/config.yaml).

## Findings and open decision

Both runs completed all **2,000** updates on one H100 each: the control on Kennel
`mastiff` (PCIe) and the gate on `dogo` (SXM), using the `standard` partition. The gate
passed acceptance and completed without numerical failures.

The restored baselines differ by **5.1e-6 nats** on the 19 held-out batches (NLL 2.689318
against 2.689324). With `a_h = 0` the gate is mathematically the parent's function;
the implementations have small floating-point differences.

| Continuation | Initial PPL | Terminal PPL | Validation-best PPL / step |
| --- | ---: | ---: | --- |
| Fixed Taylor | 14.7216 | 14.5064 | 14.5064 / 9,500 |
| Per-token query scale + direction decay | 14.7217 | 14.5056 | 14.5056 / 9,500 |

The gate ends **0.000821 PPL (0.0057%)**, or **0.0000566 nats/token**, below fixed Taylor.
The NLL difference is +0.0000027 at continuation update 500, −0.0000164 at 1,000,
−0.0000271 at 1,500, and −0.0000566 at 2,000. This is a very small difference in a
single-seed comparison on different H100 variants and executed sources; it does not
establish a meaningful held-out gain.

The gate does open. At update 9,500 its amplitudes have mean `|a_h|` **0.2094** and
maximum **0.9802**; **140 of 210 heads** exceed 0.1. On the diagnostic sequence the
per-token log-scale standard deviation, averaged across layers and heads, is **0.0181**,
and the largest absolute log-scale modulation is **0.4157**, bounding the query-scale
multiplier between **1/1.52** and **1.52**. These are diagnostics of one
sequence, not aggregate statistics over all held-out text.

For this parent, seed, and budget, the gate with direction decay does not establish a
meaningful gain over continuing fixed Taylor. This does not determine whether
query-dependent temperature would help under a different training protocol.

**Open decision:** retain or discard the topic. These results alone do not motivate a
full-pretraining run.

Measurements live only in each selected run; `summary.csv` derives every selected
evaluation with the gate's amplitude and realized-scale statistics, distinguishing
terminal from validation-best. Regenerate after intentionally incorporating new
measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/exploratory/token_query_scale_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.token_query_scale_continuation.collect
```

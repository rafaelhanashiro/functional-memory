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
[200M / 4B study](../../../README.md) and enters no canonical comparison.

## Runs and method

Both runs start from the validation-best checkpoint at update **7,500** of the canonical
fixed-Taylor degree-four parent `r816ab0fbe119` (held-out PPL **14.7242**; QK RMSNorm
with per-head scaling, head RMSNorm output scaling, Mosaic). The parent's configuration,
checkpoint, and packed-data identity are pinned by SHA256 in each child's frozen recipe
and execution record.

| Role | Run | Change |
| --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | None |
| Per-token query scale | `ra988273933f7` | `qk_norm_token_scale: cosine` |

The gate adds, per layer and head, a direction `w_h` (896 entries drawn `N(0, 1/896)`;
only its direction matters) and an amplitude `a_h` starting at zero: **188,370
parameters**. Zero amplitude reproduces the parent exactly. Both vectors are normalized
with `eps = 1e-6`, so `|delta| <= |a_h|` without saturation.

All backbone parameters continue training in both runs, preserving pretrained Adam
moments and clock, the next training-data position, and the RNG step. The gate's
parameters have zero Adam moments, their own clock, no weight decay, and a constant
learning rate of **0.001**; the backbone keeps **0.0001**. The separate rate lets `a_h`
reach order one within the budget; the query-generator continuation's amplitudes reached
at most 0.043 in 500 updates at 0.0001. Both train **2,000** additional updates, from
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
[study README](../../../README.md#matched-taylor-continuations).

## Findings and open decision

Both runs are allocated and not yet complete.

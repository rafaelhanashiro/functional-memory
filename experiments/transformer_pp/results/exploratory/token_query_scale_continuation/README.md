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

All three runs start from the validation-best checkpoint at update **7,500** of the canonical
fixed-Taylor degree-four parent `r816ab0fbe119` (held-out PPL **14.7218**; QK RMSNorm
with per-head scaling, head RMSNorm output scaling, Mosaic). The parent's configuration,
checkpoint, and packed-data identity are pinned by SHA256 in each child's frozen recipe
and execution record.

| Role | Run | Change |
| --- | --- | --- |
| Fixed Taylor control | `r82203a85ef8e` | None |
| Per-token query scale | `ra988273933f7` | `qk_norm_token_scale: cosine` |
| Per-token query scale + direction decay | `re563830cf150` | Same gate; decay only `w_h` at 0.1 |

The gate adds, per layer and head, a direction `w_h` (896 entries drawn `N(0, 1/896)`;
only its direction matters) and an amplitude `a_h` starting at zero: **188,370
parameters**. Zero amplitude reproduces the parent's function. Both vectors are
normalized with `eps = 1e-6`, so `|delta| <= |a_h|` without saturation.

All backbone parameters continue training in each run, preserving pretrained Adam
moments and clock, the next training-data position, and the RNG step. The gate's
parameters have zero Adam moments, their own clock, no weight decay in the original gate
run, and a constant learning rate of **0.001**; the backbone keeps **0.0001**. The separate rate lets `a_h`
reach order one within the budget; the query-generator continuation's amplitudes reached
at most 0.043 in 500 updates at 0.0001. All three train **2,000** additional updates, from
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

The direction-decay branch **`re563830cf150`** adds weight decay **0.1** only to
`qk_norm_token_scale_kernel` (`w_h`), with
`model.qk_norm_token_scale_kernel_weight_decay: true`. Its amplitude `a_h` remains
exempt. It uses the same Taylor checkpoint at update 7,500, 2,000 further updates,
data order, seed, RoPE, and closed gate initialization as `ra988273933f7`. Both new
gate parameters use constant LR 0.001; the backbone uses 0.0001 and retains its Adam
history. The frozen configuration is in [its run directory](../../../runs/re563830cf150/config.yaml).
It completed on one H100 SXM on Kennel; commands are in the
[study README](../../../README.md#cpt-with-projection-weight-decay).

The original control and no-decay gate runs completed all **2,000** updates on one H100
each (Kennel `mastiff`, `standard` partition): control job **228301**, gate job **228305**.
Trainer runtimes were **2 h 31 min** (control) and **2 h 12 min** (gate).

The restored baselines differ by **6e-6 nats** on the 19 held-out batches (NLL 2.689318
against 2.689324): with `a_h = 0` the gate is the parent's function, and the remaining
difference is the differently compiled Mosaic program (bitwise identical on the CPU
reference backend; 1.1e-4 nats apart on the single acceptance batch).

| Continuation | Initial PPL | Terminal PPL | Validation-best PPL / step |
| --- | ---: | ---: | --- |
| Fixed Taylor | 14.7216 | 14.5064 | 14.5064 / 9,500 |
| Per-token query scale | 14.7217 | 14.5057 | 14.5057 / 9,500 |
| Per-token query scale + direction decay | 14.7217 | 14.5056 | 14.5056 / 9,500 |

The original no-decay gate ends **0.0007 PPL (0.0048%)**, or **0.0000477 nats/token**,
below the control.
The two trajectories agree to within 1.4e-5 nats through update 1,000, after which the
gate's small advantage grows monotonically but never exceeds 5e-5 nats. This is a tie
at the resolution of the comparison; the earlier query-generator continuation's
advantage over its control was of the same order (0.0027 PPL in 500 updates).

With direction decay, terminal and validation-best PPL are both **14.505602** at update
9,500: **0.000130 PPL** (**0.00000894 nats/token**) below the no-decay gate, and
**0.000821 PPL** below fixed Taylor. The two gate runs differ by at most **0.000343 PPL**
over the recorded evaluations, and their difference changes sign during training.
This remains a tie; a single seed on different H100 variants and executed sources does
not establish a weight-decay benefit. The initial PPLs agree within 0.00000004, consistent
with unchanged initialization and a closed gate in both runs.

The original no-decay gate did open. At update 9,500 its amplitudes have mean `|a_h|`
**0.21** and maximum **0.98**; 137 of 210 heads exceed 0.1 and 54 exceed 0.3, with per-layer means
near zero except layer 13 (−0.22). The realized modulation stays small: on the
diagnostic sequence the per-token log scale `delta_th` has a standard deviation of
0.004–0.042 per layer (largest in layers 11–14), so typical queries change their scale by
1–4%, and the extreme token reaches `exp(0.43) = 1.5x`. The model therefore uses the
freedom only mildly, and that use does not move held-out loss.

For this parent, seed, and budget, neither gate variant establishes a meaningful gain
over continuing fixed Taylor. This does not determine whether query-dependent
temperature would help under a different training protocol.

**Open decision:** retain or discard the topic. Direction decay has not changed the
conclusion of this comparison; these results alone do not motivate a full-pretraining run.

Measurements live only in each selected run; `summary.csv` derives every selected
evaluation with the gate's amplitude and realized-scale statistics, distinguishing
terminal from validation-best. Regenerate after intentionally incorporating new
measurements:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_pp/results/exploratory/token_query_scale_continuation
scaffolds/transformer/.venv/bin/python -m experiments.transformer_pp.results.exploratory.token_query_scale_continuation.collect
```

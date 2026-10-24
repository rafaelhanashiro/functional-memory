# Coefficient fitting with frozen 4B backbones

Opened: **2026-10-23**. Home cluster: **Kennel**.

How much of the ability to fit one sequence can the trained query-only coefficient
generator realize? These exploratory fits use the same first packed sequence and
keep each source checkpoint's backbone frozen. They live in the
[200M / 4B study](../../../README.md) and enter no canonical comparison.

| Run | Trained parameters | Source checkpoint | Status |
| --- | --- | --- | --- |
| `r3c0f21ab7dce` | Free coefficients per layer, head, query, and order | Static `r9b8acfcbeb89`, step 7,500 | Complete |
| `rd7bf9bc37cb6` | Original tanh query coefficient generator | Query-only `r98d5895c02d0`, step 7,500 | Prepared |

The source backbones differ. These are fitting-capacity diagnostics, not a
matched-backbone comparison of coefficient parameterizations. All reported
perplexities measure the fitting sequence, not generalization.

## Free coefficients from the static checkpoint

The source is degree-eight learned-static run `r9b8acfcbeb89` from the 200M / 4B
setup, using its retained validation-best checkpoint at step **7,500**. The
backbone has 15 layers, width 896, 14 heads, QK RMSNorm, unnormalized polynomial
attention, and per-head RMSNorm on its output. Every pretrained parameter is
frozen. Free effective monomial coefficients have axes
`[sequence, layer, query, head, order]` and shape `[1, 15, 2048, 14, 9]`:
**3,870,720 trainable scalars**, initialized by broadcasting the checkpoint's
learned static coefficients across query positions. Each query uses the same
polynomial across its causal keys.

The input is the first packed evaluation sequence, with 2,048 next-token prediction
positions, also used in the earlier single-sequence oracle. It becomes training
data for this diagnostic; all NLL and perplexity are fitting metrics, with no
generalization claim. The frozen recipe and run record pin the parent recipe,
checkpoint, and dataset checksums. The parent checkpoint is read in place.

The existing coefficient-only runner uses Adam in factorial-scaled coordinates,
LR 0.01, betas 0.9/0.95, epsilon 1e-8, no decay, and gradient norm clipped to 1.
After at least 2,000 updates, improvement below 0.001 nats over 200 updates reduces
the LR tenfold. Fitting stops after 500 confirmation updates if the loss remains
on a plateau; the 10,000-update safety limit pauses an unconverged run. Only the
best fitting coefficients are retained after completion. GPU acceptance must
verify coefficient replay, finite improving updates, and exact checkpoint resume
before fitting starts.

### Findings

The run completed 2,500 updates on one B300 (Slurm job `224586`), with total
allocation time 2 minutes 25 seconds including GPU qualification. The learning
rate decreased at update 2,000 and 500 confirmation updates established a plateau.
All pretrained parameters retained their original checksum.

| Measurement | Update | Fitting NLL | Fitting PPL |
| --- | ---: | ---: | ---: |
| Initial | 0 | 2.591015100 | 13.34330953 |
| Best retained | 1,710 | 0.004305246 | 1.00431453 |
| Terminal | 2,500 | 0.004370327 | 1.00437989 |

Free query coefficients nearly memorize this sequence from the learned-static
checkpoint. The open decision is whether this capacity motivates a more expressive
shared coefficient generator for the 4B setup. This gain does not establish that a shared causal
coefficient generator can realize it, or explain the small held-out gap between
learned-static and fixed Taylor pretraining.

GPU qualification reproduced the original standalone forward exactly when the
captured coefficients were replayed. The separately compiled fitting forward
starts 0.00089169 nats/token lower because of BF16 compilation rounding, within
the recorded 0.002 tolerance. Both sides of the fitting improvement above use the
same compiled training forward; this offset is not counted as improvement.

## Original query coefficient generator

Run `rd7bf9bc37cb6` restores the trained degree-eight query-only checkpoint without
resetting its generator. Only `query_polynomial_bias`, `query_polynomial_amplitude`,
and `query_polynomial_kernel` are optimized: **1,697,220 parameters**. Q/K/V,
normalization gains, output projections, embeddings, MLPs, and the output head
remain fixed. Every forward recomputes intermediate activations normally; no
activation or coefficient table is substituted for the shared generator.

The model retains `c_r(x) = [b_r + a_r tanh(W_r x)] / r!`. Adam updates its native
stored parameters, with LR 0.001, betas 0.9/0.95, epsilon 1e-8, clipping at 1,
and no weight decay. The factorial is already applied by the model and is not
applied again to gradients or updates. The minimum 2,000 updates, 200-update
plateau window, 0.001-nat tolerance, tenfold LR reduction, and 500-update
confirmation match the free-coefficient diagnostic; a 10,000-update safety limit
pauses an unconverged run. The same 2,048 prediction targets are pinned by their
sequence checksum. GPU qualification checks original-forward replay, gradients
for all three generator parameter groups, a loss-reducing update, exact resume,
and an unchanged frozen backbone.

Findings are pending. If this generator nearly memorizes the sequence, it has
sufficient capacity for this diagnostic. A higher plateau would implicate the
generator or its optimization; it would not establish a pure representation
limit. A free-coefficient fit from the same query-only checkpoint is a possible
follow-up, not part of this execution.

## Open decision and regeneration

Decide whether these fitting results motivate changes to the shared coefficient
generator or its training. [selection.yaml](selection.yaml) names the exact runs
and their source parents. Measurements and retained fitted parameters live only
under each run. Launch commands are in the [study README](../../../README.md).

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_200m_4b/results/exploratory/coefficient_oracle
```

# Coefficient fitting with frozen 4B backbones

Opened: **2026-10-23**. Home cluster: **Kennel**.

How much can polynomial coefficients fit one sequence or a full batch, and how
does sharing coefficients across sequences constrain that capacity? These exploratory
fits start at the first packed evaluation sequence and keep each source checkpoint's
backbone frozen. They live in the
[200M / 4B study](../../../README.md) and enter no canonical comparison.

| Run | Trained parameters | Source checkpoint | Status |
| --- | --- | --- | --- |
| `r3c0f21ab7dce` | Free coefficients per layer, head, query, and order | Static `r9b8acfcbeb89`, step 7,500 | Complete |
| `rd7bf9bc37cb6` | Original tanh query coefficient generator | Query-only `r98d5895c02d0`, step 7,500 | Complete |
| `r1f2f749ff099` | Original degree-four tanh query coefficient generator | Query-only `r49b7943aee1b`, step 7,500 | Complete |
| `r0963b3e6610b` | One coefficient table shared across 256 sequences | Query-only `r98d5895c02d0`, step 7,500 | Running |
| `r22854de31b96` | Static coefficients on one sequence | Query-only `r98d5895c02d0`, step 7,500 | Complete |
| `r5b28697413fa` | Static coefficients on 256 sequences | Query-only `r98d5895c02d0`, step 7,500 | Complete |
| `r7505c6d49711` | Independent coefficients per sequence, layer, query, head, and order on 256 sequences | Query-only `r98d5895c02d0`, step 7,500 | Complete |

The source backbones differ. These are fitting-capacity diagnostics, not a
matched-backbone comparison of coefficient parameterizations. All reported
perplexities measure the fitting inputs, not held-out generalization.

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

### Findings

Slurm job `224691` completed on one B300 in the `debug` partition in **3 minutes
27 seconds**, including GPU qualification. Fitting stopped at update **2,684**,
500 updates after the plateau controller reduced the learning rate at **2,184**.
The retained best generator is from update **2,656**. All backbone parameters
retained their original checksum; all three generator parameter groups had
finite, nonzero gradients in qualification.

| Measurement | Update | Fitting NLL | Fitting PPL |
| --- | ---: | ---: | ---: |
| Initial | 0 | 2.603273392 | 13.50788234 |
| Best retained | 2,656 | 0.005425740 | 1.00544049 |
| Terminal | 2,684 | 0.005504376 | 1.00551955 |

**The original trained query generator can nearly memorize this sequence with
its backbone frozen.** Its parametrization therefore has sufficient capacity
for this diagnostic. The result does not establish an advantage on unseen
sequences or explain why joint pretraining ties fixed Taylor. The earlier free
coefficients also nearly memorized the sequence, but from a different backbone;
their small difference in fitting PPL is not a controlled comparison.

The original and restored standalone forward logits matched exactly. The
separately compiled training forward differs from the capture forward by
0.00172043 nats, within the recorded BF16 tolerance of 0.002. Both the initial
and fitted losses in the table use the same training graph. Checkpoint roundtrip
and the next resumed update were exact.

## Degree-four query coefficient generator

Opened: **2026-10-24**. Run `r1f2f749ff099` repeats the single-sequence
query-generator fit at degree four, starting from the trained query-only
checkpoint `r49b7943aee1b` at step **7,500**. The question is whether this
degree-four model can memorize the same sequence that the degree-eight generator
fit above memorized. The pretrained backbones differ, so this is a comparison of
achieved fitting capacity, not an isolated degree ablation on one backbone.

Only the original query bias, amplitude, and projection parameters are trained:
**942,900 parameters**, with `c_r(x) = [b_r + a_r tanh(W_r x)] / r!`, for orders
zero through four. The trained generator is restored without resetting it.
Q/K/V projections, normalization gains, embeddings, MLPs, and the output head
remain frozen; activations are recomputed normally on every forward. This is
coefficient-only fitting, unlike the all-parameter Taylor continuation.

The first packed evaluation sequence is reused as fitting data: **2,048 prediction
targets**, with the same token checksum as `rd7bf9bc37cb6`. Adam uses LR **0.001**,
betas **0.9/0.95**, epsilon **1e-8**, clipping at **1**, and no weight decay.
The minimum **2,000** updates, **200**-update plateau window, **0.001**-nat
improvement threshold, tenfold LR reduction, and **500** confirmation updates
match the degree-eight fit. An unconverged run pauses at the **10,000**-update
safety limit. The retained checkpoint minimizes fitting NLL, with terminal
metrics reported separately.

### Findings

Slurm job **224960** completed on one B300 on `akita` in `standard` in **2 minutes
29 seconds**, including GPU qualification. The learning rate decreased at update
**2,194**, and fitting stopped at **2,694** after 500 confirmation updates on the
plateau. The retained best generator is from update **2,685**. All backbone
parameters retained their original checksum. All 2,695 loss and gradient-norm
measurements were finite; 201 measured gradient norms exceeded the clipping
threshold of 1.

| Measurement | Update | Fitting NLL | Fitting PPL |
| --- | ---: | ---: | ---: |
| Initial | 0 | 2.616209507 | 13.68375697 |
| Best retained | 2,685 | 0.006344396 | 1.00636456 |
| Terminal | 2,694 | 0.006474757 | 1.00649576 |

**The degree-four query generator can nearly memorize this sequence with its
backbone frozen.** Degree four therefore has sufficient capacity for this
single-sequence diagnostic. The degree-eight generator reached best PPL 1.00544
on the same sequence; their small difference is not a controlled degree ablation
because their pretrained backbones differ. This result does not establish
held-out generalization or explain the small gains in joint pretraining or
Taylor continuation.

GPU qualification passed source replay, finite nonzero gradients for all three
generator groups, an improving update, exact checkpoint resume, and unchanged
backbone parameters. Original and replayed standalone logits matched exactly.
The separately compiled training forward differed by 0.00069380 nats/token,
within the recorded BF16 tolerance of 0.002. Initial and fitted metrics above
use the same training graph, so this offset is not counted as fitting improvement.

### Departure from Taylor and input dependence at degree four

The retained generator at update 2,685 was inspected on its fitting sequence in
Slurm job `224965` (one B300, 58 seconds). Measurements and checkpoint identities
are in the run's [generator_dependence.json](../../../runs/r1f2f749ff099/generator_dependence.json),
pinned in this collection's selection. No parameters were updated.

Effective coefficients were captured at all 15 layers, 14 heads, and 2,048 query
positions. In coordinates `r! * c_r`, Taylor has value one at every order.

| Measurement | Pretrained query generator | Fitted generator |
| --- | ---: | ---: |
| RMS coefficient deviation from Taylor | 0.42045 | 0.72748 |
| RMS coefficient variation across query positions | 0.09728 | 0.46609 |
| RMS input-dependent amplitude | 0.22500 | 0.60315 |
| Median coefficient-direction cosine with Taylor | 0.96553 | 0.81727 |

Query variation is measured about each layer/head/order's sequence mean.
Coefficient distances depend on the chosen coordinates; positive common scales
are largely removed by head RMSNorm. The direction statistic also changes,
showing that the departure includes polynomial shape changes.

The following ablations substitute coefficients in every layer and recompute all
activations with the same frozen backbone. Each uses the same compiled forward;
replaying the captured fitted coefficients exactly matches the native fitted
forward's NLL. The difference from the saved fitting-graph NLL is 0.00000408 nats.

| Coefficients used | Sequence fitting NLL | Sequence fitting PPL |
| --- | ---: | ---: |
| Fitted query-dependent coefficients | 0.006348467 | 1.00637 |
| Fitted coefficients averaged over queries, per layer/head/order | 3.184901237 | 24.16490 |
| Fixed Taylor coefficients `1/r!` | 4.548325062 | 94.47404 |

The fitted solution makes substantial use of query variation and departs
functionally from fixed Taylor. These interventions do not retrain the static
coefficients or backbone, and the Taylor intervention does not use a separately
trained fixed-Taylor backbone. They establish dependence in this fitted solution,
not a lower bound for optimized static models or generalization to unseen inputs.

The inspection implementation is `inspect_generator.py`. To reproduce the raw
measurement on one GPU, from a checkout with the retained parent and fitted
checkpoints available, run:

```bash
PYTHONPATH=. python experiments/transformer_200m_4b/results/exploratory/coefficient_oracle/inspect_generator.py --run experiments/transformer_200m_4b/runs/r1f2f749ff099
```

Use the cluster-selected locked environment. The command refuses to overwrite an
existing measurement; a reproduction checkout should omit that output first.

## Coefficients shared across a full batch

Run `r0963b3e6610b` restores the original query-only checkpoint `r98d5895c02d0`
at step 7,500. It fits a single effective coefficient table with axes
`[layer, query_position, head, order]`, shape `[15, 2048, 14, 9]`, and
**3,870,720 trainable scalars**. This same table is broadcast across every sequence
in the batch, and each query polynomial is shared across its causal keys. It is
position dependent but does not depend on input content. All pretrained parameters
are frozen, and the original query generator is bypassed by the table.

The fitting input is the first **256 packed evaluation sequences**, with **524,288
prediction targets**, including the first sequence used above. Initial coefficients
are the arithmetic mean of the parent model's effective coefficients over these
256 sequences at each layer, query position, head, and order. That projection
changes the model's function, so original-parent NLL, projected-initial NLL, and
fitting-graph initial NLL are recorded separately. Fitting gains compare the initial
and fitted states using the same training graph.

Adam operates in factorial-scaled coordinates with LR 0.01, betas 0.9/0.95,
epsilon 1e-8, clipping at 1, and no decay. Microbatches of eight accumulate the
mean gradient over the entire batch before **one** clipping and Adam update;
there are no separate per-sequence parameters or optimizer states. Every forward
recomputes the backbone normally. The minimum 2,000 updates, 200-update plateau
window, 0.001-nat tolerance, tenfold LR reduction, 500 confirmation updates, and
10,000-update safety limit match the other free-coefficient fit.

Slurm job `224910` is running on one H100 on `dogo` in the `debug` partition.
GPU qualification passed original coefficient replay, shared-table repeatability,
an improving full-batch update, unchanged pretrained parameters, and exact checkpoint
resume. Fitting is underway; no terminal result is available yet.

This tests whether one coefficient table can reduce loss across many fitting
sequences simultaneously. It places a stronger sharing constraint than independent
per-sequence coefficients, but the batch is training data for this diagnostic;
transfer to unseen sequences is not measured.

## Static coefficients on one sequence and a full batch

Two further fits remove both query-position and input-content dependence. Each
trains one effective monomial coefficient tensor `[layer, head, order]`, shape
`[15, 14, 9]`: **1,890 trainable scalars**. Every query position and every sequence
uses the same polynomial within a layer and head. This is the static coefficient
parametrization, evaluated as a fitting diagnostic with the backbone frozen.

Both start from query-only parent `r98d5895c02d0`, checkpoint 7,500. Run
`r22854de31b96` fits the same first packed sequence; `r5b28697413fa` fits the
same first 256 packed sequences as the position-dependent batch diagnostic. For each fit, initialization is the mean
of that parent's effective coefficients across all sequences and query positions
in its fitting set. This projection changes the original forward, so original,
projected-initial, and fitting-graph initial losses are recorded separately.
All original model parameters remain frozen; the original generator is bypassed
by the static tensor, and activations are recomputed normally.

Both use Adam in factorial coordinates with LR 0.01, betas 0.9/0.95, epsilon
1e-8, clipping at 1, and no decay. Gradients average over all fitting tokens before
one clipping and optimizer update. The batch fit accumulates microbatches of eight;
the sequence fit uses one. The minimum 2,000 updates, 200-update plateau window,
0.001-nat tolerance, tenfold LR reduction, and 500 confirmation updates match the
position-dependent fit. The 10,000-update safety limit pauses an unconverged run.

The single-sequence fit completed on one H100 in `debug`, Slurm job `224916`,
in 2 minutes 50 seconds including qualification. The independent full-batch fit
completed on one H100 in `debug`, Slurm job `224917`, in 1 hour 14 minutes
18 seconds including qualification.
Both jobs passed GPU qualification, including exact checkpoint resume, before fitting.

### Single-sequence findings

The plateau controller reduced LR from 0.01 to 0.001 at update 3,430 and stopped
at update 4,112. The best retained static tensor is from update 4,041. All pretrained
parameters remained unchanged. All measurements were finite, but optimization
was uneven: 212 of 4,113 gradient evaluations exceeded the clipping threshold,
including 24 after the LR reduction.

| Measurement | Update | Fitting NLL | Fitting PPL |
| --- | ---: | ---: | ---: |
| Initial training graph | 0 | 2.611659288 | 13.62163433 |
| Best retained | 4,041 | 2.170698166 | 8.76440091 |
| Terminal | 4,112 | 2.172165871 | 8.77727391 |

Static coefficients improve fitting, but this run does not approach memorization.
The original query generator (`rd7bf9bc37cb6`) reached PPL 1.00544 from the same
parent checkpoint and sequence. The fit optimizers operate on different parameters
with different learning rates, so this comparison demonstrates the fitting achieved
by these runs; it does not prove a global optimum for static coefficients.

Qualification replayed the original coefficients exactly. The static projected
forward is repeatable, and its training graph differed by only 0.00015569 nats
from the separately compiled evaluation graph. Initialization and final fitting
metrics above use the same training graph. Checkpoint resume was exact.

### Full-batch findings

The plateau controller reduced LR from 0.01 to 0.001 at update 2,000 and stopped
at update 2,500 after 500 confirmation updates. The best retained static tensor
is from update 2,333. All 2,501 measurements and gradient norms were finite,
no gradient was clipped, and all pretrained parameters remained unchanged.

| Measurement | Update | Fitting NLL | Fitting PPL |
| --- | ---: | ---: | ---: |
| Initial training graph | 0 | 2.728801250 | 15.31451774 |
| Best retained | 2,333 | 2.720215797 | 15.18359847 |
| Terminal | 2,500 | 2.720302105 | 15.18490899 |

The static batch fit improves only slightly from its projected initialization
and plateaus far from memorization. Its final loss is close to the original
query-only checkpoint's batch NLL, 2.719785738: fitting mostly recovers the loss
introduced by averaging away the input and position dependence. The residual
difference is smaller than the recorded BF16 cross-compilation tolerance, so it
is not evidence that either original or fitted static coefficients are better.
This result describes the achieved fit under this optimizer and stopping rule,
not a proven global optimum or a held-out generalization measurement.

The batch static tensor is a restriction of the position-dependent tensor, on the
same source backbone and fitting batch. Comparing these fitting results tests the
benefit of query-position freedom. Neither fit evaluates unseen-sequence transfer,
and neither retrains the backbone as in the original static pretraining runs.

## Independent coefficients across a full batch

Opened: **2026-10-24**. Run `r7505c6d49711` fits independent degree-eight
coefficient tables on the first **256 packed evaluation sequences**, restoring
the query-only parent `r98d5895c02d0` at its retained step **7,500**. It tests
whether successful fits across sequences share a compact family of polynomial
shapes. This is the 4B counterpart of the older 10B batch oracle; both its parent
and its retained artifacts have **Kennel** as their home cluster.

All pretrained parameters stay frozen. Effective monomial coefficients have shape
`[256, 15, 2048, 14, 9]`: **990,904,320 independent fitted scalars**, shared only
across a query's causal keys. Each sequence starts with the coefficients captured
from its own original forward pass. The batch and parent checkpoint are pinned
by SHA256. These are free coefficient tables, so a shared causal generator's
ability to predict them remains a separate question.

The same batch is used by the shared-position and static batch diagnostics.
Microbatches of eight fit the independent tables with Adam in factorial
coordinates `r! c_r`, LR **0.01**, betas **0.9/0.95**, epsilon **1e-8**, no weight
decay, and gradient clipping at **1 independently for each sequence**. Each
reported update advances every sequence once. After at least 2,000 updates,
improvement below 0.001 nats over 200 updates reduces LR tenfold. Fitting stops
after 500 confirmation updates on a plateau; 10,000 is the safety budget.
Qualification checks original coefficient replay, finite improving updates,
unchanged pretrained parameters, and exact checkpoint resume on the allocated GPU.

### Findings

Slurm job **225066** completed on one **B300** on `akita` in `long` in **1 hour
5 minutes 8 seconds**, including qualification and coefficient analysis, using
the locked CUDA 13 environment. The learning rate decreased at update **2,000**;
fitting stopped at **2,500** after 500 confirmation updates on the plateau. The
retained best coefficients are from update **2,496**. All pretrained parameters
retained their original checksum. GPU qualification passed exact original-forward
replay, finite improving gradients, and exact checkpoint resume.

| Measurement | Update | Fitting NLL | Fitting PPL |
| --- | ---: | ---: | ---: |
| Initial | 0 | 2.719759941 | 15.17667851 |
| Best retained | 2,496 | 0.003100998 | 1.00310581 |
| Terminal | 2,500 | 0.003108311 | 1.00311315 |

**Independent query coefficients nearly memorize the full fitting batch with the
backbone frozen.** Each sequence has its own fitted table, so this establishes
capacity under that freedom, not transfer to unseen sequences or the ability of
a shared causal generator to predict the tables.

### Coefficient geometry

`analyze_batch_coefficients.py` analyzed the retained checkpoint on CPU after
fitting. It samples 64 uniformly spaced query positions from all 256 sequences,
multiplies coefficients by `r!`, and divides each nine-dimensional coefficient
vector by its positive L2 norm, preserving sign. This removes positive common
scale, which is largely redundant under head RMSNorm. PCA is centered separately
for each of the **210 layer/head pairs**. Compact measurements live in the run's
[batch_coefficient_geometry.json](../../../runs/r7505c6d49711/batch_coefficient_geometry.json),
with checkpoint identity, sampling settings, and analysis source provenance.

| Components | Median coefficient variance captured | Median error reduction on other fitted sequences |
| ---: | ---: | ---: |
| 1 | 44.4% | 44.0% |
| 2 | 69.1% | 69.2% |
| 3 | 82.9% | 82.9% |
| 4 | 89.8% | 89.7% |
| 5 | 94.4% | 94.4% |
| 6 | 96.9% | 96.8% |

The first column of measurements uses all fitting sequences. The last fits the
PCA basis on the first **192 sequences** and measures reconstruction on the
remaining **64**, relative to using the training-sequence mean alone. The median
layer/head requires **five components for 90%** of variance and **six for 95%**.
The similar reconstruction on the other fitted sequences indicates that the
coefficient directions share structure across these independently fitted tables.
It motivates testing a few shared polynomial basis functions with query-dependent
mixing weights, but does not establish that such a parametrization improves
training.

These are coordinate-dependent coefficient-space statistics, without weighting
by the attention scores actually encountered. They do not show that compressed
coefficients preserve attention outputs or fitting PPL. Every table was fitted
using its own sequence's targets, including the 64 sequences used to test the
basis, so the split is not a language-model generalization measurement. Whether
a causal generator can predict the fitted coefficients also remains untested.
The open decision is whether to reconstruct coefficients with fewer components
and check fitting PPL before changing the shared generator.

To recompute the CPU summary without changing the saved measurements or provenance:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  experiments/transformer_200m_4b/results/exploratory/coefficient_oracle/analyze_batch_coefficients.py \
  --run experiments/transformer_200m_4b/runs/r7505c6d49711 --dry-run --overwrite
```

## Open decision and regeneration

Decide whether these fitting results motivate changes to the shared coefficient
generator or its training. [selection.yaml](selection.yaml) names the exact runs
and their source parents. Measurements and retained fitted parameters live only
under each run. Launch commands are in the [study README](../../../README.md).

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_200m_4b/results/exploratory/coefficient_oracle
```

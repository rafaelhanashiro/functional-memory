# RBF basis counts and coefficient learning

Opened: **2026-10-21**. Home cluster: **Kennel**.

## Question and method

How does the number of fixed scalar Gaussian RBFs approximating `exp(score)`
affect unnormalized-attention language-model quality and computational cost
when each head uses output RMSNorm?
The [study](../../../experiments/attention_scaling/README.md#fixed-rbf-basis-counts)
owns the full method, frozen recipes, and launch commands.

## Runs

| RBF count | Run ID | Status |
| --- | --- | --- |
| 3 | `re530e0627233` | complete: 10B tokens |
| 5 | `r1b62a76f9b55` | complete: 10B tokens |
| 9 | `r46c3479f9681` | complete: 10B tokens |
| 20 | `rc28d7b48dc72` | running: job `223900` |

Each run is an independent seed-zero execution of the shared 200M-scale,
10B-token recipe. Gaussian L2 fit diagnostics are stored in each frozen
configuration; they are not language-model results. The comparison selects existing baseline runs from the maintained attention-scaling collection; no new baseline training is allocated.

### 1× Chinchilla cohort

Four fresh seed-zero runs use the same RBF fits with 3,998,744,576 training
tokens (20 per total trainable parameter, rounded to a full batch), a matching
7,627-update cosine schedule, and 160-update warmup. Model, data, initialization,
and remaining optimizer settings are unchanged. They have no measured quality
outcomes yet. Their learning curves and terminal results must remain distinct
from the 10B-token cohort; an intermediate point on a longer learning-rate
schedule is not a completed 1× training run.

| RBF count | Run ID | Status |
| --- | --- | --- |
| 3 | `r4614f6fff21f` | submitted: job `224475` |
| 5 | `r0fbe4040cc36` | submitted: job `224474` |
| 9 | `r6e1b71e49350` | submitted: job `224473` |
| 20 | `r8f7f83e9b6ee` | acceptance checks: job `224472` |

### Learned 9-RBF comparison

Two fresh 1× runs test whether learning the nine fitted coefficients improves
quality: static per-layer/head weights (`r55b028eb88a6`) and query-dependent
`b + a*tanh(x W)` weights (`r564c8c864d15`). Both initialize at the same fitted
kernel as fixed control `r6e1b71e49350`; centers and widths stay fixed.
The [study recipe](../../../experiments/attention_scaling/README.md#learned-9-rbf-coefficients-at-1-chinchilla)
specifies initialization, exact budgets and launch commands. Both runs completed
on 2026-10-25, with finite diagnostics and their terminal checkpoint also validation
best. Their measurements remain outside the canonical attention-scaling selection.

| Coefficients | Run ID | Training tokens | Terminal validation NLL | Terminal validation PPL |
| --- | --- | ---: | ---: | ---: |
| Static learned | `r55b028eb88a6` | 3,998,744,576 | 2.764058 | 15.864093 |
| Query-dependent learned | `r564c8c864d15` | 4,028,104,704 | 2.757587 | 15.761758 |

Query dependence lowers terminal perplexity by 0.65% in these seed-zero runs.
Its additional parameters also receive additional training tokens under the same
20-tokens-per-parameter convention. At the common 7,600-update checkpoint
(3,984,588,800 tokens), perplexities are 15.866787 and 15.774150, respectively;
their learning-rate schedules still differ with their total budgets. This small
single-seed difference does not establish a robust advantage. Both retain native
offline W&B histories for upload once project permissions permit it. The open
decision includes whether to promote coefficient learning and input dependence.

## Findings and open decision

The 3-, 5-, and 9-RBF runs completed 10,000,269,312 tokens. Their terminal validation perplexities are 13.9501, 13.6150, and 13.6213; each terminal checkpoint is also validation best. The 20-RBF run is still training, with validation perplexity 13.6394 at 9,437,184,000 tokens. These are seed-zero observations without a selected matched baseline. Median throughput over the last five validation intervals is 192,383, 184,216, 166,527, and 112,053 tokens/s, respectively. Training intervals include checkpoint work and exclude validation. The runs use one H100 each in Kennel's standard queue, with unnormalized attention, per-head output
RMSNorm, QK normalization, the native Triton backend, and microbatch 16.
Data access is verified. The recipes request online W&B logging, but creation
in the configured project is denied for account `ltorroba`; training retains
its local logs and canonical measurements. Production is gated on
GPU kernel checks, a full-recipe pilot, and checkpoint resumption. The initial
comparison will report validation NLL against training tokens, throughput,
numerical failures, and the observed score ranges relative to the fitting
measure. Failed and incomplete coordinates will remain visible.

The comparison below uses the exact baseline IDs already selected by
`results/attention_scaling/selection.yaml`, `degree_chart.yaml`, and
`output_scaling.yaml`. All selected dot-product models match shape, initialization,
QK normalization, optimizer settings, training schedule, global batch, seed, and
train/evaluation data hashes. The normalized references do not use head RMSNorm;
the exp and learned degree-4 controls do. RBF coefficients, centers, and widths
are **fixed**, whereas the polynomial control learns its coefficients per layer
and head. Gated DeltaNet is an architecture reference with its own reference
initialization and 210.8M parameters, 5.5% more than RBF. Historical backends,
microbatches, rematerialization, GPU counts, and GPU models differ, so this is a
quality comparison, not a measured runtime comparison.

The completed 5-RBF and 9-RBF models are nearly tied (13.615 versus 13.621).
Five fixed RBFs are 0.116 PPL below plain softmax and 0.024 below the learned
degree-4 polynomial with head RMSNorm. The latter gap is too small to establish
an advantage from one seed. Exp with matching head RMSNorm remains stronger:
13.411, or 0.204 below 5-RBF. The 3-RBF model is weaker at 13.950. The 20-RBF
point is explicitly incomplete at 9.437B tokens and is not a terminal result.
Terminal and validation-best checkpoints coincide for every completed selected
run. Ten billion tokens are 2.501× the 20-tokens-per-total-parameter convention
for the 199,916,704-parameter RBF model. The 1× cohort is a separate training
schedule and is not included in this figure.

![RBF and existing baseline validation perplexities](comparison.png)

[PDF](comparison.pdf) · [Exact values](summary.csv)

<!-- BEGIN RBF COMPARISON -->
| Mechanism | Run ID | Tokens (B) | Validation PPL | Metric |
| --- | --- | ---: | ---: | --- |
| Exp + head RMSNorm | `r8ad66f3912c2` | 10.000 | 13.411 | terminal |
| Learned polynomial, degree 4 + head RMSNorm | `rc2b555b905dd` | 10.000 | 13.639 | terminal |
| Fixed RBF, 3 functions + head RMSNorm | `re530e0627233` | 10.000 | 13.950 | terminal |
| Fixed RBF, 5 functions + head RMSNorm | `r1b62a76f9b55` | 10.000 | 13.615 | terminal |
| Fixed RBF, 9 functions + head RMSNorm | `r46c3479f9681` | 10.000 | 13.621 | terminal |
| Fixed RBF, 20 functions + head RMSNorm | `rc28d7b48dc72` | 9.437 | 13.639 | incomplete: latest pinned validation |
| Softmax | `rdf00da0866a8` | 10.000 | 13.731 | terminal |
| Fixed Taylor, degree 4 | `r67e1c33d19fa` | 10.000 | 13.927 | terminal |
| Fixed Taylor, degree 8 | `r10baef5d8428` | 10.000 | 13.741 | terminal |
| Fixed Taylor, degree 16 | `r697221e420ad` | 10.000 | 13.740 | terminal |
| Gated DeltaNet (210.8M parameters) | `rdf74e4289926` | 10.000 | 13.594 | terminal |
<!-- END RBF COMPARISON -->

The open decision is whether to promote the fixed-basis comparison and the
learned-coefficient experiments into the canonical attention-scaling collection.
The 10B-token figure above uses fixed coefficients; the learned 1× results are
reported separately above.

## Regeneration

The explicit run selection and comparison settings are pinned in `selection.yaml`;
canonical measurements remain under the runs. The script verifies all plotted
source hashes, data hashes, model shapes, reference initialization, optimizer
settings, and evaluation recipes before rendering. Run from the worktree root:

```bash
scaffolds/transformer/.venv/bin/python results/exploratory/rbf_basis/compare.py
```

After deliberately updating selected run outputs, refresh their pins with
`python -m scaffolds.records refresh results/exploratory/rbf_basis`, then rerun
the comparison. The CSV, PNG, PDF, and generated table reproduce byte-identically.
The 20-RBF run's pinned partial observations must be updated from its owning
worktree when terminal results become available; do not infer completion from
another run. Refresh the study index with
`python -m scaffolds.records index experiments/attention_scaling`.

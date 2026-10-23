# Fixed RBF basis counts

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
configuration; they are not language-model results. No baseline runs have
been selected, and no new baseline training is allocated.

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

Existing baseline run IDs and recipes must be identified before making matched
claims. Match model shape, tokenizer and data manifests, optimization budget,
initialization, QK normalization, and attention normalization. Keep terminal
validation separate from the validation-best checkpoint. The open decision is
whether the resulting RBF comparison belongs in the canonical attention-scaling
collection after these controls and the results have been checked.

## Regeneration

The explicit run selection is pinned in `selection.yaml`; canonical measurements remain under the runs. The values above use terminal measurements for completed runs and the latest pinned validation for the incomplete run. Refresh pins with `python -m scaffolds.records refresh results/exploratory/rbf_basis` and regenerate the study index with `python -m scaffolds.records index experiments/attention_scaling`.

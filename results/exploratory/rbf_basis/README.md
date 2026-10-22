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
| 3 | `re530e0627233` | submitted: job `223903` |
| 5 | `r1b62a76f9b55` | acceptance: job `223902` |
| 9 | `r46c3479f9681` | running: job `223901` |
| 20 | `rc28d7b48dc72` | running: job `223900` |

Each run is an independent seed-zero execution of the shared 200M-scale,
10B-token recipe. Gaussian L2 fit diagnostics are stored in each frozen
configuration; they are not language-model results. No baseline runs have
been selected, and no new baseline training is allocated.

## Findings and open decision

Production has started for the accepted configurations; completed language-model results are not available yet. The runs use one H100 each in Kennel's standard queue, with unnormalized attention, per-head output
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

There are no measured results or derived figures yet. Measurements will remain
under the run directories. Refresh the study index with
`python -m scaffolds.records index experiments/attention_scaling`; add an
explicit pinned selection here when outputs and baseline IDs are available.

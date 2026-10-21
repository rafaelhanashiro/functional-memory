# Fixed RBF basis counts

Opened: **2026-10-21**. Home cluster: **Kennel**.

## Question and method

How does the number of fixed scalar Gaussian RBFs approximating `exp(score)`
affect normalized-attention language-model quality and computational cost?
The [study](../../../experiments/attention_scaling/README.md#fixed-rbf-basis-counts)
owns the full method, frozen recipes, data-access prerequisite, and launch commands.

## Runs

| RBF count | Run ID | Status |
| --- | --- | --- |
| 3 | `rd3733c250ced` | planned |
| 5 | `r1b60d90c83e5` | planned |
| 9 | `r467aea7027c0` | planned |
| 20 | `r5bbad3217224` | planned |

Each run is an independent seed-zero execution of the shared 200M-scale,
10B-token recipe. Gaussian L2 fit diagnostics are stored in each frozen
configuration; they are not language-model results. No baseline runs have
been selected, and no new baseline training is allocated.

## Findings and open decision

No language-model measurements are available yet. The runs are planned;
readable FineWeb data and GPU qualification are outstanding. The initial
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

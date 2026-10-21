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
| 3 | `r10b2cb22687c` | cancelled by request; job `223562` |
| 5 | `r1b486c356e1a` | cancelled by request; job `223563` |
| 9 | `r27f0d0f59eb1` | cancelled by request; job `223564` |
| 20 | `r873f76c0a542` | cancelled by request; job `223565` |

Each run is an independent seed-zero execution of the shared 200M-scale,
10B-token recipe. Gaussian L2 fit diagnostics are stored in each frozen
configuration; they are not language-model results. No baseline runs have
been selected, and no new baseline training is allocated.

## Findings and open decision

All four runs were cancelled by request before the first scheduled validation;
they provide no validation-quality comparison. Their logs retain the partial
training measurements. The 3-basis run retains a resumable checkpoint at step
500; the other three stopped before their first production checkpoint. All four
passed kernel checks, a full-recipe pilot, and checkpoint resumption on one H100
before production. The recipes use unnormalized attention, per-head output
RMSNorm, and QK normalization. Throughput optimization is pending the shared
scaffold's accumulation benchmark, separately from this scientific comparison.
The exploratory question remains open, and these incomplete coordinates remain
visible.

Existing baseline run IDs and recipes must be identified before making matched
claims. Match model shape, tokenizer and data manifests, optimization budget,
initialization, QK normalization, and attention normalization. Keep terminal
validation separate from the validation-best checkpoint. The open decision is
whether the resulting RBF comparison belongs in the canonical attention-scaling
collection after these controls and the results have been checked.

## Regeneration

There are no validation results or derived figures yet. Measurements remain
under the run directories. Refresh the study index with
`python -m scaffolds.records index experiments/attention_scaling`; add an
explicit pinned selection here when outputs and baseline IDs are available.

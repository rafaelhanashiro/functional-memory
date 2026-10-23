# 200M / 4B comparisons

This is the result collection for the [isolated recipe](../README.md). It references
only explicit run IDs from this study; raw measurements and checkpoints remain
under `../runs/`. Old 10B-token results are not included in these comparisons.

`lr_selection.yaml` pins the retained winning softmax pilot and records the canonical
peak LR 0.001 and floor 0.0001. The short LR-screening summary is in the
[study README](../README.md#lr-selection-and-requested-comparison). Only the winning
pilot remains in `summary.csv`; the screening curves are not retained.

`selection.yaml` explicitly selects the query–key variants at degrees 8 and 12.
The other attention variants have their own explicit run selections in their execution branches.
`summary.csv` distinguishes latest measurements from terminal measurements.
Numerical failures retain their stopping step separately from the last valid
validation step; validation-best measurements are never substituted for terminal ones.
`attention_comparison.png` and `.pdf` show completed PPL as dots and incomplete
latest measurements as crosses, with statuses displayed. Missing and failed
variants remain in the table; they are never replaced with another run.

From the worktree/repository root:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_200m_4b/results
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.results.plot
```

Refresh pins only after intentionally updating the selected measurements. Rendering
checks those pins and regenerates deterministic summaries and figures. Findings
will be recorded here once the corresponding measurements exist.

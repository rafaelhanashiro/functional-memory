# 200M / 4B comparisons

This is the result collection for the [isolated recipe](../README.md). It references
only explicit run IDs from this study; raw measurements and checkpoints remain
under `../runs/`. Old 10B-token results are not included in these comparisons.

`lr_selection.yaml` selects the three softmax pilots and records the rule and,
once measured, the canonical peak/floor LR. `learning_rates.png` and `.pdf` show
their held-out NLL curves. No LR has been selected while the canonical value is
null. The 1,500-step comparison is an early screen on the full 7,570-step cosine
schedule, not a full-budget LR optimum.

After LR selection, `selection.yaml` explicitly lists the nine attention variants.
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

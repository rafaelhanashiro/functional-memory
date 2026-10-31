# Transformer++ results

Results for the [Transformer++ study](../README.md) live here. Selections name explicit
run IDs and pin their scientific inputs; measurements, recipes, execution records,
and retained artifacts remain under `../runs/`.

The current collection is the exploratory
[coefficient oracle](exploratory/coefficient_oracle/README.md): fitting free coefficients, the trained query generator, or coefficient tables shared across sequences and optionally query positions
with the pretrained backbone frozen. Its fitting
perplexity is not held-out validation perplexity and enters no canonical
pretraining comparison. Findings and interpretation belong to the topic README.

The [exploratory index](exploratory/README.md) lists open questions. Refresh the
oracle's explicit selection from the repository/worktree root with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/coefficient_oracle
```

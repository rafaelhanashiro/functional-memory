# Transformer++ results

Results for the [Transformer++ study](../README.md) live here. Selections name explicit
run IDs and pin their scientific inputs; measurements, recipes, execution records,
and retained artifacts remain under `../runs/`.

The [degree scaling](degree_scaling/README.md) collection plots 1× validation
perplexity against polynomial degree for fixed Taylor, fixed Hermite, learned static,
and query coefficients with projection weight decay; its new runs are training.

The current exploratory collection is the
[coefficient oracle](exploratory/coefficient_oracle/README.md): fitting free coefficients, the trained query generator, or coefficient tables shared across sequences and optionally query positions
with the pretrained backbone frozen. Its fitting
perplexity is not held-out validation perplexity and enters no canonical
pretraining comparison. Findings and interpretation belong to the topic README.

The [exploratory index](exploratory/README.md) lists open questions, and the
[diagnostics index](diagnostics/README.md) lists closed records of model behavior. Refresh the
oracle's explicit selection from the repository/worktree root with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/coefficient_oracle
```

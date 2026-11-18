# Transformer++ results

Results for the [Transformer++ study](../README.md) live here. Selections name explicit
run IDs and pin their scientific inputs; measurements, recipes, execution records,
and retained artifacts remain under `../runs/`.

The canonical [degree scaling](degree_scaling/README.md) collection plots 1× terminal
validation perplexity against polynomial degree `m = 1-8` and 12 for fixed Taylor, fixed
Hermite, learned static, and query coefficients with projection weight decay (Taylor and
query only at 12), with softmax as the reference level; all thirty-four runs are complete.

The canonical [model scale](model_scale/README.md) collection repeats eight attention
variants at the 12-layer width-640 shape (98,977,016 parameters at 1×) and compares each
with its reference-shape partner; all sixteen runs are complete.

The canonical [score MLP](score_mlp/README.md) collection compares a conventional SiLU MLP
of the score and the query token, in place of exp at widths `J = 4` and 8, with exp, query
`m = 4` and fixed Taylor `m = 4`; both MLP runs are complete.

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

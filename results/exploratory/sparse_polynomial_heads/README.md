# Sparse polynomial approximations of pretrained attention heads

Opened **2026-10-15**. Home cluster: **Kennel**.

Can a small subset of polynomial features reproduce pretrained softmax attention,
and does choosing a different subset for each query improve accuracy at the same
number of active features? This is an exploratory comparison of frozen Llama
heads, intended to test feature sparsity before training a new language model.

## Runs and method

Run recipes and measurements belong to the
[attention study](../../../experiments/attention_scaling/README.md).
[selection.yaml](selection.yaml) explicitly names the runs and pins their
scientific inputs. The comparison uses the pretrained Llama-3.2-1B checkpoint
at revision `4e20de362430cd3b72f300e6b0f18e50e7166e08`; its Q/K/V projections,
RoPE, score scaling, and all backbone parameters remain fixed. Heads 0 and 16
from layers 2, 8, and 14 are chosen before fitting. These are zero-based query
head indices, with each query using its associated grouped-attention KV head.

Every head has the same candidate bank: the 47,905 distinct monomials of total
degree zero through three in 64 coordinates. The constant feature is always
active and counts toward the budget. Each selected feature has a learned weight.

| Variant | Active features per query | Purpose |
| --- | ---: | --- |
| Dense polynomial | 47,905 | Full candidate-bank reference |
| Learned fixed subset | 128, 512, 2,048 | Test a shared sparse set across queries |
| Query-dependent subset | 128, 512, 2,048 | Test whether different queries need different terms |
| Random fixed subset | 128, 512, 2,048 | Control for learning the subset, with fitted weights |

The fixed subset is selected once from the dense fit by ranking each learned
weight times its query-feature RMS and key-feature RMS on training documents.
Its retained weights are then fitted again. This is a screening heuristic, not
an exhaustive search over sparse subsets. Dynamic selection ranks the current
query-feature magnitude times key-feature RMS, fitted weight, and learned
relevance. Its forward pass uses an exact hard budget; fitting uses a sigmoid
straight-through gradient for relevance. Weights are positive, but monomials
and kernel values can have either sign.

The selector uses the current query and learned feature relevance; it cannot
inspect the attention state or teacher output. Key features remain dense.
Dynamic selection therefore measures the potential to reduce state reads while
retaining the full feature bank and dense writes. It does not establish a latency
improvement or a smaller stored state. Feature selection and dense polynomial
feature construction also have computational costs.

Fitting and model selection use separate text from the held-out test documents.
The frozen run recipes specify the exact token counts, context lengths, document
selection, optimizer, fitting schedule, and random seeds. Longer-context test
measurements check whether an approximation fitted on shorter sequences transfers.

## Measurements and interpretation

The main metric is relative RMS head-output error, aggregated over held-out query
positions for each head:

\[
E_{\mathrm{output}} =
\sqrt{\frac{\sum_t\|\widehat y_t-y_t\|_2^2}
{\sum_t\|y_t\|_2^2}}.
\]

The teacher and approximation use the same values. Their weights are respectively
softmax scores and the sparse polynomial kernel divided by its causal sum.
An error of zero is exact reconstruction; an error of one equals the error from
zeroing every teacher output. This metric is not language-model accuracy.
Normalized attention-weight error is reported separately, because output
reconstruction can hide weight differences when value vectors are similar.
Invalid denominators and nonfinite outputs remain visible through per-head
failure fractions and run outcomes; they must not be silently clipped away.

Held-out results use the checkpoint selected on validation data. Terminal and
validation-best validation measurements are separate series, so selection does
not obscure the terminal fit. The full polynomial is a reference, not a guaranteed
lower bound on error: query-dependent selection changes the function class.

The primary comparison is error versus the number of active features, separately
for all six preselected heads. Learned fixed selection outperforming random
selection would support useful monomial structure. Dynamic selection outperforming
fixed selection at the same budget would support query-dependent feature utility.
Neither outcome alone establishes sparse writes, polynomial GDN dynamics, or an
improvement in language-model loss after replacing heads.

## Findings and run status

No experimental findings are available yet. Runs and measurements will appear
below only after they have been explicitly selected and regenerated.

<!-- BEGIN GENERATED STATUS -->

No selected run status has been generated yet.

<!-- END GENERATED STATUS -->

## Regeneration

From this experiment's checkout and its locked environment:

```bash
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m scaffolds.records refresh results/exploratory/sparse_polynomial_heads
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m results.exploratory.sparse_polynomial_heads.plot
```

The plot verifies all selected scientific inputs before writing `summary.csv`,
the per-head figures, and the generated status table. It reads committed compact
measurements and requires neither cached activations nor a model checkpoint.
Missing or unsuccessful coordinates remain visible in the table and figures;
they are not replaced by another run. Figure and CSV output is deterministic.

## Open decision

After the planned comparison, decide whether the accuracy–feature-budget curves
justify investigating sparse key writes or structured polynomial features, whether
to promote the evidence into the maintained attention collection, or whether to
discard this topic and its runs. No outcome or promotion has been decided.

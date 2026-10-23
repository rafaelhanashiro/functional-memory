# Free polynomial coefficients on fixed inputs

Opened: **2026-10-23**. Home cluster: **Kennel**.

How much can a fixed pretrained model improve on a sequence or a batch by choosing
its degree-eight polynomial coefficients freely at every query? This is a fitting
capacity diagnostic. It does not estimate held-out performance or prove an optimum.

Both runs start from the retained step-19,074 checkpoint of query-only run
`r50be104c2a9e` (reference initialization, QK normalization, unnormalized attention,
per-head output RMSNorm, and factorially scaled learned coefficients). Every
pretrained parameter is frozen. The coefficient generator is bypassed with free
coefficients per example, layer, query position, head, and polynomial order. A query
uses the same polynomial across all its causal keys; there are no free pairwise
attention weights. Coefficients start at their actual values in the checkpoint's
original forward pass, including input dependence in every layer.

| Role | Run | Fitting sequences | Free coefficients |
| --- | --- | ---: | ---: |
| Single sequence | `r9678a74c0b2d` | 1 | 2,949,120 |
| Full batch | `r7dd028e36501` | 256 | 754,974,720 |

The inputs are the first 256 packed evaluation sequences, each with 2,048 prediction
positions. The single-sequence run uses the first of the same sequences. These
inputs become training data for this diagnostic: all reported NLL and perplexity
are **fitting metrics**, with no generalization claim. The free tables are independent
across examples, so the full batch comprises 256 independent fits with a common
stopping rule. Per-example gradient scaling and clipping preserve that independence.

The optimizer learns `r! c_r` with Adam (learning rate 0.01, betas 0.9/0.95,
epsilon 1e-8, no weight decay, gradient norm clipped to 1 independently per
sequence). After at least 2,000 updates, an improvement of less than 0.001 nats/token
over 200 updates triggers a tenfold learning-rate reduction. The fit continues for
500 confirmation updates and stops only if the loss remains on a plateau; otherwise
it continues. A 10,000-update safety budget pauses an unconverged fit. The best fitting
coefficients are retained, alongside resumable progress while fitting. The original
checkpoint is read in place and never changed.

## Findings and open decision

The runs are prepared; measurements will be recorded in each run's `metrics.jsonl`.
The open decision is whether the observed fitting improvement warrants a new
coefficient generator or further optimization work. A gain demonstrates attainable
capacity on these inputs; a plateau alone cannot establish a capacity ceiling.

The [selection](selection.yaml) pins the two executions. The
[study instructions](../../../experiments/attention_scaling/README.md#fixed-input-coefficient-oracle)
give exact launch commands. Refresh this selection when measurements change:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/exploratory/coefficient_oracle
```

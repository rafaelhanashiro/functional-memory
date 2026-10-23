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
| Single sequence | `rec7a01f3282c` | 1 | 2,949,120 |
| Full batch | `rfc37cfdf49d0` | 256 | 754,974,720 |

The inputs are the first 256 packed evaluation sequences, each with 2,048 prediction
positions. The single-sequence run uses the first of the same sequences. These
inputs become training data for this diagnostic: all reported NLL and perplexity
are **fitting metrics**, with no generalization claim. The free tables are independent
across examples, so the full batch comprises 256 independent fits with a common
stopping rule. Per-example gradient scaling and clipping preserve that independence.

The optimizer uses `r! c_r` coordinates with Adam (learning rate 0.01, betas 0.9/0.95,
epsilon 1e-8, no weight decay, gradient norm clipped to 1 independently per
sequence). Effective coefficients are stored directly to preserve the captured
initialization exactly; gradients and updates apply the factorial scaling. After
at least 2,000 updates, an improvement of less than 0.001 nats/token
over 200 updates triggers a tenfold learning-rate reduction. The fit continues for
500 confirmation updates and stops only if the loss remains on a plateau; otherwise
it continues. A 10,000-update safety budget pauses an unconverged fit. The best fitting
coefficients are retained, alongside resumable progress while fitting. The original
checkpoint is read in place and never changed.

## Findings and open decision

Both runs passed GPU qualification on Kennel. Qualification checked captured-coefficient replay, finite
gradients, a loss-reducing update, repeated forwards, and exact checkpoint
continuation. The software checks are recorded in the
[scaffold validation](../../../scaffolds/transformer/validation/README.md#direct-query-coefficient-capture-and-override).
The single-sequence run completed 2,500 updates on a B300. Initial fitting NLL was
2.532260; terminal NLL was 0.004548 (perplexity 1.004558). The best fitting NLL was
0.004539 at update 2,387, whose coefficients are retained. The learning rate was
reduced at update 2,000, followed by 500 confirmation updates. This nearly memorizes
the sequence while keeping every pretrained parameter fixed.

The 256-sequence run is running on one H100. Its initial fitting NLL is 2.662276;
terminal results are pending. Each run's `metrics.jsonl` holds its measurements.
The open decision is whether the observed fitting improvement warrants a new
coefficient generator or further optimization work. A gain demonstrates attainable
capacity on these inputs; a plateau alone cannot establish a capacity ceiling.

The [selection](selection.yaml) pins the two executions. The
[study instructions](../../../experiments/attention_scaling/README.md#fixed-input-coefficient-oracle)
give exact launch commands. Refresh this selection when measurements change:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/exploratory/coefficient_oracle
```

# Free coefficients from the 4B learned-static checkpoint

Opened: **2026-10-23**. Home cluster: **Kennel**.

How much can the 4B-token learned-static model improve on one sequence when each
query can choose its polynomial coefficients freely? This is one exploratory
fitting run, `r3c0f21ab7dce`, in
[attention scaling](../../../experiments/attention_scaling/README.md#fixed-input-coefficient-oracle).
It enters no canonical comparison.

The source is degree-eight learned-static run `r9b8acfcbeb89` from the 200M / 4B
setup, using its retained validation-best checkpoint at step **7,500**. The
backbone has 15 layers, width 896, 14 heads, QK RMSNorm, unnormalized polynomial
attention, and per-head RMSNorm on its output. Every pretrained parameter is
frozen. Free effective monomial coefficients have axes
`[sequence, layer, query, head, order]` and shape `[1, 15, 2048, 14, 9]`:
**3,870,720 trainable scalars**, initialized by broadcasting the checkpoint's
learned static coefficients across query positions. Each query uses the same
polynomial across its causal keys.

The input is the first packed evaluation sequence, with 2,048 next-token prediction
positions, also used in the earlier single-sequence oracle. It becomes training
data for this diagnostic; all NLL and perplexity are fitting metrics, with no
generalization claim. The frozen recipe and run record pin the parent recipe,
checkpoint, and dataset checksums. The parent checkpoint is read in place.

The existing coefficient-only runner uses Adam in factorial-scaled coordinates,
LR 0.01, betas 0.9/0.95, epsilon 1e-8, no decay, and gradient norm clipped to 1.
After at least 2,000 updates, improvement below 0.001 nats over 200 updates reduces
the LR tenfold. Fitting stops after 500 confirmation updates if the loss remains
on a plateau; the 10,000-update safety limit pauses an unconverged run. Only the
best fitting coefficients are retained after completion. GPU acceptance must
verify coefficient replay, finite improving updates, and exact checkpoint resume
before fitting starts.

## Findings and open decision

The run is prepared; fitting results are not yet available. The open decision is
whether free query coefficients reveal useful additional fitting capacity in this
4B learned-static checkpoint. A large gain does not establish that a shared causal
coefficient generator can realize it, or explain the small held-out gap between
learned-static and fixed Taylor pretraining.

[selection.yaml](selection.yaml) names the single run. Measurements and retained
coefficients live only under its run directory. Update the selection after new
measurements with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/exploratory/coefficient_oracle_4b
```

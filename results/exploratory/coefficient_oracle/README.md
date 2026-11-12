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

The 256-sequence run completed 2,500 updates on one H100. Initial fitting NLL was
2.662276; terminal NLL was 0.003071 (perplexity 1.003076). The best fitting NLL was
0.003062 at update 2,484, whose coefficients are retained. It also reduced the
learning rate at update 2,000 and completed 500 confirmation updates. Each run's
`metrics.jsonl` holds its measurements.
The open decision is whether the observed fitting improvement warrants a new
coefficient generator or further optimization work. A gain demonstrates attainable
capacity on these inputs; a plateau alone cannot establish a capacity ceiling.

## Last-query polynomial shapes

The single-sequence checkpoint at update 2,387 supplies the effective coefficients
for the last trained query, input position 2,048 (zero-based index 2,047), which
predicts the final target token. Each layer and head has its own degree-eight
function `p(s) = sum_r c_r s^r`; the stored coefficients already include their
factorial scaling. The argument is the dot product divided by `sqrt(64)` after
QK normalization and RoPE, using the operands passed to the attention kernel.

The [raw curves](last_query_polynomials.png) and
[shape comparison](last_query_polynomial_shapes.png) show all 16 heads in layer 10.
The corresponding [raw PDF](last_query_polynomials.pdf) and
[shape PDF](last_query_polynomial_shapes.pdf) cover all ten layers, one page per
layer. Blue is the fitted polynomial, orange is the pretrained polynomial on the
original sequence, and dashed gray is `exp(s)`. Every panel uses the fitted query's
observed score range, with score-quantile ticks; there is no extrapolation.

In the final layer, nine heads assign negative weights to at least one observed
key, and nine fitted functions are nonmonotone on the observed score interval.
Heads 1, 3, 10, and 13 visibly give positive weight to both strongly negative and
strongly positive scores, with smaller weights in between. Other heads remain
much closer to the exponential shape. Negative weights were already possible
before fitting: 13 pretrained head functions are negative on at least one of the
fitted query's scores. The median cosine similarity of kernel values to `exp(s)`
on those same scores changes from 0.9946 before fitting to 0.9042 afterward.

The shape comparison divides each curve by its own positive RMS on the same
fitted-score samples, preserving signs. This removes overall magnitude from the
visual comparison because per-head output RMSNorm largely cancels positive
scaling (up to its epsilon). Raw values remain available separately. Earlier
coefficient changes also change later-layer Q/K activations despite frozen
weights, so the original and fitted score distributions are measured separately.
The exponential is a function reference, not a separately fitted control; these
figures describe one fitted query and do not establish general superiority over
exponential attention.

The compact measurement `last_query_polynomials.json` lives with the single-sequence
run and is hash-pinned in the selection. It contains all plotted coefficients,
score quantiles, scaling factors, and checkpoint/input identities. Full score
samples remain in that run's ignored `last_query_polynomials.npz`; figures do not
require that archive or a checkpoint. The instrumented forward reproduces the
selected fitting NLL (0.004538551 versus recorded 0.004538547).

Regenerate the figures and per-head summary from committed measurements on CPU:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python results/exploratory/coefficient_oracle/plot_last_query.py \
  --selection results/exploratory/coefficient_oracle/selection.yaml
```

To re-extract the measurements on Kennel, use a one-GPU allocation and the GPU's
locked environment from [CLUSTERS.md](../../../CLUSTERS.md):

```bash
PYTHONPATH=. python results/exploratory/coefficient_oracle/extract_last_query.py \
  --run experiments/attention_scaling/runs/rec7a01f3282c --overwrite
```

After intentionally replacing that measurement, update its SHA256 in
`settings.last_query_analysis.measurement_sha256`, refresh the selection, and
regenerate the figures. Both PNGs, both PDFs, and the summary CSV reproduce
byte-identically from the pinned measurement and plotting code.

The [selection](selection.yaml) pins the two executions. The
[study instructions](../../../experiments/attention_scaling/README.md#fixed-input-coefficient-oracle)
give exact launch commands. Refresh this selection when measurements change:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/exploratory/coefficient_oracle
```

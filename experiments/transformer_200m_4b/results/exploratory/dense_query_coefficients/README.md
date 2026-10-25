# Dense query coefficients in full 4B pretraining

Opened: **2026-10-25**. Home cluster: **Kennel**.

Does a dense query-coefficient generator, whose K tanh features are mixed into every
polynomial coefficient, beat the original one-feature-per-coefficient query generator and
fixed Taylor when trained from scratch on the 200M / 4B recipe? Does decoupling K from the
degree m help? Runs live in the [200M / 4B study](../../../README.md) and enter no
canonical comparison.

## Generator

For the current query's pre-attention normalized hidden state x and attention score s:

    h_j(x) = tanh(w_j^T x),                    j = 1..K
    c_r(x) = [b_r + sum_j V[r,j] h_j(x)] / r!, r = 0..m
    P_x(s) = sum_r c_r(x) s^r

Parameters are separate per layer and head and shared across sequences and positions;
V is [m+1, K]. Fresh runs start at V = 0 with the canonical Taylor bias and query
projection draw (std 1/sqrt(896)), so every run starts exactly at Taylor, like the
original query model's zero-amplitude start. The original query generator is
c_r(x) = [b_r + a_r tanh(w_r^T x)] / r!; fixed Taylor is c_r = 1/r!.

## Runs

Every run repeats the canonical 4B recipe (seed 42, data order 0, global batch 256 x 2048,
7,570 updates, peak/minimum LR 1e-3/1e-4 with 455 warmup, QK RMSNorm, unnormalized
attention with per-head output RMSNorm, Mosaic kernel, one GPU) and changes only the
coefficient generator, degree, and the explicitly selected score/readout scaling.

| Priority | Run | Label | m | K | Generator |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `rb83b6290a229` | `dense-query-m2-k3` | 2 | 3 | dense |
| 1 | `r7de01791778b` | `dense-query-m4-k5` | 4 | 5 | dense |
| 1 | `r98403594f227` | `dense-query-m8-k9` | 8 | 9 | dense |
| 2 | `re6d9f618439d` | `dense-query-m2-k9` | 2 | 9 | dense |
| 2 | `r5397d762caa1` | `dense-query-m4-k9` | 4 | 9 | dense |
| 3 | `r1bbf3b794a14` | `query-m2` | 2 | | original query |
| 3 | `r5755f840bacc` | `fixed-m2` | 2 | | fixed Taylor |

Baselines from the 4B comparison: original query `r49b7943aee1b` (m = 4, best PPL 14.835)
and `r98d5895c02d0` (m = 8, 14.300); fixed Taylor `r79da929de109` (m = 4, 14.756) and
`r019ca96d6533` (m = 8, 14.256). [selection.yaml](selection.yaml) names every run and role.

## Per-head QK scale and reduced dense mixing

| Degree | Joint scaling run | Matched dense control |
| --- | --- | --- |
| 4 | `r04a5b2a3e8dd` | `r5397d762caa1` |
| 8 | `rfccffdd239dd` | `r98403594f227` |

Jobs **225920** (degree four) and **225921** (degree eight) are running GPU
qualification on one H100 each on Kennel `dogo`, in `standard`, with 24-hour
allocations. Production training begins only after every acceptance stage passes.

The additional degree-four and degree-eight runs both use **K = 9**. They retain
the original RMS normalization and separate Q/K channel gains shared within each
layer. Each layer/head additionally learns a positive scalar
`alpha = exp(gamma)`, multiplying its attention scores before the polynomial.
`gamma` starts at zero, so `alpha` starts at one. The new scale does not replace
RMSNorm, change its channel gains, or use cosine normalization.

The dense readout becomes

    c_r(x) = b_r/r! + (1/(3 r!)) sum_j V[r,j] tanh(w_j^T x).

Only the input-dependent mixing contribution is divided by three; the Taylor
bias and the existing factorial scaling are retained. `V=0` at initialization,
so the initial polynomial and score function match the corresponding original
dense run. All model parameters train from scratch on the same 7,570-update,
approximately 4B-token recipe. The learning rate is unchanged.

The matched dense controls are `r5397d762caa1` (m=4, K=9) and `r98403594f227`
(m=8, K=9). This comparison tests the **joint** effect of head-specific QK scaling
and smaller dense readout contributions. It does not distinguish which change
causes any observed difference. The original query and fixed-Taylor selections
remain available as reference mechanisms.

## Comparison and status

The two completed K=9 dense runs give the following **terminal** held-out PPLs at
update 7,570:

| Degree | Dense, K=9 | Original query | Fixed Taylor |
| --- | ---: | ---: | ---: |
| 4 | 15.0949 | 14.8401 | 14.7600 |
| 8 | 14.9206 | 14.3053 | 14.2609 |

Both completed dense variants trail their same-degree references. Their
validation-best PPLs are 15.0881 (degree four) and 14.9125 (degree eight), both at
step 7,500; those best checkpoints are distinct from the terminal measurements
above. These single-seed results motivate the joint scaling test, but do not
identify an optimization cause or establish the outcome of that test.

All runs train 524,288 prediction tokens per update, so held-out PPL is compared at
matched updates 500, 1,000, 1,500, 3,000, 5,000, 7,000 and 7,570, with terminal
(update 7,570) and validation-best metrics reported separately. Regenerate the table with:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh \
  experiments/transformer_200m_4b/results/exploratory/dense_query_coefficients
scaffolds/transformer/.venv/bin/python -m experiments.transformer_200m_4b.results.exploratory.dense_query_coefficients.compare
```

Refresh pins only when intentionally incorporating new measurements. The collector
rejects changed scientific inputs until that refresh, reports execution status
from each run's record, and keeps unavailable or incomplete measurements visible.
Its summary includes the per-head QK scale flag and dense readout multiplier.

**Open decision:** whether the dense generator, including the jointly scaled
variant, improves held-out performance enough to join the canonical comparison.

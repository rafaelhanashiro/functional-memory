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
coefficient generator and degree.

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

## Comparison and status

All runs train 524,288 prediction tokens per update, so held-out PPL is compared at
matched updates 500, 1,000, 1,500, 3,000, 5,000, 7,000 and 7,570, with terminal
(update 7,570) and validation-best metrics reported separately. Regenerate the table with:

```bash
scaffolds/transformer/.venv/bin/python experiments/transformer_200m_4b/results/exploratory/dense_query_coefficients/compare.py
```

Running. **Open decision:** whether the dense generator joins the canonical comparison.

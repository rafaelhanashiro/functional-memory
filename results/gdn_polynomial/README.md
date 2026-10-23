# Polynomial Gated DeltaNet

Do independent erasure and higher-degree memory improve Gated DeltaNet (GDN), and do
they let narrow keys match wide ones? Terminal held-out perplexity of 200M models after
19,074 updates (10.0B tokens), seed 0. No training happens here; the figures and
tables are built from the runs named in [`selection.yaml`](selection.yaml). Home
cluster: Kennel. Cancelled runs retain their cells and any recorded measurements.

## Model

The GDN block keeps width-four causal convolutions and SiLU on Q/K/V, L2-normalized
queries and keys, Mamba-2 scalar decay `alpha_t`, gated output RMSNorm, and no RoPE.
Every gate is a scalar per token, layer, and head.

- **Tied GDN (reference):** `S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T`,
  `beta_t = sigmoid(W_beta x_t)`; the read is `S_t^T q_t / sqrt(d_K)`.
- **Independent erase (`m = 1`):** the erase gate `lambda_t = sigmoid(W_lambda x_t)`
  replaces `beta_t` in `I - lambda_t k_t k_t^T`. `W_lambda` starts as a copy of
  `W_beta`, so the model starts identical to tied GDN.
- **Polynomial memory (`m >= 2`):**

  ```text
  F_t(q) = alpha_t F_{t-1}(P_t q) + v_t sum_(r=0..m) beta_{t,r} <q, k_t>^r,   P_t = I - lambda_t k_t k_t^T
  read_t = F_t(q_t) / sqrt(d_K)
  ```

  `beta_{t,1}` is the sigmoid write above; each added order `r = 0, 2, ..., m` has
  `beta_{t,r} = tanh(W_r x_t)` with `W_r` zero at the start, so every polynomial model
  starts as its `m = 1` control. Unrolled, token `i` contributes
  `(prod_(j=i+1..t) alpha_j) sum_r beta_{i,r} s_ti^r v_i` with
  `s_ti = q_t^T P_t ... P_{i+1} k_i`: with `lambda = 0` and `alpha = 1` this is
  unnormalized polynomial attention whose coefficients depend on the key (writing)
  token.
- **Tanh write:** degree one with `beta_t = tanh(W_beta x_t)` (signed writes from the
  first update) and the sigmoid independent erase gate.

Shared recipe: 10 layers, width 1024, 16 heads, `d_V = 64`, MLP width 3008, sequence
length 2,048, BF16, batch 256 sequences, AdamW 3e-4 to 3e-5 with 400 warmup updates,
selective weight decay 0.1, gradient clipping at 1, and each mechanism's reference
initialization. Softmax references use QK-norm.

<!-- BEGIN GENERATED RESULTS -->
### Terminal perplexity

18 cells: 10 complete, 0 incomplete, 0 pending, 7 cancelled, 1 not run, 0 failed.

![Terminal perplexity by key width](terminal_ppl.png)

[PDF](terminal_ppl.pdf)

| d_K | softmax | Gated DeltaNet | GDN m = 1 | GDN m = 2 | GDN m = 3 | GDN m = 4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 15.732 | 14.816 | cancelled | cancelled | 13.817 | cancelled |
| 16 | 14.470 | 14.217 | cancelled | cancelled | 13.439 | cancelled |
| 64 | 13.731 | 13.594 | 13.579 | 13.179 | cancelled (update 7,000: 15.64) | — |

### Against tied Gated DeltaNet

![Validation perplexity against tied Gated DeltaNet](validation_gap.png)

[PDF](validation_gap.pdf) · [Per-run measurements](summary.csv)

### Degree-one variants (d_K = 64)

| Erase gate | Linear write | Run | Terminal PPL | vs tied reference |
| --- | --- | --- | ---: | ---: |
| Tied to the write gate | sigmoid | `rdf74e4289926` | 13.594 |  |
| Tied to the write gate | sigmoid | `rf640fc13f3a8` | 13.583 | -0.08% |
| Independent | sigmoid | `rf7341031e3d5` | 13.579 | -0.11% |
| Independent | tanh | `rf65d38ec0552` | 13.927 | +2.44% |
<!-- END GENERATED RESULTS -->

## Findings

- **Quadratic memory gives the lowest canonical `d_K = 64` perplexity:** 13.179, 2.95%
  below its independent-erase control (13.579) and 3.06% below tied GDN (13.594). The
  best `d_K = 64` attention run in [attention scaling](../attention_scaling/README.md) is
  exp with head RMSNorm (13.411); softmax is 13.731.
- **Cubic memory makes narrow keys competitive:** at `d_K = 16` it reaches 13.439,
  5.5% below tied GDN at the same width (14.217) and below tied GDN at `d_K = 64`. At
  `d_K = 8` it reaches 13.817, 6.7% below tied GDN (14.816) and 12.2% below softmax
  (15.732). The width-matched independent-erase controls were cancelled before training; at `d_K = 64`
  independent and tied erase differ by 0.1%.
- **The polynomial gain holds through training.** Against tied GDN, quadratic
  `d_K = 64` stays between -2.3% and -3.1% from update 1,000 onward. Cubic `d_K = 64`
  tracks it: -3.2% at update 7,000, against -3.1% for quadratic at the same update.
- **Independent erase is a near-tie with tied erase:** 13.579 against 13.594 and
  13.583, two tied runs of one recipe on H200 and B300.
- **A tanh linear write hurts:** 13.927, 2.6% above the sigmoid control. Against tied
  GDN it starts 9% behind at update 1,000 and ends 2.4% behind.
- **The order-0 write dominates the gradient.** Polynomial runs clip far more often
  than linear ones (at most 0.5% of updates): quadratic `d_K = 64` 95.6% of updates,
  cubic `d_K = 64` 92.8% through update 7,000, cubic `d_K = 16` 27.4%, and cubic
  `d_K = 8` 1.8%. In every polynomial run the order-0 write projection carries almost
  all of the gradient norm (1.76 of 1.77 in quadratic's final window). With clipping
  at norm 1, this scales every other parameter's update down by the same factor.

## Cancelled runs

No polynomial GDN runs remain queued. The width-matched controls at `d_K = 8`
and 16 (independent erase `m = 1`, quadratic `m = 2`) and quartic `m = 4` at both
widths were cancelled before training. Cubic `d_K = 64` was also cancelled; this
collection retains its pinned measurements through update 7,000, not a terminal
result. All seven runs remain selected, with their recipes and retained artifacts.

## Limits

- One seed. Two tied GDN runs of one recipe differ by 0.08%, so smaller differences
  are not established.
- Arithmetic differs by mechanism: tied GDN uses FLA's `chunk_gated_delta_rule`,
  independent erase and tanh writes use FLA chunks with local independent-gate WY
  kernels, and polynomial runs use local transported-score Triton kernels. Hardware
  differs too (H200, B300, H100, and A100 for early cubic attempts); each record lists
  its attempts.
- Each added order adds 163,840 parameters: cubic `d_K = 64` has 211.5M against
  210.8M for tied GDN (+0.3%). At `d_K = 8` and 16 the models have 192.4–195.7M.
- The `d_K = 8` and 16 comparisons use tied GDN; their independent-erase
  controls were cancelled before training.
- The order-0 gradient also inflates clipping, so polynomial and linear runs train
  with different effective step sizes. Whether limiting it helps is untested.

## Reproduce

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh results/gdn_polynomial
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.gdn_polynomial.plot
```

The plot checks every pinned input and each recipe's mechanism, width, erase gate,
write, and degree, then writes both figures, [`summary.csv`](summary.csv) (terminal,
validation-best, and last-validation metrics, differences from tied GDN and from each
run's control, parameter counts, and gradient clipping), and the generated tables. No
GPU or dataset needed. Recipes, records, and launch commands are in the
[attention study](../../experiments/attention_scaling/README.md#polynomial-gated-deltanet).

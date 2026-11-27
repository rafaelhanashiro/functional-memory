# Exponential prefactors at Q/K width 4

Opened 2026-11-26. Do the softplus and sigmoid prefactors help the exponential at `d_K` 4, as they do at `d_K` 64
(softplus(s) exp(s) 14.132 and sigmoid(s) exp(s) 14.186 against exp's 14.257, in
[pointwise_attention](../../results/pretraining/alternative_attention/fixed_kernels/pointwise_attention/README.md))?
Each run is the shifted exponential's recipe at that width (`r2739c80cc032`) with the prefactor, on the blocked
causal strips (the one implementation that shifts a score function by its row maximum; at this width the scores pass
exp's FP32 range) and head RMSNorm's max-abs prescale
([methods and launch](../../README.md#exponential-prefactors-at-key-width-four)).

| Run | Weight | Best (terminal) | vs exp | Gradient spikes |
| --- | --- | ---: | ---: | ---: |
| `r8aead78ffa55` | softplus(s) exp(s) | 19.552 (19.559) | +1.89% | 171 |
| `r7530874e8610` | sigmoid(s) exp(s) | 19.240 (19.247) | +0.26% | 3 |
| `r2739c80cc032` | exp(s) | 19.190 (19.201) | – | 3 |

- **Neither prefactor helps at `d_K` 4.** Softplus is 1.9% behind exp; sigmoid ties it.
- **The softplus prefactor trains unstably.** Its spikes fall mostly between updates 1,000 and 3,000, where its
  validation perplexity is up to 15% behind exp's (update 2,000). Softplus grows like `s` at the large scores this
  width reaches; sigmoid is bounded by 1 and trains like exp.
- Limits: one seed. The exp run trained on the fused kernel with head RMSNorm's plain arithmetic, the same
  functions; the prefactor runs trained on one B300 each.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/narrow_exp_prefactors
```

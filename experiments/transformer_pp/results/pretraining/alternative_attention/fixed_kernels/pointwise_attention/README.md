# Sigmoid and softplus attention

Opened 2026-11-18. Home cluster: Kennel.

Does replacing the exponential score function with sigmoid or softplus improve
Transformer++ at the 15-layer width-896 reference shape?

The explicit [selection](selection.yaml) compares sigmoid `r88d8654a8a1d` and
softplus `r266c5dd1f935` with exp `r6a76e7fbde45`. For each head,
`s_ti = q_t^T k_i / sqrt(64)` and `u_t = sum_(i<=t) phi(s_ti) v_i`, followed by
per-head RMSNorm. Attention is unnormalized. The two new functions are
`phi(s) = sigmoid(s)` and `phi(s) = log(1 + exp(s))`, evaluated stably.

Both preserve the control's RoPE, QK RMSNorm with learned positive per-head
score scales, model shape, initialization, data, seed 42, optimizer, global batch
256, microbatch 16, context 2,048, and 7,570-update budget. They contain exactly
the same 198,453,586 trainable parameters as exp. No polynomial coefficients,
score offsets, or additional gates are learned.

The shared scaffold expresses each function as fixed `p(z) = z`, coefficients
`[0, 1]`, after the selected score transform. Its Mosaic transformed attention
kernels compute the causal weighted sum. The historical exp control used its
fused exponential backend. It also summed input embedding gradients for repeated
token IDs in BF16. The sigmoid and softplus runs sum those gradients in FP32.
Their comparison with each other matches embedding precision. Their comparisons
with historical exp include both the score function and the precision change.

The [study launch instructions](../../../../../README.md#sigmoid-and-softplus-attention)
use one B200 per run in `standard`. Before production, acceptance verifies the
pinned control recipe, every initialized parameter and optimizer decay mask,
forward and backward kernel correctness, the full-recipe training pilot, and
fresh-process checkpoint resume. Run records and logs hold the current status.

The [scaffold validation record](../../../../../../../scaffolds/transformer/validation/README.md#fixed-sigmoid-and-softplus-attention-2026-11-18)
describes the repeated-token gradient check and the full-model qualification.

## Results

Validation perplexity, best with terminal in parentheses. Both runs trained on one B200.

| Run | Weight | Best (terminal) | vs exp |
| --- | --- | ---: | ---: |
| `r266c5dd1f935` | softplus(s) | 15.744 (15.748) | +10.43% |
| `r88d8654a8a1d` | sigmoid(s) | 16.135 (16.142) | +13.17% |
| `r6a76e7fbde45` | exp(s) | 14.257 (14.262) | – |

- Neither fixed function comes close to exp: softplus trails it by 10.4% and sigmoid by
  13.2%. Learned polynomials of softplus(s), in the [softplus_sos](../../../parametrizations/sos/softplus_sos/README.md)
  topic, reach 14.13–14.34.
- Limits: one seed. The comparison with exp also includes the embedding-gradient precision change described
  above.

## The exponential with score functions

Added 2026-11-19. Does a parameter-free function of the score help the exponential? Three runs change
only exp's weight `exp(s)`, keeping everything else of `r6a76e7fbde45`:

| Run | Weight | Behaviour |
| --- | --- | --- |
| `rbd24b948a240` | softplus(s) exp(s) | ≈ exp(2s) for very negative s, ≈ s exp(s) for large s |
| `rd7c9f610e9dc` | sigmoid(s) exp(s) | ≈ exp(2s) for very negative s, ≈ exp(s) for large s |
| `r0dd96f88d9f8` | exp(softplus(s)) | exactly 1 + exp(s): exp plus a uniform causal average |

They train on the fused unnormalized kernel, which evaluates these functions in its forward and backward
passes (errors against FP64 about 3e-7 in FP32). The embedding-gradient caveat above applies to them too.
The softplus prefactor and input trained on one B300, the sigmoid prefactor on one H100.

| Run | Weight | Best (terminal) | vs exp |
| --- | --- | ---: | ---: |
| `rbd24b948a240` | softplus(s) exp(s) | 14.132 (14.136) | −0.88% |
| `rd7c9f610e9dc` | sigmoid(s) exp(s) | 14.186 (14.190) | −0.50% |
| `r0dd96f88d9f8` | exp(softplus(s)) | 14.329 (14.332) | +0.51% |
| `r6a76e7fbde45` | exp(s) | 14.257 (14.262) | – |

- **Both prefactors improve on exp**, softplus by 0.88% and sigmoid by 0.50%. Both damp keys with
  negative scores to about exp(2s); softplus also grows like s exp(s) for large scores. Without learned
  coefficients, softplus(s) exp(s) comes within 0.3% of the learned softplus SoS polynomials (14.095–14.128).
- **The softplus input hurts** by 0.51%: exp(softplus(s)) adds a uniform causal average to exp.
- Limits: one seed. The margins are below 1%, comparable to what the FP32 embedding-gradient sums may
  change; an exp control with current code would isolate the score function.

## Open question

Whether any fixed function, alone or multiplying the exponential, merits further comparison with exp and learned
SoS attention, starting with an exp control on current code; otherwise delete the topic with its runs and their
artifacts.

Canonical measurements remain with the runs. Refresh the explicit source pins
when updating this comparison:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/alternative_attention/fixed_kernels/pointwise_attention
```

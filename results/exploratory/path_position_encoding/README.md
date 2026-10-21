# PaTH position encoding

Opened 2026-10-19. Open decision: promote into
[`results/attention_scaling/`](../../attention_scaling/README.md) (PaTH as a position
encoding beside RoPE), extend it to more kernels and seeds, or discard.

Does replacing RoPE with PaTH (scores transported through token Householder products,
`S_ts = q_t^T H_t ... H_(s+1) k_s / sqrt(d_K)`, `H_t = I - beta_t w_t w_t^T`, the
authors' gate design; [scaffold](../../../scaffolds/transformer/README.md#path-position-encoding))
improve softmax and query-conditioned tanh polynomial attention?

Each PaTH run repeats its RoPE control with `model.positional_encoding: path`
(`path_w_rank: 32`, `path_conv_size: 3`); otherwise only the attention implementation
differs, and for softmax the global microbatch (32 instead of 64, same batch). PaTH adds
850,080 parameters (0.4%). All runs: 200M, `d_K = 64`,
QK-norm, reference initialization, seed 0, 400 warmup updates, 19,074 updates (10.0B
tokens). The tanh runs use the zero-amplitude, `N(0, 1/hidden_size)`-draw recipe with
head RMSNorm. Runs and roles: [`selection.yaml`](selection.yaml); method and launch:
[study README](../../../experiments/attention_scaling/README.md#path-position-encoding-softmax-and-tanh-m--2-4).

## Findings

Validation perplexity at update 19,074, which is also each run's validation best:

| Kernel | RoPE | PaTH | Δ |
| --- | ---: | ---: | ---: |
| Softmax | 13.731 (`rdf00da0866a8`) | **13.401** (`rf1ece7e314b3`) | -0.330 |
| Tanh query-conditioned, m = 2 | 14.562 (`rec520e9303ac`) | **13.912** (`rb901a10c4777`) | -0.650 |
| Tanh query-conditioned, m = 4 | 13.623 (`rd4f66b1959d2`) | **13.249** (`r4de2b1429c8a`) | -0.374 |
| Exp with head RMSNorm (reference) | 13.411 (`r8ad66f3912c2`) | | |

- PaTH improves all three kernels; the gain is largest at m = 2.
- PaTH tanh m = 4 (13.249) is the best run here, below exp with head RMSNorm (13.411).
- PaTH softmax (13.401) ties exp with head RMSNorm.

## Limitations

- One seed per run.
- The m = 2 and m = 4 runs changed implementation mid-run: the JAX PaTH reference
  (`blocked`) up to updates 7,500 and 6,500, then FLA's fused Triton kernels (`fused`;
  each record's `attention_backend_segments`). Fused BF16 errors are larger than the
  reference's (output and q/k/v gradients 1.5-3x, PaTH-gate and coefficient gradients
  about 8x, all below 1e-2), and its dk/dv/dw/dbeta are not bitwise repeatable. In
  1,000-update pilots of the m = 2 recipe, validation NLL was 3.7188 on `blocked` and
  3.7373 and 3.7258 in two `fused` repeats, within run-to-run variation
  ([validation](../../../scaffolds/transformer/validation/README.md#fused-path-attention-kernels)).
  Softmax ran on `blocked` throughout.

# Matched tokens

Opened: 2026-11-12. Status: **open**; each run's status is in its record and the study's `runs.csv`.

Trained on the reference shape's token count, twice its own 1x budget, how do the attention variants compare
at the 12-layer width-640 shape at degrees eight and twelve? How close does each come to its reference-shape
counterpart at the same tokens?

## Runs

Each run is a 12-layer width-640 1x run derived with `derive_recipe.py --budget 2`: 15,103 updates of
128 × 2,048 tokens (3.96B prediction tokens, against the reference shape's 3.97B), warmup 908, validation
every 200 updates through 3,000 and every 1,000 after, seed 42. Where the 1x partner has another degree, only
`attention_degree` changes. One GPU each; home cluster Kennel.

| Run | Label | Variant | `m` | Recipe source | Attention |
| --- | --- | --- | ---: | --- | --- |
| `r3759b2a57a54` | `query-m8-w-decay-l12-d640-2x` | tanh query, RoPE, W decay | 8 | `r95a5691d293f` | Mosaic |
| `r0f6d929e5992` | `query-m12-w-decay-l12-d640-2x` | tanh query, RoPE, W decay | 12 | `r95a5691d293f`, degree 12 | Mosaic |
| `r3cca4c13edab` | `static-m8-l12-d640-2x` | learned static | 8 | `r94e0f2b8cff9`, degree 8 | Mosaic |
| `r84f325593209` | `static-m12-l12-d640-2x` | learned static | 12 | `r94e0f2b8cff9`, degree 12 | Mosaic |
| `rd77db79142e6` | `fixed-m8-l12-d640-2x` | fixed Taylor | 8 | `r2c432bd3a2a4`, degree 8 | Mosaic |
| `rffab7c4eeac5` | `fixed-m12-l12-d640-2x` | fixed Taylor | 12 | `r2c432bd3a2a4`, degree 12 | Mosaic |
| `rc6a7e50d27e1` | `query-m8-w-decay-l12-d640-softplus-input-2x` | softplus input, `c_0` from 0 | 8 | `r3759b2a57a54` with the transform | Mosaic transformed, H100 |
| `r53d5bbcac16f` | `query-m12-w-decay-l12-d640-softplus-input-2x` | softplus input, `c_0` from 0 | 12 | `r0f6d929e5992` with the transform | Mosaic transformed, H100 |
| `rccd2cb5e8bab` | `query-m8-sum-of-squares-w-decay-l12-d640-2x` | sum-of-squares query | 8 | `r3759b2a57a54` with sum of squares | Mosaic |
| `r08be7093202a` | `query-m12-sum-of-squares-w-decay-l12-d640-2x` | sum-of-squares query | 12 | `r0f6d929e5992` with sum of squares | Mosaic |
| `r656e8e652b72` | `query-m8-forget-gate-no-rope-w-decay-l12-d640-2x` | FoX query: cumulative gate, no RoPE, W decay | 8 | `r3d14b598d88f`, degree 8 | causal strips |
| `rbf05f7baab5e` | `query-m12-forget-gate-no-rope-w-decay-l12-d640-2x` | FoX query | 12 | `r3d14b598d88f`, degree 12 | causal strips |
| `rd920368a922e` | `exp-l12-d640-2x` | exp | – | `r3e2fa2532c23` | fused Triton |
| `r3bb5b8bd5c6a` | `softmax-l12-d640-2x` | softmax | – | `rf81c96f76a1d` | cuDNN |
| `re634d82ee479` | `exp-forget-gate-no-rope-l12-d640-2x` | FoX exp | – | `ra5dfd978d3bd` | causal strips |

- The softplus runs are the 2x query runs with the reference-shape softplus run's changes: the transform, `c_0`
  started at 0 and gradient-event capture. Their 1x counterpart `r1d342ca6e58c` trained on the causal strips.
- The sum-of-squares runs are the 2x query runs with the reference-shape sum-of-squares runs' changes.
- The forget-gate runs train on the causal strips, where their 1x partners used the blocked scan: the same
  arithmetic in another summation order.

Recipes, acceptance and launch commands:
[study README](../../../README.md#the-12-layer-width-640-shape-at-the-reference-token-count).

## Comparisons

Each 2x run is compared with:

- its 1x run at the 12-layer width-640 shape, where one exists at its degree: query 17.592, softplus 17.488,
  exp 17.550, softmax 17.964 and FoX exp 16.753, all at degree eight or without a degree;
- the reference shape's 1x run of the same variant and degree, at almost the same tokens: query 14.323 (`m = 8`)
  and 14.226 (`m = 12`), fixed Taylor 14.218 and 14.309, static 14.294 (`m = 8`), sum of squares 14.196 and
  14.165, exp 14.257, softmax 14.543, and FoX query 13.907 (`m = 8`) and FoX exp 13.628.

Validation-best perplexity is reported, with terminal values beside it.

## Open decision

Once the runs complete: promote the topic to its own canonical collection, or discard it with its runs and
their artifacts.

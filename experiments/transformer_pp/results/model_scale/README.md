# Model scale

Canonical collection, created 2026-11-07 at the user's request.

**Question.** Do the attention-variant comparisons of the Transformer++ recipe hold at a
smaller model shape? Seven variants train at the 12-layer width-640 shape, each against its
partner at the 15-layer width-896 reference shape, both at 1× Chinchilla.

## Method

Each 12-layer run is its partner's recipe derived by the study's
[recipe rules](../../README.md#model-shapes-and-token-budgets) with
`derive_recipe.py --layers 12 --width 640`: 10 heads of width 64 for queries, keys and
values, SwiGLU width 1,664, batch 128, 7,551 updates (1.98B prediction tokens), warmup 454,
and the reference's 4,864 validation sequences (38 batches of 128) at the same cadence.
The 1× budget comes from the normalized-softmax model at the shape (98,977,016 parameters)
and holds for every variant. A query-coefficient projection initializes at $1/\sqrt{640}$.
Everything else is the partner's: data and its order, tokenizer, context 2,048, positional
encoding, QK RMSNorm with the learned head scale, per-head output RMSNorm, initialization,
peak learning rate 0.001 with floor 0.0001, AdamW settings, seed 42, and attention backend.

| Variant | 12-layer width-640 run | Reference-shape partner | Partner best |
| --- | --- | --- | ---: |
| Softmax | `rf81c96f76a1d` | `r9040f12f3d80` | 14.543 |
| Exp | `r3e2fa2532c23` | `r6a76e7fbde45` | 14.257 |
| Fixed Taylor m = 4 | `r2c432bd3a2a4` | `r816ab0fbe119` | 14.722 |
| Static m = 4 | `r94e0f2b8cff9` | `r4339a245820e` | 14.767 |
| Query m = 4 + W decay | `r4cce9a985a70` | `rca97cc9db8f2` | 14.709 |
| FoX query m = 4 + W decay, no positional encoding | `r3d14b598d88f` | `r4a80e28ce0a7` | 13.873 |
| FoX exp, no positional encoding | `ra5dfd978d3bd` | `r183f08346ddc` | 13.628 |

The [selection](selection.yaml) pins each pair. Launch commands are in the
[study README](../../README.md#model-scale).

## Findings

Pending: the 12-layer runs are allocated and not yet trained. The summary table and figure
will regenerate from the selection once they complete.

## Limits

- The step is 0.50× in total parameters but 0.41× without embeddings and about 0.25× in
  compute, because the 32,000-token vocabulary is fixed: embeddings are 41% of the smaller
  model and 29% of the reference.
- Shape and batch change together, by rule, so the update count stays at the reference's.
- The peak learning rate stays 0.001 at both shapes; it is not re-tuned for either.
- One seed per run.

## Regeneration

Re-pin the selection from the worktree root after the runs change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/model_scale
```

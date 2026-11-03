# Why query attention breaks past 2,048 tokens

Opened 2026-11-02.

**Question.** In [context_length](../context_length/README.md), the 2,048-token query + W decay
(m = 4) checkpoint reaches perplexity 3,081 by 16,384 tokens while FoX without RoPE stays near
13. Its scores at unseen distances are not larger
([extrapolation_scores](../../diagnostics/extrapolation_scores/README.md)). Does the break come
from keys beyond the training distance at all, and if so, is the trigger their RoPE phases or
how many of them there are?

## Run and method

`rb786bd19c6b1` evaluates the validation-best checkpoint of `rca97cc9db8f2` (update 7,500) per position
on the 608 held-out rows of 16,384 tokens that the original 16k evaluation
(`r2a45c518f703`) scored, with only the attention call replaced:

| Pass | Keys each query attends to | RoPE |
| --- | --- | --- |
| `full` (control) | all earlier keys | true positions |
| `window` | the most recent 2,048 keys (distance < 2,048) | true positions |
| `clamp` | all earlier keys | true positions within 2,047 tokens; distance 2,047 beyond |

`window` removes every key beyond the training distance, so no query sees more keys or
farther ones than in training. `clamp` keeps every key but gives each one farther than
2,047 tokens the phase of distance 2,047 (the query rotated to position 2,047 and the key left
at position 0), so every relative phase was seen in training while the number of keys grows.
The replacement computes the model's own unnormalized polynomial with its query
coefficients in FP32, in 256-query blocks, before the unchanged per-head output RMSNorm.
[`evaluate.py`](evaluate.py) implements it; on CPU it matches the scaffold's blocked
reference kernel, and `window` and `clamp` match a brute-force pairwise computation.

Qualification: `full` and `clamp` replay the model's 4,864-row validation set at 2,048
tokens (where all three passes are the same function) within 0.002 nats of its recorded NLL,
and `full` at 16,384 tokens must match the original Mosaic evaluation's first 2,048
positions within 0.002 nats. Nothing is trained. Launch commands are in the
[study README](../../../README.md#extrapolation-ablation).

## Findings and open decision

Pending. Open decision: keep the result in the context-length discussion, or delete this
topic with its run.

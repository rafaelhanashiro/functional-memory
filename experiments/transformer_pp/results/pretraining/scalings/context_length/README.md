# Context length

Opened 2026-11-01.

**Question.** How do the Transformer++ attention variants compare when trained on 4,096-token
sequences instead of 2,048? Their behavior past the training length is the
[extrapolation_ablation diagnostic](../../../diagnostics/length_extrapolation/extrapolation_ablation/README.md).

## Training at 4,096 tokens

Five runs repeat five 2,048-token models at 4,096 tokens with the same tokens per update and
budget; the [study README](../../../../README.md#training-at-4096-tokens) gives the recipe and
launch commands. Their validation set is 2,432 rows of 4,097 tokens; each partner's
perplexity below is its own validation-best, on its 2,048-token validation rows. Every
run's best is at update 7,500 of 7,570.

| Model | Trained at 4,096 tokens: best PPL | 2,048-token partner: best PPL |
| --- | ---: | ---: |
| forget gate, no RoPE + W decay, m = 4 | **13.484** (`r58efc973ff28`) | 13.873 (`r4a80e28ce0a7`) |
| exp | 14.098 (`r2047d29a0d9f`) | 14.257 (`r6a76e7fbde45`) |
| softmax | 14.498 (`r636f433be44f`) | 14.543 (`r9040f12f3d80`) |
| query + W decay, m = 4 | 14.556 (`r738b35a8e314`) | 14.709 (`rca97cc9db8f2`) |
| fixed Taylor, m = 4 | 14.581 (`r339a31ad8d4b`) | 14.722 (`r816ab0fbe119`) |

- The ranking is unchanged: forget gate, exp, softmax, then query and fixed Taylor. The
  forget gate's lead over exp grows from 2.7% at 2,048 tokens to 4.4%.
- The polynomial models still trail softmax (+0.4%, +0.6%; +1.1%, +1.2% at 2,048 tokens) and
  exp (+3.3%, +3.4%).
- The two validation sets differ, so the columns compare training lengths, not the same text.

## Open question

Add context length as a dimension of the Transformer++ comparison, or delete the topic.

## Regeneration

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/pretraining/scalings/context_length
```

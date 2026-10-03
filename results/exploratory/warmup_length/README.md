# Warmup length

Opened 2026-10-03. The shared recipe warms up for 2,000 of 19,074 updates (10.5%,
1.05B tokens). Pretraining recipes warm up for 0.1–4% of training: GPT-3 375M tokens
(715 updates at our batch), llm.c 700 updates for this exact 10B-token budget,
Pythia and GPT-NeoX 1%, LLaMA 2 and OLMo 2 2,000 updates of 250k–1M. The 2,000 was
inherited from the OFM recipe when `total_steps` shrank to 19,074.

Does a standard warmup shift terminal perplexity by a small constant, leaving the
comparisons in [`results/attention_scaling/`](../../attention_scaling/README.md)
intact, or does it change them materially? One run repeats the canonical QK-norm
softmax recipe at `d_K = 64` with the transformer reference initialization and changes
only `optim.warmup_steps` from 2,000 to 400 (2.1%, 210M tokens).

Runs and roles: [`selection.yaml`](selection.yaml).

| Run | Warmup | Terminal PPL | Best PPL |
| --- | ---: | ---: | ---: |
| rbde7fc4c2135 (control) | 2,000 | 13.514 | 13.514 |
| rdf00da0866a8 | 400 | pending | pending |

## Open decision

If the gap is small, existing runs stay as they are and conclusions hold. Either way,
new recipes should adopt a standard warmup; existing runs keep their frozen recipes,
and the run index distinguishes the two settings through `optim.warmup_steps`.

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

## Results

Terminal validation perplexity (validation-best is terminal for both):

| Run | Warmup | Terminal PPL |
| --- | ---: | ---: |
| rbde7fc4c2135 (control) | 2,000 | 13.514 |
| rdf00da0866a8 | 400 | 13.731 |

Gap by update (warmup 400 minus warmup 2,000):

| Update | 1,000 | 2,000 | 3,000 | 5,000 | 8,000 | 10,000 | 12,000 | 14,000 | 16,000 | 18,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gap | −24.29 | −4.35 | −1.42 | −0.40 | −0.04 | +0.04 | +0.10 | +0.16 | +0.19 | +0.21 | +0.22 |

- **The short warmup leads early and loses late.** It is ahead until about update 9,000,
  then falls behind by a margin that is still widening at the end (+0.19, +0.21, +0.22 over
  the last 3,000 updates). The 10.5% warmup is not a handicap at this learning rate and
  batch; it is worth 0.22 perplexity, about the size of the QK-norm effect (0.21).
- **Sharper attention under the short warmup.** At update 18,000 the largest attention
  score on the diagnostic sample is 20.9 against 16.6 for the control, and the learned
  QK-norm scales span 5.9–25.5 against 3.1–21.9 (same median, 14.2).
- **Execution.** Updates 1–1,000 ran on H100 PCIe (two `sweep` preemptions), 1,001–18,000
  on B300, where update 18,408 produced an infinite gradient norm at a finite loss (2.648,
  no spike in the preceding 140 updates; `numerical_failure_step_00018408.json`). Replaying
  updates 18,001–19,074 from the step-18,000 checkpoint on H100 reproduced the loss
  trajectory (2.655403 against 2.655444 at update 18,400) and passed, so the event did not
  come from the model state. Whether it was B300-specific or a one-off was not tested; the
  step-18,000 checkpoint was not retained.

## Open decision

The recipe's `warmup_steps: 2000` stays canonical; existing comparisons stand. Decide
whether to keep this run as the recipe's warmup control (promote a one-line note into the
attention scaling results) or discard the topic with its run.

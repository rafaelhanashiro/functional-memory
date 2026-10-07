# Warmup length

Opened 2026-10-03. The shared recipe warms up for 2,000 of 19,074 updates (10.5%,
1.05B tokens). Pretraining recipes warm up for 0.1–4% of training: GPT-3 375M tokens
(715 updates at our batch), llm.c 700 updates for this exact 10B-token budget,
Pythia and GPT-NeoX 1%, LLaMA 2 and OLMo 2 2,000 updates of 250k–1M. The 2,000 was
inherited from the OFM recipe when `total_steps` shrank to 19,074.

Does a standard warmup shift every mechanism's terminal perplexity by a small constant,
leaving the comparisons in [`results/attention_scaling/`](../../attention_scaling/README.md)
intact, or does it change them? Each warmup-400 run repeats one cell of the `d_K = 64`
degree chart with the transformer reference initialization and changes only
`optim.warmup_steps` from 2,000 to 400 (2.1%, 210M tokens); peak and minimum learning
rates stay 3e-4 and 3e-5. Each is paired with its warmup-2,000 source.

Runs, roles, and pairs: [`selection.yaml`](selection.yaml).

## Results across the degree chart

Terminal validation perplexity at `d_K = 64`; every run's validation best is terminal.

| Cell | Warmup 2,000 | Warmup 400 | Difference |
| --- | ---: | ---: | ---: |
| Softmax | 13.514 | 13.731 | +0.22 |
| Softmax without QK-norm | 13.932 | 14.184 | +0.25 |
| Normalized, learned nonnegative m = 2 | 14.666 | 14.771 | +0.11 |
| Normalized, learned nonnegative m = 4 | 13.689 | 13.815 | +0.13 |
| Unnormalized, learned m = 1 | 17.883 | 17.874 | −0.01 |
| Unnormalized, learned m = 2 | 15.507 | 15.461 | −0.05 |
| Unnormalized, learned m = 3 | 14.657 | 14.630 | −0.03 |
| Unnormalized, learned m = 4 | 14.720 | 14.742 | +0.02 |
| Unnormalized exp | 15.885 | 15.663 | −0.22 |

- **Not a constant shift.** Normalized attention, both softmax lines included, loses
  0.11–0.25 with the short warmup; unnormalized learned polynomials are unchanged
  (−0.05 to +0.02); unnormalized exp gains 0.22.
- **Orders within each line hold.** Unnormalized learned ranks m = 3, 4, 2, 1 at both
  warmups; softmax stays ahead of softmax without QK-norm (by 0.42, then 0.45); exp stays
  between unnormalized learned m = 1 and m = 2.
- **Distances between families shrink.** Normalized minus unnormalized gaps narrow by
  0.1–0.45: softmax against unnormalized learned m = 3 from 1.14 to 0.90, learned
  nonnegative m = 4 against unnormalized learned m = 4 from 1.03 to 0.93, softmax against
  exp from 2.37 to 1.93. Learned nonnegative m = 4 against softmax narrows from 0.18 to 0.08.
- **Learned nonnegative against normalized fixed Taylor (m = 2, 4) is pending.** At warmup
  2,000 the nonnegative kernel leads fixed Taylor by 0.25 (14.666 against 14.911) and 0.09
  (13.689 against 13.774). The fixed Taylor warmup-400 cells run on ORCD and are not merged
  yet. With the nonnegative results above, the order changes only if fixed Taylor's own
  shift is below −0.14 at m = 2 or below +0.04 at m = 4.

Limits: one seed per cell. Learned nonnegative m = 4 trained on B300 then H100, and
unnormalized learned m = 2 on H100 then B300 (each resumed from a checkpoint after a
requeue); the other Kennel cells each ran on one GPU type.

## Softmax in detail

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

The recipe's `warmup_steps: 2000` stays canonical; existing comparisons stand. Decide whether
to keep these runs as the recipe's warmup controls (promote a note on the narrowed family
gaps into the attention scaling results) or discard the topic with its runs.

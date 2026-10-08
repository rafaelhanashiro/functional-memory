# Warmup length

Opened 2026-10-03. The shared recipe warms up for 2,000 of 19,074 updates (10.5%,
1.05B tokens). Pretraining recipes warm up for 0.1–4% of training: GPT-3 375M tokens
(715 updates at our batch), llm.c 700 updates for this exact 10B-token budget,
Pythia and GPT-NeoX 1%, LLaMA 2 and OLMo 2 2,000 updates of 250k–1M. The 2,000 was
inherited from the OFM recipe when `total_steps` shrank to 19,074.

Does a standard warmup shift every mechanism's terminal perplexity by a small constant,
leaving the comparisons in [`results/attention_scaling/`](../../attention_scaling/README.md)
intact, or does it change them? Each warmup-400 run repeats one `d_K = 64` cell of the
heatmap or degree chart with its reference initialization and changes only
`optim.warmup_steps` from 2,000 to 400 (2.1%, 210M tokens); peak and minimum learning
rates stay 3e-4 and 3e-5. Each is paired with its warmup-2,000 source.

Runs, roles, and pairs: [`selection.yaml`](selection.yaml).

## Results at d_K = 64

Terminal validation perplexity at `d_K = 64`; every completed run's validation best is
terminal, and failures show their update.

| Cell | Warmup 2,000 | Warmup 400 | Difference |
| --- | ---: | ---: | ---: |
| Softmax | 13.514 | 13.731 | +0.22 |
| Softmax without QK-norm | 13.932 | 14.184 | +0.25 |
| Normalized, fixed Taylor m = 1 | fail @35 | fail @67 | — |
| Normalized, fixed Taylor m = 2 | 14.911 | 15.014 | +0.10 |
| Normalized, fixed Taylor m = 3 | fail @555 | fail @218 | — |
| Normalized, fixed Taylor m = 4 | 13.774 | 13.927 | +0.15 |
| Normalized, fixed Taylor m = 8 | 13.543 | 13.741 | +0.20 |
| Normalized, fixed Taylor m = 16 | 13.519 | 13.740 | +0.22 |
| Normalized, learned nonnegative m = 2 | 14.666 | 14.771 | +0.11 |
| Normalized, learned nonnegative m = 4 | 13.689 | 13.815 | +0.13 |
| Unnormalized, fixed Taylor m = 1 | 17.903 | 17.930 | +0.03 |
| Unnormalized, fixed Taylor m = 2 | 18.166 | 18.099 | −0.07 |
| Unnormalized, fixed Taylor m = 3 | 15.621 | 15.604 | −0.02 |
| Unnormalized, fixed Taylor m = 4 | 15.888 | 15.862 | −0.03 |
| Unnormalized, learned m = 1 | 17.883 | 17.874 | −0.01 |
| Unnormalized, learned m = 2 | 15.507 | 15.461 | −0.05 |
| Unnormalized, learned m = 3 | 14.657 | 14.630 | −0.03 |
| Unnormalized, learned m = 4 | 14.720 | 14.742 | +0.02 |
| Unnormalized exp | 15.885 | 15.663 | −0.22 |
| Gated DeltaNet | 13.705 | 13.594 | −0.11 |

Unnormalized attention with output scaling (`attention_output_scaling`), same comparison:

| Cell | Warmup 2,000 | Warmup 400 | Difference |
| --- | ---: | ---: | ---: |
| Unnormalized exp, 1/t | 14.196 | 13.933 | −0.26 |
| Unnormalized exp, head RMSNorm | 13.441 | 13.411 | −0.03 |
| Unnormalized learned m = 4, 1/t | 14.056 | 13.816 | −0.24 |
| Unnormalized learned m = 4, head RMSNorm | 13.662 | 13.639 | −0.02 |

- **Not a constant shift.** Normalized attention, both softmax lines included, loses
  0.10–0.25 with the short warmup; unnormalized polynomials, fixed Taylor and learned,
  are unchanged (−0.07 to +0.03); unnormalized exp gains 0.22 and Gated DeltaNet 0.11. Within
  normalized fixed Taylor the loss grows with degree, from 0.10 at m = 2 to 0.22 at
  m = 16, the same as softmax.
- **Orders within each line hold.** Unnormalized learned ranks m = 3, 4, 2, 1 and
  unnormalized fixed Taylor m = 3, 4, 1, 2 at both warmups; normalized fixed Taylor ranks m = 16, 8, 4, 2 (m = 16 and 8 tie at warmup 400,
  13.740 against 13.741); softmax stays ahead of softmax without QK-norm (by 0.42, then
  0.45); exp stays between unnormalized learned m = 1 and m = 2.
- **Learned coefficients are not what makes unnormalized polynomials insensitive.** Fixed
  Taylor coefficients move as little as learned ones; among unnormalized kernels only exp
  gains from the short warmup.
- **m = 3 still beats m = 4 in both unnormalized polynomial lines.** Learned: by 0.06 at
  warmup 2,000 and 0.11 at warmup 400; fixed Taylor: by 0.27 and 0.26. The inversion is
  not caused by the long warmup.
- **Gated DeltaNet changes places.** At warmup 2,000 it trails softmax by 0.19 and also
  fixed Taylor m = 8 and 16 and learned nonnegative m = 4; at warmup 400 it leads every
  `d_K = 64` cell, softmax by 0.14. The `d_K = 64` softmax-against-Gated DeltaNet
  comparison in the attention scaling results therefore reverses with the warmup.
- **Distances between families shrink.** Normalized minus unnormalized gaps narrow by
  0.1–0.45: softmax against unnormalized learned m = 3 from 1.14 to 0.90, learned
  nonnegative m = 4 against unnormalized learned m = 4 from 1.03 to 0.93, softmax against
  exp from 2.37 to 1.93. Learned nonnegative m = 4 against softmax narrows from 0.18 to 0.08.
- **Fixed Taylor's gap to softmax narrows at low degree.** m = 2 trails by 1.28 instead
  of 1.40 and m = 4 by 0.20 instead of 0.26; m = 8 and 16 stay within 0.03 of softmax at
  both warmups.
- **Learned nonnegative still leads normalized fixed Taylor.** It is ahead by 0.24 at
  m = 2 (0.25 at warmup 2,000) and by 0.11 at m = 4 (0.09).
- **Odd fixed Taylor degrees still fail.** m = 1 fails at update 67 (35 at warmup 2,000)
  and m = 3 at 218 (555): the short warmup moves the failure in both directions and
  rescues neither.

- **Output scaling decides whether unnormalized attention gains from the short warmup.**
  With 1/t, exp and learned m = 4 both gain about 0.25, as plain exp does (0.22) while
  plain learned m = 4 does not (+0.02). With head RMSNorm both are unchanged (−0.03,
  −0.02). Softmax with head RMSNorm equals exp with head RMSNorm (RMSNorm cancels the
  row sum), so it has no separate cell.

Unnormalized fixed Hermite `m = 1–4` at warmup 400 runs on ORCD.

Loss spikes: unnormalized `m = 1`, fixed Taylor and learned, spikes at both warmups and
recovers; no `m ≥ 2` run spikes. Spikes are training-loss jumps of more than 0.5 nats
over the running median after update 1,000, all between updates 2,900 and 8,100. At
warmup 400 there are fewer but larger ones: fixed Taylor `m = 1` one (+5.9 nats, against
four of up to +3.8 at warmup 2,000) and learned `m = 1` two (+7.4 and +2.7); each
recovers within 50 updates. The learned `m = 1` warmup-2,000 log is on ORCD.

Limits: one seed per cell. Learned nonnegative m = 4 trained on B300 then H100, and
unnormalized learned m = 2 and fixed Taylor m = 4 on H100 then B300, and learned m = 4
with head RMSNorm on B300 then H100 (each resumed from a checkpoint after a requeue); the other Kennel cells each ran on one GPU type. The ORCD cells (normalized
fixed Taylor and Gated DeltaNet) ran on 2×H200, except fixed Taylor m = 1 on H100 and
m = 2 and 16, which moved from H200 to H100 after updates 1,000 and 7,000. Their controls
ran on ORCD H200 (fixed Taylor m = 2 and 4) or on Kennel, so the Gated DeltaNet pair also
differs in hardware.

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
gaps and the Gated DeltaNet reversal into the attention scaling results) or discard the
topic with its runs.

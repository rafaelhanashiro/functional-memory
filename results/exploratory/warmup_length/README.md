# Warmup length

Opened 2026-10-03. The shared recipe warms up for 2,000 of 19,074 updates (10.5%,
1.05B tokens). Pretraining recipes warm up for 0.1–4% of training: GPT-3 375M tokens
(715 updates at our batch), llm.c 700 updates for this exact 10B-token budget,
Pythia and GPT-NeoX 1%, LLaMA 2 and OLMo 2 2,000 updates of 250k–1M. The 2,000 was
inherited from the OFM recipe when `total_steps` shrank to 19,074.

Does a standard warmup shift every mechanism's terminal perplexity by a small constant,
leaving the comparisons in [`results/attention_scaling/`](../../attention_scaling/README.md)
intact, or does it change them? Each warmup-400 run repeats one cell of the heatmap or
degree chart (at `d_K = 64` unless its section says otherwise) with its reference
initialization and changes only
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
| Normalized, fixed Hermite m = 1 | fail @374 | fail @192 | — |
| Normalized, fixed Hermite m = 2 | fail @176 | fail @2,875 | — |
| Normalized, fixed Hermite m = 3 | fail @549 | fail @254 | — |
| Normalized, fixed Hermite m = 4 | fail @680 | fail @1,122 | — |
| Normalized, learned m = 1 | fail @60 | fail @73 | — |
| Normalized, learned m = 2 | fail @1,923 | fail @937 | — |
| Normalized, learned m = 3 | fail @336 | fail @203 | — |
| Normalized, learned m = 4 | fail @1,143 | fail @1,395 | — |
| Normalized, learned nonnegative m = 2 | 14.666 | 14.771 | +0.11 |
| Normalized, learned nonnegative m = 4 | 13.689 | 13.815 | +0.13 |
| Unnormalized, fixed Taylor m = 1 | 17.903 | 17.930 | +0.03 |
| Unnormalized, fixed Taylor m = 2 | 18.166 | 18.099 | −0.07 |
| Unnormalized, fixed Taylor m = 3 | 15.621 | 15.604 | −0.02 |
| Unnormalized, fixed Taylor m = 4 | 15.888 | 15.862 | −0.03 |
| Unnormalized, fixed Hermite m = 1 | 18.196 | 18.233 | +0.04 |
| Unnormalized, fixed Hermite m = 2 | 15.673 | 15.624 | −0.05 |
| Unnormalized, fixed Hermite m = 3 | 15.747 | 15.794 | +0.05 |
| Unnormalized, fixed Hermite m = 4 | 15.278 | 15.238 | −0.04 |
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
  0.10–0.25 with the short warmup; unnormalized polynomials, fixed Taylor, fixed Hermite,
  and learned, are unchanged (−0.07 to +0.05); unnormalized exp gains 0.22 and Gated
  DeltaNet 0.11. Within
  normalized fixed Taylor the loss grows with degree, from 0.10 at m = 2 to 0.22 at
  m = 16, the same as softmax.
- **Orders within each line hold.** Unnormalized learned ranks m = 3, 4, 2, 1,
  unnormalized fixed Taylor m = 3, 4, 1, 2, and unnormalized fixed Hermite m = 4, 2, 3, 1
  at both warmups; normalized fixed Taylor ranks m = 16, 8, 4, 2 (m = 16 and 8 tie at warmup 400,
  13.740 against 13.741); softmax stays ahead of softmax without QK-norm (by 0.42, then
  0.45); exp stays between unnormalized learned m = 1 and m = 2.
- **Learned coefficients are not what makes unnormalized polynomials insensitive.** Fixed
  Taylor and fixed Hermite coefficients move as little as learned ones; among unnormalized kernels only exp
  gains from the short warmup.
- **m = 3 still beats m = 4 in both unnormalized polynomial lines.** Learned: by 0.06 at
  warmup 2,000 and 0.11 at warmup 400; fixed Taylor: by 0.27 and 0.26. The inversion is
  not caused by the long warmup. Fixed Hermite does not invert: its m = 4 leads m = 3 by
  0.47, then 0.56.
- **Gated DeltaNet changes places.** At warmup 2,000 it trails softmax by 0.19 and also
  fixed Taylor m = 8 and 16 and learned nonnegative m = 4; at warmup 400 it leads every
  `d_K = 64` cell, softmax by 0.14. The `d_K = 64` softmax-against-Gated DeltaNet
  comparison in the attention scaling results therefore reverses with the warmup, as it
  does at every other width (next section).
- **Distances between families shrink.** Normalized minus unnormalized gaps narrow by
  0.1–0.45: softmax against unnormalized learned m = 3 from 1.14 to 0.90, learned
  nonnegative m = 4 against unnormalized learned m = 4 from 1.03 to 0.93, softmax against
  exp from 2.37 to 1.93. Learned nonnegative m = 4 against softmax narrows from 0.18 to 0.08.
- **Fixed Taylor's gap to softmax narrows at low degree.** m = 2 trails by 1.28 instead
  of 1.40 and m = 4 by 0.20 instead of 0.26; m = 8 and 16 stay within 0.03 of softmax at
  both warmups.
- **Learned nonnegative still leads normalized fixed Taylor.** It is ahead by 0.24 at
  m = 2 (0.25 at warmup 2,000) and by 0.11 at m = 4 (0.09).
- **Normalized kernels that are signed or reach zero still fail.** All ten that fail at
  warmup 2,000 fail at warmup 400: fixed Taylor m = 1 and 3, learned m = 1–4, and fixed
  Hermite m = 1–4. The short warmup moves the failure earlier in five (fixed Taylor m = 3,
  learned m = 2 and 3, Hermite m = 1 and 3) and later in five, and rescues none. Hermite
  m = 2, nonnegative but zero at `s = −1`, lasts longest: update 2,875 against 176, at
  validation perplexity 26.5 at update 2,000 against softmax's 22.1.
- **A failure update is one draw, not a threshold.** Hermite m = 4 failed at update 1,122
  after resuming from its update-1,000 checkpoint on H100; two earlier H100 executions,
  one from update 0 and one from the same checkpoint, passed 1,122 and were preempted at
  1,283 and 1,425.

- **Output scaling decides whether unnormalized attention gains from the short warmup.**
  With 1/t, exp and learned m = 4 both gain about 0.25, as plain exp does (0.22) while
  plain learned m = 4 does not (+0.02). With head RMSNorm both are unchanged (−0.03,
  −0.02). Softmax with head RMSNorm equals exp with head RMSNorm (RMSNorm cancels the
  row sum), so it has no separate cell.

- **Unnormalized fixed Hermite moves like the other unnormalized polynomials.** Its
  differences are +0.04, −0.05, +0.05, and −0.04 at m = 1–4, within the fixed Taylor
  (−0.07 to +0.03) and learned (−0.05 to +0.02) ranges and unlike unnormalized exp
  (−0.22). Its comparisons hold: fixed Taylor stays ahead at odd degrees (m = 1 by 0.29,
  then 0.30; m = 3 by 0.13, then 0.19), Hermite at even degrees (m = 2 by 2.49, then 2.48;
  m = 4 by 0.61, then 0.62), and learned leads Hermite at every degree.

Loss spikes: unnormalized `m = 1`, fixed Taylor and learned, spikes at both warmups and
recovers; no `m ≥ 2` run spikes. Spikes are training-loss jumps of more than 0.5 nats
over the running median after update 1,000, all between updates 2,900 and 8,100. At
warmup 400 there are fewer but larger ones: fixed Taylor `m = 1` one (+5.9 nats, against
four of up to +3.8 at warmup 2,000) and learned `m = 1` two (+7.4 and +2.7); each
recovers within 50 updates. The learned `m = 1` warmup-2,000 log is on ORCD.

Limits: one seed per cell. Learned nonnegative m = 4 trained on B300 then H100, and
unnormalized learned m = 2 and fixed Taylor m = 4 on H100 then B300, and learned m = 4
with head RMSNorm on B300 then H100 (each resumed from a checkpoint after a requeue); the other Kennel cells each ran on one GPU type. Of the normalized learned and fixed Hermite cells, learned m = 1 and 2 trained on four H100s, learned m = 3 on four B300s, learned m = 4 and Hermite m = 1 on two B300s, and Hermite m = 2–4 on two H100 PCIe; two-GPU runs keep the global batch. The ORCD cells (normalized
fixed Taylor and Gated DeltaNet) ran on 2×H200, except fixed Taylor m = 1 on H100 and
m = 2 and 16, which moved from H200 to H100 after updates 1,000 and 7,000. Their controls
ran on ORCD H200 (fixed Taylor m = 2 and 4) or on Kennel, so the Gated DeltaNet pair also
differs in hardware. The ORCD unnormalized fixed Hermite cells, whose controls ran on
Kennel, trained on H100 (m = 1), on H200 then H100 after update 1,000 (m = 2), on H100
with updates 12,501–13,000 on H200 (m = 3), and on H200 then H100 NVL after update 3,000
(m = 4).

## Softmax, Gated DeltaNet, and normalized fixed Taylor across d_K

Terminal validation perplexity of the heatmap's softmax (QK-norm, transformer reference
initialization) and Gated DeltaNet (its reference initialization) cells at every width;
every run's validation best is terminal. The `d_K = 64` pair is the one above.

| `d_K` | Softmax, 2,000 | Softmax, 400 | Difference | Gated DeltaNet, 2,000 | Gated DeltaNet, 400 | Difference | Gated DeltaNet − softmax, 2,000 → 400 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 15.639 | 15.732 | +0.09 | 14.847 | 14.816 | −0.03 | −0.79 → −0.92 |
| 16 | 14.335 | 14.470 | +0.13 | 14.290 | 14.217 | −0.07 | −0.04 → −0.25 |
| 32 | 13.762 | 14.028 | +0.27 | 13.952 | 13.908 | −0.04 | +0.19 → −0.12 |
| 64 | 13.514 | 13.731 | +0.22 | 13.705 | 13.594 | −0.11 | +0.19 → −0.14 |
| 128 | 13.303 | 13.520 | +0.22 | 13.380 | 13.327 | −0.05 | +0.08 → −0.19 |

- **Gated DeltaNet leads softmax at every width at warmup 400.** At warmup 2,000 softmax
  leads at `d_K = 32`, 64, and 128 (by 0.19, 0.19, and 0.08); at warmup 400 Gated
  DeltaNet leads there by 0.12, 0.14, and 0.19, and its lead at `d_K = 8` and 16 grows
  (0.79 to 0.92, 0.04 to 0.25). The attention scaling results' "softmax is lower at 32,
  64, and 128" reverses at all three widths.
- **The two mechanisms move in opposite directions at every width.** Softmax loses
  0.09–0.27 with the short warmup, least at `d_K = 8`; Gated DeltaNet gains 0.03–0.11.
- **Both still improve with every doubling of `d_K`** at both warmups.

Limits: one seed per cell; the controls ran on Kennel. These runs record the training
telemetry. Softmax trained on H100 NVL (`d_K = 8`), on H200 then H100 after update
15,000 (16), on H100 (32), and on H100 with updates 7,001–14,500 on H200 (128); Gated
DeltaNet on H200 (8 and 128) and H100 (16 and 32).

Normalized fixed Taylor (QK-norm, transformer reference initialization) at `m = 2`, 4,
8, and 16, terminal validation perplexity; every run's validation best is terminal. The
`d_K = 64` rows are the ones above.

| `d_K` | m = 2, 2,000 | m = 2, 400 | Difference | m = 4, 2,000 | m = 4, 400 | Difference | m = 8, 2,000 | m = 8, 400 | Difference | m = 16, 2,000 | m = 16, 400 | Difference |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 19.491 | 19.622 | +0.13 | 17.063 | 17.148 | +0.08 | 16.081 | 16.170 | +0.09 | 15.749 | 15.835 | +0.09 |
| 16 | 17.154 | 17.279 | +0.13 | 15.212 | 15.369 | +0.16 | 14.563 | 14.717 | +0.15 | 14.357 | 14.529 | +0.17 |
| 32 | 15.794 | 15.895 | +0.10 | 14.278 | 14.482 | +0.20 | 13.854 | 14.119 | +0.27 | 13.775 | 14.028 | +0.25 |
| 64 | 14.911 | 15.014 | +0.10 | 13.774 | 13.927 | +0.15 | 13.543 | 13.741 | +0.20 | 13.519 | 13.740 | +0.22 |
| 128 | 14.329 | 14.434 | +0.11 | 13.455 | 13.611 | +0.16 | 13.288 | 13.495 | +0.21 | 13.307 | 13.517 | +0.21 |

- **At m = 8 and 16 fixed Taylor's loss follows softmax's across `d_K`.** The short
  warmup costs least at `d_K = 8` (0.09, as for softmax), more at 16 (0.15 and 0.17
  against 0.13), most at 32 (0.27 and 0.25 against 0.27), and 0.20–0.22 at 64 and 128.
  At m = 16 the shift is within 0.02 of softmax's at every width except `d_K = 16`
  (0.04). m = 4 has the same shape, least at `d_K = 8` (0.08) and most at 32 (0.20), but
  loses only 0.15–0.16 at 16, 64, and 128, less than softmax's 0.22 at 64 and 128.
- **At `d_K = 128` the shift matches `d_K = 64`; at `d_K = 32` it is larger from m = 4
  up.** `d_K = 128` loses 0.11, 0.16, 0.21, and 0.21 at m = 2, 4, 8, and 16 against 0.10,
  0.15, 0.20, and 0.22 at `d_K = 64`; `d_K = 32` loses 0.10, 0.20, 0.27, and 0.25.
- **m = 2 loses about 0.1 at every width:** 0.13, 0.13, 0.10, 0.10, and 0.11 at `d_K =
  8`, 16, 32, 64, and 128, with no peak at `d_K = 32`. From `d_K = 16` up it loses least
  of the four degrees, and m = 4 and 8 lose 0.15–0.27. At `d_K = 8` it loses most, so the
  shift does not grow from m = 2 to m = 8 there: 0.13, 0.08, and 0.09 at m = 2, 4, and 8.
- **Orders hold.** Within each width the degrees keep their order at both warmups
  (`d_K = 8`, 16, and 32: m = 16, 8, 4, 2; 128: m = 8, 16, 4, 2). From `d_K = 32`
  up, m = 16 stays within 0.02 of softmax at both warmups (warmup 400: 14.028 against
  14.028 at 32, 13.517 against 13.520 at 128); at `d_K = 8` it trails softmax by 0.11
  and 0.10, and at 16 by 0.02 and 0.06.
- **ORCD and Kennel replicates agree to about 0.01.** With the same recipe and seed
  (m = 2, warmup 400), ORCD and Kennel give 17.267 against 17.279, 15.883 against 15.895,
  and 14.435 against 14.434 at `d_K = 16`, 32, and 128. Hardware nondeterminism
  therefore moves terminal perplexity by about 0.01, well below the warmup shifts.

Limits: one seed per cell. The warmup-400 runs trained on Kennel with the training
telemetry: four B300s (`d_K = 32`, m = 4; `d_K = 128`, m = 2, 4, and 16), four H100s
(`d_K = 16` and 32, m = 2), two B300s
(`d_K = 128`, m = 8), four H100 PCIe (`d_K = 16`, m = 16), two H100s (`d_K = 8`, m = 8
and 16; `d_K = 16`, m = 8; `d_K = 32`, m = 16), and two H100 PCIe then two B300s after
update 16,500 (`d_K = 32`, m = 8). The warmup-400 runs on ORCD trained on two H100s
(`d_K = 8`, m = 2 and 4; `d_K = 16`, m = 4); the ORCD replicates of m = 2 trained on two
H200s (`d_K = 32` and 128) and on two H200s then two H100s after update 11,500
(`d_K = 16`). Their controls ran on ORCD's two H200s (m = 2 and 4) or on Kennel's four
H100s (`d_K = 128`, m = 8, and `d_K = 16`, m = 16: B300 and H100).

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

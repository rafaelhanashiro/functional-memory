# Even polynomial powers

Opened: 2026-11-03. Do even-only degree-eight polynomials improve static or
query-conditioned attention at the 1× Chinchilla budget?

The two even-only Transformer++ runs use orders `R = {0, 2, 4, 6, 8}`:

- Static learns `p(z) = sum_{r in R} a_r*z^r/r!`, independently per layer/head.
- Query + weight decay learns
  `p_t(z) = sum_{r in R} (b_r + a_r*tanh(w_r^T*x_t))*z^r/r!`.
  The five projection columns receive weight decay 0.1; coefficient offsets and
  amplitudes remain exempt, matching the query controls. The input `x_t` is the
  current token's activation after the block input RMSNorm.

Only five coefficient channels are stored; every odd coefficient remains exactly
zero. Both initialize to the even part of the degree-eight Taylor polynomial.
Coefficients are otherwise unconstrained, so even-only does not imply nonnegative.
All model parameters train from scratch.

The runs share seed 42, global batch 256, context 2,048, canonical QK RMSNorm with
per-head score scaling, RoPE, unnormalized attention with head-output RMSNorm,
and factorial coefficient scaling. The **1× schedule is 7,570 updates,
455 warmup updates, and 3,968,860,160 prediction tokens**. The LR schedule,
validation cadence, data order and other settings match each family's controls;
see the [study recipe](../../../README.md).

A full degree-eight static control, `r9c69dd7cfc8d`, is pending. It matches the
even-only static recipe `r45831d718944` except for
`model.attention_coefficient_orders: null`: all nine orders from zero through eight
are learned, initialized to the full Taylor polynomial. The training budget and
all other scientific settings are unchanged.

Both even-only runs completed all 7,570 updates without numerical failure. The table reports
terminal validation perplexity separately from validation-best perplexity.

| Family | Role | Run | Terminal validation PPL | Best validation PPL |
| --- | --- | --- | ---: | ---: |
| Static | Even orders through eight | `r45831d718944` | 14.620 | 14.616 |
| Static | Full degree-four control | `r4339a245820e` | 14.773 | 14.767 |
| Static | Full degree-eight control (pending) | `r9c69dd7cfc8d` | — | — |
| Query + weight decay | Even orders through eight | `r698f9a157a94` | 14.618 | 14.615 |
| Query + weight decay | Full degree-four control | `rca97cc9db8f2` | 14.714 | 14.709 |
| Query + weight decay | Full degree-eight control | `r25fffc16c2f3` | 14.328 | 14.323 |

The [selection](selection.yaml) pins the comparisons explicitly. Full degree four
controls for five coefficient channels; full degree eight controls for maximum
degree. The static comparison at equal maximum degree awaits its pending control. Removing all
odd terms changes both the available functions and initialization; this does not
isolate whether an odd highest power alone causes a performance difference.
One seed cannot establish the size of run-to-run variation.

Static even-only improves terminal PPL by **1.03%** over full degree four.
Query even-only improves it by **0.65%** over full degree four, but is **2.02% worse**
than full degree eight. The two even-only models finish effectively tied
(14.620 versus 14.618); this run provides no meaningful numerical advantage for
query conditioning over static coefficients. Both even-only validation-best checkpoints
are at update 7,500 (14.616102 and 14.615046).

The evidence favors even-only over full degree four at equal coefficient-channel
count, but does not favor discarding odd terms from full degree eight for query
coefficients; the full degree-eight static comparison is pending. The open
decision is whether either even-only parameterization merits further study;
these single-seed results remain exploratory.

## Regeneration

The [summary](summary.csv) reads only the explicit [selection](selection.yaml),
checks its scientific-input pins, and distinguishes terminal and best metrics.
Incomplete runs and numerical failures retain their status with blank terminal and
best metrics until training completes.
From the repository root:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python \
  experiments/transformer_pp/results/exploratory/even_powers/summarize.py
```

## Execution

Home cluster: Kennel. The even-only runs and full degree-eight static control use
the existing degree-eight Mosaic kernel, one GPU each, with no scheduling
dependencies. See the [study launch instructions](../../../README.md#even-polynomial-powers)
for the exact commands.

Acceptance checks retained orders, gradients, kernel correctness, backbone
initialization, production-shaped throughput and checkpoint resume before
training. Actual submissions and allocations live in each run's record.

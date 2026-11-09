# Even polynomial powers

Opened: 2026-11-03. Do even-only degree-eight polynomials improve static or
query-conditioned attention at the 1× Chinchilla budget? A third run asks the
mirror question for query coefficients: keep the intercept and only the odd orders
through seven ([below](#intercept-and-odd-orders)).

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

A full degree-eight static control, `r9c69dd7cfc8d`, matches the
even-only static recipe `r45831d718944` except for
`model.attention_coefficient_orders: null`: all nine orders from zero through eight
are learned, initialized to the full Taylor polynomial. The training budget and
all other scientific settings are unchanged.

All three runs completed all 7,570 updates without numerical failure. The table reports
terminal validation perplexity separately from validation-best perplexity.

| Family | Role | Run | Terminal validation PPL | Best validation PPL |
| --- | --- | --- | ---: | ---: |
| Static | Even orders through eight | `r45831d718944` | 14.620 | 14.616 |
| Static | Full degree-four control | `r4339a245820e` | 14.773 | 14.767 |
| Static | Full degree-eight control | `r9c69dd7cfc8d` | 14.299 | 14.294 |
| Query + weight decay | Even orders through eight | `r698f9a157a94` | 14.618 | 14.615 |
| Query + weight decay | Full degree-four control | `rca97cc9db8f2` | 14.714 | 14.709 |
| Query + weight decay | Full degree-eight control | `r25fffc16c2f3` | 14.328 | 14.323 |

The [selection](selection.yaml) pins the comparisons explicitly. Full degree four
controls for five coefficient channels; full degree eight controls for maximum
degree. Removing all
odd terms changes both the available functions and initialization; this does not
isolate whether an odd highest power alone causes a performance difference.
One seed cannot establish the size of run-to-run variation.

Static even-only improves terminal PPL by **1.03%** over full degree four, but is
**2.25% worse** than full degree eight (14.620 against 14.299).
Query even-only improves it by **0.65%** over full degree four, but is **2.02% worse**
than full degree eight. The two even-only models finish effectively tied
(14.620 versus 14.618); this run provides no meaningful numerical advantage for
query conditioning over static coefficients. Both even-only validation-best checkpoints
are at update 7,500 (14.616102 and 14.615046).

The evidence favors even-only over full degree four at equal coefficient-channel
count, but does not favor discarding odd terms from full degree eight: both even-only
models trail their full degree-eight controls by about 2% (static 2.25%, query 2.02%). The open
decision is whether either even-only parameterization merits further study;
these single-seed results remain exploratory.

## Intercept and odd orders

The query + weight decay run `r0bdac8e1b97b` (`query-odd-m7-w-decay`) keeps orders
`R = {0, 1, 3, 5, 7}` and is otherwise the even-only query run's recipe: the same
`p_t(z)` with five coefficient channels and weight decay on the five projection
columns, and orders 2, 4 and 6 fixed at exactly zero. It initializes to
`1 + z + z^3/3! + z^5/5! + z^7/7!`, the intercept plus the odd part of the Taylor
polynomial. That polynomial increases monotonically, like `exp`, and turns negative
for scores below about -0.88, whereas an even-only polynomial gives a score and its
negation the same weight. Coefficients are otherwise unconstrained, so the learned
polynomial need not stay monotonic. The run uses the even-only runs' source code.

Its comparisons are the even-only query run (five channels), the full degree-four
query control (five channels), the full degree-seven query control `rdb67453630c8`
(same highest power, also on an ORCD H100), and the full degree-eight control.
The run is in progress; this topic reports it when it completes.

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

The intercept-and-odd run `r0bdac8e1b97b` is hosted on ORCD and uses the existing
degree-seven Mosaic kernel on one H200 (Hopper, sm90, the H100's kernels), which Slurm
expected to start days before an H100. From its worktree after committing the frozen
execution inputs:

```bash
RUN=r0bdac8e1b97b
sbatch --parsable --job-name="$RUN" --output="experiments/transformer_pp/runs/$RUN/training.log" \
  --gres=gpu:h200:1 \
  scaffolds/transformer/launchers/orcd.sbatch experiments.transformer_pp.acceptance \
  "experiments/transformer_pp/runs/$RUN/config.yaml" "$(git rev-parse HEAD)" 1 H200
```

Acceptance checks retained orders, gradients, kernel correctness, backbone
initialization, production-shaped throughput and checkpoint resume before
training; its order tests cover both the even and the intercept-and-odd sets.
Actual submissions and allocations live in each run's record.

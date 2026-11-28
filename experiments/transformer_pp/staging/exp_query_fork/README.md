# Forking a trained exp model into exp and a Taylor-initialized query polynomial

Opened 2026-11-27. Does a trained exp model gain from switching to a Taylor-initialized query polynomial when both
branches get the same next 7,570 batches? Two runs fork the completed unnormalized exp run `r19dfa87cb7ed` (the
15-layer width-896 shape, `d_K = d_V = 64`, seed 42, twice the recipe's rate) at its terminal update, 7,570. One
keeps exp and every parameter. The other takes `r90776a5d4175`'s attention, the degree-eight query polynomial with
direct tanh coefficients on raw scores and W decay, unnormalized with per-head RMSNorm: its new coefficients start at
that recipe's initial draw, so every query starts at `T_8(s) = sum_{r<=8} s^r / r!`, and every shared parameter is the
parent's. Both restart AdamW and the parent's schedule (455 warmup updates to 2e-3, cosine to 2e-4 over 7,570
updates) and read the parent's packed stream from its next sequence, 3,968,860,160 fresh prediction tokens per
branch ([methods and launch](../../README.md#forks-of-the-exponential-at-its-terminal-update)). Home cluster: ORCD.

## Runs

Validation perplexity on the parent's full validation set (19 x 256 sequences). Update 0 is each branch's evaluation
before its first update. Each branch's best is its last update, 7,570.

| Branch | Run | Update 0 | Best and terminal, update 7,570 |
| --- | --- | ---: | ---: |
| exp -> exp | `rda011906861b` | 14.2151 | **13.1981** |
| exp -> query, m = 8 | `r68f1cbfed2ef` | 37.8561 | 13.2044 |
| Query − exp | | +23.6410 | +0.0063 (+0.05%) |

At matched updates:

| Update | exp -> exp | exp -> query | Query − exp |
| ---: | ---: | ---: | ---: |
| 0 | 14.2151 | 37.8561 | +23.6410 |
| 100 | 14.3552 | 14.5112 | +0.1560 |
| 200 | 14.9551 | 15.0317 | +0.0766 |
| 500 | 17.0271 | 16.9509 | −0.0762 |
| 800 | 16.9077 | 16.9564 | +0.0488 |
| 1,000 | 16.8378 | 16.8703 | +0.0326 |
| 2,000 | 16.1031 | 16.1223 | +0.0192 |
| 3,000 | 15.4298 | 15.4364 | +0.0066 |
| 4,000 | 14.8346 | 14.8359 | +0.0014 |
| 5,000 | 14.1371 | 14.1463 | +0.0091 |
| 6,000 | 13.6262 | 13.6339 | +0.0076 |
| 7,000 | 13.3260 | 13.3319 | +0.0059 |
| 7,570 | 13.1981 | 13.2044 | +0.0063 |

## Findings

- **Switching gains nothing.** The query branch ends 0.0063 perplexity (0.05%) behind the exp branch, 13.2044 against
  13.1981, far inside the 0.6–0.7% seed spread measured at `d_K` 64. It is behind at 27 of the 28 evaluations after
  the fork (ahead once, by 0.076 at update 500), by 0.04–0.06% from update 5,000 on.
- **The switch cost is large and brief.** Replacing exp with `T_8` on the exp-trained weights raises validation
  perplexity from 14.2151 to 37.8561 at the fork (+23.64, +0.979 in NLL). After 100 updates the query branch is 1.1%
  behind the exp branch, and from update 600 on at most 0.3%.
- **Both branches gain from a second budget of fresh tokens, after a rise.** Restarting the schedule first raises
  perplexity, to 17.03 at update 500 for exp and 16.96 at update 800 for the query branch. Both return below the
  fork's 14.2151 at update 5,000 and end 7.1% below the parent's best (14.2117).
- **From scratch the same pair is far apart.** At this rate the query recipe `r90776a5d4175` reaches a validation best
  of 14.742 against the parent's 14.212 (+3.7%), both at update 7,500; they end at 14.748 against 14.215 (+3.8%).
  Started from the exp-trained weights it ends within 0.05% of exp.
- **The query branch has two gradient spikes, the exp branch none:** at update 80 (pre-clip norm 5.26) and update
  3,215 (1.31). Its first update's norm is 5.97.

## Limits

- One pair at one seed. The pair shares its starting weights and batches, which no seed spread measures; the terminal
  difference is more than ten times smaller than the seed spread of independent runs.
- The parent's terminal state is rebuilt, not the original: retention kept only its update-7,500 checkpoint, and its
  last 70 updates were replayed once from it. The replay reproduces the recorded terminal validation within 2.4e-6 in
  NLL; 32 of the 2,130 QK RMSNorm gains and head scales the parent recorded at update 7,570 differ by more than 1e-4
  (at most 7.1e-4). Both branches start from the same rebuilt state, and the exp branch's update-0 evaluation equals
  the rebuilt checkpoint's.
- One restart design: a fresh AdamW and a full warmup to twice the recipe's rate. A continuation at a small constant
  rate, or one keeping the parent's Adam history, is not measured here.
- One polynomial: degree eight, direct tanh coefficients, no SoS, softplus or prefactor.
- No run trains exp for the same tokens on one schedule at this width. Interpolating NLL linearly in log tokens
  between exp's validation best at the recipe's rate for one budget (`r6a76e7fbde45`, 14.257) and for 2.5 budgets
  (`rfe12e5c36e94`, 12.449) gives about 12.87 at two budgets, 0.33 below the exp branch; the rates differ, so this is
  an estimate.
- Each branch trains on its own kernel (fused Triton exp, Mosaic query polynomial) on an H100 80GB HBM3.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/staging/exp_query_fork
```

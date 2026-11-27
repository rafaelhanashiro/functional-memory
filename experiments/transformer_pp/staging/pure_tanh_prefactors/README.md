# Pure tanh query coefficients with softplus and exp prefactors at Q/K width 4, and at width 64 (Transformer++)

Opened 2026-11-22.

**Question.** At `d_K` 4, how close to exp does a signed query polynomial with bounded coefficients come, at degrees
8 and 16, alone and under a softplus or exp prefactor of the raw score? Six runs train `P_t(s)`,
`softplus(s) P_t(s)` and `e^s P_t(s)` with

`P_t(s) = sum_r tanh(w_r^T x_hat_t + beta_r) / r! s^r`,

every power 0..m, `x_hat_t` the raw attention-block input at unit L2 norm, `w_r` starting at zero and `beta_r` at
atanh(1/2). The [study methods](../../README.md#pure-tanh-coefficients-with-softplus-and-exp-prefactors-at-key-width-four)
define the recipe and launch commands. Home cluster: Kennel.

A seventh run gives the same generator to the SoS kernel that ties exp, `z h_t(z)` with `z = softplus(s)` and a
degree-sixteen core: each square factor is `f_j tanh(w_j^T x_hat_t + beta_j)`, a fraction of its Taylor factor
`f_j`, in place of the affine `f_j (1 + b_j + a_j tanh(W_j^T x_t)) / 2`
([methods](../../README.md#pure-tanh-square-factors-for-z-h_tz-at-key-width-four)).

At `d_K` 64, three runs ask whether the pure tanh generator still beats the affine one: `P_t(s)` at degrees 8 and
16, and the affine degree-sixteen query polynomial `p_t(s)`, `(b_r + a_r tanh(w_r^T x_t)) / r!`, as its control,
which `d_K` 64 did not have. The degree-eight control is the affine run `r25fffc16c2f3`. All three take the affine
partner's recipe, with weight decay 0.1 on `W`
([methods](../../README.md#pure-tanh-coefficients-at-key-width-64)). Home cluster: ORCD.

All runs use the 15-layer width-896 shape, RoPE, QK RMSNorm with head scales, per-head output RMSNorm with the
max-abs prescale, seed 42, 7,570 updates and the recipe's learning rate (peak 1e-3), on the row-power shifted fused
kernel.

## Runs

Validation perplexity, best with terminal in parentheses; exp `r2739c80cc032` reaches 19.190 (19.201). The lead is
the gap to exp at the same update; the last column is the largest share of updates clipped in one validation window
after update 500, with the largest pre-clip gradient norm.

| Run | Weight | `m` | Best (terminal) | vs exp | vs exp at update 1,000 | at 3,000 | Clipped, largest norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rfd18b16513a0` | `P_t(s)` | 8 | 19.870 (19.877) | +3.54% | −14.8% | +2.5% | 4.8%, 31 |
| `rfd16ec25202a` | `P_t(s)` | 16 | 19.556 (19.558) | +1.91% | −15.3% | +0.8% | 6.2%, 162 |
| `rfa8c0b2f748f` | `softplus(s) P_t(s)` | 8 | 20.020 (20.031) | +4.33% | −12.5% | +3.0% | 3.0%, 3.4 |
| `r1a0e7b80c588` | `softplus(s) P_t(s)` | 16 | 19.815 (19.821) | +3.25% | −11.3% | +2.1% | 2.0%, 4.6 |
| `r6621b4589a3b` | `e^s P_t(s)` | 8 | 19.664 (19.673) | +2.47% | −15.0% | +0.9% | 0.2%, 5.7 |
| `rf51ad9acc005` | `e^s P_t(s)` | 16 | 19.483 (19.488) | +1.52% | −16.8% | −0.0% | 1.6%, 22 |
| `r70e98e1944d5` | `z h_t(z)`, pure tanh square factors | core 16 | 19.385 (19.388) | +1.02% | −14.1% | −0.1% | 4.0%, 5.4 |

References at `d_K` 4, on the same recipe and learning rate:

| Reference | Run | `m` | Best (terminal) | vs exp |
| --- | --- | ---: | ---: | ---: |
| Affine tanh query polynomial, `(b_r + a_r tanh(W_r^T x_t)) / r!` | `rd5e3ec5a4f02` | 8 | 19.925 (19.939) | +3.83% |
| Affine tanh query polynomial | `raec1c513a51b` | 16 | 19.777 (19.779) | +3.06% |
| Fixed Taylor `T_16(s)` | `rca649583af62` | 16 | 19.829 (19.835) | +3.33% |
| `z h_t(z)`, `z = softplus(s)`, SoS core with affine tanh factors | `r2e7f5617fdf2` | 16 | 19.192 (19.197) | +0.01% |

## Runs at `d_K` 64

Validation perplexity, best with terminal in parentheses; exp `r6a76e7fbde45` reaches 14.257 (14.262). The lead is
the gap to exp at the same update. Spikes counts gradient spikes over the whole run; the control's trainer predates
the counter (–). The last column is the largest share of updates clipped in one validation window after update 500,
with the largest pre-clip gradient norm. `rbb09d696a5b0` also fills the degree-sixteen query-coefficients cell of the
placed [sum of squares table](../../results/pretraining/parametrizations/sos/sum_of_squares/README.md#findings).

| Run | Weight | `m` | Best (terminal) | vs exp | vs exp at update 1,000 | at 3,000 | Spikes | Clipped, largest norm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `re0af7abe427b` | `P_t(s)`, pure tanh | 8 | 14.543 (14.550) | +2.00% | −1.0% | +1.9% | 25 | 2%, 5.9 |
| `rcb74ec29ab5f` | `P_t(s)`, pure tanh | 16 | 14.320 (14.326) | +0.44% | −1.5% | +0.3% | 14 | 3%, 5.9 |
| `rbb09d696a5b0` | `p_t(s)`, affine tanh | 16 | 14.286 (14.290) | +0.20% | −1.8% | +0.0% | 5 | 0.2%, 3.9 |
| `r25fffc16c2f3` | `p_t(s)`, affine tanh (control) | 8 | 14.323 (14.328) | +0.46% | −0.9% | +0.4% | – | 1%, 3.8 |

## Findings

- **None reaches exp.** The closest are the SoS core with pure tanh factors, 1.02% above, and `e^s P_t` at degree
  sixteen, 1.52% above. Every run leads exp by 11–17% at update 1,000 and is level with it or behind by update
  3,000.
- **Pure tanh square factors do not help the SoS core:** 19.385 against 19.192 with the affine factors (+1.01%).
  The run leads the affine one by 8.8% at update 1,000 and trails it from update 3,000, by 1.0% from update 5,000.
- **Degree sixteen beats eight for each weight:** by 1.6% without a prefactor, 1.0% under softplus and 0.9% under
  exp.
- **The exp prefactor helps and the softplus prefactor hurts.** Against no prefactor, exp is 1.0% lower at degree
  eight and 0.4% at sixteen; softplus is 0.8% and 1.3% higher.
- **Bounded coefficients train without sustained clipping.** The worst window has 6.2% of its updates clipped
  (`P_t`, degree sixteen); the prefactor runs stay at or below 3.0%.
- **The pure tanh generator ties the affine one at degree eight and beats it at sixteen:** 19.870 against 19.925
  (−0.28%) and 19.556 against 19.777 (−1.1%). The affine run at degree sixteen had 14–42% of its updates clipped
  in the validation windows between updates 2,000 and 3,500 and at most 1.6% per window after update 5,000
  (largest norm 153).
- **Learned coefficients beat fixed Taylor at degree sixteen** without a prefactor (19.556 against 19.829) and
  under exp.
- **The softplus-prefactor SoS kernel stays ahead of every signed polynomial here:** with affine factors it ties
  exp at core degree sixteen, 1.5% below the best of them, and with pure tanh factors it is 0.5% below.
- **At `d_K` 64 pure tanh does not beat the affine generator:** 14.543 against 14.323 at degree eight (+1.53%) and
  14.320 against 14.286 at sixteen (+0.23%). At `d_K` 4 it was 0.28% and 1.1% ahead.
- **At `d_K` 64 the affine generator is the stable one at degree sixteen:** 5 gradient spikes and at most 0.2% of a
  window's updates clipped after update 500, against 14 and 3% for pure tanh. At degree eight pure tanh has 25 spikes
  and up to 2% clipped (largest norm 5.9), against 1% and 3.8 for the control. At `d_K` 4 it was the affine
  degree-sixteen run that clipped 14–42% of its updates in some windows.
- **Among this topic's `d_K` 64 runs the affine degree-sixteen query polynomial is the best:** 14.286, 0.25% below
  degree eight and 0.20% above exp.

## Limits

- One seed per run, at one learning rate.
- The fixed Taylor run trained on a B300 to update 1,400 and on a B200 after. Between updates 1,750 and 2,100 19%
  of its updates were clipped (largest norm 44), and no other window had more than 0.2%. A B300 execution of the
  recipe had no clipped update through update 5,000 and was 2.4% above exp there, 1.1% below this run, so 19.829
  is probably that much too high.
- Hardware: the degree-sixteen runs trained on a B300. Each degree-eight run trained on a B300 and a B200 across
  preemptions (`rfa8c0b2f748f` mostly on the B200). The SoS run with pure tanh factors trained on a B200.
- The prefactor kernel evaluates each key in units of its own bound and takes 1.4–2.3 times the plain kernel's time
  per update: 3.3 and 5.0 s (exp, degrees 8 and 16) and 5.1 and 6.9 s (softplus) against 2.4 and 3.0 s.
- At `d_K` 64 the three runs trained on ORCD H200s and resumed from checkpoints across preemptions (3–4 attempts
  each). The degree-eight control `r25fffc16c2f3` trained on a Kennel B300 on the Mosaic kernel, with head RMSNorm's
  plain arithmetic (the same function), so the degree-eight comparison also changes the hardware and the kernel.

## Open question

Keep the topic, or delete it with its runs and their artifacts.

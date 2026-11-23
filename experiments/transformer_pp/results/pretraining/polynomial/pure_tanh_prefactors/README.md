# Pure tanh query coefficients with softplus and exp prefactors at Q/K width 4 (Transformer++)

Opened 2026-11-22.

**Question.** At `d_K` 4, how close to exp does a signed query polynomial with bounded coefficients come, at degrees
8 and 16, alone and under a softplus or exp prefactor of the raw score? Six runs train `P_t(s)`,
`softplus(s) P_t(s)` and `e^s P_t(s)` with

`P_t(s) = sum_r tanh(w_r^T x_hat_t + beta_r) / r! s^r`,

every power 0..m, `x_hat_t` the raw attention-block input at unit L2 norm, `w_r` starting at zero and `beta_r` at
atanh(1/2). The [study methods](../../../../README.md#pure-tanh-coefficients-with-softplus-and-exp-prefactors-at-key-width-four)
define the recipe and launch commands. Home cluster: Kennel.

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

References at `d_K` 4: the affine tanh query polynomial at degree eight, `b_r + a_r tanh(W_r^T x_t)` over `r!`
(`rd5e3ec5a4f02`), 19.925 (+3.83%, largest norm 62), and `z h_t(z)`, `z = softplus(s)`, with a degree-sixteen SoS
core (`r2e7f5617fdf2`), 19.192 (+0.01%).

## Findings

- **None reaches exp.** The closest is `e^s P_t` at degree sixteen, 1.52% above. Every run leads exp by 11–17% at
  update 1,000 and is level with it or behind by update 3,000.
- **Degree sixteen beats eight for each weight:** by 1.6% without a prefactor, 1.0% under softplus and 0.9% under
  exp.
- **The exp prefactor helps and the softplus prefactor hurts.** Against no prefactor, exp is 1.0% lower at degree
  eight and 0.4% at sixteen; softplus is 0.8% and 1.3% higher.
- **Bounded coefficients train without sustained clipping.** The worst window has 6.2% of its updates clipped
  (`P_t`, degree sixteen); the prefactor runs stay at or below 3.0%.
- **At degree eight the pure tanh generator ties the affine one:** 19.870 against 19.925 (−0.28%).
- **The softplus-prefactor SoS kernel stays ahead of every signed polynomial here:** it ties exp at core degree
  sixteen, 1.5% below the best of these.

## Limits

- One seed per run, at one learning rate.
- Hardware: the degree-sixteen runs trained on a B300. Each degree-eight run trained on a B300 and a B200 across
  preemptions (`rfa8c0b2f748f` mostly on the B200).
- The prefactor kernel evaluates each key in units of its own bound and takes 1.4–2.3 times the plain kernel's time
  per update: 3.3 and 5.0 s (exp, degrees 8 and 16) and 5.1 and 6.9 s (softplus) against 2.4 and 3.0 s.

## Open question

Keep the topic, or delete it with its runs and their artifacts.

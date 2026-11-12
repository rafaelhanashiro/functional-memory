# Per-degree score biases

Opened: 2026-11-11. Status: **both trials complete; neither beat the control**, so no reference-shape runs
follow. Kept as the record of this check.

Does giving every order `r = 1..8` of the degree-eight tanh query polynomial its own signed shift of the
score, `(s_ti + beta_r)^r` in place of `s_ti^r`, improve validation perplexity? Two trials at the 12-layer
width-640 shape shift each order by a constant (static) or by an amount that depends on the query token
(query).

## Runs

| Role | Run | Label | Shift | Home cluster |
| --- | --- | --- | --- | --- |
| Trial | `r8dc6e147601c` | `query-m8-w-decay-l12-d640-score-bias-query` | query | Kennel |
| Trial | `rb4e5457ef2a0` | `query-m8-w-decay-l12-d640-score-bias-static` | static | Kennel |
| Control | `r95a5691d293f` | `query-m8-w-decay-l12-d640` | – | Kennel |

The [selection](selection.yaml) names each role and pins the runs' scientific files.

## Method

With `s_ti = Q_t K_i^T / sqrt(d_K)`, each trial computes

    W(t,i) = c_0(x_t) + sum_(r=1..8) c_r(x_t) (s_ti + beta_r(t,i))^r

with the control's tanh query coefficients `c_r(x_t)` (factorial scales unchanged) and `x_t` the block's
attention input after its RMSNorm. Per layer, head and order: static `beta_r = b_r`; query
`beta_r = b_r + a_r tanh(u_r . x_t)`. They start at `b = 0`, `a = 1`, `u = 0`, so each trial starts as the
control's function. AdamW decays `u` (0.1), never `b` or `a`; the shifts are unconstrained in sign.

- Recipe: the control's (12 layers, width 640, 10 heads, RoPE, coefficient-W decay; 7,551 updates of
  128 × 2,048 tokens, warmup 454, seed 42) with only the shift added and gradient-event capture on.
- Kernels: the trials train on one H100 with the Mosaic score-bias kernels, which form each shifted power
  directly; the control trained on B300's Mosaic query-polynomial kernels.
- Gradient-event capture: after warmup, an update whose finite pre-clip gradient norm exceeds
  `max(1, 5 × median)` of the previous 100 norms is logged in the run's `gradient_events.jsonl`; the
  first such event's pre-update state and the two largest later ones are kept as diagnostic checkpoints.
- Diagnostics: at each validation the forward-only JAX reference of the same kernel, on the first held-out
  sequence (2,098,176 causal pairs), records the shifts, the shifted scores and the weights
  (`attention_diagnostics` in `metrics.jsonl` and `metadata.json`). The values below are from the final
  validation, update 7,551.

Recipe, acceptance and launch: [study README](../../../README.md#per-degree-score-biases).

## Results

| Run | Shift | Validation-best PPL (update) | vs control | Terminal PPL (update 7,551) | vs control |
| --- | --- | ---: | ---: | ---: | ---: |
| `r95a5691d293f` | – (control) | 17.592 (7,551) | – | 17.592 | – |
| `rb4e5457ef2a0` | static | 17.604 (7,551) | +0.07% | 17.604 | +0.07% |
| `r8dc6e147601c` | query | 18.216 (7,551) | +3.55% | 18.216 | +3.55% |

Validation perplexity against the control's at the same update (`metrics.jsonl`):

| Update | 200 | 400 | 500 | 600 | 700 | 800 | 1,000 | 1,500 | 2,000 | 4,000 | 7,551 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query | −6.2% | −11.4% | −11.9% | −5.0% | −2.0% | +0.2% | +1.6% | +2.7% | +3.5% | +3.5% | +3.55% |
| static | −2.0% | −4.4% | −1.4% | −1.6% | −1.1% | −0.8% | −0.4% | −0.1% | +0.3% | −0.0% | +0.07% |

Gradient norms after update 500 (pre-clip; clipping at 1). The validation rows count each window's
clipped updates and its largest norm; the trials' `gradient_events.jsonl` names the events:

| Run | Updates above 1, 501–7,551 (after 600) | Largest (update or window) | Events (update: norm) |
| --- | ---: | --- | --- |
| control | 8 (2) | 3.40 (5,501–6,000) | not logged |
| static | 10 (2) | 6.25 (561) | 561: 6.25, 2,831: 1.81, 7,461: 1.46 |
| query | 26 (17) | 6.42 (793) | 580: 4.49, 608: 3.72, 793: 6.42, 1,402: 3.11, then seven of 1.02–1.85 to 7,461 |

The query trial's log also holds a row for update 2,626, superseded by its resume from update 2,500. Each
trial keeps its three captured pre-update states (query 580, 608, 793; static 561, 2,831, 7,461), listed
in its record.

Learned shifts and weights on the diagnostic sequence, by order `r`:

| Quantity | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| query: mean `beta_r` | −0.46 | −0.59 | −0.35 | −0.38 | −0.28 | −0.06 | +0.14 | +0.59 |
| query: shifted scores below zero | 59% | 62% | 58% | 58% | 56% | 52% | 48% | 39% |
| static: mean `b_r` | −0.14 | −0.10 | −0.08 | −0.14 | −0.09 | −0.07 | −0.05 | +0.28 |
| static: heads with `b_r < 0` | 78% | 83% | 78% | 92% | 85% | 82% | 63% | 2% |
| static: shifted scores below zero | 54% | 54% | 53% | 54% | 54% | 53% | 53% | 48% |

## Findings

- **Neither trial cleared the bar** (below about 17.548, the control's 17.592 less 0.25% seed noise), so
  no reference-shape runs follow.
- **The query shift lost an early lead and finished 3.55% worse.** It led by 6–12% from update 200 to 500
  (warmup ends at 454), fell behind by update 800, during its three largest gradient events (580, 608,
  793), and stayed 3.4–3.8% behind from update 2,000 to the end.
- **The query shift spiked more.** It clipped 17 updates after update 600, against 2 in the static trial
  and 2 in the control. The large early spikes are not specific to it: the static trial reached 6.25 at
  update 561, and the control 3.40 between updates 5,501 and 6,000. All three runs clip one update between
  7,001 and 7,500 (1.46–1.69; update 7,461 in both trials).
- **The static shift tied.** It was ahead by up to 4.4% through update 800, within 0.6% from update 900,
  and within 0.07% from update 3,000 to the end.
- **The learned shifts are mostly negative.** In the query trial, orders 1–5 shift by −0.28 to −0.59 on
  average, order 6 by −0.06, order 7 by +0.14 and order 8 by +0.59; the offsets `b_r` stay within −0.13
  to +0.05 per order, so the query term `a_r tanh(u_r . x_t)` (token standard deviation 0.39–0.68) carries
  the shift. The static offsets are negative in most heads for orders 1–7 and positive for order 8.
- **The coefficients stay positive, so negative shifts make odd orders' terms negative.** In the query
  trial `c_2`–`c_8` are positive at every token of every head (`c_1` in 119 of 120 heads, `c_0` in 112),
  with means close to the control's apart from `c_1` (1.06 against 0.85). 33.5% of its causal weights
  `W(t,i)` are negative, against 21.3% in the static trial and 20.2% in the control.

## Limits

- One seed per configuration.
- Different kernels and GPUs, all rounding the weights to BF16: the trials' H100 Mosaic row-shift
  score-bias kernels (the query trial on an H100 SXM through update 4,000 and an H100 PCIe after, the
  static trial on an H100 PCIe) against the control's B300 query-polynomial kernels.
- The control logs no gradient events: its spikes are known only through its windows' largest norms and
  clipped counts (100 updates to update 1,500, then 500).
- The shift and weight statistics come from one held-out sequence at the final validation.

## Open decision

None pending: neither trial cleared the bar, so no reference-shape runs follow. The topic is kept as the
record of this check.

Refresh the selection after the runs' scientific files change:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.records refresh experiments/transformer_pp/results/exploratory/score_biases
```

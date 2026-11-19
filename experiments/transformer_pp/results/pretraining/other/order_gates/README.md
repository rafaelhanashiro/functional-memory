# Per-order coefficient gates

Opened: 2026-11-07. Does multiplicative gating of individual polynomial orders improve
query-conditioned attention without introducing a distance-dependent forget gate?

## Runs

| Role | Run | Label | Gate | Degree | Home cluster |
| --- | --- | --- | --- | ---: | --- |
| Gated query + weight decay | `r77698776d82c` | `query-m8-w-decay-order-gates` | sigmoid | 8 | ORCD |
| Gated query + weight decay | `r8c4a6ecdf4d2` | `query-m10-w-decay-order-gates` | sigmoid | 10 | ORCD |
| Ungated control | `r25fffc16c2f3` | `query-m8-dk64-w-decay` | – | 8 | Kennel |
| GDN-gated query + weight decay | `r1a5d217f030d` | `query-m4-w-decay-order-gdn-gates` | GDN | 4 | Kennel |
| Ungated control | `rca97cc9db8f2` | `query-m4-w-decay` | – | 4 | Kennel |
| Shared distance gate, no positional encoding | `r3e26412b2811` | `query-m4-query-distance-no-rope-w-decay` | – | 4 | Kennel |
| Per-order distance gates, no positional encoding | `r507193d47450` | `query-m4-per-order-query-gate-nope-wd` | – | 4 | Kennel |

No ungated degree-ten run exists, so the degree-ten run is compared with the gated
degree-eight run and the ungated degree-eight control only. The [selection](selection.yaml)
pins these runs and three references at `m = 8`: fixed Taylor `ra7ee8a4b2544`, learned
static `r9c69dd7cfc8d` and unnormalized exp `r6a76e7fbde45`. The degree-four GDN gate
and its three references are described [below](#gdn-gate-without-a-distance-exponent).

## Model

Suppressing layer and head indices, `x_t` is the activation after the block's
pre-attention RMSNorm and `s_ti = Q_t K_i^T / sqrt(d_K)` is the usual QK-normalized,
RoPE-rotated score. The existing coefficient generator is kept,

    c_r(x_t) = [b_r + a_r tanh(W_r^T x_t)] / r!,

and every layer, head and order `r = 0, ..., m` receives an independent gate:

    g_r(x_t) = sigmoid(U_r^T x_t + beta_r)
    c~_r(x_t) = 2 c_r(x_t) g_r(x_t)
    omega_ti = sum_r c~_r(x_t) s_ti^r
    output_t = head_RMSNorm(sum_{i<=t} omega_ti v_i)

There is no forget gate, cumulative gate product, distance exponent, or division by the
attention weight sum. The gate is applied inside the shared coefficient generator
(`model.attention_coefficient_gate: sigmoid`), so the existing Mosaic query-polynomial
kernel receives `c~` with no kernel change.

**Initialization and normalization.** The coefficient generator starts as in the control:
stored `b_r = 1` (so `b_r / r! = 1/r!`), `a_r = 0`, and `W_r` drawn from
`N(0, 0.0334^2)`. The gates start at `U_r = 0` and `beta_r = 0`, so every `g_r` is
exactly 1/2 and the fixed factor 2 makes the multiplier `2 g_r` exactly 1. The gated model
therefore starts as the ungated Taylor model, with the same initial derivatives with
respect to `W`, `a` and `b`, while the gates learn from the first update:
`dL/dbeta_r = (c_r / 2) dL/dc~_r` and `dL/dU_r = (c_r / 2) x_t dL/dc~_r`. The effective
multiplier `2 g_r` ranges over (0, 2), so an order can be suppressed toward zero or
amplified up to twice its ungated coefficient. The gates start at the sigmoid's steepest
point, far from saturation.

**Weight decay.** AdamW decay 0.1 applies to both projections, `W` and `U`. `a`, `b` and
`beta` are excluded, and the backbone keeps its decay policy
(`exclude_embeddings_norms_biases_v1`).

**Training.** Each run is `r25fffc16c2f3`'s recipe with only the gate added, and degree 10
for the second run. The matched 1× budget is 7,570 updates with a global batch of 256
(16 micro-batches of 16) at 2,048 tokens, seed 42, the same data order, and the same warmup
(455 updates) and cosine schedule. The recipe keeps `d_K = 64`, QK RMSNorm with a learned
per-head score scale, `transformer_truncated_reference` initialization, and per-head
output RMSNorm, on the Mosaic backend with one H200. Only the dataset manifest paths
differ; ORCD's copies hash identically to Kennel's.

## Results

Both runs completed all 7,570 updates without numerical failure, after preemptions (two
attempts at `m = 8`, three at `m = 10`, each resuming from its latest checkpoint). Both end
clearly worse than the ungated control:

| Run | Terminal PPL | Validation-best PPL (update 7,500) | Terminal vs control |
| --- | ---: | ---: | ---: |
| Gated, `m = 8` (`r77698776d82c`) | 15.532 | 15.526 | +8.40% |
| Gated, `m = 10` (`r8c4a6ecdf4d2`) | 15.350 | 15.344 | +7.14% |
| Ungated query + weight decay, `m = 8` (`r25fffc16c2f3`) | 14.328 | 14.323 | – |
| Fixed Taylor, `m = 8` (`ra7ee8a4b2544`) | 14.223 | 14.218 | −0.73% |
| Learned static, `m = 8` (`r9c69dd7cfc8d`) | 14.299 | 14.294 | −0.20% |
| Unnormalized exp (`r6a76e7fbde45`) | 14.262 | 14.257 | −0.46% |

- **The gated runs lead early and then fall behind for good.** Against the control,
  gated `m = 8` is −2.9% at update 100, −15.9% at 400 and −1.5% at 600. It crosses the
  control near update 650, reaches +9.5% at 900, and then holds at +8.4–9.0% to the end.
  `m = 10` has the same shape. The gap is about 40 times the 0.2% between equivalent runs.
- **The gates suppress the low orders, and the coefficients do not compensate.** At
  update 7,570 (`m = 8`, means over the 210 heads, orders 0 to 8):

  | Order | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Multiplier `2 g_r` | 0.50 | 0.65 | 0.45 | 0.61 | 0.59 | 1.00 | 1.21 | 1.62 | 1.56 |
  | Tilt | −1.33 | −0.29 | −1.37 | −0.70 | −0.87 | 0.43 | 0.86 | 1.74 | 1.53 |
  | Ungated `r! c_r` (median) | 0.89 | 0.91 | 0.85 | 0.88 | 0.84 | 0.90 | 0.93 | 1.25 | 1.30 |
  | Gated `r! c~_r` (median) | 0.32 | 0.42 | 0.23 | 0.28 | 0.28 | 0.85 | 1.24 | 2.21 | 2.33 |
  | Contribution share | 5.5% | 7.8% | 6.2% | 8.3% | 7.4% | 12.0% | 13.0% | 22.3% | 17.6% |

  The ungated coefficients stay near their Taylor start of 1. In heads whose multiplier
  is below 0.2, the ungated `|r! c_r|` is 0.72–0.88 times its value in the other heads,
  smaller rather than larger, and the gated `|r! c~_r|` is 0–10% of theirs. So the low
  orders are really suppressed, not offset. Orders 0–2 carried 56% of each row's absolute
  contribution at update 100 and 19% at the end; orders 7–8 went from 6% to 40%. At
  `m = 10` the multipliers are 0.49–0.74 for orders 0–4 and 1.25–1.44 for orders 7–10,
  with the contribution spread more evenly over orders 5–10.
- **The gates lock in while the gap opens.** At `m = 8`, the number of (layer, head,
  order) gates with a mean multiplier below 0.2 rose from 0 at update 100 to 138 at
  400, 332 at 900 and 507 of 1,890 at the end. Multipliers above 1.8 numbered 208 by
  update 200 and stayed near 320–370. A gate near zero also scales its coefficient's
  gradient toward zero, so neither can recover. The model trains around a configuration
  dominated by high orders that was fixed while the backbone was still early in training.
- **Degree within the gated family.** Gated `m = 10` ends 1.2% below gated `m = 8`.

## Diagnostics

At every validation event, `metrics.jsonl` records the gate statistics under
`attention_diagnostics.coefficients.coefficient_gates`. That is every 100 updates
through 1,500, then every 500, and the terminal update 7,570: 28 events per run. Each
statistic is per layer, head and order on the fixed held-out diagnostic sequence:

- the gate `g_r`: mean, standard deviation, minimum and maximum over tokens;
- the order tilt `log g_r - mean_r log g_r`: mean and standard deviation;
- the ungated coefficient `c_r`: mean and standard deviation;
- the gated coefficient `c~_r`: the existing effective-coefficient summary (mean,
  standard deviation, minimum and maximum);
- each order's contribution `c~_r(x_t) s_ti^r` over causal pairs: RMS, mean absolute
  value, and mean share of each row's total absolute contribution over orders, plus each
  head's mean `|omega_ti|`;
- the gate parameters: `beta` and the L2 norm of `U` per order.

A small gate alone does not establish suppression, because `c_r` can grow to compensate.
Per-head output RMSNorm also cancels any factor that all orders share at a token.
Suppression of an order therefore needs a small gated coefficient `c~_r` and a small
contribution share, read together with the tilt. W&B mirrors the per-order means over
layers and heads of the gate, the tilt, and the contribution share.

## GDN gate without a distance exponent

`r1a5d217f030d` asks whether the per-order query-distance gate's gain comes from gating each
order or from the distance decay. The per-order distance run `r507193d47450`
(`query-m4-per-order-query-gate-nope-wd`, no positional encoding) weights

    omega_ti = sum_r c_r(x_t) s_ti^r exp(-(t - i) lambda_tr)
    lambda_tr = exp(A_log_r) softplus(w_r^T x_t + dt_bias_r)

per layer, head and order, in Gated DeltaNet's (Mamba-2's) parameterization. Dropping
`(t - i)` leaves `alpha_tr = exp(-lambda_tr)`, which depends only on the query and folds into
the coefficient:

    c~_r(x_t) = alpha_tr c_r(x_t),   omega_ti = sum_r c~_r(x_t) s_ti^r

This is `model.attention_coefficient_gate: gdn`, a per-order coefficient gate applied in the
shared generator like the sigmoid gate. The Mosaic query-polynomial kernel receives `c~`
with no kernel change. `alpha` lies in `(0, 1]`; with per-head output RMSNorm only ratios
between orders matter, so suppressing the other orders amplifies one.

| Run | Gate | Positional encoding | Status | Validation-best PPL | Terminal PPL |
| --- | --- | --- | --- | ---: | ---: |
| `r1a5d217f030d` | per order, `exp(-lambda_tr)` | RoPE | complete | 15.981 | 15.983 |
| `rca97cc9db8f2` | none | RoPE | complete | 14.709 | 14.714 |

**Initialization.** The generator starts as the control's: stored `b_r = 1`, `a_r = 0`,
`W_r ~ N(0, 0.0334^2)`. `w`, `A_log` and `dt_bias` take the shared input-form forget gate's
distributions (`w` uniform on `+-1/sqrt(896)`, `exp(A_log)` uniform on `(0, 16)`,
`softplus(dt_bias)` log-uniform on `[0.001, 0.1]`), drawn once per layer and head and
copied to all five orders. All orders of a head therefore share `alpha_t` at each token, and
the per-head output RMSNorm cancels a shared factor up to its epsilon: the normalized output
is the ungated one times `sqrt((ms + eps alpha^2) / (ms + eps))`, with `ms` the gated head
output's mean square. At seed 42 on one 64-token sequence, `alpha` spans 0.057–1 (median
0.94; per-head means 0.11–1.00). Every other initial parameter equals the control's start
(state hash, the same at the control's launch commit). The initial logits differ from the
control's by 5.9e-5 of the largest logit, within the epsilon bound of 2.6e-3, and by 1.0e-6
(FP32 rounding) with every RMSNorm epsilon at 1e-30.

**Weight decay.** AdamW decay 0.1 applies to `W` and `w`; `a`, `b`, `A_log` and `dt_bias`
are excluded.

**Training.** `rca97cc9db8f2`'s recipe with only the gate added: 7,570 updates, global batch
256 at 2,048 tokens, seed 42, warmup 455 and the cosine schedule, `d_K = 64`, QK RMSNorm
with a learned per-head score scale, RoPE, per-head output RMSNorm,
`transformer_truncated_reference` initialization, on the Mosaic backend.

**The 2×2.** Validation-best PPL at `m = 4`:

| | Gate shared by all orders | Gate per order |
| --- | ---: | ---: |
| Distance exponent `exp(-(t - i) lambda)`, no positional encoding | 14.198 (`r3e26412b2811`) | 13.892 (`r507193d47450`) |
| No distance exponent, RoPE | 14.709 (`rca97cc9db8f2`) | 15.981 (`r1a5d217f030d`) |

- A gate shared by all orders without a distance exponent is a per-token factor that the
  per-head output RMSNorm cancels up to its epsilon, so the ungated control fills that cell.
- The rows also differ in position encoding: the distance gates carry position without
  RoPE, while attention without a distance gate needs RoPE. `r3e26412b2811` trains on the
  blocked backend.
- The [selection](selection.yaml) pins all four cells.

**Results.** Per-order gating helps only with the distance exponent. Without it, the GDN gate
ends 8.65% above its ungated control at the best update (15.981 vs 14.709; terminal 15.983
vs 14.714, +8.63%), the same loss as the sigmoid gate at `m = 8` (+8.40%), and 15.0% above
the per-order distance gate (13.892). It leads the control only through update 600 (by up to
11.7%, at update 300), crosses it near update 650, and stays 8.6–10.2% above it from update
1,500 on; it is lower at 6 of 28 evaluations.

The gates repeat the sigmoid gate's shift toward the highest orders, although `alpha` cannot
exceed one: the per-head output RMSNorm leaves only the ratios between orders. At update
7,570 (means over the 210 heads, orders 0 to 4):

| Order | 0 | 1 | 2 | 3 | 4 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multiplier `alpha_r` | 0.51 | 0.65 | 0.68 | 0.94 | 0.88 |
| Heads with `alpha_r` below 0.2 | 64 | 50 | 46 | 4 | 7 |
| Tilt | −2.64 | −1.03 | −0.76 | 2.51 | 1.93 |
| Ungated `r! c_r` (median) | 0.89 | 0.89 | 0.89 | 1.14 | 1.00 |
| Gated `r! c~_r` (median) | 0.49 | 0.59 | 0.76 | 1.13 | 0.95 |
| Contribution share | 6.0% | 11.3% | 15.1% | 38.2% | 29.3% |

Orders 0–2 carried 69% of each row's absolute contribution at update 100 and 32% at the
end. Order 0's gate is below 0.2 in 8 heads at update 100, 47 at 400, 60 at 900 and 64 at
the end. Most of the suppression forms in the first 900 updates (131 of the final 171 gates
below 0.2, against 332 of 507 for the sigmoid gate), and the shares keep moving toward
orders 3 and 4 afterwards. The per-order distance gate learns the same preference, short
ranges for the low orders (order 0's median half-life is 0.92 tokens), but every order keeps
its factor of one at distance zero, so no order loses its short-range part.

The gate reports the sigmoid gate's diagnostics, with `alpha` in the gate fields (mean,
standard deviation, minimum and maximum over tokens), the tilt
`log alpha_r - mean_r log alpha_r`, contribution shares, ungated and gated coefficients, and
the gate parameters `exp(A_log)`, `softplus(dt_bias)` and `||w_r||` per layer, head and order.

## Comparisons and limits

The control `r25fffc16c2f3` ran on a Kennel B300 at source commit `6ab34ab1`. The gated
runs trained on ORCD H200s (`m = 8` on H200 NVL and H200 nodes across its attempts) at
source commit `fa03c22d`. The comparison therefore crosses cluster, GPU and code version,
as the forget-gate comparison at `m = 8` does; a gap of 7–8% is far larger than any
cross-cluster difference seen in this study. The initial parameters match the ungated
recipe exactly, but gradient clipping's global norm includes the gate gradients, so
updates differ from the control's from the first step. There is one seed per run, and
the gate diagnostics come from one held-out sequence.

Both distance-free gates lose to their ungated controls by the same early lock-in: the
sigmoid gate at degrees 8 and 10, and the GDN gate of the successful per-order distance
runs at degree 4. A gate near zero stops both its own learning and its coefficient's, and
the configuration chosen in the first ~900 updates stays. The GDN gate's comparison with
`rca97cc9db8f2` shares the cluster, backend, seed and every recipe setting but the gate;
the source commits differ, as they do across the 2×2, whose rows also differ in position
encoding.

The open question is whether to delete this topic, since per-order gating helps only through
the distance exponent of the query_distance_decay topic, or to try a gate that can recover:
a multiplier that cannot reach zero (for example `exp(delta_r)` with `delta` decayed toward
zero), a lower learning rate for the gate parameters, or gates held fixed through the early
updates.

## Execution

The GDN-gated run's launch command and acceptance are in the
[study README](../../../../README.md#per-order-coefficient-gates); it runs on Kennel with one
B300 or H100 GPU.

Home cluster for the sigmoid runs: ORCD. Each run uses one H200 (Hopper, sm90, the H100's
kernels), with no scheduling dependency; at submission Slurm estimated an H200 start about
two days before an H100 one. From the experiment worktree after committing the frozen
execution inputs:

```bash
for RUN in r77698776d82c r8c4a6ecdf4d2; do
  sbatch --parsable --job-name="$RUN" --output="experiments/transformer_pp/runs/$RUN/training.log" \
    --gres=gpu:h200:1 \
    scaffolds/transformer/launchers/orcd.sbatch experiments.transformer_pp.coefficient_gate_acceptance \
    "experiments/transformer_pp/runs/$RUN/config.yaml" "$(git rev-parse HEAD)" 1 H200
done
```

Acceptance ([coefficient_gate_acceptance.py](../../../../coefficient_gate_acceptance.py)) pins
the control's recipe and rejects any scientific difference except the gate and degree 10.
It checks that the gates start at exact zeros, and that every other initial parameter
(by state hash) and the initial logits equal the ungated recipe's at the same seed. It
then runs the gate's CPU tests and its GPU qualification at degrees 8 and 10, alongside
the shared pilot and checkpoint-resume checks. The GPU qualification compares the gated
generator through Mosaic with FP64 at 2,048 tokens, and the gated model's loss and
gradients on Mosaic with the FP32 blocked model. Both runs passed every stage on their
H200 allocations ([qualification record](../../../../../../scaffolds/transformer/validation/coefficient_gates_h200.json)).
Submissions and allocations live in each run's record. The validation-best checkpoints
and logs are retained under
`/orcd/pool/006/rafah/functional_memory/artifacts/experiments/transformer_pp/runs/<id>/`.

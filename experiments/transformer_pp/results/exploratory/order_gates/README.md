# Per-order coefficient gates

Opened: 2026-11-07. Does multiplicative gating of individual polynomial orders improve
query-conditioned attention without introducing a distance-dependent forget gate?

## Runs

| Role | Run | Label | Degree | Home cluster |
| --- | --- | --- | ---: | --- |
| Gated query + weight decay | `r77698776d82c` | `query-m8-w-decay-order-gates` | 8 | ORCD |
| Gated query + weight decay | `r8c4a6ecdf4d2` | `query-m10-w-decay-order-gates` | 10 | ORCD |
| Ungated control | `r25fffc16c2f3` | `query-m8-dk64-w-decay` | 8 | Kennel |

Both gated runs are in progress; this topic reports them when they finish. No ungated
degree-ten run exists, so the degree-ten run is compared with the gated degree-eight run and
the ungated degree-eight control only.

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
output RMSNorm, on the Mosaic backend with one H100. Only the dataset manifest paths
differ; ORCD's copies hash identically to Kennel's.

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

## Comparisons and limits

The control `r25fffc16c2f3` ran on a Kennel B300 at source commit `6ab34ab1`. The gated
runs train on ORCD H100s at this topic's source commit. The comparison therefore crosses
cluster, GPU and code version, as the forget-gate comparison at `m = 8` does. The
initial parameters match the ungated recipe exactly, but gradient clipping's global norm
includes the gate gradients, so updates differ from the control's from the first step.
There is one seed per run. Validation-best and terminal perplexity will be reported
separately.

The open decision is whether per-order gates merit a matched comparison beyond these two
runs.

## Execution

Home cluster: ORCD. Each run uses one H200 (Hopper, sm90, the H100's kernels), with no
scheduling dependency; at submission Slurm estimated an H200 start about two days before
an H100 one. From the experiment worktree after committing the frozen execution inputs:

```bash
for RUN in r77698776d82c r8c4a6ecdf4d2; do
  sbatch --parsable --job-name="$RUN" --output="experiments/transformer_pp/runs/$RUN/training.log" \
    --gres=gpu:h200:1 \
    scaffolds/transformer/launchers/orcd.sbatch experiments.transformer_pp.coefficient_gate_acceptance \
    "experiments/transformer_pp/runs/$RUN/config.yaml" "$(git rev-parse HEAD)" 1 H200
done
```

Acceptance ([coefficient_gate_acceptance.py](../../../coefficient_gate_acceptance.py)) pins
the control's recipe and rejects any scientific difference except the gate and degree 10.
It checks that the gates start at exact zeros, and that every other initial parameter
(by state hash) and the initial logits equal the ungated recipe's at the same seed. It
then runs the gate's CPU tests and its GPU qualification at degrees 8 and 10, alongside
the shared pilot and checkpoint-resume checks. The GPU qualification compares the gated
generator through Mosaic with FP64 at 2,048 tokens, and the gated model's loss and
gradients on Mosaic with the FP32 blocked model. Submissions and allocations live in each
run's record.

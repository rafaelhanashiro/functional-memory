# Unnormalized Taylor attention

Do fixed Taylor truncations of the exponential attention kernel train when the
attention row normalization is removed? With normalization, the odd degrees
failed numerically: degree 1 at update 9 and degree 3 at update 17, when signed
row sums approached zero ([odd Taylor attention](../odd_taylor_attention/README.md)).
This sweep trains degrees **m = 1, 2, 3, 4** with

```text
y_i = sum_{j <= i} p_m(s_ij) v_j,   s_ij = q_i · k_j / sqrt(d_K),   p_m(x) = sum_{n <= m} x^n / n!
```

and no denominator, so there is no near-zero row sum to divide by.
**Status: completed.** All four degrees trained stably for the full 10B tokens,
including degrees 1 and 3, whose normalized versions failed within 17 updates.
Every truncation ends behind the unnormalized exponential (PPL 17.46), and odd
degrees end ahead of even ones: m=3 20.24, m=1 20.77, m=4 21.46, m=2 23.51.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; the packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`. A clone provides code and records,
not checkpoints, environments, logs, or datasets. The
[provenance manifest](provenance/manifest.json) is the authoritative record of
sources, configurations, runs, and artifact locations.

## Results

| Attention (200M, seed 0) | Normalized terminal PPL | Unnormalized terminal NLL | Unnormalized terminal PPL |
| --- | ---: | ---: | ---: |
| Taylor m=1 | failed at update 9 | 3.033512 | 20.7701 |
| Taylor m=2 | 16.2810 | 3.157241 | 23.5057 |
| Taylor m=3 | failed at update 17 | 3.007646 | 20.2397 |
| Taylor m=4 | 14.9836 | 3.066279 | 21.4619 |
| Exponential (`attention_degree: null`, m → ∞) | 14.6942 (softmax) | 2.859817 | 17.4583 |

Every run was best at its terminal update 19,074. Removing normalization makes
the odd degrees trainable but costs every degree substantially: unnormalized m=2
and m=4 end 0.367 and 0.359 nats behind their normalized versions, more than
twice the 0.172-nat cost for the exponential. The ordering is not monotonic in
degree; one seed per degree does not establish it across seeds. Validation NLL
at matched updates:

| Update | m=1 | m=2 | m=3 | m=4 | Unnormalized exp | Softmax |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,000 | 5.310 | 5.419 | 5.155 | 5.241 | 5.285 | 4.458 |
| 2,000 | 4.633 | 4.740 | 4.346 | 4.469 | 4.434 | 3.609 |
| 5,000 | 3.679 | 3.839 | 3.472 | 3.526 | 3.437 | 3.020 |
| 10,000 | 3.214 | 3.378 | 3.169 | 3.211 | 3.052 | 2.806 |
| 15,000 | 3.076 | 3.211 | 3.049 | 3.103 | 2.906 | 2.720 |
| 19,074 | 3.034 | 3.157 | 3.008 | 3.066 | 2.860 | 2.687 |

All 20 evaluations per run reported valid diagnostics with zero invalid scores,
kernels, weight sums, or outputs, and no run recorded a numerical failure. On the
fixed diagnostic sequence, odd degrees kept many signed weights: across
evaluations, the most-negative layer had 48–55% negative causal kernel values
in m=1 and 37–42% in m=3. Their row sums came
arbitrarily close to zero; m=1 had an exactly zero row sum at update 18,000
(layer 3) and m=3 a minimum of 1.7e-6 (update 11,000, layer 6). Normalization
would divide by these sums. Maximum row weight sums peaked early at 12,371 (m=1,
update 1,000), 115,110 (m=2, update 4,000), 140,263 (m=3, update 2,000), and
647,346 (m=4, update 4,000), all far from FP32 overflow. At completion m=1 scores
spanned -92 to 85, versus about -42 to 47 or narrower for the other degrees.

| Run | Source | Slurm job | Hardware | Elapsed | Mean tokens/s |
| --- | --- | --- | --- | ---: | ---: |
| `200m-m001-seed0` | `972d223` | `201011` | 4×B300 `akita` | 2 h 16 min | 1.44M |
| `200m-m002-seed0` | `972d223` | `201012` | 4×H100 `mastiff` | 5 h 10 min | 0.58M |
| `200m-m003-seed0` | `972d223` | `201013` | 4×H100 `mastiff` | 5 h 00 min | 0.60M |
| `200m-m004-seed0` | `972d223` | `201014` | 4×B300 `akita` | 3 h 02 min | 1.06M |

Elapsed times include acceptance. The m=1 job was first preempted on H100 node
`dogo` during CPU acceptance, before any training update, and then ran from
scratch on B300 after automatic requeue; both submission records are retained.
No run resumed from a checkpoint. Records: metrics and completion in
[`runs/`](runs/), acceptance and allocations in [`validation/`](validation/), a
[terminal summary](summary.csv), one log per run at
`logs/unnormalized_taylor_attention/<run>/slurm-<job_id>.out`, and each
run's retained step-19,074 checkpoint (2.1 GB) under
`checkpoints/unnormalized_taylor_attention/<run>/` at the artifact root.

## Scientific recipe

Each [run recipe](configs/) is the canonical
[unnormalized exponential recipe](../unnormalized_attention/configs/base.yaml)
with `model.attention_degree: m` added; a configuration test enforces that
nothing else differs except output locations. That recipe is the repository's
200M sweep recipe apart from normalization: 199,906,304 parameters, 10 layers,
width 1024, 16 full-MHA heads, `d_K=d_V=64`, SwiGLU width 3008, untied 32,000-token
embeddings, RoPE base 10,000, no dropout, seed 0, packed FineWeb in the fixed
order, global batch 256 with four accumulation steps, AdamW with peak/minimum
learning rates `3e-4`/`3e-5` and 2,000 warmup updates, **19,074 updates and
10,000,269,312 prediction tokens**. Evaluation uses the first 1,792 held-out
sequences every 1,000 updates and at completion. Initial parameters are
bit-identical across degrees and to the 200M controls. Primary endpoint:
terminal held-out perplexity; best-checkpoint metrics are retained separately.

Scores are unshifted after RoPE. Odd degrees keep signed weights; even-degree
truncations are strictly positive, and a nonpositive even-degree weight is
treated as invalid. There is no clipping, epsilon, absolute value, or length
scaling. Row sums of the kernel, their minimum absolute value, the cancellation
ratio `sum|p_m| / |sum p_m|`, and negative-weight fractions are recorded for
the fixed diagnostic sequence as measurements only. Nonfinite scores, kernel
values, outputs, gradients, or updated parameters stop training with a
numerical-failure record.

Existing 200M comparison points, all seed 0 with the same recipe apart from
the attention kernel:

Normalized Taylor results in the table above come from [polynomial attention](../polynomial_attention/README.md)
and [odd Taylor attention](../odd_taylor_attention/README.md); the normalized
exponential is the exact-softmax control. The two existing softmax controls,
trained on different clusters and kernels, agree within 0.0003 nats, so
hardware differences between these runs are negligible at this resolution.

## Implementation

The [shared Transformer scaffold](../../scaffolds/transformer/README.md)
selects this mechanism with `attention_normalization: false` and a nonnegative
`attention_degree`. The fused Pallas/Triton kernel in
`attention/backends/fused_unnormalized.py` is the same tiled unnormalized kernel
used for the exponential, with `exp` replaced by the Horner evaluation of `p_m`.
It visits only causal tiles and never materializes a sequence-square matrix.
Kernel values and accumulation are FP32; mixed FP32/BF16 products split the
FP32 operand into three BF16 components. The backward pass recomputes weights
and uses `dS = p_m'(S) * (dY Vᵀ)`, differentiating the same Horner recurrence;
there is no row-coupling term. The blocked reference supplies the scalar
diagnostics. Learned-coefficient quadratics remain normalized-only.

## Validation

From a dedicated checkout with the shared environment prepared:

```bash
JAX_PLATFORMS=cpu "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/studies/unnormalized_attention/tests \
  --import-mode=importlib -m 'not gpu'
```

CPU tests compare blocked outputs and Q/K/V gradients for degrees 0–4 with an
independent FP64 power-sum formula, and cover prefix sums at zero scores,
signed weights and exact zero row sums, strictly positive even kernels,
initialization independence, recipe equivalence, recorded identity, and short
training runs. GPU tests, run automatically by the launcher's acceptance in the
production allocation, compare the fused forward and VJP with the blocked
reference and FP64 for degrees 0–4, including odd widths, tile boundaries,
full 2,048-token BF16 context, causality, and invalid-score handling.
Acceptance then verifies full-model initialization, runs 20 full-recipe
updates with validation, and checks one-update checkpoint resume.

## Kennel launch

Each run used four B300s or four H100s, whichever were free, in the
low-priority `sweep` partition: submissions exclude the A100 and A6000 nodes,
and `GPU_TYPE=auto` records the allocated type. One Python process per run uses
global batch 256, microbatch 64, and four accumulation steps. Acceptance is
keyed to the resolved GPU type, so a requeue onto the other type reruns it
before resuming the latest checkpoint; matching passed stages are reused.
To reproduce, create a dedicated execution checkout of the recorded source
and submit from it, with fresh output paths if the original records should be
kept:

```bash
PROJECT_ROOT=/data/cl/u/rafah/home/projects/functional-memory
EXECUTION_ROOT="$PROJECT_ROOT/worktrees/unnormalized_taylor_attention-972d223"
git -C "$PROJECT_ROOT" worktree add --detach "$EXECUTION_ROOT" 972d223d89c2542028dec761569c25dbd7379b36
SOURCE_COMMIT="$(git -C "$EXECUTION_ROOT" rev-parse HEAD)"
for DEGREE in 1 2 3 4; do
  RUN="200m-m00${DEGREE}-seed0"
  CONFIG="$EXECUTION_ROOT/results/unnormalized_taylor_attention/configs/$RUN.yaml"
  LOG_DIR="$PROJECT_ROOT/logs/unnormalized_taylor_attention/$RUN"
  mkdir -p "$LOG_DIR"
  sbatch --job-name="unnorm-taylor-m${DEGREE}" --partition=sweep --gres=gpu:4 \
    --exclude=collie,samoyed,doodle,terrier --output="$LOG_DIR/slurm-%j.out" \
    "$EXECUTION_ROOT/scaffolds/transformer/studies/unnormalized_attention/launch_kennel.sbatch" \
    "$EXECUTION_ROOT" "$PROJECT_ROOT" "$SOURCE_COMMIT" "$CONFIG" 4 auto
done
```

The launcher records Slurm allocation, source, and configuration under
`<artifact-root>/.cache/unnormalized_taylor_attention/submissions/` and
acceptance under `preflight/` beside it. Metrics and resolved runtime metadata
live in `results/unnormalized_taylor_attention/runs/<run>/`, checkpoints in
`checkpoints/unnormalized_taylor_attention/<run>/`, and logs in
`logs/unnormalized_taylor_attention/<run>/` at the artifact root.
Preserve the source checkout and shared environment while jobs can run or
requeue. After completion, only each run's validation-best checkpoint is
retained, metrics and acceptance evidence are copied into this record, and the
execution checkout and cache are removed.

## Limitations

Each degree has one seed. Runs used B300 or H100 GPUs; the two softmax
controls on different hardware agree within 0.0003 nats, far below the gaps
here. Degree 0, an unnormalized causal prefix sum whose
Q/K projections receive no attention gradient, is supported but not run.
Diagnostics sample one held-out sequence per evaluation.

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
**Status: four runs prepared for Kennel.**

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; the packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`. A clone provides code and records,
not checkpoints, environments, logs, or datasets. The
[provenance manifest](provenance/manifest.json) is the authoritative record of
sources, configurations, runs, and artifact locations.

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

| Attention | Normalized terminal PPL | Unnormalized terminal PPL |
| --- | --- | --- |
| Taylor m=1 | failed at update 9 | this sweep |
| Taylor m=2 | 16.2810 | this sweep |
| Taylor m=3 | failed at update 17 | this sweep |
| Taylor m=4 | 14.9836 | this sweep |
| Exponential (`attention_degree: null`, m → ∞) | 14.6942 (softmax) | `unnormalized_attention/200m-exp-seed0-kennel` |

Normalized Taylor results come from [polynomial attention](../polynomial_attention/README.md)
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

Each run uses four B300s or four H100s, whichever are free, in the
low-priority `sweep` partition: submissions exclude the A100 and A6000 nodes,
and `GPU_TYPE=auto` records the allocated type. One Python process per run uses
global batch 256, microbatch 64, and four accumulation steps. Acceptance is
keyed to the resolved GPU type, so a requeue onto the other type reruns it
before resuming the latest checkpoint; matching passed stages are reused.
Exact submission from the frozen experiment checkout:

```bash
PROJECT_ROOT=/data/cl/u/rafah/home/projects/functional-memory
EXECUTION_ROOT="$PROJECT_ROOT/worktrees/unnormalized_taylor_attention"
SOURCE_COMMIT="$(git -C "$EXECUTION_ROOT" rev-parse HEAD)"
for DEGREE in 1 2 3 4; do
  RUN="200m-m00${DEGREE}-seed0"
  CONFIG="$EXECUTION_ROOT/results/unnormalized_taylor_attention/configs/$RUN.yaml"
  LOG_DIR="$PROJECT_ROOT/results/logs/unnormalized_taylor_attention/$RUN"
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
`results/logs/unnormalized_taylor_attention/<run>/` at the artifact root.
Preserve the source checkout and shared environment while jobs can run or
requeue. After completion, retain only each run's validation-best checkpoint,
copy metrics and acceptance evidence into this record, and record any
numerical failure with its stopping step, last valid metrics, diagnostics, and
failure log.

## Limitations

Each degree has one seed. Degree 0, an unnormalized causal prefix sum whose
Q/K projections receive no attention gradient, is supported but not run.
Diagnostics sample one held-out sequence per evaluation.

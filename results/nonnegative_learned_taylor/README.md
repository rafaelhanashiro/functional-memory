# Nonnegative learned Taylor kernels

[Selected runs and source hashes](selection.yaml) · [Run index](../../experiments/nonnegative_learned_taylor/runs.csv) · [Execution guide](../../experiments/nonnegative_learned_taylor/README.md)

Does a normalized learned polynomial kernel train once it cannot go negative?
The [QK-norm learned Taylor sweep](../qk_norm_learned_taylor/README.md) learned all
`m + 1` coefficients of `p(s) = sum_r c_r s^r` per layer and head under
normalization, and every degree failed numerically (updates 27, 1,857, 318, and
1,456) once its kernel turned negative inside the bounded score range and a row sum
crossed zero. This experiment keeps that recipe and changes only the kernel's
parameterization, so that it is nonnegative by construction:

```text
y_i = sum_{j <= i} p_{l,h}(s_ij) v_j / sum_{j <= i} p_{l,h}(s_ij),
p_{l,h}(s) = q1_{l,h}(s)^2 + q2_{l,h}(s)^2,   deg q1 = m / 2,   deg q2 = m / 2 - 1,
s_ij = exp(gamma_{l,h}) · <q_i, k_j> / (||q_i|| ||k_j||)
```

for **m = 2 and m = 4**. Every polynomial of even degree that is nonnegative on the
real line has this form, with the same `m + 1` parameters as the direct coefficients,
so the only change is the constraint. Odd degrees cannot be nonnegative on the real
line and are not part of this experiment. The factors are learned per layer and head,
FP32, without weight decay, and initialized so that `p` equals the Taylor polynomial
exactly:

```text
m = 2:  T_2(s) = ((1 + s) / sqrt 2)^2 + (1 / sqrt 2)^2
m = 4:  T_4(s) = (0.5500 + 0.4082 s + 0.2041 s^2)^2 + (0.8351 + 0.3298 s)^2
```

The `m = 2` case generalizes the earlier positive quadratic `0.5 (1 + (a s + b)^2)`,
which fixed the constant square; here all three values are learned. The kernel is
evaluated as `q1^2 + q2^2`, never through expanded coefficients, so no rounding can
make it negative; kernel values may be exactly zero, and only a zero row sum is
invalid.

**Status: concluded.** Both degrees trained all 19,074 updates without a numerical
failure; `m = 4` finished 0.28 perplexity above QK-norm softmax.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. The
[study record](../../experiments/nonnegative_learned_taylor/sweep.yaml) is the authoritative record of the
source commit, frozen execution set, acceptance, execution, and artifact locations.
The packed corpus is under `/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Results

<!-- BEGIN GENERATED RESULTS -->
Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0, d_K = 64, per-head factors and QK-norm scales, against the signed direct-coefficient runs of the same recipe and the QK-norm softmax baseline. The minimum row sum is over every validation's diagnostic sequence.

| m | Status | Terminal PPL | Δ vs QK-norm softmax | Signed control | Minimum row sum |
|---:|---|---:|---:|---|---:|
| 2 | complete | 16.1758 | +1.7184 | failed @1,857 | 0.000371 |
| 4 | complete | 14.7419 | +0.2845 | failed @1,456 | 6.22e-05 |

[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf)
<!-- END GENERATED RESULTS -->

## Findings

**Nonnegativity is enough for these kernels to train.** The signed runs of the same
recipe failed at updates 1,857 (`m = 2`) and 1,456 (`m = 4`). Written as a sum of
squares, both degrees completed, and all 20 validations of each run recorded valid
attention diagnostics, with no negative kernel values and no invalid normalizers or
outputs.

**`m = 4` comes within 0.28 of softmax.** It reached **14.7419**, against 14.4574 for
QK-norm softmax and 14.6655 for softmax without QK-norm. That is 1.64 below the
unnormalized learned `m = 4` kernel (16.3840) and 0.15 below the normalized fixed
Taylor `m = 4` at `d_K = 64` under the same defaults (14.8963,
[dK × Taylor-degree record](../key_dimension_taylor/README.md)). `m = 2` reached
**16.1758**, 1.72 above softmax, 0.33 below the fixed Taylor `m = 2` under the same
defaults (16.5107), level with the earlier positive quadratic (16.2125, a legacy run
with the original weight decay and no QK-norm), and 1.42 below the unnormalized
learned `m = 2` (17.5990).

**Both kernels learned a touching root.** The factors moved each kernel onto a zero
inside the reachable score range `|s| <= exp(gamma)`, which the signed kernels could
only reach by passing through zero:

- At `m = 2`, the constant factor `q2` fell from its initial `1/sqrt 2` to a median
  of 2.2e-5 across the 160 heads (below 1e-3 in 109), so `p(s)` is almost exactly
  `(a + b s)^2`. Its double root sits inside the range, at a median score of
  -0.6.
- At `m = 4`, every head's minimum over the reachable range is below 1e-3 of its
  maximum (below 1e-6 in 126 heads), at an interior score, with a median of -2.0.

Each head can therefore give a key near its learned root almost no weight while the
kernel stays nonnegative. The learned score scales `exp(gamma)` grew from 8 to
medians of 13.8 (`m = 2`) and 12.0 (`m = 4`). Row sums stay positive but can become
small: the smallest across all validation diagnostic sequences was 3.7e-4 at `m = 2`
and 6.2e-5 at `m = 4`. The outputs remain convex combinations of the values, and the
normalizers accumulate in FP32.

## Scientific recipe

Each [recipe](../../experiments/nonnegative_learned_taylor/runs/) is the corresponding normalized recipe of the
[QK-norm learned Taylor sweep](../../experiments/qk_norm_learned_taylor/runs/) with
`attention_coefficient_parameterization: sum_of_squares` added; a configuration test
enforces that nothing else differs except output locations. The backbone is the
repository's 200M model: 10 layers, width 1024, 16 heads, `d_K = d_V = 64`, untied
embeddings, seed 0, packed FineWeb in the fixed order, global batch 256, AdamW
with peak/minimum learning rates `3e-4`/`3e-5`, 2,000 warmup updates, selective
weight decay (`exclude_embeddings_norms_biases_v1`), QK-norm with per-head log scales
from `log(sqrt(d_K)) = log 8`, **19,074 updates and 10,000,269,312 prediction
tokens**. Evaluation uses the first 1,792 held-out sequences every 1,000 updates and
at completion. The two runs use different backends for the same mathematics: `m = 4`
runs on the scaffold's blocked reference, the only backend for normalized learned
kernels when it launched. `m = 2` first launched on the blocked reference too; when the
fused normalized polynomial kernel was qualified the same evening, that attempt was
cancelled and `m = 2` restarted from initialization on the fused kernel, which runs
about four times faster on H100. Each recipe names its backend explicitly.

Controls: the signed direct-coefficient runs of the same recipe (both failed), the
[QK-norm softmax baseline](../qk_norm_attention/README.md) (14.4574), and, for
`m = 2`, the earlier positive quadratic without QK-norm and under the original decay
policy (16.2125, [learned quadratic attention](../learned_quadratic_attention/README.md)).

## Artifacts

| Artifact | Location in `experiments/nonnegative_learned_taylor/` |
| --- | --- |
| Exact recipes | `experiments/nonnegative_learned_taylor/runs/<run_id>/config.yaml` |
| Committed metrics, metadata, and completion or failure records | `runs/<run_id>/` |
| Summary and figure | `summary.csv`, `learning_curves.{png,pdf}` |
| Canonical manifest | `experiments/nonnegative_learned_taylor/sweep.yaml` |
| Validation-best checkpoints (ignored) | `runs/<run_id>/checkpoints/<step>/` |
| Slurm logs (ignored) | `runs/<run_id>/training.log` |

## Reproduction

Regenerate the summary, figure, and results block from the committed runs; no GPU,
dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.nonnegative_learned_taylor.report
```

The runs were submitted from this experiment's worktree root with the shared Kennel
launcher, which checks the frozen execution set against the launch commit, runs
acceptance (CPU tests, distributed CPU tests, GPU tests of the recipe's kernels at
the production context, initialization, a 20-update pilot, and an exact resume
audit) in the allocation, then trains; the manifest records each submission's
partition and resources.

## Limitations

One seed per degree. `m = 2` changed hardware during training: it ran on four H100
SXM until a preemption at about update 14,570, then resumed from the update-14,500
checkpoint on four B300. The two degrees ran on different backends (fused for
`m = 2`, blocked for `m = 4`); both evaluate `q1^2 + q2^2` directly with FP32
accumulation and pass the same kernel tests, but their BF16 training trajectories
are not bitwise comparable.
The `m = 2` comparison with the earlier positive quadratic changes three things at
once (QK-norm, the decay policy, and learning the constant square).

# Nonnegative learned Taylor kernels

Does a row-normalized learned polynomial kernel train once it cannot go negative?
The [QK-norm learned Taylor sweep](../qk_norm_learned_taylor/README.md) learned all
`m + 1` coefficients of `p(s) = sum_r c_r s^r` per layer and head under row
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

**Status: running.** The results block below is written by the analysis module once
both runs have concluded.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. The
[provenance manifest](provenance/manifest.json) is the authoritative record of the
source commit, frozen execution set, acceptance, execution, and artifact locations.
The packed corpus is under `/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Results

<!-- BEGIN GENERATED RESULTS -->
Pending: the runs have not concluded.
<!-- END GENERATED RESULTS -->

## Scientific recipe

Each [recipe](configs/) is the corresponding normalized recipe of the
[QK-norm learned Taylor sweep](../qk_norm_learned_taylor/configs/) with
`attention_coefficient_parameterization: sum_of_squares` added; a configuration test
enforces that nothing else differs except output locations. The backbone is the
repository's 200M model: 10 layers, width 1024, 16 heads, `d_K = d_V = 64`, untied
embeddings, seed 0, packed FineWeb in the fixed order, global batch 256, AdamW
with peak/minimum learning rates `3e-4`/`3e-5`, 2,000 warmup updates, selective
weight decay (`exclude_embeddings_norms_biases_v1`), QK-norm with per-head log scales
from `log(sqrt(d_K)) = log 8`, **19,074 updates and 10,000,269,312 prediction
tokens**. Evaluation uses the first 1,792 held-out sequences every 1,000 updates and
at completion. The two runs use different backends for the same mathematics: `m = 4`
runs on the scaffold's blocked reference, the only backend for row-normalized learned
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
| Exact recipes | `configs/` |
| Committed metrics, metadata, and completion or failure records | `runs/<run>/` |
| Summary and figure | `summary.csv`, `figures/learning_curves.{png,pdf}` |
| Canonical manifest | `provenance/manifest.json` |
| Validation-best checkpoints (ignored) | `checkpoints/<run>/<step>/` |
| Slurm logs (ignored) | `logs/<run>/slurm-<job_id>.out` |

## Reproduction

Regenerate the summary, figure, and results block from the committed runs; no GPU,
dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.nonnegative_learned_taylor.adapter.analysis.report
```

The runs were submitted from this experiment's worktree root with the shared Kennel
launcher, which checks the frozen execution set against the launch commit, runs
acceptance (CPU tests, distributed CPU tests, GPU tests of the recipe's kernels at
the production context, initialization, a 20-update pilot, and an exact resume
audit) in the allocation, then trains; the manifest records each submission's
partition and resources.

## Limitations

One seed per degree. The two degrees ran on different backends (fused for `m = 2`,
blocked for `m = 4`); both evaluate `q1^2 + q2^2` directly with FP32 accumulation and pass the same kernel
tests, but their BF16 training trajectories are not bitwise comparable.
The `m = 2` comparison with the earlier positive quadratic changes three things at
once (QK-norm, the decay policy, and learning the constant square).

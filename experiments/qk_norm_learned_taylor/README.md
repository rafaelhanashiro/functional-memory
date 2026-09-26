# Learned Taylor kernels over QK-norm scores

Do learned polynomial attention kernels train better when their scores are bounded
cosines with a learned scale, and does bounding the scores let the row-normalized
signed kernels survive? This sweep trains degrees **m = 1, 2, 3, 4** with

```text
s_ij = exp(gamma_{l,h}) · <q_i, k_j> / (||q_i|| ||k_j||),   p_{l,h}(x) = sum_{r <= m} c_{l,h,r} x^r
```

in two forms: **unnormalized**, `y_i = sum_{j <= i} p_{l,h}(s_ij) v_j`, and
**normalized**, `y_i = sum_{j <= i} p_{l,h}(s_ij) v_j / sum_{j <= i} p_{l,h}(s_ij)`
with a signed row sum. All `m + 1` coefficients are learned independently for every
layer `l` and head `h` from the Taylor values `1/r!`, and one log scale `gamma_{l,h}`
is learned per layer and head from `log(sqrt(d_K)) = log 8`. Eight runs.

The [learned-coefficient unnormalized Taylor sweep](../learned_unnormalized_taylor/README.md)
showed that learning the coefficients helps degrees 2–4 but leaves every polynomial
well behind the exponential, with scores reaching 17–19; the
[QK-norm softmax baseline](../qk_norm_attention/README.md) showed that bounded cosine
scores with a learned scale slightly beat standard softmax. With `|s_ij| <= exp(gamma)`
a degree-`m` polynomial's approximation error to the exponential is uniformly bounded,
so this sweep measures how much of the polynomial kernels' gap closes when the scores
are bounded and the model must sharpen through the scale alone. The normalized
variants are the first row-normalized learned polynomials in the repository: the fixed
odd Taylor kernels failed at updates 9 and 17 and the normalized learned quadratic at
update 4,187 when their row sums approached zero; whether bounded scores prevent that
is part of the question.

**Status: running.** Requested 2026-09-26 from Kennel and submitted the same day;
see the [provenance manifest](provenance/manifest.json) for the launch commit, jobs,
and progress.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. While the runs execute, their
records, logs, and checkpoints live in this experiment's worktree
(`worktrees/qk_norm_learned_taylor/experiments/qk_norm_learned_taylor/`); at
completion the retained artifacts move to the same paths in the primary checkout.
The [provenance manifest](provenance/manifest.json) is the authoritative record of
sources, configurations, runs, and artifact locations. The packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Scientific recipe

Each unnormalized [run recipe](configs/) is the
[learned-coefficient unnormalized Taylor recipe](../learned_unnormalized_taylor/configs/)
of the same degree with `qk_norm: true`, `qk_norm_sharing: per_head`, the default
initial scale `sqrt(d_K)`, and `qk_norm_eps: 1e-6` added; each normalized recipe is
its unnormalized twin with `attention_normalization: true`. Configuration tests
enforce that nothing else differs except the explicit backend and output locations.
The backbone is the repository's 200M sweep model: 10 layers, width 1024, 16 full-MHA
heads, `d_K = d_V = 64`, SwiGLU width 3008, untied 32,000-token embeddings, RoPE base
10,000, seed 0, packed FineWeb in the fixed order, global batch 256 with four
accumulation steps, AdamW with peak/minimum learning rates `3e-4`/`3e-5` and 2,000
warmup updates, **19,074 updates and 10,000,269,312 prediction tokens**. Evaluation
uses the first 1,792 held-out sequences every 1,000 updates and at completion. The
primary endpoint is terminal held-out perplexity; a numerical failure is recorded with
its stopping update and last valid metrics.

The coefficients form one FP32 array of shape `[10, 16, m + 1]` and the log scales
one of shape `[10, 16, 1]`; both use the backbone learning-rate schedule and global
gradient clipping and receive no weight decay. All runs train under the canonical
selective weight-decay policy (`exclude_embeddings_norms_biases_v1`). Initial
backbone weights are bit-identical to the 200M softmax control's, and at
initialization each model computes its fixed-Taylor kernel over cosine scores.

| Run | Row normalization | Kernel at init | Backend |
| --- | --- | --- | --- |
| `200m-qk-m00{1,2,3,4}-unnormalized-seed0` | none | `p_m(8 cos)` | fused Pallas/Triton (`fused_learned_polynomial`) |
| `200m-qk-m00{1,2,3,4}-normalized-seed0` | signed row sum | `p_m(8 cos) / rowsum` | blocked reference (no fused kernel exists) |

**Expectations to test.** With `exp(gamma) = 8` at initialization, scores span
`[-8, 8]`, where `p_1` and `p_3` are negative below `-1` and `-1.6`. The normalized
odd degrees therefore start with signed weights and may fail early, as their fixed
counterparts did; whether the learned constant or the scale rescues them is the
result. Even degrees start positive and can acquire real roots only by learning.

Controls, all seed 0 with the same backbone and budget (terminal perplexity;
"policy" is the weight-decay policy):

| Kernel | Policy | m=1 | m=2 | m=3 | m=4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Unnormalized, learned coefficients, no QK-norm ([record](../learned_unnormalized_taylor/README.md)) | all parameters | 20.7136 | 20.0594 | 18.5316 | 20.1398 |
| Unnormalized, fixed Taylor, no QK-norm | all parameters | 20.7701 | 23.5057 | 20.2397 | 21.4619 |
| Normalized, fixed Taylor, no QK-norm | all parameters | failed (update 9) | 16.2810 | failed (update 17) | 14.9836 |

QK-norm softmax ends at 14.4574, the canonical-policy softmax at 14.6655, the
original softmax at 14.6942, and the unnormalized exponential at 17.4583. The
canonical policy changed the softmax control by only 0.03, so the older controls
remain informative, but the comparison across policies is not exact.

## Implementation

The [shared Transformer scaffold](../../scaffolds/transformer/README.md) selects these
mechanisms with `attention_coefficients: learned`, an integer `attention_degree`, and
`qk_norm: true`, with `attention_normalization` choosing the form. The attention
module unit-normalizes Q and K per head in FP32 before RoPE and passes
`q' = exp(gamma) sqrt(d_K) q̂` and `k' = k̂` to the kernel, whose `q'·k' / sqrt(d_K)`
is the cosine score. Unnormalized runs use the fused Pallas/Triton learned-polynomial
kernel, whose custom backward returns the coefficient gradients. Normalized runs use
the blocked reference (`attention/learned_polynomial.py`), which evaluates the kernel
by Horner's rule, divides by the signed row sum, and poisons a row only when its sum
is exactly zero or nonfinite; `attention_backend: blocked` is stated in those recipes
because no fused kernel exists. Nonfinite scores, kernel values, outputs, gradients,
or updated parameters stop training with a numerical-failure record. Every
validation records the coefficients and `gamma` of every layer and head in
`metrics.jsonl`; W&B receives their mean, minimum, and maximum.

## Validation

From the experiment worktree, sync the scaffold environment and run the CPU tests:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_qk_norm.py \
  scaffolds/transformer/tests/test_learned_polynomial.py \
  scaffolds/transformer/tests/test_learned_polynomial_normalized.py \
  experiments/qk_norm_learned_taylor/adapter/tests
```

CPU tests compare the normalized and unnormalized learned kernels, alone and
composed with QK-norm, against independent FP64 references for outputs and the Q,
K, V, coefficient, and `gamma` gradients at degrees 1–4; check parity with the fixed
Taylor kernels at the Taylor initialization; check the signed-row-sum semantics
(negative sums valid, an exactly zero sum poisons only its row), backbone identity,
parameter counts, backend selection, diagnostics, and training with checkpoint
resume. GPU tests, run by the launcher's acceptance in the production allocation,
cover the fused learned-polynomial and unnormalized kernels, the cosine operands
through every kernel, and the blocked normalized kernel at the full 2,048-token BF16
context against FP64 on a causal prefix. Acceptance then verifies the initialization
against the softmax control, runs 20 full-recipe updates with validation, and
checks a one-update checkpoint resume.

## Kennel launch

Each run uses four B300s or four H100s in the low-priority `sweep` partition,
submitted from the worktree root at the launch commit recorded in the manifest:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
for m in 1 2 3 4; do for variant in unnormalized normalized; do
  RUN="200m-qk-m00${m}-${variant}-seed0"
  mkdir -p "experiments/qk_norm_learned_taylor/logs/$RUN"
  sbatch --job-name="qk-lt-m${m}-${variant:0:1}" \
    --output="experiments/qk_norm_learned_taylor/logs/$RUN/slurm-%j.out" \
    scaffolds/transformer/launchers/kennel.sbatch experiments.qk_norm_learned_taylor.adapter.execution.acceptance \
    "experiments/qk_norm_learned_taylor/configs/$RUN.yaml" "$(git rev-parse HEAD)" 4 B300,H100
done; done
```

Keep the frozen execution set (scaffold code, lock, `configs/`, and
`adapter/execution/`) unchanged while jobs can run or requeue. The normalized runs
are several times slower per update than the fused unnormalized runs; the pilot
records each allocation's step time and projected hours.

## Artifacts

Each run keeps its validation-best checkpoint at
`experiments/qk_norm_learned_taylor/checkpoints/<run>/<step>/` and one Slurm log at
`experiments/qk_norm_learned_taylor/logs/<run>/`, on Kennel under the artifact root
and ignored by Git; the manifest records every file's SHA256 at completion. W&B
mirrors the runs in the `qk_norm_learned_taylor` group. A clone supplies the committed
run records, recipes, and this README, not checkpoints, logs, or datasets.

## Limitations

One seed per run and one initial scale, `sqrt(d_K)`, shared with the softmax
baseline rather than tuned per kernel. The learned-coefficient and fixed controls
trained under the earlier all-parameter weight decay. Normalized runs use the
blocked reference backend, so their throughput is not comparable with the fused
runs. Kernel and scale statistics come from one held-out diagnostic sequence per
validation.

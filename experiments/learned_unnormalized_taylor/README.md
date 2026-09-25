# Learned-coefficient unnormalized Taylor attention

Does unnormalized polynomial attention train better when every coefficient of its
kernel, including the constant, is learned? The
[unnormalized Taylor runs](../unnormalized_taylor_attention/README.md) fix
`p_m(x) = sum_{r<=m} x^r / r!`, so the constant term contributes a plain prefix sum
`sum_{j<=i} v_j` that grows with position; the model can shrink its relative weight
only by inflating Q/K, and those runs reached scores of 50–700. This sweep trains
degrees **m = 1, 2, 3, 4** with

```text
y_i = sum_{j <= i} p_{l,h}(s_ij) v_j,   s_ij = q_i · k_j / sqrt(d_K),   p_{l,h}(x) = sum_{r <= m} c_{l,h,r} x^r
```

where all `m + 1` coefficients are learned independently for every layer `l` and
head `h`, start at the Taylor values `1/r!`, and are unrestricted (signed kernels
are allowed). There is no row denominator.

**Status: in preparation.** Requested 2026-09-25 from Kennel.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. While the runs execute, their
records, logs, and checkpoints live in this experiment's worktree
(`worktrees/learned_unnormalized_taylor/experiments/learned_unnormalized_taylor/`);
at completion the retained artifacts move to the same paths in the primary
checkout. The [provenance manifest](provenance/manifest.json) is the authoritative
record of sources, configurations, runs, and artifact locations. The packed corpus
is under `/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Scientific recipe

Each [run recipe](configs/) is the fixed unnormalized Taylor recipe of the same
degree with `attention_coefficients: learned` and
`attention_coefficient_sharing: per_head` added; a configuration test enforces
that nothing else differs except output locations. The backbone is the
repository's 200M sweep model: 10 layers, width 1024, 16 full-MHA heads,
`d_K = d_V = 64`, SwiGLU width 3008, untied 32,000-token embeddings, RoPE base
10,000, seed 0, packed FineWeb in the fixed order, global batch 256 with four
accumulation steps, AdamW with peak/minimum learning rates `3e-4`/`3e-5` and 2,000
warmup updates, **19,074 updates and 10,000,269,312 prediction tokens**. Evaluation
uses the first 1,792 held-out sequences every 1,000 updates and at completion.
The primary endpoint is terminal held-out perplexity.

The coefficients form one FP32 array of shape `[10, 16, m + 1]` (160·(m+1) values:
320, 480, 640, 800), use the backbone learning-rate schedule and global gradient
clipping, and receive no weight decay. Initial backbone weights are bit-identical
to the 200M softmax and fixed-Taylor controls, and at initialization each model
computes the same function as its fixed-Taylor control.

Controls, all seed 0 with the same backbone and budget:

| Kernel | m=1 | m=2 | m=3 | m=4 |
| --- | ---: | ---: | ---: | ---: |
| Unnormalized, fixed Taylor coefficients | 20.7701 | 23.5057 | 20.2397 | 21.4619 |
| Normalized, fixed Taylor coefficients | failed (update 9) | 16.2810 | failed (update 17) | 14.9836 |

The unnormalized exponential ends at 17.4583 and softmax at 14.6942. The normalized
learned quadratic `1 + b s + c s^2` with per-head coefficients failed at update 4,187
after its kernel acquired real roots ([learned quadratic attention](../learned_quadratic_attention/README.md)).

Interpretation notes. A per-head overall scale of the kernel is interchangeable
with the scale of that head's value or output projection, which do receive weight
decay, so coefficient magnitudes may drift; compare scale-free ratios such as
`c_r / c_0`. For m = 1, rescaling Q/K and V already reaches every `c_0 + c_1 s`
with `c_0 != 0`, so learning changes the optimization path and admits `c_0 = 0`
rather than enlarging the kernel family; for m ≥ 2 it also frees `m - 1` shape
ratios fixed by the Taylor values. With this schedule Adam can move each raw
coefficient by at most about 3.1 over the run.

## Implementation

The [shared Transformer scaffold](../../scaffolds/transformer/README.md) selects
this mechanism with `attention_normalization: false`, an integer
`attention_degree >= 1`, and `attention_coefficients: learned`. The unnormalized
mechanism declares the `polynomial_coefficients` array (initialized at `1/r!`,
excluded from weight decay). The blocked reference evaluates the kernel by Horner's
rule per head and supplies diagnostics. The fused Pallas/Triton kernel in
`attention/backends/fused_learned_polynomial.py` uses the tiles, products, and FP32
accumulation of the fixed unnormalized kernel; its custom backward returns
`dS = p'(S) * (dY Vᵀ)` and each head's coefficient gradient
`sum_{i>=j} S_ij^r (dY_i · V_j)`, reduced from per-query-tile partials without
atomics. Nonfinite scores, kernel values, outputs, gradients, or updated
parameters stop training with a numerical-failure record; signed kernels and zero
or negative row sums are valid and measured only. Every validation records the
coefficients of every layer and head in `metrics.jsonl`; W&B receives their mean,
minimum, and maximum.

## Validation

From the experiment worktree, sync the scaffold environment and run the CPU tests:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_learned_polynomial.py \
  scaffolds/transformer/tests/test_learned_polynomial_training.py \
  experiments/learned_unnormalized_taylor/adapter/tests
```

CPU tests compare blocked outputs and Q/K/V and coefficient gradients for degrees
1–4, shared and per head, with an independent FP64 power sum; run the fused kernel's
programs in the Pallas interpreter against the same oracle; check Taylor-initialized
parity with the fixed kernel, backbone identity, parameter counts, the weight-decay
exclusion, gradient accumulation of shared coefficients, signed-kernel diagnostics,
and training with checkpoint resume. GPU tests, run by the launcher's acceptance in
the production allocation, compile the fused kernel and compare its forward and
all four VJPs with the blocked reference and FP64 at odd widths, tile boundaries,
full 2,048-token BF16 context, scores near 50, and against the fixed-coefficient
kernel at Taylor values. Acceptance then verifies the initialization, runs 20
full-recipe updates with validation, and checks a one-update checkpoint resume.

## Kennel launch

Each degree runs on four B300s or four H100s in the low-priority `sweep` partition,
submitted from the worktree root at the launch commit:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
for m in 1 2 3 4; do
  RUN="200m-m00${m}-seed0"
  mkdir -p "experiments/learned_unnormalized_taylor/logs/$RUN"
  sbatch --job-name="lut-m$m" --output="experiments/learned_unnormalized_taylor/logs/$RUN/slurm-%j.out" \
    scaffolds/transformer/launchers/kennel.sbatch experiments.learned_unnormalized_taylor.adapter.execution.acceptance \
    "experiments/learned_unnormalized_taylor/configs/$RUN.yaml" "$(git rev-parse HEAD)" 4 B300,H100
done
```

Keep the frozen execution set (scaffold code, lock, `configs/`, and
`adapter/execution/`) unchanged while jobs can run or requeue.

## Limitations

Each degree has one seed, and only per-head sharing is run. Coefficients use the
backbone learning rate, which bounds how far they can move.

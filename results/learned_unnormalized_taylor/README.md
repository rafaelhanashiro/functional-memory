# Learned-coefficient unnormalized Taylor attention

[Selected runs and source hashes](selection.yaml) · [Run index](../../experiments/learned_unnormalized_taylor/runs.csv) · [Execution guide](../../experiments/learned_unnormalized_taylor/README.md)

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

**Status: completed.** All four degrees trained the full 10B tokens without a
numerical failure. Learning the coefficients left degree 1 unchanged (PPL 20.71
versus 20.77) and improved degrees 2–4 substantially: m=2 23.51 → 20.06, m=3
20.24 → 18.53, m=4 21.46 → 20.14. The learned even-degree kernels became signed,
and the models stopped inflating their scores.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; retained checkpoints and logs
are under this directory there (see [Artifacts](#artifacts)). The
[study record](../../experiments/learned_unnormalized_taylor/sweep.yaml) is the authoritative record of
sources, configurations, runs, and artifact locations. The packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Results

<!-- BEGIN GENERATED RESULTS -->
Terminal held-out perplexity at 19,074 updates (10,000,269,312 prediction tokens), seed 0, against the fixed-coefficient unnormalized Taylor control of the same degree. Coefficient and kernel statistics are the terminal validation's 160 layer-heads. Kernel minima are over the scores observed on the fixed diagnostic sequence, relative to each head's constant, for even degrees (odd polynomials are unbounded below). Negative kernel values are the median layer's fraction of causal entries on that sequence.

| m | Learned PPL | Fixed PPL | Change | Median c0 | Median min p(s)/c0 | Heads with negative kernel | Negative kernel values | Largest abs score (fixed) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.7136 | 20.7701 | -0.0564 | 0.912 | — | 160/160 | 0.49 | 88.7 (92.0) |
| 2 | 20.0594 | 23.5057 | -3.4462 | 0.817 | -0.015 | 154/160 | 0.60 | 18.6 (47.3) |
| 3 | 18.5316 | 20.2397 | -1.7081 | 0.669 | — | 158/160 | 0.44 | 19.1 (35.0) |
| 4 | 20.1398 | 21.4619 | -1.3221 | 0.921 | -0.300 | 160/160 | 0.59 | 17.4 (34.9) |

[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf) · [Kernel usage](kernel_usage.pdf) · [Learned kernels](learned_kernels.pdf)
<!-- END GENERATED RESULTS -->

**Degree 1 is unchanged, as its kernel family predicts.** Rescaling Q/K and V already
reaches every linear kernel `c_0 + c_1 s` with `c_0 != 0`, so learning the
coefficients could only change the optimization path or reach `c_0 = 0`. It did
neither: the median constant ended at 0.91 against a linear coefficient of 0.93,
essentially Taylor's 1:1, and the learning curve, scores, and signed-weight
fraction track the fixed run throughout.

**Even degrees became signed.** Taylor coefficients make degrees 2 and 4 strictly
positive. With learned coefficients, 154/160 degree-2 heads and every degree-4 head
take negative values over the observed scores, and about 60% of causal kernel
values on the diagnostic sequence are negative at the end (0 for the fixed runs).
At degree 2 each head converged to nearly a perfect square: the scale-free shape
ratio `c_1^2 / (c_0 c_2)` has median 4.06 (Taylor 2; real roots appear above 4),
the kernel's minimum is about zero (median −0.015 of the constant, Taylor +0.5),
and its vertex sits at `s = −1.41` in almost every head (interquartile range
−1.43 to −1.39). The [learned kernels](learned_kernels.pdf) of degrees 2–4 reach
zero or dip below it at moderately negative scores (about −1 to −4), where Taylor's
kernels keep a positive floor, and grow as fast or faster for positive scores. At degree 3 the signed fraction first falls and then rises above the fixed
run's. The degree-2 switch to signed weights happens between updates 4,000 and
11,000, when its learning curve separates from the control's.

**Scores stopped inflating.** At the end the largest score magnitude was 18.6, 19.1,
and 17.4 for degrees 2–4 against 47.3, 35.0, and 34.9 with fixed coefficients, and
the largest row weight sum fell from 24,600 to 3,490 (m=2), 34,100 to 2,590 (m=3),
and 177,000 to 16,500 (m=4). This is consistent with the models suppressing keys
through the kernel's zero instead of by enlarging Q/K, though these runs do not
isolate that mechanism.

**The coefficients moved modestly.** Median constants ended at 0.91, 0.82, 0.67,
and 0.92 for m = 1–4, and 99% of coefficients changed by less than about 0.6 from their initial
values (1.1 for degree-3 constants). Only three heads moved a coefficient by more
than 1; the largest change, 2.21, took one degree-4 linear coefficient to −1.21,
within the ≈3.1 that Adam could move a coefficient over this schedule. Only a few degree-3
constants (1%) turned negative. Because a head's overall kernel scale is
interchangeable with its value projection, the shape ratios above, not raw values,
carry the result.

Degree 3 remains the best unnormalized polynomial (18.53) but trails the
unnormalized exponential (17.46), and every unnormalized kernel stays well behind
normalized attention (softmax 14.67; normalized degree-4 Taylor 14.98). Unlike the
normalized learned quadratic, which failed once its kernel acquired real roots,
signed kernels are harmless here because nothing divides by the row sum.

## Scientific recipe

Each [run recipe](../../experiments/learned_unnormalized_taylor/runs/) is the fixed unnormalized Taylor recipe of the same
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
| Normalized, fixed Taylor coefficients, QK-norm and selective decay | failed (update 31) | 16.5107 | failed (update 419) | 14.8963 |

The normalized fixed Taylor row was retrained under the scaffold's current defaults,
which the runs here predate; its comparison with them includes QK-norm and the
weight-decay policy as well as normalization.

The unnormalized exponential ends at 17.4583 and softmax at 14.6655. The normalized
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
  experiments/learned_unnormalized_taylor/tests
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

Regenerate the summary, figures, and results block from the committed runs (no GPU,
dataset, or checkpoint needed):

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.learned_unnormalized_taylor.report
```

## Kennel launch
Use the [execution guide](../../experiments/learned_unnormalized_taylor/README.md) to create a fresh run and submit it from an isolated worktree.

## Artifacts

Each run keeps its validation-best checkpoint, which is also its terminal one, at
`experiments/learned_unnormalized_taylor/runs/<run_id>/checkpoints/19074/` (about 2.2 GB
each) and one Slurm log at `experiments/learned_unnormalized_taylor/runs/<run_id>/training.log`,
both on Kennel under the artifact root and ignored by Git; the manifest records
every file's SHA256. W&B mirrors the runs in the `learned_unnormalized_taylor` group;
the manifest records each run id. A clone supplies the committed runs, recipes,
figures, and summary, not checkpoints, logs, or datasets.

## Limitations

Each degree has one seed, and only per-head sharing was run. Coefficients used the
backbone learning rate, which bounds how far they can move; three heads moved a
coefficient by more than 1, at most 2.21 of the ≈3.1 possible. Kernel statistics come from one held-out diagnostic sequence
per validation. Degree 2 trained on H100s and the others on B300s, as did their
controls on mixed hardware. The softmax baseline trained under the selective
weight-decay policy, unlike the runs here; that difference (about 0.03 PPL in the
baseline) is far below the differences reported here.

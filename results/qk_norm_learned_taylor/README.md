# Learned Taylor kernels over QK-norm scores

[Selected runs and source hashes](selection.yaml) · [Run index](../../experiments/qk_norm_learned_taylor/runs.csv) · [Execution guide](../../experiments/qk_norm_learned_taylor/README.md)

Do learned polynomial attention kernels train better when their scores are bounded
cosines with a learned scale, and does bounding the scores let the normalized
signed kernels survive? This sweep trained degrees **m = 1, 2, 3, 4** with

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
variants are the first normalized learned polynomials in the repository: the fixed
odd Taylor kernels and the normalized learned quadratic (at update 4,187) failed
numerically when their row sums approached zero; whether bounded scores prevent that
was part of the question.

**Status: completed.** The four unnormalized runs trained the full 10B tokens without
a numerical failure and every degree improved on its control without QK-norm: m=1
20.71 → 20.28, m=2 20.06 → 17.60, m=3 18.53 → 16.33, m=4 20.14 → 16.38. Degrees 3
and 4 now coincide within 0.05, below the unnormalized exponential (17.46) and 1.9
above QK-norm softmax (14.46). All four normalized runs failed numerically, at
updates 27 (m=1), 1,857 (m=2), 318 (m=3), and 1,456 (m=4), each once its kernel was
negative inside the bounded score range and a row sum crossed zero.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; retained checkpoints and logs are
under this directory there (see [Artifacts](#artifacts)). The
[study record](../../experiments/qk_norm_learned_taylor/sweep.yaml) is the authoritative record of
sources, configurations, runs, and artifact locations. The packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Results

<!-- BEGIN GENERATED RESULTS -->
Held-out perplexity at the terminal validation (19,074 updates, 10,000,269,312 prediction tokens; seed 0) or, for a numerical failure, the last valid validation and the stopping update. Controls: the learned-coefficient unnormalized Taylor run of the same degree without QK-norm (all-parameter decay) and the QK-norm softmax baseline (14.4574). Kernel statistics are the last validation's 160 layer-heads over the scores observed on the fixed diagnostic sequence; negative kernel values are the median layer's fraction of causal entries; scales are exp(γ) over the 160 heads.

| Run | Status | Terminal PPL | Learned, no QK-norm | Δ | Δ vs QK-norm softmax | Median c0 | Heads with negative kernel | Negative kernel values | exp(γ) median (min–max) | Largest abs score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| m=1 unnormalized | complete | 20.2790 | 20.7136 | -0.4347 | +5.8216 | 0.954 | 160/160 | 0.50 | 7.8 (6.1–17.8) | 10.3 |
| m=1 normalized | failed at update 27 before the first validation | — | 20.7136 | — | — | — | — | — | — (—–—) | — |
| m=2 unnormalized | complete | 17.5990 | 20.0594 | -2.4605 | +3.1416 | 0.820 | 159/160 | 0.64 | 7.4 (6.3–17.0) | 7.1 |
| m=2 normalized | failed at update 1,857; last valid 128.20 at 1,000 | — | 20.0594 | — | — | 0.944 | 0/160 | 0.00 | 8.5 (8.1–8.6) | 8.2 |
| m=3 unnormalized | complete | 16.3301 | 18.5316 | -2.2015 | +1.8727 | 0.742 | 159/160 | 0.46 | 8.4 (7.0–12.4) | 7.8 |
| m=3 normalized | failed at update 318 before the first validation | — | 18.5316 | — | — | — | — | — | — (—–—) | — |
| m=4 unnormalized | complete | 16.3840 | 20.1398 | -3.7558 | +1.9266 | 0.885 | 159/160 | 0.63 | 9.1 (7.0–12.2) | 9.5 |
| m=4 normalized | failed at update 1,456; last valid 296.85 at 1,000 | — | 20.1398 | — | — | 0.972 | 97/160 | 0.11 | 8.3 (8.0–8.5) | 8.0 |

[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf) · [Learned kernels](learned_kernels.pdf) · [Scales and signs](scales_and_signs.pdf)
<!-- END GENERATED RESULTS -->

**Bounded scores lift every unnormalized learned polynomial, most for the higher
degrees.** Against the same kernels without QK-norm (which trained under the earlier
all-parameter weight decay; the policy change moved the softmax control by only
0.03), the gains are 0.43 (m=1), 2.46 (m=2), 2.20 (m=3), and 3.76 (m=4). Degrees 3
and 4 end within 0.05 of each other, so beyond degree 3 the polynomial's
approximation quality is no longer the limit; the remaining 1.9-perplexity gap to
QK-norm softmax is the missing normalization. The QK-norm cubic and quartic also
beat the unnormalized exponential without QK-norm (17.46), which makes an
unnormalized exponential with QK-norm the natural next control. All four runs trail
QK-norm softmax heavily early (perplexity 204–244 at update 1,000 against 93) and
close most of the gap by update 10,000 (19.3 for degrees 3 and 4 against 16.3).

**Why: the kernels keep their shape but lose their tail.** Terminal negative-kernel
fractions are the same with and without QK-norm (0.50, 0.64, 0.46, 0.63 against
0.49, 0.60, 0.44, 0.59 for degrees 1–4), so learning produced the same signed
shapes: degree 1 stays essentially Taylor (median coefficients 0.95, 0.98), degree 2
converges to a near-perfect square (shape ratio `c_1^2 / (c_0 c_2)` median 4.04,
159/160 heads at or past a double root), and degrees 3 and 4 stay close to Taylor
(medians 0.74, 1.09, 0.53, 0.085 and 0.89, 1.00, 0.53, 0.18, 0.029). What changed is
the score range the kernels see: [−4.8, 10.3] with QK-norm against [−16, 19] without
(and ±85 for degree 1). At |s| ≈ 18 the quartic term is twenty times what it is at
|s| ≈ 8, so without QK-norm the largest kernel values reached 228, 805, and 6,470 for
degrees 2–4 and the largest row weight sums 3,490, 2,590, and 16,500; with QK-norm
they are 57, 167, and 644 and 727, 712, and 2,320. A uniform rescaling of the
scores is not what changed: learned coefficients absorb it at every degree
(`s -> lambda s` is undone by `c_r -> c_r / lambda^r`), which is also why the learned
scales barely moved from 8 (medians 7.8, 7.4, 8.4, 9.1; ranges 6.1–17.8) although
Adam could have raised them by a factor of 22. What changed is the shape of the
score distribution, whose tail the `s^m` term amplifies: the same kernels without
QK-norm inflated Q and K until scores reached ±17–19, whereas cosines of unit
vectors have no such tail. Degree 1 is the control for that reading: its output
is linear in the scores, so it amplifies no tail, and it gained the least (0.43,
close to the 0.21 that softmax itself gains from QK-norm) even though its scores
shrank the most. QK-norm also removes per-token key magnitudes, and this sweep
cannot separate that from the bounded range; an unnormalized learned kernel with
QK-norm at a large initial scale would.

**The normalized kernels fail by the same mechanism as before, and bounding the
scores does not help.** Normalization divides by the signed sum
`sum_j p(s_ij)`; once `p` is negative somewhere in the score range, a row sum can pass
through zero, the weights for that row blow up, and the loss goes nonfinite (the
scaffold adds no epsilon or clipping). The odd degrees are negative from the start:
with Taylor coefficients `1 + s` is negative below −1 and the cubic below −1.6,
while the cosine scores already span about ±5.3 at initialization. On the
diagnostic sequence before any training, degree 1 had 16% negative weights, a
smallest absolute row sum of 0.008, and a cancellation ratio above 12,000; degree 3
had 5% negative weights and a smallest row sum of 0.005. They failed at updates 27
and 318; the fixed-coefficient kernels trained with the same QK-norm recipe fail at
31 and 419 ([odd Taylor attention](../odd_taylor_attention/README.md)). The even degrees start
strictly positive (kernel minima 0.50 and 0.27 on the sequence) and were learned
into failure: the degree-2 shape ratio drifted from Taylor's 2 to a median of 3.31 at
update 1,500 with one head past 4 (a root near `s = −1.5`), the training loss stayed
flat at 4.3–4.4 nats through update 1,850, and the run went nonfinite at 1,857; by
update 1,000 the degree-4 kernels were negative inside the score range in 97/160
heads with a smallest row sum of 0.012, and the run failed at 1,456. The roots sit
at `s ≈ −1` to `−1.6`, well inside `[−exp(gamma), exp(gamma)] ≈ [−9, 9]`; keeping the
odd kernels positive would need a scale below their positivity radius, which
flattens attention. The unnormalized twins carry the same signed kernels harmlessly.
A positivity-preserving parametrization, not a tighter score bound, is the route to
normalized polynomial kernels.

## Scientific recipe

Each unnormalized [run recipe](../../experiments/qk_norm_learned_taylor/runs/) is the
[learned-coefficient unnormalized Taylor recipe](../../experiments/learned_unnormalized_taylor/runs/)
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
gradient clipping and receive no weight decay. All runs trained under the canonical
selective weight-decay policy (`exclude_embeddings_norms_biases_v1`). Initial
backbone weights are bit-identical to the 200M softmax control's, which every
allocation's acceptance verified, and at initialization each model computes its
fixed-Taylor kernel over cosine scores.

| Run | Normalization | Kernel at init | Backend |
| --- | --- | --- | --- |
| `200m-qk-m00{1,2,3,4}-unnormalized-seed0` | none | `p_m(8 cos)` | fused Pallas/Triton (`fused_learned_polynomial`) |
| `200m-qk-m00{1,2,3,4}-normalized-seed0` | signed row sum | `p_m(8 cos) / rowsum` | blocked reference (no fused kernel exists) |

Controls, all seed 0 with the same backbone and budget (terminal perplexity;
"policy" is the weight-decay policy):

| Kernel | Policy | m=1 | m=2 | m=3 | m=4 |
| --- | --- | ---: | ---: | ---: | ---: |
| Unnormalized, learned coefficients, no QK-norm ([record](../learned_unnormalized_taylor/README.md)) | all parameters | 20.7136 | 20.0594 | 18.5316 | 20.1398 |
| Unnormalized, fixed Taylor, no QK-norm | all parameters | 20.7701 | 23.5057 | 20.2397 | 21.4619 |
| Normalized, fixed Taylor, with QK-norm ([even](../key_dimension_taylor/README.md), [odd](../odd_taylor_attention/README.md)) | selective | failed (update 31) | 16.5107 | failed (update 419) | 14.8963 |

QK-norm softmax ends at 14.4574 and the repository's softmax baseline under the
canonical policy at 14.6655 ([key_dimension_taylor](../key_dimension_taylor/README.md));
the unnormalized exponential without QK-norm ends at 17.4583. The weight-decay
policy moves softmax by about 0.03 PPL, so controls trained with decay on all
parameters remain informative, but comparisons across policies are not exact.

## Implementation

The [shared Transformer scaffold](../../scaffolds/transformer/README.md) selects these
mechanisms with `attention_coefficients: learned`, an integer `attention_degree`, and
`qk_norm: true`, with `attention_normalization` choosing the form. The attention
module unit-normalizes Q and K per head in FP32 before RoPE and passes
`q' = exp(gamma) sqrt(d_K) q̂` and `k' = k̂` to the kernel, whose `q'·k' / sqrt(d_K)`
is the cosine score. Unnormalized runs use the fused Pallas/Triton learned-polynomial
kernel, whose custom backward returns the coefficient gradients. Normalized runs use
the blocked reference (`attention/learned_polynomial.py`, added for this sweep),
which evaluates the kernel by Horner's rule, divides by the signed row sum, and
poisons a row only when its sum is exactly zero or nonfinite; `attention_backend:
blocked` is stated in those recipes because no fused kernel exists. Nonfinite scores,
kernel values, outputs, gradients, or updated parameters stop training with a
numerical-failure record. Every validation records the coefficients and `gamma` of
every layer and head in `metrics.jsonl`; W&B receives their mean, minimum, and
maximum.

## Validation

From a checkout with the scaffold environment synced, run the CPU tests:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_qk_norm.py \
  scaffolds/transformer/tests/test_learned_polynomial.py \
  scaffolds/transformer/tests/test_learned_polynomial_normalized.py \
  experiments/qk_norm_learned_taylor/tests
```

CPU tests compare the normalized and unnormalized learned kernels, alone and
composed with QK-norm, against independent FP64 references for outputs and the Q,
K, V, coefficient, and `gamma` gradients at degrees 1–4; check parity with the fixed
Taylor kernels at the Taylor initialization; check the signed-row-sum semantics
(negative sums valid, an exactly zero sum poisons only its row), backbone identity,
parameter counts, backend selection, diagnostics, and training with checkpoint
resume. GPU tests cover the fused learned-polynomial and unnormalized kernels, the
cosine operands through every kernel, and the blocked normalized kernel at the full
2,048-token BF16 context against FP64 on a causal prefix. Before launch the full
scaffold CPU suite passed (673 tests) and 35 GPU cases passed on one B300 (`akita`,
job 202336). In every production allocation, acceptance passed 202 CPU tests, 5
distributed CPU tests, and 83 GPU tests, matched the initialization to the softmax
control's parameter hash plus the declared coefficients and log scales, ran a
20-update full-recipe pilot on the recipe's backend, and reproduced a one-update
checkpoint resume exactly; this was also the first H100 qualification of QK-norm.

Regenerate the summary, figures, and results block from the committed runs (no GPU,
dataset, or checkpoint needed):

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.qk_norm_learned_taylor.report
```

## Kennel launch
Use the [execution guide](../../experiments/qk_norm_learned_taylor/README.md) to create a fresh run and submit it from an isolated worktree.

## Artifacts

Each completed run keeps its validation-best checkpoint, which is also its terminal
one, at `experiments/qk_norm_learned_taylor/runs/<run_id>/checkpoints/19074/` (about 2.2 GB
each) and one Slurm log at `experiments/qk_norm_learned_taylor/runs/<run_id>/training.log`, both on
Kennel under the artifact root and ignored by Git; the manifest records every file's
SHA256. The failed runs keep their log and their numerical-failure record and last
valid validation in `runs/<run_id>/`; no diagnostic checkpoint is retained (the
degree-2 failure analysis above read the step-1,500 checkpoint before it was
removed). W&B mirrors the runs in the `qk_norm_learned_taylor` group; the manifest
records each run id. A clone supplies the committed run records, recipes, summary,
figures, and this README, not checkpoints, logs, or datasets.

## Limitations

One seed per run and one initial scale, `sqrt(d_K)`, shared with the softmax
baseline rather than tuned per kernel. The learned-coefficient and fixed controls
trained under the earlier all-parameter weight decay. The sweep cannot separate the
score bound from the loss of per-token key magnitudes, since QK-norm imposes both.
Normalized runs used the blocked reference backend, so their throughput is not
comparable with the fused runs, and their failures leave no terminal endpoint.
Kernel and scale statistics come from one held-out diagnostic sequence per
validation; the initialization statistics quoted above were computed on that
sequence from the recipes at seed 0. The unnormalized runs trained on mixed hardware
(three on H100s, one on B300s); earlier softmax controls on different hardware
agreed within 0.0003 nats.

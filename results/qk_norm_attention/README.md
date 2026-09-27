# QK-norm attention

[Selected runs and source hashes](selection.yaml) · [Run index](../../experiments/qk_norm_attention/runs.csv) · [Execution guide](../../experiments/qk_norm_attention/README.md)

How does exact softmax attention train when its scores are bounded cosines with a
learned scale? This record's first run is the **QK-norm softmax baseline**: the
repository's 200M softmax control (`d_K = 64`) with

```text
s_ij = exp(gamma_{l,h}) · <q_i, k_j> / (||q_i|| ||k_j||),   |s_ij| <= exp(gamma_{l,h})
```

in place of `q_i · k_j / sqrt(d_K)`, where one log scale `gamma_{l,h}` is learned per
layer `l` and head `h`, starting at `log(sqrt(d_K)) = log 8`. Everything else,
including the exact softmax and its cuDNN implementation, is the control's. The
cosine removes the magnitudes of queries and keys from the score, so a head can
sharpen or flatten its attention only through `exp(gamma)`; the run measures what
this costs against standard softmax and how far the scales move. Bounded scores
are also the regime in which the repository's polynomial (Taylor) kernels
approximate the exponential, so this baseline anchors later QK-norm polynomial runs.

**Status: completed.** The run trained the full 10B tokens without a numerical
failure and ended at held-out perplexity **14.4574**, against 14.6655 for the
softmax control under the same weight-decay policy (−0.208). It trailed the
control at update 1,000, led from
update 3,000 on, and its lead narrowed from about 0.5 to 0.2 over the second half.
The learned scales grew from 8 to a median of 14.1, rising with depth (layer-1
median 10.1, layers 7–10 about 16–17), and the largest score seen on the diagnostic
sequence was 18.7.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; the retained checkpoint and log
are under this directory there (see [Artifacts](#artifacts)). The
[study record](../../experiments/qk_norm_attention/sweep.yaml) is the authoritative record of
sources, configuration, the run, and artifact locations. The packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Results

<!-- BEGIN GENERATED RESULTS -->
Held-out perplexity at the terminal validation (19,074 updates, 10,000,269,312 prediction tokens; seed 0) against the softmax control: the repository's softmax baseline, the same recipe and weight-decay policy without QK-norm. Scales are the terminal validation's 160 layer-heads; the score bound is the largest exp(γ), and the largest |score| is over the fixed diagnostic sequence.

| Run | Status | Terminal PPL | Best PPL (update) | Softmax | Δ | exp(γ) min / median / max | Largest abs score |
|---|---|---:|---:|---:|---:|---:|---:|
| 200m-qk-softmax-seed0 | complete | 14.4574 | 14.4574 (19,074) | 14.6655 | -0.2081 | 7.67 / 14.09 / 21.91 | 18.70 |

[Summary CSV](summary.csv) · [Learning curves](learning_curves.pdf) · [Learned scales](qk_norm_scales.pdf)
<!-- END GENERATED RESULTS -->

**QK-norm softmax beats the softmax control by 0.21 perplexity.** The gap opened
early: QK-norm was 5.7
perplexity *behind* at update 1,000, level at 2,000, and 0.45–0.49 ahead from
update 3,000 to 6,000; it then narrowed steadily to 0.21 at the end, so the
advantage is largest mid-training and shrinking, not growing, at this budget
([learning curves](learning_curves.pdf), right panel). The terminal
training-loss sample is also lower (2.699 versus 2.713 nats), so the difference is
not confined to the held-out set.

**The scales did the sharpening.** Every layer's mean `exp(gamma)` left 8 within
the first thousand updates except layer 1, which first fell to 7.5 by update
3,000 and then climbed to 10; the other layers rose fastest between updates 1,000
and 7,500 and had nearly flattened by 15,000
([learned scales](qk_norm_scales.pdf)). Terminal layer medians increase
with depth (10.1, 12.9, 13.9, 13.5, 13.4, 14.6, 16.8, 16.3, 16.2, 16.5), and so does
the spread within a layer: layer 1's heads span 7.7–10.4, layers 7–10 span
10.8–21.9. 39 of 160 heads exceed 16 and one head sits below 8. Scores on the
diagnostic sequence reached −14.2 to 18.7 against the largest bound of 21.9, so
heads use most of their positive range. The optimizer's movement budget did not
bind: over this schedule Adam can move `gamma` by about 3.1, a factor of 22 in the
scale, while the largest scale moved by a factor of 2.7.

**Cost.** Throughput on four B300s was 610k against 618k tokens per second per
device for the canonical control's B300 segment, about 1% slower; the cuDNN
softmax path is unchanged and the normalization is elementwise. Training took
1 h 10 min 44 s.

## Scientific recipe

The [run recipe](../../experiments/qk_norm_attention/runs/r1feece425074/config.yaml) is the
[200M softmax control](../../experiments/key_dimension_taylor/runs/r50088d29f2b5/config.yaml) with
`qk_norm: true`, `qk_norm_sharing: per_head`, the default initial scale
`sqrt(d_K)`, and `qk_norm_eps: 1e-6` added; a configuration test enforces that
nothing else differs except output locations. The backbone is the repository's
200M sweep model: 10 layers, width 1024, 16 full-MHA heads, `d_K = d_V = 64`,
SwiGLU width 3008, untied 32,000-token embeddings, RoPE base 10,000, seed 0,
packed FineWeb in the fixed order, global batch 256 with four accumulation steps,
AdamW with peak/minimum learning rates `3e-4`/`3e-5` and 2,000 warmup updates,
**19,074 updates and 10,000,269,312 prediction tokens**. Evaluation uses the
first 1,792 held-out sequences every 1,000 updates and at completion. The
primary endpoint is terminal held-out perplexity.

The log scales form one FP32 array of shape `[10, 16, 1]` (160 values), use the
backbone learning-rate schedule and global gradient clipping, and receive no
weight decay. With this schedule Adam can move each `gamma` by at most about 3.1,
so `exp(gamma)` can range over roughly `[0.4, 180]` from its start at 8. Initial
backbone weights are bit-identical to the 200M softmax control's, which the
launcher's acceptance verified.

**Weight-decay policy.** This run trained under the scaffold's canonical selective
weight decay (`exclude_embeddings_norms_biases_v1`: no decay on the input
embedding, RMSNorm scales, biases, or the log scales), as did its control.

Control, seed 0, same backbone, budget, and decay policy:

| Kernel | Score | Terminal PPL |
| --- | --- | ---: |
| Softmax ([key_dimension_taylor](../key_dimension_taylor/README.md) `200m-softmax-seed0`, source `381f1fd`, job 202190) | `q·k / sqrt(d_K)` | 14.6655 |

## Implementation

The [shared Transformer scaffold](../../scaffolds/transformer/README.md) selects
this mechanism with `qk_norm: true` on the exact-softmax configuration. The
attention module unit-normalizes Q and K per head in FP32 before RoPE and passes
`q' = exp(gamma) sqrt(d_K) q̂` and `k' = k̂` to the unchanged softmax path, whose
`q'·k' / sqrt(d_K)` is the cosine score; with `attention_backend: fused` the
exact softmax runs in cuDNN, as in the control. The model owns the
`qk_norm_log_scale` array (initialized at `log(sqrt(d_K))`, excluded from weight
decay). Nonfinite scores, outputs, gradients, or updated parameters stop training
with a numerical-failure record. Every validation records `gamma` and
`exp(gamma)` for each layer and head in `metrics.jsonl` under
`attention_diagnostics.qk_norm_scale`; W&B receives their mean, minimum, and
maximum, and the run carries the `qk_norm` tag.

## Validation

From a checkout with the scaffold environment synced, run the CPU tests:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_qk_norm.py \
  scaffolds/transformer/tests/test_softmax_backends.py \
  experiments/qk_norm_attention/tests
```

CPU tests compare the composed operator and its Q/K/V and `gamma` gradients with
an independent FP64 reference for softmax and the polynomial kernels, check the
score bound `|s| <= exp(gamma)`, invariance of the operator to rescaling the Q or
K projection, commutation of the normalization with RoPE, parameter ownership and
counts, the weight-decay exclusion, configuration validation and round trips,
diagnostics and their W&B mirror, and training with checkpoint resume. GPU tests
check that cuDNN softmax and the fused kernels consume the cosine operands exactly
as the blocked reference does at the full 2,048-token BF16 context, and the
exact-softmax cuDNN control against the blocked reference. Before launch the five
QK-norm GPU gates passed on one B300 (`akita`, job 202131) and the full scaffold CPU
suite passed (552 tests). In the production allocation, acceptance passed 171 CPU
tests, 5 distributed CPU tests, and 12 GPU tests, matched the initialization to the
softmax control's parameter hash plus the declared log scales, ran a 20-update
full-recipe pilot on the cuDNN backend at 0.202 s per update, and reproduced a
one-update checkpoint resume exactly.

Regenerate the summary, figures, and results block from the committed run (no
GPU, dataset, or checkpoint needed):

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.qk_norm_attention.report
```

## Kennel launch
Use the [execution guide](../../experiments/qk_norm_attention/README.md) to create a fresh run and submit it from an isolated worktree.

## Artifacts

The run keeps its validation-best checkpoint, which is also its terminal one, at
`experiments/qk_norm_attention/runs/r1feece425074/checkpoints/19074/` (13 files,
2.2 GB) and one Slurm log at
`experiments/qk_norm_attention/runs/r1feece425074/training.log`, both on
Kennel under the artifact root and ignored by Git; the manifest records every
file's SHA256. W&B mirrors the run in the `qk_norm_attention` group; the manifest
records its run id. A clone supplies the committed run record, recipe, summary,
figures, and this README, not the checkpoint, log, or datasets.

## Limitations

One seed and one initial scale (`sqrt(d_K)`, not tuned); per-head sharing only.
Scale and score statistics come from one held-out diagnostic sequence per
validation. The canonical control trained on H100s for its first 1,464 updates and
on B300s thereafter, while this run used B300s only; earlier softmax controls on
different hardware agreed within 0.0003 nats, far below the 0.014-nat difference
reported here. The comparison covers exact softmax only; QK-norm's effect on the
polynomial kernels, where the bounded scores matter most, is not yet measured.

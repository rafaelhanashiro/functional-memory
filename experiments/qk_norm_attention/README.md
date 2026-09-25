# QK-norm attention

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

**Status: running.** Requested 2026-09-25 from Kennel and submitted the same day
as Slurm job 202156 at launch commit `4b5c69c`; see the
[provenance manifest](provenance/manifest.json) for progress.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. While the run executes, its
record, logs, and checkpoints live in this experiment's worktree
(`worktrees/qk_norm_attention/experiments/qk_norm_attention/`); at completion the
retained artifacts move to the same paths in the primary checkout. The
[provenance manifest](provenance/manifest.json) is the authoritative record of
sources, configuration, runs, and artifact locations. The packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`.

## Scientific recipe

The [run recipe](configs/200m-qk-softmax-seed0.yaml) is the
[200M softmax control](../key_dimension_taylor/configs/200m-softmax-seed0.yaml) with
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
backbone weights are bit-identical to the 200M softmax control's.

**Weight-decay policy.** This is the first run under the scaffold's canonical
selective weight decay (`exclude_embeddings_norms_biases_v1`: no decay on the
input embedding, RMSNorm scales, biases, or the log scales). The existing softmax
control (terminal perplexity 14.6942, source `1a8836a`) trained with decay on
every parameter, so the comparison below confounds QK-norm with the policy
change until the control retrained under the canonical policy completes: the
`decay_policy_baseline` experiment's `200m-softmax-seed0` (Slurm job 202148,
submitted 2026-09-25 from its own worktree) is that control.

Controls, seed 0, same backbone and budget:

| Kernel | Score | Decay policy | Terminal PPL |
| --- | --- | --- | ---: |
| Softmax ([key_dimension_taylor](../key_dimension_taylor/README.md)) | `q·k / sqrt(d_K)` | all parameters | 14.6942 |
| Softmax (`decay_policy_baseline`, job 202148) | `q·k / sqrt(d_K)` | canonical selective | pending |

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

From the experiment worktree, sync the scaffold environment and run the CPU tests:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_qk_norm.py \
  scaffolds/transformer/tests/test_softmax_backends.py \
  experiments/qk_norm_attention/adapter/tests
```

CPU tests compare the composed operator and its Q/K/V and `gamma` gradients with
an independent FP64 reference for softmax and the polynomial kernels, check the
score bound `|s| <= exp(gamma)`, invariance of the operator to rescaling the Q or
K projection, commutation of the normalization with RoPE, parameter ownership and
counts, the weight-decay exclusion, configuration validation and round trips,
diagnostics and their W&B mirror, and training with checkpoint resume. GPU tests,
run by the launcher's acceptance in the production allocation, check that cuDNN
softmax and the fused kernels consume the cosine operands exactly as the blocked
reference does at the full 2,048-token BF16 context, and cuDNN softmax against the
manual reference at every production width. Acceptance then verifies the
initialization against the softmax control, runs 20 full-recipe updates with
validation, and checks a one-update checkpoint resume. Before launch the five GPU
gates passed on one B300 (`akita`, job 202131) and the full scaffold CPU suite
passed (552 tests) at the pre-merge tree.

## Kennel launch

The run uses four B300s or four H100s in the low-priority `sweep` partition,
submitted from the worktree root at the launch commit recorded in the manifest:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
RUN="200m-qk-softmax-seed0"
mkdir -p "experiments/qk_norm_attention/logs/$RUN"
sbatch --job-name="qk-softmax" --output="experiments/qk_norm_attention/logs/$RUN/slurm-%j.out" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.qk_norm_attention.adapter.execution.acceptance \
  "experiments/qk_norm_attention/configs/$RUN.yaml" "$(git rev-parse HEAD)" 4 B300,H100
```

Keep the frozen execution set (scaffold code, lock, `configs/`, and
`adapter/execution/`) unchanged while the job can run or requeue.

## Artifacts

The run keeps its validation-best checkpoint at
`experiments/qk_norm_attention/checkpoints/200m-qk-softmax-seed0/<step>/` and one
Slurm log at `experiments/qk_norm_attention/logs/200m-qk-softmax-seed0/`, on Kennel
under the artifact root and ignored by Git; the manifest records every file's
SHA256 at completion. W&B mirrors the run in the `qk_norm_attention` group. A clone
supplies the committed run record, recipe, and this README, not checkpoints, logs,
or datasets.

## Limitations

One seed. The only existing softmax control trained under the earlier
all-parameter weight decay, so its 14.6942 is not a clean control for this run.
Per-head scales only; the initial scale `sqrt(d_K)` was not tuned. Scale statistics
come from one held-out diagnostic sequence per validation.

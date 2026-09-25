# Unnormalized exponential attention

This experiment asks what changes when a Transformer uses causal
`exp(QKᵀ / sqrt(d_K)) V` without dividing attention weights by their row sum.
The canonical run uses the repository's **200M sweep recipe**, so it compares
directly with the 200M softmax controls and Taylor/key-dimension sweeps.
**Status: completed.** Removing row normalization costs **0.172 nats** at 200M:
terminal validation PPL **17.46** versus 14.69 for softmax. A completed 250M run
is documented [below](#supplementary-250m-run) as a supplementary scale point.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; the packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`. A clone provides code and records,
not checkpoints, environments, logs, or datasets. The
[provenance manifest](provenance/manifest.json) is the authoritative record of
sources, configurations, runs, and artifact locations.

## Results

| Run | Terminal val NLL | Terminal val PPL | Softmax control NLL / PPL | Gap (nats) |
| --- | ---: | ---: | ---: | ---: |
| `200m-exp-seed0-kennel` | 2.859817 | 17.4583 | 2.687451 / 14.6942 | +0.1724 |

Validation was best at the terminal update 19,074. The gap to the matched
softmax control narrows during training but does not close:

| Update | 1,000 | 2,000 | 5,000 | 10,000 | 15,000 | 19,074 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unnormalized NLL | 5.285 | 4.434 | 3.437 | 3.052 | 2.906 | 2.860 |
| Softmax NLL | 4.458 | 3.609 | 3.020 | 2.806 | 2.720 | 2.687 |
| Gap | 0.827 | 0.825 | 0.417 | 0.246 | 0.186 | 0.172 |

The unnormalized model also ends behind normalized Taylor attention of degree 2
(PPL 16.28) and 4 (14.98) in the [dK × Taylor-degree record](../key_dimension_taylor/README.md).
The gap matches the 0.174 nats measured at 250M against the external OFM
baseline. All 20 evaluations reported valid diagnostics with zero invalid
scores, kernels, weight sums, or outputs. Maximum row weight sums on the fixed
diagnostic sequence peaked at 668,232 (layer 8, update 2,000, maximum score
13.4) and fell to at most 3,466 (layer 9) at completion, far from FP32 overflow.
One seed does not estimate seed variation.

The run used source `3b0d311`, Slurm job `201002`, and four B300s on `akita`
(2026-09-22), taking 2 h 28 min including acceptance, with no restarts, at about
1.33M training tokens per second. Records: metrics and completion in
[`runs/200m-exp-seed0-kennel/`](runs/200m-exp-seed0-kennel/), GPU acceptance
and allocation summarized in the [provenance manifest](provenance/manifest.json),
a [terminal summary](summary.csv), the log at
`logs/200m-exp-seed0-kennel/slurm-201002.out`, and the retained step-19,074
checkpoint (2.1 GB) under `checkpoints/200m-exp-seed0-kennel/`; logs and
checkpoints are ignored and stay on Kennel.

## Scientific recipe

The [canonical recipe](configs/base.yaml) has **199,906,304 parameters**: 10
layers, width 1024, 16 full-MHA heads, `d_K=d_V=64`, SwiGLU width 3008,
32,000-token untied embeddings, 2,048 prediction positions, RoPE base 10,000,
RMSNorm epsilon `1e-6`, and no dropout. The block and final RMSNorm layers and
vocabulary softmax remain unchanged.

The seed is 0. Packed FineWeb data, tokenizer, sequence order, AdamW parameters,
and learning-rate schedule match the 200M sweeps. Global batch 256, global
microbatch 64, and four accumulation steps give **19,074 updates and
10,000,269,312 prediction tokens**. Warmup lasts 2,000 updates; peak/minimum
learning rates are `3e-4`/`3e-5`. Evaluation uses the first 1,792 held-out
sequences every 1,000 updates and at completion. Parameters and optimizer state
are FP32; model computation is BF16. Initial parameters are bit-identical to the
independent in-repository port used by the 200M controls.

Apart from normalization, the recipe is field-for-field identical to these
existing softmax controls, both trained with this repository's code:

| Control | Cluster, attention backend | Terminal val NLL | Terminal val PPL |
| --- | --- | --- | --- |
| [`200m-softmax-seed0`](../key_dimension_taylor/README.md) (degree sweep) | Kennel (H100, then B300), fused | 2.687451 | 14.6942 |
| [`200m-dk064-seed0`](../key_dimension_taylor/README.md) (width sweep) | ORCD, cuDNN | 2.687151 | 14.6898 |

They differ by 0.0003 nats despite different clusters, GPU counts, kernels, and
source revisions, which bounds hardware/kernel effects far below the gaps
measured here. Primary endpoint: terminal held-out perplexity at the matched
token budget; best-checkpoint metrics are retained separately. Comparisons are
single-seed.

## Implementation

The [shared Transformer scaffold](../../scaffolds/transformer/README.md) owns the
model, data pipeline, optimizer, trainer, evaluation, and checkpoints. Setting
`model.attention_normalization: false` selects the exponential mechanism;
`true` retains the existing normalized mechanisms. There is no score clipping,
row-max subtraction, prefix-length division, added Q/K normalization, or added
attention-output normalization.

The fused implementation visits causal query/key tiles without materializing a
sequence-square attention matrix. Exponentials and accumulation are FP32. Mixed
FP32/BF16 products split the FP32 operand into three BF16 components;
FP32/FP32 products use TF32x3. Its derivative is `dS = exp(S) * (dY Vᵀ)`, without
the softmax row-coupling term. The shared blocked reference supplies independent
scalar diagnostics.

`attention_backend: auto` selects the CPU reference or a supported NVIDIA fused
implementation; requested and resolved choices are recorded, and resume uses
the previously resolved choice. Explicit incompatible backends fail. Nonfinite
causal scores, exponential weights, outputs, gradients, or updated parameters
stop training with diagnostics. Weight sums are measurements only: underflow to
zero or overflow of a sum does not introduce a denominator or silently alter the
mathematics.

## Validation

From the experiment worktree, sync the scaffold environment and run the CPU tests:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
JAX_PLATFORMS=cpu PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml --import-mode=importlib -m 'not gpu' \
  scaffolds/transformer/tests/test_unnormalized_attention.py \
  scaffolds/transformer/tests/test_unnormalized_training.py \
  experiments/unnormalized_attention/adapter/tests
```

Tests cover independent FP64 outputs and Q/K/V gradients, causal boundaries,
prefix sums at zero scores, sensitivity to score shifts, underflow and visible
overflow, initialization, gradient accumulation, packed-data resume, and failure
records.

GPU acceptance ([`adapter/execution/acceptance.py`](adapter/execution/acceptance.py))
runs in the production allocation before training: fused forward/VJP tests,
full-model initialization identity against the normalized softmax model with the
same seed, 20 full-context/full-microbatch updates, validation, and a one-update
checkpoint-resume audit while preserving the full learning-rate horizon.
Qualification is bound to source, recipe, hardware, and transfer/allocator
settings; changing any of them reruns acceptance.

The terminal summary regenerates from `runs/` and the manifest:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.unnormalized_attention.adapter.analysis.summary
```

## Kennel launch

The run used four B300s in `long`. One Python process uses all four GPUs;
global batch 256, microbatch 64, and four accumulation steps stay unchanged.
The shared Kennel launcher selects scheduler and transport settings, runs
acceptance, then starts production.

A rerun with the current scaffold is a new experiment with its own worktree, branch,
and record (see [AGENTS.md](../../AGENTS.md)); never relaunch into this record's `runs/`.
Copy the recipes from `configs/` with outputs under `experiments/<new>/` and
`train.source_commit: auto`, give the new adapter an acceptance spec like
[this one's](adapter/execution/acceptance.py), then submit the run from the worktree root:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
mkdir -p "experiments/<new>/logs/$RUN"
sbatch --job-name="$RUN" --output="experiments/<new>/logs/$RUN/slurm-%j.out" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.<new>.adapter.execution.acceptance \
  "experiments/<new>/configs/$RUN.yaml" "$(git rev-parse HEAD)" 4 B300
```

For an H100 fallback, pass `H100` as the final argument; it must pass the same
acceptance on its own hardware. Do not run both copies concurrently, and keep the
frozen execution set unchanged while the job can run or requeue. Resubmitting the
same recipe resumes the latest complete checkpoint, optimizer, and data cursor.

Allocation and acceptance records are written under the experiment's
`logs/<run>/` beside the Slurm log; training metrics and resolved runtime metadata go to `runs/<run>/` and checkpoints to
`checkpoints/<run>/`. Checkpoints keep latest and validation best while resumable;
after completion only the validation-best checkpoint is retained, and the manifest
summarizes the acceptance and allocation records.

The exact packed corpus has 90 training shards, one evaluation shard, and two
manifests totaling 24,031,780,324 bytes (22.38 GiB), verified in
the manifest's `data_manifests.kennel_verification`. Expected manifest
SHA256 values are:

- Train: `9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4`.
- Evaluation: `0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f`.

## Supplementary 250M run

Before the 200M recipe became canonical, the same mechanism was trained once at
the canonical OFM 250M geometry. It is retained as a completed scale point; the
current code has no path dedicated to it.

- **Recipe:** [`configs/250m-exp-seed0-kennel.yaml`](configs/250m-exp-seed0-kennel.yaml):
  14 layers (253,654,016 parameters),
  21,935 updates, 11,500,257,280 prediction tokens, otherwise the recipe above.
- **Execution:** source `51c6474`, Slurm job `200875`, four B300s on `akita`,
  2026-09-22, 3 h 18 min including acceptance, no restarts.
- **Result:** terminal and best validation NLL **2.765265**, PPL **15.883** at
  update 21,935. Every evaluation reported zero invalid scores, kernels, weight
  sums, or outputs. At the final diagnostic sample, most layers' maximum weight
  sums stayed below about 1,000, but layers 6, 12, and 13 reached about 11,900,
  2,400, and 57,500, far from FP32 overflow.
- **Comparison:** the only 250M softmax result is the OFM report
  (source `32ea743`, two H200s, cuDNN): terminal NLL 2.591487, PPL 13.3496, with
  rounded values and no retained curve. The unnormalized run is 0.174 nats
  worse. No normalized 250M model has been trained with this repository's code,
  so a codebase difference cannot be excluded at that scale.
- **Records:** metrics and completion in [`runs/250m-exp-seed0-kennel/`](runs/250m-exp-seed0-kennel/);
  CPU validation, GPU acceptance, allocation, the post-pilot walltime reduction
  from 48 to 8 hours, and the cuBLAS-warning assessment in
  the [provenance manifest](provenance/manifest.json); the
  log at `logs/250m-exp-seed0-kennel/slurm-200875.out` and the retained
  step-21,935 checkpoint (2.7 GB) under `checkpoints/250m-exp-seed0-kennel/`.
- **Origin:** the mechanism was first written as a standalone experiment at
  source `7a3e584` and run on ORCD; its recipe and dependency lock are at that
  commit, and its ORCD artifacts were never inspected or transferred.

## Limitations

Each configuration has one seed. The 200M control comparison is within this
repository's code and data; the 250M comparison relies on an external,
rounded OFM result. Diagnostics sample one held-out sequence per evaluation.

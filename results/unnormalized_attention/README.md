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
(PPL 16.28) and 4 (14.98) from [polynomial attention](../polynomial_attention/README.md).
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
and allocation in [`validation/200m-exp-seed0-kennel/`](validation/200m-exp-seed0-kennel/),
a [terminal summary](summary.csv), the log at
`logs/unnormalized_attention/200m-exp-seed0-kennel/slurm-201002.out`, and
the retained step-19,074 checkpoint (2.1 GB) under
`checkpoints/unnormalized_attention/200m-exp-seed0-kennel/` at the artifact root.

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
| [`polynomial_attention/200m-softmax-seed0`](../polynomial_attention/README.md) | Kennel (H100, then B300), fused | 2.687451 | 14.6942 |
| [`key_dimension/200m-dk064-seed0`](../key_dimension/README.md) | ORCD, cuDNN | 2.687151 | 14.6898 |

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

From a dedicated checkout, prepare the shared environment according to cluster
policy, then run CPU acceptance:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project scaffolds/transformer --locked --extra cuda
JAX_PLATFORMS=cpu "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/studies/unnormalized_attention/tests \
  --import-mode=importlib -m 'not gpu'
JAX_PLATFORMS=cpu "$UV_PROJECT_ENVIRONMENT/bin/python" -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/tests/test_frozen_parity.py -k unnormalized_attention
```

Tests cover independent FP64 outputs and Q/K/V gradients, causal boundaries,
prefix sums at zero scores, sensitivity to score shifts, underflow and visible
overflow, initialization, gradient accumulation, packed-data resume, and failure
records. Frozen-source checks export the original experiment source `7a3e584`
from Git and compare initialization, logits, every gradient, optimizer updates,
and checkpoint continuation.

GPU acceptance runs in the intended allocation before production: fused
forward/VJP tests, exact full-model initialization verification against the
independent port, 20 full-context/full-microbatch updates, validation, and a
one-update checkpoint-resume check while preserving the full learning-rate
horizon. Qualification is bound to source, recipe, hardware, and
transfer/allocator settings; changing any of them reruns acceptance.

## Kennel launch

The run used four B300s in `long`. One Python process uses all four GPUs;
global batch 256, microbatch 64, and four accumulation steps stay unchanged. The
scheduler and transport settings are selected by `launch_kennel.sbatch` and
`run_kennel.sh`, using the shared Kennel runtime helper. Acceptance runs first,
then production starts automatically. To reproduce, create a dedicated
execution checkout of the recorded source and submit from it, with fresh output
paths if the original run record should be kept:

```bash
PROJECT_ROOT=/data/cl/u/rafah/home/projects/functional-memory
EXECUTION_ROOT="$PROJECT_ROOT/worktrees/unnormalized_attention-3b0d311"
git -C "$PROJECT_ROOT" worktree add --detach "$EXECUTION_ROOT" 3b0d311655b033545d0c211467fd600376e0fdeb
SOURCE_COMMIT="$(git -C "$EXECUTION_ROOT" rev-parse HEAD)"
RUN=200m-exp-seed0-kennel
CONFIG="$EXECUTION_ROOT/results/unnormalized_attention/configs/$RUN.yaml"
LOG_DIR="$PROJECT_ROOT/logs/unnormalized_attention/$RUN"
mkdir -p "$LOG_DIR"
sbatch --job-name=unnorm-200m --output="$LOG_DIR/slurm-%j.out" \
  "$EXECUTION_ROOT/scaffolds/transformer/studies/unnormalized_attention/launch_kennel.sbatch" \
  "$EXECUTION_ROOT" "$PROJECT_ROOT" "$SOURCE_COMMIT" "$CONFIG" 4 B300
```

For an H100 fallback, add `--gres=gpu:h100:4` to `sbatch` and replace the final
`B300` argument with `H100`; it must pass the same acceptance on its own
hardware. Do not run both copies concurrently. Preserve the source checkout and
shared environment unchanged while the job can run or requeue. Reusing the same
source/config/output resumes the latest complete checkpoint, optimizer, and
data cursor.

Slurm allocation/source/config records are under
`<artifact-root>/.cache/unnormalized_attention/submissions/`; matching acceptance
records are under `preflight/` beside it. Training metrics and resolved runtime
metadata live in `results/unnormalized_attention/runs/<run>/`, checkpoints in
`checkpoints/unnormalized_attention/<run>/`, and logs in
`logs/unnormalized_attention/<run>/` at the durable artifact root.
Checkpoints keep latest and validation best while resumable; after completion
only the validation-best checkpoint is retained. Retained metrics and
acceptance evidence are copied into this record after the execution stops, and
the execution checkout is then removed.

The exact packed corpus has 90 training shards, one evaluation shard, and two
manifests totaling 24,031,780,324 bytes (22.38 GiB), verified in
[`validation/kennel-data.json`](validation/kennel-data.json). Expected manifest
SHA256 values are:

- Train: `9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4`.
- Evaluation: `0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f`.

## Supplementary 250M run

Before the 200M recipe became canonical, the same mechanism was trained once at
the canonical OFM 250M geometry. It is retained as a completed scale point; the
current code has no path dedicated to it.

- **Recipe:** [`configs/250m-exp-seed0-kennel.yaml`](configs/250m-exp-seed0-kennel.yaml),
  byte-identical to the executed file: 14 layers (253,654,016 parameters),
  21,935 updates, 11,500,257,280 prediction tokens, otherwise the recipe above.
- **Execution:** source `51c6474`, Slurm job `200875`, four B300s on `akita`,
  2026-09-22, 3 h 18 min including acceptance, no restarts.
- **Result:** terminal and best validation NLL **2.765265**, PPL **15.883** at
  update 21,935. Every evaluation reported zero invalid scores, kernels, weight
  sums, or outputs. At the final diagnostic sample, most layers' maximum weight
  sums stayed below about 1,000, but layers 6, 12, and 13 reached about 11,900,
  2,400, and 57,500, far from FP32 overflow.
- **Comparison:** the only 250M softmax result is the historical OFM report
  (source `32ea743`, two H200s, cuDNN): terminal NLL 2.591487, PPL 13.3496, with
  rounded values and no retained curve. The unnormalized run is 0.174 nats
  worse. No normalized 250M model has been trained with this repository's code,
  so a codebase difference cannot be excluded at that scale.
- **Records:** metrics and completion in [`runs/250m-exp-seed0-kennel/`](runs/250m-exp-seed0-kennel/);
  CPU validation, GPU acceptance, allocation, the post-pilot walltime reduction
  from 48 to 8 hours, and the cuBLAS-warning assessment in
  [`validation/250m-exp-seed0-kennel/`](validation/250m-exp-seed0-kennel/); the
  log at `logs/unnormalized_attention/250m-exp-seed0-kennel/slurm-200875.out`
  and the retained step-21,935 checkpoint (2.7 GB) under
  `checkpoints/unnormalized_attention/250m-exp-seed0-kennel/` at the artifact root.
- **Origin:** the mechanism was first written as a standalone experiment at
  source `7a3e584` and run on ORCD. Its [recipe](configs/historical.yaml) and
  [dependency lock](configs/historical.uv.lock) are unchanged; those ORCD
  artifacts were never inspected or transferred. Export `7a3e584` from Git to
  reproduce that implementation.

## Limitations

Each configuration has one seed. The 200M control comparison is within this
repository's code and data; the 250M comparison relies on an external,
rounded OFM result. Diagnostics sample one held-out sequence per evaluation.

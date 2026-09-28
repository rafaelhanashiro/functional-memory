# Unnormalized exponential attention

[Selected runs and source hashes](selection.yaml) · [Run index](../../experiments/attention_scaling/runs.csv) · [Execution guide](../../experiments/attention_scaling/README.md)

This experiment asks what changes when a Transformer uses causal
`exp(QKᵀ / sqrt(d_K)) V` without dividing attention weights by their row sum.
The canonical run uses the repository's **200M sweep recipe**, so it compares
directly with the 200M softmax controls and Taylor/key-dimension sweeps.
**Status: completed.** Removing normalization costs **0.174 nats** at 200M:
terminal validation PPL **17.46** versus 14.67 for softmax. A completed 250M run
is documented [below](#supplementary-250m-run) as a supplementary scale point.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`; the packed corpus is under
`/data/cl/u/rafah/home/datasets/fineweb_12b`. A clone provides code and records,
not checkpoints, environments, logs, or datasets. The
[run records](../../experiments/attention_scaling/runs/) are authoritative for
sources, configurations, runs, and artifact locations.

## Results

| Run | Terminal val NLL | Terminal val PPL | Softmax control NLL / PPL | Gap (nats) |
| --- | ---: | ---: | ---: | ---: |
| `200m-exp-seed0-kennel` | 2.859817 | 17.4583 | 2.685498 / 14.6655 | +0.1743 |

Validation was best at the terminal update 19,074. The softmax control is the
repository's softmax baseline, trained with this repository's code:

| Control | Cluster, attention backend | Terminal val NLL | Terminal val PPL |
| --- | --- | --- | --- |
| [`200m-softmax-seed0`](../key_dimension_taylor/README.md) (source `381f1fd`, selective weight decay) | Kennel (H100, then B300), cuDNN | 2.685498 | 14.6655 |

The control trained under the scaffold's selective weight-decay policy; this run
trained with decay on all parameters, a difference worth about 0.03 PPL in the
softmax baseline and far below the gap reported here.

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
  experiments/attention_scaling/tests
```

Tests cover independent FP64 outputs and Q/K/V gradients, causal boundaries,
prefix sums at zero scores, sensitivity to score shifts, underflow and visible
overflow, initialization, gradient accumulation, packed-data resume, and failure
records.

GPU acceptance (see the [execution guide](../../experiments/attention_scaling/README.md))
runs in the production allocation before training: fused forward/VJP tests,
full-model initialization identity against the normalized softmax model with the
same seed, 20 full-context/full-microbatch updates, validation, and a one-update
checkpoint-resume audit while preserving the full learning-rate horizon.
Qualification is bound to source, recipe, hardware, and transfer/allocator
settings; changing any of them reruns acceptance.

The terminal summary regenerates from `runs/` and the selection:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m results.unnormalized_attention.summary
```

## Kennel launch
Use the [execution guide](../../experiments/attention_scaling/README.md) to create a fresh run and submit it from an isolated worktree.

## Supplementary 250M run

Before the 200M recipe became canonical, the same mechanism was trained once at
the canonical OFM 250M geometry. It is retained as a completed scale point; the
current code has no path dedicated to it.

- **Recipe:** [Resolved configuration](../../experiments/attention_scaling/runs/r7e462a645696/config.yaml):
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
- **Records:** metrics and completion in [`runs/250m-exp-seed0-kennel/`](../../experiments/attention_scaling/runs/r7e462a645696/);
  CPU validation, GPU acceptance, allocation, the post-pilot walltime reduction
  from 48 to 8 hours, and the cuBLAS-warning assessment in
  the [run record](../../experiments/attention_scaling/runs/r7e462a645696/record.json); the
  log at `experiments/attention_scaling/runs/r7e462a645696/training.log` and the retained
  step-21,935 checkpoint (2.7 GB) under `experiments/attention_scaling/runs/r7e462a645696/checkpoints/`.
- **Origin:** the mechanism was first written as a standalone experiment at
  source `7a3e584` and run on ORCD; its recipe and dependency lock are at that
  commit, and its ORCD artifacts were never inspected or transferred.

## Limitations

Each configuration has one seed. The 200M control comparison is within this
repository's code and data; the 250M comparison relies on an external,
rounded OFM result. Diagnostics sample one held-out sequence per evaluation.

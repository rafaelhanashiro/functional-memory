# Shared Transformer scaffold

This package owns the JAX/Flax Transformer, data pipeline, optimizer, distributed
training, evaluation, and checkpoint machinery used by the pretraining experiments.
Scientific recipes, findings, and reproduction records belong to
`results/<experiment>/`. This README describes the reusable software API.
State-rank analysis uses the independent `scaffolds/state_rank/` capability.

## Model, runtime, and launch scripts

The model and trainer share one implementation across clusters. Generic device
placement, replication, sharding, and gradient synchronization belong in
`training/distributed.py`. Transfer options such as host staging belong in that
runtime interface, keeping cluster and GPU-model workarounds out of model and
optimizer code and scientific recipes.

Launchers are script files: Slurm `.sbatch`/shell scripts and Python submission
scripts. They request resources, select execution settings, and invoke the shared
trainer. Cluster-specific transport choices, including host staging and NCCL
environment settings, belong at this launch boundary. Backend capability checks
remain in `attention/backends/`; resolved implementations and runtime settings
belong in execution provenance. See the canonical separation rules in
[AGENTS.md](../../AGENTS.md) and cluster policy in [CLUSTERS.md](../../CLUSTERS.md).

The runtime accepts `TRANSFORMER_HOST_STAGING=1` to copy initialization and
restored state through host memory and verify every physical replica byte for
byte. It also stages replicated RNG keys; model computation stays on the GPU.
Packed training and evaluation batches always transfer directly from NumPy host
arrays to their destination shards. `metadata.json` records the resolved transfer
policy, hardware, selected environment settings, and changes across resumptions
in `execution_runtime` and `runtime_history`. `scaffold_ofm_source_commit` records
shared implementation ancestry; an experiment's scientific control source belongs
in its own provenance manifest.

Kennel launchers source `launchers/kennel_runtime.sh`, which selects host staging
and disables NCCL peer transport for A100/A6000 allocations. A6000 uses the
validated 0.90 allocator fraction. Explicit `TRANSFORMER_HOST_STAGING=0` or `1`
overrides automatic selection. This transport policy does not replace kernel and
full-recipe qualification; consult `CLUSTERS.md` for its measured scope.

## Attention and parameter ownership

`attention/multihead.py` owns Q/K/V projections, RoPE, head layout, and the output
projection. `attention/core.py` dispatches complete operations on post-RoPE
`[batch, time, heads, channels]` arrays. Q/K use `d_K`; the output has V's shape
and dtype. Mechanisms live in `softmax.py`, `taylor.py`, and
`learned_quadratic.py`. Their implementations live under `attention/backends/`.
Fused kernels consume Q/K/V directly and retain their custom backward passes.
The interface does not require a materialized attention matrix.

The existing scientific configuration fields remain supported:

| Settings | Mechanism |
|---|---|
| `attention_degree: null` | Exact softmax |
| `attention_degree: 0` | Uniform causal-prefix averaging |
| Nonnegative `attention_degree`, `quadratic_mode: fixed` | Fixed Taylor polynomial; odd degrees permit signed weights |
| `attention_degree: 2`, `quadratic_mode: positive` | Learned `0.5 * (1 + (a*s+b)^2)` |
| `attention_degree: 2`, `quadratic_mode: unrestricted` | Learned `1 + b*s + c*s^2` |

`qk_head_dim` controls projection geometry independently of the mechanism.
`quadratic_sharing: global` stores one pair across all layers and heads;
`per_head` stores `[num_layers, num_heads, 2]`. Each layer keeps its own projections.

`AttentionParameterSpec` declares initialization, sharing, the forward argument,
and weight-decay policy. The model registers each array once and selects the
appropriate device-array arguments for each layer. Gradients from shared uses
accumulate into one parameter and optimizer state. The existing
`quadratic_coefficients` checkpoint key and backbone initialization order are
preserved. No new RNG draws are introduced for these coefficients.

Adding a mechanism means implementing its attention operation and parameter
specifications, adding selection/validation to the configuration and dispatcher,
and defining its numerical diagnostics. A fused implementation must differentiate
all trainable array arguments. Data loading, the Transformer residual/MLP blocks,
and the training loop do not need separate copies.

## Execution selection and resumption

The shared configuration defaults to `attention_backend: auto`; ordinary recipes
can omit it. CPU execution selects the blocked reference. Supported NVIDIA GPU
polynomial/learned shapes select the fused kernels; exact softmax selects an
appropriate public JAX implementation. Unsupported shapes use the reference when
automatic selection is requested. Explicit overrides are validated and never
silently replaced.

The requested and resolved implementations are recorded in run metadata. The
resolved backend is stored in the run configuration and reused on resume, even
when the caller requests `auto`. A resumed run fails if its pinned implementation
cannot run on the current device. Explicit backend choices in historical recipes
are preserved.

Softmax uses stable score shifting. Taylor and learned polynomials use unshifted
scores. Signed mechanisms retain their existing signed-normalizer rules; the
scaffold does not add clipping or denominator epsilon.

## Environment and training API

Run from a dedicated development checkout. This shared JAX environment is separate
from the root PyTorch environment used for state-rank analysis:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project scaffolds/transformer --locked --extra cuda

# Tiny synthetic CPU check; both outputs are disposable.
JAX_PLATFORMS=cpu .envs/shared_transformer/bin/python -m scaffolds.transformer.train \
  scaffolds/transformer/configs/smoke.yaml \
  --workdir "$PWD/.cache/shared-smoke/results" \
  --checkpoint-dir "$PWD/.cache/shared-smoke/checkpoints"
```

For a new experiment, start with `configs/base.yaml`, store its exact resolved
recipe in `results/<experiment>/configs/`, and set the data manifests and scientific
parameters. Keep `train.source_commit: auto` until the tested execution checkout
is committed and frozen. Record its home cluster and artifact root in the
experiment's provenance. Inside the required GPU allocation, the common CLI is:

```bash
.envs/shared_transformer/bin/python -m scaffolds.transformer.train \
  results/my_experiment/configs/run.yaml \
  --workdir results/my_experiment/runs/run \
  --checkpoint-dir checkpoints/my_experiment/run
```

`--attention-degree` accepts a nonnegative degree or `softmax`; `--qk-head-dim`
selects the projection width. `--checkpoint-dir` separates large model state from
scientific records. `--stop-after-step` writes a resumable checkpoint while keeping
the recipe's full learning-rate horizon. Resume is the default, and Slurm requeue
also resumes; `--no-resume` applies only to a fresh attempt. Historical recipes and
source commits stay immutable in their retained experiment records.

## Scientific study adapters

`studies/` contains small configuration, scheduling, validation, and analysis
adapters for a scientific question. They call the shared model, attention, data,
and training implementation. Use configuration alone when sufficient; a new
experiment does not need another copy of the training stack. For example,
`python -m scaffolds.transformer.studies.polynomial_attention.plot` uses the
polynomial study's canonical retained manifest and regenerates its figures.
Consult that experiment's `results/<experiment>/README.md` for its exact command.

The `key_dimension`, `polynomial_attention`, `learned_quadratic_attention`,
`odd_taylor_attention`, and `key_dimension_taylor` study adapters all use this
same training implementation. Their retained historical runs and currently frozen
execution checkouts still identify the source revisions that actually ran them.

Each experiment has one scientific README under `results/<experiment>/`, with
recipes in `configs/`, run metrics in `runs/`, figures in `figures/`, and the
canonical manifest at `provenance/manifest.json`. Plotting updates marked result
sections without replacing authored methods. Study adapters do not own competing
experiment READMEs or per-run narrative reports.

During training, checkpoints retain the latest resumable state and validation best.
After review, keep only the validation-best checkpoint at
`checkpoints/<experiment>/<run>/<step>/`, terminal metrics independently, and the
matching successful log under `results/logs/<experiment>/`. Checkpoints, logs,
environments, and caches remain on the home cluster and are ignored by Git.

## Validation and frozen execution

```bash
JAX_PLATFORMS=cpu \
  .envs/shared_transformer/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/tests scaffolds/transformer/studies --import-mode=importlib -m 'not gpu'

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .envs/shared_transformer/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml \
  scaffolds/transformer/tests --import-mode=importlib -m 'not gpu' -k distributed
```

The shared suite covers independent FP64 attention/gradient references, signed
normalizers, learned parameter sharing, optimizer exclusions, distributed
accumulation, packed-data cursors, and interruption/resume. Frozen-source parity
tests export the recorded historical commits into temporary directories and
compare initialization, logits, every parameter gradient, optimizer updates,
and restoration/continuation of original checkpoints. They require local Git
history, but never read or write a live experiment checkout or checkpoint.
GPU-marked tests exercise the actual fused forward/backward implementations.

Run shared-scaffold validation in its own checkout and environment. Existing jobs
continue using their frozen source, environment, manifests, and artifact paths
until they finish or are explicitly migrated. Integrating a shared scaffold on
`main` does not update an active execution checkout. Future runs freeze a source
revision containing the canonical scaffold; source-parity fixtures deliberately
load historical packages from their original Git commits.

The extraction sources are recorded in `sources.json`; historical scientific
results continue to identify their original source commits.

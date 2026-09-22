# Shared Transformer scaffold

This package owns the JAX/Flax Transformer, data pipeline, optimizer, distributed
training, evaluation, and checkpoint machinery used by the pretraining experiments.
Experiment directories retain their recipes, sweep orchestration, tests of
scientific assumptions, and analysis. The state-rank analysis remains independent.

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

## Use

From the repository or dedicated development checkout:

```bash
export UV_PROJECT_ENVIRONMENT="$PWD/.envs/shared_transformer"
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --project experiments/shared_transformer --locked --extra cuda

# Small synthetic CPU smoke run, isolated from retained experiments.
JAX_PLATFORMS=cpu .envs/shared_transformer/bin/python -m experiments.shared_transformer.train \
  experiments/shared_transformer/configs/smoke.yaml --workdir "$PWD/.cache/shared-smoke"

# Production recipe: launch only from a clean, committed experiment checkout.
.envs/shared_transformer/bin/python -m experiments.shared_transformer.train \
  experiments/shared_transformer/configs/base.yaml --attention-degree 4 \
  --qk-head-dim 32 --workdir results/my_experiment/run
```

Legacy key-dimension and polynomial import/CLI paths are compatibility adapters.
Their configuration defaults and existing YAML recipes remain valid. The joint,
learned-quadratic, and odd-Taylor orchestration packages use this common core.

Runtime checkpointing keeps the latest resumable state and validation best.
After a run is complete and reviewed, experiment retention keeps only its
validation-best checkpoint in `experiments/<experiment>/checkpoints/<run>/`,
terminal metrics separately in `results/<experiment>/`, provenance, and one
matching successful Slurm log. Checkpoints and caches stay out of Git.

## Validation and rollout

```bash
JAX_PLATFORMS=cpu \
  .envs/shared_transformer/bin/python -m pytest \
  -c experiments/shared_transformer/pyproject.toml \
  experiments/shared_transformer/tests --import-mode=importlib -m 'not gpu'

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  .envs/shared_transformer/bin/python -m pytest \
  -c experiments/shared_transformer/pyproject.toml \
  experiments/shared_transformer/tests --import-mode=importlib -m 'not gpu' -k distributed
```

The shared suite covers independent FP64 attention/gradient references, signed
normalizers, learned parameter sharing, optimizer exclusions, distributed
accumulation, packed-data cursors, and interruption/resume. Frozen-source parity
tests export the recorded historical commits into temporary directories and
compare initialization, logits, every parameter gradient, optimizer updates,
and restoration/continuation of original checkpoints. They require local Git
history, but never read or write a live experiment checkout or checkpoint.
GPU-marked tests exercise the actual fused forward/backward implementations.

This refactor is developed on `codex/shared-transformer-scaffold` in its own
checkout and environment. It must not be merged into or applied to a frozen
execution checkout while that checkout's jobs can run or requeue. Existing runs
keep their original source, environment, manifests, configurations, results, and
checkpoints. Integrate the retained experiment changes and this scaffold on
`main` after those runs complete and their results are reviewed. Future runs
use frozen worktrees based on the canonical shared scaffold.

The extraction sources are recorded in `sources.json`; historical scientific
results continue to identify their original source commits.

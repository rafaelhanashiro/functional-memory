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
and dtype. Mechanisms live in `softmax.py`, `taylor.py`,
`learned_quadratic.py`, and `unnormalized.py`. Their implementations live under `attention/backends/`.
Fused kernels consume Q/K/V directly and retain their custom backward passes.
The interface does not require a materialized attention matrix.

The existing scientific configuration fields remain supported:

| Settings | Mechanism |
|---|---|
| `attention_degree: null`, `attention_normalization: true` (default) | Exact softmax |
| `attention_degree: null`, `attention_normalization: false` | Unnormalized causal `exp(QKᵀ / sqrt(d_K)) V` |
| Nonnegative `attention_degree`, `attention_normalization: false` | Unnormalized fixed Taylor `p_m(QKᵀ / sqrt(d_K)) V`; odd degrees permit signed weights |
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

### Gated linear attention

`attention_mechanism: gla` replaces the attention module inside the same residual
blocks. `kind: transformer` continues to select the shared language-model backbone;
the default mechanism `dot_product` preserves all existing recipes. GLA uses
`qk_head_dim` and `value_head_dim` independently. Its vector-gated recurrence is
`S_t = diag(sigmoid(g_t) ** (1 / tau)) S_{t-1} + k_t v_t^T`, followed by the
inclusive read `q_t^T S_t`, per-head RMSNorm, a SiLU output gate, and an output
projection. Each sequence starts with zero state. No score scaling or attention
row normalization is applied. The OFM defaults are gate rank 16, tau 16, RoPE
enabled, and outer/inner chunks 64/16, exposed by `gla_gate_rank`, `gla_gate_tau`,
`gla_use_rope`, `gla_chunk_size`, and `gla_inner_chunk_size`.

GLA accepts `attention_backend: auto`, `blocked` (the OFM chunkwise associative
scan reference), and `fused` (the repository-owned OFM chunked Triton forward and
custom backward). The fused path requires an Ampere-or-newer NVIDIA GPU, prediction
length divisible by 64, and chunk sizes 64/16. The CUDA extra installs pinned
Triton; the minimal JAX/Triton bridge is vendored with its license and source
provenance. The operation and its module retain the original OFM arithmetic;
key tiles below 16 are padded so narrow keys can use Triton matrix products.
Automatic CPU execution uses the reference; explicit unsupported fused requests
fail. Launchers configure bridge caching and XLA options separately from recipes.
Training resolves the backend from the actual prediction length. Direct model
construction with `auto` resolves from `max_seq_len`; use `blocked` for inference
at arbitrary lengths that do not satisfy the fused kernel's chunk constraint.
The bridge hands Triton's PTX (ISA 8.7 with Triton 3.6) to jaxlib, which assembles
it with the first `ptxas` among its CUDA roots; that is a node's `/usr/local/cuda`
unless `CUDA_HOME` names another toolkit, so launchers set `CUDA_HOME` to the
environment's `nvidia/cuda_nvcc`. Operands are converted to FP32 before launch:
with BF16 operands upcast inside the kernel, Triton 3.6 on B300 computed the FP32
products at TF32-level precision. The fused kernel passes its GPU tests on H100 and B300.

The OFM mixed-precision behavior is retained: RoPE promotes Q/K to FP32 and the
fused kernel returns reads in Q's dtype, while the reference module casts its
completed read to the configured compute dtype before head normalization.
Both accumulate recurrent state in FP32. Their BF16 paths are therefore not
claimed to be bitwise identical; production pins the fused implementation.

GLA diagnostics report finite projections, decay gates, recurrence reads, and
outputs. They do not impose positive dot products or invent attention row sums.
Training checks gradient and updated-parameter finiteness every step. Runtime
metadata records the actual recurrence backend, gate settings, zero-state policy,
and state size. All checkpointing, optimizer state, data cursors, mixed precision,
and distributed training continue through the shared runtime.

## Execution selection and resumption

The shared configuration defaults to `attention_backend: auto`; ordinary recipes
can omit it. CPU execution selects the blocked reference. Supported NVIDIA GPU
polynomial, learned, and unnormalized exponential/Taylor shapes select the fused kernels; exact softmax selects an
appropriate public JAX implementation. Unsupported shapes use the reference when
automatic selection is requested. Explicit overrides are validated and never
silently replaced.

The requested and resolved implementations are recorded in run metadata. The
resolved backend is stored in the run configuration and reused on resume, even
when the caller requests `auto`. A resumed run fails if its pinned implementation
cannot run on the current device. Explicit backend choices in historical recipes
are preserved.

Softmax uses stable score shifting. Taylor and learned polynomials use unshifted
scores. Unnormalized exponential and Taylor kernels also use unshifted scores
and never divide by a row sum. Nonfinite outputs, gradients, and updated parameters are surfaced as
failures; diagnostic-only weight sums may underflow or overflow. Signed mechanisms retain their existing signed-normalizer rules; the
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
`python -m scaffolds.transformer.studies.key_dimension_taylor.plot` verifies the
dK × Taylor-degree record's canonical manifest and regenerates its figures.
Consult that experiment's `results/<experiment>/README.md` for its exact command.

The `key_dimension_taylor`, `learned_quadratic_attention`, `odd_taylor_attention`,
`unnormalized_attention`, and `gla_key_dimension` study adapters all use this same training implementation.
`key_dimension_taylor` also carries the degree-sweep helpers (`degree_sweep/`) and
width-sweep configuration defaults (`width_sweep_configs/`) that its runs and the
learned/odd adapters build on. Kernel and throughput benchmark scripts live in
`scaffolds/transformer/benchmarks/`. Their retained historical runs and currently frozen
execution checkouts still identify the source revisions that actually ran them.

Each experiment has one scientific README under `results/<experiment>/`, with
recipes in `configs/`, run metrics in `runs/`, figures in `figures/`, and the
canonical manifest at `provenance/manifest.json`. Plotting updates marked result
sections without replacing authored methods. Study adapters do not own competing
experiment READMEs or per-run narrative reports.

During training, checkpoints retain the latest resumable state and validation best.
After review, keep only the validation-best checkpoint at
`checkpoints/<experiment>/<run>/<step>/`, terminal metrics independently, and the
matching successful log under `logs/<experiment>/`. Checkpoints, logs,
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

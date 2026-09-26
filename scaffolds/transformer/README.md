# Shared Transformer scaffold

This package owns the JAX/Flax Transformer, data pipeline, optimizer, distributed
training, evaluation, and checkpoint machinery used by the pretraining experiments.
Scientific recipes, findings, and reproduction records belong to
`experiments/<experiment>/`. This README describes the reusable software API.
State-rank analysis uses the independent `scaffolds/state_rank/` capability.

## Embeddings and optimizer policy

The input token embedding and output vocabulary projection are always independent
trainable arrays. `model.tie_embeddings` must be `false`; `true` is rejected.
Supporting tied embeddings requires a scaffold change, rather than a recipe toggle.

AdamW applies weight decay to projection weights, including the independent output
vocabulary projection. It excludes input embeddings, all RMSNorm scales (including
GLA's per-head scales), all biases (including the GLA retention-gate bias), and
attention parameters whose `AttentionParameterSpec` disables decay. Excluded
parameters still receive ordinary Adam gradient updates. These rules apply to every
attention mechanism and are not experiment-specific options.

Embeddings and normalization modules mark their parameters with NNX metadata
`weight_decay=False`; bias parameters use NNX's `bias` name. New modules must retain
these roles rather than infer decay from tensor dimensions. The optimizer preserves
the parameter tree and masks only AdamW's decoupled decay term. FP32 parameters and
Adam moments are retained with BF16 computation; clipping follows gradient averaging.

Resolved configurations record
`optim.weight_decay_policy: exclude_embeddings_norms_biases_v1`. The trainer refuses
to resume a run with an absent or different policy identifier. Reproduce or resume
historical runs using their recorded source commit; loading an old recipe for a new
run uses the canonical policy. Existing scientific records are unchanged.

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
| `attention_degree: m >= 1`, `attention_normalization: false`, `attention_coefficients: learned` | Unnormalized `sum_r c_r (QKᵀ / sqrt(d_K))^r V` with every coefficient learned from `1/r!`; signed |
| `attention_degree: m >= 1`, `attention_normalization: true`, `attention_coefficients: learned` | Row-normalized `sum_r c_r s^r V / rowsum` with every coefficient learned from `1/r!`; signed weights and row sums; blocked reference only |
| Even `attention_degree: m >= 2`, `attention_normalization: true`, `attention_coefficients: learned`, `attention_coefficient_parameterization: sum_of_squares` | Row-normalized `(q1(s)^2 + q2(s)^2) V / rowsum` with `deg q1 = m/2`, `deg q2 = m/2 - 1` learned and initialized to reproduce `T_m` exactly; nonnegative weights, positive row sums; blocked reference only |
| `attention_degree: 0` | Uniform causal-prefix averaging |
| Nonnegative `attention_degree`, `quadratic_mode: fixed` | Fixed Taylor polynomial; odd degrees permit signed weights |
| `attention_degree: 2`, `quadratic_mode: positive` | Learned `0.5 * (1 + (a*s+b)^2)` |
| `attention_degree: 2`, `quadratic_mode: unrestricted` | Learned `1 + b*s + c*s^2` |

With `attention_coefficient_parameterization: sum_of_squares` the learned array is
`polynomial_square_factors` instead (`attention/nonnegative_polynomial.py`): the same
`m + 1` values per layer and head, holding the ascending coefficients of `q1` then
`q2`. Every polynomial of even degree that is nonnegative on the real line has this
form, so the parameterization is complete and adds no parameters. The kernel is
evaluated as `q1(s)^2 + q2(s)^2` rather than through expanded coefficients, whose
cancellation near a zero minimum could round below zero. Kernel values may be
exactly zero; a zero row sum is invalid. Diagnostics report the factors as raw
parameters and the coefficients of `q1^2 + q2^2` as the effective ones.

`qk_head_dim` controls projection geometry independently of the mechanism.
`quadratic_sharing: global` stores one pair across all layers and heads;
`per_head` stores `[num_layers, num_heads, 2]`. Each layer keeps its own projections.
Learned polynomial coefficients are the `polynomial_coefficients` array: `[m + 1]`
with `attention_coefficient_sharing: global`, `[num_layers, num_heads, m + 1]` with
`per_head`, FP32, without weight decay. The blocked reference evaluates them by
Horner's rule with or without row normalization. Without normalization,
`attention/backends/fused_learned_polynomial.py` is the fused kernel, whose custom
backward also returns each head's coefficient gradient `sum_{i>=j} S_ij^r (dY_i · V_j)`.
With normalization (`attention/learned_polynomial.py`) the weights divide by the
signed row sum, as for odd Taylor and unrestricted quadratic kernels; there is no
fused kernel, `attention_backend: auto` resolves to `blocked`, and `fused` is rejected.

### QK-norm (cosine scores)

QK-norm is the scaffold's default for dot-product attention: a recipe that leaves
`qk_norm` unset gets `true`, and states `qk_norm: false` to use plain `q·k / sqrt(d_K)`
scores. GLA rejects `qk_norm: true`, and degree 0, which never reads Q or K, resolves
it to `false`. It composes with every dot-product mechanism above and
changes only the score. Q and K are unit-normalized per head in FP32,
`x̂ = x / sqrt(||x||² + qk_norm_eps)`, and the score becomes

```text
s_ij = exp(γ) · <q_i, k_j> / (||q_i|| ||k_j||),   |s_ij| <= exp(γ)
```

The kernels are untouched: `attention/multihead.py` feeds the mechanism
`q' = exp(γ) sqrt(d_K) q̂` and `k' = k̂` (`attention/qk_norm.py`), whose
`q'·k' / sqrt(d_K)` is the cosine score, so the blocked reference, the fused
Pallas/Triton kernels, cuDNN softmax, and all diagnostics see `s_ij` directly. The
normalization precedes RoPE, which is orthogonal, so the order is immaterial. The
log scale `γ` is the `qk_norm_log_scale` array, `[1]` with `qk_norm_sharing: global`
or `[num_layers, num_heads, 1]` with `per_head` (default), FP32, without weight
decay, initialized at `log(qk_norm_initial_scale)`; `null` (default) selects
`sqrt(d_K)`, the scale of RMSNorm-style QK-norm at initialization. The learned scale
is log-parametrized so the optimizer moves it multiplicatively and it stays positive
(a negative scale is a sign flip that the key projection absorbs). Because cosine
scores ignore Q/K magnitudes, the model can sharpen attention only through `γ`,
and rescaling the Q or K projection leaves the operator unchanged. Every
validation records `γ` and `exp(γ)` for each layer and head in `metrics.jsonl`
under `attention_diagnostics.qk_norm_scale`; W&B receives their mean, minimum,
and maximum. Recipes and metadata identify the option with `qk_norm`,
`qk_norm_sharing`, `qk_norm_initial_scale`, and `qk_norm_eps`. GLA and degree 0
do not accept it.

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
cannot run on the current device. Explicit backend choices in recipes are
preserved.

An unfinished run may resume only when its latest checkpoint covers its recorded
committed progress. Missing or older checkpoints cause an error before progress
metadata or metrics are overwritten. Restore the matching artifacts or use a new
run directory; an existing scientific record is never reused for a fresh restart.
An initialized run without committed progress can still start from step zero, and
completed runs retain their existing validation-best-only handling.

Softmax uses stable score shifting. Taylor and learned polynomials use unshifted
scores. Unnormalized exponential and Taylor kernels also use unshifted scores
and never divide by a row sum. Nonfinite outputs, gradients, and updated parameters are surfaced as
failures; diagnostic-only weight sums may underflow or overflow. Signed mechanisms retain their existing signed-normalizer rules; the
scaffold does not add clipping or denominator epsilon. Every attention mechanism
checks the gradient norm and all updated weights on every training step, including
gradient accumulation and distributed execution. An invalid update stops before
validation or checkpointing and writes a numerical-failure diagnostic. The legacy
`check_all_gradients` step argument controls additional returned diagnostics only;
it cannot disable these checks.

## Environment and training API

Each worktree syncs its own environment from this scaffold's lock; hardlinks share
package files with the uv cache, so an additional environment costs almost no disk:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink

# Tiny synthetic CPU check; both outputs are disposable.
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m scaffolds.transformer.train \
  scaffolds/transformer/configs/smoke.yaml \
  --workdir .cache/smoke/run --checkpoint-dir .cache/smoke/checkpoints
```

For a new experiment, start with `configs/base.yaml`, store its exact recipe in
`experiments/<experiment>/configs/`, and set the data manifests and scientific
parameters. Recipes use repository-relative outputs,
`experiments/<experiment>/runs/<run>` and `experiments/<experiment>/checkpoints/<run>`,
so a run writes inside the worktree that launched it. Keep `train.source_commit: auto`;
launchers export the launch commit. Inside a GPU allocation the common CLI is:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.transformer.train \
  experiments/my_experiment/configs/run.yaml
```

`--attention-degree` accepts a nonnegative degree or `softmax`; `--qk-head-dim`
selects the projection width. `--checkpoint-dir` separates large model state from
scientific records. `--stop-after-step` writes a resumable checkpoint while keeping
the recipe's full learning-rate horizon. Resume is the default, and Slurm requeue
also resumes; `--no-resume` applies only to a fresh attempt.

### Weights & Biases

`training/tracking.py` mirrors committed metrics to the private W&B project
`rafah-massachusetts-institute-of-technology/functional-memory`; `runs/<run>/metrics.jsonl`
remains the authoritative record, and a W&B failure never stops training. For a workdir
`experiments/<experiment>/runs/<run>`, the W&B group is the experiment, the name is the
run, and local W&B files go to `experiments/<experiment>/logs/<run>/wandb/`
(`train.wandb_group`, `wandb_run_name`, and `wandb_dir` override these). Tags give the
attention kernel, d_K, degree, GPU count and type, and cluster; the config holds the
full recipe plus a separate `runtime` section. Metrics use `train/` (every log step
and each validation window), `val/`, and `attention/<kernel>/layerNN/<statistic>`
against a `step` axis, so updates relogged after a resume are kept; terminal and
validation-best metrics go to the run summary. `wandb_run.json` in the run directory
pins the W&B run id across requeues.

Recipes leave `train.wandb_mode` at `disabled`; the Kennel launcher passes
`--wandb-mode online` (set `TRANSFORMER_WANDB_MODE=offline` or `disabled` when
submitting to change it), and acceptance pilots never log. Saved workspace views are
experiment analysis code built with the `views` dependency group
(`uv sync --project scaffolds/transformer --locked --extra cuda --group views`).

## Launch, acceptance, retention, and analysis

Experiments configure these shared modules from their own `adapter/`:

- `launchers/kennel.sbatch` runs any experiment on Kennel: it resolves the allocated
  GPU type, checks the frozen execution set against the launch commit
  (`scaffolds/launch/frozen.py`), uses the worktree's environment, pins `CUDA_HOME`
  to the locked CUDA toolkit, runs acceptance, then trains; logs append across requeues.
- `acceptance.py` runs identity-keyed acceptance in the production allocation: CPU
  tests, distributed CPU tests on two forced host devices, and single-GPU kernel tests
  (no skips), an initialization hash, a full-recipe pilot, and an exact fresh-process
  resume. An adapter supplies an `AcceptanceSpec` (tests, recipe check,
  required backend, initialization check, runtime environment) and nothing else.
- `retention.py` prunes a completed run to its validation-best checkpoint and records
  the selection.
- `analysis.py` reads committed validation events and retained artifacts and saves
  figures deterministically, so summaries and figures regenerate byte-identically.

Kernel and throughput benchmark scripts live in `benchmarks/`; their results and the
scaffold's hardware qualification live in [`validation/`](validation/README.md).

## Tests

```bash
JAX_PLATFORMS=cpu scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml scaffolds/transformer/tests \
  --import-mode=importlib -m 'not gpu'

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml scaffolds/transformer/tests \
  --import-mode=importlib -m 'not gpu' -k distributed
```

The suite covers independent FP64 attention and gradient references, signed and
unnormalized kernels, QK-norm cosine scores, learned parameter sharing, optimizer exclusions, distributed
accumulation, packed-data cursors, interruption and resume, checkpoint audits,
acceptance, and retention. GPU-marked tests (`-m gpu`, in a GPU allocation) exercise
the fused forward and backward implementations.

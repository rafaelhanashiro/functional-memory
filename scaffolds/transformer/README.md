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
shared implementation ancestry; a run's own source commit is in its `record.json`.

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
`learned_quadratic.py`, `learned_polynomial.py`, and `unnormalized.py`. Their implementations live under `attention/backends/`.
Fused kernels consume Q/K/V directly and retain their custom backward passes.
The interface does not require a materialized attention matrix.

The existing scientific configuration fields remain supported:

| Settings | Mechanism |
|---|---|
| `attention_degree: null`, `attention_normalization: true` (default) | Exact softmax |
| `attention_degree: null`, `attention_normalization: false` | Unnormalized causal `exp(QKᵀ / sqrt(d_K)) V` |
| Nonnegative `attention_degree`, `attention_normalization: false` | Unnormalized fixed Taylor `p_m(QKᵀ / sqrt(d_K)) V`; odd degrees permit signed weights |
| `attention_degree: m >= 1`, `attention_normalization: false`, `attention_coefficients: learned` | Unnormalized `sum_r c_r (QKᵀ / sqrt(d_K))^r V` with every coefficient learned from `1/r!`; signed |
| `attention_degree: m >= 1`, `attention_normalization: true`, `attention_coefficients: learned` | Normalized `sum_r c_r s^r V / rowsum` with every coefficient learned from `1/r!`; signed weights and row sums; fused or blocked |
| Even `attention_degree: m >= 2`, `attention_normalization: true`, `attention_coefficients: learned`, `attention_coefficient_parameterization: sum_of_squares` | Normalized `(q1(s)^2 + q2(s)^2) V / rowsum` with `deg q1 = m/2`, `deg q2 = m/2 - 1` learned and initialized to reproduce `T_m` exactly; nonnegative weights, positive row sums; fused or blocked |
| `attention_degree: m >= 1`, `attention_normalization: true`, `attention_coefficients: learned`, `attention_coefficient_parameterization: rational` | Normalized `p_a(s) V / sum_j p_b(s_j)` with a learned numerator `p_a(s) = sum_r a_r s^r` and a learned denominator with constant 1 of degree `attention_denominator_degree` (default `m`): signed `1 + sum_r b_r s^r`, or nonnegative `(1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2` (`attention_denominator_parameterization: sum_of_squares`); fused or blocked |
| `attention_degree: 0` | Uniform causal-prefix averaging |
| Nonnegative `attention_degree`, `quadratic_mode: fixed` | Fixed Taylor polynomial; odd degrees permit signed weights |
| `attention_degree: 2`, `quadratic_mode: positive` | Learned `0.5 * (1 + (a*s+b)^2)` |
| `attention_degree: 2`, `quadratic_mode: unrestricted` | Learned `1 + b*s + c*s^2` |

Fixed polynomials beyond Taylor use `attention_coefficients: fixed` and
`attention_fixed_coefficients: [c0, ..., cm]`, the ascending monomial coefficients.
For an even-degree nonnegative normalized kernel, use
`attention_coefficient_parameterization: sum_of_squares` and
`attention_fixed_square_factors: [q1_0, ..., q1_{m/2}, q2_0, ..., q2_{m/2-1}]`
instead. The kernel evaluates `q1(s)^2 + q2(s)^2` directly to avoid cancellation
from expanding near a real root. These two fixed representations are mutually
exclusive and create no polynomial parameters or optimizer state. They use the
generic polynomial backends and the same default QK normalization.

With `attention_coefficient_parameterization: sum_of_squares` the learned array is
`polynomial_square_factors` instead (`attention/nonnegative_polynomial.py`): the same
`m + 1` values per layer and head, holding the ascending coefficients of `q1` then
`q2`. Every polynomial of even degree that is nonnegative on the real line has this
form, so the parameterization is complete and adds no parameters. The kernel is
evaluated as `q1(s)^2 + q2(s)^2` rather than through expanded coefficients, whose
cancellation near a zero minimum could round below zero. Kernel values may be
exactly zero; a zero row sum is invalid. Diagnostics report the factors as raw
parameters and the coefficients of `q1^2 + q2^2` as the effective ones.

With `attention_coefficient_parameterization: rational` the weights are
`p_a(s_ij) / sum_(j<=i) p_b(s_ij)` (`attention/rational_polynomial.py`). The numerator
`a_0..a_m` is `polynomial_coefficients`. The denominator has degree `d`
(`attention_denominator_degree`, default `m`) and a fixed constant 1, which removes the
common scale of the two polynomials. `attention_denominator_parameterization` selects
its form:

| Form | Learned array | Denominator |
|---|---|---|
| `direct` (default) | `polynomial_denominator_coefficients`, `b_1..b_d` | `1 + sum_r b_r s^r`, signed |
| `sum_of_squares`, even `d = 2k` | `polynomial_denominator_square_factors`, `u_1..u_k, w_1..w_k` | `(1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2`, nonnegative |

Every nonnegative polynomial of degree `2k` with constant 1 has the second form, so it
is complete. Arrays are shared or `[num_layers, num_heads, ·]` per head. A negative
nonzero denominator sum is valid; an exactly zero or nonfinite sum poisons its row, with
no clipping or epsilon. Diagnostics report the numerator as `kernel_*`, the denominator
values as `denominator_*`, the cancellation ratio of the denominator sums, and the
expanded denominator coefficients `b_r` of either form.

`attention_coefficient_initialization` chooses where learned coefficients start:
`taylor` (default) sets every learned polynomial to `sum_r s^r / r!`, so a rational
kernel with equal degrees equals normalized Taylor attention at initialization; `linear`
sets `a_1 = 1` and every other coefficient to 0, so a direct kernel starts at `s` and a
rational kernel at linear attention divided by the number of visible keys,
`sum_(j<=i) s_ij v_j / (i + 1)`. A sum-of-squares denominator with all `w_r = 0` gives
every `w_r` a zero gradient, so its linear start sets `w_1 = 10^-3`, a denominator of
`1 + 10^-6 s^2`. Sum-of-squares kernels always start at `T_m`.

`optim.coefficient_penalty` adds `lambda` times
`sum over polynomials [a_0^2 + sum_(r>=2) r^2 a_r^2 + sum_(r>=1) r^2 b_r^2]` to the
training loss (`training/regularization.py`), for learned direct or rational
coefficients; `a_1` is unpenalized and the sum runs over every learned coefficient
set (each layer and head with `per_head` sharing).
`optim.coefficient_penalty_denominator: false` drops the `b_r` term, penalizing the
numerator only; a sum-of-squares denominator requires it. The recorded training NLL,
perplexities, and the validation loss exclude the penalty; its value at the logged
step's parameters is printed as `train_coefficient_penalty`, stored as
`train_coefficient_penalty` in each validation record, and mirrored to W&B as
`train/coefficient_penalty`. AdamW weight decay never touches polynomial
coefficients, so the penalty is the only shrinkage they receive.

`qk_head_dim` controls projection geometry independently of the mechanism.
`quadratic_sharing: global` stores one pair across all layers and heads;
`per_head` stores `[num_layers, num_heads, 2]`. Each layer keeps its own projections.
Learned polynomial coefficients are the `polynomial_coefficients` array: `[m + 1]`
with `attention_coefficient_sharing: global`, `[num_layers, num_heads, m + 1]` with
`per_head`, FP32, without weight decay. The blocked reference evaluates them by
Horner's rule with or without normalization. Without normalization,
`attention/backends/fused_learned_polynomial.py` is the fused kernel, whose custom
backward also returns each head's coefficient gradient `sum_{i>=j} S_ij^r (dY_i · V_j)`.
With normalization (`attention/learned_polynomial.py`),
`attention/backends/fused_normalized_polynomial.py` supports direct coefficients,
sum-of-squares factors, and rational numerator/denominator pairs. Direct and rational
coefficients use a two-pass forward calculation that divides by the final signed row
sum (of the denominator polynomial for rational kernels); nonnegative factors use
one-pass online normalization. Both evaluate causal tiles without storing a sequence-square matrix,
retain FP32 output for the custom backward, and differentiate every learned value
through both the numerator and denominator. Factor evaluation stays in sum-of-squares
form. Parameter gradients are reduced from query-tile partials without atomics;
globally shared arrays receive the sum across heads and layers through ordinary JAX
broadcast differentiation. `attention_backend: auto` selects fused execution on
supported NVIDIA GPUs and the blocked reference otherwise. The blocked reference
provides diagnostics while the fused output continues through subsequent layers.

### QK-norm (cosine scores)

QK-norm is required for new dot-product experiments unless the user explicitly
requests plain QK scores. A recipe that leaves `qk_norm` unset gets `true`; frozen
new recipes record `qk_norm: true`. To run an explicitly requested plain-score
experiment, set `qk_norm: false` and pass `--allow-plain-qk` when allocating it with
`python -m scaffolds.records create`. This prevents copied legacy recipes from
silently retaining plain `q·k / sqrt(d_K)` scores. Historical recipes and resumptions
retain their recorded settings. GLA rejects `qk_norm: true`, and degree 0, which
never reads Q or K, resolves
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
do not use it.

### Unnormalized output scaling

`attention_output_scaling` rescales each head's unnormalized output
`o_t = sum_(j<=t) k(s_tj) v_j` before the output projection
(`attention/output_scaling.py`); only dot-product attention accepts it.
Normalized attention equals `o_t / sum_j k(s_tj)`, so `o_t` is the normalized output
times the row's kernel sum. `head_rms_norm` also applies to normalized attention,
where it normalizes `o_t / sum_j k(s_tj)`; for a positive kernel such as exp the two
agree up to `rms_norm_eps`. `inverse_position` requires unnormalized attention.

| `attention_output_scaling` | Output |
|---|---|
| `none` (default) | `o_t` |
| `inverse_position` | `o_t / t`, with `t` the number of visible keys |
| `head_rms_norm` | `w ⊙ o_t / sqrt(mean(o_t²) + rms_norm_eps)` over the head's value channels |

`inverse_position` removes the kernel sum's growth with `t`, not its dependence on
the scores. `head_rms_norm` removes any positive per-row scale, so for a positive
kernel it equals the RMSNorm of the normalized output; unlike a row sum, its
divisor cannot be driven through zero by a signed kernel. Its gain `w` is the
`output_norm_scale` array `[layers, heads, value_head_dim]`, initialized at 1 and
excluded from weight decay. Run metadata and W&B tags name the option.

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

### Weight initialization

`model.initialization` selects the weight initialization. New runs use their
mechanism's reference scheme: `python -m scaffolds.records create` writes it into the
frozen recipe when the field is unset, and an explicit `flax_default` (the scaffold's
Flax draws: Linear kernels truncated normal with variance `1/fan_in`, embeddings
variance `1/features`) needs `--allow-flax-default-init`. The config default stays
`flax_default`, so recipes from before the option, which omit it, keep their meaning. A
reference scheme re-draws the
weights after construction from one extra draw of the run's parameter key, keeping
parameter metadata, norm scales of one, zero biases, and the declared initial values of
mechanism-owned attention parameters (`models/initialization.py`).

| `initialization` | Mechanism | Weights |
|---|---|---|
| `flax_default` | any | Flax defaults |
| `transformer_reference` | dot-product | N(0, 0.02); attention `o_proj` and MLP `down_proj` N(0, 0.02/√(2L)) (GPT-2/Megatron) |
| `gla_reference` | GLA | N(0, 0.02); query, key, and both low-rank gate projections Xavier-uniform, gain 2^-2.5 (official GLA code) |
| `gated_deltanet_reference` | Gated DeltaNet | Embeddings and output head N(0, 0.02); Linear weights Uniform(±1/√fan_in); MLP `down_proj` also divided by √(2L) (official `lit_gpt`, `mamba_init`) |

### Gated DeltaNet

`attention_mechanism: gated_deltanet` replaces the attention module with Gated DeltaNet
(`attention/gated_deltanet.py`), mirroring flash-linear-attention's layer. Queries, keys,
and values each pass a causal depthwise short convolution (`gdn_conv_size`, default 4)
and SiLU; queries and keys are L2-normalized per head. Each head keeps a
`qk_head_dim × value_head_dim` state, `S_t = a_t S_{t-1} + β_t k_t (v_t − a_t S_{t-1}ᵀ k_t)ᵀ`,
read inclusively as `S_tᵀ q_t / √d_K`, with `β_t = sigmoid(W_b x_t)` and the Mamba-2 decay
`a_t = exp(−exp(A_log) softplus(W_a x_t + dt_bias))` (`A_log`, `dt_bias` without weight
decay). The read passes `RMSNorm(o_t) · w · silu(W_g x_t)` (one `w` shared by the heads)
and the output projection. There is no positional encoding and no QK-norm.

`blocked` is a token-by-token FP32 reference. `fused`
(`attention/backends/fla_gated_delta_rule.py`) launches FLA's own chunked Triton kernels,
imported unmodified from the pinned `flash-linear-attention` in the CUDA extra, through
the vendored bridge: thin wrappers reorder arguments into the bridge's convention, the
forward and backward follow FLA's launch sequence under `jax.custom_vjp`, and the two
chunk-local cumulative sums run in JAX. FLA reads the GPU through torch, which the
environment installs CPU-only to satisfy imports, so the glue applies FLA's GPU choices
itself: TF32 triangular solves, its Hopper and Blackwell autotune menus, and tile sizes
from the device's shared memory. Triton must be ≥ 3.7.1 on Hopper (FLA #640). The fused
path requires an sm90+ GPU and prediction length divisible by 64; `auto` selects it there
and the reference elsewhere. It passes its GPU tests on H100 and B300.

### Gated linear attention

`attention_mechanism: gla` replaces the attention module inside the same residual
blocks. `kind: transformer` continues to select the shared language-model backbone;
the default mechanism `dot_product` preserves all existing recipes. GLA uses
`qk_head_dim` and `value_head_dim` independently. Its vector-gated recurrence is
`S_t = diag(sigmoid(g_t) ** (1 / tau)) S_{t-1} + k_t v_t^T`, followed by the
inclusive read `q_t^T S_t`, per-head RMSNorm, a SiLU output gate, and an output
projection. Each sequence starts with zero state. No score scaling or attention
normalization is applied. The OFM defaults are gate rank 16, tau 16, RoPE
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
The bridge hands Triton's PTX to jaxlib, which assembles it with the first `ptxas`
among its CUDA roots; that is a node's `/usr/local/cuda` unless `CUDA_HOME` names
another toolkit, so launchers set `CUDA_HOME` to the environment's `nvidia/cuda_nvcc`.
Triton 3.7 picks its PTX version from its own bundled assembler (CUDA 13.1 for
Blackwell), newer than the locked toolkit reads, so launchers also point
`TRITON_PTXAS_PATH` and `TRITON_PTXAS_BLACKWELL_PATH` at the locked `ptxas`. Operands are converted to FP32 before launch:
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

For a new study, start with `configs/base.yaml`, set its data manifests and scientific
parameters, and allocate a run with `scaffolds.records.create_run`. The study's
`sweep.yaml` contains common setup and sweep axes; each allocated
`experiments/<study>/runs/<run_id>/config.yaml` contains the complete input configuration.
Commit that file before submission. Its bytes stay fixed throughout execution.
Recipes use repository-relative `train.workdir` paths and the default colocated
`checkpoints/`, so artifacts stay in the worktree that launched the run. Keep
`train.source_commit: auto`; launchers export the launch commit. Inside a GPU
allocation the common CLI is:

```bash
scaffolds/transformer/.venv/bin/python -m scaffolds.transformer.train \
  experiments/my_experiment/runs/r0123456789ab/config.yaml
```

`--attention-degree` accepts a nonnegative degree or `softmax`; `--qk-head-dim`
selects the projection width. Scientific overrides require allocating a fresh run;
an existing run's complete recipe is checked before training. `--stop-after-step`
writes a resumable checkpoint while keeping
the recipe's full learning-rate horizon. Resume is the default, and Slurm requeue
also resumes; `--no-resume` applies only to an uninitialized run. A fresh execution
gets a fresh run ID even when its scientific settings match a previous run.

`record.json` summarizes status, source, execution attempts, acceptance, runtime,
resolved configuration and backend, data identity, and retained checkpoints.
`config.json`, `metadata.json`, completion and retention files are raw trainer
outputs retained for exact resume validation. Resolving `auto` settings updates
these execution records while preserving `config.yaml`. Metrics stay in
`metrics.jsonl`; logs, checkpoints and W&B files are colocated ignored artifacts.
Shared helpers regenerate the study's `runs.csv` index. Figures and comparisons
belong under `results/<comparison>/` and select run IDs explicitly.

### Weights & Biases

`training/tracking.py` mirrors committed metrics to the private W&B project
`rafah-massachusetts-institute-of-technology/functional-memory`; `runs/<run_id>/metrics.jsonl`
remains the authoritative record, and a W&B failure never stops training. For a workdir
`experiments/<experiment>/runs/<run_id>`, the W&B group is the experiment, the name is the
run, and local W&B files go to `experiments/<experiment>/runs/<run_id>/wandb/`
(`train.wandb_group`, `wandb_run_name`, and `wandb_dir` override these). Tags give the
attention kernel, d_K, degree, GPU count and type, and cluster; the config holds the
full recipe plus a separate `runtime` section. Metrics use `train/` (every log step
and each validation window), `val/`, and `attention/<kernel>/layerNN/<statistic>`
against a `step` axis, so updates relogged after a resume are kept; terminal and
validation-best metrics go to the run summary. `wandb_run.json` in the run directory
pins the W&B run id across requeues.

Recipes leave `train.wandb_mode` at `disabled`; the Kennel and ORCD launchers pass
`--wandb-mode online` (set `TRANSFORMER_WANDB_MODE=offline` or `disabled` when
submitting to change it), and acceptance pilots never log. Saved workspace views are
results code built with the `views` dependency group
(`uv sync --project scaffolds/transformer --locked --extra cuda --group views`).

## Launch, acceptance, retention, and analysis

Studies configure these shared modules with execution modules at their root
(for example `experiments/<study>/acceptance.py`):

- `launchers/kennel.sbatch` runs any experiment on Kennel: it resolves the allocated
  GPU type, checks the frozen execution set against the launch commit
  (`scaffolds/launch/frozen.py`), uses the worktree's environment, pins `CUDA_HOME`
  to the locked CUDA toolkit, runs acceptance, then trains; `training.log` appends
  across requeues. The frozen set includes study `sweep.yaml`, every run's
  `config.yaml`, root execution modules, and shared record/launch helpers. Metrics,
  records and `results/` remain editable while jobs run.
- `launchers/orcd.sbatch` does the same on ORCD. Its allocations fit that cluster's
  per-user GPU caps and 6-hour limit, and a job requeues itself before the limit
  (see [CLUSTERS.md](../../CLUSTERS.md)).
- `acceptance.py` runs identity-keyed acceptance in the production allocation: CPU
  tests, distributed CPU tests on two forced host devices, and single-GPU kernel tests
  (no skips), an initialization hash, a full-recipe pilot, and an exact fresh-process
  resume. `attention_spec(config, entry_module=...)` selects correctness and kernel
  tests from the recipe and resolved backend, plus the initialization check and
  runtime environment. A study execution module passes this factory to `main`;
  specialized studies can supply an `AcceptanceSpec` directly. Launchers invoke
  `<acceptance_module> <config.yaml> --runtime-env` before starting workers so GLA
  bridge settings apply only to GLA recipes.
- `retention.py` prunes a completed run to its validation-best checkpoint and records
  the selection.

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

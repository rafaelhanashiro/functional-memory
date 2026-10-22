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
| The preceding settings plus `attention_coefficient_conditioning: query`, `attention_coefficient_activation: tanh`, `attention_coefficient_sharing: per_head` | Query-dependent signed coefficients `c_r(x_t) = b_r + a_r tanh(w_r^T x_t)`, independently per layer and head; direct parameterization, fused or blocked |
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
(`attention_denominator_degree`, default `m`). `attention_denominator_parameterization`
selects its form and `attention_denominator_constant` whether its constant is fixed at 1
(default), which removes the common scale of the two polynomials, or learned:

| Form, constant | Learned array | Denominator |
|---|---|---|
| `direct`, `fixed` (default) | `polynomial_denominator_coefficients`, `b_1..b_d` | `1 + sum_r b_r s^r`, signed |
| `direct`, `learned` | `polynomial_denominator_coefficients_with_constant`, `b_0..b_d` | `sum_r b_r s^r`, signed |
| `sum_of_squares`, `fixed`, even `d = 2k` | `polynomial_denominator_square_factors`, `u_1..u_k, w_1..w_k` | `(1 + sum_r u_r s^r)^2 + (sum_r w_r s^r)^2`, nonnegative |
| `sum_of_squares`, `learned`, even `d = 2k` | `polynomial_denominator_square_factors_with_constant`, `q1_0..q1_k, q2_0..q2_(k-1)` | `q1(s)^2 + q2(s)^2`, nonnegative |

Every nonnegative polynomial of degree `2k` (with constant 1, for the fixed form) has the
sum-of-squares form, so it is complete. Arrays are shared or `[num_layers, num_heads, ·]` per head. A negative
nonzero denominator sum is valid; an exactly zero or nonfinite sum poisons its row, with
no clipping or epsilon. Diagnostics report the numerator as `kernel_*`, the denominator
values as `denominator_*`, the cancellation ratio of the denominator sums, and the
expanded denominator coefficients `b_r` of either form.

`attention_coefficient_initialization` chooses where learned coefficients start:
`taylor` (default) sets every learned polynomial to `sum_r s^r / r!`, so a rational
kernel with equal degrees equals normalized Taylor attention at initialization; `linear`
sets `a_1 = 1` and every other coefficient to 0, so a direct kernel starts at `s` and a
rational kernel at linear attention divided by the number of visible keys,
`sum_(j<=i) s_ij v_j / (i + 1)`. A sum-of-squares denominator with constant 1 and all
`w_r = 0` gives every `w_r` a zero gradient, so its linear start sets `w_1 = 10^-3`, a
denominator of `1 + 10^-6 s^2`; with a learned constant it starts exactly at 1 as
`q1 = q2 = 1/sqrt(2)`. Sum-of-squares kernels always start at `T_m`.

`optim.coefficient_penalty` adds `lambda` times
`sum over polynomials [a_0^2 + sum_(r>=2) r^2 a_r^2 + sum_(r>=1) r^2 b_r^2]` to the
training loss (`training/regularization.py`), for learned direct or rational
coefficients; `a_1` is unpenalized and the sum runs over every learned coefficient
set (each layer and head with `per_head` sharing).
`optim.coefficient_penalty_denominator: false` drops the `b_r` term, penalizing the
numerator only; a learned denominator constant requires it. A sum-of-squares
denominator is penalized through its expanded coefficients `b_r`, differentiated through
its factors. The recorded training NLL,
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

### Projecting exp onto a scalar function basis

`attention/basis.py` fits a fixed approximation `k(s) = sum_j c_j b_j(s)` to
`exp(s)` under `s ~ N(mean, std²)`. The basis can contain monomials, Gaussian
RBFs, sines, cosines, mixtures of these, or vectorized Python callables. This
changes the scalar function applied to query–key scores; it does not construct
expanded query/key features or linearize attention.

```python
import jax.numpy as jnp
from scaffolds.transformer.attention.basis import fit_exp, monomial, rbf, sine, cosine

# Named functions support both interactive evaluation and saved run recipes.
fit = fit_exp([monomial(0), rbf(-2, 1.5), rbf(0, 1.5), rbf(2, 1.5)],
              mean=0.0, std=1.0)
print(fit.coefficients, fit.diagnostics)
values = fit(jnp.linspace(-4, 4, 101))

# Callables provide the same fitting API; use JAX operations if the result
# will be differentiated or compiled. Scalar constant returns are supported.
interactive = fit_exp([lambda s: 1.0, lambda s: jnp.sin(s), lambda s: jnp.cos(s)])
```

For an interactive attention calculation, pass the fitted callable as
`attention_core(q, k, v, backend="blocked", score_kernel=interactive,
attention_normalization=True)` from `attention/core.py`, using post-RoPE Q/K/V
arrays with shape `[batch, time, heads, channels]`. Automatic differentiation
passes through the fixed function to Q and K.

`rbf(center, width)` means `exp(-0.5 * ((s-center)/width)**2)`; `sine(frequency,
phase=0)` and `cosine(frequency, phase=0)` use radians. `monomial(power)` includes
the constant at power zero. A named specification is a dictionary such as
`{"kind": "rbf", "center": 0.0, "width": 1.5}`. Unknown kinds and parameters are
rejected; specifications never execute strings as Python code.

Fitting runs once on the CPU in float64. Gauss–Hermite quadrature discretizes
the Gaussian expectation, and a column-scaled SVD least-squares solve finds the
projection in the original basis. The functions need not be orthogonal. A zero
function or numerical linear dependence raises an error instead of silently
discarding terms. The default fit uses 64 quadrature nodes; a separate 128-node
fit checks the change in the fitted function. `quadrature_order` and `check_order`
are configurable, with `basis size <= quadrature_order < check_order <= 256`.

Diagnostics include estimated absolute and relative L2 error, scaled-design
condition numbers, relative change under the higher quadrature order, and an
estimate of the Gaussian mass where the approximation is negative. The
`quadrature_converged` flag requires both relative function change and relative
error in the known exponential L2 norm to be at most `1e-7`.
These are numerical checks, not error bounds for arbitrary functions: narrow
RBFs or rapid oscillations may need more quadrature nodes. Increasing the basis
span and resolving its integrals are separate requirements.

Freeze the named basis, coefficients, Gaussian measure, fitting settings and
diagnostics into the input recipe before allocating a training run:

```python
from scaffolds.transformer.configs import Config, ModelConfig, save_config

config = Config(model=ModelConfig(
    attention_basis=fit.to_config(),
    attention_backend="blocked",  # auto also resolves to blocked on CPU or GPU
    attention_normalization=True,
    qk_norm=True,
    initialization="transformer_reference",
))
save_config(config, "basis-recipe.yaml")
```

Fill in the study's data and training recipe before allocating with
`scaffolds.records create`. Saved fits are restored without refitting, and resume
checks include the entire specification. Anonymous callables cannot be serialized;
use named functions for recorded training. The fitted coefficients are fixed and
shared across layers and heads and add no trainable parameters. They exclude the
existing polynomial degree, polynomial coefficient, and query-conditioning options.

Normalized execution computes `sum_(i<=t) k(s_ti) v_i / sum_(i<=t) k(s_ti)`;
unnormalized execution omits the denominator and supports the existing output
scaling options (use head RMSNorm for new unnormalized runs). Scores are unshifted.
The fit need not be positive: signed weights and negative nonzero sums are valid;
zero or nonfinite denominators produce explicit numerical failures, without
clipping or epsilon. Projection minimizes scalar-kernel L2 error, not normalized
attention-output error. The blocked JAX path implements arbitrary bases;
`attention_backend: triton_scalar` also supports frozen named bases on H100.
Explicit fused, Mosaic, XLA-softmax or cuDNN requests are rejected. CPU tests cover
analytic projections, forward/backward attention and training/resume; production
GPU use requires the usual hardware qualification.

### Native Triton scalar attention

`attention/backends/triton_scalar.py` exposes `scalar_attention(q, k, v,
kernel=...)` for causal BTHD operands. `ScalarKernel` contains an inlined
`@triton.jit` callback `(scores, parameters, WITH_GRAD) -> (value, derivative)`
and immutable compile-time parameters. `exponential_kernel`,
`polynomial_kernel`, and `basis_kernel` provide exponential, fixed polynomial,
and frozen named-basis callbacks. Function parameters are fixed; the custom
backward differentiates Q, K, V and the optional per-head RMSNorm gain.
`uniform_rbf_kernel` is an explicit alternative for equally spaced Gaussian
centers with a common width: two exponentials initialize a product recurrence
for the unchanged fitted sum and derivative. It starts at the endpoint nearer
each score and rejects spans wider than 24 standard deviations. The generic
`basis_kernel` evaluates each basis function independently.

The tiled implementation is adapted from Triton's official fused-attention
tutorial, pinned in the kernel source. Forward uses tensor descriptors; dQ and
dK/dV recompute scalar values and derivatives without retaining the attention
matrix. The bridge supplies descriptor scratch as a private XLA result, whose
lifetime also covers CUDA graph execution.

The modes are unnormalized attention, normalized attention
(`attention_normalization=True`), and unnormalized attention fused with per-head
RMSNorm (`output_norm_scale=gain`). The last two enable streaming accumulation
by default; `streaming=False` measures its cost independently. Streaming keeps
`Y = exp(log_scale) * accumulator`, with positive scale `1 + sum(abs(weights))`.
RMSNorm uses `eps * exp(-2*log_scale)`, so retaining epsilon does not change the
mathematics. Exponential evaluation shifts scores; other callbacks must produce
finite values on valid scores. Streaming controls magnitude, not cancellation.
Normalized exponential uses the positive weight mass without the unit prior,
preserving softmax stability for entirely negative rows as well. Normalized
signed weights still require a nonzero signed sum.

Supported operands have BF16 or FP32 dtype and head widths 16, 32, 64 or 128.
BF16 execution rounds weights and backward adjoints for tensor-core products;
`precision="tf32x3"` retains FP32 operands for those products. The model's explicit
`triton_scalar` backend currently requires a frozen basis and fuses RMSNorm when
requested. `auto` remains blocked for basis attention. Diagnostic collection
uses an additional blocked calculation, so it is not part of the fused fast path.
Correctness and benchmark evidence belongs in the
[validation record](validation/README.md).

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

### Input-dependent polynomial coefficients

`attention_coefficient_conditioning: query` predicts signed coefficients
`c_0(x_t), ..., c_m(x_t)` from the current token's incoming attention activation,
after the block's input RMSNorm. It requires learned coefficients, direct
parameterization, per-head sharing, Taylor coefficient initialization, and
unnormalized dot-product attention. The default `static` keeps input-independent
coefficients.

`attention_coefficient_activation` selects the coefficient function:

| Activation | Coefficients | Degree |
| --- | --- | --- |
| `identity` (default) | `c_r(x_t) = b_r + w_r^T x_t` | `m = 1` |
| `tanh` | `c_r(x_t) = b_r + a_r tanh(w_r^T x_t)` | `m >= 1` |

Every layer, head, and polynomial order has independent parameters. There is no
bias inside `tanh`. For fixed parameters its coefficient lies within
`b_r +/- abs(a_r)`; both offsets and signed amplitudes learn, so this is not a
fixed bound throughout training. The output before output scaling is
`sum_(i<=t) sum_(r=0)^m c_r(x_t) s_ti^r v_i`. QK normalization, score scaling,
and RoPE define `s_ti` as in input-independent attention. Query conditioning
changes the read at `t`; it does not add write gates, forgetting, or delta corrections.

`query_polynomial_kernel` has shape `[layers, heads, hidden_size, m+1]`, while
`query_polynomial_bias` and the tanh-only `query_polynomial_amplitude` have shape
`[layers, heads, m+1]`. These FP32 model-owned arrays are excluded from weight
decay. The kernel starts at zero, the bias at `b_r = 1/r!`, and the tanh amplitude
at one, without consuming RNG draws. The backbone keeps its canonical
initialization and the initial attention kernel is the degree-`m` Taylor polynomial.
For tanh, `attention_coefficient_amplitude_initial` (default 1) sets the starting
`a_r`, and `attention_coefficient_projection_initial_std` (default 0) draws each kernel
entry from `N(0, std^2)`. The draw uses a mechanism stream derived from the seed
(`attention/parameters.py`), so the backbone is still the softmax control's and
acceptance regenerates the kernel exactly. Starting at `a_r = 0` with a random kernel
keeps the Taylor start, while the amplitude's gradient `tanh(w_r^T x_t)` is nonzero;
`a_r = 0` with a zero kernel would leave both gradients zero and is rejected.
`attention_coefficient_tanh_bias: true` adds a learned offset inside the tanh,
`c_r = b_r + a_r tanh(w_r^T x_t + beta_r)`, stored as `query_polynomial_tanh_bias`
(`[layers, heads, m+1]`, FP32, no weight decay) and starting at zero, so a constant
shift of the pre-activation need not be read from `x_t` through `w_r`.
`attention_coefficient_conditioning: query_key` (tanh only) multiplies each key's
degree-`r` term by a per-key factor, `sum_(i<=t) sum_r c_r(x_t) f_r(x_i) s_ti^r v_i` with
`f_r(x_i) = 1 + a^k_r tanh(w_k_r^T x_i)` from the key token's activation
(`key_polynomial_kernel` `[layers, heads, hidden_size, m+1]`, `key_polynomial_amplitude`
`[layers, heads, m+1]`; drawn and started like `W` and `a`, so the start is unchanged).
The pair coefficient factorizes, so nothing pairwise is stored: the blocked reference
scales each key column, the fused Triton kernel (`backends/fused_query_key_polynomial.py`)
forms `c_r(x_t) f_r(x_i)` inside each tile in one pass (its key-side backward also
accumulates the key-factor gradient), and the Mosaic query kernel runs once per order with
the factor folded into degree-specific values. The per-order path rounds each partial
output to V's dtype before summing, which at degree one corrupts the gradient after head
RMSNorm. Diagnostics report the key factors beside the query coefficients.

The tanh path uses a dedicated fused kernel (or its Mosaic GPU counterpart,
`attention_backend: mosaic`) that directly evaluates the query-dependent polynomial, with gradients for Q/K/V and every coefficient.
The historical identity path retains its linear polynomial kernel plus FP32
causal value sum. Both have an explicit blocked polynomial reference.
Per-head output RMSNorm remains an independent option through
`attention_output_scaling: head_rms_norm`. Validation diagnostics report the
actual input-dependent coefficient means, standard deviations, minima, and
maxima on the fixed diagnostic sample, separately from projection parameters.

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
itself: its Hopper and Blackwell autotune menus and tile sizes from the device's
shared memory. Chunk intermediates and backward partials use FP32, with TF32x3
products including the triangular solve. Keeping only FP32 accumulators while
rounding intermediate buffers to BF16 loses accuracy in repeated-key cancellation.
The bridge scopes the dot-product precision to each compilation and includes it in
the cache identity. Outputs and returned input gradients keep their original dtypes;
saved Q/K/V retain their original storage dtype and are promoted again in backward.
Triton must be ≥ 3.7.1 on Hopper (FLA #640). The fused
path requires an sm90+ GPU and prediction length divisible by 64; `auto` selects it there
and the reference elsewhere. Hardware qualification is recorded in
[validation](validation/README.md#independent-gdn-erasewrite-kernel).

`gdn_erase_mode: independent` gives erasure its own per-head gate
`lambda_t = sigmoid(W_lambda x_t)`, while `beta_t = sigmoid(W_b x_t)` controls writing:
`S_t = a_t S_{t-1} + k_t (beta_t v_t - lambda_t a_t S_{t-1}^T k_t)^T`.
The default `tied` mode preserves the original GDN update. The independent projection
is a separate parameter copied from the write projection at initialization. It consumes
no additional initialization RNG draws, and reference initialization re-copies it after
initializing the original weights, so all shared parameters match tied GDN exactly.
The fused independent path shares FLA's triangular solve, state, and read kernels;
local WY preparation and its backward pass use separate erase/write gates and gradients.
Neither pass divides by a gate, so zero erasure with nonzero writing is supported.
Training telemetry records both gates, their difference, raw attention output statistics,
and separate erase/write/decay/output-gate parameter-group norms.

`gdn_write_activation: tanh` changes the linear write to `beta_t = tanh(W_b x_t)`.
It requires independent erasure, which remains `lambda_t = sigmoid(W_lambda x_t)`.
The default `sigmoid` preserves historical recipes and outputs. Both choices use
exactly the same reference-initialized projection weights and RNG draws; changing
the activation changes the initial function. Signed writes range from -1 to 1 and
do not change decay, Q/K normalization, or gated output RMSNorm. The activation is
part of checkpoint compatibility, run identity, and the generated study index;
tanh runs carry the `tanh_write` W&B tag.

`gdn_polynomial_degree: 2` or `3` with independent erasure replaces the linear
memory by a polynomial function:
`F_t(q) = exp(g_t) F_{t-1}((I-lambda_t k_t k_t^T)q) + v_t sum_r beta_{t,r}(q^T k_t)^r`.
The linear coefficient retains `gdn_write_activation`; separate tanh projections
for degree zero and degrees two/three start at zero, preserving the linear
initial function. Q/K normalization, the final `1/sqrt(d_K)` read scale, and
gated output RMSNorm are unchanged. The blocked backend is an explicit tensor
reference. The fused Triton backend supports key/value widths up to 64 and
lengths divisible by 64; it transports queries through prior key chunks without
materializing polynomial features. Its work and temporary backward histories
grow quadratically with sequence length.

The fused polynomial backend defaults to FP32/TF32x3 products with two warps on
Blackwell and four on Hopper. Blackwell rejects higher warp counts because its
four-warp backward path fails hardware qualification with the pinned compiler.
Experimental runtime options `POLY_GDN_VALUE_PRECISION=bf16_read` (value-read
products) or `bf16_values` (also score-adjoint products) retain FP32 transport,
inverse geometry, and accumulation. `POLY_GDN_NUM_WARPS` accepts 1, 2, 4, or 8.
`POLY_GDN_MIN_CTAS_PER_SM` accepts 0 (the default, no guidance), 1, or 2;
positive values add PTX occupancy guidance to the polynomial kernels, allowing
the assembler to make a different register-allocation choice.
Explicit `precision`/`num_warps`/`min_ctas_per_sm` keyword arguments on
`polynomial_gated_delta_rule` override these settings; environment changes take
effect when tracing a fresh executable. Backend metadata records the resolved
options. Qualification and throughput measurements belong in
[validation](validation/README.md).

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

Standard GLA/GDN training leaves persistent bridge caching disabled; keep
`EJKERNEL_TRITON_CACHE_COMPILES` and its legacy `EASYDEL_TRITON_CACHE_COMPILES`
alias unset. Benchmark launchers that enable it use a cache directory per frozen
source commit because the bridge's source key excludes called helper functions.
Standard acceptance sets `EJKERNEL_TRITON_CACHE_DIR` itself, overriding an external
directory setting; enabling persistence there requires a launcher/acceptance
change that preserves this source isolation.

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
can omit it. CPU execution selects the blocked reference. On H100 and B300, recipes with
a Mosaic GPU kernel (below) select it. Other supported NVIDIA GPU
polynomial, learned, and unnormalized exponential/Taylor shapes select the fused kernels; exact softmax selects an
appropriate public JAX implementation. Unsupported shapes use the reference when
automatic selection is requested. Explicit overrides are validated and never
silently replaced.

`attention_backend: mosaic` selects the Pallas Mosaic GPU kernels
(`attention/backends/mosaic/`): Hopper `wgmma` kernels on H100 (sm90) and Blackwell
`tcgen05`/tensor-memory kernels on B300 (sm10x), with the mathematics, FP32
accumulation, and NaN failure rows of the fused Triton kernels. Weights and score
gradients enter the tensor cores rounded once to BF16, as in FlashAttention and cuDNN,
except for normalized odd-degree Taylor attention, whose cancelling row sums keep
FP32-precision weights (`api.default_precision`; recorded as
`attention_weight_precision`). They cover normalized fixed Taylor attention (degree >= 1),
tanh query-conditioned polynomial attention (degree 1-8), and unnormalized polynomials with
static fixed or learned coefficients (degree 1-8, on the query-coefficient kernel with every
query's coefficients equal), with Q/K width 8, 16, 32, 64, or
128, value width 64, `train.dtype: bfloat16`, and a prediction length divisible by 128
(H100) or 256 (B300). An explicit `mosaic` request raises for other recipes, GPUs, or
shapes; `auto` then uses the fused kernels. Diagnostics come from the blocked reference,
as for the fused kernels. They are the canonical kernels for these mechanisms, and
`scaffolds.records create` rejects a fresh recipe for them that requests `fused` unless
`--allow-triton-attention` is supplied. Acceptance runs the Mosaic GPU tests only at the
run's weight precision (`--mosaic-precision`).

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
validation or checkpointing and writes a numerical-failure diagnostic. Training the run
again from a checkpoint supersedes that failure: the trainer removes its diagnostic and
clears the failure from the run's status, so the run keeps one outcome. The legacy
`check_all_gradients` step argument controls additional returned diagnostics only;
it cannot disable these checks.

## Environment and training API

`train.micro_batch_size` is the global microbatch, divided evenly across GPUs.
Each optimizer update consumes `data.batch_size / train.micro_batch_size`
microbatches. Single-GPU accumulation uses a compiled `lax.scan`; it averages
the FP32 gradients before clipping and applies the optimizer once per global
batch, preserving the supplied RNG for each microbatch.

To measure a microbatch size on one allocated GPU, run
`python -m scaffolds.transformer.benchmarks.benchmark_accumulation --config RECIPE
--micro-batch-size 8 --output REPORT.json`. Use a fresh process per size;
`--implementation python` supplies the former Python-loop baseline.
The report includes synchronized complete-update timings and allocator peaks
using resident synthetic tokens. Data loading, evaluation and checkpoint costs
are excluded, so these timings alone are not an end-to-end training estimate.

Each worktree syncs its own environment from this scaffold's lock; hardlinks share
package files with the uv cache, so an additional environment costs almost no disk:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
# B300 (Blackwell) environment from the same lock: CUDA 13 with cuBLAS >= 13.2.
UV_PROJECT_ENVIRONMENT=$PWD/scaffolds/transformer/.venv-cuda13 \
  uv sync --project scaffolds/transformer --locked --extra cuda13 --link-mode hardlink

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

Gradient diagnostics are standard for every mechanism. At each normal log step,
W&B plots `train/gradient_norm_pre_clip`, `train/gradient_norm_post_clip`,
`train/gradient_clip_scale`, and `train/gradient_clipped`. The norm is taken after
all microbatch and data-parallel averaging, before clipping, and includes any
coefficient penalty in the training objective. The optimizer already computes
this norm for its numerical checks; logging adds no gradient-tree reduction.
Retained training logs keep the sampled norm and clipping scale. Each validation
event in `metrics.jsonl` stores the norm count, mean, maximum, and clipped fraction
over **every update** since the preceding validation or resume, together with the
clipping threshold. W&B also plots the window mean, maximum, and clipped fraction.
The default saved workspace from `views.experiment_workspace` includes gradient
norm plots on a logarithmic axis and separate clipping-scale/fraction plots.
Historical runs without these measurements retain unknown gradient diagnostics.

Training telemetry (`training/telemetry.py`) is standard for data-parallel training
with `model.remat: false`. Every update returns, in one packed vector:

| Statistic | Reduction over microbatches and GPUs |
| --- | --- |
| `layer_<i>/residual_rms`, `final/residual_rms`: residual stream entering block *i* / the final norm | RMS |
| `layer_<i>/residual_max_abs`, `final/residual_max_abs` | maximum |
| `layer_<i>/attention_output_rms`: dot-product attention output before output scaling | RMS |
| `layer_<i>/attention_output_head_rms_min`: smallest RMS of one head at one position | minimum |
| `layer_<i>/head_rms_norm_gain_max` (head RMSNorm): largest `1/sqrt(mean(y^2) + eps)` | maximum |
| `update/<group>/{gradient_norm,parameter_norm,update_norm}` | computed on the averaged gradient |

Parameter groups are embedding, output head, attention Q/K/V, attention output
projection, MLP, norm scales, QK-norm scales, attention-output norm scales,
attention coefficients, and other. `update_norm` is the actual parameter change
after the whole AdamW step. Gated DeltaNet and GLA record the residual and update
statistics only. Recording never changes the update: on CPU the update is
bit-identical with and without it, and on GPU the differences match GPU
nondeterminism between identical updates.

Each log line adds the smallest `attention_output_head_rms_min`, largest
`head_rms_norm_gain_max`, and largest `residual_max_abs` over layers; W&B receives
every statistic as `train/telemetry/<name>`. Each validation event stores
`train_telemetry` (every statistic's mean, minimum, and maximum over every update in
the window), `train_gradient_spike_count`, and `train_gradient_spikes`: the complete
statistics of up to 20 updates whose gradient norm exceeds 5× the median of the
preceding 100 updates (after 20 updates). A numerical-failure record keeps the failed
and preceding updates' statistics. `benchmarks/benchmark_telemetry.py` measures the
cost; see the validation record.

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
  records and `results/` remain editable while jobs run. Both launchers source
  `scaffolds/launch/watchdog.sh`: when `training.log` has not grown for
  `TRANSFORMER_STALL_MINUTES` (default 20), the job logs its GPUs' state, ends the
  running step, and exits with code 124. The finalizer requeues the job, which resumes
  from its latest checkpoint, for a run's first two stalls and records a third as a failure.
  Kennel also requeues ten minutes before its allocation expires. Set
  `TRANSFORMER_ALLOCATION_HOURS` to the requested duration (default 48; use 24
  for standard). Acceptance records the full-run estimate and allows multiple
  allocations only if compilation, evaluation, and a checkpoint interval fit
  inside one allocation with two hours reserved. The CLI exposes this policy as
  `--allocation-hours HOURS --allow-requeue`; without the latter, the full run
  must fit one allocation. Requeues can repeat up to one checkpoint interval,
  and queue waits are additional to the estimate.
- `launchers/orcd.sbatch` does the same on ORCD. Its allocations fit that cluster's
  per-user GPU caps; a job asks for 12 hours, accepts `mit_normal_gpu`'s 6-hour
  maximum, and requeues itself before its limit (see [CLUSTERS.md](../../CLUSTERS.md)).
- `acceptance.py` runs identity-keyed acceptance in the production allocation: CPU
  tests, distributed CPU tests on two forced host devices, and single-GPU kernel tests
  (no skips), an initialization hash, a full-recipe pilot, and an exact fresh-process
  resume. The identity includes the allocated GPU model, so a requeue onto another
  model under the same launcher GPU type (H100 NVL after H100 80GB HBM3) repeats
  them. `attention_spec(config, entry_module=...)` selects correctness and kernel
  tests from the recipe and resolved backend, plus the initialization check and
  runtime environment. A study execution module passes this factory to `main`;
  specialized studies can supply an `AcceptanceSpec` directly. Launchers invoke
  `<acceptance_module> <config.yaml> --runtime-env` before starting workers so GLA
  bridge settings apply only to GLA recipes. Both launchers pass `--test-cache`, so a
  passed test stage is reused by the worktree's other runs that select the same tests
  under the same source, lock, runtime settings, and (for GPU tests) GPU model, such as
  H100 80GB HBM3 or H100 NVL rather than the launcher's `H100`; the
  reusing run copies the report into its own acceptance directory and records its
  origin in `success.json` under `reused_tests`.
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

# In a GPU allocation; the Triton kernels need the locked toolkit, as the launchers set it.
export CUDA_HOME="$(scaffolds/transformer/.venv/bin/python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("nvidia.cuda_nvcc").submodule_search_locations)))')"
TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas" TRITON_PTXAS_BLACKWELL_PATH="$CUDA_HOME/bin/ptxas" \
  scaffolds/transformer/.venv/bin/python -m pytest \
  -c scaffolds/transformer/pyproject.toml scaffolds/transformer/tests \
  --import-mode=importlib -m gpu
```

The suite covers independent FP64 attention and gradient references, signed and
unnormalized kernels, QK-norm cosine scores, learned parameter sharing, optimizer exclusions, distributed
accumulation, packed-data cursors, interruption and resume, checkpoint audits,
acceptance, and retention. GPU-marked tests exercise the fused forward and backward
implementations; without the locked toolkit, a node's older `ptxas` rejects the Triton
kernels' PTX and the Gated DeltaNet and GLA kernel tests fail.

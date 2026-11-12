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

## Continuation from a pretrained Taylor checkpoint

An optional `continuation` recipe section pins `parent_config`, its
`parent_config_sha256`, `parent_checkpoint`, its `parent_checkpoint_sha256`, and
`parent_step`. Packed data additionally pins `parent_data_identity_sha256`, the
SHA256 of `data_identity.json` beside the parent recipe. Empty defaults preserve
ordinary training. Continuations use zero warmup and equal peak/minimum learning
rates; `optim.total_steps` is the absolute ending step, including the parent steps.

To extend an already learned static or query model without changing its attention,
set `continuation.preserve_all_parameters: true` and leave
`continuation.new_parameter_learning_rate: null`. Every learned coefficient and
projection, along with its Adam moments and clock, is restored. All parameters use
the continuation's constant `optim.learning_rate`; no coefficient optimizer is
created or reset. This mode requires an identical model recipe except for the
execution backend and preserves the parent's weight-decay policy.

The shared trainer restores every pretrained parameter and its Adam moments/count
without change, continues the parent's batch cursor and RNG clock, and independently
evaluates and checkpoints the starting state before the first new update. A child
may retain fixed Taylor or introduce direct static or query-conditioned coefficients.
Query continuations use `c_r(x) = (b_r + a_r tanh(x W_r)) / r!`, with `b_r = 1`
and either zero amplitudes or zero projections, so every input starts at Taylor.
Both coefficient and amplitude scales must be `factorial`; an inner tanh bias is
not supported for continuations. Zero amplitudes with random projections initially
learn through the amplitudes; the projections begin receiving gradients once those
amplitudes move.
New coefficients use a separate Adam state whose moments and bias-correction count
start at zero; global clipping still acts once on the combined averaged gradient.
Backbone/data recipe changes are rejected, and `continuation.json` records verified
parent/backbone hashes. Subsequent resumptions restore the child's full checkpoint
through the normal trainer. Acceptance pilots are offset by the parent step.

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
validated 0.90 allocator fraction. Blackwell allocations (B200, B300) run in the CUDA 13
environment `.venv-cuda13` with collectives on the compute stream; every other GPU runs in
the CUDA 12 `.venv` ([CLUSTERS.md](../../CLUSTERS.md)). `launchers/kennel.sbatch` accepts the
GPU types its `GPU_TYPES` argument lists (default `B300,H100`; B200 when listed). Explicit `TRANSFORMER_HOST_STAGING=0` or `1`
overrides automatic selection. This transport policy does not replace kernel and
full-recipe qualification; consult `CLUSTERS.md` for its measured scope.

## Attention and parameter ownership

`model.attention_forget_gate: true` adds the existing per-layer/head gate
`log(alpha_t) = -exp(A_log) * softplus(x_t w_alpha + b_dt)`, computed in FP32 from
the pre-attention RMSNorm activation, to softmax, unnormalized exponential, and
unnormalized nonlinear query-polynomial attention ([Forget gate](#forget-gate) lists
positions and backends). `attention_forget_gate_mode: cumulative` (the historical
default) multiplies weights by `exp(L_t - L_i)`, where `L_t = sum_{j<=t} log(alpha_j)`.
For query polynomials on the blocked backend, the `query_distance` mode instead
multiplies by `exp((t-i) * log(alpha_t))`; only the current query chooses the decay rate.
On softmax and the unnormalized exponential the mode adds `(t-i) * log(alpha_t)` to every
score ([Forget gate](#forget-gate)).
Both modes apply attenuation after the polynomial, keep diagonal attenuation one,
and preserve the coefficient generator and attention-output normalization.
They share identical parameters and initialization: the gate projection decays,
while `A_log` and `b_dt` do not. The separate coefficient-projection decay switch
retains its existing behavior. New-mode GPU acceptance compares outputs and all
operand gradients against an explicit dense reference at 2,048 tokens.

Input-form `query_distance` gates also support two independent options on the
blocked backend. `attention_forget_gate_conditioning: query_key` adds
`forget_gate_key_kernel` and uses the endpoint rate
`exp(A_log) * softplus(wq·x_t + wk·x_i + b_dt)`. The key projection starts at zero
and takes weight decay, exactly recovering the query-only control initially.
Compact query/key projections pass to the attention core; pair rates and causal
statistics are evaluated within query blocks. Polynomial coefficients stay
query-conditioned. On softmax and the unnormalized exponential the same query-key rate
adds `(t-i) * log(alpha_ti)` to every score ([Forget gate](#forget-gate)); the bounded
sigmoid rate below stays with the query polynomials.

With query-only gating, `attention_forget_gate_rate_activation: sigmoid` replaces
softplus by sigmoid, giving a positive rate bounded by learned `exp(A_log)`.
Initialization preserves the rate/projection draws and sets `b_dt = logit(dt)`
from the original Mamba-2 `dt` draw, matching the control's zero-input rate.
Rates on nonzero inputs need not match. Both new fields default to the historical
`query`/`softplus` behavior; combining key dependence with sigmoid is unqualified
and rejected. Gate diagnostics report endpoint rates over causal pairs when
key-conditioned, otherwise over query tokens.

`attention_forget_gate_per_order: true` gives every order, including zero, its
own softplus query-distance gate on the blocked backend or the Mosaic decay kernels
(see `attention_backend: mosaic` below for their GPU limits). With query-only
coefficients the weight is `sum_r c_r(x_t) s_ti^r exp(-(t-i) lambda_tir)`.
The query variant uses `lambda_tr = exp(A_r) softplus(wq_r·x_t + b_r)`;
`query_key` adds `wk_r·x_i` inside softplus. With `attention_forget_gate_mode:
cumulative` (query conditioning, blocked backend only) each order instead keeps its
own cumulative sum `L_tr = sum_(j<=t) log(alpha_jr)`, and the weight is
`sum_r c_r(x_t) s_ti^r exp(L_tr - L_ir)` ([Forget gate](#forget-gate)). Gate arrays
add an order axis before their final channel axis. All orders initially duplicate
the shared gate's draw; key projections start at zero. The copies then learn
independently. Projection matrices receive WD; rates and biases remain exempt.
Diagnostics retain the layer, head, and order axes. Sigmoid rates, other gate
forms, and other backends reject this option, and Mosaic rejects cumulative
per-order gates. Existing recipes default to a shared gate.

`attention_forget_gate_conditioning: none` is the input-free gate of learned static
coefficients: `lambda = exp(A_log) * softplus(b_dt)` per layer and head reads no token,
so every weight is `sum_r c_r s_ti^r exp(-(t-i) lambda)`. It keeps the query gate's
`A_log` and `b_dt` arrays, names and Gated DeltaNet draws (both exempt from weight decay)
and drops its projection; with a rate that reads no token, cumulative and query-distance
decay coincide. It requires learned direct `per_head` static coefficients with the Taylor
start and every order, the input form with softplus rates in `query_distance` mode, RoPE or
no positional encoding, and the blocked (scan or strips) or Mosaic backend;
`attention_forget_gate_per_order: true` gives each order its own copy of the gate. The model
passes the represented coefficients and `-lambda` to the query-coefficient decay operation,
broadcast to every query and token, so their gradients sum back to `c`, `A_log` and `b_dt`;
on Mosaic these are the `query_decay` and `per_order_query_decay` kernels. Diagnostics report
each layer's, head's (and order's) rate and half-life `ln 2 / lambda` beside the coefficients.
Static coefficients reject every gate that reads tokens, and query coefficients reject this one.
Acceptance adds `tests/test_input_free_gate.py` to both test stages of these recipes: on CPU,
outputs and every gradient against FP64 through the scan and the strips, the model's operands,
initialization, masks, diagnostics, recipe rules, and H100 kernel lowering at 2,048 tokens; on
the GPU, the Mosaic path against FP64 at 2,048 tokens, duplicated orders against the shared
kernels, and a two-layer model against the FP32 blocked model.

`attention_forget_gate_form: input` (the default) is that gate. The `constant` form
drops its token dependence: one learned log rate per layer and head,
`log(alpha_t) = -exp(A_log)`, so every weight decays as `exp(-exp(A_log) * (t - i))`,
ALiBi's linear bias in log space; softmax and the exponential add `-exp(A_log) * (t - i)`
to every score, which for softmax is ALiBi. Rates start at ALiBi's slopes `2^(-8h/H)` for
heads `h = 1..H`, identically in each layer, and are exempt from weight decay.
`attention_forget_gate_constant_rate: fixed` holds them at those slopes with no trainable
parameter (`learned`, the default and the meaning of recipes without the field, trains them);
the fixed rate and the learned rate's start are one FP32 array, `forget_gate.alibi_log_rate`.
The constant form is qualified for softmax, the exponential, and query polynomials with the
cumulative mode under RoPE or no positional encoding on the blocked backend (scan or strips);
configuration rejects it with `query_distance` or PaTH, and a fixed rate with any other form.
The `sigmoid` form is the Forgetting Transformer's original
gate, `alpha_t = sigmoid(x_t w_f + b_f)`, with the paper's initialization (`w_f ~ N(0, 0.02^2)`,
`b_f = 0`) and weight decay on `w_f` only; it equals the input form with `exp(A_log) = 1` and the
projection negated. It is qualified for softmax, the exponential, and query polynomials with the
cumulative mode under RoPE or no positional encoding on the blocked backend.

`attention/multihead.py` owns Q/K/V projections, RoPE, head layout, and the output
projection. `attention/core.py` dispatches complete operations on post-RoPE
`[batch, time, heads, channels]` arrays. Q/K use `d_K`; the output has V's shape
and dtype. Mechanisms live in `softmax.py`, `taylor.py`,
`learned_quadratic.py`, `learned_polynomial.py`, `unnormalized.py`, and `spline.py`. Their implementations live under `attention/backends/`.
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
| `attention_degree: p >= 1`, `attention_normalization: false`, `attention_coefficients: learned`, `attention_kernel_basis: bspline`, `attention_spline_knots: [u_0, ..., u_K]` | Unnormalized `sum_j theta_j B_j(QKᵀ / sqrt(d_K)) V` on a fixed degree-`p` B-spline basis with `K + p` functions, static or tanh query-conditioned `theta`; blocked only ([B-spline score function](#b-spline-score-function)) |
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
`per_head`, FP32, without weight decay. For learned static/direct or diagonal tanh-query polynomials,
`attention_coefficient_orders: [0, 2, 4, 6, 8]` stores only five entries and fixes
all omitted orders at zero. Orders must be strictly increasing and include the
configured highest degree. Taylor initialization retains `1/r!` on these orders;
with `attention_coefficient_scale: factorial`, the stored entries start at one
and divide by the factorial of their actual order, not their compact array index.
The existing attention kernels receive the expanded monomial vector, with exactly
zero omitted coefficients. Query generators store and project only the selected
columns of W and entries of a,b (and an optional inner bias); projection weight
decay still applies to W only. Both bias and amplitude scaling use the selected
orders' factorials before expanding the effective coefficients. The blocked reference evaluates them by
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

### QK normalization

New dot-product experiments allocated with `python -m scaffolds.records create`
use `qk_norm: true`, `qk_norm_type: rms_norm`, `qk_norm_sharing: per_layer`,
`qk_norm_initial_scale: null`, and `qk_norm_head_scale: true`. The allocator fills
omitted fields with these settings and rejects copied older settings unless
`--allow-legacy-qk-normalization` explicitly opts into that comparison. Plain
scores require `qk_norm: false` and `--allow-plain-qk`. GLA and degree-zero
attention do not use QK normalization. Historical config loading, checkpoint
resumption, and checkpoint-based diagnostics preserve their recorded settings;
the runtime's defaults remain `cosine_log_scale` with head scaling disabled.

RMSNorm applies `g_Q * q / sqrt(mean(q²) + eps)` and
`g_K * k / sqrt(mean(k²) + eps)` before RoPE. Statistics are independent for each
token and head and computed in FP32; each layer's two gain vectors have
`qk_head_dim` entries and broadcast across heads. The FP32 arrays
`qk_norm_q_gain` and `qk_norm_k_gain`, both `[num_layers, qk_head_dim]`, initialize
to one and receive no weight decay. Channel gains generally do not commute with
RoPE, so their placement before RoPE is part of this mode's definition.

With `qk_norm_head_scale: true`, the model also multiplies normalized Q by
`exp(gamma)` once before RoPE. Each layer/head learns its own scalar in
`qk_norm_head_log_scale` (`[num_layers, num_heads, 1]`, FP32, no weight decay).
Zero initialization gives scale one and preserves the original RMSNorm function
exactly. Q/K channel gains stay shared across heads; the additional scalar permits
independent score magnitudes. The kernels retain `QKᵀ / sqrt(d_K)`, so this scale
multiplies the score once, for softmax, exp, and polynomial kernels alike.
Validation records channel gains in `attention_diagnostics.qk_norm_gains` and
head log scales/effective scales in `attention_diagnostics.qk_norm_head_scales`;
W&B mirrors their mean, minimum, and maximum. Historical RMSNorm recipes with
`qk_norm_head_scale: false` have no additional scalar or parameter array.

`qk_norm_token_scale: cosine` (default `none`; requires the head scale) gives each
query its own bounded log scale from the attention input `x_t`, the activation after
the block's pre-attention RMSNorm: `delta_th = a_h cos(w_h, x_t)`, and Q is multiplied
by `exp(gamma_h + delta_th)` instead of `exp(gamma_h)`. Each layer/head owns a direction
`qk_norm_token_scale_kernel` (`[num_layers, num_heads, hidden_size]`, drawn
`N(0, 1/hidden_size)` from the mechanism stream; only its direction matters) and an
amplitude `qk_norm_token_scale_amplitude` (`[num_layers, num_heads, 1]`, initially zero),
both FP32 and exempt from weight decay by default. Set
`qk_norm_token_scale_kernel_weight_decay: true` to apply `optim.weight_decay` to the
direction `w_h` only; the amplitude remains exempt. This policy also applies to the
new-parameter optimizer in continuations and is recorded in attention diagnostics.
Both vectors are normalized as `v / sqrt(||v||² + eps)`
with `qk_norm_eps`, so `|delta| <= |a_h|` without saturation, and a zero amplitude
reproduces the head-scaled model exactly. The gate restores the per-token query-norm
freedom that RMSNorm removes; a per-query temperature of the degree-`m` Taylor kernel is
the same operation. Validation records amplitudes, direction norms, and the realized
`delta` statistics in `attention_diagnostics.qk_norm_token_scales`; W&B mirrors the
amplitudes. A continuation may introduce the gate beside a fixed-Taylor parent and give
it its own constant learning rate (`continuation.new_parameter_learning_rate`).

The historical `qk_norm_type: cosine_log_scale` normalizes Q and K per head in
FP32 as `x̂ = x / sqrt(||x||² + qk_norm_eps)`, then feeds
`q' = exp(gamma) sqrt(d_K) q̂` and `k' = k̂` to the unchanged kernels. Its score is
`exp(gamma) * <q̂, k̂>`. The scalar `qk_norm_log_scale` is `[1]` with
`qk_norm_sharing: global` or `[num_layers, num_heads, 1]` with `per_head`, FP32
without weight decay. It initializes at `log(qk_norm_initial_scale)`, or at
`log(sqrt(d_K))` when that setting is null. Validation retains these scales under
`attention_diagnostics.qk_norm_scale`. Unlike channel gains, unit normalization
and the scalar commute with RoPE. Existing cosine-score recipes retain their
arithmetic and parameter trees.

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
decay, except that `model.attention_coefficient_kernel_weight_decay: true` decays the
kernel `W` with `optim.weight_decay`. The kernel starts at zero, the bias at `b_r = 1/r!`, and the tanh amplitude
at one, without consuming RNG draws. The backbone keeps its canonical
initialization and the initial attention kernel is the degree-`m` Taylor polynomial.
For learned query-conditioned tanh coefficients with diagonal readout,
`optim.query_coefficient_learning_rate_scale` (default `1.0`) multiplies the learning
rate of the stored bias `b` and amplitude `a` at every layer, head, and order. It
scales the completed Adam update after global gradient clipping; the projection
`W` and all other parameters keep the base schedule. Both `a` and `b` remain
excluded from weight decay. A value of `3.0` gives them three times both peak and
minimum LR with the same warmup and cosine horizon, without changing their forward
factorial scaling or optimizer moments. The optimizer state layout stays unchanged.
The multiplier is frozen in each run's recipe and supports ordinary checkpoint
resumption. Checkpoint continuations from a different recipe use their existing
`continuation.new_parameter_learning_rate` option instead.
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
`attention_coefficient_gate: sigmoid` (default `none`) gates every order independently
per layer and head: `c~_r(x_t) = 2 g_r(x_t) c_r(x_t)` with
`g_r(x_t) = sigmoid(U_r^T x_t + beta_r)`, where `x_t` is the coefficient generator's input.
The kernel `query_polynomial_gate_kernel` (`U`, `[layers, heads, hidden_size, m+1]`) and
the bias `query_polynomial_gate_bias` (`beta`, `[layers, heads, m+1]`) are FP32, start at
exactly zero, and draw no random numbers, so every other parameter keeps the ungated
recipe's initialization. The fixed factor 2 normalizes the gate: `2 g_r` is exactly 1 at
the start, preserving the ungated function and its derivatives with respect to `W`, `a`
and `b`, and ranges over `(0, 2)` thereafter, so an order can be suppressed or amplified;
the gate never starts near saturation. `U` follows
`attention_coefficient_kernel_weight_decay`, like `W`; `beta` never decays. The gate is
applied inside the coefficient generator, so every backend receives the gated `c~`
unchanged, with no distance-dependent decay or row-sum division. It is qualified for
diagonal tanh query coefficients with RoPE, without selected orders, an inner tanh bias, a
forget gate, continuation, or a query learning-rate scale. Diagnostics report the gate,
the ungated `c`, the order tilt `log g_r - mean_r log g_r`, and each order's contribution
`c~_r s_ti^r` over causal pairs, since a small gate alone does not show suppression.
`attention_coefficient_gate: gdn` multiplies instead by Gated DeltaNet's decay without a
distance exponent, `c~_r(x_t) = alpha_r(x_t) c_r(x_t)` with
`alpha_r(x_t) = exp(-exp(A_log_r) softplus(w_r^T x_t + dt_bias_r))` in `(0, 1]`, computed in
FP32 per layer, head and order. Its arrays are `query_polynomial_gate_kernel` (`w`,
`[layers, heads, hidden_size, m+1]`), `query_polynomial_gate_log_rate` (`A_log`) and
`query_polynomial_gate_dt_bias` (`[layers, heads, m+1]`), with the input-form forget gate's
initialization (`w` uniform in `+-1/sqrt(hidden_size)`, `exp(A_log)` uniform on `(0, 16)`,
`softplus(dt_bias)` log-uniform on `[0.001, 0.1]`), drawn once per layer and head from the
mechanism stream and copied to every order. Every order therefore starts with the same
`alpha_t`, which per-head output RMSNorm cancels up to its epsilon: the normalized output is
the ungated one times `sqrt((ms + eps alpha^2) / (ms + eps))`, with `ms` the gated head output's
mean square. `w` follows `attention_coefficient_kernel_weight_decay`; `A_log` and `dt_bias`
never decay. Nonfinite gate inputs give NaN rather than a saturated gate. The qualification
restrictions are the sigmoid gate's, and diagnostics report `alpha` in the gate fields
together with `exp(A_log)`, `softplus(dt_bias)` and `||w_r||`.
`acceptance.gdn_gate_start_parity` checks a recipe's initial logits against its ungated
counterpart: to FP32 rounding with the RMSNorm epsilon at `1e-30`, and within the epsilon's
bound at the recipe's.
`attention_coefficient_amplitude_scale: factorial` (tanh only; default `none`) uses the
effective amplitude `a_r/r!`, `c_r = b_r + (a_r/r!) tanh(w_r^T x_t)`, with no new
parameters. Because Adam's step is invariant to the gradient's scale, order `r`'s
modulation then moves at `lr/r!`, in proportion to its Taylor bias. Diagnostics report
the stored `a_r` and the effective amplitude.
`attention_coefficient_scale: factorial` does the same for the input-independent part of
learned direct unnormalized coefficients, static `c_r` or the tanh query bias `b_r`: the
stored `c~_r` starts at `r!` times the Taylor value (so at 1) and attention uses
`c~_r / r!`. Order `r` then moves at `lr/r!`, its gradient carries `s^r / r!` instead of
`s^r`, and the coefficient penalty is rejected. Diagnostics report stored and represented
values (`effective` coefficients, `effective_projection_bias`).
To scale the entire query-dependent coefficient, enable both factorial options:
`c_r(x_t) = [b_r + a_r tanh(w_r^T x_t)] / r!`, with stored `b_r = 1` and
`a_r = 0` for a Taylor start. Static coefficients need only
`attention_coefficient_scale: factorial`.
`attention_coefficient_readout: dense` replaces diagonal tanh amplitudes with a
matrix that mixes all tanh features into every polynomial order. With both
factorial options enabled, `c_r(x) = [b_r + sum_j V_rj tanh(w_j^T x)] / r!`.
It requires query-only tanh conditioning. `attention_coefficient_feature_count`
sets the number of features `K` independently of the degree; null uses `m+1`.
`query_polynomial_kernel` then has shape `[layers, heads, hidden_size, K]`, and
`query_polynomial_mixing` has shape `[layers, heads, m+1, K]` (FP32, no weight
decay), replacing the amplitude array. For `K = m+1`, initialization places the
requested amplitude on the mixing diagonal, exactly reproducing the diagonal
readout. Other `K` require zero initial amplitude and start mixing at zero.
The bias and backbone initialization stay unchanged. Diagonal remains the
runtime default for old checkpoints.

`attention_coefficient_readout_scale` (default `1.0`) multiplies only the dense
input-dependent term: `c_r = [b_r + scale * sum_j V_rj tanh(w_j^T x)] / r!`.
For example, `1/3` reduces the effective mixing contribution while leaving the
Taylor bias unchanged. It must be positive and finite; values other than one
require dense readout. Diagnostics record the stored and effective mixing.

`training.coefficient_distillation_model` provides native teacher-input capture,
free-table checkpoint streaming, Taylor-preserving dense student construction,
packed generator parameters, and ordinary student evaluation. Teacher capture
observes the attention module's actual normalized input and exposes both joint
and independently compiled forward checks. `training.coefficient_distillation_regression`
fits packed generators against factorial-coordinate targets. Each layer/head has
its own mean loss and gradient clipping before Adam; deployment uses the same
production coefficient projection. Dataset selection, fitting budgets, and result
interpretation belong to the calling study.

`attention_coefficient_conditioning: query_key` (tanh only) multiplies each key's
degree-`r` term by a per-key factor, `sum_(i<=t) sum_r c_r(x_t) f_r(x_i) s_ti^r v_i` with
`f_r(x_i) = 1 + a^k_r tanh(w_k_r^T x_i)` from the key token's activation
(`key_polynomial_kernel` `[layers, heads, hidden_size, m+1]`, `key_polynomial_amplitude`
`[layers, heads, m+1]`; drawn, started, and decayed like `W` and `a`, so the start is
unchanged).
With both factorial options enabled its entire pair coefficient is
`[b_r + a_r tanh(w_r^T x_t)] [1 + a^k_r tanh(w_k_r^T x_i)] / r!`.
The key factor is dimensionless and receives no second factorial division.
The pair coefficient factorizes, so nothing pairwise is stored: the blocked reference
scales each key column, the fused Triton kernel (`backends/fused_query_key_polynomial.py`)
forms `c_r(x_t) f_r(x_i)` inside each tile in one pass (its key-side backward also
accumulates the key-factor gradient), and the Mosaic query kernel runs once per order with
the factor folded into degree-specific values. The per-order path rounds each partial
output to V's dtype before summing, which at degree one corrupts the gradient after head
RMSNorm. Diagnostics report the key factors beside the query coefficients.

For H100 and B300 BF16 query-key attention at context 2048, Q/K and value width 64,
and degree 4, 8 or 12, the fused backend uses 32-by-32 backward tiles and keeps the
original forward tiles. Other devices and shapes keep their original geometry.
The arithmetic precision and coefficient parameterization are unchanged.
Kernel timings and qualification are in the
[validation record](validation/README.md#query-key-polynomial-backward-tiles).

The tanh path uses a dedicated fused kernel (or its Mosaic GPU counterpart,
`attention_backend: mosaic`) that directly evaluates the query-dependent polynomial, with gradients for Q/K/V and every coefficient.
The historical identity path retains its linear polynomial kernel plus FP32
causal value sum. Both have an explicit blocked polynomial reference.
Per-head output RMSNorm remains an independent option through
`attention_output_scaling: head_rms_norm`. Validation diagnostics report the
actual input-dependent coefficient means, standard deviations, minima, and
maxima on the fixed diagnostic sample, separately from projection parameters.

For direct coefficient adaptation with a frozen backbone, unnormalized direct learned
static and query-only models expose
`logits, c = model(tokens, return_coefficients=True)`. The returned FP32 `c` has
shape `[layers, batch, time, heads, m+1]` and contains the actual coefficients used
in that forward pass, after bias and amplitude scaling. Static coefficients are
broadcast across examples and query positions, with the normal forward unchanged.
Calling `model(tokens, query_coefficients=c)` replaces the static coefficients or
bypasses the query coefficient generator and uses
these effective values directly, without further factorial scaling or tanh.
Each example and query has its own polynomial, shared across its causal keys.
All other attention operations and model parameters are unchanged; gradients flow
through the frozen layers to the supplied array. Optimize only this array to keep
the backbone fixed. Both operations support JIT, require `model.remat: false`, and
can be combined to return the supplied coefficients. Capture and `return_diagnostics`
are mutually exclusive. Query-key conditioning rejects both operations.

`training.coefficient_fit` runs fixed-input adaptation from a pinned pretrained
checkpoint. Its `free_per_example_query_coefficients` mode optimizes captured
coefficient tables; `query_coefficient_generator` mode optimizes only the original
query generator's bias, amplitude, projection, and optional inner tanh bias, and for
a query-key parent also its trained key projection and amplitude (key factors carry
no factorial scaling). A query-key parent's model is restored unchanged, and its
restored forward must agree with the blocked reference backend before fitting.
Generator mode accepts one sequence or a fixed batch. It recomputes all activations
through the frozen backbone and uses native stored parameter coordinates, keeping
factorial scaling inside the model. `shared_batch_query_coefficients` optimizes
one `[layer,query,head,order]` table shared across every fitting sequence. It
initializes from the arithmetic mean of captured effective coefficients across
the full fitting batch and records original-parent and projected-initial losses
separately. A scan computes microbatch gradients at the same table, averages them
over the complete batch, and then performs one clipping/Adam update in factorial
coordinates. It never maintains a separate coefficient table or optimizer for
each sequence. `static_coefficients` uses the same optimizer over one
`[layer,head,order]` table, shared across query positions as well as sequences.
Its initialization averages captured effective coefficients over all fitting
sequences and query positions. This mode accepts one sequence or a full batch
without changing its trainable parameter count. All modes verify input identity, forward
replay, an improving update, frozen-parameter checksums, and exact checkpoint
resume before fitting. The study provides the frozen recipe and launches:
`python -m scaffolds.transformer.training.coefficient_fit --run <run-directory>`.
The retained best checkpoint contains only fitted parameters and references the
source checkpoint for the frozen backbone. Per-update measurements are fitting
metrics; no held-out evaluation is implied. Shared-table qualification checks
original replay before projection and compares the training forward with the
projected model; changing the initial function is not counted as fitting progress.

### Coefficient release schedule

A release schedule keeps learned coefficients at their initial values for the first
part of training, then releases them (`training/coefficient_release.py`). It
applies to learned direct static coefficients (`polynomial_coefficients`, `[m+1]` or
`[layers, heads, m+1]`) and to diagonal tanh query coefficients without an inner
bias (`query_polynomial_bias` and `query_polynomial_amplitude`, `[layers, heads, m+1]`,
and `query_polynomial_kernel`, `[layers, heads, hidden_size, m+1]`). Every array must
store all orders `0..m` on its last axis, so `attention_coefficient_orders` is
rejected, as are query-key conditioning, dense readout, other activations,
continuations, and non-Taylor initialization. Order `r` is entry `r` of that axis:
static `c_r`, or the query generator's `b_r`, `a_r`, and column `r` of `W` together.
All other parameters, including QK gains, head score scales, output RMSNorm gains,
and forget-gate parameters, train from the first update.

| Setting | Meaning |
| --- | --- |
| `optim.coefficient_release_schedule` | `all` or `by_order`; null (default) trains the coefficients from the first update |
| `optim.coefficient_release_step` | first release, with `0 < step < optim.total_steps` |
| `optim.coefficient_release_end_step` | `by_order` only: the last order's release, `step < end < optim.total_steps` |

Counts `k` are optimizer updates on the learning-rate schedule's own clock: the update
with count `k` is logged as step `k + 1` and uses `schedule(k)`. Order `r` is frozen
for `k < s_r` and trains for `k >= s_r`. `all` sets `s_r` to the release step for every
order; `by_order` sets `s_r = step + floor(r (end - step) / m)`, lowest order first.
With step 455, end 1500 and `m = 8`, orders 0-8 release at 455, 585, 716, 846, 977,
1108, 1238, 1369, and 1500.

Freezing has two masks, both functions of the update count alone. A frozen entry's
gradient is zeroed after microbatch and data-parallel averaging, before global-norm
clipping and AdamW: it adds nothing to the clip norm, the rest of the model trains
exactly as if those coefficients had no gradient, and its Adam moments stay exactly
zero. Its final update is zeroed after AdamW, so decoupled weight decay cannot shrink
a decayed `W` (`attention_coefficient_kernel_weight_decay`). Frozen entries stay
bitwise equal to their initial values until release. A released entry starts from
zero moments while bias correction keeps the global count: with betas (0.9, 0.95) and a
steady gradient, its first update is 0.45 of a steady Adam step, and later ones overshoot
it by up to 10% (20 updates after release) before settling. It joins the shared
learning-rate schedule at its current value (`b` and `a` times
`optim.query_coefficient_learning_rate_scale`, when set), with no separate warmup, Adam
clock, or learning rate. The logged `gradient_norm_pre_clip`, clipping scale, numerical
update checks, and telemetry use the masked gradient that clipping sees.

The release wraps the unchanged clipping and AdamW chain, reading the schedule's count
from its state, so the optimizer state layout is the same with or without a schedule
and a checkpoint resume continues the schedule exactly. It names the arrays it freezes
and raises before the first update if any is missing, rather than letting it train.
Without a schedule, the optimizer and train steps are built exactly as before. The
training log prints
`coefficient_release_schedule=by_order release_updates=455,585,...` at start and
`coefficient_release update=455 step=456 orders=0` at each release; W&B receives
`train/coefficient_trainable_orders` at that step and the tag
`coefficient_release_all` or `coefficient_release_by_order`. The attention identity
and formula, recorded in metrics and metadata, include the schedule and every
`s_r` (`coefficient_release_counts`) only when one is set, and `runs.csv` has
`coefficient_release_schedule`, `coefficient_release_step`, and
`coefficient_release_end_step` columns, empty without a schedule. Acceptance selects
`tests/test_coefficient_release.py` for every recipe with a schedule.

### Per-degree score biases

`attention_score_bias` (default `none`) gives every order `r = 1..m` of diagonal tanh query
coefficients its own signed shift of the score (`attention/score_bias.py`); order zero is unchanged:

    W(t,i) = c_0(x_t) G_0(t,i) + sum_(r=1..m) c_r(x_t) G_r(t,i) (s_ti + beta_r(t,i))^r

`c_r` keeps its generator and factorial scales. `G_r = 1`, or each order's endpoint gate
`exp(-(t-i) exp(A_r) softplus(x_t w_query_r + x_i w_key_r + dt_bias_r))` (the per-order query–key
gate above). `x` is the attention input after the block's RMSNorm. Each layer, head, and order has its
own shift:

| Mode | `beta_r(t,i)` | Arrays | Added parameters (15 layers, 14 heads, width 896, `m = 8`) |
| --- | --- | --- | ---: |
| `static` | `b_r` | `score_bias_offset` (`b`) | 1,680 |
| `query` | `b_r + a_r tanh(x_t u_r)` | `b`, `score_bias_amplitude` (`a`), `score_bias_query_kernel` (`u`) | 1,508,640 |
| `query_key` | `b_r + a_r tanh(x_t u_r + x_i v_r)` | the query mode's and `score_bias_key_kernel` (`v`) | 3,013,920 |

- Arrays: FP32, `[layers, heads, m]` (`b`, `a`) and `[layers, heads, m, hidden_size]` (`u`, `v`). `b`
  starts at 0, `a` at 1, `u` and `v` at 0, with no random draw, so every shift starts at exactly zero,
  every other array keeps its initialization, and `u` and `v` receive gradients from the first update.
  AdamW decays `u` and `v`; `b` and `a` never decay. Telemetry reports them as the `score_bias` group.
- Recipes: unnormalized dot-product attention with learned direct `per_head` Taylor-initialized
  diagonal tanh query coefficients of degree 1-12, without selected orders, a coefficient gate, or a
  B-spline basis; RoPE or no positional encoding; and no forget gate or the per-order query–key
  endpoint gate (input form, `query_distance`, softplus rates). Configuration rejects every other
  recipe, coefficient release, `optim.query_coefficient_learning_rate_scale` other than 1, and
  continuations; the model rejects coefficient capture and override.
- Backends: `mosaic` trains them on H100's score-bias kernels (`backends/mosaic/score_bias.py`; no gate
  or the endpoint gate, with a row shift for `static` and `query` or a pair shift for `query_key`).
  Every other GPU refuses those kernels before tracing one, and `ModelConfig.mosaic_score_bias_problem`
  names a shape H100's shared memory cannot hold. `blocked` (`backends/reference.py`) trains them on any
  GPU with its causal strips, which `scaffolds.records create` gives a fresh blocked recipe. The scan is
  their reference for CPU tests and diagnostics, but no GPU test qualifies it for training them:
  `scaffolds.records create` (also with `--allow-scan-blocked-attention`) and acceptance's test
  selection refuse a score-bias recipe on the scan. Both form each shifted power directly (the
  kernels inside their causal tiles, the blocked backend one `[batch, heads, queries, keys]` block per
  order), never expanding it into powers of `s`, which would change how the orders cancel. `auto`,
  `fused`, and the other backends reject these recipes, and every other attention path raises on
  score-bias operands rather than ignoring them.
- Operands: `static` and `query` pass the row shift `beta_r(t)` `[batch, time, heads, m]`, whose gradient
  autodiff carries to `b`, `a`, and `u`; `query_key` passes the projections `x u` and `x v`
  `[batch, time, heads, m]` with `b` and `a` `[heads, m]`, and each pair's shift is formed in its tile.
- Records: the attention identity records the mode, `score_bias_parameter_count`,
  `score_bias_weight_decay`, and the formula with `(s_ti + beta_r(t,i))^r`; `attention_implementation`
  names the kernels (for example `pallas_mosaic_gpu_query_polynomial_pair_score_bias`); W&B tags the run
  `score_bias_<mode>`. `runs.csv` has an `attention_score_bias` column (`none` for every model recipe
  that predates it).
- Diagnostics: at each validation, the blocked reference of the same shifted kernel, forward only on the
  fixed diagnostic sample, reports `coefficients.score_biases`: the parameters; per layer, head, and
  order, the moments and extremes of the shifted scores (and of the query or query–key shifts), the
  fractions of shifted scores below zero and within `1e-3` of zero, and each order's term
  `T_r = c_r G_r (s + beta_r)^r`; shifted-score histograms on [-16, 16] per layer and order, pooled
  over heads; and the cancellation across orders per layer and head (`sum_r |T_r| / |sum_r T_r|` per
  pair, `sum_r ||o_r|| / ||o||` per row). Nonfinite parameters are its only issue. A positive shifted
  score does not imply a positive weight: the coefficients are signed.
- Acceptance adds `tests/test_score_bias.py` to the CPU tests of these recipes (FP64 references through
  the scan and the strips with every gradient, operands, starts, refusals, the model's layers on the
  strips, and diagnostics). On `mosaic` it adds `tests/test_score_bias_lowering.py` (the H100 kernels'
  shared memory and tile mathematics lowered on the host) and, on the GPU, `tests/test_mosaic_score_bias.py`
  with the GPU case of `tests/test_score_bias.py`, at the kernel family's precision, beside the
  backbone's query and per-order gate checks. On `blocked` the GPU stage is
  `tests/test_score_bias_strips.py` alone: the strips through `attention_core` at production precision
  (Q, K, V rounded to BF16 in the program, FP32 operands, degree eight, 2,048 tokens) for every mode on
  both backbones, with negative and exactly zero shifted scores, against host FP64; the output and dQ,
  dK, dV within 1.2x their BF16 rounding floors and every FP32 gradient within 1e-4 per order. The
  backbone's other GPU cases qualify the Triton and Mosaic kernels these runs never execute and can skip
  or refuse on a GPU those kernels are not qualified on, so they are not selected.

### Score transforms

`attention_score_transform` (default `identity`) evaluates the diagonal tanh query polynomial at a
parameter-free function of the score instead of the score itself (`attention/score_transform.py`):

    W(t,i) = sum_(r=0..m) c_r(x_t) z_ti^r,    z_ti = phi(s_ti)

| Transform | `phi(s)` | Very negative `s` |
| --- | --- | --- |
| `identity` | `s` (every historical recipe) | `c_m s^m` dominates: a large weight of sign `(-1)^m` |
| `softplus` | `log(1 + exp(s))`, as `max(s, 0) + log1p(exp(-abs(s)))` | `z -> 0`, so `W -> c_0` |
| `abs` | `abs(s)` | the weight of `-s` |

- Derivatives: `sigmoid(s)` for softplus and `sign(s)` (zero at `s = 0`) for abs, given as JVP rules;
  autodiff of the softplus formula would give 0 at `s = 0`, since JAX takes `|s|`'s derivative there as 1.
  Masked entries evaluate `phi` at their zero score and are zeroed as for every kernel; a nonfinite score
  poisons its row as before.
- `attention_coefficient_constant_initial` (default null) starts the order-0 offset `b_0` of the same
  coefficients at its value instead of Taylor's 1 (stored `b_0 * 0!`, every other array unchanged); with
  `attention_coefficient_amplitude_initial: 0`, `c_0` starts at exactly that value. With 0 and softplus the
  kernel starts at `sum_(r=1..m) z^r/r!`, the Taylor polynomial of `e^z - 1 = e^s`: about `e^s` for negative
  scores and `T_m(s) - 1` for large positive ones. Neither field adds a parameter.
- Recipes: unnormalized dot-product attention with learned direct `per_head` Taylor-initialized diagonal tanh
  query coefficients of degree >= 1, without selected orders, a coefficient gate, or a B-spline basis (both
  fields); the transform also requires no forget gate, no score biases, and `attention_backend: blocked` (RoPE
  or no positional encoding) or `mosaic` (`mosaic_score_transform_problem()`: RoPE, and the widths and degrees
  of the query polynomial's Mosaic kernels at the recipe's length). Configuration rejects coefficient release,
  `optim.query_coefficient_learning_rate_scale` other than 1, and continuations with either field; the model
  rejects coefficient capture and override with a transform.
- Backends: the blocked backend evaluates a transform in its scan and strips, one `[batch, heads, queries,
  keys]` block at a time; a blocked recipe trains on the strips, which `scaffolds.records create` gives a fresh
  blocked recipe and requires (also with `--allow-scan-blocked-attention`). An explicit `mosaic` backend runs
  `mosaic.api.transformed_query_polynomial_attention(q, k, v, coefficients, transform=...)`: the query
  polynomial's own H100 and B300 kernels with `phi` evaluated in their causal tiles (`common.QUERY_SOFTPLUS`,
  `common.QUERY_ABS`). Backend selection admits them on H100 (sm90) alone and refuses a B200 or B300 with the
  reason, as it refuses the score biases: B300's kernels lower, but no Blackwell GPU has run them. Softplus is
  `(s + max(-s, 0)) + ln(1 + y)`, `y = e^-abs(s)`, from one `ex2.approx` and one `lg2.approx` (below
  `y = 2^-6` the cubic series of `ln(1 + y)`): `s + max(-s, 0)` is `max(s, 0)` for every finite score and NaN at
  `s = -inf`, so a causal score that overflowed makes its row NaN as in the query kernels and the blocked
  reference, where the limit 0 would give the pair the finite weight `c_0`. `sigmoid(s) =
  e^(min(s, 0) - ln(1 + y))` takes one more `ex2.approx`; abs evaluates no transcendental and keeps the query
  family's compiler settings.
  dS is `dP p'(z) phi'(s)` in the dQ and dK passes, and the dQ pass sums `dc_r = sum_i dP z^r`; shared memory,
  vectors, precision (BF16 weight products), shapes and failure behavior are the query family's, and the
  identity transform is `query_polynomial_attention` itself. `auto` never selects these kernels
  (`mosaic_supported()` is false for a transformed recipe; `mosaic_score_transform_supported()` holds where
  they apply). The fused kernels, `auto`, every other backend, and every other attention path raise on a
  transform rather than computing the untransformed polynomial. The constant start alone runs on any
  backend.
- Records: the attention identity records `attention_score_transform` and `attention_coefficient_constant_initial`
  when set, and the formula reads `z_ti^r` with `phi`'s definition and `b_0`'s start; on Mosaic the
  implementation is `pallas_mosaic_gpu_query_polynomial_<name>_scores`; W&B tags the run
  `score_transform_<name>` and `coefficient_constant_start`. `runs.csv` has an `attention_score_transform`
  column (`identity` for every model recipe that predates it) and a `coefficient_constant_initial` column
  (empty unless the recipe sets the start). Untransformed recipes trace identical programs.
- Diagnostics: the scan of the same kernel, forward only on the fixed diagnostic sample, reports
  `coefficients.score_transform`: per layer and head the raw scores' and the inputs' moments and extrema, the
  fraction of raw scores below zero, and the fraction of causal pairs at which the polynomial decreases in its
  input (`sum_r r c_r(x_t) z^(r-1) < 0`); per layer, histograms of `s` and `z` on [-16, 16] pooled over heads.
  The coefficient statistics beside it give each order's sign per head. `kernel_values` is
  `query_polynomial_of_<name>_scores`.
- Acceptance adds `tests/test_score_transform.py` to the CPU tests of a transformed recipe (FP64 references
  through the scan and the strips with every gradient, special values, masking, the softplus start, refusals,
  allocation, the model start, diagnostics). On the blocked backend it adds `tests/test_blocked_strips.py`,
  whose families include two transforms, and its GPU stage is `tests/test_score_transform_strips.py` (the
  strips through `attention_core` at production precision, degree eight and 2,048 tokens, for each transform
  with unit and wide scores, against host FP64: output and dQ, dK, dV within 1.2x their BF16 floors, the
  coefficient gradient within 1e-4 per order) and the strips' score-bias qualification,
  `tests/test_score_bias_strips.py`, alone. On Mosaic it adds `tests/test_score_transform_lowering.py` on CPU
  (counts, support, exact shared memory and lowering of both architectures' kernels, refusals, dispatch, and
  the tile mathematics in FP64 against explicit references) and `tests/test_mosaic_score_transform.py` to the
  GPU stage (degrees 1-12, every Q/K width, unit and wide scores, the production context against host FP64
  and the strips, future-token invariance, NaN rows, repeatability, dispatch), beside the query kernels' own
  GPU cases. A recipe with only the constant start adds `tests/test_score_transform.py` on CPU.

### B-spline score function

`attention_kernel_basis: bspline` (default `polynomial`, the monomials above) replaces the
score polynomial by a fixed B-spline basis (`attention/spline.py`). The attention output is
`sum_(i<=t) f(s_ti) v_i` with `f(s) = sum_j theta_j B_j(s)` and the usual scores `s_ti`
(after QK normalization, head scale, and RoPE), unnormalized and signed; per-head output
RMSNorm remains a separate option.

- Basis: `B_j` are the B-splines of degree `p = attention_degree` on the distinct knots
  `attention_spline_knots` (`u_0 < ... < u_K`, strictly increasing), clamped: the knot
  vector repeats each endpoint `p + 1` times. There are `K + p = len(knots) + degree - 1`
  coefficients, never `degree + 1`: degree 4 on `[-32, -6, 0, 6, 32]` (knot vector
  `[-32 x5, -6, 0, 6, 32 x5]`) has eight, `B_0..B_7`. `f` is a degree-`p` polynomial on each
  interval with `p - 1` continuous derivatives at the simple interior knots. The same basis
  serves every layer, head, and token; knots are not learned.
- Extrapolation: outside `[u_0, u_K]` `f` continues its first or last polynomial piece. Scores
  are never clipped and `f` is never zeroed.
- Coefficients (FP32, never factorial-scaled):

  | Conditioning | Arrays (per layer and head) | Weight decay |
  | --- | --- | --- |
  | `static` | `spline_coefficients` `[layers, heads, K + p]`: `theta` | none |
  | `query`, `attention_coefficient_activation: tanh` | `query_spline_bias` `b` and `query_spline_amplitude` `a`, `[layers, heads, K + p]`; `query_spline_kernel` `W`, `[layers, heads, hidden_size, K + p]` | `W` only, with `attention_coefficient_kernel_weight_decay: true` |

  The query form is the monomial diagonal tanh generator with `K + p` outputs,
  `theta_j(x_t) = b_j + a_j tanh(w_j^T x_t)` from the pre-attention RMSNorm activation, with
  no inner tanh bias and no key factors; `attention_coefficient_amplitude_initial` and
  `attention_coefficient_projection_initial_std` set `a` and the `N(0, std^2)` draw of `W` from
  the mechanism stream, as for monomial coefficients.
- Initialization (`attention_coefficient_initialization: taylor`, the only start): `theta`
  (or `b`) holds the exact B-spline coefficients of `T_p(s) = sum_(r<=p) s^r / r!`, derived by
  blossoming in rationals (`spline.taylor_coefficients`, which checks every piece) and rounded
  once to FP32, so `f = T_p` on the whole line, extrapolation included. Degree 4 on the knots
  above gives `[116131/3, 38029/6, -931/6, 38, -42, 2351/6, 63935/6, 49697]`; with `a = 0` the
  query form starts at the same function. The tests require the FP32 start to reproduce `T_4`
  within 5e-6 relative inside the knots and 1e-6 outside; the error is largest near `T_4`'s
  minimum (0.27 at `s = -1.6`), where the B-spline terms cancel: `sum_j |theta_j B_j(s)|` is
  166 times `T_4` there.
- FP32 storage quantizes updates: a step changes a stored coefficient only when it exceeds half
  the coefficient's FP32 spacing. Wide knots make the end coefficients large: above, `theta_0`
  and `theta_7` (or `b_0` and `b_7`) start at `T_4(-32) = 38710.33` and `T_4(32) = 49697`,
  spacing `2^-8`. Adam steps are at most about 1.17 times the learning rate with betas
  (0.9, 0.95), so at learning rates below about 1.7e-3 these coefficients never move and the
  static `f(±32)` stays `T_4(±32)`; `theta_1` and `theta_6` (spacing 4.9e-4 and 9.8e-4) move by
  whole spacings, only while steps exceed half of one. Even in exact arithmetic Adam moves a
  coefficient by at most about 1.17 times the summed learning rate.
- Evaluation: a fixed linear map (`spline.piece_matrix`, exact in rationals, then rounded to
  the compute dtype) takes `theta` to each piece's local polynomial
  `sum_r P_jr (s - c_j)^r`, once per head (static) or per query token and head (query), before
  any per-score work. The centre `c_j` is the point of the piece nearest zero (`-6, 0, 0, 6`
  above), where scores concentrate, so most evaluations use a small local variable; at the
  Taylor start each piece's local polynomial is `T_p` expanded at `c_j`, the two central ones
  `T_p` itself. A score selects its piece by comparison with the interior
  knots (`s < u_1`, ..., `s >= u_(K-1)`; a NaN score stays NaN) and is evaluated by Horner's
  rule in FP32.
- Backends: the blocked backend only (`attention_backend: blocked`), with either
  implementation: the spline is a kernel branch of the per-block arithmetic that the scan and
  the causal strips share, so diagnostics (from the scan) and training (strips) run one copy.
  Invalid rows follow the signed learned polynomial's rules: a nonfinite score or kernel
  poisons its row. No fused, Mosaic, or cuDNN kernel implements the basis, and configuration
  rejects them and `auto`. Per score the spline costs more than a monomial kernel (piece
  selection and a select per Horner coefficient in the forward, per-piece coefficient sums in
  the backward), so its throughput and memory come from measurement, not from monomial runs:
  qualify it on each GPU type before production with
  `validation/blocked_strips_qualification.py` (the kernel stage has static and query spline
  cases; the model stage times a recipe's full micro-step), and the acceptance pilot projects
  each run's runtime.
- Restrictions: unnormalized dot-product attention with RoPE or no positional encoding;
  learned direct `per_head` coefficients; no forget gate, PaTH, fixed coefficients,
  `attention_coefficient_orders`, factorial coefficient or amplitude scales, coefficient
  penalty, query readout LR scale, coefficient release schedule, or coefficient override and
  capture (all defined on monomial orders); continuations only with
  `continuation.preserve_all_parameters`. Historical recipes, which lack both fields, keep
  the monomial basis and resume unchanged.
- Diagnostics: each layer records the pooled fractions of measured causal scores that each
  piece evaluates (`spline_piece_<j>_fraction`) and that lie below `u_0` or above `u_K`
  (`spline_below_range_fraction`, `spline_above_range_fraction`, the polynomial
  continuation), beside the usual score and kernel extrema. `spline_score_coverage` holds the
  same fractions and the score extrema per layer and head; `coefficients` reports static
  `theta` (`B0..B7`), or the query generator's `b`, `a`, the column norms of `W`, and the
  generated `theta` on the diagnostic sample. Run metadata records the basis, knots, knot
  vector, piece centres, the exact initial coefficients, and the formula
  (`attention_kernel: learned_spline_unnormalized` or `query_spline_unnormalized`); the run
  index names the family `spline`.
- Acceptance adds `tests/test_spline_attention.py` to both test stages of spline recipes. CPU:
  values and gradients against an independent Cox-de Boor reference, the exact Taylor start and
  its FP32 reproduction, continuity at the knots, both end knots and extrapolation, attention
  and gradients of Q, K, V, `theta`, and the generator against a dense float64 reference
  through the scan and the strips, parameter ownership, initialization, weight decay, and
  scan/strips model equivalence. GPU: production precision with BF16 Q/K/V at 2,048 tokens and
  width 64, scores in every piece and both continuations, against the scan's arithmetic in
  FP64: the strips' output and Q/K/V gradients and the scan's output within 1.25 times the
  BF16 rounding floor, and the coefficient gradients within 1e-4.

### Forget gate

`attention_forget_gate: true` adds the Forgetting Transformer's decay with Gated DeltaNet's
per-head gate (`attention/forget_gate.py`): `log alpha_t = -exp(A_log) softplus(x_t w_alpha + dt_bias)`
from the block's normalized input, with GDN's initialization (`exp(A_log)` uniform on (0, 16),
`softplus(dt_bias)` log-uniform on [0.001, 0.1]); `w_alpha` takes weight decay, `A_log` and
`dt_bias` do not. With `L_t = sum_(j<=t) log alpha_j` (FP32), each weight from key `i` to query
`t` is multiplied by `prod_(j=i+1..t) alpha_j = exp(L_t - L_i)`, one difference per pair:

| Kernel | Gated weight | Positions | Backends |
| --- | --- | --- | --- |
| softmax | `softmax_i(s_ti + L_t - L_i)` | RoPE, none, or PaTH | `blocked`; PaTH also `fused` |
| unnormalized exponential | `exp(s_ti + L_t - L_i)` | RoPE, none, or PaTH | `blocked`; PaTH also `fused` |
| nonlinear query coefficients | `p_t(s_ti) exp(L_t - L_i)` | RoPE, none, or PaTH (tanh) | `blocked`; PaTH also `fused` |
| learned static coefficients, input-free gate | `p(s_ti) exp(-(t-i) lambda)` | RoPE or none | `blocked`, `mosaic` |

`attention_forget_gate_form: sigmoid` replaces the gate with the Forgetting Transformer's
original `log alpha_t = log sigmoid(x_t w_f + b_f)` (Lin et al., 2025), initialized as in the
paper's code (`w_f ~ N(0, 0.02^2)` from the mechanism stream, `b_f = 0`, so every gate starts near
1/2); `w_f` takes weight decay and `b_f` does not. The decay enters each kernel as above.
Softmax and the exponential take the cumulative mode with the input, sigmoid, or constant gate
form, or the query-distance mode with the input form (below); the sigmoid and constant forms need
RoPE or no positional encoding.
`attention_forget_gate_form: constant` gives every head one rate, `log alpha_t = -m_h` at every
token, so `L_t - L_i = -m_h (t - i)`: softmax with this decay is ALiBi (Press et al., 2022) and the
exponential gains ALiBi's bias. `m_h = exp(A_log)` is learned from ALiBi's slopes `2^(-8 h / H)`,
or, with `attention_forget_gate_constant_rate: fixed`, held at those slopes in every layer with no
parameter. This geometric rule is the paper's for every `H`; ALiBi's code uses it only when `H` is a
power of two (for 14 heads it takes `2^-1..2^-8` and `2^-0.5, 2^-1.5, ..., 2^-5.5`). `tests/test_constant_decay.py` checks both kernels against FP64 ALiBi through the scan
and the strips; acceptance runs it for every constant-form recipe on CPU and, for softmax and the
exponential, its 2,048-token BF16 strips case on the GPU.

`attention_forget_gate_mode: query_distance` with the input gate replaces `L_t - L_i` by
`(t - i) log alpha_t`, the query's own rate times the key distance: softmax and the unnormalized
exponential add it to every score, ALiBi with a learned, input-dependent slope per query token
(softmax's row maximum is taken over the decayed logits). It takes the softplus rate with query
conditioning or its query-key form (below), RoPE or no positional encoding, and the blocked backend
(scan or strips); PaTH, bounded rates, per-order gates, the other gate forms, and other backends
are rejected. A nonfinite or positive raw `log alpha_t` poisons its query's row. Its parameters and
their initialization are cumulative FoX's. `tests/test_exponential_query_distance.py` checks both
kernels against FP64 all-pairs references through the scan and the strips; acceptance runs it for
these recipes on CPU and its 2,048-token BF16 strips cases on the GPU.

`attention_forget_gate_conditioning: query_key` gives that gate the query polynomials' endpoint
rate: each pair reads both of its tokens, `log alpha_ti = -exp(A_log) softplus(x_t w_query +
x_i w_key + dt_bias)`, and softmax and the exponential add `(t - i) log alpha_ti` to its score.
Only the token projections reach the attention core; pair rates are evaluated inside the score
blocks, and a nonfinite projection or log rate poisons the rows of the causal pairs it enters (its
query's row, the rows from a key's token on, or every row of a rate's head). The key projection
`w_key` (`forget_gate_key_kernel`, per layer and head) starts at zero and takes weight decay, so the
model starts exactly as its query-gate counterpart; it adds `num_layers * num_heads * hidden_size`
parameters (188,160 in the Transformer++ shape). Diagnostics report `log alpha_ti` over every
causal pair. `tests/test_exponential_query_key.py` checks both kernels against FP64 all-pairs
references through the scan and the strips; acceptance runs it for these recipes on CPU and its
2,048-token BF16 strips cases on the GPU, and `tests/test_endpoint_decay.py` on CPU only (its GPU
cases qualify the query polynomials' kernels).

`attention_forget_gate_per_order: true` with the cumulative mode gives every order `r` of
nonlinear query coefficients (including zero) its own input gate and sum `L_tr`:
`W_ti = sum_r c_r(x_t) s_ti^r exp(L_tr - L_ir)`, each block evaluating one order's
`[B, H, Q, K]` factor at a time. It takes query conditioning, RoPE or no positional encoding,
and the blocked backend (scan or strips); Mosaic has no cumulative kernels. The gates start as
copies of the shared draw, so the model starts as cumulative FoX; each extra order adds
`(hidden_size + 2) * num_layers * num_heads` parameters (188,580 in the Transformer++ shape). A
nonfinite `log alpha` poisons the rows from its token on, as for a shared gate.
`tests/test_per_order_cumulative.py` checks degrees four and eight against FP64; acceptance runs it
for these recipes on CPU and its 2,048-token BF16 strips cases on the GPU.

`positional_encoding: none` skips RoPE, so positions enter only through the causal mask
and the decay. Diagnostics report each head's `log alpha` statistics (per order for per-order
gates, with their half-lives) and the gate's parameters (for a fixed rate, the slopes); `runs.csv`
has an `attention_forget_gate_constant_rate` column, empty without the constant form.

### PaTH position encoding

`positional_encoding: path` replaces RoPE with PaTH (Yang et al., arXiv:2505.16381):
the score between query `t` and key `s <= t` is
`q_t^T H_t H_(t-1) ... H_(s+1) k_s / sqrt(d_K)` with `H_t = I - beta_t w_t w_t^T`.
As in the authors' flash-linear-attention layer, `w_t` comes from a rank-`path_w_rank`
(32) projection shared by a layer's heads, a width-`path_conv_size` (3) causal depthwise
convolution with SiLU, and L2 normalization per head, and
`beta_t = 2 sigmoid(x_t W_beta + b_beta)` lies in (0, 2). The arrays `path_w_down`
(`[layers, hidden, rank]`), `path_w_up`, `path_w_conv`, `path_beta_kernel`, and
`path_beta_bias` (`[layers, heads, ...]`) are drawn from the mechanism stream
(projections N(0, 0.02^2), the convolution with PyTorch's default variance, a zero bias),
so the backbone keeps the softmax control's weights; all but the bias take weight decay.

Set `attention_forget_gate: true` to combine PaTH with FoX (PaTH-FoX). With `L` the
cumulative per-token log forget gate, softmax and the exponential add `L_t - L_s` to each
transported score, and the tanh query polynomial multiplies its weight by `exp(L_t - L_s)`,
on the blocked reference and the fused kernels alike. The public `path_attention` and
`fused_path_attention` APIs accept this optional FP32 `[batch, time, heads]` array as
`log_decay` for every kernel; its gradients propagate through the cumulative sum into the
gate parameters. Causal differences are masked before exponentiation, and gate values are
never factored into separate global exponentials. The fused backward adds O(BTH)
gate-gradient buffers. Only `attention_forget_gate_mode: cumulative` with the input gate
form composes with PaTH; configuration rejects `query_distance` and the constant form with PaTH.

`attention_backend: blocked` runs `attention/path.py`, the authors' chunked algorithm in
JAX: a UT transform per 128-token chunk, then a sweep over chunk distance that scores each
query chunk against the key chunk behind it and moves the queries through that chunk's
Householder product. Automatic differentiation of a rematerialized forward pass gives the
derivative. It supports softmax (normalized, with a running maximum), the unnormalized
exponential, and tanh query-conditioned unnormalized polynomials, each with the forget
gate above. It is tested against the definition and the authors' PyTorch reference (with and
without their gate), and is the reference for the fused kernels and their diagnostics.

`attention_backend: fused` runs FLA's fused Triton kernels
(`attention/backends/fla_path_attention.py`, through `triton_port`, below). Softmax
launches the eleven kernels of FLA's `parallel_path_attn` unchanged and in FLA's order,
with its tiles (64-token UT-transform chunks and key blocks, 128-token query blocks, and
512-token splits on H100 and B300), warps, stages, autotune menus, TF32 products, FP16
copies of `w`, and buffers; outputs FLA updates in place are aliased outputs. The tanh
query polynomial uses the same launches with FLA's six softmax-dependent kernels replaced
by a fork (`fla_path_polynomial_kernels.py`): `o_t = sum_s p_t(s_ts) v_s` without
normalization, so no running maximum, normalizer, or `delta`; the backward's softmax
derivative becomes `dS_ts = p_t'(s_ts) dP_ts / sqrt(d_K)`, and the kernels that visit a
(query, key) block also return `dc_r(t) = sum_s s_ts^r dP_ts`. The unnormalized
exponential is FLA's softmax output times its row sum `2^lse` (FLA's base-2 log-sum-exp);
its derivative is FLA's softmax backward given `do 2^lse` and `delta = 0`, which is exact
because the row sum's own derivative cancels `delta`. For softmax and the exponential,
the forget gate enables FLA's own gate path in six of the kernels (`USE_G`, `USE_GATE`);
FLA adds its gate to the unscaled scores, so the kernels receive `L sqrt(d_K)`, and the
four kernels that form its gradient each write a separate FP32 part that the wrapper sums.
The polynomial fork multiplies each weight by `exp(L_t - L_s)` itself and receives `L`.
Transport, Householder, and UT-transform arithmetic are FLA's. It needs an sm90+ GPU,
`train.dtype: bfloat16` (with FP32 tiles one of FLA's kernels exceeds the shared memory), a
prediction length divisible by 128, equal Q/K and value widths of 16, 32, 64, or 128 (up to
64 on B300, where FLA's dq kernel fails to launch at width 128), and degree 1-4; `auto`
selects it there and the JAX implementation elsewhere. A row with a nonfinite output becomes NaN,
as in `path.py`; a softmax score of -inf gets zero weight, as in FlashAttention. FLA
rounds several intermediates to BF16 (dw and dbeta are stored in BF16): against FP32 its
errors in the output and the q, k, v gradients are 1.5-3 times the JAX implementation's,
and in dw, dbeta, and the coefficient gradients about 8 times (all below 1e-2 at the
production shape). Its backward accumulates dk, dv, dw, and the split products'
gradient (and the gate's) with FP32 atomic additions, so dk, dv, dw, dbeta, and the decay
gradient are not bitwise repeatable; outputs, dq, and the coefficient gradients are.
Accuracy, speed, and training pilots are in
[validation](validation/README.md#fused-path-attention-kernels).

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
and weight-decay policy; `repeat_initial_axis` makes a named random draw once without that
axis and repeats it along the axis, the copies training independently afterwards. The model registers each array once and selects the
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
| `transformer_truncated_reference` | dot-product | N(0, 0.02) conditioned on ±0.04 without variance compensation (actual std ≈0.0175925); attention `o_proj` and MLP `down_proj` draws divided by √(2L) |
| `gla_reference` | GLA | N(0, 0.02); query, key, and both low-rank gate projections Xavier-uniform, gain 2^-2.5 (official GLA code) |
| `gated_deltanet_reference` | Gated DeltaNet | Embeddings and output head N(0, 0.02); Linear weights Uniform(±1/√fan_in); MLP `down_proj` also divided by √(2L) (official `lit_gpt`, `mamba_init`) |

`model.rms_norm_fp32: true` computes backbone RMSNorm squares, means, reciprocal
square roots, and gain multiplication in FP32. Its default is `false`, preserving
historical arithmetic. The truncated initializer is also explicit: adding it does
not change the reference chosen when a new recipe leaves initialization unset.

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
`triton_port` (below): thin wrappers reorder arguments into the bridge's convention, the
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

### Triton kernels written for PyTorch

`attention/backends/triton_port.py` brings Triton kernels written for PyTorch (FLA's)
into JAX without changing them. It replaces their torch host code: `device()` reads the
compute capability from JAX and the opt-in shared memory from the CUDA driver (the
scaffold's CPU-only torch cannot), `Device.has_shared_memory` and `Device.tile` reproduce
FLA's `check_shared_mem` tiers and the tile sizes derived from them, `per_architecture`
selects per-family autotune menus (FLA's reduced Blackwell menus), `autotune` hands a menu
to jaxlib's autotuner, `jit_function` exposes an upstream kernel body, and `chunk_indices`
and `chunk_offsets` build FLA's variable-length chunk tables. `launch(kernel, *inputs,
outputs=..., grid=..., name=..., dot_precision=..., num_warps=..., num_stages=...,
**constexprs)` declares one launch, with outputs `empty(shape, dtype)`, `zeros(shape,
dtype)` (upstream `torch.zeros`, e.g. for atomic sums), or `update(array)` (an array the
kernel modifies in place, aliased to its output). It lowers to the vendored ejkernel
`triton_call` on a GPU; inside `triton_port.interpreter()` the same launches run in
Triton's interpreter on the CPU through `jax.pure_callback`, with tiles chosen for an
emulated device, so a port's orchestration and gradients can be tested without a GPU
(FP32 or FP16 operands: the interpreter cannot compute in BF16). Only Triton kernels are
in scope; CUDA/C++ extensions and calls into PyTorch are not.

Each ported kernel follows one convention:

1. **Wrappers** (`<name>_kernels.py`): `@triton.jit` functions in the bridge's argument
   order (inputs, arrays updated in place, outputs, constexprs) that call the upstream
   body from `jit_function` with its heuristic flags fixed. A kernel whose mathematics
   changes is copied into its own module with the upstream license header and a
   docstring listing every change; unchanged kernels are never copied.
2. **Operation** (`<name>.py`): a `jax.custom_vjp` whose forward and backward follow the
   upstream autograd function's launch sequence, dtype casts, and buffer semantics, with
   tiles, warps, stages, and menus taken from `triton_port.device()` through the
   upstream branches, and an explicit dot precision.
3. **Reference oracle**: a pure-JAX implementation of the same contract, which is the
   recipe's `blocked` backend and computes its diagnostics.
4. **Tests**: CPU tests in `interpreter()` comparing outputs and every gradient with the
   oracle (including the upstream tile branches), and GPU tests (`-m gpu`, no skips) in
   BF16 at the production shape against the FP32-HIGHEST oracle, at tile-boundary
   lengths, for causality, nonfinite inputs, repeatability, and the model's dispatch.
   `acceptance.py` adds both for recipes that resolve to the kernel.
5. **Qualification**: the GPU tests pass on one H100 (`dogo`, `.venv`) and one B300
   (`akita`, `.venv-cuda13`); results, benchmarks, and limitations are recorded in a
   `validation/` section with JSON under `validation/runs/`. Launches run outside XLA
   command buffers (`--xla_gpu_enable_command_buffer=`, set by acceptance's runtime
   environment) with the locked toolkit's `ptxas`.

FLA's Gated DeltaNet and PaTH attention follow it; FLA's GDN launches are unchanged by
the move onto `triton_port` (compared launch for launch and bitwise in
[validation](validation/README.md#fused-path-attention-kernels)).

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
tanh query-conditioned polynomial attention, and unnormalized polynomials with static fixed
or learned coefficients (on the query-coefficient kernel with every query's coefficients
equal), both at degrees 1-12 where B300's dK/dV kernel holds every query's coefficients in
shared memory (`api.query_degree_supported`: 12 at `d_K <= 64` and 9 at `d_K = 128`, with
2,048 tokens), with Q/K width 8, 16, 32, 64, or
128, value width 64, `train.dtype: bfloat16`, and a prediction length divisible by 128
(H100) or 256 (B300). An explicit `mosaic` request raises for other recipes, GPUs, or
shapes; `auto` then uses the fused kernels. Query-distance forget gates of tanh or
sigmoid query polynomials, and the input-free gate of learned static coefficients (on the same
query-decay kernels, every query's coefficients and rates equal), run on the Mosaic decay
kernels only when `mosaic` is requested (`ModelConfig.mosaic_decay_supported`), and so does the tanh query
polynomial evaluated at a transformed score, on the query kernels with `phi(s)` in their tiles, on H100 alone
(`ModelConfig.mosaic_score_transform_supported`; [Score transforms](#score-transforms)). Shared gates must fit B300 like the gate-free
kernels. Per-order gates need only fit H100's kernels, which stage their vectors per tile
(`api.hopper_supported`, every degree to 12). B300's kernels keep every query's and key's
per-order vectors for the whole sequence. They raise `ValueError` before tracing a kernel
for shapes `api.blackwell_supported` rejects: above degree 7 (query) or 6 (query–key) at
`d_K = 64` and 2,048 tokens. Diagnostics come from the blocked reference,
as for the fused kernels. They are the canonical kernels for these mechanisms, and
`scaffolds.records create` rejects a fresh recipe for them that requests `fused` unless
`--allow-triton-attention` is supplied. Acceptance runs the Mosaic GPU tests only at the
run's weight precision (`--mosaic-precision`).

`model.attention_blocked_implementation` chooses how the blocked backend runs dot-product
attention with RoPE or no positional encoding. Both choices live in
`attention/backends/reference.py` and evaluate the same per-block arithmetic, one shared
function: the same scores, kernels, normalizations, and invalid-row rules for every mechanism.

| Value | Function | Execution |
| --- | --- | --- |
| `scan` (default) | `blocked_dot_product_attention` | Checkpointed scan of 128-query blocks, each against all keys. Autodiff sums each block's K and V cotangents in the input dtype: BF16 dK and dV carry 1.8-2.8x the BF16 rounding floor at 2,048 tokens ([validation](validation/README.md)). |
| `strips` | `strip_dot_product_attention` | Causal 512-query strips (`STRIP_SIZE`; the last may be shorter), each against its key prefix only, each under `jax.checkpoint` so its backward recomputes it. K, V, and the other per-key operands are converted to FP32 once before the strips, so their cotangents sum in FP32 and round once. |

- Precision: the products read the BF16 values (exact in FP32 and TF32) with FP32
  accumulation, weights stay FP32, and the output has V's dtype, as in the scan. dQ has no sum
  across strips. Every conversion of a BF16 operand to FP32 (each strip's queries; K, V, and
  the other operands once) and of the output back to V's dtype is written with
  `reduce_precision` (`_rounded_to`). With the default `xla_allow_excess_precision`, XLA GPU
  removes an f32 -> bf16 -> f32 conversion pair, so plain conversions next to the model's FP32
  math (QK RMSNorm, RoPE, head RMSNorm) would hand the forward unrounded FP32 values and the
  backward's recomputation the saved BF16 ones. `tests/test_blocked_strips_h100.py` checks the
  roundings in compile-only H100 programs (it skips without the CUDA plugin; acceptance does not
  run it).
- Memory: a strip's working set is FP32 [batch, heads, 512, keys] blocks, 4x the scan's
  128-query rows, and K and V are kept in FP32 for the backward. Compile-only H100 estimates of
  one update's temporary memory on the blocked backend, scan then strips: the 15-layer
  width-896 softmax recipe at micro-batch 16, 23.6 and 26.0 GB; FoX query m = 4 there, 26.8 and
  26.1 GB; 10-layer width-1024 recipes with 16 heads at micro-batch 64, query m = 4 66.1 and
  82.4 GB (73.9 GB with 256-query strips) and normalized learned m = 4 63.3 and 57.2 GB. A
  recipe near a GPU's limit may need a smaller micro-batch; gradient accumulation keeps its
  scientific batch.
- The one intended difference: masked keys are never multiplied, so a nonfinite V entry poisons
  the rows from the start of its strip onward. The scan multiplies every key's V by each block's
  weights, and a masked zero weight times NaN or inf poisons every row.
- The strips' checkpoints keep CSE prevention on. Unlike the scan's loop body, unrolled strips
  have no loop boundary: without it XLA merges each recomputation with its forward and keeps
  every layer's scores until its backward. The strips assume a static sequence length.
- Diagnostics (`return_diagnostics`) still come from the scan, forward only, as for the fused
  kernels. Kernels are still qualified against the scan, and the coefficient fit's
  blocked-reference check runs it whatever the parent recipe sets.
- PaTH, Gated DeltaNet, and GLA keep their own blocked code and reject `strips`. Other resolved
  backends ignore the field, so `auto` recipes use the strips wherever they resolve to `blocked`.
- Historical recipes and resumptions keep `scan`, the default. `scaffolds.records create` writes
  `strips` into fresh dot-product recipes (RoPE or no positional encoding) on the `blocked` or
  `auto` backend that leave the field unset, and rejects an explicit `scan` there unless
  `--allow-scan-blocked-attention` is supplied. Recipes on other backends keep the inert `scan`,
  so they match their historical partners. A launcher that allocates copied recipes with
  `create_run` for the blocked backend applies the same rule
  (`configs.base.fresh_recipe_uses_blocked_strips`), as the Transformer++ continuation launcher
  does.
- The field is an execution setting, like the backend: a continuation, including one that
  preserves all parameters, may train on the strips from a parent that trained on the scan.
- Run metadata records `attention_implementation: blocked_causal_strips`,
  `attention_blocked_implementation`, `attention_strip_size`, and the diagnostic scan
  (`diagnostic_backend`, `diagnostic_query_block_size`). Acceptance adds
  `tests/test_blocked_strips.py` to the CPU tests of recipes that train on the strips.

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

Each worktree syncs its own environment from this scaffold's lock; hardlinks share
package files with the uv cache, so an additional environment costs almost no disk:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
# Blackwell (B200, B300) environment from the same lock: CUDA 13 with cuBLAS >= 13.2.
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

`train.eval_early_every` and `train.eval_early_until` optionally freeze a denser early
evaluation cadence. Steps through `eval_early_until` use `eval_early_every`; later
steps use `eval_every`, and the terminal step always evaluates. The schedule uses
global steps across checkpoint resumptions. Unset/zero early settings preserve the
original single cadence.

`record.json` summarizes status, source, execution attempts, acceptance, runtime,
resolved configuration and backend, data identity, and retained checkpoints.
`config.json`, `metadata.json`, completion and retention files are raw trainer
outputs retained for exact resume validation. Resolving `auto` settings updates
these execution records while preserving `config.yaml`. Metrics stay in
`metrics.jsonl`; logs, checkpoints and W&B files are colocated ignored artifacts.
Shared helpers regenerate the study's `runs.csv` index. Figures and comparisons
belong under `results/<comparison>/` and select run IDs explicitly.

### Packed-data tools

A packed corpus is one token stream of documents, each followed by an end-of-document
token, cut into fixed rows. Both tools check every source shard against its manifest
checksum, write a new packed corpus that `PackedBatchIterator` reads, and record the source
manifest and its SHA256 in the new manifest's metadata.

```bash
# The same stream re-cut into rows of another length (metadata.repacked_from).
python -m scaffolds.transformer.data.repack SOURCE/manifest.json DESTINATION --sequence-length 4097
# One row per complete document of at least sequence_length - 1 tokens whose first
# --opening-tokens tokens (default 1,024) no earlier document begins with, in stream order:
# an end-of-document token, then the document's first sequence_length - 1 tokens
# (metadata.long_documents_from).
python -m scaffolds.transformer.data.long_documents SOURCE/manifest.json DESTINATION --sequence-length 16385
```

`long_documents` also writes `documents.npy`, every complete document's
`[start, length, first_start]` in the stream (where its opening first appears), and
`rows.npy`, each row's document; the manifest pins both by SHA256. Every row is the first
appearance of its document.
Offsets are stream tokens, which repacking preserves, so a run that trained on rows
`[0, n)` of a corpus with row length L consumed stream tokens `[0, n * L)`. The rows it never
saw, in content as well as in position, are the suffix from `first_unseen_row(manifest,
n * L)`, and `row_documents(manifest)` gives each row's document.

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
coefficient penalty in the training objective; under a coefficient release schedule
it excludes still-frozen coefficient entries, as clipping does. The optimizer already computes
this norm for its numerical checks; logging adds no gradient-tree reduction.
Retained training logs keep the sampled norm and clipping scale. Each validation
event in `metrics.jsonl` stores the norm count, mean, maximum, and clipped fraction
over **every update** since the preceding validation or resume, together with the
clipping threshold. W&B also plots the window mean, maximum, and clipped fraction.
The default saved workspace from `views.experiment_workspace` includes gradient
norm plots on a logarithmic axis and separate clipping-scale/fraction plots.
Historical runs without these measurements retain unknown gradient diagnostics.

Training telemetry (`training/telemetry.py`) is standard for training with
`model.remat: false`, on one GPU or several. Every update's statistics reach the host
in one packed vector:

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
bit-identical with and without it, on one device and on two, and on GPU the
differences match GPU nondeterminism between identical updates. Statistics of one
batch are the same whether its equal-sized microbatches run on one device or split
across several.

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

### Gradient-event capture

`train.gradient_event_capture: true` saves the exact state an update started from when its
pre-clip gradient norm spikes after warmup (`training/gradient_events.py`). The rule and
retention are recipe fields, fixed before launch:

| Field | Default | Meaning |
| --- | --- | --- |
| `gradient_event_factor` | 5.0 | multiple of the running median the norm must exceed |
| `gradient_event_min_norm` | 1.0 | floor of the threshold |
| `gradient_event_history` | 100 | finite pre-clip norms of earlier updates in the median |
| `gradient_event_min_history` | 20 | earlier norms required before any event |
| `gradient_event_max_states` | 3 | kept states: the first event and the largest later ones |

- Rule: update *k* is an event when *k* > `optim.warmup_steps`, its norm *g* is finite, at
  least `min_history` earlier norms are known, and *g* > max(`min_norm`, `factor` × median).
  The history includes warmup updates and events, and resumes with the checkpoint.
- Capture: the pre-update state (parameters, Adam moments, `step` *k* − 1) is saved
  synchronously in the checkpoint manager's `{"state": state}` layout at
  `<checkpoints>/gradient_events/<k>/<k-1>/default`, with the event's record as
  `<k>/event.json`. Each `<k>` is a one-step checkpoint-manager root whose step directory
  names the state's own step; `checkpointing.restore_state(<k>/<k-1>, template)`, or a
  `StandardCheckpointer` restore of `<k>/<k-1>/default`, reads it back. The run's step
  manager ignores the non-numeric `gradient_events` root, and Git ignores it with the
  checkpoints.
- Log: every event, kept or not, is appended to the run's `gradient_events.jsonl` and printed
  as `gradient_event`. A record holds the update and pre-update step, loss, pre-clip norm,
  clip scale, running median and threshold, learning rate, sequence range and microbatch
  layout, batch and RNG SHA256 (as the checkpoint audit hashes them), seed and RNG
  derivation, the update's and preceding update's telemetry, device kind and count, backend
  and implementation, source commit, frozen-execution hash, attempt ID, the pre-update
  state's SHA256, and whether it was kept and which capture it replaced. The log is
  append-only: a resume appends a `resume` row with its `restored_step`, and rows that earlier
  attempts logged for later updates are superseded, because the resumed run repeats them.
- Retention: a larger later event replaces the smallest kept later event (the first is never
  replaced); the replaced directory is deleted once no restorable checkpoint references it.
  A nonfinite update also keeps its pre-update state, outside the cap, before its numerical
  failure is recorded.
- Resumption: `gradient_events/state.json` records the history and kept list of each
  checkpoint a resume may restore, written before that checkpoint's save. A resume restores
  its checkpoint's snapshot, and deletes and logs (`capture_discarded`, with the reason for
  each directory) captures that snapshot does not reference: those made after that checkpoint,
  whose updates it repeats, the capture of a superseded numerical failure, and a capture
  replaced at or before that checkpoint.
- Records: a completed run and a numerical failure list their kept captures in `record.json`
  under `diagnostic_checkpoints`: directory descriptors of `<k>` with SHA256 and a
  justification ("pre-update state of gradient event at update *k*, norm *g*"). A run that
  trains again supersedes that list and lists none until it reaches its own outcome, as a
  first execution does. `retention.py` keeps them and nothing else of the capture root, and
  lists them in `retention.json` (`retained_gradient_event_updates`). `validate_record`
  checks them like checkpoints and reports them unavailable on other clusters.
- Cost: capture never changes training; on CPU losses, gradient norms, and states are
  bit-identical with and without it, on one device and on two. One-device steps do not
  donate their input, so holding the pre-update state across the update costs no device
  memory. The data-parallel step donates it, so each of its updates first copies the state
  (one more state per device: 2.4 GB for the 15-layer width-896 recipes). An event costs a
  state hash, and a kept one a synchronous save comparable to a checkpoint save.
- Acceptance adds `tests/test_gradient_events.py` to the CPU tests of every recipe with capture on
  (its data-parallel case runs in the two-device stage).

## Launch, acceptance, retention, and analysis

Both shared launchers default to one GPU. Independent training runs are submitted
without inter-run dependencies. Keep the recipe's global batch size fixed; the
training runtime uses gradient accumulation for the configured microbatch. On one
device the whole update, all of its microbatches, is one compiled step (a `lax.scan`),
as in the data-parallel step; dispatching each microbatch separately had left a single
GPU idle for about 40% of every update. An
explicit multi-GPU allocation must override both Slurm's `--gres` request and the
launcher's fourth positional argument (`GPU_COUNT`), for example `--gres=gpu:2`
and `2` on Kennel, or `--gres=gpu:h200:2` and `2` on ORCD. Allocation changes are
recorded as runtime provenance and trigger acceptance for that allocation.
Kennel defaults to `--partition=standard,long` with a 24-hour allocation. Submitters
use the available standard and long allowances before placing overflow in sweep,
checking current limits and queued/running allocations as documented in
[CLUSTERS.md](../../CLUSTERS.md). Longer jobs explicitly request `--partition=long`
and an appropriate time limit. Routine launches add no artificial nice penalty.
The Kennel launcher resolves Slurm's granted `TimeLimit` through
`scaffolds/launch/allocation.py` and exports `TRANSFORMER_ALLOCATION_SECONDS`.
Acceptance records this budget and includes it in its identity, so a shorter
allocation cannot reuse a passed runtime gate from a longer one. The projected
runtime must fit with a two-hour reserve. ORCD deliberately resumes across its
6-hour or 12-hour allocations, so it retains the adapter's `allocation_hours`
qualification horizon (48 hours by default) instead of limiting full training to
one backfilled allocation. Other callers use that same fallback unless they set
the runtime override; this budget never changes the recipe.

The runtime projection separates the disposable pilot's elapsed time (including
startup and compilation, counted once) from production work. The pilot enables
`run_training(..., profile_timing=True)` to warm diagnostics and evaluation
separately, then measure diagnostics once per validation and at least three warm
global evaluation batches. Median warm update, diagnostic, and batch timings are
multiplied only by their corresponding production counts; checkpoint estimates
use the largest measured write time. The pilot's updates never replace production
updates. Continuations count their remaining updates and baseline validation.
`runtime-projection.json` and `pilot.json` expose the component costs, with the
projection also printed before the allocation check. The two-hour reserve covers
other acceptance work and runtime variation. Normal training leaves profiling off;
its validation data, metrics, optimizer, and update sequence are unchanged.

Studies configure these shared modules with execution modules at their root
(for example `experiments/<study>/acceptance.py`):

- `launchers/kennel.sbatch` runs any experiment on Kennel: it resolves the allocated
  GPU type, checks the frozen execution set against the launch commit
  (`scaffolds/launch/frozen.py`), uses the worktree's environment, pins `CUDA_HOME`
  to the locked CUDA toolkit, runs acceptance, then trains; `training.log` appends
  across requeues. An optional sixth positional argument, `STOP_AFTER_STEP`, ends
  a pilot at that global update while keeping the recipe's full LR horizon. It is
  recorded in the attempt, and final checkpoint retention is skipped for pilots;
  continuation without that argument performs final retention. The frozen set includes study `sweep.yaml`, every run's
  `config.yaml`, root execution modules, and shared record/launch helpers. Metrics,
  records and `results/` remain editable while jobs run. Both launchers source
  `scaffolds/launch/watchdog.sh`: when `training.log` has not grown for
  `TRANSFORMER_STALL_MINUTES` (default 20), the job logs its GPUs' state, ends the
  running step, and exits with code 124. The finalizer requeues the job, which resumes
  from its latest checkpoint, for a run's first two stalls and records a third as a failure.
- `launchers/orcd.sbatch` does the same on ORCD. Its allocations fit that cluster's
  per-user GPU caps; a job asks for 12 hours, accepts `mit_normal_gpu`'s 6-hour
  maximum, and requeues itself before its limit (see [CLUSTERS.md](../../CLUSTERS.md)).
- `acceptance.py` runs identity-keyed acceptance in the production allocation: CPU
  tests, distributed CPU tests on two forced host devices, and single-GPU kernel tests
  (no skips), an initialization hash, a full-recipe pilot, and an exact fresh-process
  resume. The kernel tests run with `JAX_PLATFORMS=cuda,cpu`: CUDA is the default, and
  the host CPU is there for FP64 references; the workers and training list CUDA alone.
  The identity includes the allocated GPU model, so a requeue onto another
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
  the selection, keeping a capturing run's kept gradient-event states as recorded
  diagnostic checkpoints.

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

Each command runs its tests in one process. `tests/conftest.py` clears JAX's caches after
any test that leaves the process holding more than half of `vm.max_map_count` memory
mappings, and stops the run with an explicit error if clearing cannot keep it below nine
tenths.

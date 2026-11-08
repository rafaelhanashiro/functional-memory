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
validated 0.90 allocator fraction. Explicit `TRANSFORMER_HOST_STAGING=0` or `1`
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
query-conditioned.

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
`query_key` adds `wk_r·x_i` inside softplus. Gate arrays add an order axis
before their final channel axis. All orders initially duplicate the shared
gate's draw; key projections start at zero. The copies then learn independently.
Projection matrices receive WD; rates and biases remain exempt. Diagnostics
retain the layer, head, and order axes. Cumulative gates, sigmoid rates, and
other backends reject this option. Existing recipes default to a shared gate.

`attention_forget_gate_form: input` (the default) is that gate. The `constant` form
drops its token dependence: one learned log rate per layer and head,
`log(alpha_t) = -exp(A_log)`, so every weight decays as `exp(-exp(A_log) * (t - i))`,
ALiBi's linear bias in log space. Rates start at ALiBi's slopes `2^(-8h/H)` for heads
`h = 1..H`, identically in each layer, and are exempt from weight decay. The constant
form is qualified only for query polynomials with the cumulative mode under RoPE or no
positional encoding on the blocked backend; configuration rejects it with `query_distance`,
PaTH, softmax, or the exponential.

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

Softmax and the exponential take the cumulative mode with the input gate form only.
`positional_encoding: none` skips RoPE, so positions enter only through the causal mask
and the decay. Diagnostics report each head's `log alpha` statistics and the gate's
parameters.

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
sigmoid query polynomials run on the Mosaic decay kernels only when `mosaic` is requested
(`ModelConfig.mosaic_decay_supported`). Shared gates must fit B300 like the gate-free
kernels. Per-order gates need only fit H100's kernels, which stage their vectors per tile
(`api.hopper_supported`, every degree to 12). B300's kernels keep every query's and key's
per-order vectors for the whole sequence. They raise `ValueError` before tracing a kernel
for shapes `api.blackwell_supported` rejects: above degree 7 (query) or 6 (query–key) at
`d_K = 64` and 2,048 tokens. Diagnostics come from the blocked reference,
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

Each command runs its tests in one process. `tests/conftest.py` clears JAX's caches after
any test that leaves the process holding more than half of `vm.max_map_count` memory
mappings, and stops the run with an explicit error if clearing cannot keep it below nine
tenths.

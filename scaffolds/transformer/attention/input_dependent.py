"""Unnormalized polynomials with query-dependent coefficients.

The incoming attention activation (after the block's pre-attention RMSNorm)
produces coefficients per query and head. The score operands, including QK normalization
and RoPE, retain the ordinary dot-product path. Coefficient projection and basis
combination use FP32; the final attention output has V's dtype.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp

# The feature f of affine generators c = b + a*f(x W) and of pure generators c = f(x W).
AFFINE_FEATURES = {"tanh": jnp.tanh, "affine_sigmoid": jax.nn.sigmoid}
PURE_FEATURES = {"sigmoid": jax.nn.sigmoid, "pure_tanh": jnp.tanh}
# Per-order multipliers of diagonal tanh coefficients: 2 sigmoid(x U + beta), or Gated
# DeltaNet's exp(-exp(A_log) softplus(x w + dt_bias)) without a distance exponent.
COEFFICIENT_GATES = ("none", "sigmoid", "gdn")
GDN_GATE_PARAMETERS = ("query_polynomial_gate_kernel", "query_polynomial_gate_log_rate",
                       "query_polynomial_gate_dt_bias")


def parameter_specs(*, hidden_size: int, degree: int = 1, activation: str = "identity",
                    amplitude_initial: float = 1.0, projection_initial_std: float = 0.0,
                    tanh_bias: bool = False, conditioning: str = "query", readout: str = "diagonal",
                    feature_count: int | None = None, kernel_weight_decay: bool = False, gate: str = "none"):
    """b starts at 1/r!; W at zero or N(0, projection_initial_std^2); tanh's a at amplitude_initial
    and its optional inner bias beta at zero. "query_key" adds the key factor's W_k (same draw
    rule, independent draws) and a_k (same start as a). Dense readout projects onto K =
    feature_count tanh features (degree+1 when null) and mixes them through V [degree+1, K],
    diagonal at amplitude_initial when K = degree+1 and otherwise zero. affine_sigmoid has the
    same parameters as tanh. Pure sigmoid and pure_tanh coefficients have no b or a: W alone,
    and dense V. The "sigmoid" coefficient gate adds U [D, degree+1] and its bias [degree+1],
    both exactly zero and drawing no random numbers. The "gdn" gate adds w [D, degree+1],
    A_log and dt_bias [degree+1] with the input-form forget gate's distributions, drawn once
    per layer and head and repeated over the orders. kernel_weight_decay decays the projections
    W, W_k, U and w; b, a, a_k, beta, the gate biases, A_log, and V never decay."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    if readout not in ("diagonal", "dense"):
        raise ValueError(f"Unknown coefficient readout: {readout}")
    if gate not in COEFFICIENT_GATES:
        raise ValueError(f"Unknown coefficient gate: {gate!r}")
    if gate != "none" and (activation != "tanh" or readout != "diagonal" or conditioning != "query" or tanh_bias):
        raise ValueError("Coefficient gates require diagonal tanh query coefficients without an inner bias")
    if readout == "dense" and (activation not in (*AFFINE_FEATURES, *PURE_FEATURES) or conditioning != "query"):
        raise ValueError("Dense coefficient readout requires nonlinear query-only coefficients")
    count = degree + 1
    features = count if feature_count is None else feature_count
    if features != count and (readout != "dense" or amplitude_initial != 0):
        raise ValueError("A feature count other than degree+1 requires dense readout with V starting at zero")
    kernel = AttentionParameterSpec(
        name="query_polynomial_kernel", argument="query_polynomial_kernel",
        initial=(0.0,) * (hidden_size * features), initial_shape=(hidden_size, features),
        sharing="per_head", weight_decay=bool(kernel_weight_decay), initial_std=float(projection_initial_std),
    )
    if activation in PURE_FEATURES:
        if tanh_bias or conditioning != "query":
            raise ValueError(f"Pure {activation} coefficients are query-only, without an inner bias")
        if amplitude_initial != 1.0:
            raise ValueError(f"Pure {activation} coefficients have no amplitude; dense V starts at the identity")
        if readout == "diagonal":
            return (kernel,)
        return (kernel, AttentionParameterSpec(
            name="query_polynomial_mixing", argument="query_polynomial_mixing",
            initial=tuple(1.0 if output == feature else 0.0 for output in range(count) for feature in range(features)),
            initial_shape=(count, features), sharing="per_head", weight_decay=False,
        ))
    specs = (
        AttentionParameterSpec(
            name="query_polynomial_bias", argument="query_polynomial_bias",
            initial=tuple(1.0 / math.factorial(r) for r in range(count)), sharing="per_head", weight_decay=False,
        ),
        kernel,
    )
    if activation in AFFINE_FEATURES:
        if readout == "dense":
            specs += (AttentionParameterSpec(
                name="query_polynomial_mixing", argument="query_polynomial_mixing",
                initial=tuple(float(amplitude_initial) if output == feature else 0.0
                              for output in range(count) for feature in range(features)),
                initial_shape=(count, features), sharing="per_head", weight_decay=False,
            ),)
        else:
            specs += (AttentionParameterSpec(
                name="query_polynomial_amplitude", argument="query_polynomial_amplitude",
                initial=(float(amplitude_initial),) * count, sharing="per_head", weight_decay=False,
            ),)
        if tanh_bias:
            specs += (AttentionParameterSpec(
                name="query_polynomial_tanh_bias", argument="query_polynomial_tanh_bias",
                initial=(0.0,) * features, sharing="per_head", weight_decay=False,
            ),)
    elif activation != "identity":
        raise ValueError(f"Unknown coefficient activation: {activation}")
    elif tanh_bias:
        raise ValueError("A bias inside tanh requires tanh coefficients")
    if conditioning == "query_key":
        if activation != "tanh":
            raise ValueError("Query-key coefficients require tanh coefficients")
        specs += (
            AttentionParameterSpec(
                name="key_polynomial_kernel", argument="key_polynomial_kernel",
                initial=(0.0,) * (hidden_size * count), initial_shape=(hidden_size, count),
                sharing="per_head", weight_decay=bool(kernel_weight_decay), initial_std=float(projection_initial_std),
            ),
            AttentionParameterSpec(
                name="key_polynomial_amplitude", argument="key_polynomial_amplitude",
                initial=(float(amplitude_initial),) * count, sharing="per_head", weight_decay=False,
            ),
        )
    elif conditioning != "query":
        raise ValueError(f"Unknown input-dependent coefficient conditioning: {conditioning}")
    if gate == "sigmoid":
        specs += (
            AttentionParameterSpec(
                name="query_polynomial_gate_kernel", argument="query_polynomial_gate_kernel",
                initial=(0.0,) * (hidden_size * count), initial_shape=(hidden_size, count),
                sharing="per_head", weight_decay=bool(kernel_weight_decay),
            ),
            AttentionParameterSpec(
                name="query_polynomial_gate_bias", argument="query_polynomial_gate_bias",
                initial=(0.0,) * count, sharing="per_head", weight_decay=False,
            ),
        )
    elif gate == "gdn":
        # One forget-gate draw per layer and head (the draw omits the order axis, so the
        # kernel's fan-in is hidden_size), repeated over the orders: every order starts with
        # the same multiplier, which the per-head output RMSNorm cancels up to its epsilon.
        kernel_name, log_rate_name, dt_bias_name = GDN_GATE_PARAMETERS
        specs += (
            AttentionParameterSpec(
                name=kernel_name, argument=kernel_name, initial=(0.0,) * (hidden_size * count),
                initial_shape=(hidden_size, count), sharing="per_head", weight_decay=bool(kernel_weight_decay),
                initializer="uniform_fan_in", repeat_initial_axis=-1,
            ),
            AttentionParameterSpec(
                name=log_rate_name, argument=log_rate_name, initial=(0.0,) * count, sharing="per_head",
                weight_decay=False, initializer="mamba2_log_rate", repeat_initial_axis=-1,
            ),
            AttentionParameterSpec(
                name=dt_bias_name, argument=dt_bias_name, initial=(0.0,) * count, sharing="per_head",
                weight_decay=False, initializer="mamba2_dt_bias", repeat_initial_axis=-1,
            ),
        )
    return specs


def _diagonal_readout(features, amplitude, bias, amplitude_scale, orders=None):
    """The same rounded FP32 diagonal computation in both readout families."""
    amplitude = amplitude.astype(jnp.float32)
    if amplitude_scale == "factorial":
        amplitude = amplitude / jnp.asarray([math.factorial(r) for r in
                                             (range(bias.shape[-1]) if orders is None else orders)], jnp.float32)
    # The incoming bias may itself have been factorial-scaled by the caller.
    # Preserve each rounded operand before multiplying/adding so the compiler
    # cannot reassociate scaling or contract a different expression in one arm.
    amplitude, bias = jax.lax.optimization_barrier((amplitude, bias.astype(jnp.float32)))
    product = jax.lax.optimization_barrier(amplitude * features)
    return jax.lax.optimization_barrier(product + bias)


def _pure_coefficients(x, kernel, mixing, amplitude_scale, readout_scale, feature):
    """FP32 c_r = f(x w_r) [/ r!], or dense c_r = readout_scale * sum_j (V_rj [/ r!]) f(x w_j)."""
    if x.ndim != 3 or kernel.ndim != 3 or kernel.shape[1] != x.shape[-1]:
        raise ValueError("Pure coefficient projection requires x [B,T,D] and kernel [H,D,K]")
    if mixing is not None and (mixing.ndim != 3 or mixing.shape[0] != kernel.shape[0]
                               or mixing.shape[-1] != kernel.shape[-1]):
        raise ValueError("Dense pure coefficients require a mixing matrix [head,output_order,feature]")
    if amplitude_scale not in ("none", "factorial"):
        raise ValueError(f"Unknown amplitude scale: {amplitude_scale}")
    projected = jnp.einsum(
        "btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )
    features = feature(projected)
    count = kernel.shape[-1] if mixing is None else mixing.shape[1]
    factorials = jnp.asarray([math.factorial(r) for r in range(count)], jnp.float32)
    if mixing is None:
        return features / factorials if amplitude_scale == "factorial" else features
    mixing = mixing.astype(jnp.float32)
    if amplitude_scale == "factorial":
        mixing = mixing / factorials[None, :, None]
    if readout_scale != 1.0:
        mixing = mixing * jnp.asarray(readout_scale, jnp.float32)
    return jnp.einsum("bthj,hrj->bthr", features, mixing,
                      precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)


def coefficient_gate_logits(x, kernel, bias):
    """FP32 per-order gate pre-activations x U + beta with shape [B,T,H,R]."""
    if (x.ndim != 3 or kernel.ndim != 3 or kernel.shape[1] != x.shape[-1]
            or bias.shape != (kernel.shape[0], kernel.shape[-1])):
        raise ValueError("Coefficient gate projection requires x [B,T,D], kernel [H,D,R], bias [H,R]")
    projected = jnp.einsum(
        "btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )
    return projected + bias.astype(jnp.float32)


def project_coefficient_gates(x, kernel, bias):
    """FP32 per-order gates g = sigmoid(x U + beta) with shape [B,T,H,R]; exactly 1/2 while U = beta = 0."""
    return jax.nn.sigmoid(coefficient_gate_logits(x, kernel, bias))


def gdn_gate_log_multipliers(logits, log_rate):
    """FP32 log alpha = -exp(A_log) softplus(x w + dt_bias) with shape [B,T,H,R].

    ``logits`` are the softplus inputs x w + dt_bias (coefficient_gate_logits with dt_bias as
    its bias) and ``log_rate`` is A_log [H,R]. A nonfinite input or A_log gives NaN rather
    than a multiplier saturated at zero or one.
    """
    if logits.ndim != 4 or log_rate.ndim != 2 or logits.shape[-2:] != log_rate.shape:
        raise ValueError("The GDN coefficient gate requires inputs [B,T,H,R] and A_log [H,R]")
    logits, log_rate = logits.astype(jnp.float32), log_rate.astype(jnp.float32)
    log_alpha = -jnp.exp(log_rate) * jax.nn.softplus(logits)
    return jnp.where(jnp.isfinite(logits) & jnp.isfinite(log_rate), log_alpha, jnp.nan)


def project_coefficients(x, kernel, bias, *, amplitude=None, mixing=None, activation="identity", tanh_bias=None,
                         amplitude_scale="none", readout_scale=1.0, orders=None, gate_kernel=None, gate_bias=None,
                         gate_log_rate=None):
    """FP32 c=b+xW or c=b+a*f(xW+beta), f = tanh or sigmoid; the inner bias beta is optional and
    otherwise absent. Pure coefficients c=f(xW) take bias=None (see _pure_coefficients).

    A dense mixing matrix V [head,output_order,feature] replaces diagonal amplitude; its K
    features (kernel [H,D,K]) may differ in number from the R = degree+1 coefficients.
    amplitude_scale="factorial" divides output order r's amplitude or mixing row by r!.
    readout_scale multiplies only the dense modulation; it never rescales the bias.
    orders selects the actual monomial orders for compact diagonal tanh features;
    omitted outputs are zero, and factorial scaling uses those actual orders.
    gate_kernel U [H,D,R] and gate_bias [H,R] multiply diagonal tanh coefficients by
    2*sigmoid(x U + gate_bias) per order: exactly 1 while both are zero. With gate_log_rate
    A_log [H,R] the multiplier is instead Gated DeltaNet's exp(-exp(A_log) softplus(x U +
    gate_bias)), gate_bias acting as dt_bias, with no distance exponent.
    """
    if (gate_kernel is None) != (gate_bias is None):
        raise ValueError("A coefficient gate requires both its kernel and its bias")
    if gate_log_rate is not None and (gate_kernel is None or gate_log_rate.shape != gate_bias.shape):
        raise ValueError("The GDN coefficient gate requires its kernel, dt_bias and A_log [H,R]")
    if gate_kernel is not None and (activation != "tanh" or mixing is not None or tanh_bias is not None
                                    or orders is not None or bias is None
                                    or gate_kernel.shape != kernel.shape or gate_bias.shape != bias.shape):
        raise ValueError("Coefficient gates require diagonal tanh coefficients without an inner bias or "
                         "selected orders, with kernel [H,D,R] and bias [H,R] matching the generator's")
    if orders is not None and (activation != "tanh" or mixing is not None or bias is None
                                or len(orders) != kernel.shape[-1]):
        raise ValueError("Selected orders require matching compact diagonal tanh features")
    if (isinstance(readout_scale, bool) or not isinstance(readout_scale, (int, float))
            or not math.isfinite(readout_scale) or readout_scale <= 0):
        raise ValueError("Dense readout scale must be finite and positive")
    if readout_scale != 1.0 and (activation not in (*AFFINE_FEATURES, *PURE_FEATURES) or mixing is None):
        raise ValueError("A nondefault readout scale requires dense tanh or sigmoid coefficients")
    if activation in PURE_FEATURES:
        if bias is not None or amplitude is not None or tanh_bias is not None:
            raise ValueError("Pure sigmoid and tanh coefficients have no bias, amplitude, or inner bias")
        return _pure_coefficients(x, kernel, mixing, amplitude_scale, readout_scale, PURE_FEATURES[activation])
    if (x.ndim != 3 or kernel.ndim != 3 or kernel.shape[1] != x.shape[-1] or bias.ndim != 2
            or bias.shape[0] != kernel.shape[0]):
        raise ValueError("Query coefficient projection requires x [B,T,D], kernel [H,D,K], bias [H,R]")
    rectangular = kernel.shape[-1] != bias.shape[-1]
    if rectangular and (activation not in AFFINE_FEATURES or mixing is None):
        raise ValueError("Only dense tanh readout may use a feature count other than the coefficient count")
    projected = jnp.einsum(
        "btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )
    if activation in AFFINE_FEATURES:
        if mixing is None:
            if amplitude is None or amplitude.shape != bias.shape:
                raise ValueError("Tanh coefficients require a model-owned amplitude with the bias shape")
        elif amplitude is not None or mixing.shape != (*bias.shape, kernel.shape[-1]):
            raise ValueError("Dense tanh coefficients require only a mixing matrix [head,output_order,feature]")
        if tanh_bias is not None:
            if tanh_bias.shape != (kernel.shape[0], kernel.shape[-1]):
                raise ValueError("The bias inside tanh must have the feature shape [H,K]")
            projected = projected + tanh_bias.astype(jnp.float32)
        if amplitude_scale not in ("none", "factorial"):
            raise ValueError(f"Unknown amplitude scale: {amplitude_scale}")
        # Dense readout reuses these features for its diagonal and residual.
        # Materialize them in both families: fusing tanh only in the diagonal
        # arm introduced one-ulp differences amplified by later BF16 residuals.
        features = jax.lax.optimization_barrier(AFFINE_FEATURES[activation](projected))
        if mixing is None:
            coefficients = _diagonal_readout(features, amplitude, bias, amplitude_scale, orders)
            if orders is not None:
                from scaffolds.transformer.attention.parameters import expand_polynomial_coefficients

                coefficients = expand_polynomial_coefficients(coefficients, orders[-1], orders)
            if gate_kernel is not None:
                # A separate projection leaves W's projection and the readout as they were;
                # 2*sigmoid(0) is exactly 1, so zero U and bias reproduce the ungated values.
                if gate_log_rate is None:
                    multiplier = jax.lax.optimization_barrier(
                        2.0 * project_coefficient_gates(x, gate_kernel, gate_bias))
                else:
                    multiplier = jax.lax.optimization_barrier(jnp.exp(gdn_gate_log_multipliers(
                        coefficient_gate_logits(x, gate_kernel, gate_bias), gate_log_rate)))
                coefficients = coefficients * multiplier
            return coefficients
        mixing = mixing.astype(jnp.float32)
        if readout_scale != 1.0:
            # Preserve the historical square/rectangular arithmetic at scale 1.
            # The new scale applies to each output-order/feature contribution,
            # after any factorial scaling, while b retains its original value.
            if amplitude_scale == "factorial":
                factorials = jnp.asarray([math.factorial(r) for r in range(bias.shape[-1])], jnp.float32)
                mixing = mixing / factorials[None, :, None]
            mixing = mixing * jnp.asarray(readout_scale, jnp.float32)
            return bias.astype(jnp.float32) + jnp.einsum(
                "bthj,hrj->bthr", features, mixing,
                precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
            )
        if rectangular:
            # No diagonal exists: c_r = b_r + sum_j (V_rj [/ r!]) h_j, every V entry differentiated.
            if amplitude_scale == "factorial":
                factorials = jnp.asarray([math.factorial(r) for r in range(bias.shape[-1])], jnp.float32)
                mixing = mixing / factorials[None, :, None]
            readout = jnp.einsum(
                "bthj,hrj->bthr", features, mixing,
                precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
            )
            return bias.astype(jnp.float32) + readout
        amplitude = jnp.diagonal(mixing, axis1=-2, axis2=-1)
        diagonal = _diagonal_readout(features, amplitude, bias, amplitude_scale)
        off_diagonal = jnp.where(jnp.eye(bias.shape[-1], dtype=bool), 0.0, mixing)
        if amplitude_scale == "factorial":
            factorials = jnp.asarray([math.factorial(r) for r in range(bias.shape[-1])], jnp.float32)
            off_diagonal = off_diagonal / factorials[None, :, None]
        residual = jnp.einsum(
            "bthj,hrj->bthr", features, off_diagonal,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        # The zero residual is still differentiated: every off-diagonal entry
        # can learn immediately, including at V = diag(a).
        return diagonal + residual
    elif (activation != "identity" or amplitude is not None or mixing is not None or tanh_bias is not None
          or amplitude_scale != "none"):
        raise ValueError("Identity coefficients cannot receive an amplitude, its scale, or an inner bias; "
                         "unknown activation otherwise")
    return projected + bias.astype(jnp.float32)


def project_key_factors(x, kernel, amplitude):
    """FP32 per-key factors f = 1 + a*tanh(xW) with shape [B,T,H,R]; exactly one while a = 0."""
    if (x.ndim != 3 or kernel.ndim != 3 or kernel.shape[1] != x.shape[-1]
            or amplitude.shape != (kernel.shape[0], kernel.shape[-1])):
        raise ValueError("Key factor projection requires x [B,T,D], kernel [H,D,R], amplitude [H,R]")
    projected = jnp.einsum(
        "btd,hdr->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )
    return 1.0 + amplitude.astype(jnp.float32) * jnp.tanh(projected)


def query_key_polynomial_attention(q, k, v, coefficients, key_factors, *, backend="blocked",
                                   return_diagnostics=False, blocked_implementation="scan"):
    """sum_(i<=t) sum_r c_r(x_t) f_r(x_i) s_ti^r v_i: per-query coefficients times per-key factors.

    The blocked reference evaluates the pairwise coefficient directly, and the fused Triton
    kernel forms it inside each tile in one pass. The Mosaic query-coefficient kernel runs once
    per order with the key factor folded into the values, v_i^(r) = f_r(x_i) v_i, and the
    coefficient vector c_r(x_t) e_r; its m+1 partial outputs are rounded to V's dtype before
    they are summed, which at degree one corrupts the gradient after head RMSNorm.
    """
    from scaffolds.transformer.attention.backends.reference import (
        blocked_backend_attention, blocked_dot_product_attention,
    )

    if coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3] or coefficients.shape[-1] < 2:
        raise ValueError("Query coefficients require [batch,time,heads,degree+1], degree >= 1")
    if key_factors.shape != coefficients.shape:
        raise ValueError("Key factors must match the query coefficient shape [batch,time,heads,degree+1]")
    degree = coefficients.shape[-1] - 1
    arguments = dict(attention_degree=degree, attention_normalization=False,
                     query_polynomial_coefficients=coefficients, key_polynomial_factors=key_factors)
    reference = partial(blocked_dot_product_attention, q, k, v, **arguments)
    if backend == "blocked":
        result = blocked_backend_attention(q, k, v, implementation=blocked_implementation,
                                           return_diagnostics=return_diagnostics, **arguments)
        if not return_diagnostics:
            return result
        output, diagnostics = result
    elif backend == "fused":
        from scaffolds.transformer.attention.backends.fused_query_key_polynomial import (
            fused_query_key_polynomial_attention,
        )

        output = fused_query_key_polynomial_attention(q, k, v, coefficients.astype(jnp.float32),
                                                      key_factors.astype(jnp.float32))
        if not return_diagnostics:
            return output
        _, diagnostics = reference(return_diagnostics=True)
    elif backend == "mosaic":
        from scaffolds.transformer.attention.backends.mosaic.api import query_polynomial_attention as attend

        coefficients32, factors32 = coefficients.astype(jnp.float32), key_factors.astype(jnp.float32)
        total = None
        for order in range(degree + 1):
            selected = jnp.zeros_like(coefficients32).at[..., order].set(coefficients32[..., order])
            values = (v.astype(jnp.float32) * factors32[..., order, None]).astype(v.dtype)
            partial_output = attend(q, k, values, selected).astype(jnp.float32)
            total = partial_output if total is None else total + partial_output
        output = total.astype(v.dtype)
        output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
        if not return_diagnostics:
            return output
        _, diagnostics = reference(return_diagnostics=True)
    else:
        raise ValueError("Query-key polynomial attention requires a blocked, fused or mosaic backend")
    diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(coefficients)))
    diagnostics.update(jax.lax.stop_gradient(key_factor_statistics(key_factors)))
    return output, diagnostics


def key_factor_statistics(key_factors):
    """Actual per-key factors on the diagnostic sample, per head and order."""
    values = key_factors.astype(jnp.float32)
    return {
        "key_factor_mean": jnp.mean(values, axis=(0, 1)),
        "key_factor_std": jnp.std(values, axis=(0, 1)),
        "key_factor_min": jnp.min(values, axis=(0, 1)),
        "key_factor_max": jnp.max(values, axis=(0, 1)),
    }


def query_polynomial_attention(q, k, v, coefficients, *, backend="blocked", return_diagnostics=False,
                               log_decay=None, decay_mode="cumulative",
                               query_gate_input=None, key_gate_input=None, gate_log_rate=None,
                               blocked_implementation="scan"):
    """Apply a per-query polynomial directly in one fused attention operation.

    ``log_decay`` [batch,time,heads] is the cumulative FP32 sum L_t of log alpha in
    ``cumulative`` mode, multiplying each weight by exp(L_t-L_i). In ``query_distance``
    mode it is log alpha_t itself, giving exp((t-i)*log alpha_t); endpoint gate inputs give
    exp(-(t-i) exp(A) softplus(u_t + v_i)). Cumulative decay uses the blocked backend;
    query-distance and endpoint decay also run on the Mosaic kernels.
    An extra order axis applies independent decay inside the polynomial sum, on the blocked
    and Mosaic backends. These gates include the constant term.
    """
    from scaffolds.transformer.attention.backends.reference import (
        blocked_backend_attention, blocked_dot_product_attention,
    )

    if (coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3]
            or coefficients.shape[-1] < 2):
        raise ValueError("Query coefficients require [batch,time,heads,degree+1], degree >= 1")
    if decay_mode not in ("cumulative", "query_distance"):
        raise ValueError(f"Unknown query decay mode: {decay_mode}")
    endpoint = any(value is not None for value in (query_gate_input, key_gate_input, gate_log_rate))
    per_order = ((log_decay is not None and log_decay.ndim == 4)
                 or (query_gate_input is not None and query_gate_input.ndim == 4))
    if endpoint and (any(value is None for value in (query_gate_input, key_gate_input, gate_log_rate))
                     or log_decay is not None or decay_mode != "query_distance"):
        raise ValueError("Endpoint decay requires all three operands and query-distance mode without log decay")
    if decay_mode != "cumulative" and log_decay is None and not endpoint:
        raise ValueError("A query-distance decay mode requires log_decay")
    if (log_decay is not None or endpoint) and not (
            backend == "blocked" or backend == "mosaic" and decay_mode == "query_distance"):
        raise ValueError("The forget gate runs on the blocked reference backend; query-distance and endpoint "
                         "decay also run on mosaic")
    degree = coefficients.shape[-1] - 1
    if backend == "blocked":
        result = blocked_backend_attention(
            q, k, v, implementation=blocked_implementation, attention_degree=degree,
            attention_normalization=False, query_polynomial_coefficients=coefficients,
            query_log_decay=log_decay, query_decay_mode=decay_mode, query_gate_input=query_gate_input,
            key_gate_input=key_gate_input, gate_log_rate=gate_log_rate,
            return_diagnostics=return_diagnostics,
        )
        if not return_diagnostics:
            return result
        output, diagnostics = result
    elif backend in ("fused", "mosaic"):
        if backend == "fused":
            from scaffolds.transformer.attention.backends.fused_query_polynomial import (
                fused_query_polynomial_attention as attend,
            )
        else:
            from scaffolds.transformer.attention.backends.mosaic import api as mosaic

            attend = mosaic.query_polynomial_attention
            if endpoint:
                attend = partial(mosaic.per_order_endpoint_decay_attention if per_order
                                 else mosaic.endpoint_decay_attention, query_gate_input=query_gate_input,
                                 key_gate_input=key_gate_input, gate_log_rate=gate_log_rate)
            elif log_decay is not None:
                attend = partial(mosaic.per_order_query_decay_attention if per_order
                                 else mosaic.query_decay_attention, log_decay=log_decay)

        output = attend(q, k, v, coefficients)
        if not return_diagnostics:
            return output
        # Diagnostics, including the forget gate's, come from the blocked reference.
        _, diagnostics = blocked_dot_product_attention(
            q, k, v, attention_degree=degree, attention_normalization=False,
            query_polynomial_coefficients=coefficients, query_log_decay=log_decay,
            query_decay_mode=decay_mode, query_gate_input=query_gate_input,
            key_gate_input=key_gate_input, gate_log_rate=gate_log_rate, return_diagnostics=True,
        )
    else:
        raise ValueError("Query-conditioned polynomial attention requires a blocked, fused or mosaic backend")
    diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(coefficients)))
    return output, diagnostics


def coefficient_statistics(coefficients):
    """Actual coefficients on the diagnostic sample, per head and order."""
    values = coefficients.astype(jnp.float32)
    return {
        "query_coefficient_mean": jnp.mean(values, axis=(0, 1)),
        "query_coefficient_std": jnp.std(values, axis=(0, 1)),
        "query_coefficient_min": jnp.min(values, axis=(0, 1)),
        "query_coefficient_max": jnp.max(values, axis=(0, 1)),
    }


def coefficient_gate_statistics(gate_logits, ungated, gated, q, k, *, gate_log_rate=None):
    """Gated query coefficients on the diagnostic sample, per head and order [H,R] (kernel: [H]).

    Gates g_r = sigmoid(gate_logits) [B,T,H,R] or, given A_log [H,R] as gate_log_rate, the GDN
    multipliers alpha_r = exp(-exp(A_log_r) softplus(gate_logits)); the ungated c_r; and the
    order tilt log g_r - mean_r log g_r, which a factor shared by all orders (cancelled by per-head
    output RMSNorm) leaves unchanged. Each order's contribution c~_r(x_t) s_ti^r is measured
    over causal pairs i <= t with the reference's scores s = q k^T / sqrt(d_K) (BF16 operands,
    FP32 accumulation): RMS, mean |.|, and its mean share of each row's total |.| over orders;
    attention_kernel_abs_mean [H] is the mean |sum_r c~_r s^r|. The gated c~ itself is summarized
    by coefficient_statistics. Measurements only: nothing here enters the forward pass.
    """
    logits = gate_logits.astype(jnp.float32)
    if gate_log_rate is None:
        gates = jax.nn.sigmoid(logits)
        log_gates = jax.nn.log_sigmoid(logits)
    else:
        log_gates = gdn_gate_log_multipliers(logits, gate_log_rate)
        gates = jnp.exp(log_gates)
    tilt = log_gates - jnp.mean(log_gates, axis=-1, keepdims=True)
    ungated = ungated.astype(jnp.float32)
    batch, time, heads, count = gated.shape
    if k.shape[2] != q.shape[2]:
        k = jnp.repeat(k, q.shape[2] // k.shape[2], axis=2)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) * (1.0 / math.sqrt(q.shape[-1]))
    power = jnp.broadcast_to(jnp.tril(jnp.ones((time, time), jnp.float32)), scores.shape)
    coefficients = jnp.transpose(gated.astype(jnp.float32), (0, 2, 1, 3))
    kernel = jnp.zeros_like(scores)
    squares, magnitudes, rows = [], [], []
    for order in range(count):
        if order:
            power = power * scores
        term = coefficients[..., order, None] * power
        kernel = kernel + term
        squares.append(jnp.sum(term * term, axis=(0, 2, 3)))
        magnitudes.append(jnp.sum(jnp.abs(term), axis=(0, 2, 3)))
        rows.append(jnp.sum(jnp.abs(term), axis=-1))
    pairs = batch * time * (time + 1) / 2
    rows = jnp.stack(rows, axis=-1)
    totals = jnp.sum(rows, axis=-1, keepdims=True)
    shares = jnp.where(totals > 0, rows / jnp.where(totals > 0, totals, 1.0), 0.0)
    causal = jnp.tril(jnp.ones((time, time), bool))
    return jax.lax.stop_gradient({
        "coefficient_gate_mean": jnp.mean(gates, axis=(0, 1)),
        "coefficient_gate_std": jnp.std(gates, axis=(0, 1)),
        "coefficient_gate_min": jnp.min(gates, axis=(0, 1)),
        "coefficient_gate_max": jnp.max(gates, axis=(0, 1)),
        "coefficient_gate_log_tilt_mean": jnp.mean(tilt, axis=(0, 1)),
        "coefficient_gate_log_tilt_std": jnp.std(tilt, axis=(0, 1)),
        "coefficient_gate_ungated_mean": jnp.mean(ungated, axis=(0, 1)),
        "coefficient_gate_ungated_std": jnp.std(ungated, axis=(0, 1)),
        "coefficient_gate_contribution_rms": jnp.sqrt(jnp.stack(squares, axis=-1) / pairs),
        "coefficient_gate_contribution_abs_mean": jnp.stack(magnitudes, axis=-1) / pairs,
        "coefficient_gate_contribution_row_share": jnp.mean(shares, axis=(0, 2)),
        "coefficient_gate_attention_kernel_abs_mean": jnp.sum(jnp.where(causal, jnp.abs(kernel), 0.0),
                                                              axis=(0, 2, 3)) / pairs,
    })


def query_linear_attention(q, k, v, coefficients, *, backend="blocked", return_diagnostics=False,
                           blocked_implementation="scan"):
    """sum_(i<=t) (c0_t + c1_t * <q_t,k_i>/sqrt(d_K)) v_i.

    The fused path reuses the linear learned-polynomial kernel and adds the
    constant basis via a FP32 prefix sum. Its forward tiles also check the actual
    c0+c1*s kernel, so nonfinite weights cannot disappear behind zero values.
    Both bases must be representable in FP32 before their combination; extreme
    coefficients/cancellation can therefore fail earlier than direct evaluation.
    Its VJP propagates through both bases,
    both coefficients, and the coefficient projection. The blocked path directly
    evaluates the query-dependent polynomial and supplies kernel diagnostics.
    """
    from scaffolds.transformer.attention.backends.reference import (
        blocked_backend_attention, blocked_dot_product_attention,
    )

    if coefficients.shape != (*q.shape[:3], 2):
        raise ValueError("Query coefficients must have shape [batch,time,heads,2]")
    if backend == "blocked":
        result = blocked_backend_attention(
            q, k, v, implementation=blocked_implementation, attention_degree=1, attention_normalization=False,
            query_polynomial_coefficients=coefficients, return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            output, diagnostics = result
        else:
            return result
    elif backend == "fused":
        from scaffolds.transformer.attention.backends.fused_learned_polynomial import (
            fused_learned_polynomial_attention,
        )

        linear = fused_learned_polynomial_attention(
            q, k, v, jnp.broadcast_to(jnp.asarray([0.0, 1.0], jnp.float32), (q.shape[2], 2)),
            keep_output_float32=True, query_kernel_guard=coefficients,
        )
        prefix = jnp.cumsum(v.astype(jnp.float32), axis=1)
        c = coefficients.astype(jnp.float32)
        output = (c[..., 0, None] * prefix + c[..., 1, None] * linear).astype(v.dtype)
        output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
        if not return_diagnostics:
            return output
        _, diagnostics = blocked_dot_product_attention(
            q, k, v, attention_degree=1, attention_normalization=False,
            query_polynomial_coefficients=coefficients, return_diagnostics=True,
        )
    else:
        raise ValueError("Query-conditioned attention requires a blocked or fused backend")
    diagnostics.update(jax.lax.stop_gradient(coefficient_statistics(coefficients)))
    return output, diagnostics

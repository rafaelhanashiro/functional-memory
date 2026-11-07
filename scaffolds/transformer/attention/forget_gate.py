"""Forgetting Transformer decay with Gated DeltaNet's per-head Mamba-2 gate.

Each attention weight from key i to query t is multiplied by prod_(j=i+1..t) alpha_j in
cumulative FoX, or by alpha_t ** (t-i) in query-distance mode, with
log alpha_t = -exp(A_log) softplus(x_t w_alpha + dt_bias) per layer and head, computed from
the attention input x (after the block's pre-attention RMSNorm). The parameterization and the
initialization are Gated DeltaNet's: exp(A_log) ~ Uniform(0, 16), softplus(dt_bias)
log-uniform on [0.001, 0.1] (floored at 1e-4), and w_alpha a PyTorch-default Linear row,
Uniform(+-1/sqrt(hidden_size)). As in Gated DeltaNet, w_alpha decays and A_log, dt_bias do not.

For query-distance gates, ``query_key`` conditioning adds a zero-initialized key projection
inside softplus; its pairwise input uses just the two endpoints. The query-only ``sigmoid``
rate option bounds lambda by exp(A_log), initializing sigmoid(dt_bias) to the same legacy dt
draw. Its rate agrees at zero input; tokenwise rates and derivatives are not identical.

The ``constant`` form drops the token dependence: log alpha_t = -exp(A_log) per layer and
head, so every weight decays as exp(-lambda_h (t - i)), ALiBi's linear bias in log space.
lambda_h starts at ALiBi's geometric slopes 2^(-8 h / H), h = 1..H, in every layer.
"""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp

PARAMETERS = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")
KEY_PARAMETER = "forget_gate_key_kernel"
ALL_PARAMETERS = (*PARAMETERS, KEY_PARAMETER)
CONSTANT_PARAMETERS = ("forget_gate_log_rate",)
FORMS = ("input", "constant")


def parameter_names(form: str = "input", conditioning: str = "query") -> tuple[str, ...]:
    if form not in FORMS:
        raise ValueError(f"Unknown forget gate form: {form}")
    if conditioning not in ("query", "query_key") or (form == "constant" and conditioning != "query"):
        raise ValueError("Unsupported forget gate conditioning")
    return CONSTANT_PARAMETERS if form == "constant" else (*PARAMETERS, KEY_PARAMETER) if conditioning == "query_key" else PARAMETERS


def alibi_log_slopes(num_heads: int) -> list[float]:
    """log of ALiBi's slopes 2^(-8 h / H), h = 1..H: half-lives from about one token upward."""
    return [-8.0 * (head + 1) / num_heads * math.log(2.0) for head in range(num_heads)]


def parameter_specs(*, hidden_size: int, form: str = "input", conditioning: str = "query",
                    rate_activation: str = "softplus", order_count: int | None = None):
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    parameter_names(form, conditioning)
    if order_count is not None and (form != "input" or rate_activation != "softplus"
                                   or isinstance(order_count, bool) or not isinstance(order_count, int)
                                   or order_count < 2):
        raise ValueError("Per-order gates require input-form softplus and at least two orders")
    if rate_activation not in ("softplus", "sigmoid") or (rate_activation != "softplus" and (form != "input" or conditioning != "query")):
        raise ValueError("Unsupported forget gate rate activation")
    if form == "constant":
        return (AttentionParameterSpec(name="forget_gate_log_rate", argument="forget_gate_log_rate", initial=(0.0,),
                                       sharing="per_head", weight_decay=False, initializer="alibi_log_slope"),)
    specs = (
        AttentionParameterSpec(name="forget_gate_kernel", argument="forget_gate_kernel",
                               initial=(0.0,) * hidden_size, initial_shape=(hidden_size,), sharing="per_head",
                               weight_decay=True, initializer="uniform_fan_in"),
        AttentionParameterSpec(name="forget_gate_log_rate", argument="forget_gate_log_rate", initial=(0.0,),
                               sharing="per_head", weight_decay=False, initializer="mamba2_log_rate"),
        AttentionParameterSpec(name="forget_gate_dt_bias", argument="forget_gate_dt_bias", initial=(0.0,),
                               sharing="per_head", weight_decay=False, initializer="mamba2_dt_bias" if rate_activation == "softplus" else "mamba2_sigmoid_bias"),
    )
    if conditioning == "query_key":
        specs += (AttentionParameterSpec(name=KEY_PARAMETER, argument=KEY_PARAMETER, initial=(0.0,) * hidden_size,
                                        initial_shape=(hidden_size,), sharing="per_head", weight_decay=True),)
    if order_count is not None:
        specs = tuple(replace(spec, initial=spec.initial * order_count,
                              initial_shape=(order_count, len(spec.initial)),
                              repeat_initial_axis=-2) for spec in specs)
    return specs


def constant_log_decay(x, log_rate):
    """FP32 log alpha [batch,time,heads] = -exp(A_log) for every token, from x [B,T,D] and log_rate [H,1]."""
    if x.ndim != 3 or log_rate.ndim != 2 or log_rate.shape[1] != 1:
        raise ValueError("The constant forget gate requires x [B,T,D] and log_rate [H,1]")
    rate = jnp.exp(log_rate[:, 0].astype(jnp.float32))
    return jnp.broadcast_to(-rate, (x.shape[0], x.shape[1], log_rate.shape[0]))


def log_decay(x, kernel, log_rate, dt_bias, *, rate_activation="softplus"):
    """FP32 log alpha [B,T,H] or [B,T,H,R], with an optional order axis in every parameter."""
    if (x.ndim != 3 or kernel.ndim not in (2, 3) or kernel.shape[-1] != x.shape[-1]
            or log_rate.shape != (*kernel.shape[:-1], 1) or log_rate.shape != dt_bias.shape):
        raise ValueError("The forget gate requires x [B,T,D], kernel [H,D], log_rate and dt_bias [H,1]")
    projected = jnp.einsum("btd,hd->bth" if kernel.ndim == 2 else "btd,hrd->bthr",
                           x.astype(jnp.float32), kernel.astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    rate = jnp.exp(log_rate[..., 0].astype(jnp.float32))
    if rate_activation not in ("softplus", "sigmoid"):
        raise ValueError(f"Unknown forget gate rate activation: {rate_activation}")
    inputs = projected + dt_bias[..., 0].astype(jnp.float32)
    if rate_activation == "softplus":
        result = -rate * jax.nn.softplus(inputs)
        if kernel.ndim == 3:
            return jnp.where(jnp.isfinite(inputs) & jnp.isfinite(log_rate[..., 0]), result, jnp.nan)
        return result
    # Saturating finite inputs may yield zero rate, but nonfinite parameters/activations
    # must not be hidden by sigmoid's finite endpoints.
    valid = jnp.isfinite(inputs) & jnp.isfinite(log_rate[..., 0])
    return jnp.where(valid, -rate * jax.nn.sigmoid(inputs), jnp.nan)


def project_gate_inputs(x, kernel, dt_bias, key_kernel):
    """Endpoint projections [B,T,H] or [B,T,H,R]; evaluate pair rates inside score blocks."""
    if (x.ndim != 3 or kernel.ndim not in (2, 3) or kernel.shape[-1] != x.shape[-1]
            or key_kernel.shape != kernel.shape or dt_bias.shape != (*kernel.shape[:-1], 1)):
        raise ValueError("Endpoint gate requires x [B,T,D], kernels [H,D], and bias [H,1]")
    def project(weight):
        return jnp.einsum("btd,hd->bth" if kernel.ndim == 2 else "btd,hrd->bthr",
                          x.astype(jnp.float32), weight.astype(jnp.float32),
                          precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return project(kernel) + dt_bias[..., 0].astype(jnp.float32), project(key_kernel)

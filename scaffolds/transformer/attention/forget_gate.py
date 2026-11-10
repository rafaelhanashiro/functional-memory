"""Forgetting Transformer decay with Gated DeltaNet's per-head Mamba-2 gate.

Each attention weight from key i to query t is multiplied by prod_(j=i+1..t) alpha_j in
cumulative FoX, or by alpha_t ** (t-i) in query-distance mode (softmax and the exponential add
the factor's logarithm to the score), with
log alpha_t = -exp(A_log) softplus(x_t w_alpha + dt_bias) per layer and head, computed from
the attention input x (after the block's pre-attention RMSNorm). The parameterization and the
initialization are Gated DeltaNet's: exp(A_log) ~ Uniform(0, 16), softplus(dt_bias)
log-uniform on [0.001, 0.1] (floored at 1e-4), and w_alpha a PyTorch-default Linear row,
Uniform(+-1/sqrt(hidden_size)). As in Gated DeltaNet, w_alpha decays and A_log, dt_bias do not.
Per-order gates (``order_count``) give every polynomial order its own input-form parameters,
copies of the shared draw at initialization, in either mode.

For query-distance gates, ``query_key`` conditioning adds a zero-initialized key projection
inside softplus; its pairwise input uses just the two endpoints, and on softmax and the exponential
the pair's (t-i) log alpha_ti joins its score. The query-only ``sigmoid``
rate option bounds lambda by exp(A_log), initializing sigmoid(dt_bias) to the same legacy dt
draw. Its rate agrees at zero input; tokenwise rates and derivatives are not identical.

The ``constant`` form drops the token dependence: log alpha_t = -exp(A_log) per layer and
head, so every weight decays as exp(-lambda_h (t - i)), ALiBi's linear bias in log space; on
softmax and the exponential the decay adds -lambda_h (t - i) to every score, ALiBi itself.
lambda_h starts at ALiBi's geometric slopes 2^(-8 h / H), h = 1..H, in every layer. With the
``fixed`` constant rate lambda_h stays at those slopes and the form has no parameter: the
fixed log rate and the learned rate's start are the same FP32 array (``alibi_log_rate``).

The ``sigmoid`` form is the Forgetting Transformer's original gate (Lin et al., 2025):
alpha_t = sigmoid(x_t w_f + b_f) per layer and head, so log alpha_t = -softplus(-(x_t w_f + b_f)).
As in the paper's code, w_f ~ N(0, 0.02^2) and b_f = 0, so every gate starts near 1/2, and
w_f decays while b_f does not. The input form equals sigmoid(-(x_t w_alpha + dt_bias))
raised to the per-head power exp(A_log). It takes query conditioning and the softplus rate.
"""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp

PARAMETERS = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")
KEY_PARAMETER = "forget_gate_key_kernel"
CONSTANT_PARAMETERS = ("forget_gate_log_rate",)
SIGMOID_PARAMETERS = ("forget_gate_kernel", "forget_gate_bias")
ALL_PARAMETERS = (*PARAMETERS, KEY_PARAMETER, "forget_gate_bias")
FORMS = ("input", "constant", "sigmoid")
# The constant form's rate: one learned log rate per layer and head, or ALiBi's slopes fixed.
CONSTANT_RATES = ("learned", "fixed")
# The Forgetting Transformer's N(0, 0.02^2) draw for w_f (HuggingFace LLaMA's initializer_range).
SIGMOID_KERNEL_STD = 0.02


def parameter_names(form: str = "input", conditioning: str = "query",
                    constant_rate: str = "learned") -> tuple[str, ...]:
    if form not in FORMS:
        raise ValueError(f"Unknown forget gate form: {form}")
    if conditioning not in ("query", "query_key") or (form != "input" and conditioning != "query"):
        raise ValueError("Unsupported forget gate conditioning")
    if constant_rate not in CONSTANT_RATES:
        raise ValueError(f"Unknown constant forget gate rate: {constant_rate}")
    if constant_rate != "learned" and form != "constant":
        raise ValueError("A fixed forget gate rate requires the constant forget gate form")
    if form != "input":
        return {"constant": CONSTANT_PARAMETERS if constant_rate == "learned" else (),
                "sigmoid": SIGMOID_PARAMETERS}[form]
    return (*PARAMETERS, KEY_PARAMETER) if conditioning == "query_key" else PARAMETERS


def alibi_log_slopes(num_heads: int) -> list[float]:
    """log of ALiBi's slopes 2^(-8 h / H), h = 1..H: half-lives from about one token upward."""
    return [-8.0 * (head + 1) / num_heads * math.log(2.0) for head in range(num_heads)]


def alibi_log_rate(num_heads: int) -> jax.Array:
    """FP32 log rates [heads, 1] at ALiBi's slopes: the fixed constant rate and the learned one's start."""
    return jnp.asarray(alibi_log_slopes(num_heads), jnp.float32).reshape(num_heads, 1)


def parameter_specs(*, hidden_size: int, form: str = "input", conditioning: str = "query",
                    rate_activation: str = "softplus", constant_rate: str = "learned",
                    order_count: int | None = None):
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    names = parameter_names(form, conditioning, constant_rate)
    if order_count is not None and (form != "input" or rate_activation != "softplus"
                                   or isinstance(order_count, bool) or not isinstance(order_count, int)
                                   or order_count < 2):
        raise ValueError("Per-order gates require input-form softplus and at least two orders")
    if rate_activation not in ("softplus", "sigmoid") or (rate_activation != "softplus" and (form != "input" or conditioning != "query")):
        raise ValueError("Unsupported forget gate rate activation")
    if form == "constant":
        # A fixed rate is ALiBi's slopes themselves (alibi_log_rate): nothing is trained.
        return tuple(AttentionParameterSpec(name=name, argument=name, initial=(0.0,), sharing="per_head",
                                            weight_decay=False, initializer="alibi_log_slope") for name in names)
    if form == "sigmoid":
        return (
            AttentionParameterSpec(name="forget_gate_kernel", argument="forget_gate_kernel",
                                   initial=(0.0,) * hidden_size, initial_shape=(hidden_size,), sharing="per_head",
                                   weight_decay=True, initial_std=SIGMOID_KERNEL_STD),
            AttentionParameterSpec(name="forget_gate_bias", argument="forget_gate_bias", initial=(0.0,),
                                   sharing="per_head", weight_decay=False),
        )
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


def sigmoid_log_decay(x, kernel, bias):
    """FP32 log alpha [batch,time,heads] = log sigmoid(x w_f + b_f) from x [B,T,D], kernel [H,D], bias [H,1]."""
    if x.ndim != 3 or kernel.shape != (bias.shape[0], x.shape[-1]) or bias.ndim != 2 or bias.shape[1] != 1:
        raise ValueError("The sigmoid forget gate requires x [B,T,D], kernel [H,D] and bias [H,1]")
    projected = jnp.einsum("btd,hd->bth", x.astype(jnp.float32), kernel.astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    return jax.nn.log_sigmoid(projected + bias[:, 0].astype(jnp.float32))


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

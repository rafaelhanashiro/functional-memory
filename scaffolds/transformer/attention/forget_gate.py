"""Forgetting Transformer decay with Gated DeltaNet's per-head Mamba-2 gate.

Each attention weight from key i to query t is multiplied by prod_(j=i+1..t) alpha_j in
cumulative FoX, or by alpha_t ** (t-i) in query-distance mode, with
log alpha_t = -exp(A_log) softplus(x_t w_alpha + dt_bias) per layer and head, computed from
the attention input x (after the block's pre-attention RMSNorm). The parameterization and the
initialization are Gated DeltaNet's: exp(A_log) ~ Uniform(0, 16), softplus(dt_bias)
log-uniform on [0.001, 0.1] (floored at 1e-4), and w_alpha a PyTorch-default Linear row,
Uniform(+-1/sqrt(hidden_size)). As in Gated DeltaNet, w_alpha decays and A_log, dt_bias do not.

The ``constant`` form drops the token dependence: log alpha_t = -exp(A_log) per layer and
head, so every weight decays as exp(-lambda_h (t - i)), ALiBi's linear bias in log space.
lambda_h starts at ALiBi's geometric slopes 2^(-8 h / H), h = 1..H, in every layer.

The ``sigmoid`` form is the Forgetting Transformer's original gate (Lin et al., 2025):
alpha_t = sigmoid(x_t w_f + b_f) per layer and head, so log alpha_t = -softplus(-(x_t w_f + b_f)).
As in the paper's code, w_f ~ N(0, 0.02^2) and b_f = 0, so every gate starts near 1/2, and
w_f decays while b_f does not. The input form equals sigmoid(-(x_t w_alpha + dt_bias))
raised to the per-head power exp(A_log).
"""

import math

import jax
import jax.numpy as jnp

PARAMETERS = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")
CONSTANT_PARAMETERS = ("forget_gate_log_rate",)
SIGMOID_PARAMETERS = ("forget_gate_kernel", "forget_gate_bias")
ALL_PARAMETERS = (*PARAMETERS, "forget_gate_bias")
FORMS = ("input", "constant", "sigmoid")
# The Forgetting Transformer's N(0, 0.02^2) draw for w_f (HuggingFace LLaMA's initializer_range).
SIGMOID_KERNEL_STD = 0.02


def parameter_names(form: str = "input") -> tuple[str, ...]:
    if form not in FORMS:
        raise ValueError(f"Unknown forget gate form: {form}")
    return {"input": PARAMETERS, "constant": CONSTANT_PARAMETERS, "sigmoid": SIGMOID_PARAMETERS}[form]


def alibi_log_slopes(num_heads: int) -> list[float]:
    """log of ALiBi's slopes 2^(-8 h / H), h = 1..H: half-lives from about one token upward."""
    return [-8.0 * (head + 1) / num_heads * math.log(2.0) for head in range(num_heads)]


def parameter_specs(*, hidden_size: int, form: str = "input"):
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    if form not in FORMS:
        raise ValueError(f"Unknown forget gate form: {form}")
    if form == "constant":
        return (AttentionParameterSpec(name="forget_gate_log_rate", argument="forget_gate_log_rate", initial=(0.0,),
                                       sharing="per_head", weight_decay=False, initializer="alibi_log_slope"),)
    if form == "sigmoid":
        return (
            AttentionParameterSpec(name="forget_gate_kernel", argument="forget_gate_kernel",
                                   initial=(0.0,) * hidden_size, initial_shape=(hidden_size,), sharing="per_head",
                                   weight_decay=True, initial_std=SIGMOID_KERNEL_STD),
            AttentionParameterSpec(name="forget_gate_bias", argument="forget_gate_bias", initial=(0.0,),
                                   sharing="per_head", weight_decay=False),
        )
    return (
        AttentionParameterSpec(name="forget_gate_kernel", argument="forget_gate_kernel",
                               initial=(0.0,) * hidden_size, initial_shape=(hidden_size,), sharing="per_head",
                               weight_decay=True, initializer="uniform_fan_in"),
        AttentionParameterSpec(name="forget_gate_log_rate", argument="forget_gate_log_rate", initial=(0.0,),
                               sharing="per_head", weight_decay=False, initializer="mamba2_log_rate"),
        AttentionParameterSpec(name="forget_gate_dt_bias", argument="forget_gate_dt_bias", initial=(0.0,),
                               sharing="per_head", weight_decay=False, initializer="mamba2_dt_bias"),
    )


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


def log_decay(x, kernel, log_rate, dt_bias):
    """FP32 log alpha [batch,time,heads] from x [B,T,D], kernel [H,D], log_rate and dt_bias [H,1]."""
    if x.ndim != 3 or kernel.shape != (log_rate.shape[0], x.shape[-1]) or log_rate.shape != dt_bias.shape:
        raise ValueError("The forget gate requires x [B,T,D], kernel [H,D], log_rate and dt_bias [H,1]")
    projected = jnp.einsum("btd,hd->bth", x.astype(jnp.float32), kernel.astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    rate = jnp.exp(log_rate[:, 0].astype(jnp.float32))
    return -rate * jax.nn.softplus(projected + dt_bias[:, 0].astype(jnp.float32))

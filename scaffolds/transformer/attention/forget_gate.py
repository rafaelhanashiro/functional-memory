"""Forgetting Transformer decay with Gated DeltaNet's per-head Mamba-2 gate.

Each attention weight from key i to query t is multiplied by prod_(j=i+1..t) alpha_j, with
log alpha_t = -exp(A_log) softplus(x_t w_alpha + dt_bias) per layer and head, computed from
the attention input x (after the block's pre-attention RMSNorm). The parameterization and the
initialization are Gated DeltaNet's: exp(A_log) ~ Uniform(0, 16), softplus(dt_bias)
log-uniform on [0.001, 0.1] (floored at 1e-4), and w_alpha a PyTorch-default Linear row,
Uniform(+-1/sqrt(hidden_size)). As in Gated DeltaNet, w_alpha decays and A_log, dt_bias do not.
"""

import jax
import jax.numpy as jnp

PARAMETERS = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")


def parameter_specs(*, hidden_size: int):
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    return (
        AttentionParameterSpec(name="forget_gate_kernel", argument="forget_gate_kernel",
                               initial=(0.0,) * hidden_size, initial_shape=(hidden_size,), sharing="per_head",
                               weight_decay=True, initializer="uniform_fan_in"),
        AttentionParameterSpec(name="forget_gate_log_rate", argument="forget_gate_log_rate", initial=(0.0,),
                               sharing="per_head", weight_decay=False, initializer="mamba2_log_rate"),
        AttentionParameterSpec(name="forget_gate_dt_bias", argument="forget_gate_dt_bias", initial=(0.0,),
                               sharing="per_head", weight_decay=False, initializer="mamba2_dt_bias"),
    )


def log_decay(x, kernel, log_rate, dt_bias):
    """FP32 log alpha [batch,time,heads] from x [B,T,D], kernel [H,D], log_rate and dt_bias [H,1]."""
    if x.ndim != 3 or kernel.shape != (log_rate.shape[0], x.shape[-1]) or log_rate.shape != dt_bias.shape:
        raise ValueError("The forget gate requires x [B,T,D], kernel [H,D], log_rate and dt_bias [H,1]")
    projected = jnp.einsum("btd,hd->bth", x.astype(jnp.float32), kernel.astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
    rate = jnp.exp(log_rate[:, 0].astype(jnp.float32))
    return -rate * jax.nn.softplus(projected + dt_bias[:, 0].astype(jnp.float32))

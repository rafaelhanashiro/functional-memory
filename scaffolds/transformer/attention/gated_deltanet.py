"""Gated DeltaNet (Yang, Kautz & Hatamizadeh, ICLR 2025) as the shared Transformer's token mixer.

Mirrors flash-linear-attention's ``GatedDeltaNet`` layer. Queries, keys, and values
are projected and each passes a causal depthwise short convolution and SiLU; queries
and keys are L2-normalized per head. Each head keeps a K x V state

    S_t = a_t S_{t-1} + beta_t k_t (v_t - a_t S_{t-1}^T k_t)^T,   o_t = S_t^T q_t / sqrt(d_K),

with ``beta_t = sigmoid(W_b x_t)`` and the Mamba-2 decay ``a_t = exp(g_t)``,
``g_t = -exp(A_log) softplus(W_a x_t + dt_bias)``. The read passes a gated RMSNorm,
``RMSNorm(o_t) * w * silu(W_g x_t)`` (one ``w`` shared by the heads), and the output
projection. There is no positional encoding; the short convolutions carry local order.
The ``fused`` backend runs FLA's chunked Triton kernels; ``blocked`` is a token-by-token
FP32 reference.

With ``erase_mode="independent"``, erasure instead uses a separate sigmoid
``lambda_t = sigmoid(W_lambda x_t)``: the rank-one update is
``k_t (beta_t v_t - lambda_t a_t S_{t-1}^T k_t)^T``. Its projection starts as an
independent copy of the write projection, recovering ordinary GDN initially when
the write activation retains its default sigmoid. With independent erasure,
``write_activation="tanh"`` gives signed writes ``beta_t = tanh(W_b x_t)``
without changing the projection weights or sigmoid erasure.

With ``polynomial_degree=m`` for m=2 or 3, memory is a polynomial function, transformed as
``F_t(q) = alpha_t F_{t-1}((I-lambda_t k_t k_t^T)q)`` before adding
``v_t sum_{r=0}^m beta_r <q,k_t>^r``. ``beta_1`` retains the configured
linear write activation; independent tanh projections produce the other coefficients
and start at zero, nesting the linear model. The final read has a single global
``1/sqrt(d_K)`` scale, before the same gated output RMSNorm.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from scaffolds.transformer.attention.backends.fla_gated_delta_rule import MAX_KEY_DIM
from scaffolds.transformer.training import telemetry

A_LOG_RANGE = (0.0, 16.0)
DT_RANGE = (0.001, 0.1)
DT_FLOOR = 1e-4
L2_NORM_EPS = 1e-6


def _l2_normalize(x: jax.Array) -> jax.Array:
    x32 = x.astype(jnp.float32)
    return (x32 * jax.lax.rsqrt(jnp.sum(jnp.square(x32), axis=-1, keepdims=True) + L2_NORM_EPS)).astype(x.dtype)


def gated_delta_rule_reference(q, k, v, g, beta, scale, erase=None):
    """FP32 recurrence with optional independent erase gate, all gates [B,T,H].

    ``q, k`` are normalized [B,T,H,K], ``v`` is [B,T,H,V]. Omitting ``erase``
    preserves the original tied-gate recurrence and its arithmetic.
    """
    q, k, v, g, beta = (x.astype(jnp.float32) for x in (q, k, v, g, beta))
    tied = erase is None
    erase = beta if tied else erase.astype(jnp.float32)

    def step(state, inputs):
        q_t, k_t, v_t, g_t, beta_t, erase_t = inputs
        state = state * jnp.exp(g_t)[..., None, None]
        prediction = jnp.einsum("bhk,bhkv->bhv", k_t, state)
        update = (beta_t[..., None] * (v_t - prediction) if tied
                  else beta_t[..., None] * v_t - erase_t[..., None] * prediction)
        state = state + jnp.einsum("bhk,bhv->bhkv", k_t, update)
        return state, jnp.einsum("bhk,bhkv->bhv", q_t * scale, state)

    batch, _, heads, key_dim = q.shape
    initial = jnp.zeros((batch, heads, key_dim, v.shape[-1]), jnp.float32)
    _, reads = jax.lax.scan(step, initial, tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta, erase)))
    return jnp.moveaxis(reads, 0, 1)


class ShortConvolution(nnx.Module):
    """Causal depthwise convolution over time, ``y_t = sum_i w_i x_(t-K+1+i)``, then SiLU."""

    def __init__(self, channels: int, kernel_size: int, *, rngs: nnx.Rngs):
        # PyTorch's Conv1d default for a depthwise kernel: Uniform(+-1/sqrt(kernel_size)).
        bound = 1.0 / math.sqrt(kernel_size)
        self.kernel_size = kernel_size
        self.weight = nnx.Param(jax.random.uniform(rngs.params(), (kernel_size, channels), jnp.float32,
                                                   -bound, bound))

    def __call__(self, x: jax.Array) -> jax.Array:
        padded = jnp.pad(x, ((0, 0), (self.kernel_size - 1, 0), (0, 0)))
        length = x.shape[1]
        weight = self.weight[...].astype(x.dtype)
        y = sum(padded[:, i:i + length] * weight[i] for i in range(self.kernel_size))
        return nnx.silu(y)


class GatedDeltaNet(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_size: int = 4,
        norm_eps: float = 1e-6,
        backend: str = "blocked",
        *,
        erase_mode: str = "tied",
        write_activation: str = "sigmoid",
        polynomial_degree: int = 1,
        rngs: nnx.Rngs,
    ):
        if min(hidden_size, num_heads, key_head_dim, value_head_dim, conv_size) <= 0:
            raise ValueError("Gated DeltaNet dimensions and conv_size must be positive")
        if backend not in ("blocked", "fused"):
            raise ValueError(f"Unsupported Gated DeltaNet backend: {backend}")
        if backend == "fused" and polynomial_degree == 1 and key_head_dim > MAX_KEY_DIM:
            raise ValueError(f"The fused Gated DeltaNet backend supports key_head_dim up to {MAX_KEY_DIM}")
        if erase_mode not in ("tied", "independent"):
            raise ValueError(f"Unsupported Gated DeltaNet erase mode: {erase_mode}")
        if write_activation not in ("sigmoid", "tanh"):
            raise ValueError(f"Unsupported Gated DeltaNet write activation: {write_activation}")
        if write_activation == "tanh" and erase_mode != "independent":
            raise ValueError("Gated DeltaNet tanh write activation requires independent erasure")
        if type(polynomial_degree) is not int or polynomial_degree not in (1, 2, 3):
            raise ValueError("Gated DeltaNet polynomial_degree must be 1, 2, or 3")
        if polynomial_degree > 1 and erase_mode != "independent":
            raise ValueError("Polynomial Gated DeltaNet requires independent erasure")
        self.polynomial_degree = polynomial_degree
        self.num_heads, self.key_head_dim, self.value_head_dim = num_heads, key_head_dim, value_head_dim
        self.key_dim, self.value_dim = num_heads * key_head_dim, num_heads * value_head_dim
        self.norm_eps = norm_eps
        self.backend = backend
        self.erase_mode = erase_mode
        self.write_activation = write_activation
        self.dtype = jnp.float32
        self.q_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(hidden_size, self.key_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(hidden_size, self.value_dim, use_bias=False, rngs=rngs)
        self.a_proj = nnx.Linear(hidden_size, num_heads, use_bias=False, rngs=rngs)
        self.b_proj = nnx.Linear(hidden_size, num_heads, use_bias=False, rngs=rngs)
        self.q_conv = ShortConvolution(self.key_dim, conv_size, rngs=rngs)
        self.k_conv = ShortConvolution(self.key_dim, conv_size, rngs=rngs)
        self.v_conv = ShortConvolution(self.value_dim, conv_size, rngs=rngs)
        # Mamba-2 decay parameters: exp(A_log) ~ Uniform(0, 16); softplus(dt_bias) log-uniform
        # on [0.001, 0.1] (floored at 1e-4). Neither decays, as in FLA and the paper.
        a = jax.random.uniform(rngs.params(), (num_heads,), jnp.float32, *A_LOG_RANGE)
        self.A_log = nnx.Param(jnp.log(a), weight_decay=False)
        log_dt = jax.random.uniform(rngs.params(), (num_heads,), jnp.float32,
                                    math.log(DT_RANGE[0]), math.log(DT_RANGE[1]))
        dt = jnp.maximum(jnp.exp(log_dt), DT_FLOOR)
        self.dt_bias = nnx.Param(dt + jnp.log(-jnp.expm1(-dt)), weight_decay=False)
        self.g_proj = nnx.Linear(hidden_size, self.value_dim, use_bias=False, rngs=rngs)
        self.o_norm_scale = nnx.Param(jnp.ones((value_head_dim,), jnp.float32), weight_decay=False)
        self.o_proj = nnx.Linear(self.value_dim, hidden_size, use_bias=False, rngs=rngs)
        if erase_mode == "independent":
            # Clone the graph/Param, not the RNG stream: every existing parameter
            # in this layer and subsequent layers retains the tied model's draw.
            self.lambda_proj = nnx.clone(self.b_proj)
            # nnx.clone shares array storage. Donated training state requires
            # distinct buffers for these independently updated parameters.
            self.lambda_proj.kernel[...] = self.b_proj.kernel[...].copy()
        if polynomial_degree > 1:
            # Clone only the module structure; no random draw or change to the
            # initialization of this layer or any subsequent layer. Each new
            # zero-initialized weight needs its own donated training buffer.
            for degree in (0, *range(2, polynomial_degree + 1)):
                projection = nnx.clone(self.b_proj)
                projection.kernel[...] = jnp.zeros_like(self.b_proj.kernel[...])
                setattr(self, f"b{degree}_proj", projection)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
        return_diagnostics: bool = False,
        coefficients: jax.Array | None = None,
        attention_parameters: dict[str, jax.Array] | None = None,
    ):
        if coefficients is not None or attention_parameters:
            raise ValueError("Gated DeltaNet does not accept attention parameters")
        batch, length, _ = x.shape
        heads = self.num_heads
        q = self.q_conv(self.q_proj(x).astype(self.dtype)).reshape(batch, length, heads, self.key_head_dim)
        k = self.k_conv(self.k_proj(x).astype(self.dtype)).reshape(batch, length, heads, self.key_head_dim)
        v = self.v_conv(self.v_proj(x).astype(self.dtype)).reshape(batch, length, heads, self.value_head_dim)
        q, k = _l2_normalize(q), _l2_normalize(k)
        write_activation = jnp.tanh if self.write_activation == "tanh" else jax.nn.sigmoid
        beta = write_activation(self.b_proj(x).astype(jnp.float32)).astype(self.dtype)
        erase = (jax.nn.sigmoid(self.lambda_proj(x).astype(jnp.float32)).astype(self.dtype)
                 if self.erase_mode == "independent" else None)
        log_decay = -jnp.exp(self.A_log[...]) * jax.nn.softplus(
            self.a_proj(x).astype(jnp.float32) + self.dt_bias[...])
        scale = self.key_head_dim ** -0.5
        write_coefficients = None
        if self.polynomial_degree > 1:
            write_coefficients = jnp.stack(tuple(
                beta if degree == 1 else jnp.tanh(
                    getattr(self, f"b{degree}_proj")(x).astype(jnp.float32)).astype(self.dtype)
                for degree in range(self.polynomial_degree + 1)
            ), axis=-1)
            if self.backend == "fused":
                from scaffolds.transformer.attention.backends.polynomial_gdn import (
                    polynomial_gated_delta_rule,
                )

                read = polynomial_gated_delta_rule(q, k, v, log_decay, write_coefficients, erase, scale)
            else:
                from scaffolds.transformer.attention.polynomial_gdn_reference import (
                    polynomial_gated_delta_rule_reference,
                )

                read = polynomial_gated_delta_rule_reference(
                    q, k, v, log_decay, write_coefficients, erase, scale).astype(self.dtype)
        elif self.backend == "fused":
            if erase is None:
                from scaffolds.transformer.attention.backends.fla_gated_delta_rule import fla_gated_delta_rule

                read = fla_gated_delta_rule(q, k, v, log_decay, beta, scale)
            else:
                from scaffolds.transformer.attention.backends.fla_gated_delta_rule import fla_independent_gated_delta_rule

                read = fla_independent_gated_delta_rule(q, k, v, log_decay, beta, erase, scale)
        else:
            read = gated_delta_rule_reference(q, k, v, log_decay, beta, scale, erase=erase).astype(self.dtype)
        read32 = read.astype(jnp.float32)
        telemetry.record_attention_output(read32, head_rms_norm_eps=self.norm_eps)
        if telemetry.active():
            gates = {"beta": beta, "lambda": beta if erase is None else erase}
            for name, value in gates.items():
                value = value.astype(jnp.float32)
                telemetry.record(f"{name}_mean", jnp.mean(value), "mean")
                telemetry.record(f"{name}_min", jnp.min(value), "min")
                telemetry.record(f"{name}_max", jnp.max(value), "max")
            if write_coefficients is not None:
                for degree in range(self.polynomial_degree + 1):
                    value = write_coefficients[..., degree].astype(jnp.float32)
                    telemetry.record(f"beta_degree_{degree}_mean", jnp.mean(value), "mean")
                    telemetry.record(f"beta_degree_{degree}_min", jnp.min(value), "min")
                    telemetry.record(f"beta_degree_{degree}_max", jnp.max(value), "max")
                    telemetry.record(f"beta_degree_{degree}_rms", jnp.mean(jnp.square(value)), "rms")
            gap = (gates["lambda"].astype(jnp.float32) - beta.astype(jnp.float32))
            telemetry.record("lambda_minus_beta_rms", jnp.mean(jnp.square(gap)), "rms")
        gate = self.g_proj(x).astype(jnp.float32).reshape(batch, length, heads, self.value_head_dim)
        normed = read32 * jax.lax.rsqrt(jnp.mean(jnp.square(read32), axis=-1, keepdims=True) + self.norm_eps)
        y = (normed * self.o_norm_scale[...] * nnx.silu(gate)).astype(self.dtype)
        output = self.o_proj(y.reshape(batch, length, self.value_dim))
        if not return_diagnostics:
            return output
        diagnostics = {
            "log_decay_min": jnp.min(log_decay),
            "log_decay_max": jnp.max(log_decay),
            "beta_min": jnp.min(beta.astype(jnp.float32)),
            "beta_max": jnp.max(beta.astype(jnp.float32)),
            "read_abs_max": jnp.max(jnp.abs(read32)),
            "invalid_projection_count": sum(jnp.sum(~jnp.isfinite(value), dtype=jnp.int32)
                                            for value in (q, k, v, beta, gate)),
            "invalid_gate_count": jnp.sum(~jnp.isfinite(log_decay) | (log_decay > 0), dtype=jnp.int32),
            "invalid_read_count": jnp.sum(~jnp.isfinite(read32), dtype=jnp.int32),
            "invalid_output_count": jnp.sum(~jnp.isfinite(output), dtype=jnp.int32),
        }
        if write_coefficients is not None:
            coefficients32 = write_coefficients.astype(jnp.float32)
            diagnostics["invalid_coefficient_count"] = jnp.sum(
                ~jnp.isfinite(coefficients32) | (jnp.abs(coefficients32) > 1), dtype=jnp.int32)
            for degree in range(self.polynomial_degree + 1):
                value = coefficients32[..., degree]
                diagnostics.update({
                    f"beta_degree_{degree}_mean": jnp.mean(value),
                    f"beta_degree_{degree}_min": jnp.min(value),
                    f"beta_degree_{degree}_max": jnp.max(value),
                    f"beta_degree_{degree}_rms": jnp.sqrt(jnp.mean(jnp.square(value))),
                })
        if self.write_activation == "tanh":
            beta32 = beta.astype(jnp.float32)
            diagnostics["invalid_write_count"] = jnp.sum(
                ~jnp.isfinite(beta32) | (jnp.abs(beta32) > 1), dtype=jnp.int32)
        if erase is not None:
            erase32, beta32 = erase.astype(jnp.float32), beta.astype(jnp.float32)
            diagnostics.update({
                "lambda_min": jnp.min(erase32),
                "lambda_max": jnp.max(erase32),
                "lambda_minus_beta_rms": jnp.sqrt(jnp.mean(jnp.square(erase32 - beta32))),
                "invalid_erase_count": jnp.sum(~jnp.isfinite(erase32) | (erase32 < 0) | (erase32 > 1),
                                              dtype=jnp.int32),
            })
        return output, diagnostics

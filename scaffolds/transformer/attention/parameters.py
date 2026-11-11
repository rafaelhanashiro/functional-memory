"""Model-level attention parameter ownership, independent of layer execution."""

from dataclasses import dataclass, replace
import math
from typing import Literal
import zlib

import jax
import jax.numpy as jnp
from flax import nnx

# fold_in tag of the mechanism's random stream, far above the model's draw counts.
MECHANISM_STREAM = 2**31 - 2


@dataclass(frozen=True)
class AttentionParameterSpec:
    name: str
    argument: str
    initial: tuple[float, ...]
    sharing: Literal["global", "per_head", "per_layer"]
    weight_decay: bool = False
    initial_shape: tuple[int, ...] | None = None
    # Independent N(0, initial_std^2) noise added to every stored entry, drawn from the
    # mechanism stream so the backbone's initialization is unchanged.
    initial_std: float = 0.0
    # A named random draw (RANDOM_INITIALIZERS) from the same mechanism stream; it replaces
    # ``initial``, which then only fixes the shape.
    initializer: str | None = None
    # Draw the shared control's array once and repeat it on this axis at initialization.
    # The stored copies are independent trainable parameters afterwards.
    repeat_initial_axis: int | None = None


def _mamba2_dt_bias(key, shape):
    from scaffolds.transformer.attention.gated_deltanet import DT_FLOOR, DT_RANGE

    dt = jnp.maximum(jnp.exp(jax.random.uniform(key, shape, jnp.float32, math.log(DT_RANGE[0]),
                                                math.log(DT_RANGE[1]))), DT_FLOOR)
    return dt + jnp.log(-jnp.expm1(-dt))      # inverse softplus


def _mamba2_sigmoid_bias(key, shape):
    """Reuse the legacy dt draw, matching sigmoid(b_new) = softplus(b_old) at zero input."""
    dt = jax.nn.softplus(_mamba2_dt_bias(key, shape))
    return jnp.log(dt) - jnp.log1p(-dt)


def _mamba2_log_rate(key, shape):
    from scaffolds.transformer.attention.gated_deltanet import A_LOG_RANGE

    return jnp.log(jax.random.uniform(key, shape, jnp.float32, *A_LOG_RANGE))


def _uniform_fan_in(key, shape):
    bound = 1.0 / math.sqrt(shape[-1])
    return jax.random.uniform(key, shape, jnp.float32, -bound, bound)


def _alibi_log_slope(key, shape):
    """Deterministic: log 2^(-8 h / H) for head h of H, the same in every layer; [L, H, 1].

    The same FP32 values as the constant forget gate's fixed rate (forget_gate.alibi_log_rate)."""
    from scaffolds.transformer.attention.forget_gate import alibi_log_rate

    if len(shape) != 3 or shape[-1] != 1:
        raise ValueError("ALiBi slopes initialize a per-head [layers, heads, 1] parameter")
    return jnp.broadcast_to(alibi_log_rate(shape[1])[None], shape)


# Gated DeltaNet's decay-gate draws: a PyTorch-default Linear row over the last axis,
# exp(A_log) ~ Uniform(0, 16), and softplus(dt_bias) log-uniform on [0.001, 0.1].
RANDOM_INITIALIZERS = {"uniform_fan_in": _uniform_fan_in, "mamba2_log_rate": _mamba2_log_rate,
                       "mamba2_dt_bias": _mamba2_dt_bias, "mamba2_sigmoid_bias": _mamba2_sigmoid_bias, "alibi_log_slope": _alibi_log_slope}


# The input-independent coefficient arrays that ``attention_coefficient_scale`` stores scaled.
SCALED_COEFFICIENTS = ("polynomial_coefficients", "query_polynomial_bias")


def factorial_scale(values: jax.Array, orders: tuple[int, ...] | None = None) -> jax.Array:
    """Divide order r (the last axis) by r!: the coefficients a stored c~_r represents."""
    factorials = jnp.asarray([math.factorial(order) for order in
                             (range(values.shape[-1]) if orders is None else orders)], jnp.float32)
    return values.astype(jnp.float32) / factorials


def expand_polynomial_coefficients(values: jax.Array, degree: int, orders: tuple[int, ...] | None) -> jax.Array:
    """Expand compact stored coefficients; omitted powers are constants with zero gradient."""
    if orders is None:
        return values
    if values.shape[-1] != len(orders):
        raise ValueError("Stored coefficient count must match attention_coefficient_orders")
    expanded = jnp.zeros((*values.shape[:-1], degree + 1), dtype=values.dtype)
    return expanded.at[..., jnp.asarray(orders)].set(values)


def attention_parameter_specs(config) -> tuple[AttentionParameterSpec, ...]:
    """Ask the configured mechanism, then the score option, which trainable arrays they need."""
    if (getattr(config, "attention_coefficients", "fixed") == "learned"
            and getattr(config, "attention_kernel_basis", "polynomial") == "bspline"):
        # B-spline coefficients (attention/spline.py): not monomial orders, never factorial-scaled.
        from scaffolds.transformer.attention.spline import parameter_specs as spline_specs

        specs = spline_specs(
            hidden_size=config.hidden_size, degree=config.attention_degree, knots=config.attention_spline_knots,
            conditioning=config.attention_coefficient_conditioning,
            amplitude_initial=getattr(config, "attention_coefficient_amplitude_initial", 1.0),
            projection_initial_std=getattr(config, "attention_coefficient_projection_initial_std", 0.0),
            kernel_weight_decay=getattr(config, "attention_coefficient_kernel_weight_decay", False))
    elif getattr(config, "attention_coefficients", "fixed") == "learned":
        parameterization = getattr(config, "attention_coefficient_parameterization", "direct")
        conditioning = getattr(config, "attention_coefficient_conditioning", "static")
        if conditioning in ("query", "query_key"):
            from scaffolds.transformer.attention.input_dependent import parameter_specs

            specs = parameter_specs(
                hidden_size=config.hidden_size, degree=config.attention_degree,
                activation=config.attention_coefficient_activation,
                amplitude_initial=getattr(config, "attention_coefficient_amplitude_initial", 1.0),
                projection_initial_std=getattr(config, "attention_coefficient_projection_initial_std", 0.0),
                tanh_bias=getattr(config, "attention_coefficient_tanh_bias", False),
                kernel_weight_decay=getattr(config, "attention_coefficient_kernel_weight_decay", False),
                conditioning=conditioning,
                initialization=getattr(config, "attention_coefficient_initialization", "taylor"),
                parameterization=parameterization,
                readout=getattr(config, "attention_coefficient_readout", "diagonal"),
                feature_count=getattr(config, "attention_coefficient_feature_count", None),
                gate=getattr(config, "attention_coefficient_gate", "none"))
        elif parameterization == "sum_of_squares":
            from scaffolds.transformer.attention.nonnegative_polynomial import parameter_specs

            specs = parameter_specs(degree=config.attention_degree, sharing=config.attention_coefficient_sharing)
        else:
            initialization = getattr(config, "attention_coefficient_initialization", "taylor")
            if parameterization == "rational":
                from scaffolds.transformer.attention.rational_polynomial import parameter_specs

                specs = parameter_specs(
                    degree=config.attention_degree, sharing=config.attention_coefficient_sharing,
                    initialization=initialization,
                    denominator_degree=getattr(config, "attention_denominator_degree", None),
                    denominator_parameterization=getattr(config, "attention_denominator_parameterization", "direct"),
                    denominator_constant=getattr(config, "attention_denominator_constant", "fixed"))
            else:
                from scaffolds.transformer.attention.unnormalized import parameter_specs

                specs = parameter_specs(degree=config.attention_degree, sharing=config.attention_coefficient_sharing,
                                        initialization=initialization)
        if getattr(config, "attention_coefficient_scale", "none") == "factorial":
            # Store r! times each starting coefficient, so the represented start is unchanged.
            specs = tuple(replace(spec, initial=tuple(value * math.factorial(order)
                                                      for order, value in enumerate(spec.initial)))
                          if spec.name in SCALED_COEFFICIENTS else spec for spec in specs)
        orders = getattr(config, "attention_coefficient_orders", None)
        if orders is not None:
            # Select after factorial storage scaling, so indices still denote actual powers.
            selected = []
            for spec in specs:
                if spec.name == "query_polynomial_kernel":
                    width, count = spec.initial_shape
                    spec = replace(spec, initial=tuple(spec.initial[row * count + order]
                                                       for row in range(width) for order in orders),
                                   initial_shape=(width, len(orders)))
                elif spec.name in ("polynomial_coefficients", "query_polynomial_bias",
                                    "query_polynomial_amplitude", "query_polynomial_tanh_bias"):
                    spec = replace(spec, initial=tuple(spec.initial[order] for order in orders))
                selected.append(spec)
            specs = tuple(selected)
    elif config.quadratic_mode == "fixed":
        specs = ()
    else:
        from scaffolds.transformer.attention.learned_quadratic import parameter_specs

        specs = parameter_specs(mode=config.quadratic_mode, sharing=config.quadratic_sharing)
    if getattr(config, "qk_norm", False):
        from scaffolds.transformer.attention.qk_norm import (
            initial_scale, parameter_specs as qk_norm_specs, rms_parameter_specs, rms_head_scale_specs,
            token_scale_specs,
        )

        specs += (rms_parameter_specs(qk_head_dim=config.qk_head_dim)
                  if config.qk_norm_type == "rms_norm" else
                  qk_norm_specs(initial_scale=initial_scale(config), sharing=config.qk_norm_sharing))
        if config.qk_norm_head_scale:
            specs += rms_head_scale_specs()
        if getattr(config, "qk_norm_token_scale", "none") == "cosine":
            specs += token_scale_specs(
                hidden_size=config.hidden_size,
                kernel_weight_decay=getattr(config, "qk_norm_token_scale_kernel_weight_decay", False),
            )
    if getattr(config, "attention_forget_gate", False):
        from scaffolds.transformer.attention.forget_gate import parameter_specs as forget_gate_specs

        specs += forget_gate_specs(hidden_size=config.hidden_size,
                                   form=getattr(config, "attention_forget_gate_form", "input"),
                                   conditioning=getattr(config, "attention_forget_gate_conditioning", "query"),
                                   rate_activation=getattr(config, "attention_forget_gate_rate_activation", "softplus"),
                                   constant_rate=getattr(config, "attention_forget_gate_constant_rate", "learned"),
                                   order_count=(config.attention_degree + 1
                                                if getattr(config, "attention_forget_gate_per_order", False) else None))
    if getattr(config, "positional_encoding", "rope") == "path":
        from scaffolds.transformer.attention.path import parameter_specs as path_specs

        specs += path_specs(hidden_size=config.hidden_size, qk_head_dim=config.qk_head_dim,
                            rank=config.path_w_rank, conv_size=config.path_conv_size)
    if getattr(config, "attention_output_scaling", "none") == "head_rms_norm":
        from scaffolds.transformer.attention.output_scaling import parameter_specs as output_specs

        specs += output_specs(value_head_dim=config.value_head_dim)
    return specs


def mechanism_key(base_key: jax.Array) -> jax.Array:
    """The mechanism's random stream: a function of the seed that consumes none of the model's draws."""
    return jax.random.fold_in(base_key, MECHANISM_STREAM)


def initial_attention_parameter_values(specs, *, num_layers: int, num_heads: int,
                                       key: jax.Array | None = None) -> dict[str, jax.Array]:
    """Each declared array's starting value; random entries come from ``key`` by parameter name."""
    values = {}
    for spec in specs:
        if spec.sharing not in ("global", "per_head", "per_layer"):
            raise ValueError(f"Unknown attention parameter sharing: {spec.sharing}")
        initial = jnp.asarray(spec.initial, dtype=jnp.float32)
        if spec.initial_shape is not None:
            initial = initial.reshape(spec.initial_shape)
        shape = (initial.shape if spec.sharing == "global" else (num_layers, *initial.shape)
                 if spec.sharing == "per_layer" else (num_layers, num_heads, *initial.shape))
        value = jnp.broadcast_to(initial, shape)
        if spec.initial_std or spec.initializer:
            if key is None:
                raise ValueError(f"{spec.name} has a random initialization and needs the mechanism key")
            name_key = jax.random.fold_in(key, zlib.crc32(spec.name.encode()) & 0x7FFFFFFF)
            if spec.initializer:
                if spec.initial_std or spec.initializer not in RANDOM_INITIALIZERS:
                    raise ValueError(f"{spec.name} needs one known random initialization")
                if spec.repeat_initial_axis is None:
                    value = RANDOM_INITIALIZERS[spec.initializer](name_key, shape)
                else:
                    axis = spec.repeat_initial_axis % len(shape)
                    draw = RANDOM_INITIALIZERS[spec.initializer](name_key, shape[:axis] + shape[axis + 1:])
                    value = jnp.broadcast_to(jnp.expand_dims(draw, axis), shape)
            else:
                value = value + spec.initial_std * jax.random.normal(name_key, shape, jnp.float32)
        values[spec.name] = value
    return values


def initialize_attention_parameters(owner, specs, *, num_layers: int, num_heads: int,
                                    key: jax.Array | None = None) -> None:
    """Register once on the model, keeping existing names and initialization order.

    No model RNG is consumed; random entries use the mechanism ``key``. Globally shared
    parameters are stored once; independent layer/head parameters remain one compact
    array in the checkpoint tree.
    """
    values = initial_attention_parameter_values(specs, num_layers=num_layers, num_heads=num_heads, key=key)
    for spec in specs:
        setattr(owner, spec.name, nnx.Param(values[spec.name]))


def attention_parameter_count(config) -> int:
    """Count declared trainable values without allocating any device arrays."""
    multiplier = {"global": 1, "per_layer": config.num_layers, "per_head": config.num_layers * config.num_heads}
    return sum(len(spec.initial) * multiplier[spec.sharing] for spec in attention_parameter_specs(config))


def parameters_for_layer(owner, specs, layer_index: int) -> dict:
    """Return device-array arguments; shared arrays are never copied per layer."""
    return {
        spec.argument: (getattr(owner, spec.name)[...] if spec.sharing == "global"
                        else getattr(owner, spec.name)[layer_index])
        for spec in specs
    }

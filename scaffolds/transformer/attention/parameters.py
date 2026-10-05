"""Model-level attention parameter ownership, independent of layer execution."""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from flax import nnx


@dataclass(frozen=True)
class AttentionParameterSpec:
    name: str
    argument: str
    initial: tuple[float, ...]
    sharing: Literal["global", "per_head"]
    weight_decay: bool = False


def attention_parameter_specs(config) -> tuple[AttentionParameterSpec, ...]:
    """Ask the configured mechanism, then the score option, which trainable arrays they need."""
    if getattr(config, "attention_coefficients", "fixed") == "learned":
        parameterization = getattr(config, "attention_coefficient_parameterization", "direct")
        if parameterization == "sum_of_squares":
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
                    denominator_parameterization=getattr(config, "attention_denominator_parameterization", "direct"))
            else:
                from scaffolds.transformer.attention.unnormalized import parameter_specs

                specs = parameter_specs(degree=config.attention_degree, sharing=config.attention_coefficient_sharing,
                                        initialization=initialization)
    elif config.quadratic_mode == "fixed":
        specs = ()
    else:
        from scaffolds.transformer.attention.learned_quadratic import parameter_specs

        specs = parameter_specs(mode=config.quadratic_mode, sharing=config.quadratic_sharing)
    if getattr(config, "qk_norm", False):
        from scaffolds.transformer.attention.qk_norm import initial_scale, parameter_specs as qk_norm_specs

        specs += qk_norm_specs(initial_scale=initial_scale(config), sharing=config.qk_norm_sharing)
    if getattr(config, "attention_output_scaling", "none") == "head_rms_norm":
        from scaffolds.transformer.attention.output_scaling import parameter_specs as output_specs

        specs += output_specs(value_head_dim=config.value_head_dim)
    return specs


def initialize_attention_parameters(owner, specs, *, num_layers: int, num_heads: int) -> None:
    """Register once on the model, keeping existing names and initialization order.

    No RNG is consumed. Globally shared parameters are stored once; independent
    layer/head parameters remain one compact array in the checkpoint tree.
    """
    for spec in specs:
        if spec.sharing not in ("global", "per_head"):
            raise ValueError(f"Unknown attention parameter sharing: {spec.sharing}")
        initial = jnp.asarray(spec.initial, dtype=jnp.float32)
        shape = initial.shape if spec.sharing == "global" else (num_layers, num_heads, *initial.shape)
        setattr(owner, spec.name, nnx.Param(jnp.broadcast_to(initial, shape)))


def attention_parameter_count(config) -> int:
    """Count declared trainable values without allocating any device arrays."""
    return sum(
        len(spec.initial) * (1 if spec.sharing == "global" else config.num_layers * config.num_heads)
        for spec in attention_parameter_specs(config)
    )


def parameters_for_layer(owner, specs, layer_index: int) -> dict:
    """Return device-array arguments; shared arrays are never copied per layer."""
    return {
        spec.argument: (getattr(owner, spec.name)[...] if spec.sharing == "global"
                        else getattr(owner, spec.name)[layer_index])
        for spec in specs
    }

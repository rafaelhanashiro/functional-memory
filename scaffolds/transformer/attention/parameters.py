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
    """Ask the configured mechanism which trainable arrays it needs."""
    if config.quadratic_mode == "fixed":
        return ()
    from scaffolds.transformer.attention.learned_quadratic import parameter_specs

    return parameter_specs(mode=config.quadratic_mode, sharing=config.quadratic_sharing)


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

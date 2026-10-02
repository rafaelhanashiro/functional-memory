"""Reference weight initializations: each architecture's published training recipe.

The scaffold's Flax defaults draw every Linear kernel from a truncated normal with
variance 1/fan_in and the embedding table with variance 1/features. A reference
scheme re-draws the weights after construction from one key derived from the
run's seed, keeping each parameter's metadata (for example weight-decay
exclusions). Normalization scales stay one, biases zero, and mechanism-owned
attention parameters (QK-norm scales, learned coefficients) keep their declared
initial values.

- ``transformer_reference`` (GPT-2/Megatron): every Linear and embedding weight
  N(0, 0.02); the residual output projections (attention ``o_proj``, MLP
  ``down_proj``) N(0, 0.02 / sqrt(2 * num_layers)).
- ``gla_reference`` (berlino/gated_linear_attention): every Linear and embedding
  weight N(0, 0.02), then the query, key, and both low-rank gate projections
  Xavier-uniform with gain 2^-2.5.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

INITIALIZATIONS = ("flax_default", "transformer_reference", "gla_reference")
# Each reference scheme belongs to one attention mechanism.
MECHANISMS = {"transformer_reference": "dot_product", "gla_reference": "gla"}
_GLA_XAVIER = ("q_proj", "k_proj", "gate_low_rank", "gate_proj")


def _normal(std: float):
    return jax.nn.initializers.normal(stddev=std)


def _xavier_uniform(gain: float):
    return jax.nn.initializers.variance_scaling(gain ** 2, "fan_avg", "uniform")


def _weight_initializer(scheme: str, path: tuple[str, ...], num_layers: int):
    parent, name = (path[-2] if len(path) > 1 else ""), path[-1]
    if scheme == "transformer_reference":
        residual = (parent, name) in (("attention", "o_proj"), ("feed_forward", "down_proj"))
        return _normal(0.02 / math.sqrt(2 * num_layers) if residual else 0.02)
    if scheme == "gla_reference":
        return _xavier_uniform(2 ** -2.5) if parent == "attention" and name in _GLA_XAVIER else _normal(0.02)
    raise ValueError(f"Unknown initialization: {scheme}")


def apply_initialization(model: nnx.Module, scheme: str, key: jax.Array, *, num_layers: int) -> None:
    """Re-draw ``model``'s Linear and embedding weights under ``scheme`` (no-op for Flax defaults)."""
    if scheme not in INITIALIZATIONS:
        raise ValueError(f"Unknown initialization: {scheme}")
    if scheme == "flax_default":
        return
    index = 0
    for path, node in nnx.iter_graph(model):
        if isinstance(node, (nnx.Linear, nnx.Embed)):
            path = tuple(str(part) for part in path)
            weight = node.kernel if isinstance(node, nnx.Linear) else node.embedding
            initializer = _weight_initializer(scheme, path, num_layers)
            weight[...] = initializer(jax.random.fold_in(key, index), weight.shape, jnp.float32)
            if isinstance(node, nnx.Linear) and node.bias is not None:
                node.bias[...] = jnp.zeros(node.bias.shape, jnp.float32)
            index += 1

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
- ``gated_deltanet_reference`` (NVlabs/GatedDeltaNet ``lit_gpt`` with ``mamba_init``):
  embeddings and the (there tied) output head N(0, 0.02); every Linear weight
  PyTorch's default ``kaiming_uniform_(a=sqrt(5))``, Uniform(+-1/sqrt(fan_in)); the MLP
  down projection the same divided by sqrt(2 * num_layers). Its name-matched rescaling
  does not reach the recurrent ``o_proj``. Short convolutions keep PyTorch's default
  draw and ``A_log``/``dt_bias`` their Mamba-2 draws, both made at construction.
  An independent erase projection copies the initialized beta projection without
  changing any of the original model's draws.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

INITIALIZATIONS = ("flax_default", "transformer_reference", "gla_reference", "gated_deltanet_reference")
# Each reference scheme belongs to one attention mechanism.
MECHANISMS = {"transformer_reference": "dot_product", "gla_reference": "gla",
              "gated_deltanet_reference": "gated_deltanet"}
# The canonical initialization of each attention mechanism: its reference scheme.
REFERENCE = {mechanism: scheme for scheme, mechanism in MECHANISMS.items()}
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
    if scheme == "gated_deltanet_reference":
        if name in ("token_embedding", "lm_head"):
            return _normal(0.02)
        torch_default = jax.nn.initializers.variance_scaling(1.0 / 3.0, "fan_in", "uniform")
        if (parent, name) == ("feed_forward", "down_proj"):
            scale = math.sqrt(2 * num_layers)
            return lambda key, shape, dtype: torch_default(key, shape, dtype) / scale
        return torch_default
    raise ValueError(f"Unknown initialization: {scheme}")


def apply_initialization(model: nnx.Module, scheme: str, key: jax.Array, *, num_layers: int) -> None:
    """Re-draw ``model``'s Linear and embedding weights under ``scheme`` (no-op for Flax defaults)."""
    if scheme not in INITIALIZATIONS:
        raise ValueError(f"Unknown initialization: {scheme}")
    if scheme == "flax_default":
        return
    index = 0
    independent_gdn = []
    if scheme == "gated_deltanet_reference":
        from scaffolds.transformer.attention.gated_deltanet import GatedDeltaNet

    for path, node in nnx.iter_graph(model):
        if scheme == "gated_deltanet_reference":
            if isinstance(node, GatedDeltaNet) and node.erase_mode == "independent":
                independent_gdn.append(node)
        if isinstance(node, (nnx.Linear, nnx.Embed)):
            path = tuple(str(part) for part in path)
            if scheme == "gated_deltanet_reference" and path[-1] in ("lambda_proj", "b0_proj", "b2_proj", "b3_proj", "b4_proj"):
                # Do not consume an initialization index: adding a gate must keep
                # all of the original GDN model's weights bit-identical.
                continue
            weight = node.kernel if isinstance(node, nnx.Linear) else node.embedding
            initializer = _weight_initializer(scheme, path, num_layers)
            weight[...] = initializer(jax.random.fold_in(key, index), weight.shape, jnp.float32)
            if isinstance(node, nnx.Linear) and node.bias is not None:
                node.bias[...] = jnp.zeros(node.bias.shape, jnp.float32)
            index += 1
    for attention in independent_gdn:
        # Match the values without aliasing a buffer donated by the train step.
        attention.lambda_proj.kernel[...] = attention.b_proj.kernel[...].copy()
        if attention.polynomial_degree > 1:
            # Zero tangent gates nest the linear model while retaining nonzero
            # derivatives through tanh. Allocate each projection independently.
            for degree in (0, *range(2, attention.polynomial_degree + 1)):
                getattr(attention, f"b{degree}_proj").kernel[...] = jnp.zeros_like(attention.b_proj.kernel[...])

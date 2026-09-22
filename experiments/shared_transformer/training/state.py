import jax
import jax.numpy as jnp
from flax import nnx

from experiments.shared_transformer.configs import Config
from experiments.shared_transformer.models import build_model
from experiments.shared_transformer.attention.backends.selection import resolve_config_backend
from experiments.shared_transformer.training.optim import make_optimizer


class TrainState(nnx.TrainState):
    """Parameters, optimizer moments, and completed update count (the RNG/cursor clock)."""


def create_state(config: Config, rng: jax.Array) -> TrainState:
    resolve_config_backend(config)
    if config.data.source == "synthetic" and config.data.synthetic_vocab_size > config.model.vocab_size:
        raise ValueError("data.synthetic_vocab_size must be <= model.vocab_size")
    model = build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[config.train.dtype]
    # The graph walker also reaches blocks held inside NNX lists.
    for _, module in nnx.iter_graph(model):
        if isinstance(module, nnx.Module) and hasattr(module, "dtype"):
            module.dtype = dtype
    graphdef, params = nnx.split(model, nnx.Param)
    if any(leaf.dtype != jnp.float32 for leaf in jax.tree.leaves(params)):
        raise ValueError("All trainable parameters must initialize in FP32")
    return TrainState.create(graphdef, params=params, tx=make_optimizer(config))

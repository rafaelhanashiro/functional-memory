import jax
from flax import nnx


def model_forward_rngs(rng: jax.Array) -> nnx.Rngs:
    """Derive the model stream from the seed/update/microbatch/device RNG clock."""
    return nnx.Rngs(dropout=jax.random.fold_in(rng, 0))

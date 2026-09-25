import jax
import optax
from functools import partial
from flax import nnx

from scaffolds.transformer.configs import Config
from scaffolds.transformer.attention.parameters import attention_parameter_specs


def make_learning_rate_schedule(config: Config):
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optim.learning_rate,
        warmup_steps=config.optim.warmup_steps,
        decay_steps=max(1, config.optim.total_steps),
        end_value=config.optim.min_learning_rate,
    )


def make_optimizer(config: Config):
    schedule = make_learning_rate_schedule(config)
    excluded = frozenset(
        spec.name for spec in attention_parameter_specs(config.model) if not spec.weight_decay
    )
    return optax.chain(
        optax.clip_by_global_norm(config.optim.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=config.optim.beta1,
            b2=config.optim.beta2,
            weight_decay=config.optim.weight_decay,
            mask=partial(attention_weight_decay_mask, excluded_names=excluded),
        ),
    )


def attention_weight_decay_mask(params, *, excluded_names):
    """Exclude biases, role-tagged parameters, and mechanism-owned exceptions.

    Embeddings and normalization modules mark their parameters with
    ``weight_decay=False``. This handles multidimensional normalization scales
    without assuming that every matrix should decay. Biases follow NNX's
    ``bias`` parameter name, wherever the module lives in the model tree.
    """
    def mask_parameter(path, parameter):
        keys = tuple(entry.key for entry in path if isinstance(entry, jax.tree_util.DictKey))
        decay = not keys or (keys[0] not in excluded_names and keys[-1] != "bias")
        if isinstance(parameter, nnx.Param):
            decay = decay and parameter.get_metadata("weight_decay", True)
        # Keep variable wrappers and their metadata so the mask has exactly the
        # same tree structure as parameters, gradients, and optimizer moments.
        return jax.tree.map(lambda _: decay, parameter)

    return jax.tree.map_with_path(
        mask_parameter, params, is_leaf=lambda value: isinstance(value, nnx.Param),
    )


def quadratic_weight_decay_mask(params):
    """Historical helper retained for callers of the learned-quadratic API."""
    return attention_weight_decay_mask(params, excluded_names={"quadratic_coefficients"})

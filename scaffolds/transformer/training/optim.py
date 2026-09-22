import jax
import optax
from functools import partial

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
            # Mechanisms declare optimizer policy with their parameter ownership.
            # The callable preserves the NNX State tree shape expected by Optax.
            mask=(partial(attention_weight_decay_mask, excluded_names=excluded) if excluded else None),
        ),
    )


def attention_weight_decay_mask(params, *, excluded_names):
    return jax.tree.map_with_path(
        lambda path, _: getattr(path[0], "key", None) not in excluded_names, params,
    )


def quadratic_weight_decay_mask(params):
    """Historical helper retained for callers of the learned-quadratic API."""
    return attention_weight_decay_mask(params, excluded_names={"quadratic_coefficients"})

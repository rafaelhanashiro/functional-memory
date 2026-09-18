import optax

from experiments.polynomial_attention.configs import Config


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
    return optax.chain(
        optax.clip_by_global_norm(config.optim.grad_clip_norm),
        optax.adamw(
            learning_rate=schedule,
            b1=config.optim.beta1,
            b2=config.optim.beta2,
            weight_decay=config.optim.weight_decay,
        ),
    )

import jax
import optax
from functools import partial
from flax import nnx
from typing import NamedTuple

from scaffolds.transformer.configs import Config
from scaffolds.transformer.attention.parameters import attention_parameter_specs
from scaffolds.transformer.training.coefficient_release import CoefficientReleaseOptimizer, coefficient_release


def make_learning_rate_schedule(config: Config):
    if config.optim.warmup_steps == 0 and config.optim.learning_rate == config.optim.min_learning_rate:
        return optax.constant_schedule(config.optim.learning_rate)
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
    adam = optax.adamw(
            learning_rate=schedule,
            b1=config.optim.beta1,
            b2=config.optim.beta2,
            weight_decay=config.optim.weight_decay,
            mask=partial(attention_weight_decay_mask, excluded_names=excluded),
        )
    if (config.continuation.parent_checkpoint and config.continuation.optimizer == "parent"
            and continuation_introduces_parameters(config)):
        # Newly introduced parameters have their own Adam clock; using the backbone's
        # old count with zero moments would misapply bias correction. They may also
        # use their own constant learning rate. A fork (optimizer: fresh) restarts every
        # parameter's AdamW together, so it keeps the plain optimizer and its state layout.
        # Weight decay follows each array's declaration, as in pretraining: only arrays
        # declared with decay (the query coefficient projection W under
        # attention_coefficient_kernel_weight_decay, or the forget gate's w_alpha) decay.
        new_rate = config.continuation.new_parameter_learning_rate
        coefficient_adam = optax.adamw(schedule if new_rate is None else optax.constant_schedule(new_rate),
                                       b1=config.optim.beta1, b2=config.optim.beta2,
                                       weight_decay=config.optim.weight_decay,
                                       mask=partial(attention_weight_decay_mask, excluded_names=excluded))
        backbone_adam = adam

        def init(params):
            coefficients, backbone = split_continuation_parameters(params)
            return ContinuationAdamState(backbone_adam.init(backbone), coefficient_adam.init(coefficients))

        def update(updates, state, params=None):
            coefficient_grads, backbone_grads = split_continuation_parameters(updates)
            coefficient_params, backbone_params = split_continuation_parameters(params)
            backbone_updates, backbone_state = backbone_adam.update(backbone_grads, state.backbone, backbone_params)
            coefficient_updates, coefficient_state = coefficient_adam.update(
                coefficient_grads, state.coefficients, coefficient_params)
            return nnx.State.merge(backbone_updates, coefficient_updates), ContinuationAdamState(
                backbone_state, coefficient_state)

        adam = optax.GradientTransformation(init, update)
    query_scale = config.optim.query_coefficient_learning_rate_scale
    if query_scale != 1.0:
        adam_update = adam.update

        def update_query_coefficients(updates, state, params=None):
            # Scale Adam's completed update, after global clipping and moment
            # normalization. Multiplying raw gradients would not multiply Adam's LR.
            updates, state = adam_update(updates, state, params)

            def scale_parameter(path, value):
                keys = tuple(entry.key for entry in path if isinstance(entry, jax.tree_util.DictKey))
                return value * query_scale if keys and keys[0] in QUERY_COEFFICIENT_READOUTS else value

            return jax.tree.map_with_path(scale_parameter, updates), state

        # The multiplier has no state. Keep the original optimizer state layout,
        # moments, and counters, including the unchanged default checkpoint format.
        adam = optax.GradientTransformation(adam.init, update_query_coefficients)
    optimizer = optax.chain(optax.clip_by_global_norm(config.optim.grad_clip_norm), adam)
    release = coefficient_release(config)
    if release is None:
        return optimizer
    # Frozen coefficient entries lose their gradients before clipping and their final
    # updates after AdamW (training/coefficient_release.py). The release reads the
    # schedule's own count and adds no optimizer state.
    return CoefficientReleaseOptimizer(optimizer, release)


class ContinuationAdamState(NamedTuple):
    backbone: object
    coefficients: object


QUERY_COEFFICIENT_READOUTS = frozenset({"query_polynomial_bias", "query_polynomial_amplitude"})


# The arrays a continuation can introduce beside its fixed-Taylor parent: learned static or
# query coefficients, the per-token query scale gate, and the forget gate.
CONTINUATION_PARAMETERS = frozenset({
    "polynomial_coefficients", "query_polynomial_bias", "query_polynomial_kernel",
    "query_polynomial_amplitude", "query_polynomial_mixing",
    "qk_norm_token_scale_kernel", "qk_norm_token_scale_amplitude",
    "forget_gate_kernel", "forget_gate_key_kernel", "forget_gate_log_rate", "forget_gate_dt_bias",
    "forget_gate_bias",
})


def continuation_introduces_parameters(config) -> bool:
    """Whether the recipe adds parameter arrays that its parent checkpoint lacks."""
    if config.continuation.preserve_all_parameters:
        return False
    return (config.model.attention_coefficients == "learned"
            or getattr(config.model, "qk_norm_token_scale", "none") != "none"
            or getattr(config.model, "attention_forget_gate", False))


def split_continuation_parameters(params):
    """Separate the introduced coefficient or token-scale parameters from the retained backbone."""
    return params.split(lambda path, _: path[0] in CONTINUATION_PARAMETERS, ...)


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

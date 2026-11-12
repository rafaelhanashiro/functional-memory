"""Supervised fitting of independent dense coefficient generators.

Inputs have shape [layer, example, hidden], where an example is one query token.
Targets are *factorial coordinates* u_r = r! c_r, with shape
[layer, example, head, order]. The packed model parameters retain their native
stored coordinates: kernel [layer,head,hidden,feature], bias [layer,head,order],
mixing [layer,head,order,feature], and optional tanh_bias [layer,head,feature].

Each layer/head is an independent regression problem. Its loss is averaged over
examples and output coordinates, and gradients are clipped separately for that
network before Adam. Adding another layer or head therefore does not change an
existing network's gradient, clipping, or update.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax

from scaffolds.transformer.attention.input_dependent import project_coefficients


def _validate_shapes(generator, inputs):
    required = {"kernel", "bias", "mixing"}
    if not required <= set(generator) or set(generator) - required - {"tanh_bias"}:
        raise ValueError("Dense regression requires kernel, bias, mixing, and optional tanh_bias")
    kernel, bias, mixing = (generator[key] for key in ("kernel", "bias", "mixing"))
    if inputs.ndim != 3 or kernel.ndim != 4 or bias.ndim != 3 or mixing.ndim != 4:
        raise ValueError("Expected inputs [L,N,D], kernel [L,H,D,K], bias [L,H,R], mixing [L,H,R,K]")
    if (inputs.shape[0] != kernel.shape[0] or inputs.shape[-1] != kernel.shape[-2]
            or kernel.shape[:2] != bias.shape[:2]
            or mixing.shape != (*bias.shape, kernel.shape[-1])):
        raise ValueError("Dense regression parameter and input shapes disagree")
    if "tanh_bias" in generator and generator["tanh_bias"].shape != (*bias.shape[:2], kernel.shape[-1]):
        raise ValueError("tanh_bias must have shape [L,H,K]")


def predict_factorial(generator, inputs, readout_scale=1.0):
    """Return u=r!c using the production FP32 coefficient projection exactly.

Multiplication by r! occurs only on the output of the native projection. The
native bias and mixing each already receive their one factorial division; no
additional gradient preconditioner belongs in this supervised optimizer.
    """
    _validate_shapes(generator, inputs)
    factors = jnp.asarray([math.factorial(r) for r in range(generator["bias"].shape[-1])], jnp.float32)

    def project(x, kernel, bias, mixing, inner):
        return project_coefficients(
            x[None, :, :], kernel, bias / factors[None, :], mixing=mixing,
            activation="tanh", tanh_bias=inner, amplitude_scale="factorial",
            readout_scale=readout_scale,
        )[0] * factors

    return jax.vmap(project, in_axes=(0, 0, 0, 0, 0 if "tanh_bias" in generator else None))(
        inputs, generator["kernel"], generator["bias"], generator["mixing"],
        generator.get("tanh_bias"),
    )


def _directions(values):
    squared = jnp.sum(jnp.square(values.astype(jnp.float32)), axis=-1, keepdims=True)
    return values / jnp.sqrt(jnp.maximum(squared, 1e-24))


def regression_metrics(prediction, targets):
    """Compact mean errors plus per-layer/head errors, without selecting a fit."""
    if prediction.shape != targets.shape or prediction.ndim != 4:
        raise ValueError("Prediction and targets must share [L,N,H,R] shape")
    error = prediction.astype(jnp.float32) - targets.astype(jnp.float32)
    per_network = jnp.mean(jnp.square(error), axis=(1, 3))
    predicted_direction, target_direction = _directions(prediction), _directions(targets)
    cosine = jnp.clip(jnp.sum(predicted_direction * target_direction, axis=-1), -1., 1.)
    direction_error = jnp.sum(jnp.square(predicted_direction - target_direction), axis=-1)
    target_energy = jnp.mean(jnp.square(targets.astype(jnp.float32)))
    return {
        "factorial_mse": jnp.mean(per_network),
        "factorial_mse_by_network": per_network,
        "factorial_relative_l2": jnp.sqrt(jnp.mean(per_network) / jnp.maximum(target_energy, 1e-24)),
        "cosine_distance": jnp.mean(1. - cosine),
        "cosine_distance_by_network": jnp.mean(1. - cosine, axis=1),
        "direction_squared_error": jnp.mean(direction_error),
        "angle_degrees": jnp.mean(jnp.arccos(cosine)) * (180. / jnp.pi),
    }


def clip_network_gradients(gradient, maximum_norm):
    """Clip each [layer,head] network independently; return original norms."""
    squared = sum(jnp.sum(jnp.square(value.astype(jnp.float32)), axis=tuple(range(2, value.ndim)))
                  for value in jax.tree.leaves(gradient))
    norms = jnp.sqrt(squared)
    multiplier = jnp.minimum(1., maximum_norm / jnp.maximum(norms, 1e-30))
    clipped = jax.tree.map(
        lambda value: value * multiplier.reshape((*multiplier.shape, *((1,) * (value.ndim - 2)))),
        gradient,
    )
    return clipped, norms


def make_regression_functions(*, readout_scale=1.0, beta1=.9, beta2=.95,
                              epsilon=1e-8, clip_norm=1.0, objective="factorial_mse"):
    """Return ``init(generator), step(...), evaluate(generator, inputs, targets)``.

``step(generator, optimizer, inputs, targets, learning_rate)`` returns the new
generator, optimizer, and metrics for the pre-update prediction. All are JITed.
The default raw factorial MSE preserves the exact fitted teacher vectors.
``direction_mse`` is an explicit alternative, not an implicit normalization.
    """
    if objective not in {"factorial_mse", "direction_mse"}:
        raise ValueError("Unknown coefficient regression objective")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(readout_scale) or readout_scale <= 0:
        raise ValueError("readout_scale must be finite and positive")
    tx = optax.scale_by_adam(b1=beta1, b2=beta2, eps=epsilon)

    def loss(generator, inputs, targets):
        prediction = predict_factorial(generator, inputs, readout_scale)
        if prediction.shape != targets.shape:
            raise ValueError("Targets must have shape [L,N,H,R]")
        difference = (prediction - targets) if objective == "factorial_mse" else (
            _directions(prediction) - _directions(targets))
        per_network = jnp.mean(jnp.square(difference), axis=(1, 3))
        # Sum independent network means: do not shrink one network's gradients
        # when another network is added to the batch of regressions.
        return jnp.sum(per_network), prediction

    @jax.jit
    def step(generator, optimizer, inputs, targets, learning_rate):
        (_, prediction), gradient = jax.value_and_grad(loss, has_aux=True)(generator, inputs, targets)
        clipped, norms = clip_network_gradients(gradient, clip_norm)
        updates, optimizer = tx.update(clipped, optimizer, generator)
        updates = jax.tree.map(lambda value: -learning_rate * value, updates)
        candidate = optax.apply_updates(generator, updates)
        metrics = regression_metrics(prediction, targets)
        metrics.update(
            gradient_norm_mean=jnp.mean(norms), gradient_norm_max=jnp.max(norms),
            gradient_norm_by_network=norms, clipping_fraction=jnp.mean(norms > clip_norm),
            finite=jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in jax.tree.leaves(
                (candidate, optimizer, gradient, prediction, targets))])),
        )
        return candidate, optimizer, metrics

    @jax.jit
    def evaluate(generator, inputs, targets):
        return regression_metrics(predict_factorial(generator, inputs, readout_scale), targets)

    return jax.jit(tx.init), step, evaluate

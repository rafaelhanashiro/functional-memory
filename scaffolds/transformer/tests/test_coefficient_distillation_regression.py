"""Regression coordinates and independent fitting agree with native generators."""
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.input_dependent import project_coefficients
from scaffolds.transformer.training.coefficient_distillation_regression import (
    clip_network_gradients, make_regression_functions, predict_factorial,
)


def _example(seed=3, layers=2, heads=2, features=9, orders=5):
    rng = np.random.default_rng(seed)
    generator = {
        "kernel": jnp.asarray(rng.normal(size=(layers, heads, 4, features)) * .3, jnp.float32),
        "bias": jnp.asarray(rng.normal(size=(layers, heads, orders)) * .1 + 1., jnp.float32),
        "mixing": jnp.asarray(rng.normal(size=(layers, heads, orders, features)) * .3, jnp.float32),
    }
    x = jnp.asarray(rng.normal(size=(layers, 64, 4)), jnp.float32)
    return generator, x


@pytest.mark.parametrize("features", (5, 9))
@pytest.mark.parametrize("inner_bias", (False, True))
def test_predictions_and_derivatives_use_native_factorial_coordinates(features, inner_bias):
    generator, x = _example(features=features)
    if inner_bias:
        generator["tanh_bias"] = jnp.full((2, 2, features), .12, jnp.float32)
    factors = jnp.asarray([math.factorial(r) for r in range(5)], jnp.float32)

    def native(p):
        outputs = []
        for layer in range(2):
            outputs.append(project_coefficients(
                x[layer][None], p["kernel"][layer], p["bias"][layer] / factors,
                mixing=p["mixing"][layer], activation="tanh", amplitude_scale="factorial",
                readout_scale=1 / 3,
                tanh_bias=p["tanh_bias"][layer] if inner_bias else None,
            )[0] * factors)
        return jnp.stack(outputs)

    predicted = jax.jit(lambda p: predict_factorial(p, x, 1 / 3))(generator)
    np.testing.assert_allclose(predicted, jax.jit(native)(generator), rtol=2e-6, atol=2e-7)
    cotangent = jax.random.normal(jax.random.PRNGKey(3), predicted.shape)
    actual = jax.grad(lambda p: jnp.sum(predict_factorial(p, x, 1 / 3) * cotangent))(generator)
    expected = jax.grad(lambda p: jnp.sum(native(p) * cotangent))(generator)
    for key in generator:
        np.testing.assert_allclose(actual[key], expected[key], rtol=3e-6, atol=2e-6, err_msg=key)


def test_one_network_gradient_does_not_clip_or_rescale_another():
    gradient = {
        "bias": jnp.asarray([[[3., 4.], [.03, .04]]]),
        "mixing": jnp.zeros((1, 2, 2, 3)),
        "kernel": jnp.zeros((1, 2, 4, 3)),
    }
    clipped, norms = clip_network_gradients(gradient, 1.)
    np.testing.assert_allclose(norms, [[5., .05]], rtol=1e-6)
    np.testing.assert_allclose(clipped["bias"], [[[.6, .8], [.03, .04]]], rtol=1e-6)


def test_adding_networks_leaves_each_adam_update_unchanged():
    generator, x = _example(features=3, orders=3)
    target = predict_factorial(generator, x, 1 / 3) + jnp.asarray([.7, -.3, .9])
    init, step, _ = make_regression_functions(readout_scale=1 / 3, clip_norm=.02)
    combined = step(generator, init(generator), x, target, .01)[0]
    for layer in range(2):
        for head in range(2):
            individual = {key: value[layer:layer + 1, head:head + 1] for key, value in generator.items()}
            fitted = step(individual, init(individual), x[layer:layer + 1],
                          target[layer:layer + 1, :, head:head + 1], .01)[0]
            for key in generator:
                np.testing.assert_allclose(combined[key][layer:layer + 1, head:head + 1], fitted[key],
                                           rtol=2e-6, atol=2e-7, err_msg=key)


def test_synthetic_dense_teacher_can_be_fitted_and_optimizer_resume_is_exact():
    teacher, x = _example(features=3, orders=3)
    targets = predict_factorial(teacher, x, 1 / 3)
    student = {"kernel": teacher["kernel"] + .04, "bias": jnp.ones_like(teacher["bias"]),
               "mixing": jnp.zeros_like(teacher["mixing"])}
    init, step, evaluate = make_regression_functions(readout_scale=1 / 3)
    initial_loss = float(evaluate(student, x, targets)["factorial_mse"])
    optimizer = init(student)
    for _ in range(120):
        student, optimizer, metrics = step(student, optimizer, x, targets, .02)
        assert bool(metrics["finite"])
    final_loss = float(evaluate(student, x, targets)["factorial_mse"])
    assert final_loss < initial_loss * .01
    # A checkpoint round trip is simply a lossless device-to-host tree transfer;
    # no hidden RNG or optimizer counter may alter the next update.
    restored = jax.tree.map(lambda value: jnp.asarray(np.asarray(value)), (student, optimizer))
    direct = step(student, optimizer, x, targets, .002)
    resumed = step(*restored, x, targets, .002)
    for expected, actual in zip(jax.tree.leaves(direct), jax.tree.leaves(resumed)):
        np.testing.assert_array_equal(expected, actual)


def test_nonfinite_target_is_reported_and_invalid_shapes_are_rejected():
    generator, x = _example()
    target = predict_factorial(generator, x, 1 / 3).at[0, 0, 0, 0].set(jnp.nan)
    init, step, _ = make_regression_functions(readout_scale=1 / 3)
    assert not bool(step(generator, init(generator), x, target, .001)[2]["finite"])
    with pytest.raises(ValueError, match="shapes disagree"):
        predict_factorial(generator, x[..., :-1], 1 / 3)

"""Scalar-basis attention: independent dense oracle, gradients, recipes, and resume."""
from copy import deepcopy
from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scaffolds.transformer.attention.basis import fit_exp, monomial, rbf
from scaffolds.transformer.attention.core import attention_core
from scaffolds.transformer.attention.backends.reference import blocked_dot_product_attention
from scaffolds.transformer.attention.backends.selection import resolve_attention_backend
from scaffolds.transformer.configs import Config, ModelConfig, config_to_dict, load_config, save_config
from scaffolds.transformer.configs.loader import config_dicts_equal
from scaffolds.transformer.tests.test_training import tiny_config
from scaffolds.transformer.training import loop
from scaffolds.transformer.training.diagnostics import attention_identity
from scaffolds.transformer.acceptance import attention_test_paths, matches_softmax_backbone_initialization
from scaffolds.records import _attention_family


@pytest.fixture
def projection():
    return fit_exp([monomial(0), rbf(-1, 1.5), rbf(1, 1.5)])


@pytest.mark.parametrize("normalized", [False, True])
def test_rbf_attention_and_all_gradients_match_independent_dense_oracle(projection, normalized):
    arrays = [jnp.asarray(a, dtype=jnp.float32) for a in np.random.default_rng(17).normal(
        scale=0.15, size=(3, 2, 7, 2, 4))]
    c0, c1, c2 = projection.coefficients

    def oracle(q, k, v):
        scores = jnp.einsum("bthd,bshd->bhts", q, k) / 2
        kernel = c0 + c1*jnp.exp(-0.5*((scores+1)/1.5)**2) + c2*jnp.exp(-0.5*((scores-1)/1.5)**2)
        kernel = jnp.where(jnp.tril(jnp.ones((7, 7), dtype=bool)), kernel, 0)
        if normalized:
            kernel = kernel / kernel.sum(-1, keepdims=True)
        return jnp.einsum("bhts,bshd->bthd", kernel, v)

    def blocked(q, k, v):
        return blocked_dot_product_attention(q, k, v, score_kernel=projection,
                                             attention_normalization=normalized, query_block_size=3)

    np.testing.assert_allclose(jax.jit(blocked)(*arrays), oracle(*arrays), atol=2e-6, rtol=2e-5)
    actual = jax.grad(lambda *xs: (blocked(*xs)**2).sum(), argnums=(0, 1, 2))(*arrays)
    expected = jax.grad(lambda *xs: (oracle(*xs)**2).sum(), argnums=(0, 1, 2))(*arrays)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=3e-6, rtol=5e-5)
    changed_values = arrays[2].at[:, -1].add(100)
    np.testing.assert_array_equal(blocked(*arrays)[:, :-1], blocked(*arrays[:2], changed_values)[:, :-1])


def test_exp_callable_reproduces_softmax_without_score_shifting():
    q, k, v = [jnp.asarray(x, dtype=jnp.float32) for x in np.random.default_rng(1).normal(size=(3, 1, 5, 1, 2))]
    actual = attention_core(q, k, v, score_kernel=jnp.exp)
    expected = attention_core(q, k, v)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_signed_zero_and_nonfinite_normalizers_keep_failure_semantics():
    x = jnp.ones((1, 3, 1, 2))
    result, diagnostics = attention_core(x, x, x, score_kernel=lambda s: -jnp.ones_like(s), return_diagnostics=True)
    np.testing.assert_array_equal(result, x)  # A negative, nonzero sum is valid.
    assert diagnostics["negative_kernel_count"] == 6
    assert diagnostics["invalid_normalizer_count"] == 0
    for kernel in (lambda s: jnp.zeros_like(s), lambda s: jnp.full_like(s, jnp.inf)):
        result, diagnostics = attention_core(x, x, x, score_kernel=kernel, return_diagnostics=True)
        assert np.isnan(result).all()
        assert diagnostics["invalid_normalizer_count"] == 3
    result = attention_core(x, x, x, score_kernel=lambda s: jnp.zeros_like(s), attention_normalization=False)
    np.testing.assert_array_equal(result, jnp.zeros_like(x))


def test_named_projection_config_roundtrip_and_legacy_compatibility(tmp_path, projection):
    config = Config(model=ModelConfig(attention_basis=projection.to_config()))
    path = tmp_path / "recipe.yaml"
    save_config(config, path)
    assert config_to_dict(load_config(path)) == config_to_dict(config)
    assert attention_identity(config)["attention_kernel"] == "fixed_basis"
    assert _attention_family(config_to_dict(config)["model"], {}, None) == "basis"
    for platform in ("cpu", "gpu"):
        assert resolve_attention_backend(config.model, platform=platform) == "blocked"
    old = config_to_dict(Config())
    current = deepcopy(old)
    del old["model"]["attention_basis"]
    assert config_dicts_equal(old, current)
    assert not config_dicts_equal(old, config_to_dict(config))
    paths = attention_test_paths(config_to_dict(config), "blocked", cpu=True)
    assert "scaffolds/transformer/tests/test_basis_attention.py" in paths


@pytest.mark.parametrize("override", [{"attention_degree": 2}, {"attention_mechanism": "gla"},
                                       {"attention_coefficients": "learned"}, {"attention_backend": "fused"},
                                       {"attention_backend": "mosaic"}, {"attention_backend": "cudnn"},
                                       {"attention_forget_gate": True}, {"attention_kernel_basis": "bspline"},
                                       {"attention_coefficient_readout": "dense"},
                                       {"attention_blocked_implementation": "strips"}])
def test_conflicting_configuration_rejected(projection, override):
    with pytest.raises(ValueError):
        ModelConfig(attention_basis=projection.to_config(), **override)


@pytest.mark.parametrize("options", [
    {"query_log_decay": jnp.zeros((1, 3, 1))}, {"spline_knots": (-1.0, 0.0, 1.0)},
    {"query_decay_mode": "query_distance"}, {"blocked_implementation": "strips"},
])
def test_scalar_dispatch_rejects_unimplemented_kernel_combinations(options):
    x = jnp.ones((1, 3, 1, 2))
    with pytest.raises(ValueError, match="scalar score kernel"):
        attention_core(x, x, x, score_kernel=jnp.exp, **options)


def test_checkpoint_resume_keeps_frozen_fit_and_trains_with_output_scaling(tmp_path):
    fit = fit_exp([monomial(0), monomial(1), monomial(2)])
    config = tiny_config(tmp_path)
    config.model = replace(config.model, attention_basis=fit.to_config(), qk_norm=True,
                           attention_normalization=False, attention_output_scaling="head_rms_norm",
                           initialization="transformer_reference")
    matches_softmax_backbone_initialization(config)
    loop.run_training(config, stop_after_step=2)
    loop.run_training(config)
    completion = json.loads((tmp_path / "completion.json").read_text())
    assert completion["final_step"] == 4
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["attention_kernel"] == "fixed_basis_unnormalized"
    assert metadata["attention_basis"] == fit.to_config()
    diagnostics = completion["attention_diagnostics"]
    assert diagnostics["valid"]
    assert diagnostics["kernel_values"] == "unshifted_fixed_basis"
    changed = deepcopy(config)
    changed.model.attention_basis["coefficients"][0] += 0.01
    with pytest.raises(ValueError, match="config|Config"):
        loop.run_training(changed)

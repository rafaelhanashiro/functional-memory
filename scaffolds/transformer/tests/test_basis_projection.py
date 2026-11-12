"""Check numerical projections against analytic Gaussian integrals, not another fit."""
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from scaffolds.transformer.attention.basis import (
    basis_from_spec, cosine, fit_exp, monomial, projection_from_config, rbf, sine,
)


@pytest.mark.parametrize("degree", [0, 1, 2, 4, 8])
@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.7, 1.3)])
def test_polynomial_fit_recovers_gaussian_hermite_projection(degree, mean, std):
    fit = fit_exp([monomial(order) for order in range(degree + 1)], mean=mean, std=std)
    hermite_coefficients = [math.exp(mean + std**2 / 2) * std**r / math.factorial(r)
                            for r in range(degree + 1)]
    in_standard_coordinate = Polynomial(np.polynomial.hermite_e.herme2poly(hermite_coefficients))
    expected = in_standard_coordinate(Polynomial([-mean / std, 1 / std])).coef
    np.testing.assert_allclose(fit.coefficients, expected, atol=3e-9, rtol=2e-9)
    assert fit.diagnostics["quadrature_converged"]
    # Parseval: omitted Hermite terms have energy exp(2mu+sigma²) * sum_{r>m} sigma^(2r)/r!.
    tail = sum(std ** (2*r) / math.factorial(r) for r in range(degree + 1, 70))
    expected_error = math.exp(mean + std**2 / 2) * math.sqrt(tail)
    assert fit.diagnostics["l2_error"] == pytest.approx(expected_error, rel=2e-9)


def test_rbf_fit_matches_closed_form_gaussian_integral():
    mean, std, center, width = 0.3, 1.2, 0.8, 1.1
    fit = fit_exp([rbf(center, width)], mean=mean, std=std)
    cross = (math.exp(mean + std**2 / 2) * width / math.sqrt(width**2 + std**2)
             * math.exp(-(mean + std**2 - center)**2 / (2 * (width**2 + std**2))))
    norm = width / math.sqrt(width**2 + 2 * std**2) * math.exp(-(mean-center)**2 / (width**2 + 2*std**2))
    assert fit.coefficients[0] == pytest.approx(cross / norm, rel=1e-11)


@pytest.mark.parametrize("kind", [sine, cosine])
def test_trigonometric_fit_matches_gaussian_characteristic_function(kind):
    frequency = 0.8
    fit = fit_exp([kind(frequency)])
    cross = math.exp((1-frequency**2)/2) * (math.sin(frequency) if kind is sine else math.cos(frequency))
    norm = (1 + (-1 if kind is sine else 1) * math.exp(-2*frequency**2)) / 2
    assert fit.coefficients[0] == pytest.approx(cross / norm, rel=1e-12)


def test_callable_fit_preserves_precision_setting_and_is_jittable_and_differentiable():
    before = jax.config.x64_enabled
    fit = fit_exp([lambda x: 1.0, lambda x: x, lambda x: x*x])
    assert jax.config.x64_enabled == before
    x = jnp.linspace(-2, 2, 19)
    expected = math.sqrt(math.e) * (0.5 + x + 0.5*x*x)
    np.testing.assert_allclose(jax.jit(fit)(x), expected, atol=5e-7)
    derivative = jax.jit(jax.grad(lambda x: fit(x).sum()))(x)
    np.testing.assert_allclose(derivative, math.sqrt(math.e)*(1+x), atol=5e-7)
    with pytest.raises(ValueError, match="Only named"):
        fit.to_config()


def test_serialized_named_basis_roundtrip_never_refits(monkeypatch):
    fit = fit_exp([monomial(0), rbf(-1, 1.5), rbf(1, 1.5), sine(0.5), cosine(0.5)])
    spec = json.loads(json.dumps(fit.to_config(), allow_nan=False))
    monkeypatch.setattr(np.linalg, "lstsq", lambda *a, **kw: pytest.fail("Restoring must not refit"))
    restored = projection_from_config(spec)
    assert restored.to_config() == spec
    x = jnp.linspace(-3, 3, 31)
    np.testing.assert_array_equal(restored(x), fit(x))


def test_increasing_span_cannot_increase_projection_error():
    errors = [fit_exp([monomial(r) for r in range(degree+1)]).diagnostics["relative_l2_error"]
              for degree in range(6)]
    assert all(a > b for a, b in zip(errors, errors[1:]))


def test_coarse_quadrature_reports_unconverged_oscillatory_fit():
    fit = fit_exp([monomial(0), cosine(15)], quadrature_order=16, check_order=128)
    assert not fit.diagnostics["quadrature_converged"]


def test_quadrature_checks_exponential_tail_even_when_fitted_function_changes_little():
    fit = fit_exp([monomial(0)], std=8, quadrature_order=16, check_order=32)
    assert fit.diagnostics["relative_quadrature_change"] < 1e-7
    assert fit.diagnostics["relative_target_norm_error"] > 0.1
    assert not fit.diagnostics["quadrature_converged"]


@pytest.mark.parametrize("basis", [[], [monomial(0), monomial(0)], [lambda x: 0*x],
                                    [lambda x: x*jnp.nan], [lambda x: jnp.exp(1j*x)], ["exp"]])
def test_invalid_or_dependent_basis_raises(basis):
    with pytest.raises(ValueError):
        fit_exp(basis)


@pytest.mark.parametrize("kwargs", [{"std": 0}, {"mean": float("nan")}, {"rcond": 1},
                                     {"quadrature_order": 0}, {"check_order": 32}, {"check_order": 512}])
def test_invalid_fitting_options_raise(kwargs):
    with pytest.raises(ValueError):
        fit_exp([monomial(0)], **kwargs)


@pytest.mark.parametrize("spec", [{"kind": "unknown"}, {"kind": "sin", "frequency": 1},
                                   {"kind": "rbf", "center": 0, "width": -1},
                                   {"kind": "monomial", "power": True},
                                   {"kind": "monomial", "power": 2, "extra": 1}])
def test_invalid_named_functions_raise(spec):
    with pytest.raises(ValueError):
        basis_from_spec(spec)

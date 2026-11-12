"""Gaussian L2 projection of exp onto user-supplied scalar functions.

Fit once on the CPU in float64, then evaluate the fixed sum with JAX. This is
an approximation of the scalar score kernel, not a query/key feature map.
Quadrature approximates the continuous projection; independent higher-order
quadrature reports its accuracy and whether the fitted function has stabilized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np


def _number(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return float(value)


@dataclass(frozen=True)
class BasisFunction:
    """A named, serializable, differentiable scalar function. Use the factories below."""

    kind: str
    parameters: tuple[tuple[str, float], ...]

    def __post_init__(self):
        expected = {"monomial": {"power"}, "rbf": {"center", "width"},
                    "sin": {"frequency", "phase"}, "cos": {"frequency", "phase"}}
        values = dict(self.parameters)
        if (self.kind not in expected or set(values) != expected[self.kind]
                or len(values) != len(self.parameters)):
            raise ValueError(f"Invalid parameters for basis function {self.kind!r}")
        for name, value in values.items():
            _number(value, name, positive=name == "width")
        if self.kind == "monomial" and (type(values["power"]) is not int or values["power"] < 0):
            raise ValueError("Monomial power must be a nonnegative integer")

    def __call__(self, x):
        x = jnp.asarray(x)
        values = dict(self.parameters)
        if self.kind == "monomial":
            return x ** values["power"]
        if self.kind == "rbf":
            return jnp.exp(-0.5 * ((x - values["center"]) / values["width"]) ** 2)
        phase = values["frequency"] * x + values["phase"]
        return jnp.sin(phase) if self.kind == "sin" else jnp.cos(phase)

    def to_spec(self):
        return {"kind": self.kind, **dict(self.parameters)}


def monomial(power: int) -> BasisFunction:
    return BasisFunction("monomial", (("power", power),))


def rbf(center: float, width: float) -> BasisFunction:
    """exp(-0.5 * ((x - center) / width)**2)."""
    return BasisFunction("rbf", (("center", _number(center, "center")),
                                 ("width", _number(width, "width", positive=True))))


def sine(frequency: float, phase: float = 0.0) -> BasisFunction:
    return BasisFunction("sin", (("frequency", _number(frequency, "frequency")),
                                 ("phase", _number(phase, "phase"))))


def cosine(frequency: float, phase: float = 0.0) -> BasisFunction:
    return BasisFunction("cos", (("frequency", _number(frequency, "frequency")),
                                 ("phase", _number(phase, "phase"))))


def basis_from_spec(spec: dict) -> BasisFunction:
    if not isinstance(spec, dict) or not isinstance(spec.get("kind"), str):
        raise ValueError("A basis specification needs a string kind")
    return BasisFunction(spec["kind"], tuple(sorted((key, value) for key, value in spec.items() if key != "kind")))


def _orders(quadrature_order, check_order, count):
    # numpy's quadrature is reliable at these moderate orders; reject larger
    # requests rather than silently overflow its Hermite recurrence.
    if (type(quadrature_order) is not int or type(check_order) is not int
            or not count <= quadrature_order < check_order <= 256):
        raise ValueError("Require basis size <= quadrature_order < check_order <= 256")


@dataclass(frozen=True)
class ExpProjection:
    basis: tuple[Callable, ...]
    coefficients: tuple[float, ...]
    mean: float
    std: float
    quadrature_order: int
    check_order: int
    rcond: float
    diagnostics: dict = field(compare=False, hash=False)

    def __call__(self, scores):
        scores = jnp.asarray(scores)
        scores = scores.astype(jnp.result_type(scores.dtype, jnp.float32))
        output = jnp.zeros_like(scores)
        for coefficient, function in zip(self.coefficients, self.basis, strict=True):
            output = output + coefficient * function(scores)
        return output

    def to_config(self) -> dict:
        """Freeze a named basis and its fitted coefficients in model.attention_basis.

        Anonymous callables are intentionally not serialized or evaluated as code.
        """
        if not all(isinstance(function, BasisFunction) for function in self.basis):
            raise ValueError("Only named basis functions can be saved; callables are for the Python API")
        return {
            "version": 1, "target": "exp", "basis": [function.to_spec() for function in self.basis],
            "coefficients": list(self.coefficients), "mean": self.mean, "std": self.std,
            "quadrature_order": self.quadrature_order, "check_order": self.check_order,
            "rcond": self.rcond, "diagnostics": dict(self.diagnostics),
        }


def projection_from_config(spec: dict) -> ExpProjection:
    """Validate and restore a frozen fit without rerunning quadrature."""
    keys = {"version", "target", "basis", "coefficients", "mean", "std", "quadrature_order",
            "check_order", "rcond", "diagnostics"}
    if not isinstance(spec, dict) or set(spec) != keys or type(spec["version"]) is not int or spec["version"] != 1:
        raise ValueError("Invalid attention_basis projection specification (version 1 required)")
    if spec["target"] != "exp" or not isinstance(spec["basis"], list) or not spec["basis"]:
        raise ValueError("attention_basis requires target exp and a nonempty basis list")
    basis = tuple(basis_from_spec(function) for function in spec["basis"])
    values = spec["coefficients"]
    if not isinstance(values, (tuple, list)) or len(values) != len(basis):
        raise ValueError("attention_basis needs one coefficient per basis function")
    coefficients = tuple(_number(value, "coefficient") for value in values)
    mean, std = _number(spec["mean"], "mean"), _number(spec["std"], "std", positive=True)
    _orders(spec["quadrature_order"], spec["check_order"], len(basis))
    rcond = _number(spec["rcond"], "rcond", positive=True)
    if rcond >= 1 or not isinstance(spec["diagnostics"], dict):
        raise ValueError("Invalid projection rcond or diagnostics")
    return ExpProjection(basis, coefficients, mean, std, spec["quadrature_order"], spec["check_order"],
                         rcond, dict(spec["diagnostics"]))


def _sample(basis, mean, std, order):
    nodes, weights = np.polynomial.hermite_e.hermegauss(order)
    nodes = mean + std * nodes
    weights = weights / math.sqrt(2 * math.pi)
    # Callables may use either NumPy or JAX. Keep fitting in float64 even when
    # training uses JAX's default float32, without changing the global setting.
    with jax.enable_x64(), jax.default_device(jax.devices("cpu")[0]):
        columns = []
        for function in basis:
            values = np.asarray(function(nodes))
            if np.iscomplexobj(values):
                raise ValueError("Basis functions must be real-valued; use real sine/cosine pairs")
            columns.append(np.broadcast_to(values.astype(np.float64), nodes.shape))
    matrix = np.stack(columns, axis=1)
    with np.errstate(over="ignore", invalid="ignore"):
        target = np.exp(nodes)
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite basis or exponential values at quadrature nodes")
    return matrix, target, weights


def _solve(matrix, target, weights, rcond):
    root_weights = np.sqrt(weights)
    weighted = matrix * root_weights[:, None]
    scales = np.linalg.norm(weighted, axis=0)
    if not np.isfinite(scales).all() or np.any(scales == 0):
        raise ValueError("Basis contains a zero or nonfinite quadrature norm")
    # Column scaling makes the rank check independent of a function's units.
    # Solve the weighted design directly: forming its Gram matrix squares its
    # condition number and unnecessarily loses accuracy.
    scaled_coefficients, _, rank, singular = np.linalg.lstsq(
        weighted / scales, target * root_weights, rcond=rcond)
    if rank != matrix.shape[1]:
        raise ValueError(f"Basis is numerically linearly dependent: rank {rank} of {matrix.shape[1]}")
    coefficients = scaled_coefficients / scales
    if not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite fitted coefficients")
    return coefficients, float(singular[0] / singular[-1])


def fit_exp(basis, *, mean=0.0, std=1.0, quadrature_order=64, check_order=None, rcond=1e-12) -> ExpProjection:
    """Approximate argmin_c E[(exp(X) - sum_j c_j b_j(X))**2], X ~ N(mean,std²).

    Accept a sequence of named functions, specification dictionaries, or
    vectorized callables (constant scalar returns are broadcast). Callables used
    in attention must also support JAX tracing and differentiation. Coefficients
    are fixed, shared across heads/layers, and expressed in the original basis.

    A second, higher quadrature order checks the fitted function and estimates
    its error. ``quadrature_converged`` is a numerical diagnostic, not a proof
    for arbitrary callables; narrow or rapidly oscillating functions may need
    more nodes. Rank-deficient bases raise instead of silently dropping terms.
    """
    basis = tuple(basis_from_spec(function) if isinstance(function, dict) else function for function in basis)
    if not basis or not all(callable(function) for function in basis):
        raise ValueError("basis must be a nonempty sequence of callable functions or named specifications")
    mean, std = _number(mean, "mean"), _number(std, "std", positive=True)
    rcond = _number(rcond, "rcond", positive=True)
    if rcond >= 1:
        raise ValueError("rcond must be less than 1")
    check_order = 2 * quadrature_order if check_order is None else check_order
    _orders(quadrature_order, check_order, len(basis))
    matrix, target, weights = _sample(basis, mean, std, quadrature_order)
    coefficients, condition = _solve(matrix, target, weights, rcond)
    check_matrix, check_target, check_weights = _sample(basis, mean, std, check_order)
    checked_coefficients, check_condition = _solve(check_matrix, check_target, check_weights, rcond)
    fitted = check_matrix @ coefficients
    root_weights = np.sqrt(check_weights)
    try:
        target_norm = math.exp(mean + std**2)  # Exact sqrt(E[exp(2X)]).
    except OverflowError as error:
        raise ValueError("Exponential L2 norm exceeds float64 range for this Gaussian") from error
    if target_norm == 0:
        raise ValueError("Exponential L2 norm underflows for this Gaussian")
    target_norm_error = abs(np.linalg.norm(root_weights * check_target) / target_norm - 1)
    error = np.linalg.norm(root_weights * (fitted - check_target))
    fit_change = np.linalg.norm(root_weights * (check_matrix @ (checked_coefficients - coefficients)))
    relative_error, relative_change = error / target_norm, fit_change / target_norm
    diagnostics = {
        "rank": len(basis), "condition_number": condition, "check_condition_number": check_condition,
        "l2_error": float(error), "relative_l2_error": float(relative_error),
        "relative_quadrature_change": float(relative_change),
        "relative_target_norm_error": float(target_norm_error),
        "quadrature_converged": bool(relative_change <= 1e-7 and target_norm_error <= 1e-7),
        "negative_mass_estimate": float(check_weights[fitted < 0].sum()),
    }
    if not all(np.isfinite(value) for value in diagnostics.values()):
        raise ValueError("Nonfinite projection diagnostics; reduce the Gaussian scale or basis magnitudes")
    return ExpProjection(basis, tuple(map(float, coefficients)), mean, std,
                         quadrature_order, check_order, rcond, diagnostics)

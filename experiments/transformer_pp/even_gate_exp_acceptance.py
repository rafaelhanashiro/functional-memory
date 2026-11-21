"""Qualify the matched even polynomial and exponential gate runs of degrees eight and sixteen."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import hashlib

import numpy as np

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.attention.even_gate_exp import DEGREES, even_orders
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

PARENTS = {
    ("rms_norm_cosine", "zero"): ("rf1185ae8a89c", "3f51fc1b16bd82d32377c238fc299b418530d09e924f969c485bdb80598639b5"),
    ("rms_norm", "zero"): ("r4ec7b748d430", "79e2f77b0b95be51f65a37fbf57e1e19ecc796c7bcc9e11186587c34205d1f04"),
    ("rms_norm", "logit_taylor"): ("r2bb06d5ee682", "6fe4fb6759c928f6334a36c4025435a2bf44e9cb6779a0fd97a51f7e2fdbd5e1"),
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")


def recipe(score, bias, degree=16):
    """Add the exponential gate to the corresponding m=8 parent with even powers through degree, on the fused
    kernel; keep its train recipe."""
    if (score, bias) not in PARENTS:
        raise ValueError(f"No matched parent for score {score!r} and sigmoid bias {bias!r}")
    if degree not in DEGREES:
        raise ValueError(f"The gated exponential runs use a degree in {DEGREES}, not {degree!r}")
    run_id, digest = PARENTS[score, bias]
    path = Path(__file__).parent / "runs" / run_id / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Recipe digest changed for parent {run_id}")
    control = load_config(path)
    return replace(control, model=replace(control.model, attention_backend="fused", attention_degree=degree,
                                          attention_coefficient_orders=list(even_orders(degree)),
                                          attention_coefficient_exp_gate=True))


def check_recipe(config):
    validate_config(config)
    model = config.model
    expected = config_to_dict(recipe(model.qk_norm_type, model.attention_coefficient_sigmoid_bias_initial,
                                     model.attention_degree))
    actual = config_to_dict(config)
    for entry in (expected, actual):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
    if actual != expected:
        changed = [f"{section}.{name}" for section in actual for name, value in actual[section].items()
                   if value != expected[section].get(name)]
        raise ValueError(f"Recipe differs from the matched degree-{model.attention_degree} hybrid: {changed}")


def check_pilot(config, outcome, validation):
    diagnostic = validation["attention_diagnostics"]
    stats = diagnostic["coefficients"]
    head_shape = (config.model.num_layers, config.model.num_heads)
    degree = config.model.attention_degree
    orders = list(even_orders(degree))
    polynomial = stats["effective_summary"]
    low, high = (np.asarray(polynomial[key]) for key in ("min", "max"))
    if low.shape != (*head_shape, degree + 1) or high.shape != low.shape:
        raise RuntimeError("Unexpected polynomial gate summary shape")
    absent = (0, *range(1, degree, 2))
    if np.any(low[..., absent] != 0) or np.any(high[..., absent] != 0):
        raise RuntimeError("A constant or odd polynomial coefficient is nonzero")
    exp_gate = stats["exp_gate_summary"]
    exp_low, exp_high = (np.asarray(exp_gate[key]) for key in ("min", "max"))
    if exp_low.shape != head_shape or exp_high.shape != head_shape:
        raise RuntimeError("Unexpected exponential gate summary shape")
    for name, lo, hi in (("polynomial", low[..., orders], high[..., orders]), ("exponential", exp_low, exp_high)):
        if not np.all((0 <= lo) & (lo <= hi) & (hi <= 1)):
            raise RuntimeError(f"Invalid {name} sigmoid gate range")
    lows = np.asarray([layer["score_min"] for layer in diagnostic["layers"]])
    highs = np.asarray([layer["score_max"] for layer in diagnostic["layers"]])
    if not np.isfinite(lows).all() or not np.isfinite(highs).all():
        raise RuntimeError("Nonfinite score in the final pilot validation")
    if config.model.qk_norm_type == "rms_norm_cosine" and max(abs(lows.min()), abs(highs.max())) > 1.01:
        raise RuntimeError("Cosine scores exceed BF16 rounding tolerance")
    return {"polynomial_orders": orders, "min_polynomial_gate": float(low[..., orders].min()),
            "max_polynomial_gate": float(high[..., orders].max()),
            "min_exp_gate": float(exp_low.min()), "max_exp_gate": float(exp_high.max()),
            "score_min": float(lows.min()), "score_max": float(highs.max())}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_pilot=check_pilot,
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_pp/tests/test_even_gate_exp_acceptance.py"))


def main(argv=None):
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

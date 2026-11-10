"""Qualify learned static coefficients with the input-free distance gate against their pinned recipes.

A shared-gate run is a static control (``r4339a245820e`` at degree four, ``r9c69dd7cfc8d`` at degree
eight) with RoPE replaced by the gate: ``positional_encoding: none`` and the gate fields alone differ.
Every other parameter starts at the control's seed-42 start, and the gate's A_log and dt_bias are the
query-distance gates' Gated DeltaNet draws (``r3e26412b2811``, ``r9320e5cd3071``), which the input-free
gate shares by name and shape. A per-order run is its shared-gate run with
``attention_forget_gate_per_order`` alone changed: collapsing its duplicated order axes recovers the
shared run's start.
"""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance, matches_softmax_backbone_initialization
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# Static control of each degree: run, recipe SHA256, seed-42 initial parameters (hash and count, built
# identically at the control's launch commit and at this branch's base).
CONTROLS = {
    4: ("r4339a245820e", "231679d1be72a9b3524f3ee5d7ac71931aa05d00d83ac50919e853196ed8bf9d",
        "460f976c4169b66ea7823b811691d5bf3f6f90759ca8ae904ea9275002adfdb6", 198454636),
    8: ("r9c69dd7cfc8d", "f17be1f96b471b36d6c0369b4e4e7dc1ac93b0960376c920b8b40972760c9b8f",
        "ccf7631fc9802097a27a69ccedf70790c81f960fd77f7ceb238cde41f5b2caf7", 198455476),
}
# The query-distance gate of each degree, whose A_log and dt_bias draws the input-free gate starts at.
QUERY_GATES = {4: ("r3e26412b2811", "3fa5cddfc7d5633c6f5919b30bda0a1e9eda145e04e891ee6ad9668f03210a26"),
               8: ("r9320e5cd3071", "d10d77ffe87d39f80247e6b29451c0a144831bf891cb28678009ebad3d02af3f")}
# Shared-gate run of each degree, the per-order run's control: run, recipe SHA256, seed-42 initial
# parameters (hash and count, computed on CPU from its recipe).
SHARED_GATES = {
    4: ("r0e76396bdee5", "37bd9073be22efa16f01848a92d188b93dcf9dd2c5f04b4c5d00e131c6ad9307",
        "2565fbe812dcd24557b522b51945d7fe1e9129e94cbbb29ebc4ee196532affa4", 198455056),
    8: ("rc60345dfcff7", "6861194937981022c88fd0119de12edae142ae44d3fc500db4505887d3e772b0",
        "1f3aac4e23279c616bd2456f7c16512c28098091f37a6f86c19d7adcc4140435", 198455896),
}
GATE = ("forget_gate_log_rate", "forget_gate_dt_bias")
# The settings a shared-gate run changes from its static control.
GATE_SETTINGS = {"positional_encoding": "none", "attention_forget_gate": True,
                 "attention_forget_gate_mode": "query_distance", "attention_forget_gate_form": "input",
                 "attention_forget_gate_conditioning": "none", "attention_forget_gate_rate_activation": "softplus",
                 "attention_forget_gate_per_order": False}
# Execution names and paths; the degree-four control's recipe was written under the study's earlier name.
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")


def _pinned(run: str, digest: str) -> Config:
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned recipe of {run} changed")
    return load_config(path)


def _degree(config: Config) -> int:
    if config.model.attention_degree not in CONTROLS:
        raise ValueError(f"No static control at degree {config.model.attention_degree}")
    return config.model.attention_degree


def control_config(degree: int = 4) -> Config:
    return _pinned(*CONTROLS[degree][:2])


def query_gate_config(degree: int = 4) -> Config:
    return _pinned(*QUERY_GATES[degree])


def shared_gate_config(degree: int = 4) -> Config:
    return _pinned(*SHARED_GATES[degree][:2])


def _differences(actual: dict, expected: dict) -> list[str]:
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    return sorted(f"{section}.{key}" for section in set(actual) | set(expected)
                  for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                  if actual.get(section, {}).get(key) != expected.get(section, {}).get(key))


def check_recipe(config: Config) -> None:
    """A shared-gate run differs from its static control only by GATE_SETTINGS; a per-order run from
    its shared-gate run only by the per-order switch."""
    validate_config(config)
    model = config.model
    if (model.attention_coefficient_conditioning != "static" or not model.attention_forget_gate
            or model.attention_forget_gate_conditioning != "none"):
        raise ValueError("This experiment trains learned static coefficients with the input-free gate")
    degree = _degree(config)
    actual = config_to_dict(config)
    if model.attention_forget_gate_per_order:
        expected = config_to_dict(shared_gate_config(degree))
        actual["model"]["attention_forget_gate_per_order"] = False
        name = f"shared-gate run {SHARED_GATES[degree][0]}"
    else:
        expected = config_to_dict(control_config(degree))
        for key, value in GATE_SETTINGS.items():
            if actual["model"][key] != value:
                raise ValueError(f"The shared-gate runs set model.{key}: {value}")
            actual["model"][key] = expected["model"][key]
        name = f"static control {CONTROLS[degree][0]}"
    differences = _differences(actual, expected)
    if differences:
        raise ValueError(f"Scientific settings differ from the {name}: {differences}")


def _initial_parameters(config: Config):
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    rng = jax.random.PRNGKey(config.train.seed)  # Same derivation as training/state.py create_state.
    # The blocked reference builds the same parameters as any backend.
    model = build_model(replace(config.model, attention_backend="blocked"),
                        nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    return nnx.state(model, nnx.Param)


def _check_initialized_parameters(params, config: Config) -> dict:
    """The gate arrays hold the query-distance gate's draws (per order: one draw repeated over the orders);
    without them every parameter is the static control's start, and collapsing the order axes gives the
    shared-gate run's start."""
    import jax
    import numpy as np
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    from scaffolds.transformer.training.checkpointing import state_sha256

    model, degree = config.model, _degree(config)
    per_order = model.attention_forget_gate_per_order
    control_run, _, control_sha256, control_count = CONTROLS[degree]
    shared_run, _, shared_sha256, shared_count = SHARED_GATES[degree]
    gate_count = 2 * model.num_layers * model.num_heads * ((degree + 1) if per_order else 1)
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    if count != control_count + gate_count:
        raise RuntimeError(f"Unexpected parameter count: {count}")
    actual_sha256 = state_sha256(params)
    query_gate = query_gate_config(degree)
    draws = initial_attention_parameter_values(
        [spec for spec in attention_parameter_specs(query_gate.model) if spec.name in GATE],
        num_layers=model.num_layers, num_heads=model.num_heads, key=mechanism_key(jax.random.PRNGKey(config.train.seed)))
    shape = (model.num_layers, model.num_heads, *((degree + 1,) if per_order else ()), 1)
    for name in GATE:
        values = np.asarray(params[name][...])
        if values.shape != shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise RuntimeError(f"{name} must be a finite FP32 array of shape {shape}")
        if per_order:
            if not np.array_equal(values, np.broadcast_to(values[:, :, :1], shape)):
                raise RuntimeError(f"{name} must start as one draw repeated over the orders")
            values = values[:, :, 0]
            params[name] = params[name].replace(value=params[name][...][:, :, 0])
        if not np.array_equal(values, np.asarray(draws[name])):
            raise RuntimeError(f"{name} differs from the query-distance gate's draw ({QUERY_GATES[degree][0]})")
    collapsed_sha256 = state_sha256(params) if per_order else actual_sha256
    if per_order and collapsed_sha256 != shared_sha256:
        raise RuntimeError(f"Collapsed initialization differs from the shared-gate run {shared_run}'s: {collapsed_sha256}")
    for name in GATE:
        del params[name]
    remaining_sha256 = state_sha256(params)
    if remaining_sha256 != control_sha256:
        raise RuntimeError(f"Initial parameters other than the gate differ from {control_run}'s start: {remaining_sha256}")
    return {
        "parameter_count": count, "parameter_sha256": actual_sha256, "gate_parameter_count": gate_count,
        "static_control_run_id": control_run, "static_control_parameter_sha256": control_sha256,
        "parameters_other_than_the_gate_match_static_control": True,
        "gate_draws_match_query_distance_gate_run_id": QUERY_GATES[degree][0],
        **({"shared_gate_run_id": shared_run, "collapsed_parameter_sha256": collapsed_sha256,
            "gates_repeat_one_draw_over_orders": True} if per_order else {}),
    }


def check_initialization(config: Config) -> dict:
    """The shared softmax-backbone check, then this experiment's (``_check_initialized_parameters``)."""
    check_recipe(config)
    facts = matches_softmax_backbone_initialization(config)
    return {**facts, "static_distance_gate": _check_initialized_parameters(_initial_parameters(config), config)}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_static_distance_gate_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

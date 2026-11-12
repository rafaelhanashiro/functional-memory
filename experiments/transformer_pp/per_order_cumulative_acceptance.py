"""Qualify per-order cumulative FoX gates against their pinned shared-gate controls."""
from dataclasses import fields, replace
from functools import partial
import hashlib
from pathlib import Path

import yaml

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import Config, config_to_dict
from scaffolds.transformer.configs.base import ModelConfig, validate_config
from scaffolds.transformer.configs.loader import _update_dataclass

ROOT = Path(__file__).resolve().parent
# Shared cumulative FoX control of each polynomial degree: run ID and config SHA256.
CONTROLS = {
    4: ("r4a80e28ce0a7", "cbcbe1ebbf1c370e1b82711a636fddae03e58738822e61f65fe4463f748d8763"),
    8: ("r9353f554b917", "9ff6d8e01a865f759e4a86e192533e46bed2ccd3d20c9a52927d47c2b71a2b65"),
}
# Each control's initial parameters at seed 42: degree four's recorded at its launch, degree eight's computed
# on CPU with initialization_facts on its recipe.
INITIALIZATION_SHA256 = {4: "41222e5822311b7c7828965e5f4e03b72e820cc14f0928d6af073c8649a7b7d7",
                         8: "d2f533debb5896ecf1e2d456b1e2a962de3f62934fb42ad43fd3eacca189241e"}
PARAMETER_COUNT = {4: 199585066, 8: 200339386}
# Keys a control's source branch recorded that this scaffold no longer defines, with the only value each may
# hold: degree four's declared a key-generator decay switch, which a query-only model never reads.
RETIRED_MODEL_KEYS = {"attention_coefficient_key_kernel_weight_decay": False}
# The controls' recorded data identity. Degree four read Kennel's manifests and degree eight ORCD's copies;
# a recipe may read another copy only if it hashes identically.
MANIFEST_SHA256 = {
    "train_manifest": "9de775ed836ab7c61857207335a981c09630caa3226d2dc0ee0b799cdcb2f4f4",
    "eval_manifest": "0e8615538d9dccad17956d8d9839394d05a56f8ec210dc67716db829a543634f",
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")
GATE_PARAMETERS = ("forget_gate_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")


def manifest_sha256(path) -> str | None:
    """SHA256 of a manifest copy, or None where this machine cannot read it."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def control_recipe(raw: dict, *, context: str) -> Config:
    """Load a recorded control recipe, dropping only retired keys that hold their one permitted value."""
    known = {field.name for field in fields(ModelConfig)}
    for key in sorted(set(raw["model"]) - known):
        if key not in RETIRED_MODEL_KEYS or raw["model"][key] is not RETIRED_MODEL_KEYS[key]:
            raise ValueError(f"Unexpected model.{key}: {raw['model'][key]!r} in {context}")
        del raw["model"][key]
    config = Config()
    _update_dataclass(config, raw, context=context)
    validate_config(config)
    return config


def control_config(degree: int) -> Config:
    if degree not in CONTROLS:
        raise ValueError(f"No cumulative FoX control at degree {degree}")
    run, digest = CONTROLS[degree]
    path = ROOT / "runs" / run / "config.yaml"
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != digest:
        raise ValueError(f"The pinned FoX control recipe {run} changed")
    return control_recipe(yaml.safe_load(contents), context=str(path))


def check_recipe(config: Config) -> None:
    """Only per-order gates (and the causal strips) differ from the degree's shared-gate FoX control."""
    validate_config(config)
    model = config.model
    if not model.attention_forget_gate_per_order or model.attention_forget_gate_mode != "cumulative":
        raise ValueError("This comparison requires per-order cumulative FoX gates")
    if model.attention_blocked_implementation != "strips":
        # The GPU tests qualify per-order cumulative decay through the strips at the production context.
        raise ValueError("Per-order cumulative gates train on the blocked backend's causal strips")
    control = control_config(model.attention_degree)
    run = CONTROLS[model.attention_degree][0]
    actual, expected = config_to_dict(config), config_to_dict(control)
    actual["model"]["attention_forget_gate_per_order"] = False
    # The causal strips compute the scan's arithmetic, on which the controls trained.
    actual["model"]["attention_blocked_implementation"] = expected["model"]["attention_blocked_implementation"]
    for key, digest in MANIFEST_SHA256.items():
        path, control_path = actual["data"].pop(key), expected["data"].pop(key)
        if path != control_path and manifest_sha256(path) != digest:
            raise ValueError(f"data.{key} is not a copy of the control's dataset manifest")
    # Execution locations and monitoring labels are not scientific settings.
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from FoX control {run}: {sorted(differences)}")


def _check_initialized_parameters(params, control_params, config: Config) -> dict:
    """Each order's gate is a copy of the control's shared gate, and every other parameter is the control's.

    Collapsing the duplicated order axes then recovers the control's full initialization fingerprint.
    """
    import jax
    import numpy as np
    from flax import nnx
    from scaffolds.transformer.training.checkpointing import state_sha256

    degree = config.model.attention_degree
    orders = degree + 1
    actual = {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(params)}
    expected = {path: np.asarray(value[...]) for path, value in nnx.to_flat_state(control_params)}
    if set(actual) != set(expected):
        raise RuntimeError(f"Parameter names differ from the shared-gate control: {sorted(set(actual) ^ set(expected))}")
    gates = {(name,) for name in GATE_PARAMETERS}
    if not gates <= set(expected):
        raise RuntimeError(f"The control lacks the shared gate parameters {GATE_PARAMETERS}")
    for path, value in expected.items():
        # Per-order gate arrays carry an order axis after the layer and head axes.
        wanted = (np.broadcast_to(value[:, :, None], (*value.shape[:2], orders, *value.shape[2:]))
                  if path in gates else value)
        if actual[path].shape != wanted.shape or actual[path].dtype != wanted.dtype or not np.array_equal(
                actual[path], wanted):
            raise RuntimeError(f"Initial parameter {'/'.join(map(str, path))} differs from the shared-gate control"
                               + (" in some order" if path in gates else ""))
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    gate_count = sum(expected[path].size for path in gates)
    if count != PARAMETER_COUNT[degree] + (orders - 1) * gate_count:
        raise RuntimeError(f"Unexpected per-order parameter count: {count}")
    actual_sha256 = state_sha256(params)
    for name in GATE_PARAMETERS:
        params[name] = params[name].replace(value=params[name][...][:, :, 0])
    recovered = state_sha256(params)
    if recovered != INITIALIZATION_SHA256[degree]:
        raise RuntimeError(f"Collapsed initialization differs from FoX control {CONTROLS[degree][0]}: {recovered}")
    return {"parameter_count": count, "parameter_sha256": actual_sha256,
            "control_run_id": CONTROLS[degree][0], "control_parameter_count": PARAMETER_COUNT[degree],
            "recovered_control_parameter_sha256": recovered, "added_gate_parameter_count": (orders - 1) * gate_count,
            "unchanged_parameters_match_control": True, "all_order_gates_match_shared_control": True,
            "initialization": "duplicate the shared gate across orders 0..m"}


def check_initialization(config: Config) -> dict:
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    check_recipe(config)

    def initial_parameters(model):
        rng = jax.random.PRNGKey(config.train.seed)  # Same derivation as training/state.py create_state.
        # Initialization does not depend on the attention backend.
        built = build_model(replace(model, attention_backend="blocked"),
                            nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
        return nnx.state(built, nnx.Param)

    control = control_config(config.model.attention_degree)
    return _check_initialized_parameters(initial_parameters(config.model), initial_parameters(control.model), config)


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_per_order_cumulative_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

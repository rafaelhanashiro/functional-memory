"""Qualify per-order gates against their pinned shared-gate controls."""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

ROOT = Path(__file__).resolve().parent
# Shared-gate control of each per-order variant, by polynomial degree and gate conditioning. No shared
# query-key gate run exists at degrees eight and twelve, so the per-order query-key gates there pin the
# shared query gate and also differ from it by the gate's key conditioning (a key projection starting
# at zero).
CONTROLS = {
    (4, "query"): ("r3e26412b2811", "3fa5cddfc7d5633c6f5919b30bda0a1e9eda145e04e891ee6ad9668f03210a26"),
    (4, "query_key"): ("r02f984133092", "3a9b2dbb4a22b364833ab7cbded2e82f920ca823c69f6f6e48fc409080f81131"),
    (8, "query"): ("r9320e5cd3071", "d10d77ffe87d39f80247e6b29451c0a144831bf891cb28678009ebad3d02af3f"),
    (12, "query"): ("r47bfe39ce9b4", "bd1078341f708ae52b4efaba766a6c6a98133df01aee36f539a166d2df31be3a"),
    (8, "query_key"): ("r9320e5cd3071", "d10d77ffe87d39f80247e6b29451c0a144831bf891cb28678009ebad3d02af3f"),
    (12, "query_key"): ("r47bfe39ce9b4", "bd1078341f708ae52b4efaba766a6c6a98133df01aee36f539a166d2df31be3a"),
}
# Initialization fingerprint and parameter count of each degree's query-gate control: degree four's
# verified by query_distance_decay_acceptance at launch, degree eight's and twelve's computed on CPU from
# their recipes (query_distance_degree_acceptance requires each to equal that of cumulative FoX
# r9353f554b917's recipe at the same degree).
INITIALIZATION_SHA256 = {4: "41222e5822311b7c7828965e5f4e03b72e820cc14f0928d6af073c8649a7b7d7",
                         8: "d2f533debb5896ecf1e2d456b1e2a962de3f62934fb42ad43fd3eacca189241e",
                         12: "1045f57c64e45b0534c327493c9539671a664ecaa53f85a0dc55a5298f6c5e2c"}
PARAMETER_COUNT = {4: 199585066, 8: 200339386, 12: 201093706}


def control_config(conditioning="query", degree=4):
    if (degree, conditioning) not in CONTROLS:
        raise ValueError(f"No shared-gate control for {conditioning} gates at degree {degree}")
    run, sha = CONTROLS[degree, conditioning]
    path = ROOT / "runs" / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError("The pinned shared-gate control recipe changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    if not config.model.attention_forget_gate_per_order:
        raise ValueError("This comparison requires independent per-order gates")
    conditioning = config.model.attention_forget_gate_conditioning
    control = control_config(conditioning, config.model.attention_degree)
    actual, expected = config_to_dict(config), config_to_dict(control)
    actual["model"]["attention_forget_gate_per_order"] = False
    if (expected["model"]["attention_forget_gate_conditioning"], conditioning) == ("query", "query_key"):
        actual["model"]["attention_forget_gate_conditioning"] = "query"
    # The Mosaic kernels compute the blocked reference's weights, so a blocked control's variant may
    # use them; a Mosaic control's variant must keep them.
    if expected["model"]["attention_backend"] != "mosaic":
        actual["model"]["attention_backend"] = expected["model"]["attention_backend"]
    for recipe in (actual, expected):
        for key in ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir", "wandb_mode"):
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Unrelated scientific settings differ from the shared-gate control: {sorted(differences)}")


def _check_initialized_parameters(params, config):
    import jax
    import numpy as np
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )
    from scaffolds.transformer.training.checkpointing import state_sha256

    degree = config.model.attention_degree
    baseline = control_config("query", degree)
    declared = initial_attention_parameter_values(
        attention_parameter_specs(baseline.model), num_layers=baseline.model.num_layers,
        num_heads=baseline.model.num_heads, key=mechanism_key(jax.random.PRNGKey(config.train.seed)))
    orders = degree + 1
    endpoint = config.model.attention_forget_gate_conditioning == "query_key"
    shared_gate_count = (config.model.hidden_size + 2) * config.model.num_layers * config.model.num_heads
    key_count = config.model.hidden_size * config.model.num_layers * config.model.num_heads
    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    if count != PARAMETER_COUNT[degree] + (orders - 1) * shared_gate_count + (orders * key_count if endpoint else 0):
        raise RuntimeError(f"Unexpected per-order parameter count: {count}")
    actual_sha = state_sha256(params)
    for name, expected in declared.items():
        actual = np.asarray(params[name][...])
        if name.startswith("forget_gate_"):
            wanted = np.broadcast_to(np.asarray(expected)[:, :, None, :],
                                     (*expected.shape[:2], orders, expected.shape[-1]))
        else:
            wanted = np.asarray(expected)
        if not np.array_equal(actual, wanted):
            raise RuntimeError(f"Initial parameter {name} differs from the shared control")
        if name.startswith("forget_gate_"):
            params[name] = params[name].replace(value=params[name][...][:, :, 0, :])
    if endpoint:
        key = np.asarray(params["forget_gate_key_kernel"][...])
        if key.shape != (config.model.num_layers, config.model.num_heads, orders, config.model.hidden_size) or np.any(key):
            raise RuntimeError("Every key projection must initialize at zero")
        del params["forget_gate_key_kernel"]
    recovered = state_sha256(params)
    if recovered != INITIALIZATION_SHA256[degree]:
        raise RuntimeError(f"Restored initialization differs from the control: {recovered}")
    return dict(parameter_count=count, parameter_sha256=actual_sha,
                matched_control_run_id=CONTROLS[degree, config.model.attention_forget_gate_conditioning][0],
                recovered_query_control_parameter_sha256=recovered,
                unchanged_parameters_match_control=True, all_order_gates_match_shared_control=True,
                initialization="duplicate the shared gate across orders, including order zero")


def check_initialization(config):
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    check_recipe(config)
    rng = jax.random.PRNGKey(config.train.seed)
    model = build_model(config.model, nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    return _check_initialized_parameters(nnx.state(model, nnx.Param), config)


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_per_order_decay_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None):
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

"""Qualify query-distance gates, query or query-key, on softmax and exp against their pinned no-RoPE FoX controls."""
from copy import deepcopy
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import (
    attention_spec, initialization_facts, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# The cumulative FoX control of each kernel: run ID, config SHA256, and its initial parameters' SHA256 and
# count at seed 42, computed once on CPU with initialization_facts on the control recipe. Both decay rules
# read the same gate parameters, so the query-distance variant starts from exactly these parameters.
CONTROLS = {
    "softmax": ("r5adc29a91e3b", "d2c4a2626e9828dbf124a14ab2d0900d394d2e7ac6b75ada83cfcaf34114c08b",
                "d8c5081278741619be5291ed20c77792ca7f623cb7444fdeef0149dba92f8a9f", 198628726),
    "exp": ("r183f08346ddc", "6eb0100dd7d4a8332b6c1ff5ec0c5526c99006cd3013d1d35499c12507e0689e",
            "7632f5fd650f635070b0d3044daa7a39844e1737c91a02b72002fec809e6f0ee", 198642166),
}
# A query-key gate's rate also reads the key token through w_key, one row of the hidden width per layer
# and head (15 x 14 x 896 in the controls' shape): the only parameters it adds, all zero at the start.
KEY_PROJECTION = "forget_gate_key_kernel"
KEY_PROJECTION_PARAMETERS = 188_160
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")


def kernel(config: Config) -> str:
    """Softmax normalizes the decayed exponential; exp leaves it unnormalized."""
    return "softmax" if config.model.attention_normalization else "exp"


def key_conditioned(config: Config) -> bool:
    """Each pair's rate reads both of its tokens: log alpha_ti = -exp(A) softplus(wq.x_t + wk.x_i + b)."""
    return config.model.attention_forget_gate_conditioning == "query_key"


def control_config(name: str) -> Config:
    if name not in CONTROLS:
        raise ValueError(f"No FoX control for kernel {name!r}")
    run, digest = CONTROLS[name][:2]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned FoX control recipe {run} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """Only the decay mode, optionally the gate's query-key conditioning, and the causal strips differ from the
    kernel's cumulative FoX control."""
    validate_config(config)
    model = config.model
    if model.attention_degree is not None or model.attention_forget_gate_mode != "query_distance":
        raise ValueError("This experiment requires query-distance decay of softmax or the unnormalized exponential")
    if model.attention_blocked_implementation != "strips":
        # The GPU tests qualify this decay through the strips at the production context, not the scan.
        raise ValueError("Query-distance softmax and exp train on the blocked backend's causal strips")
    run = CONTROLS[kernel(config)][0]
    actual, expected = config_to_dict(config), config_to_dict(control_config(kernel(config)))
    actual["model"]["attention_forget_gate_mode"] = "cumulative"
    if key_conditioned(config):
        # The query-key gate is the query gate plus a key projection; every other gate setting is the control's.
        actual["model"]["attention_forget_gate_conditioning"] = "query"
    # The causal strips compute the scan's arithmetic, on which the controls trained.
    actual["model"]["attention_blocked_implementation"] = expected["model"]["attention_blocked_implementation"]
    # Execution locations and monitoring labels are not scientific settings.
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from FoX control {run}: {sorted(differences)}")


def initial_parameters(config: Config):
    """The parameters training starts from, derived as initialization_facts derives them."""
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    rng = jax.random.PRNGKey(config.train.seed)  # Same derivation as training/state.py create_state.
    # Initialization does not depend on the attention backend.
    model = build_model(replace(config.model, attention_backend="blocked"),
                        nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    return nnx.state(model, nnx.Param)


def without_key_projection(params, config: Config) -> dict:
    """Count every initial parameter, require the key projection to start at exactly zero, and fingerprint the
    rest: the parameters the control also has, which must hash as the control's initialization."""
    import jax
    import numpy as np
    from scaffolds.transformer.training.checkpointing import state_sha256

    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    model = config.model
    shape = (model.num_layers, model.num_heads, model.hidden_size)
    key = np.asarray(params[KEY_PROJECTION][...]) if KEY_PROJECTION in params else None
    if key is None or key.shape != shape or np.any(key != 0):
        raise RuntimeError(f"The query-key gate's key projection {KEY_PROJECTION} {list(shape)} must start at "
                           "exactly zero")
    del params[KEY_PROJECTION]
    return {"parameter_count": count, "control_parameter_sha256": state_sha256(params)}


def check_initialization(config: Config) -> dict:
    """Every initial parameter equals the FoX control's at the same seed, except a query-key gate's key
    projection, which the control lacks and which starts at exactly zero."""
    check_recipe(config)
    run, _, digest, count = CONTROLS[kernel(config)]
    if key_conditioned(config):
        facts = without_key_projection(initial_parameters(config), config)
        if facts["control_parameter_sha256"] != digest or facts["parameter_count"] != count + KEY_PROJECTION_PARAMETERS:
            raise RuntimeError(f"Initialization without the key projection differs from FoX control {run}: {facts}")
        return {**matches_softmax_backbone_initialization(config),
                "control_run_id": run, "control_parameter_sha256": digest, "control_parameter_count": count,
                "key_projection_parameter_count": KEY_PROJECTION_PARAMETERS, "key_projection_starts_at_zero": True,
                "all_other_initial_parameters_match_control": True}
    facts = initialization_facts(deepcopy(config))
    if facts["parameter_sha256"] != digest or facts["parameter_count"] != count:
        raise RuntimeError(f"Initialization differs from FoX control {run}: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": run, "control_parameter_sha256": digest, "control_parameter_count": count,
            "all_initial_parameters_match_control": True}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests,
                   "experiments/transformer_pp/tests/test_exponential_query_distance_acceptance.py"),
                   check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None) -> None:
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

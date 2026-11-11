"""Qualify per-degree score biases against their pinned degree-eight controls.

Each run is one control's recipe with ``model.attention_score_bias`` (static, query or query_key) and
gradient-event capture under the default policy: order r = 1..8 of the tanh query polynomial reads
(s_ti + beta_r)^r. The controls are the query polynomial with RoPE and coefficient-W decay
(``r25fffc16c2f3``) and the per-order query-key gated polynomial without positional encoding
(``r227b0ed7c605``); the recipe's forget gate selects its control. The biases add b [L, H, m] at zero
(every mode), a [L, H, m] at one (query, query_key), u (query, query_key) and v (query_key) [L, H, m, D]
at zero. Nothing is drawn for them, so every other initial array is the control's and removing them
recovers the control's pinned initialization fingerprint. The runs train on the blocked backend's causal
strips (plain JAX, on B200, B300 or H100), while both controls trained on the Mosaic kernels: the recipe
must be blocked with strips, the only execution settings beyond locations and labels that differ from
the control's.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# Each backbone's control: run ID, config SHA256, and its initial parameters' SHA256 and count at seed 42,
# computed on CPU with acceptance.initialization_facts on the control recipe (the count is also the one
# its completion record holds).
CONTROLS = {
    "query": ("r25fffc16c2f3", "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2",
              "bacacd7eeee7c67dd4d5deb55ebfeef925ff4631a188d50d2e45524b6618c74d", 200_150_806),
    "per_order_query_key_gate": ("r227b0ed7c605", "799d743ade4f155bc2cae7c697812e870826bb6738ad25408ad13ddae420f1b7",
                                 "a2d5e960774132ca4fc386cdfbaf32cf130f154337bc11df057b9431573e8641", 203_541_466),
}
MODES = ("static", "query", "query_key")
# The arrays each mode adds, per layer and head: name, trailing shape ("orders" = m, "projection" = m x D),
# and exact initial value. b and a never decay; u and v take AdamW decay.
ARRAYS = {
    "score_bias_offset": ("orders", 0.0),
    "score_bias_amplitude": ("orders", 1.0),
    "score_bias_query_kernel": ("projection", 0.0),
    "score_bias_key_kernel": ("projection", 0.0),
}
MODE_ARRAYS = {"static": ("score_bias_offset",),
               "query": ("score_bias_offset", "score_bias_amplitude", "score_bias_query_kernel"),
               "query_key": tuple(ARRAYS)}
# Added parameters at the controls' shape (15 layers, 14 heads, width 896, degree 8).
ADDED_PARAMETERS = {"static": 1_680, "query": 1_508_640, "query_key": 3_013_920}
# Gradient-event capture, on with the default rule and retention cap (configs/base.py TrainConfig): after
# warmup, a finite pre-clip norm above max(1, 5 x the median of the previous 100 norms, once 20 exist)
# saves the pre-update state; at most three such states are kept.
CAPTURE_POLICY = {"gradient_event_capture": True, "gradient_event_factor": 5.0, "gradient_event_min_norm": 1.0,
                  "gradient_event_history": 100, "gradient_event_min_history": 20, "gradient_event_max_states": 3}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")
# The attention implementation, required here and the controls' own (Mosaic) in the comparison.
BACKEND = {"attention_backend": "blocked", "attention_blocked_implementation": "strips"}
# The strips are plain JAX; their GPU tests run on each of these.
GPU_TYPES = ("B200", "B300", "H100")
OWN_TESTS = "experiments/transformer_pp/tests/test_score_bias_acceptance.py"


def control_name(config: Config) -> str:
    """The backbone's control: the per-order query-key gate for a gated recipe, else the RoPE query polynomial."""
    return "per_order_query_key_gate" if config.model.attention_forget_gate else "query"


def control_config(name: str) -> Config:
    if name not in CONTROLS:
        raise ValueError(f"No score-bias control {name!r}")
    run, digest = CONTROLS[name][:2]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned control recipe {run} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """Only the score biases, gradient-event capture, the attention implementation (the blocked strips
    against the controls' Mosaic kernels), and execution locations and labels differ from the control."""
    validate_config(config)
    model = config.model
    if model.attention_score_bias not in MODES:
        raise ValueError(f"This experiment requires per-degree score biases ({', '.join(MODES)}), not "
                         f"{model.attention_score_bias!r}")
    backend = {key: getattr(model, key) for key in BACKEND}
    if backend != BACKEND:
        raise ValueError(f"Per-degree score biases train on the blocked backend's causal strips: the recipe must set "
                         f"{BACKEND}, not {backend}")
    policy = {key: getattr(config.train, key) for key in CAPTURE_POLICY}
    if policy != CAPTURE_POLICY:
        raise ValueError(f"Gradient-event capture must be on with the default policy {CAPTURE_POLICY}, not {policy}")
    run = CONTROLS[control_name(config)][0]
    actual, expected = config_to_dict(config), config_to_dict(control_config(control_name(config)))
    actual["model"]["attention_score_bias"] = "none"
    actual["model"].update({key: expected["model"][key] for key in BACKEND})
    for recipe in (actual, expected):
        for key in (*EXECUTION_KEYS, *CAPTURE_POLICY):
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Scientific settings differ from control {run}: {sorted(differences)}")


def score_bias_arrays(config: Config) -> dict[str, tuple[tuple[int, ...], float]]:
    """Each score-bias array of the recipe's mode: its full shape and exact initial value."""
    model = config.model
    heads = (model.num_layers, model.num_heads, model.attention_degree)
    shapes = {"orders": heads, "projection": (*heads, model.hidden_size)}
    return {name: (shapes[ARRAYS[name][0]], ARRAYS[name][1]) for name in MODE_ARRAYS[model.attention_score_bias]}


def initial_parameters(config: Config):
    """The parameters training starts from, derived as acceptance.initialization_facts derives them."""
    import jax
    from flax import nnx
    from scaffolds.transformer.models import build_model

    rng = jax.random.PRNGKey(config.train.seed)  # Same derivation as training/state.py create_state.
    # Initialization does not depend on the attention backend; built on the blocked backend, as
    # acceptance's initialization check builds every recipe.
    model = build_model(replace(config.model, attention_backend="blocked"),
                        nnx.Rngs(params=rng, dropout=jax.random.fold_in(rng, 0)))
    return nnx.state(model, nnx.Param)


def without_score_biases(params, config: Config) -> dict:
    """Count every initial parameter, require exactly the mode's score-bias arrays with their shapes and exact
    FP32 initial values, remove them, and fingerprint the rest: the parameters the control also has."""
    import jax
    import numpy as np
    from scaffolds.transformer.training.checkpointing import state_sha256

    count = sum(leaf.size for leaf in jax.tree.leaves(params))
    expected = score_bias_arrays(config)
    present = sorted(str(name) for name in params if str(name).startswith("score_bias_"))
    if present != sorted(expected):
        raise RuntimeError(f"{config.model.attention_score_bias} score biases add exactly {sorted(expected)}, "
                           f"not {present}")
    for name, (shape, value) in expected.items():
        array = np.asarray(params[name][...])
        if (array.shape != shape or array.dtype != np.float32
                or array.tobytes() != np.full(shape, value, np.float32).tobytes()):
            raise RuntimeError(f"{name} must start at exactly {value} with FP32 shape {list(shape)}, not "
                               f"{array.dtype} {list(array.shape)}")
        del params[name]
    return {"parameter_count": count, "control_parameter_sha256": state_sha256(params)}


def check_initialization(config: Config) -> dict:
    """Every initial parameter but the score biases equals the control's at the same seed (its pinned
    fingerprint and count); each bias array starts exactly at b = 0, a = 1, u = v = 0."""
    check_recipe(config)
    run, _, digest, count = CONTROLS[control_name(config)]
    mode = config.model.attention_score_bias
    added = ADDED_PARAMETERS[mode]
    facts = without_score_biases(initial_parameters(config), config)
    if facts["control_parameter_sha256"] != digest or facts["parameter_count"] != count + added:
        raise RuntimeError(f"Initialization without the score biases differs from control {run}: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": run, "control_parameter_sha256": digest, "control_parameter_count": count,
            "score_bias_mode": mode, "score_bias_parameter_count": added,
            "score_bias_arrays": {name: {"shape": list(shape), "initial": value}
                                  for name, (shape, value) in score_bias_arrays(config).items()},
            "all_other_initial_parameters_match_control": True}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, OWN_TESTS), check_config=check_recipe,
                   required_backend="blocked", check_initialization=check_initialization)


def require_gpu_type(argv) -> None:
    """Refuse an allocation of any GPU type but B200, B300 and H100 before acceptance starts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")  # acceptance's own default
    gpu_type = parser.parse_known_args(argv)[0].gpu_type
    if gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"Per-degree score biases train only on {', '.join(GPU_TYPES)}, where the causal strips' GPU "
                         f"tests run, not on {gpu_type}. Submit with GPU types {','.join(GPU_TYPES)}.")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

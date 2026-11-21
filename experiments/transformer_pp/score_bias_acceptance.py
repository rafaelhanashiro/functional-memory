"""Qualify per-degree score biases against their pinned degree-eight control at the 12-layer width-640 shape.

Each run is the control's recipe with ``model.attention_score_bias`` (static, query or query_key) and
gradient-event capture under the default policy: order r = 1..8 of the tanh query polynomial reads
(s_ti + beta_r)^r. The control (``r95a5691d293f``) is the query polynomial with RoPE and coefficient-W decay
at the 12-layer width-640 shape; any other difference from it, its depth, width and absent forget gate
included, is refused. The biases add b [L, H, m] at zero (every mode), a [L, H, m] at one (query,
query_key), u (query, query_key) and v (query_key) [L, H, m, D] at zero. Nothing is drawn for them, so every
other initial array is the control's and removing them recovers the control's pinned initialization
fingerprint.

The runs keep their control's attention settings and train on the Mosaic score-bias kernels, which run only
on H100.
"""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, comparison_recipe, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import Config, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# The control: run ID, config SHA256, and its initial parameters' SHA256 and count at seed 42, computed on CPU
# with acceptance.initialization_facts on the control recipe (the count is also the one its completion record
# holds).
CONTROL = ("r95a5691d293f", "e27c5364bfb1a47c6fde45598a3c31da8ad6c8c4402d29b873d76b0a7571f458",
           "72d75524ec107d4acdeb4f77d8cdea838d4c57d54f79aa9c924da4a154b03667", 99_678_056)
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
# Added parameters at the control's 12 layers, 10 heads, width 640 and degree 8.
ADDED_PARAMETERS = {"static": 960, "query": 616_320, "query_key": 1_230_720}
# Gradient-event capture, on with the default rule and retention cap (configs/base.py TrainConfig): after
# warmup, a finite pre-clip norm above max(1, 5 x the median of the previous 100 norms, once 20 exist)
# saves the pre-update state; at most three such states are kept.
CAPTURE_POLICY = {"gradient_event_capture": True, "gradient_event_factor": 5.0, "gradient_event_min_norm": 1.0,
                  "gradient_event_history": 100, "gradient_event_min_history": 20, "gradient_event_max_states": 3}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")
# The control's attention implementation, which the runs keep: the Mosaic kernels (the blocked field is inert
# there).
MOSAIC = {"attention_backend": "mosaic", "attention_blocked_implementation": "scan"}
# The Mosaic score-bias kernels, qualified on H100, refuse every other GPU before tracing a kernel.
GPU_TYPES = ("H100",)
OWN_TESTS = "experiments/transformer_pp/tests/test_score_bias_acceptance.py"


def control_config() -> Config:
    run, digest = CONTROL[:2]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned control recipe {run} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """Only the score biases, gradient-event capture, and execution locations and labels differ from the
    control."""
    validate_config(config)
    model = config.model
    if model.attention_score_bias not in MODES:
        raise ValueError(f"This experiment requires per-degree score biases ({', '.join(MODES)}), not "
                         f"{model.attention_score_bias!r}")
    run = CONTROL[0]
    backend = {key: getattr(model, key) for key in MOSAIC}
    if backend != MOSAIC:
        raise ValueError(f"Per-degree score biases against control {run} train on the Mosaic score-bias kernels "
                         f"on H100: the recipe must set {MOSAIC}, not {backend}")
    policy = {key: getattr(config.train, key) for key in CAPTURE_POLICY}
    if policy != CAPTURE_POLICY:
        raise ValueError(f"Gradient-event capture must be on with the default policy {CAPTURE_POLICY}, not {policy}")
    actual, expected = comparison_recipe(config), comparison_recipe(control_config())
    actual["model"]["attention_score_bias"] = "none"
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
    run, _, digest, count = CONTROL
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
    # The pilot must resolve the Mosaic score-bias kernels the run trains on.
    return replace(shared, cpu_tests=(*shared.cpu_tests, OWN_TESTS), check_config=check_recipe,
                   required_backend="mosaic", check_initialization=check_initialization)


def require_gpu_type(argv) -> None:
    """Refuse an allocation of a GPU type other than H100 before acceptance starts. The launcher's
    --runtime-env query allocates nothing and names no GPU type."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")  # acceptance's own default
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if args.runtime_env:
        return
    if args.gpu_type.upper() not in GPU_TYPES:
        raise ValueError(f"This recipe trains only on {', '.join(GPU_TYPES)} (the Mosaic score-bias kernels run "
                         f"only there), not on {args.gpu_type}. Submit it with GPU types {','.join(GPU_TYPES)}.")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

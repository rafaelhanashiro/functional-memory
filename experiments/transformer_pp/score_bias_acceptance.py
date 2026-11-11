"""Qualify per-degree score biases against their pinned degree-eight controls.

Each run is one control's recipe with ``model.attention_score_bias`` (static, query or query_key) and
gradient-event capture under the default policy: order r = 1..8 of the tanh query polynomial reads
(s_ti + beta_r)^r. The controls are the query polynomial with RoPE and coefficient-W decay at the
reference shape (``r25fffc16c2f3``) and at the 12-layer width-640 shape (``r95a5691d293f``), and the
per-order query-key gated polynomial without positional encoding at the reference shape
(``r227b0ed7c605``). The recipe's depth, width and forget gate select its control; a recipe that matches
none is refused. The biases add b [L, H, m] at zero (every mode), a [L, H, m] at one (query, query_key),
u (query, query_key) and v (query_key) [L, H, m, D] at zero. Nothing is drawn for them, so every other
initial array is the control's and removing them recovers the control's pinned initialization
fingerprint.

All three controls trained on the Mosaic kernels. The reference-shape runs train on the blocked backend's
causal strips (plain JAX), the only execution settings beyond locations and labels that differ from their
control's. Their hardware follows each recipe's measured update at micro-batch 16
(scaffolds/transformer/validation): every recipe trains on B200, B300 or H100 but the gated query_key one,
which fits only the Blackwell GPUs, and the gated recipes raise the allocator's share of the device. The
12-layer width-640 runs keep their control's attention settings and train on the Mosaic score-bias kernels,
which run only on H100.
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
    "query_l12_d640": ("r95a5691d293f", "e27c5364bfb1a47c6fde45598a3c31da8ad6c8c4402d29b873d76b0a7571f458",
                       "72d75524ec107d4acdeb4f77d8cdea838d4c57d54f79aa9c924da4a154b03667", 99_678_056),
}
# The recipe settings that select each control: depth, width, and forget gate.
SELECTING_FIELDS = ("model.num_layers", "model.hidden_size", "model.attention_forget_gate")
CONTROL_SHAPES = {"query": (15, 896, False), "per_order_query_key_gate": (15, 896, True),
                  "query_l12_d640": (12, 640, False)}
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
# Added parameters at each control's shape and degree 8: 15 layers, 14 heads and width 896 at the reference
# shape; 12 layers, 10 heads and width 640.
REFERENCE_SHAPE_ADDITIONS = {"static": 1_680, "query": 1_508_640, "query_key": 3_013_920}
ADDED_PARAMETERS = {"query": REFERENCE_SHAPE_ADDITIONS, "per_order_query_key_gate": REFERENCE_SHAPE_ADDITIONS,
                    "query_l12_d640": {"static": 960, "query": 616_320, "query_key": 1_230_720}}
# Gradient-event capture, on with the default rule and retention cap (configs/base.py TrainConfig): after
# warmup, a finite pre-clip norm above max(1, 5 x the median of the previous 100 norms, once 20 exist)
# saves the pre-update state; at most three such states are kept.
CAPTURE_POLICY = {"gradient_event_capture": True, "gradient_event_factor": 5.0, "gradient_event_min_norm": 1.0,
                  "gradient_event_history": 100, "gradient_event_min_history": 20, "gradient_event_max_states": 3}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir",
                  "wandb_mode")
# The attention implementation each control's runs require. The reference-shape runs train on the blocked
# strips, where both of their controls trained on the Mosaic kernels; the 12-layer width-640 runs keep their
# control's settings, the Mosaic kernels (the blocked field is inert there).
STRIPS = {"attention_backend": "blocked", "attention_blocked_implementation": "strips"}
MOSAIC = {"attention_backend": "mosaic", "attention_blocked_implementation": "scan"}
BACKENDS = {"query": STRIPS, "per_order_query_key_gate": STRIPS, "query_l12_d640": MOSAIC}
TRAINS_ON = {"blocked": "the blocked backend's causal strips", "mosaic": "the Mosaic score-bias kernels on H100"}
# The strips are plain JAX; their GPU tests run on each of these.
GPU_TYPES = ("B200", "B300", "H100")
# The Mosaic score-bias kernels, qualified on H100, refuse every other GPU before tracing a kernel.
MOSAIC_GPU_TYPES = ("H100",)
# Measured allocator peaks of the full recipes' updates at micro-batch 16 on one GPU (validation record): RoPE
# 32-49 GiB; gated static and query 56-67 GiB; gated query_key 147 GiB, beyond an 80 GB H100 at any allocator
# share, so it trains only on the Blackwell GPUs (B200 180 GB, B300 288 GB). XLA's allocator may take 0.75 of
# the device by default (59.4 GiB on an H100, 133.8 GiB on a B200); the gated recipes take 0.95 (75.2 and 169.4
# GiB), which the gated query recipe needs on an H100 and the gated query_key recipe on a B200.
BLACKWELL_GPU_TYPES = ("B200", "B300")
GATED_MEMORY_FRACTION = "0.95"
OWN_TESTS = "experiments/transformer_pp/tests/test_score_bias_acceptance.py"


def control_name(config: Config) -> str:
    """The control at the recipe's depth, width and forget gate; refuse a recipe that matches none."""
    model = config.model
    selected = (model.num_layers, model.hidden_size, model.attention_forget_gate)
    for name, shape in CONTROL_SHAPES.items():
        if shape == selected:
            return name
    controls = "; ".join(f"{CONTROLS[name][0]} {list(shape)}" for name, shape in CONTROL_SHAPES.items())
    raise ValueError(f"No score-bias control has this recipe's {list(SELECTING_FIELDS)} = {list(selected)}; "
                     f"the controls have {controls}")


def control_config(name: str) -> Config:
    if name not in CONTROLS:
        raise ValueError(f"No score-bias control {name!r}")
    run, digest = CONTROLS[name][:2]
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"The pinned control recipe {run} changed")
    return load_config(path)


def check_recipe(config: Config) -> None:
    """Only the score biases, gradient-event capture, execution locations and labels, and (against the
    reference-shape controls' Mosaic kernels) the blocked strips differ from the control."""
    validate_config(config)
    model = config.model
    if model.attention_score_bias not in MODES:
        raise ValueError(f"This experiment requires per-degree score biases ({', '.join(MODES)}), not "
                         f"{model.attention_score_bias!r}")
    name = control_name(config)
    run, required = CONTROLS[name][0], BACKENDS[name]
    backend = {key: getattr(model, key) for key in required}
    if backend != required:
        raise ValueError(f"Per-degree score biases against control {run} train on "
                         f"{TRAINS_ON[required['attention_backend']]}: the recipe must set {required}, not {backend}")
    policy = {key: getattr(config.train, key) for key in CAPTURE_POLICY}
    if policy != CAPTURE_POLICY:
        raise ValueError(f"Gradient-event capture must be on with the default policy {CAPTURE_POLICY}, not {policy}")
    actual, expected = config_to_dict(config), config_to_dict(control_config(name))
    actual["model"]["attention_score_bias"] = "none"
    actual["model"].update({key: expected["model"][key] for key in required})
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
    name = control_name(config)
    run, _, digest, count = CONTROLS[name]
    mode = config.model.attention_score_bias
    added = ADDED_PARAMETERS[name][mode]
    facts = without_score_biases(initial_parameters(config), config)
    if facts["control_parameter_sha256"] != digest or facts["parameter_count"] != count + added:
        raise RuntimeError(f"Initialization without the score biases differs from control {run}: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": run, "control_parameter_sha256": digest, "control_parameter_count": count,
            "score_bias_mode": mode, "score_bias_parameter_count": added,
            "score_bias_arrays": {name: {"shape": list(shape), "initial": value}
                                  for name, (shape, value) in score_bias_arrays(config).items()},
            "all_other_initial_parameters_match_control": True}


def gpu_types(config: Config) -> tuple[str, ...]:
    """The GPU types the recipe trains on: H100 alone on the Mosaic score-bias kernels; on the strips B200, B300
    and H100, but only the Blackwell GPUs for the gated query_key recipe, whose update needs 147 GiB."""
    if config.model.attention_backend == "mosaic":
        return MOSAIC_GPU_TYPES
    if config.model.attention_forget_gate and config.model.attention_score_bias == "query_key":
        return BLACKWELL_GPU_TYPES
    return GPU_TYPES


def runtime_env(config: Config) -> dict[str, str]:
    """The allocator's share of the device: 0.95 for the gated recipes, the default (0.75) for the RoPE ones."""
    return {"XLA_PYTHON_CLIENT_MEM_FRACTION": GATED_MEMORY_FRACTION} if config.model.attention_forget_gate else {}


def experiment_spec(config: Config, *, entry_module: str):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    # The pilot must resolve the recipe's own backend: the blocked strips or the Mosaic score-bias kernels. The
    # launcher exports runtime_env for acceptance and training, and acceptance records it in its identity.
    return replace(shared, cpu_tests=(*shared.cpu_tests, OWN_TESTS), check_config=check_recipe,
                   required_backend=BACKENDS[control_name(config)]["attention_backend"],
                   check_initialization=check_initialization,
                   runtime_env={**shared.runtime_env, **runtime_env(config)})


def require_gpu_type(argv) -> None:
    """Refuse an allocation of a GPU type the recipe does not train on before acceptance starts. The
    launcher's --runtime-env query allocates nothing and names no GPU type."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config", nargs="?", type=Path)
    parser.add_argument("--gpu-type", default="H100")  # acceptance's own default
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if args.runtime_env:
        return
    allowed = GPU_TYPES if args.config is None else gpu_types(load_config(args.config))
    if args.gpu_type.upper() not in allowed:
        reason = ("its update needs 147 GiB at micro-batch 16, more than an 80 GB H100 holds"
                  if allowed == BLACKWELL_GPU_TYPES else
                  "the Mosaic score-bias kernels run only there" if allowed == MOSAIC_GPU_TYPES else
                  "the causal strips' GPU tests run there")
        raise ValueError(f"This recipe trains only on {', '.join(allowed)} ({reason}), not on {args.gpu_type}. "
                         f"Submit it with GPU types {','.join(allowed)}.")


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else list(argv)
    require_gpu_type(argv)
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

"""Qualify degree-eight polynomials with structurally absent constants, on softplus or raw scores.

Direct query/static polynomials omit c0. Query SoS omits U0 and V0 before
square expansion and has no input prefactor. Surviving parameters preserve their
pinned parents' initialization, including the parent's full W draw before slicing.
Every production allocation requires the shared full-recipe pilot and resume.
"""
from dataclasses import replace
from functools import partial
import argparse
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, main as run_acceptance, matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config


RUNS = Path(__file__).resolve().parent / "runs"
CONTROLS = {
    ("query", "direct"): {
        "run_id": "ra3b54e2489f8",
        "config_sha256": "adcdb8b7e740ccba18fb7f2bb58fa61fbfdbd14b8012af7d340084b9230387e8",
    },
    ("static", "direct"): {
        "run_id": "r9c69dd7cfc8d",
        "config_sha256": "f17be1f96b471b36d6c0369b4e4e7dc1ac93b0960376c920b8b40972760c9b8f",
    },
    ("query", "sum_of_squares"): {
        "run_id": "r41a4c74c5f58",
        "config_sha256": "851c389a0e00d5e595c5c02b30e09255249412fc3d4f9d11a5ba0551e70a7110",
    },
}
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
CAPTURE_POLICY = {
    "gradient_event_capture": True, "gradient_event_factor": 5.0,
    "gradient_event_min_norm": 1.0, "gradient_event_history": 100,
    "gradient_event_min_history": 20, "gradient_event_max_states": 3,
}
CPU_TESTS = ("scaffolds/transformer/tests/test_softplus_zero_constant.py",
             "experiments/transformer_pp/tests/test_softplus_zero_constant_acceptance.py")
GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_softplus_zero_constant.py",
             "scaffolds/transformer/tests/test_mosaic_query.py")
# Raw scores (the identity transform) train on the plain Mosaic query kernels with c_0 = 0 (or U_0 = V_0 = 0).
RAW_GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_query.py",)
TRANSFORMS = ("softplus", "identity")
COMPACT_PARAMETERS = frozenset(("polynomial_coefficients", "query_polynomial_kernel",
                               "query_polynomial_amplitude", "query_polynomial_bias"))


def control_key(config):
    key = (config.model.attention_coefficient_conditioning,
           config.model.attention_coefficient_parameterization)
    if key not in CONTROLS:
        raise ValueError(f"No zero-constant parent for conditioning/parameterization {key!r}")
    return key


def control_config(key):
    if key not in CONTROLS:
        raise ValueError(f"No zero-constant parent {key!r}")
    control = CONTROLS[key]
    path = RUNS / control["run_id"] / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != control["config_sha256"]:
        raise ValueError(f"Pinned zero-constant parent recipe {control['run_id']} changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    key = control_key(config)
    transform = config.model.attention_score_transform
    if transform not in TRANSFORMS:
        raise ValueError(f"Zero-constant runs read softplus or raw scores, not {transform!r}")
    actual, expected = config_to_dict(config), config_to_dict(control_config(key))
    expected["model"].update(attention_coefficient_zero_constant=True,
                             attention_score_transform=transform)
    expected["train"].update(CAPTURE_POLICY)
    for recipe in (actual, expected):
        for name in EXECUTION_KEYS:
            recipe["train"].pop(name, None)
    if actual != expected:
        differences = [f"{section}.{name}" for section in set(actual) | set(expected)
                       for name in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(name) != expected.get(section, {}).get(name)]
        raise ValueError(f"Settings differ from pinned zero-constant recipe {CONTROLS[key]['run_id']}: "
                         f"{sorted(differences)}")


def check_surviving_initialization(config, parent):
    """Audit declared mechanism arrays independently of the compact allocation logic."""
    import jax
    import numpy as np
    from scaffolds.transformer.attention.parameters import (
        attention_parameter_specs, initial_attention_parameter_values, mechanism_key,
    )

    # The indices name the original degree-eight coordinate layout, not an
    # expanded polynomial restriction: SoS removes U0 and V0 before squaring.
    indices = ((1, 2, 3, 4, 6, 7, 8) if config.model.attention_coefficient_parameterization == "sum_of_squares"
               else tuple(range(1, 9)))

    def initial(current):
        return initial_attention_parameter_values(
            attention_parameter_specs(current.model), num_layers=current.model.num_layers,
            num_heads=current.model.num_heads, key=mechanism_key(jax.random.PRNGKey(current.train.seed)))

    actual, original = initial(config), initial(parent)
    if set(actual) != set(original):
        raise RuntimeError("Zero-constant restriction changed mechanism parameter names")
    removed = 0
    for name, value in original.items():
        expected = np.take(np.asarray(value), indices, axis=-1) if name in COMPACT_PARAMETERS else np.asarray(value)
        observed = np.asarray(actual[name])
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"Surviving initial parameter {name} differs from its parent coordinates")
        removed += value.size - observed.size
    return {"surviving_parent_coordinates": list(indices), "removed_parameter_count": removed,
            "all_surviving_mechanism_parameters_match_parent": True,
            "parent_projection_draw_preserved": config.model.attention_coefficient_conditioning == "query"}


def check_initialization(config):
    check_recipe(config)
    key = control_key(config)
    control = CONTROLS[key]
    return {**matches_softmax_backbone_initialization(config),
            **check_surviving_initialization(config, control_config(key)),
            "control_run_id": control["run_id"], "control_config_sha256": control["config_sha256"],
            "score_transform": config.model.attention_score_transform, "polynomial_degree": 8, "input_power": 0,
            "constant_policy": "U0=V0=0" if key[1] == "sum_of_squares" else "c0=0",
            "initial_kernel": ("(U_T8({x})-U_T8(0))^2+(V_T8({x})-V_T8(0))^2"
                               if key[1] == "sum_of_squares" else "T_8({x})-1").format(
                x="z" if config.model.attention_score_transform == "softplus" else "s")}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    gpu_tests = GPU_TESTS if config.model.attention_score_transform == "softplus" else RAW_GPU_TESTS
    return replace(shared, cpu_tests=(*shared.cpu_tests, *CPU_TESTS),
                   gpu_tests=(*shared.gpu_tests, *gpu_tests), required_backend="mosaic",
                   check_config=check_recipe, check_initialization=check_initialization)


def _raw_scores(argv) -> bool:
    """Whether the recipe named on the command line reads raw scores (the identity transform)."""
    recipes = [Path(argument) for argument in argv if argument.endswith(".yaml") and Path(argument).is_file()]
    return bool(recipes) and load_config(recipes[0]).model.attention_score_transform == "identity"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    gpu = args.gpu_type.upper()
    if not args.runtime_env and gpu not in ("H100", "B200") and not (gpu == "B300" and _raw_scores(argv)):
        raise ValueError("These Mosaic softplus runs qualify and train on H100 or B200 only; raw-score runs, on "
                         "the plain Mosaic query kernels, also on B300")
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

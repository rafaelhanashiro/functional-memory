"""Qualify softplus composed with query SoS, optionally multiplied by its input.

The SoS core has degree eight or twelve. The actual polynomial has degree m
for h(z), m+1 for z*h(z), z=softplus(s). Each run starts from a completed
raw-score query-SoS control's exact parameters: h=T_m. The prefactor changes no learned parameter and is
represented by a structural zero prepended to the expanded coefficients. Mosaic
evaluates the resulting polynomial and differentiates through the full composition.

Prefactor runs, W = f(s) h_t(g(s)) with input power zero, take a parameter-free prefactor f of the raw score
(softplus, sigmoid, exp or swish) and g the identity or softplus, at the degree-eight reference control
(reeefbcc19abf): its recipe with g, f and gradient-event capture, and its exact initial parameters. Their
comparison is the input power's z h_t(z) (r841f35a24241), which is softplus(s) h_t(softplus(s)) itself, so that
pair is refused here.

The recipe's SoS core degree, depth, width and update count select its control: the degree-eight and
degree-twelve controls at the 15-layer width-896 reference shape, and the degree-twelve controls at the 12-layer
width-640 shape with twice and four times its 1x budget. No raw-score SoS run of core degree twelve exists at that
shape's 1x budget, so a 1x recipe is the 2x control's with derive_recipe's 1x schedule
(``derive_recipe.py --budget 1``) and starts from the 2x control's parameters: initialization does not depend on
the budget.
"""
from dataclasses import replace
from functools import partial
import argparse
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import (
    attention_spec, comparison_recipe, initialization_facts, main as run_acceptance,
    matches_softmax_backbone_initialization,
)
from scaffolds.transformer.configs import load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
# derive_recipe's 1x schedule at the 12-layer width-640 shape, batch 128 unchanged, in place of the 2x control's
# budget-derived fields: 7,551 updates, warmup 454, validation every 100 updates through 1,500 and every 500 after.
L12_D640_1X_SCHEDULE = {"optim": {"total_steps": 7551, "warmup_steps": 454},
                        "train": {"eval_early_every": 100, "eval_early_until": 1500, "eval_every": 500}}
# Each completed raw-score SoS control, selected by the recipe's (SoS core degree, depth, width, update count): its
# recipe SHA256 and its initial parameters' SHA256 and count at seed 42, computed on CPU with
# acceptance.initialization_facts on the control recipe (the count is also the one its completion holds). The
# prefactor must not add any learned parameter. The 12-layer width-640 controls share one fingerprint and count.
CONTROLS = {
    (8, 15, 896, 7570): {
        "run_id": "reeefbcc19abf",
        "config_sha256": "7d3fca30ff8fb22520f5bdb6df9bffc3a962b2dc5ab539436a17b4d2b0e3d326",
        "parameter_sha256": "bacacd7eeee7c67dd4d5deb55ebfeef925ff4631a188d50d2e45524b6618c74d",
        "parameter_count": 200_150_806,
    },
    (12, 15, 896, 7570): {
        "run_id": "r3d52b7ab43d1",
        "config_sha256": "9b76b2bf414442b851519a08f5bcf193598d96381f055bb1de31dbd238f6c0fa",
        "parameter_sha256": "9ece005626c3c8ecc0a53f69547fc9f393b1fee1b0dd83b71a0d20f38471ce16",
        "parameter_count": 200_905_126,
    },
    (12, 12, 640, 7551): {
        "run_id": "r08be7093202a",
        "config_sha256": "4683fd9da0993fd319e38e593a4dedafdf9cbe73d26cedf38795e1fa518c2a3f",
        "parameter_sha256": "c8a7ed88e12ea4c33806ee8a701e0c75655fc86f176790db6e07b189948e89b9",
        "parameter_count": 99_986_216,
        "schedule": L12_D640_1X_SCHEDULE,
    },
    (12, 12, 640, 15103): {
        "run_id": "r08be7093202a",
        "config_sha256": "4683fd9da0993fd319e38e593a4dedafdf9cbe73d26cedf38795e1fa518c2a3f",
        "parameter_sha256": "c8a7ed88e12ea4c33806ee8a701e0c75655fc86f176790db6e07b189948e89b9",
        "parameter_count": 99_986_216,
    },
    (12, 12, 640, 30205): {
        "run_id": "r1f935f18fe21",
        "config_sha256": "8351d24f33e1c4f6a54983fe7a069c76911d7fdf6db35e537ed122aaf7a51cec",
        "parameter_sha256": "c8a7ed88e12ea4c33806ee8a701e0c75655fc86f176790db6e07b189948e89b9",
        "parameter_count": 99_986_216,
    },
}
SELECTING_FIELDS = ("SoS core degree", "model.num_layers", "model.hidden_size", "optim.total_steps")
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
CAPTURE_POLICY = {
    "gradient_event_capture": True, "gradient_event_factor": 5.0,
    "gradient_event_min_norm": 1.0, "gradient_event_history": 100,
    "gradient_event_min_history": 20, "gradient_event_max_states": 3,
}
CPU_TESTS = ("scaffolds/transformer/tests/test_softplus_sos.py",
             "experiments/transformer_pp/tests/test_softplus_sos_acceptance.py")
GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_score_transform.py",
             "scaffolds/transformer/tests/test_mosaic_softplus_sos.py")
# Prefactor runs: at this control, input power zero, these prefactors f and transforms g (the pair f = g = softplus
# is the input power's kernel, run r841f35a24241). Their own CPU and GPU suites.
PREFACTOR_CONTROL = (8, 15, 896, 7570)
PREFACTORS = ("softplus", "sigmoid", "exp", "swish")
PREFACTOR_TRANSFORMS = ("identity", "softplus")
INPUT_POWER_RUN = "r841f35a24241"
PREFACTOR_CPU_TESTS = ("scaffolds/transformer/tests/test_score_prefactor.py",)
PREFACTOR_GPU_TESTS = ("scaffolds/transformer/tests/test_mosaic_score_prefactor.py",)


def control_key(config):
    """The recipe's (SoS core degree, depth, width, update count); refuse one that selects no control."""
    model = config.model
    key = (model.attention_degree - model.attention_coefficient_input_power, model.num_layers, model.hidden_size,
           config.optim.total_steps)
    if key not in CONTROLS:
        known = "; ".join(f"{control['run_id']} {list(selected)}" for selected, control in CONTROLS.items())
        raise ValueError(f"No raw-score SoS control has this recipe's {list(SELECTING_FIELDS)} = {list(key)}; "
                         f"the controls have {known}")
    return key


def control_config(key):
    if key not in CONTROLS:
        raise ValueError(f"No raw-score SoS control {key!r}")
    control = CONTROLS[key]
    path = RUNS / control["run_id"] / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != control["config_sha256"]:
        raise ValueError(f"Pinned SoS control recipe {control['run_id']} changed")
    return load_config(path)


def control_recipe(key):
    """The control's recipe at the selected budget: its own, or the 2x control's with the derived 1x schedule."""
    recipe = comparison_recipe(control_config(key))
    for section, values in CONTROLS[key].get("schedule", {}).items():
        recipe[section].update(values)
    return recipe


def check_prefactor(config):
    """A prefactor run's composition: input power zero at the degree-eight reference control, f in PREFACTORS and
    g in PREFACTOR_TRANSFORMS, never f = g = softplus (the input power's z h_t(z), run INPUT_POWER_RUN)."""
    model = config.model
    prefactor, transform = model.attention_score_prefactor, model.attention_score_transform
    if prefactor == "none":
        return
    if prefactor not in PREFACTORS or transform not in PREFACTOR_TRANSFORMS:
        raise ValueError(f"Prefactor runs take 'model.attention_score_prefactor' in {PREFACTORS} and "
                         f"'model.attention_score_transform' in {PREFACTOR_TRANSFORMS}, not {prefactor!r} and "
                         f"{transform!r}")
    if model.attention_coefficient_input_power or control_key(config) != PREFACTOR_CONTROL:
        raise ValueError(f"Prefactor runs are the degree-eight reference control {CONTROLS[PREFACTOR_CONTROL]['run_id']} "
                         f"{list(PREFACTOR_CONTROL)} with input power zero")
    if (transform, prefactor) == ("softplus", "softplus"):
        raise ValueError(f"softplus(s) h_t(softplus(s)) is the input power's z h_t(z): run {INPUT_POWER_RUN}")


def check_recipe(config):
    validate_config(config)
    power = config.model.attention_coefficient_input_power
    if (config.model.attention_degree, power) not in ((8, 0), (9, 1), (12, 0), (13, 1)):
        raise ValueError("Runs require a degree-eight or degree-twelve SoS core: "
                         "(total degree, input power) = (8,0), (9,1), (12,0) or (13,1)")
    selected = control_key(config)
    check_prefactor(config)
    prefactor = config.model.attention_score_prefactor
    core_degree, control = selected[0], CONTROLS[selected]
    actual, expected = comparison_recipe(config), control_recipe(selected)
    expected["model"].update(attention_score_transform=("softplus" if prefactor == "none"
                                                        else config.model.attention_score_transform),
                             attention_score_prefactor=prefactor,
                             attention_degree=core_degree + power,
                             attention_coefficient_input_power=power)
    expected["train"].update(CAPTURE_POLICY)
    for recipe in (actual, expected):
        for key in EXECUTION_KEYS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Unrelated settings differ from SoS control {control['run_id']}: {sorted(differences)}")


def check_initialization(config):
    check_recipe(config)
    selected = control_key(config)
    core_degree, control = selected[0], CONTROLS[selected]
    # Parameter initialization is backend independent; use the host reference.
    facts = initialization_facts(replace(config, model=replace(config.model, attention_backend="blocked")))
    expected = {key: control[key] for key in ("parameter_count", "parameter_sha256")}
    if facts != expected:
        raise RuntimeError(f"Initial parameters differ from the degree-{core_degree} SoS control "
                           f"{control['run_id']}: {facts}")
    return {**matches_softmax_backbone_initialization(config),
            "control_run_id": control["run_id"], "control_parameter_sha256": control["parameter_sha256"],
            "control_parameter_count": control["parameter_count"],
            "control_selection": dict(zip(SELECTING_FIELDS, selected)),
            "derived_1x_schedule": "schedule" in control,
            "sos_core_degree": core_degree, "polynomial_degree": config.model.attention_degree,
            "input_power": config.model.attention_coefficient_input_power,
            "initial_kernel": initial_kernel(config, core_degree),
            "score_transform": config.model.attention_score_transform,
            "score_prefactor": config.model.attention_score_prefactor,
            "all_initial_parameters_match_control": True}


def initial_kernel(config, core_degree):
    """The kernel the run starts at: z^power T_m(z) with z = softplus(s), or f(s) T_m(g(s)) with a prefactor."""
    model = config.model
    if model.attention_score_prefactor == "none":
        return ("z*" if model.attention_coefficient_input_power else "") + f"T_{core_degree}(z)"
    argument = "s" if model.attention_score_transform == "identity" else "z"
    return f"{model.attention_score_prefactor}(s)*T_{core_degree}({argument})"


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    prefactor = config.model.attention_score_prefactor != "none"
    return replace(shared, cpu_tests=(*shared.cpu_tests, *CPU_TESTS, *(PREFACTOR_CPU_TESTS if prefactor else ())),
                   gpu_tests=(*shared.gpu_tests, *GPU_TESTS, *(PREFACTOR_GPU_TESTS if prefactor else ())),
                   required_backend="mosaic", check_config=check_recipe, check_initialization=check_initialization)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")
    parser.add_argument("--runtime-env", action="store_true")
    args = parser.parse_known_args(argv)[0]
    if not args.runtime_env and args.gpu_type.upper() not in ("H100", "B200"):
        raise ValueError("These Mosaic softplus-SoS runs qualify and train on H100 or B200 only")
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

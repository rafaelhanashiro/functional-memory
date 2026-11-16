"""Qualify static query-key-gate, sum-of-squares, and Hermite-start runs against their pinned partner recipes."""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config

RUNS = Path(__file__).resolve().parent / "runs"
TESTS = "scaffolds/transformer/tests/test_query_coefficient_forms.py"
# Each form's partner recipe (run, SHA256) and the model fields it changes. The static gate
# runs change the partner's degree too: query-key m = 8 is the m = 4 recipe at degree 8.
PARTNERS = {"static_gate": ("r03be769d2069", "c48468ebbf4691922df582e347d7adc7fbcb11be951b668f31f8a3175aec265c"), "sum_of_squares": ("r25fffc16c2f3", "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2"),
            "hermite": ("r25fffc16c2f3", "ffa72e4d99f016bedfd07b7b858199734d71596b3af18a325813bd810e3620b2")}
CHANGES = {
    "static_gate": dict(attention_coefficient_conditioning="static", attention_coefficient_activation="identity",
                        attention_coefficient_kernel_weight_decay=False, attention_coefficient_amplitude_initial=1.0,
                        attention_coefficient_projection_initial_std=0.0, attention_coefficient_amplitude_scale="none"),
    "sum_of_squares": dict(attention_coefficient_parameterization="sum_of_squares", attention_coefficient_scale="none",
                           attention_coefficient_amplitude_scale="none"),
    "hermite": dict(attention_coefficient_initialization="hermite"),
}
# A sum-of-squares run at degree four or twelve copies the query run of that degree. Degree sixteen,
# beyond the Mosaic kernels, copies the degree-twelve sum-of-squares run on the blocked backend's strips.
DEGREE_PARTNERS = {("sum_of_squares", 4): ("rca97cc9db8f2", "75fa1e3e30eaf98aa49d5b4ca833860a2664877e5892ecca11c008eff1978f58"),
                   ("sum_of_squares", 16): ("r3d52b7ab43d1", "9b76b2bf414442b851519a08f5bcf193598d96381f055bb1de31dbd238f6c0fa"),
                   ("sum_of_squares", 12): ("r0fceea0627c8", "785e3530949ee2733addde255b679ecf0d295d188fd19fd9ca29ebd9d5a7d2e0")}
DEGREE_CHANGES = {("sum_of_squares", 16): dict(attention_degree=16, attention_backend="blocked",
                                              attention_blocked_implementation="strips")}
# At the 12-layer width-640 shape with twice or four times its 1x budget (depth, width, updates), a sum-of-squares
# run copies the query run of its degree at that shape and budget.
SHAPE_PARTNERS = {("sum_of_squares", 8, 12, 640, 15103): ("r3759b2a57a54", "aa7585f1a530ab4508cdd033d2243af900ae0316950c4b0ab8b46f21371534fb"),
                  ("sum_of_squares", 12, 12, 640, 15103): ("r0f6d929e5992", "c4f33f9ad753a7586bf1e1746590ee97d4c2fdf4fc5f5d96bf88e1ee2b89ef1c"),
                  ("sum_of_squares", 12, 12, 640, 30205): ("r2f41d054e944", "b600516341a2c3236d8342f8a834ffdc2bfb97fd2c517f7da9c4240cb4dc4c3b")}
PATH_FIELDS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name", "wandb_group", "wandb_dir", "wandb_mode")


def form(config):
    model = config.model
    if model.attention_coefficient_parameterization == "sum_of_squares":
        return "sum_of_squares"
    if model.attention_coefficient_initialization == "hermite":
        return "hermite"
    if model.attention_coefficient_conditioning == "static" and model.attention_forget_gate_conditioning == "query_key":
        return "static_gate"
    raise ValueError("This adapter covers static query-key gates, sum-of-squares, and Hermite starts only")


def partner_config(name, degree=None, shape=None):
    """The pinned partner of a form at a degree and, at another shape and budget, (depth, width, updates)."""
    key = (name, degree, *shape) if shape else None
    run, sha = SHAPE_PARTNERS[key] if key in SHAPE_PARTNERS else DEGREE_PARTNERS.get((name, degree), PARTNERS[name])
    path = RUNS / run / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError(f"The pinned partner recipe {run} changed")
    return load_config(path)


def check_recipe(config):
    validate_config(config)
    name = form(config)
    shape = (config.model.num_layers, config.model.hidden_size, config.optim.total_steps)
    expected = config_to_dict(replace(partner_config(name, config.model.attention_degree, shape)))
    expected["model"].update(CHANGES[name])
    expected["model"].update(DEGREE_CHANGES.get((name, config.model.attention_degree), {}))
    actual = config_to_dict(config)
    if name == "static_gate":
        if config.model.attention_degree not in (4, 8):
            raise ValueError("The static query-key gate runs are degree 4 and 8")
        expected["model"]["attention_degree"] = config.model.attention_degree
    for recipe in (actual, expected):
        for key in PATH_FIELDS:
            recipe["train"].pop(key, None)
    if actual != expected:
        differences = [f"{section}.{key}" for section in set(actual) | set(expected)
                       for key in set(actual.get(section, {})) | set(expected.get(section, {}))
                       if actual.get(section, {}).get(key) != expected.get(section, {}).get(key)]
        raise ValueError(f"Unrelated scientific settings differ from the partner: {sorted(differences)}")


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, cpu_tests=(*shared.cpu_tests, TESTS), check_config=check_recipe)


def main(argv=None):
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

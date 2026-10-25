"""Qualify scalar RBF attention on Rafael's executed 200M / 4B recipe."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "experiments/transformer_200m_4b/runs/r83139ea7389c/config.yaml"
REFERENCE_SHA256 = "9bcd49b8acb9b34aed37c21e51fe201d7b3e1859b1c7ffe0ddefc6881587726d"
REFERENCE_COMMIT = "36b074480751897c8b8f9754cbc13c22594024be"
# From the executed reference's CPU initialization-result.json acceptance audit.
REFERENCE_BACKBONE_SHA256 = "bb6ab6df9c4f8b8f3df92843890b150eee2c34dd7cf8be2ba2cb8c663c67d13b"
FIT = ROOT / "experiments/attention_scaling/runs/r6e1b71e49350/config.yaml"
TESTS = "scaffolds/transformer/tests/test_basis_projection_scale.py"
MODEL_CHANGES = {
    "attention_backend", "attention_basis", "attention_coefficients",
    "attention_coefficient_sharing", "attention_coefficient_initialization",
    "attention_coefficient_scale", "attention_coefficient_amplitude_scale",
    "attention_coefficient_conditioning", "attention_coefficient_activation",
    "attention_coefficient_amplitude_initial", "attention_coefficient_projection_initial_std",
}
RUN_IDENTITY = {"workdir", "checkpoint_dir", "wandb_run_name", "wandb_group", "wandb_dir", "wandb_mode", "source_commit"}


def reference_config():
    if hashlib.sha256(REFERENCE.read_bytes()).hexdigest() != REFERENCE_SHA256:
        raise ValueError("The executed Rafael reference recipe changed")
    return load_config(REFERENCE)


def recipe(mode):
    """Change only the scalar attention family and its coefficient parameterization."""
    if mode not in ("fixed", "static", "query"):
        raise ValueError(f"Unknown RBF comparison mode: {mode}")
    config = reference_config()
    m = config.model
    m.attention_backend = "triton_scalar"
    m.attention_basis = load_config(FIT).model.attention_basis
    if mode != "fixed":
        m.attention_coefficients = "learned"
        m.attention_coefficient_sharing = "per_head"
        m.attention_coefficient_initialization = "projection"
        m.attention_coefficient_scale = "projection"
    if mode == "query":
        m.attention_coefficient_conditioning = "query"
        m.attention_coefficient_activation = "tanh"
        m.attention_coefficient_amplitude_scale = "projection"
        m.attention_coefficient_amplitude_initial = 0.0
        m.attention_coefficient_projection_initial_std = 1 / math.sqrt(m.hidden_size)
    return config


def check_config(config):
    m = config.model
    mode = "fixed" if m.attention_coefficients == "fixed" else m.attention_coefficient_conditioning
    expected, actual = config_to_dict(recipe(mode)), config_to_dict(config)
    for section in expected:
        for name, value in expected[section].items():
            if section == "train" and name in RUN_IDENTITY:
                continue
            if actual[section][name] != value:
                raise ValueError(f"Matched Rafael recipe differs at {section}.{name}")


def spec(config):
    check_config(config)
    shared = attention_spec(config, entry_module=__name__)

    def check_initialization(value):
        facts = shared.check_initialization(value)
        if facts["softmax_counterpart_parameter_sha256"] != REFERENCE_BACKBONE_SHA256:
            raise ValueError("Backbone initialization differs from Rafael's executed reference audit")
        return facts

    def select_tests(values, backend, cpu):
        paths = list(shared.select_tests(values, backend, cpu))
        if cpu:
            paths.extend((TESTS, "experiments/attention_scaling/tests/test_rbf_matched.py",
                          "scaffolds/transformer/tests/test_qk_rms_norm.py",
                          "scaffolds/transformer/tests/test_normalization.py"))
        else:
            model = values["model"]
            mode = "fixed" if model["attention_coefficients"] == "fixed" else model["attention_coefficient_conditioning"]
            paths.append(f"{TESTS}::test_full_context_rbf_qk_rms[{mode}]")
        return tuple(paths)

    return replace(shared, check_config=check_config, check_initialization=check_initialization,
                   select_tests=select_tests)


def main(argv=None):
    run_acceptance(spec, argv)


if __name__ == "__main__":
    main()

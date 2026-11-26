"""Compare exp(s) h_t(softplus(s)) to the same-rate softplus(s) h_t(softplus(s)) run at key width four."""
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path

from experiments.transformer_pp.shifted_sos_acceptance import EXECUTION_KEYS
from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config

CONTROLS = {
    0.001: ("r2e7f5617fdf2", "53e71c960ce46cbe21b420a810610ff8a80d4ec9484d0d8f4ca218b6a65473bf"),
    0.002: ("rd5cc0aec9712", "278895a6a86a6b05321e83093cf20cd40ab6405e62adabe7f4f0937acbb6e859"),
    0.004: ("rdc43b5c9d2d0", "9ebd4be8e725651b6c2488e827d9588150b12f06dffe8d5f0ac8f617e5502a9a"),
}


def recipe(rate):
    run_id, digest = CONTROLS[rate]
    path = Path(__file__).parent / "runs" / run_id / "config.yaml"
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Changed softplus-prefactor control recipe {run_id}")
    control = load_config(path)
    # The control encodes its prefactor as one extra power of the polynomial input.
    model = replace(control.model, attention_degree=16, attention_coefficient_input_power=0,
                    attention_score_prefactor="exp")
    return replace(control, model=model)


def check_recipe(config):
    validate_config(config)
    actual = config_to_dict(config)
    expected = config_to_dict(recipe(config.optim.learning_rate))
    for item in (actual, expected):
        for key in EXECUTION_KEYS:
            item["train"].pop(key, None)
    if actual != expected:
        raise ValueError("The run must match its same-rate affine-tanh SoS control with an exponential prefactor")


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    return replace(attention_spec(config, entry_module=entry_module), check_config=check_recipe,
                   required_backend="fused")


if __name__ == "__main__":
    run_acceptance(partial(experiment_spec, entry_module=__name__))

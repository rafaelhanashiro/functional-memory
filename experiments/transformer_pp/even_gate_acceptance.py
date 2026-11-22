"""Qualify degree-eight even sigmoid gates with the width-four Transformer++ recipe."""
import argparse
from dataclasses import replace
from functools import partial
import hashlib
from pathlib import Path
import sys

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import config_to_dict, load_config
from scaffolds.transformer.configs.base import validate_config


CONTROL = Path(__file__).parent / "runs/rd5e3ec5a4f02/config.yaml"
CONTROL_SHA256 = "f0126846c2326b3e3c4cc74ef4c9151f442ef39452e950d41f3d82dcacb7a34d"
EXECUTION_KEYS = ("workdir", "checkpoint_dir", "source_commit", "wandb_run_name",
                  "wandb_group", "wandb_dir", "wandb_mode")
ORDER = (2, 4, 6, 8)
VARIANTS = (("rms_norm_cosine", "zero"), ("rms_norm", "zero"), ("rms_norm", "logit_taylor"))
# One learning rate for every gate, or gate r at 2/r of it (optim.coefficient_order_learning_rate).
ORDER_RATES = ("constant", "two_over_order")
FEATURE_TESTS = ("scaffolds/transformer/tests/test_gated_even_coefficients.py",
                 "scaffolds/transformer/tests/test_qk_rms_cosine.py",
                 "scaffolds/transformer/tests/test_coefficient_order_learning_rate.py")


def control_config():
    if hashlib.sha256(CONTROL.read_bytes()).hexdigest() != CONTROL_SHA256:
        raise ValueError("Pinned width-four query partner rd5e3ec5a4f02 changed")
    return load_config(CONTROL)


def recipe(score, bias, order_rate="constant"):
    if (score, bias) not in VARIANTS:
        raise ValueError(f"Unsupported even-gate score and bias initialization {(score, bias)}")
    if order_rate not in ORDER_RATES:
        raise ValueError(f"Unsupported gate learning-rate rule {order_rate!r}")
    base = control_config()
    model = replace(base.model, attention_coefficient_orders=list(ORDER),
                    attention_coefficient_activation="sigmoid", attention_coefficient_amplitude_initial=1.0,
                    attention_coefficient_sigmoid_bias_initial=bias, attention_coefficient_scale="none",
                    attention_coefficient_amplitude_scale="none", attention_coefficient_input_normalization="l2",
                    attention_coefficient_input_source="pre_rmsnorm", attention_head_rms_prescale="max_abs",
                    qk_norm_type=score, qk_norm_head_scale=(score == "rms_norm"))
    return replace(base, model=model, train=replace(base.train, gradient_event_capture=True),
                   optim=replace(base.optim, coefficient_order_learning_rate=order_rate))


def check_recipe(config):
    validate_config(config)
    reference = recipe(config.model.qk_norm_type, config.model.attention_coefficient_sigmoid_bias_initial,
                       config.optim.coefficient_order_learning_rate)
    actual, expected = config_to_dict(config), config_to_dict(reference)
    for entry in (actual, expected):
        for key in EXECUTION_KEYS:
            entry["train"].pop(key, None)
    if actual != expected:
        changed = [f"{section}.{name}" for section in actual for name, value in actual[section].items()
                   if value != expected[section].get(name)]
        raise ValueError(f"Settings differ from the pinned even-gate recipe: {changed}")


def check_pilot(config, outcome, validation):
    """Check the trained model's coefficient slots, range, and bounded score on its fixed diagnostic sequence."""
    import numpy as np

    diagnostic = validation["attention_diagnostics"]
    summary = diagnostic["coefficients"]["effective_summary"]
    gates_min, gates_max = (np.asarray(summary[key]) for key in ("min", "max"))
    shape = (config.model.num_layers, config.model.num_heads, config.model.attention_degree + 1)
    if gates_min.shape != shape or gates_max.shape != shape:
        raise RuntimeError(f"Expected coefficient diagnostics of shape {shape}")
    absent = [i for i in range(shape[-1]) if i not in ORDER]
    if np.any(gates_min[..., absent] != 0) or np.any(gates_max[..., absent] != 0):
        raise RuntimeError("The diagnostic found a nonzero constant or odd-order coefficient")
    if not (np.all(gates_min[..., ORDER] >= 0) and np.all(gates_max[..., ORDER] <= 1)):
        raise RuntimeError("Effective gate coefficients leave the sigmoid range")
    low = min(row["score_min"] for row in diagnostic["layers"])
    high = max(row["score_max"] for row in diagnostic["layers"])
    # BF16 operands round the unit directions immediately before the dot product.
    if config.model.qk_norm_type == "rms_norm_cosine" and max(abs(low), abs(high)) > 1.01:
        raise RuntimeError(f"The cosine score exceeds BF16 rounding tolerance: {low}, {high}")
    return {"coefficient_orders": list(ORDER), "min_effective_gate": float(gates_min[..., ORDER].min()),
            "max_effective_gate": float(gates_max[..., ORDER].max()), "score_min": low, "score_max": high}


def experiment_spec(config, *, entry_module):
    check_recipe(config)
    shared = attention_spec(config, entry_module=entry_module)
    return replace(shared, check_config=check_recipe, check_pilot=check_pilot, required_backend="mosaic",
                   cpu_tests=(*shared.cpu_tests, *FEATURE_TESTS,
                              "experiments/transformer_pp/tests/test_even_gate_acceptance.py"))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-type", default="H100")
    args, _ = parser.parse_known_args(argv)
    if args.gpu_type.upper() not in ("B200", "B300", "H100"):
        raise ValueError(f"Width-four gated Mosaic runs qualify on B200, B300 and H100 GPUs, got {args.gpu_type}")
    run_acceptance(partial(experiment_spec, entry_module=__name__), argv)


if __name__ == "__main__":
    main()

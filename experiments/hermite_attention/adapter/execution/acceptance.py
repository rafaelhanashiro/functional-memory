"""Qualify fixed Hermite coefficients with and without row normalization."""
import math

from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_initialization,
)

# Probabilists' Hermite expansion of exp(s) under N(0, 1), in monomial order.
# The factor cancels in normalized attention and is retained for the unnormalized analogue.
HERMITE_SCALE = math.exp(0.5)
HERMITE_COEFFICIENTS = {
    1: (1.0, 1.0),
    2: (0.5, 1.0, 0.5),
    3: (0.5, 0.5, 0.5, 1.0 / 6.0),
    4: (0.625, 0.5, 0.25, 1.0 / 6.0, 1.0 / 24.0),
}


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product"
            or model.attention_degree not in HERMITE_COEFFICIENTS
            or model.attention_coefficients != "fixed" or model.quadratic_mode != "fixed"
            or model.attention_coefficient_parameterization != "direct"
            or model.qk_norm or model.qk_head_dim != 64 or model.value_head_dim != 64
            or model.attention_backend != "fused"):
        raise ValueError("This sweep uses fixed Hermite degrees 1-4, raw scores, d_K=d_V=64, and fused attention")
    expected = tuple(HERMITE_SCALE * value for value in HERMITE_COEFFICIENTS[model.attention_degree])
    if model.attention_fixed_coefficients is None or tuple(model.attention_fixed_coefficients) != expected:
        raise ValueError("Recipe coefficients must be the fixed standard-normal Hermite projection of exp(s)")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_fixed_polynomial.py",
               "scaffolds/transformer/tests/test_fixed_polynomial_config.py",
               "experiments/hermite_attention/adapter/tests"),
    gpu_tests=("scaffolds/transformer/tests/test_fixed_polynomial_fused.py",),
    check_config=check_config,
    check_initialization=matches_softmax_initialization,
    required_backend="fused",
    # A long pilot would preselect against the early scientific failures being studied.
    # This validates two full-recipe updates, evaluation, and a third update after restart.
    pilot_steps=2,
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

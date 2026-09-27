"""Qualify fixed degree-two Hermite attention evaluated as a square."""
import math

from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_initialization,
)

SQUARE_FACTOR = math.sqrt(math.exp(0.5) / 2.0)
EXPECTED_FACTORS = (SQUARE_FACTOR, SQUARE_FACTOR, 0.0)


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product"
            or model.attention_degree != 2
            or model.attention_coefficients != "fixed"
            or model.attention_coefficient_parameterization != "sum_of_squares"
            or model.attention_normalization is not True
            or model.attention_fixed_coefficients is not None
            or model.quadratic_mode != "fixed"
            or model.qk_norm
            or model.qk_head_dim != 64 or model.value_head_dim != 64
            or model.attention_backend != "fused"):
        raise ValueError("The experiment requires fixed normalized degree-two Hermite attention with raw scores and fused evaluation")
    if (model.attention_fixed_square_factors is None
            or tuple(model.attention_fixed_square_factors) != EXPECTED_FACTORS):
        raise ValueError("Square factors must express exp(1/2) (s+1)^2 / 2 exactly")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_fixed_square_factors.py",
               "scaffolds/transformer/tests/test_fixed_polynomial_config.py",
               "experiments/hermite_factored_m2/adapter/tests"),
    gpu_tests=("scaffolds/transformer/tests/test_fixed_square_factors_fused.py",),
    check_config=check_config,
    check_initialization=matches_softmax_initialization,
    required_backend="fused",
    pilot_steps=2,
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

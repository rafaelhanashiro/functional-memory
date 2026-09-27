"""Qualify matched fixed Hermite/Taylor attention over canonical QK-norm scores."""
import math

from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_backbone_initialization,
)

DEGREES = (1, 2, 3, 4)
HERMITE_SCALE = math.exp(0.5)
HERMITE_COEFFICIENTS = {
    1: (1.0, 1.0),
    2: (0.5, 1.0, 0.5),
    3: (0.5, 0.5, 0.5, 1.0 / 6.0),
    4: (0.625, 0.5, 0.25, 1.0 / 6.0, 1.0 / 24.0),
}
SQUARE_FACTOR = math.sqrt(HERMITE_SCALE / 2.0)
EXPECTED_FACTORS = (SQUARE_FACTOR, SQUARE_FACTOR, 0.0)


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product" or model.attention_degree not in DEGREES
            or model.attention_coefficients != "fixed" or model.quadratic_mode != "fixed"
            or model.qk_norm is not True or model.qk_norm_sharing != "per_head"
            or model.qk_norm_initial_scale is not None or model.qk_norm_eps != 1e-6
            or model.qk_head_dim != 64 or model.value_head_dim != 64
            or model.attention_backend != "fused"):
        raise ValueError("This sweep requires fixed degrees 1-4, per-head QK-norm with initial scale "
                         "sqrt(d_K), d_K=d_V=64, and fused attention")
    if model.tie_embeddings or config.optim.weight_decay_policy != "exclude_embeddings_norms_biases_v1":
        raise ValueError("The canonical recipe requires untied embeddings and selective weight decay")
    if model.attention_coefficient_parameterization == "sum_of_squares":
        if (model.attention_degree != 2 or not model.attention_normalization
                or model.attention_fixed_coefficients is not None
                or model.attention_fixed_square_factors is None
                or tuple(model.attention_fixed_square_factors) != EXPECTED_FACTORS):
            raise ValueError("Fixed square factors must express normalized exp(1/2) (s+1)^2 / 2")
        return
    if model.attention_coefficient_parameterization != "direct" or model.attention_fixed_square_factors is not None:
        raise ValueError("Hermite monomial coefficients and the Taylor controls use direct parameterization")
    if model.attention_fixed_coefficients is None:
        if model.attention_normalization:
            raise ValueError("This study's Taylor controls are unnormalized; normalized controls already exist")
        return
    expected = tuple(HERMITE_SCALE * value for value in HERMITE_COEFFICIENTS[model.attention_degree])
    if tuple(model.attention_fixed_coefficients) != expected:
        raise ValueError("Recipe coefficients must be the fixed standard-normal Hermite projection of exp(s)")
    if model.attention_degree == 2 and model.attention_normalization:
        raise ValueError("Normalized Hermite degree 2 must use fixed square factors to preserve its double root")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_fixed_polynomial.py",
               "scaffolds/transformer/tests/test_fixed_polynomial_config.py",
               "scaffolds/transformer/tests/test_fixed_square_factors.py",
               "scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_parameter_policy.py",
               "experiments/hermite_attention/tests"),
    gpu_tests=("scaffolds/transformer/tests/test_fixed_polynomial_fused.py",
               "scaffolds/transformer/tests/test_fixed_square_factors_fused.py",
               "scaffolds/transformer/tests/test_qk_norm.py",
               "experiments/hermite_attention/tests/test_qk_norm.py"),
    check_config=check_config,
    check_initialization=matches_softmax_backbone_initialization,
    required_backend="fused",
    pilot_steps=20,
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

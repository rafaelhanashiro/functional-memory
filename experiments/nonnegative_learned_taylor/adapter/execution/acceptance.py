"""Acceptance for row-normalized sum-of-squares (nonnegative) polynomial kernels over QK-norm scores."""
from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_backbone_initialization,
)

DEGREES = (2, 4)
QK_HEAD_DIM = 64


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product" or model.attention_degree not in DEGREES
            or not model.attention_normalization or model.attention_coefficients != "learned"
            or model.attention_coefficient_parameterization != "sum_of_squares"
            or model.attention_coefficient_sharing != "per_head" or model.quadratic_mode != "fixed"
            or model.qk_head_dim != QK_HEAD_DIM or not model.qk_norm or model.qk_norm_sharing != "per_head"
            or model.qk_norm_initial_scale is not None or model.attention_backend != "blocked"):
        raise ValueError("This experiment runs row-normalized sum-of-squares kernels of degree 2 or 4 with "
                         "per-head factors and per-head QK-norm at d_K = 64 on the blocked reference only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_learned_polynomial_nonnegative.py",
               "scaffolds/transformer/tests/test_learned_polynomial_normalized.py",
               "scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_parameter_policy.py",
               "experiments/nonnegative_learned_taylor/adapter/tests"),
    # The blocked sum-of-squares and direct normalized kernels at the production BF16 context,
    # and cosine operands through every kernel.
    gpu_tests=("scaffolds/transformer/tests/test_learned_polynomial_nonnegative.py",
               "scaffolds/transformer/tests/test_learned_polynomial_normalized.py",
               "scaffolds/transformer/tests/test_qk_norm.py"),
    check_config=check_config,
    # Backbone equals the softmax control's; factors start at the Taylor decomposition and the
    # log scales at log(sqrt(d_K)).
    check_initialization=matches_softmax_backbone_initialization,
    required_backend="blocked",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

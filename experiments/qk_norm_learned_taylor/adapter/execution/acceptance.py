"""Acceptance for learned per-head polynomial kernels over QK-norm cosine scores, normalized or not."""
from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_backbone_initialization,
)

DEGREES = (1, 2, 3, 4)
QK_HEAD_DIM = 64


def expected_backend(normalized: bool) -> str:
    """Unnormalized kernels run fused; normalized learned kernels exist only in the blocked reference."""
    return "blocked" if normalized else "fused"


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product" or model.attention_degree not in DEGREES
            or model.attention_coefficients != "learned" or model.attention_coefficient_sharing != "per_head"
            or model.quadratic_mode != "fixed" or model.qk_head_dim != QK_HEAD_DIM
            or not model.qk_norm or model.qk_norm_sharing != "per_head" or model.qk_norm_initial_scale is not None):
        raise ValueError("This experiment runs learned per-head polynomial kernels of degrees 1-4 with per-head "
                         "QK-norm at d_K = 64 and the default initial scale sqrt(d_K) only")
    if model.attention_backend != expected_backend(model.attention_normalization):
        raise ValueError("Recipes state their backend explicitly: fused without row normalization, "
                         "blocked (the only implementation) with it")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_learned_polynomial.py",
               "scaffolds/transformer/tests/test_learned_polynomial_normalized.py",
               "scaffolds/transformer/tests/test_learned_polynomial_training.py",
               "scaffolds/transformer/tests/test_unnormalized_attention.py",
               "scaffolds/transformer/tests/test_unnormalized_training.py",
               "experiments/qk_norm_learned_taylor/adapter/tests"),
    # Cosine operands through every kernel, the fused learned-polynomial and unnormalized kernels,
    # and the blocked normalized learned kernel at the production BF16 context.
    gpu_tests=("scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_learned_polynomial_fused_attention.py",
               "scaffolds/transformer/tests/test_unnormalized_fused_attention.py",
               "scaffolds/transformer/tests/test_learned_polynomial_normalized.py"),
    check_config=check_config,
    # Backbone equals the softmax control's; coefficients start at 1/r! and the log scales at log(sqrt(d_K)).
    check_initialization=matches_softmax_backbone_initialization,
    # Recipes pin their backend explicitly and check_config enforces the pairing above.
    required_backend=None,
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

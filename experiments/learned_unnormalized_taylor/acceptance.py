"""Acceptance for unnormalized attention with learned per-head polynomial coefficients."""
from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_backbone_initialization,
)

DEGREES = (1, 2, 3, 4)


def check_config(config) -> None:
    model = config.model
    if (model.attention_normalization or model.attention_coefficients != "learned"
            or model.attention_coefficient_sharing != "per_head" or model.attention_degree not in DEGREES):
        raise ValueError("This experiment runs unnormalized degrees 1-4 with learned per-head coefficients only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_learned_polynomial.py",
               "scaffolds/transformer/tests/test_learned_polynomial_training.py",
               "scaffolds/transformer/tests/test_unnormalized_attention.py",
               "scaffolds/transformer/tests/test_unnormalized_training.py",
               "experiments/learned_unnormalized_taylor/tests", "results/learned_unnormalized_taylor/tests"),
    gpu_tests=("scaffolds/transformer/tests/test_learned_polynomial_fused_attention.py",
               "scaffolds/transformer/tests/test_unnormalized_fused_attention.py"),
    check_config=check_config,
    # Backbone equals the softmax control's; coefficients start at the Taylor values 1/r!.
    check_initialization=matches_softmax_backbone_initialization,
    required_backend="fused",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

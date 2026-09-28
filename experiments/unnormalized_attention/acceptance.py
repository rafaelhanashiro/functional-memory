"""Acceptance for unnormalized exponential attention."""
from scaffolds.transformer.acceptance import AcceptanceSpec, main as run_acceptance, matches_softmax_initialization


def check_config(config) -> None:
    if config.model.attention_normalization or config.model.attention_degree is not None:
        raise ValueError("This record runs the unnormalized exponential only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_unnormalized_attention.py", "scaffolds/transformer/tests/test_unnormalized_training.py",
               "scaffolds/transformer/tests/test_qk_norm.py",
               "experiments/unnormalized_attention/tests", "results/unnormalized_attention/tests"),
    gpu_tests=("scaffolds/transformer/tests/test_unnormalized_fused_attention.py", "scaffolds/transformer/tests/test_qk_norm.py"),
    check_config=check_config,
    check_initialization=matches_softmax_initialization,
    required_backend="fused",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

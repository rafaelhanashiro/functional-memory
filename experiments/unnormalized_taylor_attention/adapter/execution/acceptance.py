"""Acceptance for unnormalized truncated-Taylor attention."""
from scaffolds.transformer.acceptance import AcceptanceSpec, main as run_acceptance, matches_softmax_initialization


def check_config(config) -> None:
    if config.model.attention_normalization or config.model.attention_degree is None:
        raise ValueError("This record runs unnormalized Taylor degrees only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_unnormalized_attention.py", "scaffolds/transformer/tests/test_unnormalized_training.py",
               "experiments/unnormalized_taylor_attention/adapter/tests"),
    gpu_tests=("scaffolds/transformer/tests/test_unnormalized_fused_attention.py",),
    check_config=check_config,
    check_initialization=matches_softmax_initialization,
    required_backend="fused",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

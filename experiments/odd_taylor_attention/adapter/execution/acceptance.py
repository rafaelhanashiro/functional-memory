"""Acceptance for the odd-degree (signed) Taylor attention record."""
from scaffolds.transformer.acceptance import AcceptanceSpec, main as run_acceptance


def check_config(config) -> None:
    if config.model.attention_degree not in (1, 3) or not config.model.attention_normalization:
        raise ValueError("This record runs row-normalized odd Taylor degrees 1 and 3 only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_odd_attention.py", "scaffolds/transformer/tests/test_odd_training.py",
               "scaffolds/transformer/tests/test_qk_norm.py", "scaffolds/transformer/tests/test_parameter_policy.py",
               "experiments/odd_taylor_attention/adapter/tests"),
    # The signed two-pass kernel, and cosine operands through every kernel (the recipes use QK-norm).
    gpu_tests=("scaffolds/transformer/tests/test_odd_attention.py", "scaffolds/transformer/tests/test_fused_attention.py",
               "scaffolds/transformer/tests/test_qk_norm.py"),
    check_config=check_config,
    required_backend="fused",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

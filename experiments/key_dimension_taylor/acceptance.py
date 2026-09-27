"""Acceptance for the dK x Taylor-degree record: softmax and row-normalized Taylor recipes."""
from scaffolds.transformer.acceptance import AcceptanceSpec, main as run_acceptance


def check_config(config) -> None:
    if config.model.attention_mechanism != "dot_product" or not config.model.attention_normalization:
        raise ValueError("This record runs row-normalized dot-product attention only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_attention.py", "scaffolds/transformer/tests/test_model.py", "scaffolds/transformer/tests/test_training.py",
               "scaffolds/transformer/tests/test_checkpoint_audit.py", "scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_parameter_policy.py", "experiments/key_dimension_taylor/tests", "results/key_dimension_taylor/tests"),
    # The fused Taylor kernels at every production width and degree, and cosine operands through every
    # kernel (the dK=64 Taylor recipes use QK-norm).
    gpu_tests=("scaffolds/transformer/tests/test_fused_attention.py", "scaffolds/transformer/tests/test_taylor_kernel_widths.py",
               "scaffolds/transformer/tests/test_qk_norm.py"),
    check_config=check_config,
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

"""Acceptance for the learned-coefficient quadratic attention record."""
from scaffolds.transformer.acceptance import AcceptanceSpec, main as run_acceptance


def check_config(config) -> None:
    if config.model.quadratic_mode == "fixed" or config.model.attention_degree != 2:
        raise ValueError("This record runs learned degree-2 quadratic recipes only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_learned_quadratic.py", "scaffolds/transformer/tests/test_learned_training.py",
               "experiments/learned_quadratic_attention/tests", "results/learned_quadratic_attention/tests"),
    gpu_tests=("scaffolds/transformer/tests/test_learned_fused_attention.py",),
    check_config=check_config,
    required_backend="fused",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

"""Acceptance for the GLA d_K sweep: shared checks configured for the fused GLA kernel."""
from scaffolds.transformer.acceptance import AcceptanceSpec, main as run_acceptance


def check_config(config) -> None:
    if config.model.attention_mechanism != "gla":
        raise ValueError("This experiment runs GLA recipes only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    # Recurrence oracle, all gradients, bridge tracing, configuration, and this sweep's recipes.
    cpu_tests=("scaffolds/transformer/tests/test_gla.py", "scaffolds/transformer/tests/test_gla_bridge.py",
               "scaffolds/transformer/tests/test_gla_integration.py", "experiments/gla_key_dimension/adapter/tests"),
    # Fused kernels against the sequential recurrence for every production width.
    gpu_tests=("scaffolds/transformer/tests/test_gla_kernel.py",),
    check_config=check_config,
    required_backend="fused",
    # The Triton custom-call bridge runs outside XLA command buffers.
    runtime_env={"XLA_FLAGS": "--xla_gpu_enable_command_buffer=",
                 "EJKERNEL_TRITON_CACHE_DIR": ".cache/gla_key_dimension/triton"},
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

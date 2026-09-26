"""Acceptance for the softmax baseline under the canonical selective weight-decay policy."""
from scaffolds.transformer.acceptance import AcceptanceSpec, main as run_acceptance

WEIGHT_DECAY_POLICY = "exclude_embeddings_norms_biases_v1"


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product" or not model.attention_normalization
            or model.attention_degree is not None):
        raise ValueError("This experiment runs the exact-softmax reference only")
    if config.optim.weight_decay_policy != WEIGHT_DECAY_POLICY:
        raise ValueError(f"This experiment runs the {WEIGHT_DECAY_POLICY} weight-decay policy only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    # Attention and model references, the optimizer's decay policy, training, resume audits, and this recipe.
    cpu_tests=("scaffolds/transformer/tests/test_attention.py", "scaffolds/transformer/tests/test_model.py",
               "scaffolds/transformer/tests/test_parameter_policy.py", "scaffolds/transformer/tests/test_config.py",
               "scaffolds/transformer/tests/test_training.py", "scaffolds/transformer/tests/test_checkpoint_audit.py",
               "experiments/decay_policy_baseline/adapter/tests"),
    # cuDNN softmax against the manual reference at the production widths, on the allocation's
    # GPU type. d_K = 256 is excluded: cuDNN rejects it on B300 when d_K differs from d_V.
    gpu_tests=tuple(f"scaffolds/transformer/tests/test_softmax_backends.py::{test}[{width}]"
                    for test in ("test_cudnn_matches_manual_outputs_input_and_parameter_gradients",
                                 "test_cudnn_production_context_forward_and_backward_are_finite")
                    for width in (2, 4, 8, 16, 32, 64, 128)),
    check_config=check_config,
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

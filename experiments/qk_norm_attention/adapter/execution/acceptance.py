"""Acceptance for exact softmax attention over QK-norm cosine scores at the 200M backbone."""
from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_backbone_initialization,
)

QK_HEAD_DIM = 64


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product" or not model.qk_norm or model.qk_norm_sharing != "per_head"
            or model.attention_degree is not None or not model.attention_normalization
            or model.quadratic_mode != "fixed" or model.attention_coefficients != "fixed"
            or model.qk_head_dim != QK_HEAD_DIM or model.qk_norm_initial_scale is not None):
        raise ValueError("This experiment runs exact softmax with per-head QK-norm at d_K = 64 "
                         "and the default initial scale sqrt(d_K) only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_attention.py",
               "scaffolds/transformer/tests/test_softmax_backends.py",
               "scaffolds/transformer/tests/test_model.py",
               "scaffolds/transformer/tests/test_training.py",
               "scaffolds/transformer/tests/test_checkpoint_audit.py",
               "scaffolds/transformer/tests/test_parameter_policy.py",
               "experiments/qk_norm_attention/adapter/tests"),
    # Cosine operands through cuDNN and the fused kernels at the production width, plus the
    # exact-softmax cuDNN control. The softmax-backends GPU cases stay out: their width-256
    # cases exercise a cuDNN limit on B300 that this d_K = 64 recipe never reaches.
    gpu_tests=("scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_attention.py"),
    check_config=check_config,
    # Backbone equals the softmax control's; the log scale starts at log(sqrt(d_K)).
    check_initialization=matches_softmax_backbone_initialization,
    required_backend="fused",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

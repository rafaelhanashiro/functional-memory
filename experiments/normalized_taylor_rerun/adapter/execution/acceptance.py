"""Acceptance for row-normalized fixed Taylor kernels at d_K = 64 under the scaffold's current defaults."""
from scaffolds.transformer.acceptance import (
    AcceptanceSpec, main as run_acceptance, matches_softmax_backbone_initialization,
)

DEGREES = (1, 2, 3, 4, 8, 16)
QK_HEAD_DIM = 64
WEIGHT_DECAY_POLICY = "exclude_embeddings_norms_biases_v1"


def check_config(config) -> None:
    model = config.model
    if (model.attention_mechanism != "dot_product" or not model.attention_normalization
            or model.attention_degree not in DEGREES or model.attention_coefficients != "fixed"
            or model.quadratic_mode != "fixed" or model.qk_head_dim != QK_HEAD_DIM
            or not model.qk_norm or model.qk_norm_sharing != "per_head" or model.qk_norm_initial_scale is not None
            or model.attention_backend != "fused" or config.optim.weight_decay_policy != WEIGHT_DECAY_POLICY):
        raise ValueError("This experiment runs row-normalized fixed Taylor kernels of degree 1, 2, 3, 4, 8, or 16 at "
                         "d_K = 64 with per-head QK-norm, selective weight decay, and the fused kernel only")


SPEC = AcceptanceSpec(
    entry_module=__name__,
    cpu_tests=("scaffolds/transformer/tests/test_attention.py", "scaffolds/transformer/tests/test_model.py",
               "scaffolds/transformer/tests/test_training.py", "scaffolds/transformer/tests/test_checkpoint_audit.py",
               "scaffolds/transformer/tests/test_odd_training.py", "scaffolds/transformer/tests/test_qk_norm.py",
               "scaffolds/transformer/tests/test_parameter_policy.py",
               "experiments/normalized_taylor_rerun/adapter/tests"),
    # The fused even and signed odd Taylor kernels at every width and degree, and cosine operands
    # through every kernel.
    gpu_tests=("scaffolds/transformer/tests/test_fused_attention.py",
               "scaffolds/transformer/tests/test_taylor_kernel_widths.py",
               "scaffolds/transformer/tests/test_odd_attention.py",
               "scaffolds/transformer/tests/test_qk_norm.py"),
    check_config=check_config,
    # Backbone equals the softmax control's; the QK-norm log scales start at log(sqrt(d_K)).
    check_initialization=matches_softmax_backbone_initialization,
    required_backend="fused",
)


def main(argv=None) -> None:
    run_acceptance(SPEC, argv)


if __name__ == "__main__":
    main()

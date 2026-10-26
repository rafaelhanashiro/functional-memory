"""Qualify the isolated 200M / 4B reference recipe before production."""
from dataclasses import replace
import math

from scaffolds.transformer.acceptance import attention_spec, main as run_acceptance
from scaffolds.transformer.configs import validate_config


RMS_TESTS = "scaffolds/transformer/tests/test_qk_rms_norm.py"
BACKEND_TESTS = "scaffolds/transformer/tests/test_qk_rms_attention_backends.py"


def check_config(config):
    """Reject a legacy score/init recipe or partially order-scaled comparison."""
    validate_config(config)
    expected = {
        "kind": "transformer", "vocab_size": 32000, "max_seq_len": 2048,
        "num_layers": 15, "hidden_size": 896, "num_heads": 14, "num_kv_heads": 14,
        "qk_head_dim": 64, "value_head_dim": 64, "intermediate_size": 2304,
        "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "rms_norm_fp32": True,
        "dropout": 0.0, "tie_embeddings": False, "attention_mechanism": "dot_product",
        "qk_norm": True, "qk_norm_type": "rms_norm", "qk_norm_sharing": "per_layer",
        "qk_norm_initial_scale": None, "qk_norm_eps": 1e-6,
        "initialization": "transformer_truncated_reference", "positional_encoding": "rope",
    }
    for name, value in expected.items():
        if getattr(config.model, name) != value:
            raise ValueError(f"200M / 4B setup requires model.{name}={value!r}")
    if config.data.batch_size != 256 or config.data.sequence_length != 2049:
        raise ValueError("200M / 4B setup requires 256 sequences of 2048 prediction tokens")
    if config.optim.total_steps != 7570 or config.optim.warmup_steps != 455:
        raise ValueError("200M / 4B setup requires the full 7570-step schedule and 455 warmup steps")
    if not math.isclose(config.optim.min_learning_rate, 0.1 * config.optim.learning_rate, rel_tol=1e-12):
        raise ValueError("200M / 4B setup requires a minimum LR of 10% of the peak")
    if (config.optim.weight_decay_policy != "exclude_embeddings_norms_biases_v1"
            or config.optim.coefficient_penalty != 0):
        raise ValueError("200M / 4B setup excludes embeddings, norm gains, biases and coefficients from decay")
    if config.train.dtype != "bfloat16" or config.train.seed != 42:
        raise ValueError("200M / 4B setup uses BF16 computation with seed 42")
    model = config.model
    if model.attention_degree is not None and model.attention_degree not in (2, 4, 8, 12):
        raise ValueError("This study includes polynomial degrees 2, 4, 8, and 12")
    if model.attention_normalization:
        if model.attention_degree is not None or model.attention_output_scaling != "none":
            raise ValueError("The normalized control is ordinary softmax without head output RMSNorm")
    elif model.attention_output_scaling != "head_rms_norm":
        raise ValueError("The unnormalized comparison requires per-head output RMSNorm")
    if model.attention_coefficients == "learned":
        sigmoid = model.attention_coefficient_activation == "sigmoid"
        if (model.attention_coefficient_scale != ("none" if sigmoid else "factorial")
                or model.attention_coefficient_parameterization != "direct"
                or model.attention_coefficient_sharing != "per_head"):
            raise ValueError("Learned comparison coefficients use direct per-head whole-coefficient order "
                             "scaling; sigmoid coefficients have no input-independent part to scale")
        if sigmoid:
            # Pure sigmoid c_r = sigmoid(w_r^T x) [/ r!], with or without order scaling.
            if (model.attention_coefficient_conditioning != "query"
                    or model.attention_coefficient_projection_initial_std != 0):
                raise ValueError("Sigmoid coefficients are query-only with W starting at zero")
        elif model.attention_coefficient_conditioning in ("query", "query_key"):
            if (model.attention_coefficient_amplitude_scale != "factorial"
                    or model.attention_coefficient_activation != "tanh"
                    or model.attention_coefficient_amplitude_initial != 0
                    or model.attention_coefficient_tanh_bias
                    or not math.isclose(model.attention_coefficient_projection_initial_std,
                                        1 / math.sqrt(model.hidden_size), rel_tol=1e-12)):
                raise ValueError("Query and query-key coefficients scale both b and a, with zero-amplitude Taylor start")
        # Dense readout: V [degree+1, K] starts at zero and is order-scaled like a.
        if model.attention_coefficient_readout == "dense" and model.attention_coefficient_conditioning != "query":
            raise ValueError("Dense coefficient readout is a query-only generator")


def spec(config):
    check_config(config)
    shared = attention_spec(config, entry_module=__name__)

    def select_tests(recipe, backend, cpu):
        tests = list(shared.select_tests(recipe, backend, cpu))
        if cpu:
            return (*tests, RMS_TESTS, "scaffolds/transformer/tests/test_normalization.py")
        model = recipe["model"]
        degree = model["attention_degree"]
        family = ("softmax" if model["attention_normalization"] else "exp") if degree is None else (
            "fixed" if model["attention_coefficients"] == "fixed" else model["attention_coefficient_conditioning"])
        tests.append(f"{BACKEND_TESTS}::test_full_context_qk_rms_backend[{family}-{degree or 0}-{backend}]")
        return tuple(tests)

    return replace(shared, check_config=check_config, select_tests=select_tests,
                   cpu_tests=(*shared.cpu_tests, "experiments/transformer_200m_4b/tests/test_recipe.py"))


def main(argv=None):
    run_acceptance(spec, argv)


if __name__ == "__main__":
    main()

import jax
from flax import nnx

from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.attention.parameters import attention_parameter_count
from scaffolds.transformer.models.architectures.transformer import TransformerLM


MODEL_REGISTRY = {"transformer": TransformerLM}


def build_model(config: ModelConfig, rngs: nnx.Rngs | jax.Array | int | None = None):
    config.__post_init__()
    if rngs is None:
        rngs = nnx.Rngs(0)
    elif isinstance(rngs, int):
        rngs = nnx.Rngs(rngs)
    elif isinstance(rngs, jax.Array):
        rngs = nnx.Rngs(params=rngs, dropout=jax.random.fold_in(rngs, 0))
    return TransformerLM(config, rngs=rngs)


def model_parameter_count(config: ModelConfig) -> int:
    """Exact trainable count without allocating a production model."""
    config.__post_init__()
    h = config.hidden_size
    embedding = config.vocab_size * h
    attention = 2 * h * config.num_heads * (config.qk_head_dim + config.value_head_dim)
    if config.attention_mechanism == "gla":
        key_dim = config.num_heads * config.qk_head_dim
        value_dim = config.num_heads * config.value_head_dim
        # Output gate, low-rank vector decay (including bias), and head RMSNorm.
        attention += h * value_dim + h * config.gla_gate_rank + config.gla_gate_rank * key_dim + key_dim + value_dim
    if config.attention_mechanism == "gated_deltanet":
        key_dim = config.num_heads * config.qk_head_dim
        value_dim = config.num_heads * config.value_head_dim
        # Output gate, decay and beta projections, short convolutions, A_log and dt_bias, gated-norm scale.
        attention += (h * value_dim + 2 * h * config.num_heads + config.gdn_conv_size * (2 * key_dim + value_dim)
                      + 2 * config.num_heads + config.value_head_dim)
        if config.gdn_erase_mode == "independent":
            attention += h * config.num_heads
        if config.gdn_polynomial_degree == 2:
            attention += 2 * h * config.num_heads
    feed_forward = 3 * h * config.intermediate_size
    return (
        2 * embedding
        + config.num_layers * (attention + feed_forward + 2 * h)
        + h
        + attention_parameter_count(config)
    )

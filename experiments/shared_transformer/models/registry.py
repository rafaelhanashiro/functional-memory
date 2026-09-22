import jax
from flax import nnx

from experiments.shared_transformer.configs import ModelConfig
from experiments.shared_transformer.attention.parameters import attention_parameter_count
from experiments.shared_transformer.models.architectures.transformer import TransformerLM


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
    feed_forward = 3 * h * config.intermediate_size
    return (
        embedding * (1 if config.tie_embeddings else 2)
        + config.num_layers * (attention + feed_forward + 2 * h)
        + h
        + attention_parameter_count(config)
    )

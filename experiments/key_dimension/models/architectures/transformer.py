import jax
from flax import nnx

from experiments.key_dimension.configs import ModelConfig
from experiments.key_dimension.models.components.attention import MultiHeadAttention
from experiments.key_dimension.models.components.normalization import RMSNorm


def _call_block_remat(
    block: "Block",
    x: jax.Array,
    deterministic: bool,
    rngs: nnx.Rngs | jax.Array | None,
) -> jax.Array:
    return block(x, deterministic=deterministic, rngs=rngs)


class SwiGLU(nnx.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, *, rngs: nnx.Rngs):
        self.gate_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=False, rngs=rngs)
        self.up_proj = nnx.Linear(hidden_size, intermediate_size, use_bias=False, rngs=rngs)
        self.down_proj = nnx.Linear(intermediate_size, hidden_size, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nnx.silu(gate) * up)


class Block(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        cfg = config
        self.attention_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.attention = MultiHeadAttention(
            cfg.hidden_size,
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.qk_head_dim,
            cfg.value_head_dim,
            rope_theta=cfg.rope_theta,
            dropout=cfg.dropout,
            backend=cfg.attention_backend,
            rngs=rngs,
        )
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.feed_forward = SwiGLU(cfg.hidden_size, cfg.intermediate_size, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        h = self.attention_norm(x)
        h = self.attention(h, deterministic=deterministic, rngs=rngs)
        x = x + h
        h = self.ffn_norm(x)
        h = self.feed_forward(h)
        return x + h


class TransformerLM(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.token_embedding = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.dropout = nnx.Dropout(config.dropout)
        self.layer_names = tuple(f"layers_{idx}" for idx in range(config.num_layers))
        for layer_name in self.layer_names:
            setattr(self, layer_name, Block(config, rngs=rngs))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        if config.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)

    def __call__(
        self,
        tokens: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
    ) -> jax.Array:
        cfg = self.config
        if tokens.shape[1] > cfg.max_seq_len:
            raise ValueError(f"Sequence length {tokens.shape[1]} exceeds max_seq_len={cfg.max_seq_len}")
        x = self.token_embedding(tokens)
        x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        for layer_name in self.layer_names:
            layer = getattr(self, layer_name)
            if cfg.remat:
                x = nnx.remat(_call_block_remat, static_argnums=(2,))(layer, x, deterministic, rngs)
            else:
                x = layer(x, deterministic=deterministic, rngs=rngs)

        x = self.final_norm(x)
        if cfg.tie_embeddings:
            return self.token_embedding.attend(x)
        if self.lm_head is None:
            raise ValueError("lm_head is not initialized when tie_embeddings=False")
        return self.lm_head(x)

from dataclasses import replace

import jax
from flax import nnx

from scaffolds.transformer.configs import ModelConfig
from scaffolds.transformer.attention.multihead import attention_from_config
from scaffolds.transformer.attention.parameters import (
    attention_parameter_specs, initialize_attention_parameters, mechanism_key, parameters_for_layer,
)
from scaffolds.transformer.models.components.normalization import RMSNorm
from scaffolds.transformer.models.initialization import apply_initialization
from scaffolds.transformer.training import telemetry


def _call_block_remat(
    block: "Block",
    x: jax.Array,
    deterministic: bool,
    rngs: nnx.Rngs | jax.Array | None,
    return_diagnostics: bool,
    attention_parameters: dict[str, jax.Array],
):
    return block(x, deterministic=deterministic, rngs=rngs, return_diagnostics=return_diagnostics,
                 attention_parameters=attention_parameters)


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
        self.attention = attention_from_config(cfg, rngs=rngs)
        self.ffn_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.feed_forward = SwiGLU(cfg.hidden_size, cfg.intermediate_size, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
        return_diagnostics: bool = False,
        coefficients: jax.Array | None = None,
        attention_parameters: dict[str, jax.Array] | None = None,
    ):
        with jax.named_scope("attention"):
            h = self.attention_norm(x)
            result = self.attention(h, deterministic=deterministic, rngs=rngs,
                                    return_diagnostics=return_diagnostics, coefficients=coefficients,
                                    attention_parameters=attention_parameters)
        h, diagnostics = result if return_diagnostics else (result, None)
        x = x + h
        with jax.named_scope("mlp"):
            h = self.ffn_norm(x)
            h = self.feed_forward(h)
        output = x + h
        return (output, diagnostics) if return_diagnostics else output


class TransformerLM(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        config.__post_init__()
        if config.attention_backend == "auto":
            from scaffolds.transformer.attention.backends.selection import resolve_attention_backend

            config = replace(config, attention_backend=resolve_attention_backend(config))
        self.config = config
        self.token_embedding = nnx.Embed(config.vocab_size, config.hidden_size, rngs=rngs)
        self.token_embedding.embedding.set_metadata(weight_decay=False)
        self.dropout = nnx.Dropout(config.dropout)
        self.layer_names = tuple(f"layers_{idx}" for idx in range(config.num_layers))
        for layer_name in self.layer_names:
            setattr(self, layer_name, Block(config, rngs=rngs))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nnx.Linear(config.hidden_size, config.vocab_size, use_bias=False, rngs=rngs)
        self.attention_parameter_specs = attention_parameter_specs(config)
        random_mechanism = any(spec.initial_std for spec in self.attention_parameter_specs)
        initialize_attention_parameters(
            self, self.attention_parameter_specs,
            num_layers=config.num_layers, num_heads=config.num_heads,
            key=mechanism_key(rngs.params.key[...]) if random_mechanism else None,
        )
        if config.initialization != "flax_default":
            # One extra draw after construction, so Flax-default models keep their exact weights.
            apply_initialization(self, config.initialization, rngs.params(), num_layers=config.num_layers)

    def __call__(
        self,
        tokens: jax.Array,
        *,
        deterministic: bool = True,
        rngs: nnx.Rngs | jax.Array | None = None,
        return_diagnostics: bool = False,
    ):
        cfg = self.config
        if tokens.shape[1] > cfg.max_seq_len:
            raise ValueError(f"Sequence length {tokens.shape[1]} exceeds max_seq_len={cfg.max_seq_len}")
        x = self.token_embedding(tokens)
        x = self.dropout(x, deterministic=deterministic, rngs=rngs)

        if cfg.remat and telemetry.active():
            raise ValueError("Training telemetry requires model.remat == False")
        diagnostics = []
        for index, layer_name in enumerate(self.layer_names):
            layer = getattr(self, layer_name)
            parameters = parameters_for_layer(self, self.attention_parameter_specs, index)
            if cfg.remat:
                result = nnx.remat(_call_block_remat, static_argnums=(2, 4))(
                    layer, x, deterministic, rngs, return_diagnostics, parameters,
                )
            else:
                with telemetry.scope(f"layer_{index:02d}"):
                    telemetry.record_residual(x)
                    result = layer(x, deterministic=deterministic, rngs=rngs,
                                   return_diagnostics=return_diagnostics, attention_parameters=parameters)
            if return_diagnostics:
                x, layer_diagnostics = result
                diagnostics.append(layer_diagnostics)
            else:
                x = result

        with telemetry.scope("final"):
            telemetry.record_residual(x)
        x = self.final_norm(x)
        with jax.named_scope("vocabulary_projection"):
            logits = self.lm_head(x)
        return (logits, tuple(diagnostics)) if return_diagnostics else logits

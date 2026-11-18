"""Read-only diagnostic of full-model BF16 repeatability; no training or recipe writes."""
from dataclasses import replace
import json
import os

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.tests.test_pointwise_attention import _config
from scaffolds.transformer.models import build_model
from scaffolds.transformer.attention.multihead import MultiHeadAttention


def arrays(result):
    (loss, logits), gradients = result
    out = {'loss': np.asarray(loss), 'logits': np.asarray(logits)}
    out.update({'/'.join(map(str, path)): np.asarray(value[...])
                for path, value in nnx.to_flat_state(gradients)})
    return out


def differences(first, second):
    out = {}
    for key in first:
        a, b = first[key].astype(np.float64), second[key].astype(np.float64)
        if not np.array_equal(a, b):
            out[key] = {'shape': list(a.shape), 'unequal': int(np.count_nonzero(a != b)),
                        'max_abs': float(np.max(np.abs(a - b))),
                        'relative_l2': float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)),
                        'finite': bool(np.isfinite(a).all() and np.isfinite(b).all())}
    return out


def main():
    gpu, cpu = jax.devices('gpu')[0], jax.devices('cpu')[0]
    print(json.dumps({'gpu': str(gpu), 'kind': gpu.device_kind,
                      'job': os.environ.get('SLURM_JOB_ID'), 'jax': jax.__version__}), flush=True)
    for transform in ('sigmoid', 'softplus', 'exp'):
        for unique in (False, True):
            vocab = 512 if unique else 64
            config = _config('softplus' if transform == 'exp' else transform,
                             max_seq_len=256, vocab_size=vocab, hidden_size=64,
                             qk_head_dim=64, value_head_dim=64, intermediate_size=128,
                             attention_backend='blocked' if transform == 'exp' else 'mosaic')
            if transform == 'exp':
                config.model = replace(config.model, attention_degree=None,
                                       attention_fixed_coefficients=None, attention_score_transform='identity')
            with jax.default_device(cpu):
                model = build_model(config.model, 29)
                for _, module in nnx.iter_graph(model):
                    if isinstance(module, nnx.Module) and hasattr(module, 'dtype'):
                        module.dtype = jnp.bfloat16
                graph, params = nnx.split(model, nnx.Param)
                for _, module in nnx.iter_graph(model):
                    if isinstance(module, MultiHeadAttention):
                        module.backend = 'blocked'
                reference_graph, _ = nnx.split(model, nnx.Param)
                rng = np.random.default_rng(31)
                values = rng.permutation(vocab)[:256][None] if unique else rng.integers(0, vocab, (1, 256))
                tokens = jnp.asarray(values, jnp.int32)

            def program(g):
                def loss(parameters, inputs):
                    logits = nnx.merge(g, parameters)(inputs).astype(jnp.float32)
                    return jnp.mean(logits ** 2), logits
                return jax.jit(jax.value_and_grad(loss, has_aux=True))

            compiled = program(graph)
            operands = jax.device_put(params, gpu), jax.device_put(tokens, gpu)
            first = arrays(compiled(*operands))
            repeat = [differences(first, arrays(compiled(*operands))) for _ in range(3)]
            fresh = differences(first, arrays(program(graph)(*operands)))
            ref = arrays(program(reference_graph)(jax.device_put(params, cpu), tokens))
            comparison = differences(first, ref)
            print(json.dumps({'transform': transform, 'unique_tokens': unique,
                              'repeat_differences': repeat, 'fresh_compile_differences': fresh,
                              'cpu_reference_differences': comparison}), flush=True)


if __name__ == '__main__':
    main()

"""Isolate a GPU failure without training or changing the per-order implementation."""
import argparse
from functools import partial
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.attention import forget_gate
from scaffolds.transformer.tests.test_per_order_decay import inputs, operation, dense

parser = argparse.ArgumentParser()
parser.add_argument('--phase', choices=('gate', 'forward', 'backward', 'original'), default='backward')
parser.add_argument('--endpoint', action='store_true')
parser.add_argument('--precision', default='highest')
parser.add_argument('--time', type=int, default=2048)
parser.add_argument('--degree', type=int, default=4)
parser.add_argument('--dump', type=Path)
args = parser.parse_args()

def mark(stage, **extra):
    print(json.dumps({'stage': stage, 'seconds': time.monotonic(), **extra}), flush=True)

def sync(tree):
    jax.block_until_ready(tree)
    arrays = [np.asarray(a, np.float64) for a in jax.tree.leaves(tree)]
    mark('synchronized', shapes=[list(a.shape) for a in arrays], finite=[bool(np.isfinite(a).all()) for a in arrays])
    return arrays

mark('start', arguments=vars(args) | {'dump': str(args.dump)}, devices=[str(d) for d in jax.devices()],
     xla_flags=os.environ.get('XLA_FLAGS'), jax_version=jax.__version__)
arrays = list(inputs(args.time, args.degree, width=64, value_width=64))
arrays[:3] = [a.astype(jnp.bfloat16) for a in arrays[:3]]
cotangent = jnp.asarray(np.random.default_rng(5).normal(size=arrays[2].shape), jnp.bfloat16)
sync((arrays, cotangent))
if args.phase == 'original':
    mark('reference_start')
    with jax.enable_x64(), jax.default_matmul_precision('highest'):
        out, backward = jax.vjp(partial(dense, endpoint=args.endpoint), *[a.astype(jnp.float64) for a in arrays])
        expected = sync((out, backward(cotangent.astype(jnp.float64))))
    mark('reference_finished')

def evaluate(*operands):
    if args.phase == 'gate':
        q, k, v, c, x, wq, wk, rate, bias = operands
        return (forget_gate.project_gate_inputs(x, wq, bias, wk) if args.endpoint else
                forget_gate.log_decay(x, wq, rate, bias))
    if args.phase == 'forward':
        return operation(*operands, endpoint=args.endpoint)
    out, backward = jax.vjp(partial(operation, endpoint=args.endpoint), *operands)
    return out, backward(cotangent)

with jax.default_matmul_precision(args.precision):
    mark('lower_start')
    lowered = jax.jit(evaluate).lower(*arrays)
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        (args.dump / 'stablehlo.txt').write_text(str(lowered.compiler_ir()))
    mark('compile_start')
    compiled = lowered.compile()
    if args.dump:
        (args.dump / 'optimized_hlo.txt').write_text(compiled.as_text())
    mark('execute_start')
    actual = sync(compiled(*arrays))
mark('production_finished')
if args.phase == 'original':
    errors = [float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)) for a, b in zip(actual, expected, strict=True)]
    mark('errors', relative_l2=errors)
    assert max(errors) < .01, errors
mark('pass')

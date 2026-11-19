"""Recipe receipts: what a recipe's model computes, at a tiny shape on fixed inputs.

A receipt builds a recipe's model at a tiny shape (2 layers, width 128, 2 heads, 64 tokens,
vocabulary 256) on the CPU and on the blocked backend, single-threaded, and hashes its parameter
names and initial values, the loss on fixed tokens and every gradient. Two versions of the code
give a recipe the same receipt exactly when they build the same model with the same mathematics,
so a refactor or a merge that must not change a recipe checks it by recomputing its receipt. GPU
kernels are qualified separately, against the blocked backend, by each run's acceptance.

Receipts are bitwise only within one environment (one lock, one CPU type); elsewhere compare the
loss and gradient norms with a tolerance.

    python -m scaffolds.transformer.receipt [--arrays] [--keys] <config.yaml>...

prints one JSON line per recipe. ``--keys`` prints only each recipe's tiny-model key (recipes with
one key share a receipt), and ``--arrays`` adds a hash and norm per array to locate a difference.
``--code-root=<checkout>`` measures that checkout's code with this receipt definition (the
environment's own installation would otherwise take precedence over PYTHONPATH).
"""

import hashlib
import json
import os
import sys

TINY_SHAPE = dict(num_layers=2, hidden_size=128, num_heads=2, num_kv_heads=2, intermediate_size=256,
                  max_seq_len=64, vocab_size=256)
SEED = 42
# Backends the CPU cannot run; their mathematics is the blocked backend's (acceptance checks the kernels).
GPU_BACKENDS = ("mosaic", "fused", "auto", "cudnn")
VERSION = 1


def tiny_model_config(model):
    """The recipe's model configuration at the receipt's shape, on a backend the CPU runs."""
    from dataclasses import replace

    backend = "blocked" if model.attention_backend in GPU_BACKENDS else model.attention_backend
    return replace(model, attention_backend=backend, **TINY_SHAPE)


def model_key(model) -> str:
    """Recipes whose tiny models have one key share a receipt."""
    from dataclasses import asdict

    text = json.dumps(asdict(tiny_model_config(model)), sort_keys=True, default=repr)
    return hashlib.sha256(text.encode()).hexdigest()


def compute(model, *, arrays: bool = False) -> dict:
    """The receipt of a model configuration: parameter and gradient hashes and the loss."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    from scaffolds.transformer.models import build_model

    tiny = tiny_model_config(model)
    network = build_model(tiny, SEED)
    tokens = (jnp.arange(2 * tiny.max_seq_len).reshape(2, tiny.max_seq_len) * 7 + 3) % tiny.vocab_size
    targets = (tokens * 5 + 1) % tiny.vocab_size

    def loss(network):
        logits = network(tokens).astype(jnp.float32)
        return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), targets[..., None], axis=-1))

    value, gradients = nnx.value_and_grad(loss)(network)
    named = {}
    for kind, state in (("param", nnx.state(network, nnx.Param)), ("grad", gradients)):
        for path, leaf in jax.tree_util.tree_leaves_with_path(state):
            named[f"{kind}{jax.tree_util.keystr(path)}"] = np.asarray(leaf)
    digest = hashlib.sha256()
    loss_value = np.asarray(value)
    digest.update(f"loss {loss_value.dtype} {loss_value.tobytes().hex()}\n".encode())
    per_array = {}
    for name in sorted(named):
        array = np.ascontiguousarray(named[name])
        header = f"{name} {array.dtype} {array.shape}\n".encode()
        digest.update(header)
        digest.update(array.tobytes())
        if arrays:
            per_array[name] = dict(sha256=hashlib.sha256(header + array.tobytes()).hexdigest()[:16],
                                   norm=float(np.linalg.norm(array.astype(np.float64).ravel())))
    receipt = dict(version=VERSION, loss=float(loss_value), digest=digest.hexdigest(), arrays_count=len(named))
    if arrays:
        receipt["arrays"] = per_array
    return receipt


def _configure_cpu() -> None:
    """Single-threaded CPU arithmetic, so a receipt does not depend on the machine's core count."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_cpu_multi_thread_eigen" not in flags:
        os.environ["XLA_FLAGS"] = (flags + " --xla_cpu_multi_thread_eigen=false").strip()


def main(argv: list[str]) -> int:
    roots = [argument.split("=", 1)[1] for argument in argv if argument.startswith("--code-root=")]
    if roots:
        if "scaffolds" in sys.modules:
            raise RuntimeError("--code-root must be selected before the scaffolds package is imported")
        sys.path.insert(0, os.path.abspath(roots[-1]))
    arrays = "--arrays" in argv
    keys_only = "--keys" in argv
    recipes = [argument for argument in argv if not argument.startswith("--")]
    _configure_cpu()
    from scaffolds.transformer.configs import load_config

    computed = {}
    for recipe in recipes:
        line = dict(recipe=recipe)
        try:
            model = load_config(recipe).model
            key = model_key(model)
            line["key"] = key
            if not keys_only:
                if key not in computed:
                    computed[key] = compute(model, arrays=arrays)
                line.update(computed[key])
        except Exception as error:  # A recipe the code rejects has that rejection as its receipt.
            line["error"] = f"{type(error).__name__}: {str(error).splitlines()[0] if str(error) else ''}"
        print(json.dumps(line), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

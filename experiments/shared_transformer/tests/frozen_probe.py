"""Subprocess probe used against both frozen source and the shared scaffold."""

import importlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


def main():
    package, settings_json, output_path, checkpoint_path, action = sys.argv[1:]
    settings = json.loads(settings_json)
    configs = importlib.import_module(f"{package}.configs")
    states = importlib.import_module(f"{package}.training.state")
    loop = importlib.import_module(f"{package}.training.loop")
    checkpoints = importlib.import_module(f"{package}.training.checkpointing")
    config = configs.Config()
    config.model = configs.ModelConfig(
        vocab_size=16, max_seq_len=4, num_layers=2, hidden_size=8,
        num_heads=2, num_kv_heads=2, qk_head_dim=2, value_head_dim=4,
        intermediate_size=12, tie_embeddings=False, **settings,
    )
    config.data.synthetic_vocab_size = 16
    config.data.sequence_length = 5
    config.data.batch_size = 2
    config.train.micro_batch_size = 2
    config.train.workdir = Path(checkpoint_path)
    config.optim.total_steps = 4
    config.optim.warmup_steps = 0
    config.optim.grad_clip_norm = 0.01
    rng = jax.random.PRNGKey(11)
    state = states.create_state(config, rng)
    arrays = {}

    def capture(name, tree):
        for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
            arrays[f"{name}:{jax.tree_util.keystr(path)}"] = np.asarray(leaf)

    capture("initial", state.params)
    if action == "restore":
        with checkpoints.make_checkpoint_manager(config) as manager:
            state, step = checkpoints.restore_latest_checkpoint(manager, state)
            assert step == 1
        capture("restored", (state.params, state.opt_state, state.step))
    batch = jnp.arange(10, dtype=jnp.int32).reshape(2, 5) % 16
    arrays["logits"] = np.asarray(nnx.merge(state.graphdef, state.params)(batch[:, :-1]))
    loss, grads = loop.loss_and_grads(state, batch, rng)
    arrays["loss"] = np.asarray(loss)
    capture("gradients", grads)
    state = state.apply_gradients(grads=grads)
    capture("updated", (state.params, state.opt_state, state.step))
    if action == "save":
        with checkpoints.make_checkpoint_manager(config) as manager:
            checkpoints.save_checkpoint(manager, 1, state, metrics={"validation_nll": float(loss)})
            manager.wait_until_finished()
    np.savez(output_path, **arrays)


if __name__ == "__main__":
    main()

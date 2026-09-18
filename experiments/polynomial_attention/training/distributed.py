from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from experiments.polynomial_attention.configs import Config
from experiments.polynomial_attention.training.randomness import model_forward_rngs
from experiments.polynomial_attention.training.state import TrainState
from experiments.polynomial_attention.training.numerics import checked_update_loss


DATA_AXIS = "data"


@dataclass(frozen=True)
class BatchLayout:
    device_count: int
    global_batch_size: int
    local_batch_size: int
    global_micro_batch_size: int
    local_micro_batch_size: int
    grad_accum_steps: int


@dataclass(frozen=True)
class DataParallelFunctions:
    mesh: Mesh
    replicated_sharding: NamedSharding
    train_batch_sharding: NamedSharding
    eval_batch_sharding: NamedSharding
    train_step: Callable[[TrainState, jax.Array, jax.Array], tuple[TrainState, jax.Array]]
    eval_step: Callable[[TrainState, jax.Array, jax.Array], jax.Array]


def resolve_batch_layout(
    config: Config,
    *,
    available_device_count: int | None = None,
) -> BatchLayout:
    if available_device_count is None:
        available_device_count = jax.local_device_count()
    if available_device_count < 1:
        raise ValueError("JAX reported no locally available devices")
    device_count = available_device_count

    global_batch_size = config.data.batch_size
    global_micro_batch_size = config.train.micro_batch_size
    if global_batch_size < 1:
        raise ValueError("data.batch_size must be positive")
    if global_micro_batch_size < 1:
        raise ValueError("train.micro_batch_size must be positive")
    if global_batch_size % global_micro_batch_size != 0:
        raise ValueError("data.batch_size must be divisible by train.micro_batch_size")
    if global_micro_batch_size % device_count != 0:
        raise ValueError(
            "train.micro_batch_size must be divisible by the data-parallel device count"
        )

    return BatchLayout(
        device_count=device_count,
        global_batch_size=global_batch_size,
        local_batch_size=global_batch_size // device_count,
        global_micro_batch_size=global_micro_batch_size,
        local_micro_batch_size=global_micro_batch_size // device_count,
        grad_accum_steps=global_batch_size // global_micro_batch_size,
    )


def select_data_parallel_devices(layout: BatchLayout) -> tuple[jax.Device, ...]:
    if jax.process_count() != 1:
        raise ValueError(
            "This scaffold supports one host only; use a single-node Slurm allocation"
        )
    devices = tuple(jax.local_devices())
    if len(devices) != layout.device_count:
        raise ValueError(
            f"Batch layout resolved {layout.device_count} devices but JAX now reports "
            f"{len(devices)} locally visible devices"
        )
    return devices


def _loss_and_grads(
    graphdef,
    params,
    optimizer_step: jax.Array,
    batch: jax.Array,
    rng: jax.Array,
):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    def loss_fn(current_params):
        model = nnx.merge(graphdef, current_params)
        logits = model(
            inputs,
            deterministic=False,
            rngs=model_forward_rngs(rng),
        )
        return _cross_entropy(logits, targets)

    # ``params`` is already a pure NNX State pytree, so the base JAX transform
    # composes cleanly with the surrounding scan and shard_map traces.
    return jax.value_and_grad(loss_fn)(params)


def _cross_entropy(logits: jax.Array, targets: jax.Array) -> jax.Array:
    with jax.named_scope("token_cross_entropy"):
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.astype(jnp.float32), targets
        ).mean()


def fold_in_data_parallel_rng(
    step_rng: jax.Array,
    microbatch_index: jax.Array | int,
    device_index: jax.Array | int,
) -> jax.Array:
    microbatch_rng = jax.random.fold_in(step_rng, microbatch_index)
    return jax.random.fold_in(microbatch_rng, device_index)


def make_data_parallel_functions(
    state: TrainState,
    devices: Sequence[jax.Device],
) -> DataParallelFunctions:
    if len(devices) < 2:
        raise ValueError("Data-parallel functions require at least two devices")

    mesh = Mesh(np.asarray(devices), (DATA_AXIS,))
    replicated = NamedSharding(mesh, P())
    train_batch_sharding = NamedSharding(mesh, P(None, DATA_AXIS, None))
    eval_batch_sharding = NamedSharding(mesh, P(DATA_AXIS, None))
    graphdef = state.graphdef

    def local_train_step(
        current_state: TrainState,
        local_batches: jax.Array,
        step_rng: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        grad_sum = jax.tree.map(
            lambda value: jax.lax.pcast(
                jnp.zeros_like(value),
                (DATA_AXIS,),
                to="varying",
            ),
            current_state.params,
        )
        loss_sum = jax.lax.pcast(
            jnp.zeros((), dtype=jnp.float32),
            (DATA_AXIS,),
            to="varying",
        )
        device_index = jax.lax.axis_index(DATA_AXIS)

        def accumulate(carry, values):
            current_loss_sum, current_grad_sum = carry
            micro_batch, micro_index = values
            micro_rng = fold_in_data_parallel_rng(
                step_rng,
                micro_index,
                device_index,
            )
            loss, grads = _loss_and_grads(
                graphdef,
                current_state.params,
                current_state.step,
                micro_batch,
                micro_rng,
            )
            return (
                current_loss_sum + loss,
                jax.tree.map(jnp.add, current_grad_sum, grads),
            ), None

        (loss_sum, grad_sum), _ = jax.lax.scan(
            accumulate,
            (loss_sum, grad_sum),
            (local_batches, jnp.arange(local_batches.shape[0], dtype=jnp.uint32)),
        )
        scale = jnp.asarray(1.0 / local_batches.shape[0], dtype=jnp.float32)
        local_loss = loss_sum * scale
        local_grads = jax.tree.map(lambda grad: grad * scale, grad_sum)

        global_loss = jax.lax.pmean(local_loss, DATA_AXIS)
        global_grads = jax.lax.pmean(local_grads, DATA_AXIS)
        updated = current_state.apply_gradients(grads=global_grads)
        return updated, checked_update_loss(current_state.params, global_grads, updated.params, global_loss)

    def local_eval_step(
        current_state: TrainState,
        local_batch: jax.Array,
        batch_rng: jax.Array,
    ) -> jax.Array:
        inputs = local_batch[:, :-1]
        targets = local_batch[:, 1:]
        local_rng = jax.random.fold_in(
            batch_rng,
            jax.lax.axis_index(DATA_AXIS),
        )
        model = nnx.merge(graphdef, current_state.params)
        logits = model(
            inputs,
            deterministic=True,
            rngs=model_forward_rngs(local_rng),
        )
        return jax.lax.pmean(_cross_entropy(logits, targets), DATA_AXIS)

    mapped_train_step = jax.shard_map(
        local_train_step,
        mesh=mesh,
        in_specs=(P(), P(None, DATA_AXIS, None), P()),
        out_specs=(P(), P()),
        # cuDNN attention's custom VJP does not currently preserve shard_map's
        # explicit varying-axis annotation even though its values are correctly
        # local to each data shard. The explicit pmean above establishes the
        # replicated output contract, so disable only this annotation checker.
        check_vma=False,
    )
    mapped_eval_step = jax.shard_map(
        local_eval_step,
        mesh=mesh,
        in_specs=(P(), P(DATA_AXIS, None), P()),
        out_specs=P(),
        check_vma=False,
    )
    compiled_train_step = jax.jit(
        mapped_train_step,
        in_shardings=(replicated, train_batch_sharding, replicated),
        out_shardings=(replicated, replicated),
        donate_argnums=(0,),
    )
    compiled_eval_step = jax.jit(
        mapped_eval_step,
        in_shardings=(replicated, eval_batch_sharding, replicated),
        out_shardings=replicated,
    )
    def train_step_with_mesh(
        current_state: TrainState,
        batches: jax.Array,
        step_rng: jax.Array,
    ) -> tuple[TrainState, jax.Array]:
        with mesh:
            return compiled_train_step(current_state, batches, step_rng)

    def eval_step_with_mesh(
        current_state: TrainState,
        batch: jax.Array,
        batch_rng: jax.Array,
    ) -> jax.Array:
        with mesh:
            return compiled_eval_step(current_state, batch, batch_rng)

    return DataParallelFunctions(
        mesh=mesh,
        replicated_sharding=replicated,
        train_batch_sharding=train_batch_sharding,
        eval_batch_sharding=eval_batch_sharding,
        train_step=train_step_with_mesh,
        eval_step=eval_step_with_mesh,
    )


def put_replicated(state: TrainState, sharding: NamedSharding) -> TrainState:
    return jax.tree.map(
        lambda value: jax.device_put(value, sharding)
        if isinstance(value, jax.Array)
        else value,
        state,
    )


def put_train_batches(batches: np.ndarray, sharding: NamedSharding) -> jax.Array:
    return jax.device_put(jnp.asarray(batches, dtype=jnp.int32), sharding)


def put_eval_batch(batch: np.ndarray, sharding: NamedSharding) -> jax.Array:
    return jax.device_put(jnp.asarray(batch, dtype=jnp.int32), sharding)

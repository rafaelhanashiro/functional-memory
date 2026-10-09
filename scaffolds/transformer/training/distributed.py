from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from scaffolds.transformer.configs import Config
from scaffolds.transformer.training.randomness import model_forward_rngs
from scaffolds.transformer.training.state import TrainState
from scaffolds.transformer.training.numerics import checked_update_loss, update_numerics
from scaffolds.transformer.training.regularization import coefficient_penalty_sum
from scaffolds.transformer.training import telemetry


DATA_AXIS = "data"
HOST_STAGING_ENV = "TRANSFORMER_HOST_STAGING"


def single_device_call(function, *args, device):
    """Run an explicitly local operation under a one-device shard-map context.

    Custom kernels can require explicit manual sharding even when a diagnostic
    uses only one device from a larger allocation. Arguments must already be
    placed on that device. Call inside a cached jit to reuse its executable.
    """
    mesh = Mesh(np.asarray([device]), ("local",))
    mapped = jax.shard_map(function, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False)
    return mapped(*args)


def _host_staging_enabled() -> bool:
    value = os.environ.get(HOST_STAGING_ENV, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{HOST_STAGING_ENV} must be 0 or 1, got {value!r}")
    return value == "1"


def _put_from_host(value, sharding: NamedSharding, *, verify: str | None = None):
    # A NumPy source sends each shard directly from host memory. Staging via
    # jnp.asarray first would put it on the default GPU and require peer copies.
    host = np.asarray(jax.device_get(value))
    placed = jax.device_put(host, sharding)
    if verify is not None:
        # Check every physical replica, not just np.asarray(placed), which may
        # return only replica zero. Process leaves/replicas sequentially so the
        # check does not retain a second full model/optimizer state on the host.
        expected_bytes = host.ravel(order="C").view(np.uint8)
        for shard in placed.addressable_shards:
            actual = np.asarray(shard.data)
            if (actual.shape != host.shape or actual.dtype != host.dtype
                    or not np.array_equal(actual.ravel(order="C").view(np.uint8), expected_bytes)):
                raise RuntimeError(
                    f"Host-staged state replica mismatch at {verify} on {shard.device}"
                )
    return placed



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
    *,
    coefficient_penalty: float = 0.0,
    coefficient_penalty_denominator: bool = True,
    telemetry_reductions: dict[str, str] | None = None,
):
    """The microbatch NLL and the gradient of NLL + coefficient_penalty * coefficient penalty sum.

    With ``telemetry_reductions`` (a dict filled at trace time with each statistic's
    reduction), also return the forward pass's recorded activation statistics.
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    def loss_fn(current_params):
        model = nnx.merge(graphdef, current_params)
        statistics = None
        if telemetry_reductions is None:
            logits = model(
                inputs,
                deterministic=False,
                rngs=model_forward_rngs(rng),
            )
        else:
            with telemetry.collect() as collector:
                logits = model(
                    inputs,
                    deterministic=False,
                    rngs=model_forward_rngs(rng),
                )
            telemetry_reductions.update(collector.reductions)
            statistics = collector.values
        nll = _cross_entropy(logits, targets)
        if coefficient_penalty:
            penalty = coefficient_penalty_sum(current_params, model.config,
                                              include_denominator=coefficient_penalty_denominator)
            return nll + coefficient_penalty * penalty, (nll, statistics)
        return nll, (nll, statistics)

    # ``params`` is already a pure NNX State pytree, so the base JAX transform
    # composes cleanly with the surrounding scan and shard_map traces.
    (_, (nll, statistics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    if telemetry_reductions is None:
        return nll, grads
    return nll, grads, statistics


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
    *, check_all_gradients: bool = False,
    coefficient_penalty: float = 0.0,
    coefficient_penalty_denominator: bool = True,
    collect_telemetry: bool = False,
) -> DataParallelFunctions:
    """With ``collect_telemetry``, the train step also returns a dict of activation
    and per-group update statistics (see ``training.telemetry``) as its last output."""
    if len(devices) < 2:
        raise ValueError("Data-parallel functions require at least two devices")

    host_staging = _host_staging_enabled()
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
        reductions: dict[str, str] = {}

        def accumulate(carry, values):
            current_loss_sum, current_grad_sum = carry
            micro_batch, micro_index = values
            micro_rng = fold_in_data_parallel_rng(
                step_rng,
                micro_index,
                device_index,
            )
            result = _loss_and_grads(
                graphdef,
                current_state.params,
                current_state.step,
                micro_batch,
                micro_rng,
                **({"coefficient_penalty": coefficient_penalty,
                    "coefficient_penalty_denominator": coefficient_penalty_denominator}
                   if coefficient_penalty else {}),
                **({"telemetry_reductions": reductions} if collect_telemetry else {}),
            )
            loss, grads, statistics = result if collect_telemetry else (*result, None)
            return (
                current_loss_sum + loss,
                jax.tree.map(jnp.add, current_grad_sum, grads),
            ), statistics

        (loss_sum, grad_sum), stacked_statistics = jax.lax.scan(
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
        checked_loss = checked_update_loss(current_state.params, global_grads, updated.params, global_loss,
                                          check_all_gradients=check_all_gradients)
        outputs = (updated, checked_loss)
        if check_all_gradients:
            outputs += (update_numerics(global_grads, updated.params, global_loss),)
        if collect_telemetry:
            statistics = telemetry.reduce_devices(
                telemetry.reduce_microbatches(stacked_statistics, reductions), reductions, DATA_AXIS)
            statistics.update(telemetry.update_statistics(current_state.params, global_grads, updated.params))
            outputs += (statistics,)
        return outputs

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
        out_specs=(P(),) * (2 + int(check_all_gradients) + int(collect_telemetry)),
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
        out_shardings=(replicated,) * (2 + int(check_all_gradients) + int(collect_telemetry)),
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
        if host_staging:
            step_rng = _put_from_host(step_rng, replicated)
        with mesh:
            return compiled_train_step(current_state, batches, step_rng)

    def eval_step_with_mesh(
        current_state: TrainState,
        batch: jax.Array,
        batch_rng: jax.Array,
    ) -> jax.Array:
        if host_staging:
            batch_rng = _put_from_host(batch_rng, replicated)
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
    if _host_staging_enabled():
        # Called after initialization and after checkpoint restoration, before
        # any update: parameters, optimizer moments, and counters all need the
        # same verified transport. Computation and precision stay unchanged.
        return jax.tree_util.tree_map_with_path(
            lambda path, value: _put_from_host(value, sharding, verify=jax.tree_util.keystr(path))
            if isinstance(value, (jax.Array, np.ndarray)) else value,
            state,
        )
    return jax.tree.map(
        lambda value: jax.device_put(value, sharding)
        if isinstance(value, jax.Array)
        else value,
        state,
    )



def put_train_batches(batches: np.ndarray, sharding: NamedSharding) -> jax.Array:
    # Packed batches originate on the host. Send each shard straight to its
    # destination instead of routing through the default device.
    return jax.device_put(np.asarray(batches, dtype=np.int32), sharding)


def put_eval_batch(batch: np.ndarray, sharding: NamedSharding) -> jax.Array:
    return jax.device_put(np.asarray(batch, dtype=np.int32), sharding)


def runtime_provenance() -> dict:
    """Execution choices belong in metadata, outside the scientific recipe."""
    return {
        "state_transfer": "host_staged_verified" if _host_staging_enabled() else "direct",
        "batch_transfer": "host_to_shard",
        "devices": [{"id": device.id, "platform": device.platform,
                     "kind": device.device_kind} for device in jax.local_devices()],
        "environment": {key: os.environ.get(key) for key in (
            HOST_STAGING_ENV, "NCCL_P2P_DISABLE", "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_FLAGS",
        )},
    }

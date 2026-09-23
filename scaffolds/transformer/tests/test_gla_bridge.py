"""JAX tracing compatibility for the vendored Triton bridge's device guard."""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, PartitionSpec as P

from scaffolds.transformer.attention.backends.ejkernel_bridge import _triton_call as bridge
from scaffolds.transformer.training.distributed import single_device_call


def test_bridge_requires_manual_axes_not_just_an_active_mesh():
    assert not bridge._in_shard_map_context()
    mesh = Mesh(np.asarray(jax.local_devices()[:1]), ("bridge_test",))
    observed = []

    def probe(x):
        observed.append(bridge._in_shard_map_context())
        return x + 1

    with jax.set_mesh(mesh):
        assert not bridge._in_shard_map_context()
        jax.jit(probe)(jnp.zeros(2)).block_until_ready()
    jax.vmap(probe, axis_name="unrelated_batch_axis")(jnp.zeros(2)).block_until_ready()
    assert observed == [False, False]


def test_bridge_detects_shared_training_shard_map_inside_scan_and_gradient():
    mesh = Mesh(np.asarray(jax.local_devices()), ("data",))
    observed = []

    def local_step(x):
        def loss(value):
            observed.append(bridge._in_shard_map_context())
            return jnp.sum(value ** 2)

        def step(carry, _):
            return carry + jax.grad(loss)(carry), None

        return jax.lax.scan(step, x, None, length=2)[0]

    mapped = jax.shard_map(local_step, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False)
    with jax.set_mesh(mesh):
        result = jax.jit(mapped)(jnp.ones(2))
    np.testing.assert_array_equal(result, np.full(2, 9))
    assert observed and all(observed)
    assert not bridge._in_shard_map_context()


def test_bridge_detects_diagnostic_single_device_call_inside_cached_nnx_jit():
    observed = []

    def local_diagnostic(x):
        observed.append(bridge._in_shard_map_context())
        return x + 2

    @nnx.jit
    def diagnostic(x):
        return single_device_call(local_diagnostic, x, device=jax.local_devices()[0])

    first = diagnostic(jnp.zeros(2))
    second = diagnostic(jnp.ones(2))
    np.testing.assert_array_equal(first, np.full(2, 2))
    np.testing.assert_array_equal(second, np.full(2, 3))
    assert observed == [True]  # the second invocation reuses its executable
    assert not bridge._in_shard_map_context()

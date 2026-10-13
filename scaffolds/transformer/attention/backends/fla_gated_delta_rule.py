"""FLA's chunked gated delta rule as a differentiable JAX operation.

The forward and backward passes launch flash-linear-attention's own Triton kernels
(``fla.ops.gated_delta_rule.chunk``) in FLA's order through the vendored bridge;
``fla_gated_delta_rule_kernels`` holds the argument-order wrappers. The two
chunk-local cumulative sums of the log decay run in JAX. Launch parameters follow
FLA's GPU branches: its autotune menus, tile sizes from the device's shared memory,
and the reduced Blackwell menus FLA uses while a Triton ``tl.dot`` recurrence race
is outstanding. FLA cannot detect these itself, because the scaffold imports it
with a CPU-only torch.

Inputs: ``q, k`` [B, T, H, K] already L2-normalized, ``v`` [B, T, H, V], the
per-token natural-log decay ``g`` [B, T, H] (<= 0) and ``beta`` [B, T, H]. Output
``o_t = S_t^T (scale q_t)`` with ``S_t = a_t S_{t-1} + b_t k_t (v_t - a_t S_{t-1}^T k_t)^T``.
The independent-gate entry point keeps the write gate beta and accepts a separate
erase gate: ``S_t = a_t (I - erase_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T``.

The chunk algebra runs in FP32 with TF32x3 products. BF16 intermediate rounding
before the backward pass's cancellations corrupts repeated-key gradients; FP32
storage with ordinary TF32 products is also insufficient. Public outputs and
input cotangents retain their input dtypes. Saved q/k/v retain their original
storage dtype and are promoted again when recomputing the backward intermediates.
"""

from functools import lru_cache, partial
import math

import jax
import jax.numpy as jnp

CHUNK_SIZE = 64
RCP_LN2 = 1.0 / math.log(2.0)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _max_shared_memory_per_block(ordinal: int) -> int:
    """The opt-in shared-memory limit Triton reports, read from the CUDA driver directly.

    Triton's own driver object asks torch whether CUDA is available, which the
    scaffold's CPU-only torch denies.
    """
    import ctypes

    cuda = ctypes.CDLL("libcuda.so.1")
    device, value = ctypes.c_int(), ctypes.c_int()
    for status in (cuda.cuInit(0), cuda.cuDeviceGet(ctypes.byref(device), ordinal),
                   # CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN
                   cuda.cuDeviceGetAttribute(ctypes.byref(value), 97, device)):
        if status:
            raise RuntimeError(f"CUDA driver query failed with status {status}")
    return value.value


@lru_cache(maxsize=None)
def _device_class() -> tuple[str, int]:
    """("hopper" | "blackwell" | "other", maximum shared memory per block) of the local GPUs."""
    device = jax.local_devices()[0]
    major = int(str(device.compute_capability).split(".")[0])
    shared = _max_shared_memory_per_block(0)
    return ("hopper" if major == 9 else "blackwell" if major in (10, 12) else "other"), shared


@lru_cache(maxsize=None)
def _kernels():
    """Autotuned wrappers with FLA's menus for this device class (imports FLA; GPU only)."""
    import triton
    from packaging.version import Version

    from scaffolds.transformer.attention.backends import fla_gated_delta_rule_kernels as raw

    arch, shared = _device_class()
    if arch == "hopper" and Version("3.4.0") <= Version(triton.__version__) < Version("3.7.1"):
        # FLA refuses this combination: chunk_bwd_dqkwg returns wrong gradients (FLA #640).
        raise RuntimeError(f"Triton {triton.__version__} on Hopper corrupts FLA's gated dqkwg; use >= 3.7.1")
    blackwell = arch == "blackwell"
    config = triton.Config
    o_warps = (2, 4) if arch == "hopper" else (2, 4, 8)
    fwd_o_configs = ([] if blackwell else [config({"BK": 128, "BV": 128}, num_warps=8, num_stages=3)]) + [
        config({"BK": 64, "BV": 64}, num_warps=4, num_stages=3),
        config({"BK": 32, "BV": 32}, num_warps=2, num_stages=3)]
    menus = {
        "kkt_solve": [config({"BK": bk}, num_warps=w) for bk in (32, 64) for w in (1, 2, 4)],
        "recompute_w_u": [config({}, num_warps=w, num_stages=s) for w in (2, 4, 8) for s in (2, 3, 4)],
        "fwd_h": [config({"BV": bv}, num_warps=w, num_stages=s)
                  for w in ((2,) if blackwell else (2, 4)) for s in (2, 3, 4) for bv in (32, 64)],
        "fwd_o": fwd_o_configs,
        "bwd_dv_local": [config({}, num_warps=w, num_stages=s) for w in o_warps for s in (2, 3, 4)],
        "bwd_dhu": [config({"BV": bv}, num_warps=w, num_stages=s) for w in (2, 4) for s in (2, 3, 4)
                    for bv in (32, 64)],
        "bwd_dqkwg": [config({}, num_warps=w, num_stages=s) for w in o_warps for s in (2, 3, 4)],
        "wy_bwd": [config({}, num_warps=w, num_stages=s)
                   for w in ((2,) if blackwell else (2, 4)) for s in ((4,) if blackwell else (2, 3, 4))],
    }
    menus["recompute_w_u_independent"] = menus["recompute_w_u"]
    menus["wy_bwd_independent"] = menus["wy_bwd"]
    # The bridge benchmarks the configurations itself; a dummy benchmarker keeps Triton's
    # Autotuner from asking its (torch-backed) driver for one.
    tuned = {name: triton.autotune(configs=configs, key=["K", "V"] if name != "kkt_solve" else ["K"],
                                   do_bench=lambda *args, **kwargs: 0.0)(getattr(raw, name))
             for name, configs in menus.items()}
    # FLA's tile caps from the device's shared memory (check_shared_mem 'hopper' / 'ada' / default).
    tile = 128 if shared >= 232448 else 64 if shared >= 101376 else 32
    wy_tile = 64 if shared >= 102400 else 32
    return tuned, tile, wy_tile


def _call(kernel, *args, out, grid, name, zeroed=(), **meta):
    from scaffolds.transformer.attention.backends.ejkernel_bridge import triton_call

    return triton_call(*args, kernel=kernel, out_shape=out, grid=grid, name=f"fla::gdn::{name}",
                       zeroed_outputs=zeroed, default_dot_input_precision="tf32x3", **meta)


def _shape(shape, dtype):
    return jax.ShapeDtypeStruct(shape, dtype)


def _chunk_cumsum(g, *, reverse=False):
    """Cumulative sum within each chunk of ``CHUNK_SIZE`` tokens, in FP32 ([B, T, H])."""
    batch, length, heads = g.shape
    chunks = g.astype(jnp.float32).reshape(batch, length // CHUNK_SIZE, CHUNK_SIZE, heads)
    if reverse:
        chunks = jnp.flip(jnp.cumsum(jnp.flip(chunks, 2), axis=2), 2)
    else:
        chunks = jnp.cumsum(chunks, axis=2)
    return chunks.reshape(batch, length, heads)


def _prepare_w_u(kernels, k, v, g, beta, a, erase=None):
    """Share chunk machinery; only WY preparation differs for independent gates."""
    batch, length, heads, key_dim = k.shape
    name = "recompute_w_u" if erase is None else "recompute_w_u_independent"
    operands = (k, v, beta, a, g, length) if erase is None else (k, v, beta, erase, a, g, length)
    return _call(kernels[0][name], *operands,
                 out=[_shape(k.shape, k.dtype), _shape(v.shape, v.dtype)],
                 grid=(length // CHUNK_SIZE, batch * heads), name=name,
                 H=heads, HV=heads, K=key_dim, V=v.shape[-1], BT=CHUNK_SIZE, BK=64, BV=64)


def _forward_pieces(kernels, k, v, g, beta, erase=None):
    tuned, _, _ = kernels
    batch, length, heads, key_dim = k.shape
    value_dim, chunks = v.shape[-1], length // CHUNK_SIZE
    meta = dict(H=heads, HV=heads, K=key_dim, BT=CHUNK_SIZE)
    erase_gate = beta if erase is None else erase
    a = _call(tuned["kkt_solve"], k, g, erase_gate, length, out=_shape((batch, length, heads, CHUNK_SIZE), k.dtype),
              grid=(chunks, batch * heads), name="kkt_solve", zeroed=(0,), BC=16, **meta)
    w, u = _prepare_w_u(kernels, k, v, g, beta, a, erase)
    v_new, h = _call(tuned["fwd_h"], k, u, w, g, length,
                     out=[_shape(v.shape, v.dtype), _shape((batch, chunks, heads, key_dim, value_dim), k.dtype)],
                     grid=lambda m: (-(-value_dim // m["BV"]) * batch * heads,), name="fwd_h", V=value_dim, **meta)
    return a, w, v_new, h


def _gated_delta_rule_fwd(q, k, v, g, beta, scale, erase=None):
    original_q, original_k, original_v, original_beta, original_erase = q, k, v, beta, erase
    q, k, v, beta = (a.astype(jnp.float32) for a in (q, k, v, beta))
    erase = None if erase is None else erase.astype(jnp.float32)
    kernels = _kernels()
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if length % CHUNK_SIZE:
        raise ValueError(f"Sequence length must be a multiple of {CHUNK_SIZE}")
    g_chunk = _chunk_cumsum(g) * RCP_LN2  # FLA's kernels use exp2 of the base-2 chunk-local cumulative decay.
    a, _, v_new, h = _forward_pieces(kernels, k, v, g_chunk, beta, erase)
    o = _call(kernels[0]["fwd_o"], q, k, v_new, h, g_chunk, float(scale), length, out=_shape(v.shape, v.dtype),
              grid=lambda m: (-(-value_dim // m["BV"]), length // CHUNK_SIZE, batch * heads), name="fwd_o",
              H=heads, HV=heads, K=key_dim, V=value_dim, BT=CHUNK_SIZE)
    residuals = (original_q, original_k, original_v, g_chunk, original_beta, a)
    return o.astype(original_v.dtype), residuals if erase is None else (*residuals, original_erase)


def _gated_delta_rule_bwd(scale, residuals, do):
    q, k, v, g_chunk, beta, a = residuals[:6]
    erase = residuals[6] if len(residuals) == 7 else None
    input_dtypes = (q.dtype, k.dtype, v.dtype, jnp.float32, beta.dtype)
    if erase is not None:
        input_dtypes = (*input_dtypes, erase.dtype)
    q, k, v, beta = (a.astype(jnp.float32) for a in (q, k, v, beta))
    erase = None if erase is None else erase.astype(jnp.float32)
    kernels = _kernels()
    tuned, tile, wy_tile = kernels
    batch, length, heads, key_dim = q.shape
    value_dim, chunks = v.shape[-1], length // CHUNK_SIZE
    do = do.astype(v.dtype)
    meta = dict(H=heads, HV=heads, K=key_dim, V=value_dim, BT=CHUNK_SIZE)
    bk = min(max(_next_power_of_2(key_dim), 16), tile)
    bv = min(max(_next_power_of_2(value_dim), 16), tile)
    w, u = _prepare_w_u(kernels, k, v, g_chunk, beta, a, erase)
    v_new, h = _call(tuned["fwd_h"], k, u, w, g_chunk, length,
                     out=[_shape(v.shape, v.dtype), _shape((batch, chunks, heads, key_dim, value_dim), k.dtype)],
                     grid=lambda m: (-(-value_dim // m["BV"]) * batch * heads,), name="fwd_h", **meta)
    dv = _call(tuned["bwd_dv_local"], q, k, g_chunk, do, float(scale), length, out=_shape(v.shape, do.dtype),
               grid=(chunks, batch * heads), name="bwd_dv_local", BK=bk, BV=bv, **meta)
    dh, dv = _call(tuned["bwd_dhu"], q, k, w, g_chunk, do, dv, float(scale), length,
                   out=[_shape((batch, chunks, heads, key_dim, value_dim), q.dtype), _shape(dv.shape, dv.dtype)],
                   grid=lambda m: (-(-value_dim // m["BV"]) * batch * heads,), name="bwd_dhu", **meta)
    nk = -(-key_dim // bk)
    dq, dk, dw, dg = _call(tuned["bwd_dqkwg"], q, k, v_new, g_chunk, h, do, dh, dv, float(scale), length,
                           out=[_shape(q.shape, q.dtype), _shape(k.shape, k.dtype), _shape(w.shape, w.dtype),
                                _shape((nk, batch, length, heads), jnp.float32)],
                           grid=(nk, chunks, batch * heads), name="bwd_dqkwg", B=batch, BK=bk, BV=bv, **meta)
    wy_bk = min(max(_next_power_of_2(key_dim), 16), wy_tile)
    wy_bv = min(max(_next_power_of_2(value_dim), 16), wy_tile)
    if erase is None:
        dk2, dv, db, dg2 = _call(tuned["wy_bwd"], k, v, beta, g_chunk, a, dw, dv, length,
                                 out=[_shape(k.shape, k.dtype), _shape(v.shape, v.dtype), _shape(beta.shape, beta.dtype),
                                      _shape(g_chunk.shape, jnp.float32)],
                                 grid=(chunks, batch * heads), name="wy_bwd", BK=wy_bk, BV=wy_bv, **meta)
    else:
        dk2, dv, db, de, dg2 = _call(tuned["wy_bwd_independent"], k, v, beta, erase, g_chunk, a, dw, dv, length,
                                     out=[_shape(k.shape, k.dtype), _shape(v.shape, v.dtype),
                                          _shape(beta.shape, beta.dtype), _shape(erase.shape, erase.dtype),
                                          _shape(g_chunk.shape, jnp.float32)],
                                     grid=(chunks, batch * heads), name="wy_bwd_independent",
                                     BK=wy_bk, BV=wy_bv, **meta)
    dk = (dk.astype(jnp.float32) + dk2.astype(jnp.float32)).astype(k.dtype)
    dg = _chunk_cumsum(dg.sum(0) + dg2, reverse=True)
    gradients = (dq, dk, dv, dg, db)
    gradients = gradients if erase is None else (*gradients, de)
    return tuple(gradient.astype(dtype) for gradient, dtype in zip(gradients, input_dtypes, strict=True))


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def fla_gated_delta_rule(q, k, v, g, beta, scale):
    """Chunked gated delta rule with FLA's kernels; differentiable in q, k, v, g, and beta."""
    return _gated_delta_rule_fwd(q, k, v, g, beta, scale)[0]


def _fwd(q, k, v, g, beta, scale):
    return _gated_delta_rule_fwd(q, k, v, g, beta, scale)


def _bwd(scale, residuals, do):
    dq, dk, dv, dg, db = _gated_delta_rule_bwd(scale, residuals, do)
    _, _, _, g_chunk, beta, _ = residuals
    return dq, dk, dv, dg.astype(jnp.float32), db.astype(beta.dtype)


fla_gated_delta_rule.defvjp(_fwd, _bwd)


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def fla_independent_gated_delta_rule(q, k, v, g, beta, erase, scale):
    """GDN with independent write beta and erase gates, including exact zero erase."""
    return _gated_delta_rule_fwd(q, k, v, g, beta, scale, erase)[0]


def _independent_fwd(q, k, v, g, beta, erase, scale):
    return _gated_delta_rule_fwd(q, k, v, g, beta, scale, erase)


def _independent_bwd(scale, residuals, do):
    dq, dk, dv, dg, db, de = _gated_delta_rule_bwd(scale, residuals, do)
    beta, erase = residuals[4], residuals[6]
    return dq, dk, dv, dg.astype(jnp.float32), db.astype(beta.dtype), de.astype(erase.dtype)


fla_independent_gated_delta_rule.defvjp(_independent_fwd, _independent_bwd)

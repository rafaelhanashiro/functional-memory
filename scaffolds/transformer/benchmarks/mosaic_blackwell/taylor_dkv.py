"""Prototype: Blackwell dV and dK for normalized even-degree Taylor attention (mirrors fused_taylor._dkv_kernel).

Transposed tiles: lanes are keys. S^T = K Q^T / sqrt(d), W^T = p(S^T) / Z, dP^T = V dO^T,
dS^T = (dP^T - delta) * p'(S^T) / Z (query vectors broadcast along columns).
which="dv": dV = W^T dO.  which="dk": dK = dS^T Q / sqrt(d).
FP32 W^T / dS^T enter the MMA as three truncated BF16 components in TMEM, aliased over S^T (and dP^T).
CTA = two compute warpgroups (128 keys each) + a producer warpgroup (TMA warp, MMA warp).
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

L = plgpu.Layout.TCGEN05
BK = 128
WGS = 2


def _transforms(cols, dtype):
    swizzle = plgpu.find_swizzle(cols * jnp.dtype(dtype).itemsize * 8)
    return (plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle))


def _truncate(x):
    bits = lax.bitcast_convert_type(x, jnp.uint32)
    return lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), jnp.float32)


def _taylor(s, degree):
    value = jnp.full_like(s, 1.0 / math.factorial(degree))
    for order in range(degree - 1, -1, -1):
        value = value * s + 1.0 / math.factorial(order)
    return value


@partial(jax.jit, static_argnames=("which", "degree", "block_q", "stages"))
def dkv(q, k, v, do, delta, inv_z, *, which, degree, block_q=64, stages=3):
    """delta, inv_z: [batch, heads, time] FP32. Returns dV or dK [batch, time, heads, width] in its input dtype."""
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    assert which in ("dv", "dk") and time % (BK * WGS) == 0 and BK % block_q == 0
    scale = 1.0 / math.sqrt(dqk)
    width = dv if which == "dv" else dqk
    per_wg = BK // block_q  # query tiles per warpgroup key range

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, delta_gmem, invz_gmem, out_gmem,
               k_smem, v_smem, q_smem, do_smem, vec_smem, out_smem, regions, acc_tmem,
               in_bar, tile_bar, tile_free, s_ready, ds_ready, done):
        wg = lax.axis_index("wg")
        b, h = lax.axis_index("b"), lax.axis_index("h")
        k_index = lax.axis_index("k")
        base = k_index * (BK * WGS)
        start = k_index * (BK * WGS // block_q)  # first query tile of this CTA
        n_tiles = time // block_q - start

        def first(w):  # first local tile index touching warpgroup w's keys
            return w * per_wg

        def compute(w):
            k0 = base + w * BK
            plgpu.barrier_wait(in_bar.at[0])

            def body(j, masked):
                if which == "dv":
                    s_tmem, parts_tmem = regions[w]
                else:
                    (s_tmem, dp_tmem), parts_tmem = regions[w]
                plgpu.barrier_wait(s_ready.at[w])
                s = plgpu.async_load_tmem(s_tmem, layout=L) * scale
                if which == "dk":
                    dp = plgpu.async_load_tmem(dp_tmem, layout=L)
                plgpu.wait_load_tmem()
                cols = pl.ds((start + j) * block_q, block_q)
                invz = plgpu.load(vec_smem.at[1], cols, layout=L.reduce(0), optimized=False)
                invz = plgpu.layout_cast(lax.broadcast_in_dim(invz, s.shape, [1]), L)
                if which == "dv":
                    x = _taylor(s, degree) * invz
                else:
                    d = plgpu.layout_cast(lax.broadcast_in_dim(plgpu.load(vec_smem.at[0], cols, layout=L.reduce(0), optimized=False), s.shape, [1]), L)
                    x = (dp - d) * _taylor(s, degree - 1) * invz
                if masked:
                    keys = plgpu.broadcasted_iota(jnp.int32, (BK, block_q), 0, layout=L) + k0
                    queries = plgpu.broadcasted_iota(jnp.int32, (BK, block_q), 1, layout=L) + (start + j) * block_q
                    x = jnp.where(queries >= keys, x, 0.0)
                high = _truncate(x)
                rest = x - high
                middle = _truncate(rest)
                for i, part in enumerate((high, middle, rest - middle)):
                    plgpu.async_store_tmem(parts_tmem[i], part.astype(jnp.bfloat16))
                plgpu.commit_tmem()
                plgpu.barrier_arrive(ds_ready.at[w])

            lax.fori_loop(first(w), first(w) + per_wg, lambda j, c: (body(j, True), c)[1], None)
            lax.fori_loop(first(w) + per_wg, n_tiles, lambda j, c: (body(j, False), c)[1], None)
            plgpu.barrier_wait(done.at[w])
            acc = plgpu.async_load_tmem(acc_tmem.at[:, pl.ds(w * width, width)], layout=L)
            plgpu.wait_load_tmem()
            if which == "dk":
                acc = acc * scale
            out_smem.at[w][...] = acc.astype(out_smem.dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(out_smem.at[w], out_gmem.at[b, pl.ds(k0, BK), h])
            plgpu.wait_smem_to_gmem(0)

        for w in range(WGS):
            pl.when(wg == w)(partial(compute, w))

        @pl.when(wg == WGS)
        def _producer():
            @pl.core_map(plgpu.WarpMesh(axis_name="warp"))
            def _per_warp():
                warp = lax.axis_index("warp")

                @pl.when(warp == 0)
                def _tma():
                    for w in range(WGS):
                        rows = pl.ds(base + w * BK, BK)
                        plgpu.copy_gmem_to_smem(k_gmem.at[b, rows, h], k_smem.at[w], in_bar.at[0])
                        plgpu.copy_gmem_to_smem(v_gmem.at[b, rows, h], v_smem.at[w], in_bar.at[0])
                    for c in range(time // 256):  # TMA moves at most 256 elements per dimension
                        chunk = pl.ds(c * 256, 256)
                        plgpu.copy_gmem_to_smem(delta_gmem.at[b, h, chunk], vec_smem.at[0, chunk], in_bar.at[0])
                        plgpu.copy_gmem_to_smem(invz_gmem.at[b, h, chunk], vec_smem.at[1, chunk], in_bar.at[0])

                    def load(j, _):
                        slot = lax.rem(j, stages)

                        @pl.when(j >= stages)
                        def _():
                            plgpu.barrier_wait(tile_free.at[slot])

                        rows = pl.ds((start + j) * block_q, block_q)
                        plgpu.copy_gmem_to_smem(q_gmem.at[b, rows, h], q_smem.at[slot], tile_bar.at[slot])
                        plgpu.copy_gmem_to_smem(do_gmem.at[b, rows, h], do_smem.at[slot], tile_bar.at[slot])

                    lax.fori_loop(0, n_tiles, load, None)

                @pl.when(warp == 1)
                def _mma():
                    plgpu.barrier_wait(in_bar.at[0])

                    def issue_s(j, w):
                        slot = lax.rem(j, stages)
                        if which == "dv":
                            s_tmem, _ = regions[w]
                        else:
                            (s_tmem, dp_tmem), _ = regions[w]
                        plgpu.tcgen05_mma(s_tmem, k_smem.at[w], plgpu.transpose_ref(q_smem.at[slot], (1, 0)), accumulate=False)
                        if which == "dk":
                            plgpu.tcgen05_mma(dp_tmem, v_smem.at[w], plgpu.transpose_ref(do_smem.at[slot], (1, 0)),
                                              accumulate=False)
                        plgpu.tcgen05_commit_arrive(s_ready.at[w])

                    plgpu.barrier_wait(tile_bar.at[0])
                    issue_s(0, 0)

                    def step(j, _):
                        slot = lax.rem(j, stages)
                        for w in range(WGS):
                            @pl.when(j >= first(w))
                            def _():
                                plgpu.barrier_wait(ds_ready.at[w])
                                _, parts_tmem = regions[w]
                                rhs = do_smem.at[slot] if which == "dv" else q_smem.at[slot]
                                for i in range(3):
                                    plgpu.tcgen05_mma(acc_tmem.at[:, pl.ds(w * width, width)], parts_tmem[i], rhs,
                                                      accumulate=jnp.logical_or(j > first(w), i > 0))

                                @pl.when(j == n_tiles - 1)
                                def _():
                                    plgpu.tcgen05_commit_arrive(done.at[w])
                        plgpu.tcgen05_commit_arrive(tile_free.at[slot])

                        @pl.when(j + 1 < n_tiles)
                        def _():
                            plgpu.barrier_wait(tile_bar.at[lax.rem(j + 1, stages)])
                            for w in range(WGS):
                                @pl.when(j + 1 >= first(w))  # after its tile-j MMAs: in-order tensor core
                                def _():
                                    issue_s(j + 1, w)

                    lax.fori_loop(0, n_tiles, step, None)

    t = lambda cols: _transforms(cols, jnp.bfloat16)
    s_region = plgpu.TMEM((BK, block_q), jnp.float32)
    if which == "dk":
        s_region = (s_region, plgpu.TMEM((BK, block_q), jnp.float32))
    out_dtype = v.dtype if which == "dv" else k.dtype
    f = plgpu.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct((batch, time, heads, width), out_dtype),
        grid=(time // (BK * WGS), batch, heads), grid_names=("k", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=dict(
            k_smem=plgpu.SMEM((WGS, BK, dqk), jnp.bfloat16, transforms=t(dqk)),
            v_smem=plgpu.SMEM((WGS, BK, dv), jnp.bfloat16, transforms=t(dv)),
            q_smem=plgpu.SMEM((stages, block_q, dqk), jnp.bfloat16, transforms=t(dqk)),
            do_smem=plgpu.SMEM((stages, block_q, dv), jnp.bfloat16, transforms=t(dv)),
            vec_smem=plgpu.SMEM((2, time), jnp.float32),
            out_smem=plgpu.SMEM((WGS, BK, width), out_dtype),
            regions=tuple(plgpu.RefUnion(
                s_region, tuple(plgpu.TMEM((BK, block_q), jnp.bfloat16, packed=True) for _ in range(3)))
                for _ in range(WGS)),
            acc_tmem=plgpu.TMEM((BK, WGS * width), jnp.float32),
            in_bar=plgpu.Barrier(num_arrivals=2 * WGS + 2 * (time // 256), num_barriers=1),
            tile_bar=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
            tile_free=plgpu.Barrier(num_arrivals=1, num_barriers=stages, orders_tensor_core=True),
            s_ready=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            ds_ready=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            done=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
        ),
    )
    return f(q, k, v, do, delta, inv_z)

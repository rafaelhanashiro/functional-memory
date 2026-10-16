"""Prototype: Blackwell dQ for normalized even-degree Taylor attention (mirrors fused_taylor._dq_kernel).

dS = (dO V^T - delta) * p'(s) / Z with delta = rowsum(dO * O) and Z the forward row sum;
dQ = dS K / sqrt(d). The FP32 dS enters the MMA as three truncated BF16 components in TMEM.
CTA = two compute warpgroups (128 queries each) + a producer warpgroup (TMA warp, MMA warp).
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

L = plgpu.Layout.TCGEN05
BQ = 128
WGS = 2


def _transforms(cols, dtype):
    swizzle = plgpu.find_swizzle(cols * jnp.dtype(dtype).itemsize * 8)
    return (plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle))


def _truncate(x):
    bits = lax.bitcast_convert_type(x, jnp.uint32)
    return lax.bitcast_convert_type(bits & jnp.uint32(0xFFFF0000), jnp.float32)


def _taylor(s, degree):
    """T_degree(s) by Horner on the coefficients 1/r!."""
    value = jnp.full_like(s, 1.0 / math.factorial(degree))
    for order in range(degree - 1, -1, -1):
        value = value * s + 1.0 / math.factorial(order)
    return value


@partial(jax.jit, static_argnames=("degree", "block_k", "stages"))
def dq(q, k, v, do, delta, inv_z, *, degree, block_k=64, stages=3):
    """delta, inv_z: [batch, heads, time] FP32. Returns dQ [batch, time, heads, dq] in q.dtype."""
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    assert time % (BQ * WGS) == 0 and BQ % block_k == 0 and degree >= 1
    scale = 1.0 / math.sqrt(dqk)
    nb = 1  # one S/dP region per warpgroup, reused for the dS components

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, delta_gmem, invz_gmem, dq_gmem,
               q_smem, do_smem, k_smem, v_smem, vec_smem, out_smem, regions, acc_tmem,
               in_bar, kv_bar, kv_free, s_ready, ds_ready, done):
        wg = lax.axis_index("wg")
        b, h = lax.axis_index("b"), lax.axis_index("h")
        q_index = lax.axis_index("q")
        base = q_index * (BQ * WGS)

        def tiles(w):
            return q_index * (BQ * WGS // block_k) + (w + 1) * (BQ // block_k)

        n_tiles = tiles(WGS - 1)


        def compute(w):
            q0 = base + w * BQ
            plgpu.barrier_wait(in_bar.at[0])
            delta_w = plgpu.layout_cast(vec_smem.at[w, 0][...], L.reduce(1))
            invz_w = plgpu.layout_cast(vec_smem.at[w, 1][...], L.reduce(1))

            def body(j, masked):
                (s_tmem, dp_tmem), ds_tmem = regions[w]
                plgpu.barrier_wait(s_ready.at[w])
                s = plgpu.async_load_tmem(s_tmem, layout=L) * scale
                dp = plgpu.async_load_tmem(dp_tmem, layout=L)
                plgpu.wait_load_tmem()
                ds = ((dp - lax.broadcast_in_dim(delta_w, dp.shape, [0])) * _taylor(s, degree - 1)
                      * lax.broadcast_in_dim(invz_w, dp.shape, [0]))
                if masked:
                    rows = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 0, layout=L) + q0
                    kcols = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 1, layout=L) + j * block_k
                    ds = jnp.where(rows >= kcols, ds, 0.0)
                high = _truncate(ds)
                rest = ds - high
                middle = _truncate(rest)

                for i, part in enumerate((high, middle, rest - middle)):  # overwrite this tile's S/dP region
                    plgpu.async_store_tmem(ds_tmem[i], part.astype(jnp.bfloat16))
                plgpu.commit_tmem()
                plgpu.barrier_arrive(ds_ready.at[w])

            full = q_index * (BQ * WGS // block_k) + w * (BQ // block_k)
            lax.fori_loop(0, full, lambda j, c: (body(j, False), c)[1], None)
            lax.fori_loop(full, tiles(w), lambda j, c: (body(j, True), c)[1], None)
            plgpu.barrier_wait(done.at[w])
            acc = plgpu.async_load_tmem(acc_tmem.at[:, pl.ds(w * dqk, dqk)], layout=L)
            plgpu.wait_load_tmem()
            out_smem.at[w][...] = (acc * scale).astype(out_smem.dtype)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(out_smem.at[w], dq_gmem.at[b, pl.ds(q0, BQ), h])
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
                        rows = pl.ds(base + w * BQ, BQ)
                        plgpu.copy_gmem_to_smem(q_gmem.at[b, rows, h], q_smem.at[w], in_bar.at[0])
                        plgpu.copy_gmem_to_smem(do_gmem.at[b, rows, h], do_smem.at[w], in_bar.at[0])
                        plgpu.copy_gmem_to_smem(delta_gmem.at[b, h, rows], vec_smem.at[w, 0], in_bar.at[0])
                        plgpu.copy_gmem_to_smem(invz_gmem.at[b, h, rows], vec_smem.at[w, 1], in_bar.at[0])

                    def load(j, _):
                        slot = lax.rem(j, stages)

                        @pl.when(j >= stages)
                        def _():
                            plgpu.barrier_wait(kv_free.at[slot])

                        rows = pl.ds(j * block_k, block_k)
                        plgpu.copy_gmem_to_smem(k_gmem.at[b, rows, h], k_smem.at[slot], kv_bar.at[slot])
                        plgpu.copy_gmem_to_smem(v_gmem.at[b, rows, h], v_smem.at[slot], kv_bar.at[slot])

                    lax.fori_loop(0, n_tiles, load, None)

                @pl.when(warp == 1)
                def _mma():
                    plgpu.barrier_wait(in_bar.at[0])

                    def issue_s(j, w):  # S_j and dP_j for warpgroup w into its region
                        slot = lax.rem(j, stages)
                        (s_tmem, dp_tmem), _ = regions[w]
                        plgpu.tcgen05_mma(s_tmem, q_smem.at[w], plgpu.transpose_ref(k_smem.at[slot], (1, 0)), accumulate=False)
                        plgpu.tcgen05_mma(dp_tmem, do_smem.at[w], plgpu.transpose_ref(v_smem.at[slot], (1, 0)), accumulate=False)
                        plgpu.tcgen05_commit_arrive(s_ready.at[w])

                    plgpu.barrier_wait(kv_bar.at[0])
                    for w in range(WGS):
                        issue_s(0, w)

                    def step(j, _):
                        slot = lax.rem(j, stages)

                        @pl.when(j + 1 < n_tiles)
                        def _():
                            plgpu.barrier_wait(kv_bar.at[lax.rem(j + 1, stages)])

                        for w in range(WGS):
                            @pl.when(j < tiles(w))
                            def _():
                                plgpu.barrier_wait(ds_ready.at[w])
                                _, ds_tmem = regions[w]
                                for i in range(3):
                                    plgpu.tcgen05_mma(acc_tmem.at[:, pl.ds(w * dqk, dqk)], ds_tmem[i], k_smem.at[slot],
                                                      accumulate=jnp.logical_or(j > 0, i > 0))

                                @pl.when(j == tiles(w) - 1)
                                def _():
                                    plgpu.tcgen05_commit_arrive(done.at[w])

                                @pl.when(j + 1 < tiles(w))  # in-order tensor core: dQ_j reads dS before S_(j+1) lands
                                def _():
                                    issue_s(j + 1, w)
                        plgpu.tcgen05_commit_arrive(kv_free.at[slot])

                    lax.fori_loop(0, n_tiles, step, None)

    t = lambda cols: _transforms(cols, jnp.bfloat16)
    f = plgpu.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(time // (BQ * WGS), batch, heads), grid_names=("q", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=dict(
            q_smem=plgpu.SMEM((WGS, BQ, dqk), jnp.bfloat16, transforms=t(dqk)),
            do_smem=plgpu.SMEM((WGS, BQ, dv), jnp.bfloat16, transforms=t(dv)),
            k_smem=plgpu.SMEM((stages, block_k, dqk), jnp.bfloat16, transforms=t(dqk)),
            v_smem=plgpu.SMEM((stages, block_k, dv), jnp.bfloat16, transforms=t(dv)),
            vec_smem=plgpu.SMEM((WGS, 2, BQ), jnp.float32),
            out_smem=plgpu.SMEM((WGS, BQ, dqk), q.dtype),
            regions=tuple(plgpu.RefUnion(
                (plgpu.TMEM((BQ, block_k), jnp.float32), plgpu.TMEM((BQ, block_k), jnp.float32)),
                tuple(plgpu.TMEM((BQ, block_k), jnp.bfloat16, packed=True) for _ in range(3))) for _ in range(WGS)),
            acc_tmem=plgpu.TMEM((BQ, WGS * dqk), jnp.float32),
            in_bar=plgpu.Barrier(num_arrivals=4 * WGS, num_barriers=1),
            kv_bar=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
            kv_free=plgpu.Barrier(num_arrivals=1, num_barriers=stages, orders_tensor_core=True),
            s_ready=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            ds_ready=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            done=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
        ),
    )
    return f(q, k, v, do, delta, inv_z)

"""Blackwell (tcgen05) Mosaic GPU kernels for normalized causal Taylor attention.

Each CTA has two compute warpgroups and a producer warpgroup (warp 0 issues TMA, warp 1
issues tcgen05 MMAs). Scores, polynomials, sums, and accumulators are FP32; FP32 weights
and score gradients enter the MMA as three truncated BF16 components stored in tensor
memory. forward returns the FP32 output and row sums; dq and dkv compute the
gradients from delta = rowsum(dO * O) and 1 / Z ([batch, heads, time]). The caller
(taylor.py) validates shapes; scale is 1 / sqrt(d_K) for the unpadded width.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from scaffolds.transformer.attention.backends.mosaic.common import finalize as _finalize, taylor as _taylor, truncate as _truncate

L = plgpu.Layout.TCGEN05
WGS = 2


def _transforms(cols, dtype):
    swizzle = plgpu.find_swizzle(cols * jnp.dtype(dtype).itemsize * 8)
    return (plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle))


@partial(jax.jit, static_argnames=("degree", "scale", "block_k", "stages"))
def forward(q, k, v, *, degree, scale, block_k=64, stages=3):
    BQ = 128
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]

    def kernel(q_gmem, k_gmem, v_gmem, out_gmem, norm_gmem,
               q_smem, k_smem, v_smem, o_smem, n_smem, s_tmem, o_tmem, p_tmem,
               q_bar, kv_bar, kv_free, s_ready, s_free, p_ready, p_free, o_done):
        wg = lax.axis_index("wg")
        b, h = lax.axis_index("b"), lax.axis_index("h")
        q_index = lax.axis_index("q")
        base = q_index * (BQ * WGS)

        def tiles(w):  # causal KV tiles needed by compute warpgroup w (exact: block_k divides BQ)
            return q_index * (BQ * WGS // block_k) + (w + 1) * (BQ // block_k)

        n_tiles = tiles(WGS - 1)

        def compute(w):
            q0 = base + w * BQ
            my_tiles = tiles(w)

            def body(j, rowsum, masked):
                plgpu.barrier_wait(s_ready.at[w])
                s = plgpu.async_load_tmem(s_tmem.at[:, pl.ds(w * block_k, block_k)], layout=L) * scale
                plgpu.wait_load_tmem()
                plgpu.barrier_arrive(s_free.at[w])
                if masked:  # tiles crossing the causal diagonal
                    rows = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 0, layout=L) + q0
                    cols = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 1, layout=L) + j * block_k
                    causal = rows >= cols
                    p = jnp.where(causal, _taylor(jnp.where(causal, s, 0.0), degree), 0.0)
                else:
                    p = _taylor(s, degree)
                # A nonfinite score or weight always makes the row sum nonfinite (opposite
                # infinities give NaN), so the final row-sum test detects it exactly.
                rowsum = rowsum + p.sum(axis=1)
                high = _truncate(p)
                rest = p - high
                middle = _truncate(rest)
                parts = (high, middle, rest - middle)

                @pl.when(j > 0)
                def _():
                    plgpu.barrier_wait(p_free.at[w])  # PV of tile j-1 has read the weights

                for i, part in enumerate(parts):
                    plgpu.async_store_tmem(p_tmem[w][i], part.astype(jnp.bfloat16))
                plgpu.commit_tmem()
                plgpu.barrier_arrive(p_ready.at[w])
                return rowsum

            zeros = plgpu.layout_cast(jnp.zeros((BQ,), jnp.float32), L.reduce(1))
            full = q_index * (BQ * WGS // block_k) + w * (BQ // block_k)  # tiles strictly below the diagonal
            rowsum = lax.fori_loop(0, full, lambda j, r: body(j, r, False), zeros)
            rowsum = lax.fori_loop(full, my_tiles, lambda j, r: body(j, r, True), rowsum)
            plgpu.barrier_wait(o_done.at[w])
            o = plgpu.async_load_tmem(o_tmem.at[:, pl.ds(w * dv, dv)], layout=L)
            plgpu.wait_load_tmem()
            out, denominator = _finalize(o, rowsum, degree)
            o_smem.at[w][...] = out
            n_smem.at[w][...] = denominator
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem.at[w], out_gmem.at[b, pl.ds(q0, BQ), h])
            plgpu.copy_smem_to_gmem(n_smem.at[w], norm_gmem.at[b, h, pl.ds(q0, BQ)])
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
                        plgpu.copy_gmem_to_smem(q_gmem.at[b, pl.ds(base + w * BQ, BQ), h], q_smem.at[w], q_bar.at[0])

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
                    plgpu.barrier_wait(q_bar.at[0])

                    def issue_s(j):  # S_j for both warpgroups (K_j must be resident)
                        slot = lax.rem(j, stages)
                        plgpu.barrier_wait(kv_bar.at[slot])
                        for w in range(WGS):
                            @pl.when(j < tiles(w))
                            def _():
                                @pl.when(j >= 1)
                                def _():
                                    plgpu.barrier_wait(s_free.at[w])  # S_(j-1) has been loaded
                                plgpu.tcgen05_mma(s_tmem.at[:, pl.ds(w * block_k, block_k)], q_smem.at[w],
                                                  plgpu.transpose_ref(k_smem.at[slot], (1, 0)), accumulate=False)
                                plgpu.tcgen05_commit_arrive(s_ready.at[w])

                    issue_s(0)

                    def step(j, _):
                        slot = lax.rem(j, stages)

                        @pl.when(j + 1 < n_tiles)
                        def _():
                            issue_s(j + 1)

                        for w in range(WGS):
                            @pl.when(j < tiles(w))
                            def _():
                                plgpu.barrier_wait(p_ready.at[w])
                                for i in range(3):
                                    plgpu.tcgen05_mma(o_tmem.at[:, pl.ds(w * dv, dv)], p_tmem[w][i], v_smem.at[slot],
                                                      accumulate=jnp.logical_or(j > 0, i > 0))
                                plgpu.tcgen05_commit_arrive(p_free.at[w])

                                @pl.when(j == tiles(w) - 1)
                                def _():
                                    plgpu.tcgen05_commit_arrive(o_done.at[w])
                        plgpu.tcgen05_commit_arrive(kv_free.at[slot])

                    lax.fori_loop(0, n_tiles, step, None)

    f = plgpu.kernel(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((batch, time, heads, dv), jnp.float32),
                   jax.ShapeDtypeStruct((batch, heads, time), jnp.float32)),
        grid=(time // (BQ * WGS), batch, heads), grid_names=("q", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=dict(
            q_smem=plgpu.SMEM((WGS, BQ, dq), jnp.bfloat16, transforms=_transforms(dq, jnp.bfloat16)),
            k_smem=plgpu.SMEM((stages, block_k, dq), jnp.bfloat16, transforms=_transforms(dq, jnp.bfloat16)),
            v_smem=plgpu.SMEM((stages, block_k, dv), jnp.bfloat16, transforms=_transforms(dv, jnp.bfloat16)),
            o_smem=plgpu.SMEM((WGS, BQ, dv), jnp.float32),
            n_smem=plgpu.SMEM((WGS, BQ), jnp.float32),
            s_tmem=plgpu.TMEM((BQ, WGS * block_k), jnp.float32),
            o_tmem=plgpu.TMEM((BQ, WGS * dv), jnp.float32),
            p_tmem=tuple(tuple(plgpu.TMEM((BQ, block_k), jnp.bfloat16, packed=True) for _ in range(3)) for _ in range(WGS)),
            q_bar=plgpu.Barrier(num_arrivals=WGS, num_barriers=1),
            kv_bar=plgpu.Barrier(num_arrivals=2, num_barriers=stages),
            kv_free=plgpu.Barrier(num_arrivals=1, num_barriers=stages, orders_tensor_core=True),
            s_ready=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            s_free=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            p_ready=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            p_free=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            o_done=plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
        ),
    )
    out, norm = f(q, k, v)
    return out, norm.transpose(0, 2, 1)


@partial(jax.jit, static_argnames=("degree", "scale", "block_k", "stages"))
def dq(q, k, v, do, delta, inv_z, *, degree, scale, block_k=64, stages=3):
    """delta, inv_z: [batch, heads, time] FP32. Returns dQ [batch, time, heads, dq] in q.dtype."""
    BQ = 128
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    assert time % (BQ * WGS) == 0 and BQ % block_k == 0 and degree >= 1
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


@partial(jax.jit, static_argnames=("which", "degree", "scale", "block_q", "stages"))
def dkv(q, k, v, do, delta, inv_z, *, which, degree, scale, block_q=64, stages=3):
    """delta, inv_z: [batch, heads, time] FP32. Returns dV or dK [batch, time, heads, width] in its input dtype."""
    BK = 128
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    assert which in ("dv", "dk") and time % (BK * WGS) == 0 and BK % block_q == 0
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

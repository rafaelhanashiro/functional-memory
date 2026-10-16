"""Prototype v3: v2 with leaner elementwise work (diagonal-only masks, coefficient Horner,
row-sum-based failure detection, which is exact for positive even-degree Taylor kernels).

CTA = two compute warpgroups (128 query rows each) + one producer warpgroup whose
warp 0 issues TMA loads (K/V ring of `stages` slots) and warp 1 issues tcgen05 MMAs.
Per compute warpgroup: S is double-buffered in TMEM, O accumulates in TMEM, and the
three BF16 components of the FP32 weights live in one SMEM buffer.
Numerics match bw_taylor.forward (and fused_taylor._forward_kernel).
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


def _finite(x):
    return jnp.abs(x) < jnp.float32(jnp.inf)


def _polynomial(s, degree):
    value = jnp.full_like(s, 1.0 / math.factorial(degree))
    for order in range(degree - 1, -1, -1):
        value = value * s + 1.0 / math.factorial(order)
    return value


@partial(jax.jit, static_argnames=("degree", "block_k", "stages", "layout", "profile_dir", "ablate"))
def forward(q, k, v, *, degree, block_k=64, stages=2, layout="TCGEN05", profile_dir=None, ablate=None):
    L = getattr(plgpu.Layout, layout)
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    assert time % (BQ * WGS) == 0 and BQ % block_k == 0 and degree % 2 == 0
    scale = 1.0 / math.sqrt(dq)

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

        def s_cols(w, buf):
            return pl.ds((w * 2 + buf) * block_k, block_k)

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
                    p = jnp.where(causal, _polynomial(jnp.where(causal, s, 0.0), degree), 0.0)
                else:
                    p = _polynomial(s, degree)
                # Even-degree Taylor weights are positive wherever finite, so a nonfinite score or
                # weight always makes the row sum nonfinite: the final row-sum test is exact.
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
            invalid = ~_finite(rowsum) | (rowsum <= 0)
            denominator = jnp.where(invalid, jnp.nan, rowsum)
            o_smem.at[w][...] = o / lax.broadcast_in_dim(denominator, o.shape, [0])
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
                                for i in range(0 if ablate == "no_pv" else 3):
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
        compiler_params=plgpu.CompilerParams(profile_space=64 if profile_dir else 0, profile_dir=profile_dir or ""),
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

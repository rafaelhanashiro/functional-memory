"""Hopper (wgmma) Mosaic GPU kernels for normalized causal Taylor attention.

FlashAttention-3 structure: two compute warpgroups (64 rows each) and one TMA warpgroup per
CTA, visiting only causal tiles. Scores, polynomials, sums, and accumulators are FP32; FP32
weights and score gradients enter wgmma from registers as three truncated BF16 components.
``forward`` returns the FP32 output and row sums; ``dq`` and ``dkv`` compute the gradients
from delta = rowsum(dO * O) and 1 / Z ([batch, heads, time]). The caller (``taylor.py``)
validates shapes; ``scale`` is 1 / sqrt(d_K) for the unpadded width.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from scaffolds.transformer.attention.backends.mosaic.common import bf16_parts as _parts, finalize as _finalize, taylor as _taylor

L = plgpu.Layout.WGMMA
B = 64  # rows per compute warpgroup
WGS = 2


def _smem(shape):
    return plgpu.SMEM(shape, jnp.bfloat16, transforms=(plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128)))


@partial(jax.jit, static_argnames=("degree", "scale", "block_k", "stages"))
def forward(q, k, v, *, degree, scale, block_k=64, stages=3):
    BQ = B
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    max_steps = (BQ * WGS) // block_k  # tiles beyond a CTA's diagonal block are never needed

    def kernel(q_gmem, k_gmem, v_gmem, out_gmem, norm_gmem,
               q_smem, k_smem, v_smem, o_smem, n_smem, q_bar, k_bar, v_bar, k_free, v_free):
        wg = lax.axis_index("wg")
        b, h = lax.axis_index("b"), lax.axis_index("h")
        q_index = lax.axis_index("q")
        base = q_index * (BQ * WGS)
        cta_steps = q_index * max_steps + max_steps  # KV tiles the CTA's last row needs

        @pl.when(wg < WGS)
        def _compute():
            plgpu.set_max_registers(232, action="increase")
            q0 = base + wg * BQ
            plgpu.copy_gmem_to_smem(q_gmem.at[b, pl.ds(q0, BQ), h], q_smem.at[wg], q_bar.at[wg])
            plgpu.barrier_wait(q_bar.at[wg])
            my_steps = lax.div(q0 + BQ + block_k - 1, block_k)
            full_steps = lax.div(q0, block_k)

            def step(j, carry, masked):
                acc, rowsum = carry
                slot = lax.rem(j, stages)
                plgpu.barrier_wait(k_bar.at[slot])

                def qk(acc_ref):
                    plgpu.wgmma(acc_ref, q_smem.at[wg], plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
                    return acc_ref[...]

                s = pl.run_scoped(qk, plgpu.ACC((BQ, block_k), jnp.float32)) * scale
                plgpu.barrier_arrive(k_free.at[slot])
                if masked:
                    rows = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 0, layout=L) + q0
                    cols = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 1, layout=L) + j * block_k
                    causal = rows >= cols
                    p = jnp.where(causal, _taylor(jnp.where(causal, s, 0.0), degree), 0.0)
                else:
                    p = _taylor(s, degree)
                rowsum = rowsum + p.sum(axis=1)
                parts = _parts(p)
                plgpu.barrier_wait(v_bar.at[slot])

                def pv(acc_ref):
                    for part in parts:
                        plgpu.wgmma(acc_ref, part, v_smem.at[slot])

                acc = pl.run_state(pv)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(v_free.at[slot])
                return acc, rowsum

            acc = plgpu.layout_cast(jnp.zeros((BQ, dv), jnp.float32), L)
            rowsum = plgpu.layout_cast(jnp.zeros((BQ,), jnp.float32), L.reduce(1))
            carry = lax.fori_loop(0, full_steps, partial(step, masked=False), (acc, rowsum))
            acc, rowsum = lax.fori_loop(full_steps, my_steps, partial(step, masked=True), carry)

            def flush(j, _):  # keep the shared ring's consumed counts in step with the other warpgroup
                slot = lax.rem(j, stages)
                plgpu.barrier_arrive(k_free.at[slot])
                plgpu.barrier_arrive(v_free.at[slot])

            lax.fori_loop(my_steps, cta_steps, flush, None)
            out, denominator = _finalize(acc, rowsum, degree)
            o_smem.at[wg][...] = out
            n_smem.at[wg][...] = denominator
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem.at[wg], out_gmem.at[b, pl.ds(q0, BQ), h])
            plgpu.copy_smem_to_gmem(n_smem.at[wg], norm_gmem.at[b, h, pl.ds(q0, BQ)])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == WGS)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def load(j, _):
                slot = lax.rem(j, stages)

                @pl.when(j >= stages)
                def _():
                    plgpu.barrier_wait(k_free.at[slot])

                rows = pl.ds(j * block_k, block_k)
                plgpu.copy_gmem_to_smem(k_gmem.at[b, rows, h], k_smem.at[slot], k_bar.at[slot])

                @pl.when(j >= stages)
                def _():
                    plgpu.barrier_wait(v_free.at[slot])

                plgpu.copy_gmem_to_smem(v_gmem.at[b, rows, h], v_smem.at[slot], v_bar.at[slot])

            lax.fori_loop(0, cta_steps, load, None)

    tiling, swizzle = plgpu.TilingTransform((8, 64)), plgpu.SwizzleTransform(128)
    f = plgpu.kernel(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((batch, time, heads, dv), jnp.float32),
                   jax.ShapeDtypeStruct((batch, heads, time), jnp.float32)),
        grid=(time // (BQ * WGS), batch, heads), grid_names=("q", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=dict(
            q_smem=plgpu.SMEM((WGS, BQ, dq), jnp.bfloat16, transforms=(tiling, swizzle)),
            k_smem=plgpu.SMEM((stages, block_k, dq), jnp.bfloat16, transforms=(tiling, swizzle)),
            v_smem=plgpu.SMEM((stages, block_k, dv), jnp.bfloat16, transforms=(tiling, swizzle)),
            o_smem=plgpu.SMEM((WGS, BQ, dv), jnp.float32),
            n_smem=plgpu.SMEM((WGS, BQ), jnp.float32),
            q_bar=plgpu.Barrier(num_barriers=WGS),
            k_bar=plgpu.Barrier(num_barriers=stages),
            v_bar=plgpu.Barrier(num_barriers=stages),
            k_free=plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
            v_free=plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
        ),
    )
    out, norm = f(q, k, v)
    return out, norm.transpose(0, 2, 1)


@partial(jax.jit, static_argnames=("degree", "scale", "block_k", "stages"))
def dq(q, k, v, do, delta, inv_z, *, degree, scale, block_k=64, stages=2):
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    tiles_per_cta = B * WGS // block_k

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, delta_gmem, invz_gmem, dq_gmem,
               q_smem, do_smem, vec_smem, k_smem, v_smem, in_bar, k_bar, v_bar, k_free, v_free):
        wg = lax.axis_index("wg")
        b, h = lax.axis_index("b"), lax.axis_index("h")
        q_index = lax.axis_index("q")
        base = q_index * (B * WGS)
        cta_steps = (q_index + 1) * tiles_per_cta

        @pl.when(wg < WGS)
        def _compute():
            plgpu.set_max_registers(232, action="increase")
            q0 = base + wg * B
            rows = pl.ds(q0, B)
            plgpu.copy_gmem_to_smem(q_gmem.at[b, rows, h], q_smem.at[wg], in_bar.at[wg])
            plgpu.copy_gmem_to_smem(do_gmem.at[b, rows, h], do_smem.at[wg], in_bar.at[wg])
            plgpu.copy_gmem_to_smem(delta_gmem.at[b, h, rows], vec_smem.at[wg, 0], in_bar.at[wg])
            plgpu.copy_gmem_to_smem(invz_gmem.at[b, h, rows], vec_smem.at[wg, 1], in_bar.at[wg])
            plgpu.barrier_wait(in_bar.at[wg])
            delta_w = plgpu.load(vec_smem.at[wg, 0], (), layout=L.reduce(1))
            invz_w = plgpu.load(vec_smem.at[wg, 1], (), layout=L.reduce(1))
            my_steps = lax.div(q0 + B + block_k - 1, block_k)
            full_steps = lax.div(q0, block_k)

            def step(j, acc, masked):
                slot = lax.rem(j, stages)
                plgpu.barrier_wait(k_bar.at[slot])
                plgpu.barrier_wait(v_bar.at[slot])

                def sdp(refs):
                    s_ref, dp_ref = refs
                    plgpu.wgmma(s_ref, q_smem.at[wg], plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
                    plgpu.wgmma(dp_ref, do_smem.at[wg], plgpu.transpose_ref(v_smem.at[slot], (1, 0)))

                zeros = plgpu.layout_cast(jnp.zeros((B, block_k), jnp.float32), L)
                s, dp = pl.run_state(sdp)((plgpu.ACC.init(zeros), plgpu.ACC.init(zeros)))
                plgpu.barrier_arrive(v_free.at[slot])
                ds = ((dp - lax.broadcast_in_dim(delta_w, dp.shape, [0])) * _taylor(s * scale, degree - 1)
                      * lax.broadcast_in_dim(invz_w, dp.shape, [0]))
                if masked:
                    r = plgpu.broadcasted_iota(jnp.int32, (B, block_k), 0, layout=L) + q0
                    c = plgpu.broadcasted_iota(jnp.int32, (B, block_k), 1, layout=L) + j * block_k
                    ds = jnp.where(r >= c, ds, 0.0)
                parts = _parts(ds)

                def acc_dq(acc_ref):
                    for part in parts:
                        plgpu.wgmma(acc_ref, part, k_smem.at[slot])

                acc = pl.run_state(acc_dq)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(k_free.at[slot])
                return acc

            acc = plgpu.layout_cast(jnp.zeros((B, dqk), jnp.float32), L)
            acc = lax.fori_loop(0, full_steps, partial(step, masked=False), acc)
            acc = lax.fori_loop(full_steps, my_steps, partial(step, masked=True), acc)

            def flush(j, _):
                slot = lax.rem(j, stages)
                plgpu.barrier_arrive(k_free.at[slot])
                plgpu.barrier_arrive(v_free.at[slot])

            lax.fori_loop(my_steps, cta_steps, flush, None)
            q_smem.at[wg][...] = (acc * scale).astype(jnp.bfloat16)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem.at[wg], dq_gmem.at[b, rows, h])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == WGS)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def load(j, _):
                slot = lax.rem(j, stages)
                rows = pl.ds(j * block_k, block_k)

                @pl.when(j >= stages)
                def _():
                    plgpu.barrier_wait(k_free.at[slot])
                plgpu.copy_gmem_to_smem(k_gmem.at[b, rows, h], k_smem.at[slot], k_bar.at[slot])

                @pl.when(j >= stages)
                def _():
                    plgpu.barrier_wait(v_free.at[slot])
                plgpu.copy_gmem_to_smem(v_gmem.at[b, rows, h], v_smem.at[slot], v_bar.at[slot])

            lax.fori_loop(0, cta_steps, load, None)

    f = plgpu.kernel(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q.shape, q.dtype),
        grid=(time // (B * WGS), batch, heads), grid_names=("q", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=dict(
            q_smem=_smem((WGS, B, dqk)), do_smem=_smem((WGS, B, dv)),
            vec_smem=plgpu.SMEM((WGS, 2, B), jnp.float32),
            k_smem=_smem((stages, block_k, dqk)), v_smem=_smem((stages, block_k, dv)),
            in_bar=plgpu.Barrier(num_arrivals=4, num_barriers=WGS),
            k_bar=plgpu.Barrier(num_barriers=stages), v_bar=plgpu.Barrier(num_barriers=stages),
            k_free=plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
            v_free=plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
        ),
    )
    return f(q, k, v, do, delta, inv_z)


@partial(jax.jit, static_argnames=("degree", "scale", "block_q", "stages"))
def dkv(q, k, v, do, delta, inv_z, *, degree, scale, block_q=64, stages=2):
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    per_wg = B // block_q

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, delta_gmem, invz_gmem, dk_gmem, dv_gmem,
               k_smem, v_smem, q_smem, do_smem, vec_smem, in_bar, t_bar, t_free):
        wg = lax.axis_index("wg")
        b, h = lax.axis_index("b"), lax.axis_index("h")
        k_index = lax.axis_index("k")
        base = k_index * (B * WGS)
        start = k_index * (B * WGS // block_q)
        n_tiles = time // block_q - start

        @pl.when(wg < WGS)
        def _compute():
            plgpu.set_max_registers(232, action="increase")
            k0 = base + wg * B
            rows = pl.ds(k0, B)
            plgpu.copy_gmem_to_smem(k_gmem.at[b, rows, h], k_smem.at[wg], in_bar.at[wg])
            plgpu.copy_gmem_to_smem(v_gmem.at[b, rows, h], v_smem.at[wg], in_bar.at[wg])
            plgpu.barrier_wait(in_bar.at[wg])
            first = wg * per_wg

            def skip(j, _):  # tiles before this warpgroup's keys: consume their phases (waits track parity)
                slot = lax.rem(j, stages)
                plgpu.barrier_wait(t_bar.at[slot])
                plgpu.barrier_arrive(t_free.at[slot])

            lax.fori_loop(0, first, skip, None)

            def step(j, carry, masked):
                dk_acc, dv_acc = carry
                slot = lax.rem(j, stages)
                plgpu.barrier_wait(t_bar.at[slot])

                def st(acc_ref):
                    plgpu.wgmma(acc_ref, k_smem.at[wg], plgpu.transpose_ref(q_smem.at[slot], (1, 0)))
                    return acc_ref[...]

                s = pl.run_scoped(st, plgpu.ACC((B, block_q), jnp.float32)) * scale
                invz = lax.broadcast_in_dim(plgpu.load(vec_smem.at[slot, 1], (), layout=L.reduce(0)), s.shape, [1])
                d = lax.broadcast_in_dim(plgpu.load(vec_smem.at[slot, 0], (), layout=L.reduce(0)), s.shape, [1])
                w = _taylor(s, degree) * invz
                if masked:
                    keys = plgpu.broadcasted_iota(jnp.int32, (B, block_q), 0, layout=L) + k0
                    queries = plgpu.broadcasted_iota(jnp.int32, (B, block_q), 1, layout=L) + (start + j) * block_q
                    causal = queries >= keys
                    w = jnp.where(causal, w, 0.0)
                w_parts = _parts(w)

                def dv_dp(refs):
                    dv_ref, dp_ref = refs
                    for part in w_parts:
                        plgpu.wgmma(dv_ref, part, do_smem.at[slot])
                    plgpu.wgmma(dp_ref, v_smem.at[wg], plgpu.transpose_ref(do_smem.at[slot], (1, 0)))

                zeros = plgpu.layout_cast(jnp.zeros((B, block_q), jnp.float32), L)
                dv_acc, dp = pl.run_state(dv_dp)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
                ds = (dp - d) * _taylor(s, degree - 1) * invz
                if masked:
                    ds = jnp.where(causal, ds, 0.0)
                ds_parts = _parts(ds)

                def acc_dk(acc_ref):
                    for part in ds_parts:
                        plgpu.wgmma(acc_ref, part, q_smem.at[slot])

                dk_acc = pl.run_state(acc_dk)(plgpu.ACC.init(dk_acc))
                plgpu.barrier_arrive(t_free.at[slot])
                return dk_acc, dv_acc

            zero = plgpu.layout_cast(jnp.zeros((B, dqk), jnp.float32), L)
            carry = (zero, plgpu.layout_cast(jnp.zeros((B, dv), jnp.float32), L))
            carry = lax.fori_loop(first, first + per_wg, partial(step, masked=True), carry)
            dk_acc, dv_acc = lax.fori_loop(first + per_wg, n_tiles, partial(step, masked=False), carry)
            k_smem.at[wg][...] = (dk_acc * scale).astype(jnp.bfloat16)
            v_smem.at[wg][...] = dv_acc.astype(jnp.bfloat16)
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(k_smem.at[wg], dk_gmem.at[b, rows, h])
            plgpu.copy_smem_to_gmem(v_smem.at[wg], dv_gmem.at[b, rows, h])
            plgpu.wait_smem_to_gmem(0)

        @pl.when(wg == WGS)
        def _memory():
            plgpu.set_max_registers(40, action="decrease")

            def load(j, _):
                slot = lax.rem(j, stages)
                rows = pl.ds((start + j) * block_q, block_q)

                @pl.when(j >= stages)
                def _():
                    plgpu.barrier_wait(t_free.at[slot])
                plgpu.copy_gmem_to_smem(q_gmem.at[b, rows, h], q_smem.at[slot], t_bar.at[slot])
                plgpu.copy_gmem_to_smem(do_gmem.at[b, rows, h], do_smem.at[slot], t_bar.at[slot])
                plgpu.copy_gmem_to_smem(delta_gmem.at[b, h, rows], vec_smem.at[slot, 0], t_bar.at[slot])
                plgpu.copy_gmem_to_smem(invz_gmem.at[b, h, rows], vec_smem.at[slot, 1], t_bar.at[slot])

            lax.fori_loop(0, n_tiles, load, None)

    f = plgpu.kernel(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(time // (B * WGS), batch, heads), grid_names=("k", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=dict(
            k_smem=_smem((WGS, B, dqk)), v_smem=_smem((WGS, B, dv)),
            q_smem=_smem((stages, block_q, dqk)), do_smem=_smem((stages, block_q, dv)),
            vec_smem=plgpu.SMEM((stages, 2, block_q), jnp.float32),
            in_bar=plgpu.Barrier(num_arrivals=2, num_barriers=WGS),
            t_bar=plgpu.Barrier(num_arrivals=4, num_barriers=stages),
            t_free=plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
        ),
    )
    return f(q, k, v, do, delta, inv_z)

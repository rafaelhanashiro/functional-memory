"""Hopper (wgmma) Mosaic GPU kernels for causal polynomial attention.

FlashAttention-3 structure: two compute warpgroups (64 rows each) and one TMA warpgroup per
CTA, visiting only causal tiles. Scores, polynomials, sums, and accumulators are FP32; FP32
weights and score gradients enter wgmma from registers as BF16 operands at ``precision``
(``common.weight_parts``).
The elementwise mathematics of each kernel family lives in ``common.py`` (``family`` and
``degree`` select it); per-query FP32 vectors arrive as ``vectors`` [n, batch, heads, time].
``forward`` returns the FP32 output and a per-row statistic [batch, heads, time]; ``dq``
returns dQ and, for families with per-query sums, those sums [n, batch, heads, time];
``dkv`` returns dK and dV. Callers validate shapes; ``scale`` is 1 / sqrt(d_K) for the
unpadded width.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from scaffolds.transformer.attention.backends.mosaic import common

L = plgpu.Layout.WGMMA
B = 64  # rows per compute warpgroup
WGS = 2


def _smem(shape):
    """BF16 tile with the widest swizzle its rows allow: 128 bytes at width >= 64, 64 at 32, 32 at 16.

    wgmma contracts in groups of one swizzle width, so narrow Q/K need narrow swizzles."""
    swizzle = plgpu.find_swizzle(shape[-1] * 16)
    return plgpu.SMEM(shape, jnp.bfloat16, transforms=(plgpu.TilingTransform((8, swizzle // 2)), plgpu.SwizzleTransform(swizzle)))


@partial(jax.jit, static_argnames=("family", "degree", "scale", "precision", "block_k", "stages"))
def forward(q, k, v, vectors, *, family, degree, scale, precision, block_k=64, stages=3):
    BQ = B
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    max_steps = (BQ * WGS) // block_k  # tiles beyond a CTA's diagonal block are never needed
    assert BQ // block_k * (WGS - 1) <= stages, "unfreed trailing tiles must never be reloaded"
    n_vec = common.vector_counts(family, degree)[0]

    def kernel(q_gmem, k_gmem, v_gmem, *refs):
        if n_vec:
            vec_gmem, out_gmem, stat_gmem, *scratch = refs
        else:
            out_gmem, stat_gmem, *scratch = refs
        q_smem, k_smem, v_smem, vec_smem, o_smem, n_smem, q_bar, k_bar, v_bar, k_free, v_free = scratch
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
            for i in range(n_vec):
                plgpu.copy_gmem_to_smem(vec_gmem.at[i, b, h, pl.ds(q0, BQ)], vec_smem.at[wg, i], q_bar.at[wg])
            plgpu.barrier_wait(q_bar.at[wg])
            rows = [plgpu.load(vec_smem.at[wg, i], (), layout=L.reduce(1)) for i in range(n_vec)]
            my_steps = lax.div(q0 + BQ + block_k - 1, block_k)
            full_steps = lax.div(q0, block_k)

            def step(j, carry, masked):
                acc, statistic = carry
                slot = lax.rem(j, stages)
                plgpu.barrier_wait(k_bar.at[slot])

                def qk(acc_ref):
                    plgpu.wgmma(acc_ref, q_smem.at[wg], plgpu.transpose_ref(k_smem.at[slot], (1, 0)))
                    return acc_ref[...]

                s = pl.run_scoped(qk, plgpu.ACC((BQ, block_k), jnp.float32)) * scale
                plgpu.barrier_arrive(k_free.at[slot])
                vecs = [lax.broadcast_in_dim(r, s.shape, [0]) for r in rows]
                if masked:
                    r = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 0, layout=L) + q0
                    c = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 1, layout=L) + j * block_k
                    causal = r >= c
                    p = jnp.where(causal, common.forward_weights(family, degree, jnp.where(causal, s, 0.0), vecs), 0.0)
                else:
                    p = common.forward_weights(family, degree, s, vecs)
                statistic = statistic + common.forward_statistic(family, p).sum(axis=1)
                parts = common.weight_parts(p, precision)
                plgpu.barrier_wait(v_bar.at[slot])

                def pv(acc_ref):
                    for part in parts:
                        plgpu.wgmma(acc_ref, part, v_smem.at[slot])

                acc = pl.run_state(pv)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(v_free.at[slot])
                return acc, statistic

            acc = plgpu.layout_cast(jnp.zeros((BQ, dv), jnp.float32), L)
            statistic = plgpu.layout_cast(jnp.zeros((BQ,), jnp.float32), L.reduce(1))
            carry = lax.fori_loop(0, full_steps, partial(step, masked=False), (acc, statistic))
            acc, statistic = lax.fori_loop(full_steps, my_steps, partial(step, masked=True), carry)
            # Tiles past this warpgroup's diagonal are neither consumed nor freed here. Freeing a
            # tile before it lands would let a ring slot complete a phase without the other
            # warpgroup, so the loader could overwrite a tile that warpgroup still reads, or
            # advance a barrier two phases past its parity wait (a deadlock). No load waits on
            # these tiles' free phases: they are the CTA's last tiles and number at most stages.
            out, statistic = common.forward_finish(family, degree, acc, statistic)
            o_smem.at[wg][...] = out
            n_smem.at[wg][...] = statistic
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem.at[wg], out_gmem.at[b, pl.ds(q0, BQ), h])
            plgpu.copy_smem_to_gmem(n_smem.at[wg], stat_gmem.at[b, h, pl.ds(q0, BQ)])
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

    f = plgpu.kernel(
        kernel,
        out_shape=(jax.ShapeDtypeStruct((batch, time, heads, dv), jnp.float32),
                   jax.ShapeDtypeStruct((batch, heads, time), jnp.float32)),
        grid=(time // (BQ * WGS), batch, heads), grid_names=("q", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=[
            _smem((WGS, BQ, dq)),
            _smem((stages, block_k, dq)),
            _smem((stages, block_k, dv)),
            plgpu.SMEM((WGS, max(n_vec, 1), BQ), jnp.float32),
            plgpu.SMEM((WGS, BQ, dv), jnp.float32),
            plgpu.SMEM((WGS, BQ), jnp.float32),
            plgpu.Barrier(num_arrivals=1 + n_vec, num_barriers=WGS),
            plgpu.Barrier(num_barriers=stages),
            plgpu.Barrier(num_barriers=stages),
            plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
            plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
        ],
    )
    return f(q, k, v, vectors) if n_vec else f(q, k, v)


@partial(jax.jit, static_argnames=("family", "degree", "scale", "precision", "block_k", "stages"))
def dq(q, k, v, do, vectors, *, family, degree, scale, precision, block_k=64, stages=2):
    """dQ [batch, time, heads, d_K] in q.dtype, and per-query sums [n, batch, heads, time] FP32 or None."""
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    tiles_per_cta = B * WGS // block_k
    assert B // block_k * (WGS - 1) <= stages, "unfreed trailing tiles must never be reloaded"
    _, n_vec, n_sums = common.vector_counts(family, degree)

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, vec_gmem, dq_gmem, *refs):
        if n_sums:
            sums_gmem, *scratch = refs
        else:
            scratch = refs
        q_smem, do_smem, vec_smem, sums_smem, k_smem, v_smem, in_bar, k_bar, v_bar, k_free, v_free = scratch
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
            for i in range(n_vec):
                plgpu.copy_gmem_to_smem(vec_gmem.at[i, b, h, rows], vec_smem.at[wg, i], in_bar.at[wg])
            plgpu.barrier_wait(in_bar.at[wg])
            row_vectors = [plgpu.load(vec_smem.at[wg, i], (), layout=L.reduce(1)) for i in range(n_vec)]
            my_steps = lax.div(q0 + B + block_k - 1, block_k)
            full_steps = lax.div(q0, block_k)

            def step(j, carry, masked):
                acc, sums = carry
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
                s = s * scale
                vecs = [lax.broadcast_in_dim(r, s.shape, [0]) for r in row_vectors]
                if masked:
                    r = plgpu.broadcasted_iota(jnp.int32, (B, block_k), 0, layout=L) + q0
                    c = plgpu.broadcasted_iota(jnp.int32, (B, block_k), 1, layout=L) + j * block_k
                    causal = r >= c
                    s, dp = jnp.where(causal, s, 0.0), jnp.where(causal, dp, 0.0)
                    ds = jnp.where(causal, common.score_gradient(family, degree, s, dp, vecs), 0.0)
                else:
                    ds = common.score_gradient(family, degree, s, dp, vecs)
                sums = tuple(a + x for a, x in zip(sums, common.row_sums(family, degree, s, dp)))
                parts = common.weight_parts(ds, precision)

                def acc_dq(acc_ref):
                    for part in parts:
                        plgpu.wgmma(acc_ref, part, k_smem.at[slot])

                acc = pl.run_state(acc_dq)(plgpu.ACC.init(acc))
                plgpu.barrier_arrive(k_free.at[slot])
                return acc, sums

            acc = plgpu.layout_cast(jnp.zeros((B, dqk), jnp.float32), L)
            sums = tuple(plgpu.layout_cast(jnp.zeros((B,), jnp.float32), L.reduce(1)) for _ in range(n_sums))
            carry = lax.fori_loop(0, full_steps, partial(step, masked=False), (acc, sums))
            acc, sums = lax.fori_loop(full_steps, my_steps, partial(step, masked=True), carry)
            # As in forward, tiles past this warpgroup's diagonal are neither consumed nor freed.
            q_smem.at[wg][...] = (acc * scale).astype(jnp.bfloat16)
            for i, total in enumerate(sums):
                sums_smem.at[wg, i][...] = total
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem.at[wg], dq_gmem.at[b, rows, h])
            for i in range(n_sums):
                plgpu.copy_smem_to_gmem(sums_smem.at[wg, i], sums_gmem.at[i, b, h, rows])
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

    out_shapes = [jax.ShapeDtypeStruct(q.shape, q.dtype)]
    if n_sums:
        out_shapes.append(jax.ShapeDtypeStruct((n_sums, batch, heads, time), jnp.float32))
    f = plgpu.kernel(
        kernel,
        out_shape=tuple(out_shapes),
        grid=(time // (B * WGS), batch, heads), grid_names=("q", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=[
            _smem((WGS, B, dqk)), _smem((WGS, B, dv)),
            plgpu.SMEM((WGS, n_vec, B), jnp.float32),
            plgpu.SMEM((WGS, max(n_sums, 1), B), jnp.float32),
            _smem((stages, block_k, dqk)), _smem((stages, block_k, dv)),
            plgpu.Barrier(num_arrivals=2 + n_vec, num_barriers=WGS),
            plgpu.Barrier(num_barriers=stages), plgpu.Barrier(num_barriers=stages),
            plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
            plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
        ],
    )
    outputs = f(q, k, v, do, vectors)
    return (outputs[0], outputs[1]) if n_sums else (outputs[0], None)


@partial(jax.jit, static_argnames=("family", "degree", "scale", "precision", "block_q", "stages"))
def dkv(q, k, v, do, vectors, *, family, degree, scale, precision, block_q=64, stages=2):
    """(dK, dV) in the input dtypes. Lanes are keys; per-query vectors stream with each query tile."""
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    per_wg = B // block_q
    n_vec = common.vector_counts(family, degree)[1]

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, vec_gmem, dk_gmem, dv_gmem,
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
                vecs = [lax.broadcast_in_dim(plgpu.load(vec_smem.at[slot, i], (), layout=L.reduce(0)), s.shape, [1])
                        for i in range(n_vec)]
                if masked:
                    keys = plgpu.broadcasted_iota(jnp.int32, (B, block_q), 0, layout=L) + k0
                    queries = plgpu.broadcasted_iota(jnp.int32, (B, block_q), 1, layout=L) + (start + j) * block_q
                    causal = queries >= keys
                    s = jnp.where(causal, s, 0.0)
                    w = jnp.where(causal, common.value_weights(family, degree, s, vecs), 0.0)
                else:
                    w = common.value_weights(family, degree, s, vecs)
                w_parts = common.weight_parts(w, precision)

                def dv_dp(refs):
                    dv_ref, dp_ref = refs
                    for part in w_parts:
                        plgpu.wgmma(dv_ref, part, do_smem.at[slot])
                    plgpu.wgmma(dp_ref, v_smem.at[wg], plgpu.transpose_ref(do_smem.at[slot], (1, 0)))

                zeros = plgpu.layout_cast(jnp.zeros((B, block_q), jnp.float32), L)
                dv_acc, dp = pl.run_state(dv_dp)((plgpu.ACC.init(dv_acc), plgpu.ACC.init(zeros)))
                if masked:
                    dp = jnp.where(causal, dp, 0.0)
                    ds = jnp.where(causal, common.score_gradient(family, degree, s, dp, vecs), 0.0)
                else:
                    ds = common.score_gradient(family, degree, s, dp, vecs)
                ds_parts = common.weight_parts(ds, precision)

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
                for i in range(n_vec):
                    plgpu.copy_gmem_to_smem(vec_gmem.at[i, b, h, rows], vec_smem.at[slot, i], t_bar.at[slot])

            lax.fori_loop(0, n_tiles, load, None)

    f = plgpu.kernel(
        kernel,
        out_shape=(jax.ShapeDtypeStruct(k.shape, k.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        grid=(time // (B * WGS), batch, heads), grid_names=("k", "b", "h"),
        num_threads=WGS + 1, thread_name="wg",
        scratch_shapes=[
            _smem((WGS, B, dqk)), _smem((WGS, B, dv)),
            _smem((stages, block_q, dqk)), _smem((stages, block_q, dv)),
            plgpu.SMEM((stages, n_vec, block_q), jnp.float32),
            plgpu.Barrier(num_arrivals=2, num_barriers=WGS),
            plgpu.Barrier(num_arrivals=2 + n_vec, num_barriers=stages),
            plgpu.Barrier(num_arrivals=WGS, num_barriers=stages),
        ],
    )
    return f(q, k, v, do, vectors)

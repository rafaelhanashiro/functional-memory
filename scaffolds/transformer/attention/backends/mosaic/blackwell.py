"""Blackwell (tcgen05) Mosaic GPU kernels for causal polynomial attention.

Each CTA has two compute warpgroups and a producer warpgroup (warp 0 issues TMA, warp 1
issues tcgen05 MMAs). Scores, polynomials, sums, and accumulators are FP32; FP32 weights
and score gradients enter the MMA as BF16 operands at ``precision`` (``common.weight_parts``:
one, or three components), stored in tensor memory. The elementwise mathematics of each kernel family lives in ``common.py``
(``family`` and ``degree`` select it); per-query FP32 vectors arrive as ``vectors``
[n, batch, heads, time]. ``forward`` returns the FP32 output and a per-row statistic
[batch, heads, time]; ``dq`` returns dQ and, for families with per-query sums, those sums
[n, batch, heads, time]; ``dkv`` returns dV or dK. Callers validate shapes; ``scale`` is
1 / sqrt(d_K) for the unpadded width. Decay families also take per-key FP32 ``key_vectors``
[n, batch, heads, time] (held whole in shared memory, or a CTA's keys in ``dkv``), see the
distance t-i in every tile, and the dK pass also returns their per-key sums. Families with
per-head scalars take ``head_constants`` [heads, width] FP32.
"""
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu

from scaffolds.transformer.attention.backends.mosaic import common

L = plgpu.Layout.TCGEN05
WGS = 2
TMA_ELEMENTS = 256  # TMA moves at most 256 elements per dimension
SMEM_BYTES = common.BLACKWELL_SMEM_BYTES


def _pair(family, shape, offset, keys, causal, *, transposed=False, heads=()):
    """``common.Pair`` of a decay family's tile, else None: the FP32 query-minus-key distance of
    each entry, from the tile's loop-invariant local pattern plus ``offset`` (the distance of its
    first entry). Transposed tiles hold keys in rows; ``keys`` are broadcast per-key vectors."""
    if family not in common.DECAY_FAMILIES:
        return None
    rows, cols = (plgpu.broadcasted_iota(jnp.float32, shape, d, layout=L) for d in (0, 1))
    local = cols - rows if transposed else rows - cols
    return common.Pair(local + offset.astype(jnp.float32), keys, causal, heads)


def _inputs(*arrays):
    """The optional kernel inputs that are present, in order."""
    return [x for x in arrays if x is not None]


def _params(family):
    return plgpu.CompilerParams(approx_math=True) if common.approx_math(family) else None


def _key_columns(key_smem, n_key, cols, shape, *, lazy=False):
    """Per-key vectors of one tile, broadcast down its rows (keys are columns); ``lazy`` builds
    each on use (``common.Lazy``)."""
    build = lambda i: plgpu.layout_cast(lax.broadcast_in_dim(
        plgpu.load(key_smem.at[i], cols, layout=L.reduce(0), optimized=False), shape, [1]), L)
    return common.Lazy(n_key, build) if lazy else [build(i) for i in range(n_key)]


def _rows_lazy(smem, count, shape):
    """``common.Lazy`` row-broadcast vectors read from ``smem`` [count, rows] at each use."""
    return common.Lazy(count, lambda i: lax.broadcast_in_dim(
        plgpu.layout_cast(smem.at[i][...], L.reduce(1)), shape, [0]))


def _heads(head_smem, scalars, family):
    """Per-head scalars: read at each use by (possibly traced) order index for per-order families."""
    if family in common.PER_ORDER_FAMILIES and scalars:
        return common.Lazy(len(scalars), lambda i: head_smem[i])
    return scalars


def _transforms(cols, dtype):
    swizzle = plgpu.find_swizzle(cols * jnp.dtype(dtype).itemsize * 8)
    return (plgpu.TilingTransform((8, swizzle // jnp.dtype(dtype).itemsize)), plgpu.SwizzleTransform(swizzle))


@partial(jax.jit, static_argnames=("family", "degree", "scale", "precision", "block_k", "stages"))
def forward(q, k, v, vectors, key_vectors=None, head_constants=None, *, family, degree, scale, precision,
            block_k=64, stages=3):
    BQ = 128
    n_parts = common.part_count(precision)
    batch, time, heads, dq = q.shape
    dv = v.shape[-1]
    n_vec = common.vector_counts(family, degree)[0]
    n_key = common.key_counts(family, degree)[0]
    n_head = common.head_count(family, degree)
    chunks = time // TMA_ELEMENTS

    def kernel(q_gmem, k_gmem, v_gmem, *refs):
        refs = list(refs)
        vec_gmem = refs.pop(0) if n_vec else None
        key_gmem = refs.pop(0) if n_key else None
        head_gmem = refs.pop(0) if n_head else None
        out_gmem, stat_gmem, q_smem, k_smem, v_smem, vec_smem, *scratch = refs
        key_smem = scratch.pop(0) if n_key else None
        head_smem = scratch.pop(0) if n_head else None
        (o_smem, n_smem, s_tmem, o_tmem, p_tmem,
         q_bar, kv_bar, kv_free, s_ready, s_free, p_ready, p_free, o_done) = scratch
        wg = lax.axis_index("wg")
        b, h = lax.axis_index("b"), lax.axis_index("h")
        q_index = lax.axis_index("q")
        base = q_index * (BQ * WGS)

        def tiles(w):  # causal KV tiles needed by compute warpgroup w (exact: block_k divides BQ)
            return q_index * (BQ * WGS // block_k) + (w + 1) * (BQ // block_k)

        n_tiles = tiles(WGS - 1)

        def compute(w):
            q0 = base + w * BQ
            rows = []
            if n_vec:
                plgpu.barrier_wait(q_bar.at[0])
                rows = [plgpu.layout_cast(vec_smem.at[w, i][...], L.reduce(1)) for i in range(n_vec)]
            head_scalars = [head_smem[i] for i in range(n_head)]

            def body(j, statistic, masked):
                plgpu.barrier_wait(s_ready.at[w])
                s = plgpu.async_load_tmem(s_tmem.at[:, pl.ds(w * block_k, block_k)], layout=L) * scale
                plgpu.wait_load_tmem()
                plgpu.barrier_arrive(s_free.at[w])
                vecs = (_rows_lazy(vec_smem.at[w], n_vec, s.shape) if family in common.PER_ORDER_FAMILIES
                        else [lax.broadcast_in_dim(r, s.shape, [0]) for r in rows])
                causal = None
                if masked:  # tiles crossing the causal diagonal
                    r = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 0, layout=L) + q0
                    c = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 1, layout=L) + j * block_k
                    causal = r >= c
                keys = _key_columns(key_smem, n_key, pl.ds(j * block_k, block_k), s.shape,
                                    lazy=family in common.PER_ORDER_FAMILIES)
                pair = _pair(family, s.shape, q0 - j * block_k, keys, causal, heads=_heads(head_smem, head_scalars, family))
                if masked:
                    p = jnp.where(causal, common.forward_weights(family, degree, jnp.where(causal, s, 0.0), vecs, pair), 0.0)
                else:
                    p = common.forward_weights(family, degree, s, vecs, pair)
                # A nonfinite score or weight makes the row statistic nonfinite, so the final
                # per-row test detects it exactly.
                statistic = statistic + common.forward_statistic(family, p).sum(axis=1)
                parts = common.weight_parts(p, precision)

                @pl.when(j > 0)
                def _():
                    plgpu.barrier_wait(p_free.at[w])  # PV of tile j-1 has read the weights

                for i, part in enumerate(parts):
                    plgpu.async_store_tmem(p_tmem[w][i], part)
                plgpu.commit_tmem()
                plgpu.barrier_arrive(p_ready.at[w])
                return statistic

            zeros = plgpu.layout_cast(jnp.zeros((BQ,), jnp.float32), L.reduce(1))
            full = q_index * (BQ * WGS // block_k) + w * (BQ // block_k)  # tiles strictly below the diagonal
            statistic = lax.fori_loop(0, full, lambda j, r: body(j, r, False), zeros)
            statistic = lax.fori_loop(full, tiles(w), lambda j, r: body(j, r, True), statistic)
            plgpu.barrier_wait(o_done.at[w])
            o = plgpu.async_load_tmem(o_tmem.at[:, pl.ds(w * dv, dv)], layout=L)
            plgpu.wait_load_tmem()
            out, statistic = common.forward_finish(family, degree, o, statistic)
            o_smem.at[w][...] = out
            n_smem.at[w][...] = statistic
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(o_smem.at[w], out_gmem.at[b, pl.ds(q0, BQ), h])
            plgpu.copy_smem_to_gmem(n_smem.at[w], stat_gmem.at[b, h, pl.ds(q0, BQ)])
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
                        plgpu.copy_gmem_to_smem(q_gmem.at[b, rows, h], q_smem.at[w], q_bar.at[0])
                        for i in range(n_vec):
                            plgpu.copy_gmem_to_smem(vec_gmem.at[i, b, h, rows], vec_smem.at[w, i], q_bar.at[0])
                    for i in range(n_key):
                        for c in range(chunks):
                            chunk = pl.ds(c * TMA_ELEMENTS, TMA_ELEMENTS)
                            plgpu.copy_gmem_to_smem(key_gmem.at[i, b, h, chunk], key_smem.at[i, chunk], q_bar.at[0])
                    if n_head:
                        plgpu.copy_gmem_to_smem(head_gmem.at[h], head_smem, q_bar.at[0])

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
                                for i in range(n_parts):
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
        num_threads=WGS + 1, thread_name="wg", compiler_params=_params(family),
        scratch_shapes=[
            plgpu.SMEM((WGS, BQ, dq), jnp.bfloat16, transforms=_transforms(dq, jnp.bfloat16)),
            plgpu.SMEM((stages, block_k, dq), jnp.bfloat16, transforms=_transforms(dq, jnp.bfloat16)),
            plgpu.SMEM((stages, block_k, dv), jnp.bfloat16, transforms=_transforms(dv, jnp.bfloat16)),
            plgpu.SMEM((WGS, max(n_vec, 1), BQ), jnp.float32),
            *([plgpu.SMEM((n_key, time), jnp.float32)] if n_key else []),
            *([plgpu.SMEM((common.head_width(n_head),), jnp.float32)] if n_head else []),
            plgpu.SMEM((WGS, BQ, dv), jnp.float32),
            plgpu.SMEM((WGS, BQ), jnp.float32),
            plgpu.TMEM((BQ, WGS * block_k), jnp.float32),
            plgpu.TMEM((BQ, WGS * dv), jnp.float32),
            tuple(tuple(plgpu.TMEM((BQ, block_k), jnp.bfloat16, packed=True) for _ in range(n_parts))
                  for _ in range(WGS)),
            plgpu.Barrier(num_arrivals=WGS * (1 + n_vec) + n_key * chunks + bool(n_head), num_barriers=1),
            plgpu.Barrier(num_arrivals=2, num_barriers=stages),
            plgpu.Barrier(num_arrivals=1, num_barriers=stages, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
        ],
    )
    return f(q, k, v, *_inputs(vectors if n_vec else None, key_vectors if n_key else None,
                               head_constants if n_head else None))


@partial(jax.jit, static_argnames=("family", "degree", "scale", "precision", "block_k", "stages"))
def dq(q, k, v, do, vectors, key_vectors=None, head_constants=None, *, family, degree, scale, precision,
       block_k=64, stages=3):
    """dQ [batch, time, heads, d_K] in q.dtype, and per-query sums [n, batch, heads, time] FP32 or None."""
    BQ = 128
    n_parts = common.part_count(precision)
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    _, n_vec, n_sums = common.vector_counts(family, degree)
    n_key = common.key_counts(family, degree)[0]
    n_head = common.head_count(family, degree)
    chunks = time // TMA_ELEMENTS

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, vec_gmem, *refs):
        refs = list(refs)
        key_gmem = refs.pop(0) if n_key else None
        head_gmem = refs.pop(0) if n_head else None
        dq_gmem, *refs = refs
        if n_sums:
            sums_gmem, *scratch = refs
        else:
            scratch = refs
        q_smem, do_smem, k_smem, v_smem, vec_smem, sums_smem, *scratch = scratch
        key_smem = scratch.pop(0) if n_key else None
        head_smem = scratch.pop(0) if n_head else None
        regions, acc_tmem, in_bar, kv_bar, kv_free, s_ready, ds_ready, done = scratch
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
            rows = [plgpu.layout_cast(vec_smem.at[w, i][...], L.reduce(1)) for i in range(n_vec)]
            head_scalars = [head_smem[i] for i in range(n_head)]

            def body(j, sums, masked):
                (s_tmem, dp_tmem), ds_tmem = regions[w]
                plgpu.barrier_wait(s_ready.at[w])
                s = plgpu.async_load_tmem(s_tmem, layout=L) * scale
                dp = plgpu.async_load_tmem(dp_tmem, layout=L)
                plgpu.wait_load_tmem()
                vecs = (_rows_lazy(vec_smem.at[w], n_vec, s.shape) if family in common.PER_ORDER_FAMILIES
                        else [lax.broadcast_in_dim(r, s.shape, [0]) for r in rows])
                causal = None
                if masked:
                    r = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 0, layout=L) + q0
                    c = plgpu.broadcasted_iota(jnp.int32, (BQ, block_k), 1, layout=L) + j * block_k
                    causal = r >= c
                keys = _key_columns(key_smem, n_key, pl.ds(j * block_k, block_k), s.shape,
                                    lazy=family in common.PER_ORDER_FAMILIES)
                pair = _pair(family, s.shape, q0 - j * block_k, keys, causal, heads=_heads(head_smem, head_scalars, family))
                if masked:
                    s, dp = jnp.where(causal, s, 0.0), jnp.where(causal, dp, 0.0)
                ds, row = common.dq_terms(family, degree, s, dp, vecs, pair)
                if masked:
                    ds = jnp.where(causal, ds, 0.0)
                sums = tuple(a + x for a, x in zip(sums, row))
                for i, part in enumerate(common.weight_parts(ds, precision)):  # overwrite this tile's S/dP region
                    plgpu.async_store_tmem(ds_tmem[i], part)
                plgpu.commit_tmem()
                plgpu.barrier_arrive(ds_ready.at[w])
                return sums

            zeros = tuple(plgpu.layout_cast(jnp.zeros((BQ,), jnp.float32), L.reduce(1)) for _ in range(n_sums))
            full = q_index * (BQ * WGS // block_k) + w * (BQ // block_k)
            sums = lax.fori_loop(0, full, lambda j, c: body(j, c, False), zeros)
            sums = lax.fori_loop(full, tiles(w), lambda j, c: body(j, c, True), sums)
            plgpu.barrier_wait(done.at[w])
            acc = plgpu.async_load_tmem(acc_tmem.at[:, pl.ds(w * dqk, dqk)], layout=L)
            plgpu.wait_load_tmem()
            # Every MMA reading this warpgroup's Q has completed, so its Q buffer stages dQ.
            q_smem.at[w][...] = (acc * scale).astype(q_smem.dtype)
            for i, total in enumerate(sums):
                sums_smem.at[w, i][...] = total
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(q_smem.at[w], dq_gmem.at[b, pl.ds(q0, BQ), h])
            for i in range(n_sums):
                plgpu.copy_smem_to_gmem(sums_smem.at[w, i], sums_gmem.at[i, b, h, pl.ds(q0, BQ)])
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
                        for i in range(n_vec):
                            plgpu.copy_gmem_to_smem(vec_gmem.at[i, b, h, rows], vec_smem.at[w, i], in_bar.at[0])
                    for i in range(n_key):
                        for c in range(chunks):
                            chunk = pl.ds(c * TMA_ELEMENTS, TMA_ELEMENTS)
                            plgpu.copy_gmem_to_smem(key_gmem.at[i, b, h, chunk], key_smem.at[i, chunk], in_bar.at[0])
                    if n_head:
                        plgpu.copy_gmem_to_smem(head_gmem.at[h], head_smem, in_bar.at[0])

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
                                for i in range(n_parts):
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
    out_shapes = [jax.ShapeDtypeStruct(q.shape, q.dtype)]
    if n_sums:
        out_shapes.append(jax.ShapeDtypeStruct((n_sums, batch, heads, time), jnp.float32))
    f = plgpu.kernel(
        kernel,
        out_shape=tuple(out_shapes),
        grid=(time // (BQ * WGS), batch, heads), grid_names=("q", "b", "h"),
        num_threads=WGS + 1, thread_name="wg", compiler_params=_params(family),
        scratch_shapes=[
            plgpu.SMEM((WGS, BQ, dqk), jnp.bfloat16, transforms=t(dqk)),
            plgpu.SMEM((WGS, BQ, dv), jnp.bfloat16, transforms=t(dv)),
            plgpu.SMEM((stages, block_k, dqk), jnp.bfloat16, transforms=t(dqk)),
            plgpu.SMEM((stages, block_k, dv), jnp.bfloat16, transforms=t(dv)),
            plgpu.SMEM((WGS, n_vec, BQ), jnp.float32),
            plgpu.SMEM((WGS, max(n_sums, 1), BQ), jnp.float32),
            *([plgpu.SMEM((n_key, time), jnp.float32)] if n_key else []),
            *([plgpu.SMEM((common.head_width(n_head),), jnp.float32)] if n_head else []),
            tuple(plgpu.RefUnion(
                (plgpu.TMEM((BQ, block_k), jnp.float32), plgpu.TMEM((BQ, block_k), jnp.float32)),
                tuple(plgpu.TMEM((BQ, block_k), jnp.bfloat16, packed=True) for _ in range(n_parts))) for _ in range(WGS)),
            plgpu.TMEM((BQ, WGS * dqk), jnp.float32),
            plgpu.Barrier(num_arrivals=WGS * (2 + n_vec) + n_key * chunks + bool(n_head), num_barriers=1),
            plgpu.Barrier(num_arrivals=2, num_barriers=stages),
            plgpu.Barrier(num_arrivals=1, num_barriers=stages, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
        ],
    )
    outputs = f(q, k, v, do, vectors, *_inputs(key_vectors if n_key else None, head_constants if n_head else None))
    return (outputs[0], outputs[1]) if n_sums else (outputs[0], None)


@partial(jax.jit, static_argnames=("which", "family", "degree", "scale", "precision", "block_q", "stages"))
def dkv(q, k, v, do, vectors, key_vectors=None, head_constants=None, *, which, family, degree, scale, precision,
        block_q=64,
        stages=None):
    """dV or dK [batch, time, heads, width] in its input dtype; for families with per-key sums,
    the dK pass returns (dK, sums [n, batch, heads, time]).

    Transposed tiles: lanes are keys, so per-query vectors broadcast along columns. Each
    warpgroup stages its output in its own K (dK) or V (dV) buffer once its MMAs are done.
    """
    BK = 128
    batch, time, heads, dqk = q.shape
    dv = v.shape[-1]
    assert which in ("dv", "dk") and time % TMA_ELEMENTS == 0
    width = dv if which == "dv" else dqk
    per_wg = BK // block_q  # query tiles per warpgroup key range
    n_vec = common.vector_counts(family, degree)[1]
    n_key, n_key_sums = common.key_counts(family, degree)
    n_head = common.head_count(family, degree)
    if which == "dv":
        n_key_sums = 0
    n_parts = common.part_count(precision)
    chunks = time // TMA_ELEMENTS
    if stages is None:  # every query's vectors share the block with K/V and the Q/dO ring
        stages = common.blackwell_dkv_stages(dqk, dv, n_vec, time, block_q=block_q, block_k=BK, warpgroups=WGS,
                                             extra_bytes=common.dkv_extra_bytes(family, which, degree))
        if stages is None:
            raise ValueError(f"Blackwell dK/dV cannot hold {n_vec} per-query vectors for {time} queries "
                             f"beside width {dqk} keys in shared memory")

    def kernel(q_gmem, k_gmem, v_gmem, do_gmem, vec_gmem, *refs):
        refs = list(refs)
        key_gmem = refs.pop(0) if n_key else None
        head_gmem = refs.pop(0) if n_head else None
        out_gmem, *refs = refs
        if n_key_sums:
            key_sums_gmem, *refs = refs
        k_smem, v_smem, q_smem, do_smem, vec_smem, *refs = refs
        key_smem = refs.pop(0) if n_key else None
        key_sums_smem = refs.pop(0) if n_key_sums else None
        head_smem = refs.pop(0) if n_head else None
        regions, acc_tmem, in_bar, tile_bar, tile_free, s_ready, ds_ready, done = refs
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
            key_rows = [plgpu.layout_cast(key_smem.at[w, i][...], L.reduce(1)) for i in range(n_key)]
            head_scalars = [head_smem[i] for i in range(n_head)]

            def body(j, key_sums, masked):
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
                vecs = _key_columns(vec_smem, n_vec, cols, s.shape, lazy=family in common.PER_ORDER_FAMILIES)
                causal = None
                if masked:
                    keys = plgpu.broadcasted_iota(jnp.int32, (BK, block_q), 0, layout=L) + k0
                    queries = plgpu.broadcasted_iota(jnp.int32, (BK, block_q), 1, layout=L) + (start + j) * block_q
                    causal = queries >= keys
                pair = _pair(family, s.shape, (start + j) * block_q - k0,
                             (_rows_lazy(key_smem.at[w], n_key, s.shape) if n_key else ())
                             if family in common.PER_ORDER_FAMILIES
                             else [lax.broadcast_in_dim(x, s.shape, [0]) for x in key_rows], causal, transposed=True,
                             heads=_heads(head_smem, head_scalars, family))
                if masked:
                    s = jnp.where(causal, s, 0.0)
                    if which == "dk":
                        dp = jnp.where(causal, dp, 0.0)
                if which == "dv":
                    x = common.value_weights(family, degree, s, vecs, pair)
                else:
                    x, sums = common.dk_terms(family, degree, s, dp, vecs, pair)
                    key_sums = tuple(a + y for a, y in zip(key_sums, sums))
                if masked:
                    x = jnp.where(causal, x, 0.0)
                for i, part in enumerate(common.weight_parts(x, precision)):
                    plgpu.async_store_tmem(parts_tmem[i], part)
                plgpu.commit_tmem()
                plgpu.barrier_arrive(ds_ready.at[w])
                return key_sums

            zeros = tuple(plgpu.layout_cast(jnp.zeros((BK,), jnp.float32), L.reduce(1)) for _ in range(n_key_sums))
            key_sums = lax.fori_loop(first(w), first(w) + per_wg, lambda j, c: body(j, c, True), zeros)
            key_sums = lax.fori_loop(first(w) + per_wg, n_tiles, lambda j, c: body(j, c, False), key_sums)
            plgpu.barrier_wait(done.at[w])  # every MMA reading this warpgroup's K and V has completed
            acc = plgpu.async_load_tmem(acc_tmem.at[:, pl.ds(w * width, width)], layout=L)
            plgpu.wait_load_tmem()
            staging = k_smem.at[w] if which == "dk" else v_smem.at[w]
            staging[...] = (acc * scale if which == "dk" else acc).astype(jnp.bfloat16)
            for i, total in enumerate(key_sums):
                key_sums_smem.at[w, i][...] = total
            plgpu.commit_smem()
            plgpu.copy_smem_to_gmem(staging, out_gmem.at[b, pl.ds(k0, BK), h])
            for i in range(n_key_sums):
                plgpu.copy_smem_to_gmem(key_sums_smem.at[w, i], key_sums_gmem.at[i, b, h, pl.ds(k0, BK)])
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
                        for i in range(n_key):
                            plgpu.copy_gmem_to_smem(key_gmem.at[i, b, h, rows], key_smem.at[w, i], in_bar.at[0])
                    if n_head:
                        plgpu.copy_gmem_to_smem(head_gmem.at[h], head_smem, in_bar.at[0])
                    for i in range(n_vec):
                        for c in range(chunks):
                            chunk = pl.ds(c * TMA_ELEMENTS, TMA_ELEMENTS)
                            plgpu.copy_gmem_to_smem(vec_gmem.at[i, b, h, chunk], vec_smem.at[i, chunk], in_bar.at[0])

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
                                for i in range(n_parts):
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
    out_shape = jax.ShapeDtypeStruct((batch, time, heads, width), out_dtype)
    if n_key_sums:
        out_shape = (out_shape, jax.ShapeDtypeStruct((n_key_sums, batch, heads, time), jnp.float32))
    f = plgpu.kernel(
        kernel,
        out_shape=out_shape,
        grid=(time // (BK * WGS), batch, heads), grid_names=("k", "b", "h"),
        num_threads=WGS + 1, thread_name="wg", compiler_params=_params(family),
        scratch_shapes=[
            plgpu.SMEM((WGS, BK, dqk), jnp.bfloat16, transforms=t(dqk)),
            plgpu.SMEM((WGS, BK, dv), jnp.bfloat16, transforms=t(dv)),
            plgpu.SMEM((stages, block_q, dqk), jnp.bfloat16, transforms=t(dqk)),
            plgpu.SMEM((stages, block_q, dv), jnp.bfloat16, transforms=t(dv)),
            plgpu.SMEM((n_vec, time), jnp.float32),
            *([plgpu.SMEM((WGS, n_key, BK), jnp.float32)] if n_key else []),
            *([plgpu.SMEM((WGS, n_key_sums, BK), jnp.float32)] if n_key_sums else []),
            *([plgpu.SMEM((common.head_width(n_head),), jnp.float32)] if n_head else []),
            tuple(plgpu.RefUnion(
                s_region, tuple(plgpu.TMEM((BK, block_q), jnp.bfloat16, packed=True) for _ in range(n_parts)))
                for _ in range(WGS)),
            plgpu.TMEM((BK, WGS * width), jnp.float32),
            plgpu.Barrier(num_arrivals=2 * WGS + n_vec * chunks + WGS * n_key + bool(n_head), num_barriers=1),
            plgpu.Barrier(num_arrivals=2, num_barriers=stages),
            plgpu.Barrier(num_arrivals=1, num_barriers=stages, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
            plgpu.Barrier(num_arrivals=1, num_barriers=WGS, orders_tensor_core=True),
        ],
    )
    return f(q, k, v, do, vectors, *_inputs(key_vectors if n_key else None, head_constants if n_head else None))

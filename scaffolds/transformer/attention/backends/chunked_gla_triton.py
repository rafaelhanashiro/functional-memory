"""Fixed-shape chunk-parallel Triton training kernel for vector-gated GLA.

The implementation is intentionally narrow: equal-length batches, sequence
length divisible by 64, and no initial state. It computes the same inclusive
recurrence as ``GatedLinearAttention._chunkwise_gla`` while replacing token
state materialization with three GPU stages:

1. propagate and checkpoint one recurrent state per 64-token chunk;
2. construct each chunk's causal local interaction matrix from independent
   16-token block pairs in parallel;
3. combine the chunk-start state and local interactions into the read output.

The first stage is sequential over outer chunks for each state tile (32
iterations at the 2048-token target), while state tiles, heads, batches,
subchunk pairs, and output tiles are parallel. There is exactly one recurrent
input/output state per 64-token outer chunk. The 16-token subchunks are only a
numerical tiling of the local C x C term; no state is propagated between them.

The chunk formulation and tiling strategy are informed by the MIT-licensed
Flash Linear Attention project (copyright 2023-2026 Songlin Yang, Yu Zhang,
Zhiyuan Li), but this is a simplified fixed-shape JAX/Triton implementation.
The JAX/Triton call bridge comes from Apache-2.0-licensed ejkernel.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from scaffolds.transformer.attention.backends.ejkernel_bridge import triton_call


CHUNK_SIZE = 64
SUBCHUNK_SIZE = 16


@triton.jit
def _chunk_state_fwd_kernel(
    k,
    v,
    g_prefix,
    boundary_states,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Propagate one tiled state through chunks and save chunk-start states."""
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]
    state = tl.zeros([BK, BV], dtype=tl.float32)

    for i_c in range(N_CHUNKS):
        state_base = ((((i_b * (N_CHUNKS + 1) + i_c) * H + i_h) * K) * V).to(tl.int64)
        state_ptr = boundary_states + state_base + o_k[:, None] * V + o_v[None, :]
        tl.store(state_ptr, state, mask=mask_state)

        o_t = i_c * C + tl.arange(0, C)
        mask_t = o_t < T
        token_base_k = ((i_b * T + o_t[None, :]) * H + i_h).to(tl.int64) * K
        token_base_v = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * V
        k_ptr = k + token_base_k + o_k[:, None]
        v_ptr = v + token_base_v + o_v[None, :]
        g_ptr = g_prefix + token_base_k + o_k[:, None]
        last_g_base = (((i_b * T + i_c * C + C - 1) * H + i_h) * K).to(tl.int64)
        last_g = tl.load(g_prefix + last_g_base + o_k, mask=mask_k, other=0.0)

        k_block = tl.load(k_ptr, mask=mask_k[:, None] & mask_t[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptr, mask=mask_t[:, None] & mask_v[None, :], other=0.0).to(tl.float32)
        g_block = tl.load(g_ptr, mask=mask_k[:, None] & mask_t[None, :], other=0.0).to(tl.float32)
        weighted_k = k_block * tl.exp(last_g[:, None] - g_block)

        state *= tl.exp(last_g)[:, None]
        if DOT_IEEE:
            state += tl.dot(weighted_k, v_block, input_precision="ieee")
        else:
            state += tl.dot(weighted_k, v_block, input_precision="tf32")

    final_base = ((((i_b * (N_CHUNKS + 1) + N_CHUNKS) * H + i_h) * K) * V).to(tl.int64)
    final_ptr = boundary_states + final_base + o_k[:, None] * V + o_v[None, :]
    tl.store(final_ptr, state, mask=mask_state)


@triton.jit
def _chunk_local_matrix_fwd_kernel(
    q,
    k,
    g_prefix,
    local_matrix,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    SC: tl.constexpr,
    N_SUBCHUNKS: tl.constexpr,
    BK: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Construct one independent causal subchunk-pair score block.

    Off-diagonal blocks share the log-gate value immediately before the query
    subchunk. Because cumulative log gates are non-increasing, both resulting
    exponential factors are <= 1. Diagonal blocks instead form the physical
    relative decay directly and mask non-causal entries before ``exp``.
    """
    i_c = tl.program_id(0)
    i_pair = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H

    # For C=64,c=16 there are four subchunks and ten causal pairs. Decode
    # rows with triangular starts [0, 1, 3, 6]. The branch-free form avoids a
    # Triton loop-carried scalar while preserving one program per pair.
    tl.static_assert(N_SUBCHUNKS == 4)
    i_q_subchunk = (
        (i_pair >= 1).to(tl.int32)
        + (i_pair >= 3).to(tl.int32)
        + (i_pair >= 6).to(tl.int32)
    )
    i_k_subchunk = i_pair - i_q_subchunk * (i_q_subchunk + 1) // 2
    is_diagonal = i_q_subchunk == i_k_subchunk

    local_t = tl.arange(0, SC)
    local_s = tl.arange(0, SC)
    outer_t = i_q_subchunk * SC + local_t
    outer_s = i_k_subchunk * SC + local_s
    o_t = i_c * C + outer_t
    o_s = i_c * C + outer_s
    causal = local_t[:, None] >= local_s[None, :]
    score = tl.zeros([SC, SC], dtype=tl.float32)

    for i_k in range(0, tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < K
        q_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * K
        k_base = ((i_b * T + o_s[None, :]) * H + i_h).to(tl.int64) * K
        q_ptr = q + q_base + o_k[None, :]
        k_ptr = k + k_base + o_k[:, None]
        gq_ptr = g_prefix + q_base + o_k[None, :]
        gk_ptr = g_prefix + k_base + o_k[:, None]
        q_block = tl.load(q_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
        k_block = tl.load(k_ptr, mask=mask_k[:, None], other=0.0).to(tl.float32)
        gq = tl.load(gq_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
        gk = tl.load(gk_ptr, mask=mask_k[:, None], other=0.0).to(tl.float32)
        boundary_t = i_c * C + i_q_subchunk * SC - 1
        boundary_base = (((i_b * T + boundary_t) * H + i_h) * K).to(tl.int64)
        anchor = tl.load(
            g_prefix + boundary_base + o_k,
            mask=(i_q_subchunk > 0) & mask_k,
            other=0.0,
        ).to(tl.float32)

        # Disable the unused factorized path on diagonal programs before the
        # exponentials are evaluated. The actual off-diagonal arguments are
        # non-positive by construction.
        q_delta = tl.where(is_diagonal, 0.0, gq - anchor[None, :])
        k_delta = tl.where(is_diagonal, 0.0, anchor[:, None] - gk)
        q_scaled = q_block * tl.exp(q_delta)
        k_scaled = k_block * tl.exp(k_delta)
        if DOT_IEEE:
            off_diagonal_score = tl.dot(q_scaled, k_scaled, input_precision="ieee")
        else:
            off_diagonal_score = tl.dot(q_scaled, k_scaled, input_precision="tf32")

        relative_gate = gq[:, None, :] - tl.trans(gk)[None, :, :]
        safe_relative_gate = tl.where(causal[:, :, None], relative_gate, -float("inf"))
        diagonal_score = tl.sum(
            q_block[:, None, :]
            * tl.trans(k_block)[None, :, :]
            * tl.exp(safe_relative_gate),
            axis=2,
        )
        score += tl.where(is_diagonal, diagonal_score, off_diagonal_score)

    score = tl.where(is_diagonal & ~causal, 0.0, score)
    matrix_base = (((i_b * N_CHUNKS + i_c) * H + i_h) * C * C).to(tl.int64)
    matrix_ptr = local_matrix + matrix_base + outer_t[:, None] * C + outer_s[None, :]
    tl.store(matrix_ptr, score)


@triton.jit
def _chunk_output_fwd_kernel(
    q,
    v,
    g_prefix,
    boundary_states,
    local_matrix,
    output,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Add inter-chunk state reads and intra-chunk causal reads."""
    i_v = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_t = i_c * C + tl.arange(0, C)
    local_t = tl.arange(0, C)
    local_s = tl.arange(0, C)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V
    result = tl.zeros([C, BV], dtype=tl.float32)

    state_base = ((((i_b * (N_CHUNKS + 1) + i_c) * H + i_h) * K) * V).to(tl.int64)
    for i_k in range(0, tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < K
        q_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * K
        q_ptr = q + q_base + o_k[None, :]
        g_ptr = g_prefix + q_base + o_k[None, :]
        state_ptr = boundary_states + state_base + o_k[:, None] * V + o_v[None, :]
        q_block = tl.load(q_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
        g_block = tl.load(g_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
        state_block = tl.load(
            state_ptr,
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        q_scaled = q_block * tl.exp(g_block)
        if DOT_IEEE:
            result += tl.dot(q_scaled, state_block, input_precision="ieee")
        else:
            result += tl.dot(q_scaled, state_block, input_precision="tf32")

    matrix_base = (((i_b * N_CHUNKS + i_c) * H + i_h) * C * C).to(tl.int64)
    matrix_ptr = local_matrix + matrix_base + local_t[:, None] * C + local_s[None, :]
    v_base = ((i_b * T + (i_c * C + local_s)[:, None]) * H + i_h).to(tl.int64) * V
    v_ptr = v + v_base + o_v[None, :]
    local_scores = tl.load(matrix_ptr).to(tl.float32)
    value_block = tl.load(v_ptr, mask=mask_v[None, :], other=0.0).to(tl.float32)
    if DOT_IEEE:
        result += tl.dot(local_scores, value_block, input_precision="ieee")
    else:
        result += tl.dot(local_scores, value_block, input_precision="tf32")

    output_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * V
    output_ptr = output + output_base + o_v[None, :]
    tl.store(output_ptr, result, mask=mask_v[None, :])


@triton.jit
def _chunk_state_bwd_kernel(
    q,
    g_prefix,
    do,
    dfinal_state,
    boundary_state_grads,
    dinitial_state,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Reverse-propagate state adjoints and save dS at every chunk output."""
    i_k = tl.program_id(0)
    i_v = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_state = mask_k[:, None] & mask_v[None, :]

    final_base = ((i_bh * K) * V).to(tl.int64)
    final_ptr = dfinal_state + final_base + o_k[:, None] * V + o_v[None, :]
    state_grad = tl.load(final_ptr, mask=mask_state, other=0.0).to(tl.float32)

    for reverse_i in range(N_CHUNKS):
        i_c = N_CHUNKS - 1 - reverse_i
        grad_base = ((((i_b * N_CHUNKS + i_c) * H + i_h) * K) * V).to(tl.int64)
        grad_ptr = boundary_state_grads + grad_base + o_k[:, None] * V + o_v[None, :]
        tl.store(grad_ptr, state_grad, mask=mask_state)

        o_t = i_c * C + tl.arange(0, C)
        q_base = ((i_b * T + o_t[None, :]) * H + i_h).to(tl.int64) * K
        do_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * V
        q_ptr = q + q_base + o_k[:, None]
        g_ptr = g_prefix + q_base + o_k[:, None]
        do_ptr = do + do_base + o_v[None, :]
        q_block = tl.load(q_ptr, mask=mask_k[:, None], other=0.0).to(tl.float32)
        g_block = tl.load(g_ptr, mask=mask_k[:, None], other=0.0).to(tl.float32)
        do_block = tl.load(do_ptr, mask=mask_v[None, :], other=0.0).to(tl.float32)
        q_scaled = q_block * tl.exp(g_block)
        last_g_base = (((i_b * T + i_c * C + C - 1) * H + i_h) * K).to(tl.int64)
        last_g = tl.load(g_prefix + last_g_base + o_k, mask=mask_k, other=0.0)
        state_grad *= tl.exp(last_g)[:, None]
        if DOT_IEEE:
            state_grad += tl.dot(q_scaled, do_block, input_precision="ieee")
        else:
            state_grad += tl.dot(q_scaled, do_block, input_precision="tf32")

    initial_base = ((i_bh * K) * V).to(tl.int64)
    initial_ptr = dinitial_state + initial_base + o_k[:, None] * V + o_v[None, :]
    tl.store(initial_ptr, state_grad, mask=mask_state)


@triton.jit
def _chunk_dlocal_matrix_kernel(
    v,
    do,
    dlocal_matrix,
    T: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Compute dA = dY V^T for every local causal interaction matrix."""
    i_c = tl.program_id(0)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    local_t = tl.arange(0, C)
    local_s = tl.arange(0, C)
    o_t = i_c * C + local_t
    o_s = i_c * C + local_s
    dmatrix = tl.zeros([C, C], dtype=tl.float32)

    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = o_v < V
        do_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * V
        v_base = ((i_b * T + o_s[None, :]) * H + i_h).to(tl.int64) * V
        do_block = tl.load(
            do + do_base + o_v[None, :], mask=mask_v[None, :], other=0.0
        ).to(tl.float32)
        v_block = tl.load(
            v + v_base + o_v[:, None], mask=mask_v[:, None], other=0.0
        ).to(tl.float32)
        if DOT_IEEE:
            dmatrix += tl.dot(do_block, v_block, input_precision="ieee")
        else:
            dmatrix += tl.dot(do_block, v_block, input_precision="tf32")

    dmatrix = tl.where(local_t[:, None] >= local_s[None, :], dmatrix, 0.0)
    matrix_base = (((i_b * N_CHUNKS + i_c) * H + i_h) * C * C).to(tl.int64)
    matrix_ptr = dlocal_matrix + matrix_base + local_t[:, None] * C + local_s[None, :]
    tl.store(matrix_ptr, dmatrix)


@triton.jit
def _chunk_local_dqk_kernel(
    q,
    k,
    g_prefix,
    dlocal_matrix,
    dq_local,
    dk_local,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    SC: tl.constexpr,
    N_SUBCHUNKS: tl.constexpr,
    N_PAIRS: tl.constexpr,
    N_K_TILES: tl.constexpr,
    BK: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Accumulate physical Q/K gradients from parallel local block pairs.

    Each program handles one (outer chunk, causal subchunk pair, key tile).
    Multiple pairs contribute to a token, so fp32 atomics perform only the
    required reduction; there is no state or ordering dependency among pairs.
    """
    i_c = tl.program_id(0)
    i_pair_k = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    i_pair = i_pair_k // N_K_TILES
    i_k_tile = i_pair_k % N_K_TILES

    tl.static_assert(N_SUBCHUNKS == 4)
    i_q_subchunk = (
        (i_pair >= 1).to(tl.int32)
        + (i_pair >= 3).to(tl.int32)
        + (i_pair >= 6).to(tl.int32)
    )
    i_k_subchunk = i_pair - i_q_subchunk * (i_q_subchunk + 1) // 2
    is_diagonal = i_q_subchunk == i_k_subchunk

    local_t = tl.arange(0, SC)
    local_s = tl.arange(0, SC)
    outer_t = i_q_subchunk * SC + local_t
    outer_s = i_k_subchunk * SC + local_s
    o_t = i_c * C + outer_t
    o_s = i_c * C + outer_s
    o_k = i_k_tile * BK + tl.arange(0, BK)
    mask_k = o_k < K
    causal = local_t[:, None] >= local_s[None, :]

    q_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * K
    k_base = ((i_b * T + o_s[:, None]) * H + i_h).to(tl.int64) * K
    q_ptr = q + q_base + o_k[None, :]
    k_ptr = k + k_base + o_k[None, :]
    gq_ptr = g_prefix + q_base + o_k[None, :]
    gk_ptr = g_prefix + k_base + o_k[None, :]
    q_block = tl.load(q_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
    k_block = tl.load(k_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
    gq = tl.load(gq_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
    gk = tl.load(gk_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)

    matrix_base = (((i_b * N_CHUNKS + i_c) * H + i_h) * C * C).to(tl.int64)
    dmatrix_ptr = dlocal_matrix + matrix_base + outer_t[:, None] * C + outer_s[None, :]
    dmatrix = tl.load(dmatrix_ptr).to(tl.float32)

    boundary_t = i_c * C + i_q_subchunk * SC - 1
    boundary_base = (((i_b * T + boundary_t) * H + i_h) * K).to(tl.int64)
    anchor = tl.load(
        g_prefix + boundary_base + o_k,
        mask=(i_q_subchunk > 0) & mask_k,
        other=0.0,
    ).to(tl.float32)
    q_delta = tl.where(is_diagonal, 0.0, gq - anchor[None, :])
    k_delta = tl.where(is_diagonal, 0.0, anchor[None, :] - gk)
    q_factor = tl.exp(q_delta)
    k_factor = tl.exp(k_delta)
    q_scaled = q_block * q_factor
    k_scaled = k_block * k_factor
    if DOT_IEEE:
        dq_scaled = tl.dot(dmatrix, k_scaled, input_precision="ieee")
        dk_scaled = tl.dot(tl.trans(dmatrix), q_scaled, input_precision="ieee")
    else:
        dq_scaled = tl.dot(dmatrix, k_scaled, input_precision="tf32")
        dk_scaled = tl.dot(tl.trans(dmatrix), q_scaled, input_precision="tf32")
    dq_off_diagonal = dq_scaled * q_factor
    dk_off_diagonal = dk_scaled * k_factor

    relative_gate = gq[:, None, :] - gk[None, :, :]
    safe_relative_gate = tl.where(causal[:, :, None], relative_gate, -float("inf"))
    weighted_dmatrix = dmatrix[:, :, None] * tl.exp(safe_relative_gate)
    dq_diagonal = tl.sum(weighted_dmatrix * k_block[None, :, :], axis=1)
    dk_diagonal = tl.sum(weighted_dmatrix * q_block[:, None, :], axis=0)

    dq_contribution = tl.where(is_diagonal, dq_diagonal, dq_off_diagonal)
    dk_contribution = tl.where(is_diagonal, dk_diagonal, dk_off_diagonal)
    tl.atomic_add(dq_local + q_base + o_k[None, :], dq_contribution, mask=mask_k[None, :])
    tl.atomic_add(dk_local + k_base + o_k[None, :], dk_contribution, mask=mask_k[None, :])


@triton.jit
def _chunk_state_dqkg_kernel(
    q,
    k,
    v,
    g_prefix,
    do,
    boundary_states,
    boundary_state_grads,
    dq_local,
    dk_local,
    dq,
    dk,
    dg,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Add outer-state gradients and convert prefix-gate gradients."""
    i_k = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    local_t = tl.arange(0, C)
    local_s = tl.arange(0, C)
    o_t = i_c * C + local_t
    o_k = i_k * BK + tl.arange(0, BK)
    mask_k = o_k < K

    token_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * K
    q_ptr = q + token_base + o_k[None, :]
    k_ptr = k + token_base + o_k[None, :]
    g_ptr = g_prefix + token_base + o_k[None, :]
    q_block = tl.load(q_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
    k_block = tl.load(k_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
    g_block = tl.load(g_ptr, mask=mask_k[None, :], other=0.0).to(tl.float32)
    dq_block = tl.load(
        dq_local + token_base + o_k[None, :], mask=mask_k[None, :], other=0.0
    ).to(tl.float32)
    dk_block = tl.load(
        dk_local + token_base + o_k[None, :], mask=mask_k[None, :], other=0.0
    ).to(tl.float32)
    dg_prefix = dq_block * q_block - dk_block * k_block

    state0_base = ((((i_b * (N_CHUNKS + 1) + i_c) * H + i_h) * K) * V).to(tl.int64)
    state1_base = ((((i_b * (N_CHUNKS + 1) + i_c + 1) * H + i_h) * K) * V).to(tl.int64)
    dstate1_base = ((((i_b * N_CHUNKS + i_c) * H + i_h) * K) * V).to(tl.int64)
    last_g_base = (((i_b * T + i_c * C + C - 1) * H + i_h) * K).to(tl.int64)
    last_g = tl.load(g_prefix + last_g_base + o_k, mask=mask_k, other=0.0)
    last_gate_grad = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(0, tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        mask_v = o_v < V
        do_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * V
        v_base = do_base
        do_block = tl.load(
            do + do_base + o_v[None, :], mask=mask_v[None, :], other=0.0
        ).to(tl.float32)
        v_block = tl.load(
            v + v_base + o_v[None, :], mask=mask_v[None, :], other=0.0
        ).to(tl.float32)
        state0 = tl.load(
            boundary_states + state0_base + o_k[:, None] * V + o_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        state1 = tl.load(
            boundary_states + state1_base + o_k[:, None] * V + o_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        dstate1 = tl.load(
            boundary_state_grads + dstate1_base + o_k[:, None] * V + o_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        if DOT_IEEE:
            dq_state_base = tl.dot(do_block, tl.trans(state0), input_precision="ieee")
            dk_state_base = tl.dot(v_block, tl.trans(dstate1), input_precision="ieee")
        else:
            dq_state_base = tl.dot(do_block, tl.trans(state0), input_precision="tf32")
            dk_state_base = tl.dot(v_block, tl.trans(dstate1), input_precision="tf32")
        dq_state = dq_state_base * tl.exp(g_block)
        dk_state = dk_state_base * tl.exp(last_g[None, :] - g_block)
        dq_block += dq_state
        dk_block += dk_state
        dg_prefix += dq_state * q_block - dk_state * k_block
        last_gate_grad += tl.sum(dstate1 * state1, axis=1)

    dg_prefix += tl.where(
        local_t[:, None] == C - 1,
        last_gate_grad[None, :],
        0.0,
    )
    dg_block = tl.cumsum(dg_prefix, axis=0, reverse=True)
    tl.store(dq + token_base + o_k[None, :], dq_block, mask=mask_k[None, :])
    tl.store(dk + token_base + o_k[None, :], dk_block, mask=mask_k[None, :])
    tl.store(dg + token_base + o_k[None, :], dg_block, mask=mask_k[None, :])


@triton.jit
def _chunk_dv_kernel(
    k,
    g_prefix,
    do,
    boundary_state_grads,
    local_matrix,
    dv,
    T: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    N_CHUNKS: tl.constexpr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    DOT_IEEE: tl.constexpr,
):
    """Compute value gradients from local outputs and chunk state updates."""
    i_v = tl.program_id(0)
    i_c = tl.program_id(1)
    i_bh = tl.program_id(2).to(tl.int64)
    i_b = i_bh // H
    i_h = i_bh % H
    local_t = tl.arange(0, C)
    local_s = tl.arange(0, C)
    o_t = i_c * C + local_t
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < V

    matrix_base = (((i_b * N_CHUNKS + i_c) * H + i_h) * C * C).to(tl.int64)
    matrix_ptr = local_matrix + matrix_base + local_t[:, None] * C + local_s[None, :]
    matrix = tl.load(matrix_ptr).to(tl.float32)
    do_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * V
    do_block = tl.load(
        do + do_base + o_v[None, :], mask=mask_v[None, :], other=0.0
    ).to(tl.float32)
    if DOT_IEEE:
        dv_block = tl.dot(tl.trans(matrix), do_block, input_precision="ieee")
    else:
        dv_block = tl.dot(tl.trans(matrix), do_block, input_precision="tf32")

    dstate1_base = ((((i_b * N_CHUNKS + i_c) * H + i_h) * K) * V).to(tl.int64)
    for i_k in range(0, tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        mask_k = o_k < K
        token_base = ((i_b * T + o_t[:, None]) * H + i_h).to(tl.int64) * K
        k_block = tl.load(
            k + token_base + o_k[None, :], mask=mask_k[None, :], other=0.0
        ).to(tl.float32)
        g_block = tl.load(
            g_prefix + token_base + o_k[None, :], mask=mask_k[None, :], other=0.0
        ).to(tl.float32)
        last_g_base = (((i_b * T + i_c * C + C - 1) * H + i_h) * K).to(tl.int64)
        last_g = tl.load(g_prefix + last_g_base + o_k, mask=mask_k, other=0.0)
        dstate1 = tl.load(
            boundary_state_grads + dstate1_base + o_k[:, None] * V + o_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
        weighted_k = k_block * tl.exp(last_g[None, :] - g_block)
        if DOT_IEEE:
            dv_block += tl.dot(weighted_k, dstate1, input_precision="ieee")
        else:
            dv_block += tl.dot(weighted_k, dstate1, input_precision="tf32")

    dv_ptr = dv + do_base + o_v[None, :]
    tl.store(dv_ptr, dv_block, mask=mask_v[None, :])


def _validate_and_dimensions(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    *,
    chunk_size: int,
    subchunk_size: int,
    dot_precision: str,
) -> tuple[int, int, int, int, int, int, int, int, int, int, bool]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or log_alpha.ndim != 4:
        raise ValueError("q, k, v, and log_alpha must be rank-4 [B,T,H,D] arrays")
    if q.shape != k.shape or q.shape != log_alpha.shape:
        raise ValueError("q, k, and log_alpha must have identical shapes")
    if q.shape[:3] != v.shape[:3]:
        raise ValueError("q/k and v must share [B,T,H]")
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"Two-level kernel requires outer chunk_size={CHUNK_SIZE}")
    if q.shape[1] % chunk_size:
        raise ValueError("Sequence length must be divisible by chunk_size")
    if subchunk_size != SUBCHUNK_SIZE:
        raise ValueError(f"Two-level kernel requires subchunk_size={SUBCHUNK_SIZE}")
    if chunk_size % subchunk_size:
        raise ValueError("chunk_size must be divisible by subchunk_size")
    if dot_precision not in {"tf32", "ieee"}:
        raise ValueError("dot_precision must be 'tf32' or 'ieee'")

    batch, seq_len, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    num_chunks = seq_len // chunk_size
    num_subchunks = chunk_size // subchunk_size
    num_pairs = num_subchunks * (num_subchunks + 1) // 2
    # Triton dot operands need tiles of at least 16, including narrow d_K.
    bk = max(16, min(64, triton.next_power_of_2(key_dim)))
    bv = max(16, min(64, triton.next_power_of_2(value_dim)))
    dot_ieee = dot_precision == "ieee"
    return (
        batch,
        seq_len,
        heads,
        key_dim,
        value_dim,
        num_chunks,
        num_subchunks,
        num_pairs,
        bk,
        bv,
        dot_ieee,
    )


def _fp32_operands(*arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Upcast operands before launch; every kernel loads them as FP32 anyway.

    With BF16 operands upcast inside the kernel, Triton 3.6 on sm_103 (B300)
    lowers the FP32 IEEE products at TF32-level precision. The conversion is
    exact, so the requested arithmetic is unchanged on every GPU.
    """
    return tuple(array.astype(jnp.float32) for array in arrays)


def _gate_prefix(log_alpha: jax.Array, num_chunks: int, chunk_size: int) -> jax.Array:
    batch, _, heads, key_dim = log_alpha.shape
    gate_chunks = log_alpha.astype(jnp.float32).reshape(
        batch, num_chunks, chunk_size, heads, key_dim
    )
    return jnp.cumsum(gate_chunks, axis=2).reshape(log_alpha.shape)


def chunked_gla_forward_intermediates(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    *,
    chunk_size: int = CHUNK_SIZE,
    subchunk_size: int = SUBCHUNK_SIZE,
    dot_precision: str = "ieee",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return output, final state, boundary states, and local score matrices."""
    output_dtype = q.dtype
    q, k, v = _fp32_operands(q, k, v)
    (
        batch,
        seq_len,
        heads,
        key_dim,
        value_dim,
        num_chunks,
        num_subchunks,
        num_pairs,
        bk,
        bv,
        dot_ieee,
    ) = _validate_and_dimensions(
        q,
        k,
        v,
        log_alpha,
        chunk_size=chunk_size,
        subchunk_size=subchunk_size,
        dot_precision=dot_precision,
    )

    gate_prefix = _gate_prefix(log_alpha, num_chunks, chunk_size)

    boundary_states = triton_call(
        k,
        v,
        gate_prefix,
        kernel=_chunk_state_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch, num_chunks + 1, heads, key_dim, value_dim), jnp.float32
        ),
        name="ofm::triton::chunked_gla_state_fwd",
        grid=(triton.cdiv(key_dim, bk), triton.cdiv(value_dim, bv), batch * heads),
        num_warps=4,
        num_stages=2,
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        BK=bk,
        BV=bv,
        DOT_IEEE=dot_ieee,
    )
    local_matrix = triton_call(
        q,
        k,
        gate_prefix,
        kernel=_chunk_local_matrix_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch, num_chunks, heads, chunk_size, chunk_size), jnp.float32
        ),
        name="ofm::triton::chunked_gla_two_level_local_fwd",
        grid=(num_chunks, num_pairs, batch * heads),
        num_warps=4,
        num_stages=2,
        zeroed_outputs=(0,),
        T=seq_len,
        H=heads,
        K=key_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        SC=subchunk_size,
        N_SUBCHUNKS=num_subchunks,
        BK=bk,
        DOT_IEEE=dot_ieee,
    )
    output_fp32 = triton_call(
        q,
        v,
        gate_prefix,
        boundary_states,
        local_matrix,
        kernel=_chunk_output_fwd_kernel,
        out_shape=jax.ShapeDtypeStruct(
            (batch, seq_len, heads, value_dim), jnp.float32
        ),
        name="ofm::triton::chunked_gla_output_fwd",
        grid=(triton.cdiv(value_dim, bv), num_chunks, batch * heads),
        num_warps=8,
        num_stages=2,
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        BK=bk,
        BV=bv,
        DOT_IEEE=dot_ieee,
    )
    return (
        output_fp32.astype(output_dtype),
        boundary_states[:, -1],
        boundary_states,
        local_matrix,
    )


def chunked_gla_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    *,
    chunk_size: int = CHUNK_SIZE,
    subchunk_size: int = SUBCHUNK_SIZE,
    dot_precision: str = "ieee",
) -> tuple[jax.Array, jax.Array]:
    """Run the fixed-shape chunk-parallel forward and return output/final state."""
    output, final_state, _, _ = chunked_gla_forward_intermediates(
        q,
        k,
        v,
        log_alpha,
        chunk_size=chunk_size,
        subchunk_size=subchunk_size,
        dot_precision=dot_precision,
    )
    return output, final_state


def chunked_gla_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    do: jax.Array,
    dfinal_state: jax.Array,
    boundary_states: jax.Array,
    local_matrix: jax.Array,
    *,
    chunk_size: int = CHUNK_SIZE,
    subchunk_size: int = SUBCHUNK_SIZE,
    dot_precision: str = "ieee",
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Run the explicit chunked backward for Q, K, V, and log-alpha."""
    dtypes = q.dtype, k.dtype, v.dtype
    q, k, v, do = _fp32_operands(q, k, v, do)
    (
        batch,
        seq_len,
        heads,
        key_dim,
        value_dim,
        num_chunks,
        num_subchunks,
        num_pairs,
        bk,
        bv,
        dot_ieee,
    ) = _validate_and_dimensions(
        q,
        k,
        v,
        log_alpha,
        chunk_size=chunk_size,
        subchunk_size=subchunk_size,
        dot_precision=dot_precision,
    )
    expected_boundaries = (batch, num_chunks + 1, heads, key_dim, value_dim)
    expected_local = (batch, num_chunks, heads, chunk_size, chunk_size)
    if boundary_states.shape != expected_boundaries:
        raise ValueError(
            f"boundary_states has shape {boundary_states.shape}, expected {expected_boundaries}"
        )
    if local_matrix.shape != expected_local:
        raise ValueError(f"local_matrix has shape {local_matrix.shape}, expected {expected_local}")
    if do.shape != v.shape:
        raise ValueError(f"do has shape {do.shape}, expected {v.shape}")
    expected_final = (batch, heads, key_dim, value_dim)
    if dfinal_state.shape != expected_final:
        raise ValueError(
            f"dfinal_state has shape {dfinal_state.shape}, expected {expected_final}"
        )

    gate_prefix = _gate_prefix(log_alpha, num_chunks, chunk_size)
    boundary_state_grads, _ = triton_call(
        q,
        gate_prefix,
        do,
        dfinal_state,
        kernel=_chunk_state_bwd_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(
                (batch, num_chunks, heads, key_dim, value_dim), jnp.float32
            ),
            jax.ShapeDtypeStruct(expected_final, jnp.float32),
        ],
        name="ofm::triton::chunked_gla_state_bwd",
        grid=(triton.cdiv(key_dim, bk), triton.cdiv(value_dim, bv), batch * heads),
        num_warps=4,
        num_stages=2,
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        BK=bk,
        BV=bv,
        DOT_IEEE=dot_ieee,
    )
    dlocal_matrix = triton_call(
        v,
        do,
        kernel=_chunk_dlocal_matrix_kernel,
        out_shape=jax.ShapeDtypeStruct(expected_local, jnp.float32),
        name="ofm::triton::chunked_gla_dlocal",
        grid=(num_chunks, batch * heads),
        num_warps=8,
        num_stages=2,
        T=seq_len,
        H=heads,
        V=value_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        BV=bv,
        DOT_IEEE=dot_ieee,
    )
    num_k_tiles = triton.cdiv(key_dim, bk)
    dq_local, dk_local = triton_call(
        q,
        k,
        gate_prefix,
        dlocal_matrix,
        kernel=_chunk_local_dqk_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, jnp.float32),
            jax.ShapeDtypeStruct(k.shape, jnp.float32),
        ],
        name="ofm::triton::chunked_gla_two_level_local_dqk",
        grid=(num_chunks, num_pairs * num_k_tiles, batch * heads),
        num_warps=4,
        num_stages=2,
        zeroed_outputs=(0, 1),
        T=seq_len,
        H=heads,
        K=key_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        SC=subchunk_size,
        N_SUBCHUNKS=num_subchunks,
        N_PAIRS=num_pairs,
        N_K_TILES=num_k_tiles,
        BK=bk,
        DOT_IEEE=dot_ieee,
    )
    dq_fp32, dk_fp32, dg = triton_call(
        q,
        k,
        v,
        gate_prefix,
        do,
        boundary_states,
        boundary_state_grads,
        dq_local,
        dk_local,
        kernel=_chunk_state_dqkg_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, jnp.float32),
            jax.ShapeDtypeStruct(k.shape, jnp.float32),
            jax.ShapeDtypeStruct(log_alpha.shape, jnp.float32),
        ],
        name="ofm::triton::chunked_gla_state_dqkg",
        grid=(triton.cdiv(key_dim, bk), num_chunks, batch * heads),
        num_warps=8,
        num_stages=2,
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        BK=bk,
        BV=bv,
        DOT_IEEE=dot_ieee,
    )
    dv_fp32 = triton_call(
        k,
        gate_prefix,
        do,
        boundary_state_grads,
        local_matrix,
        kernel=_chunk_dv_kernel,
        out_shape=jax.ShapeDtypeStruct(v.shape, jnp.float32),
        name="ofm::triton::chunked_gla_dv",
        grid=(triton.cdiv(value_dim, bv), num_chunks, batch * heads),
        num_warps=8,
        num_stages=2,
        T=seq_len,
        H=heads,
        K=key_dim,
        V=value_dim,
        N_CHUNKS=num_chunks,
        C=chunk_size,
        BK=bk,
        BV=bv,
        DOT_IEEE=dot_ieee,
    )
    return dq_fp32.astype(dtypes[0]), dk_fp32.astype(dtypes[1]), dv_fp32.astype(dtypes[2]), dg


def chunked_gla_forward_backward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
    do: jax.Array,
    *,
    chunk_size: int = CHUNK_SIZE,
    subchunk_size: int = SUBCHUNK_SIZE,
    dot_precision: str = "ieee",
) -> tuple[jax.Array, jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    """Explicit combined path used to validate and time the prototype VJP."""
    output, final_state, boundaries, local_matrix = chunked_gla_forward_intermediates(
        q,
        k,
        v,
        log_alpha,
        chunk_size=chunk_size,
        subchunk_size=subchunk_size,
        dot_precision=dot_precision,
    )
    gradients = chunked_gla_backward(
        q,
        k,
        v,
        log_alpha,
        do,
        jnp.zeros_like(final_state),
        boundaries,
        local_matrix,
        chunk_size=chunk_size,
        subchunk_size=subchunk_size,
        dot_precision=dot_precision,
    )
    return output, final_state, gradients


@jax.custom_vjp
def chunked_gla(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Training-ready two-level IEEE GLA with an explicit custom VJP."""
    return chunked_gla_forward(
        q,
        k,
        v,
        log_alpha,
        chunk_size=CHUNK_SIZE,
        subchunk_size=SUBCHUNK_SIZE,
        dot_precision="ieee",
    )


def _chunked_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    log_alpha: jax.Array,
):
    output, final_state, boundaries, local_matrix = chunked_gla_forward_intermediates(
        q,
        k,
        v,
        log_alpha,
        chunk_size=CHUNK_SIZE,
        subchunk_size=SUBCHUNK_SIZE,
        dot_precision="ieee",
    )
    residual = q, k, v, log_alpha, boundaries, local_matrix
    return (output, final_state), residual


def _chunked_gla_bwd(
    residual: tuple[jax.Array, ...],
    cotangents: tuple[jax.Array, jax.Array],
):
    q, k, v, log_alpha, boundaries, local_matrix = residual
    do, dfinal_state = cotangents
    return chunked_gla_backward(
        q,
        k,
        v,
        log_alpha,
        do,
        dfinal_state,
        boundaries,
        local_matrix,
        chunk_size=CHUNK_SIZE,
        subchunk_size=SUBCHUNK_SIZE,
        dot_precision="ieee",
    )


chunked_gla.defvjp(_chunked_gla_fwd, _chunked_gla_bwd)


__all__ = [
    "CHUNK_SIZE",
    "SUBCHUNK_SIZE",
    "chunked_gla",
    "chunked_gla_backward",
    "chunked_gla_forward",
    "chunked_gla_forward_backward",
    "chunked_gla_forward_intermediates",
]

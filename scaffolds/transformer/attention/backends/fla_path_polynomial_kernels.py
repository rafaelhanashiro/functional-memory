# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license (flash-linear-attention).
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
"""FLA's six softmax-dependent PaTH kernels with an unnormalized per-query polynomial.

Copied from flash-linear-attention 0.6.0 (``fla/ops/path_attn``: intra_chunk_preprocess_fwd,
parallel_path_fwd, intra_chunk_preprocess_bwd_prepare, parallel_path_bwd_inter_dkv,
parallel_path_bwd_inter_dqh, parallel_path_bwd_intra). The forget-gate and variable-length
branches are removed and the kernels take the scaffold's bridge argument order (inputs,
arrays updated in place, outputs, constexprs); transport, Householder, and UT-transform
arithmetic, dtype casts, and tiling are FLA's, line for line.

The kernel value of a transported score ``s = scale * S_ts`` is ``p_t(s) = sum_r c_r(t) s^r``
with FP32 coefficients ``c`` [B, T, HQ, DEGREE + 1] (DEGREE 1-4), evaluated in FP32 by
Horner's rule. The output is ``o_t = sum_s p_t(s_ts) v_s`` with no normalization, so the
online-softmax statistics (L, M) and the backward's ``delta`` disappear: the softmax's
``dS = P (dP - delta)`` becomes ``dS = p_t'(s) dP`` (times ``scale`` for the raw score), and
each kernel that visits a (query, key) block also writes the query's coefficient gradient
``dc_r(t) = sum_s s^r dP_ts`` for the keys it covers.
"""

import triton
import triton.language as tl


@triton.jit
def _coefficient(c, rows, mask, HQ: tl.constexpr, R: tl.constexpr, r: tl.constexpr):
    """c_r for the given token rows (zero for orders above the degree)."""
    if r < R:
        value = tl.load(c + rows * (HQ * R) + r, mask=mask, other=0.0)
    else:
        value = tl.zeros(rows.shape, dtype=tl.float32)
    return value


@triton.jit
def _polynomial(s, c0, c1, c2, c3, c4, DEGREE: tl.constexpr):
    """p(s) by Horner's rule; each c_r is already shaped to broadcast against s."""
    if DEGREE == 1:
        p = c1 * s + c0
    elif DEGREE == 2:
        p = (c2 * s + c1) * s + c0
    elif DEGREE == 3:
        p = ((c3 * s + c2) * s + c1) * s + c0
    else:
        p = (((c4 * s + c3) * s + c2) * s + c1) * s + c0
    return p


@triton.jit
def _derivative(s, c1, c2, c3, c4, DEGREE: tl.constexpr):
    """p'(s) by Horner's rule."""
    if DEGREE == 1:
        d = c1 + tl.zeros(s.shape, dtype=tl.float32)
    elif DEGREE == 2:
        d = 2.0 * c2 * s + c1
    elif DEGREE == 3:
        d = (3.0 * c3 * s + 2.0 * c2) * s + c1
    else:
        d = ((4.0 * c4 * s + 3.0 * c3) * s + 2.0 * c2) * s + c1
    return d


@triton.jit
def _moments(s, dp, m0, m1, m2, m3, m4, DEGREE: tl.constexpr, AXIS: tl.constexpr):
    """Add sum_s s^r dP along AXIS to the coefficient-gradient accumulators."""
    m0 += tl.sum(dp, AXIS)
    power = dp * s
    m1 += tl.sum(power, AXIS)
    if DEGREE >= 2:
        power = power * s
        m2 += tl.sum(power, AXIS)
    if DEGREE >= 3:
        power = power * s
        m3 += tl.sum(power, AXIS)
    if DEGREE >= 4:
        power = power * s
        m4 += tl.sum(power, AXIS)
    return m0, m1, m2, m3, m4


@triton.jit
def _store_moments(dc, rows, mask, m0, m1, m2, m3, m4, HQ: tl.constexpr, R: tl.constexpr):
    p = dc + rows * (HQ * R)
    tl.store(p, m0, mask=mask)
    tl.store(p + 1, m1, mask=mask)
    if R > 2:
        tl.store(p + 2, m2, mask=mask)
    if R > 3:
        tl.store(p + 3, m3, mask=mask)
    if R > 4:
        tl.store(p + 4, m4, mask=mask)


@triton.jit(do_not_specialize=['T'])
def intra_fwd(q, k, v, w, beta, c, A, scale, T, o, w2, q_new, k_new, H: tl.constexpr, G: tl.constexpr,
              HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
              BT: tl.constexpr, DEGREE: tl.constexpr):
    """intra_chunk_preprocess_fwd_kernel: the chunk-local UT transform and outputs."""
    R: tl.constexpr = DEGREE + 1
    i_t, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G
    bos = (i_n * T).to(tl.int64)

    A += (bos*H + i_h) * BT
    q += (bos*HQ + i_hq) * K
    q_new += (bos*HQ + i_hq) * K
    k += (bos*H + i_h) * K
    k_new += (bos*H + i_h) * K
    w2 += (bos*H + i_h) * K
    w += (bos*H + i_h) * K
    v += (bos*H + i_h) * V
    o += (bos*HQ + i_hq) * V
    beta += (bos*H + i_h)
    c += (bos*HQ + i_hq) * R

    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_r = o_t < T
    m_k = m_r[:, None] & (o_d[None, :] < K)
    m_v = m_r[:, None] & (o_v[None, :] < V)
    p_q = q + o_t[:, None] * (HQ*K) + o_d[None, :]
    p_k = k + o_d[:, None] + o_t[None, :] * (H*K)
    p_w = w + o_t[:, None] * (H*K) + o_d[None, :]
    p_v = v + o_t[:, None] * (H*V) + o_v[None, :]
    p_beta = beta + o_t * H
    p_T = A + o_t[:, None] * (BT*H) + o_i[None, :]

    b_beta = tl.load(p_beta, mask=m_r, other=0.0)
    b_q = tl.load(p_q, mask=m_k, other=0.0)
    b_kt = tl.load(p_k, mask=(o_d[:, None] < K) & m_r[None, :], other=0.0)
    b_v = tl.load(p_v, mask=m_v, other=0.0)
    b_w = tl.load(p_w, mask=m_k, other=0.0)
    b_T = tl.load(p_T, mask=m_r[:, None] & (o_i[None, :] < BT), other=0.0)
    b_T = b_T * b_beta[None, :]

    m_t = o_i[:, None] >= o_i[None, :]

    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w.to(b_q.dtype))), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T.to(b_q.dtype)).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w.to(b_q.dtype), b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT.to(b_q.dtype), b_wbk), 0)

    b_q = b_q.to(tl.float32) - tl.dot(b_qwT, b_w.to(b_q.dtype))
    p_q_new = q_new + o_t[:, None] * (K*HQ) + o_d[None, :]
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), mask=m_k)

    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w)
        p_w2 = w2 + o_t[:, None] * (K*H) + o_d[None, :]
        tl.store(p_w2, b_Twb.to(p_w2.dtype.element_ty), mask=m_k)
        b_T_wbk = tl.dot(b_T.to(b_kt.dtype), b_wbk).to(b_kt.dtype)
        p_k_new = k_new + o_d[:, None] + o_t[None, :] * (K*H)
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w.to(b_kt.dtype)), b_T_wbk)
                           ).to(p_k_new.dtype.element_ty), mask=(o_d[:, None] < K) & m_r[None, :])

    # The polynomial replaces the chunk-local online softmax.
    c0 = _coefficient(c, o_t, m_r, HQ, R, 0)[:, None]
    c1 = _coefficient(c, o_t, m_r, HQ, R, 1)[:, None]
    c2 = _coefficient(c, o_t, m_r, HQ, R, 2)[:, None]
    c3 = _coefficient(c, o_t, m_r, HQ, R, 3)[:, None]
    c4 = _coefficient(c, o_t, m_r, HQ, R, 4)[:, None]
    b_p = tl.where(m_t, _polynomial(b_A * scale, c0, c1, c2, c3, c4, DEGREE), 0)
    b_o = tl.dot(b_p.to(b_v.dtype), b_v)
    p_o = o + o_t[:, None] * (V*HQ) + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_v)


@triton.jit(do_not_specialize=['T'])
def path_fwd(q, k, v, o, w1, w2, c, scale, T, o_new, G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
             K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
             BV: tl.constexpr, DEGREE: tl.constexpr):
    """parallel_path_fwd_kernel: keys outside the query's chunk, moving the queries back."""
    R: tl.constexpr = DEGREE + 1
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int64)

    o_q = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_q = o_q < T
    m_o = m_q[:, None] & (o_v[None, :] < V)
    p_q = q + (bos * HQ + i_hq) * K + o_q[:, None] * (HQ*K) + o_d[None, :]
    b_q = tl.zeros([BT, BK], dtype=tl.float32)
    b_q += tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    p_o = o + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    b_o += tl.load(p_o, mask=m_o, other=0.0)

    c += (bos * HQ + i_hq) * R
    c0 = _coefficient(c, o_q, m_q, HQ, R, 0)[:, None]
    c1 = _coefficient(c, o_q, m_q, HQ, R, 1)[:, None]
    c2 = _coefficient(c, o_q, m_q, HQ, R, 2)[:, None]
    c3 = _coefficient(c, o_q, m_q, HQ, R, 3)[:, None]
    c4 = _coefficient(c, o_q, m_q, HQ, R, 4)[:, None]

    for offset in range((i_t + 1) * BT - 2 * BS, i_t*BT-BS, -BS):
        o_k = offset + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (K*H)  # GQA when H!=HQ
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (V*H) + o_v[None, :]  # GQA when H!=HQ
        p_w1 = w1 + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (K*H)
        p_w2 = w2 + (bos * H + i_h) * K + o_k[:, None] * (K*H) + o_d[None, :]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_w1 = tl.load(p_w1, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_w2 = tl.load(p_w2, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
        m_s = i_t * BT + tl.arange(0, BT) >= (offset + BS)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        b_p = tl.where(m_s[:, None], _polynomial(b_s * scale, c0, c1, c2, c3, c4, DEGREE), 0)
        b_o = tl.dot(b_p.to(b_v.dtype), b_v, b_o)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_s2 = tl.where(m_s[:, None], b_s2, 0)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)

    tl.debug_barrier()

    for offset in range(i_t * BT - BS, -BS, -BS):
        o_k = offset + tl.arange(0, BS)
        m_k = o_k < T
        p_k = k + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (K*H)  # GQA when H!=HQ
        p_v = v + (bos * H + i_h) * V + o_k[:, None] * (V*H) + o_v[None, :]  # GQA when H!=HQ
        p_w1 = w1 + (bos * H + i_h) * K + o_d[:, None] + o_k[None, :] * (K*H)
        p_w2 = w2 + (bos * H + i_h) * K + o_k[:, None] * (K*H) + o_d[None, :]
        b_k = tl.load(p_k, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
        b_w1 = tl.load(p_w1, mask=(o_d[:, None] < K) & m_k[None, :], other=0.0)
        b_w2 = tl.load(p_w2, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
        b_s = tl.dot(b_q.to(b_k.dtype), b_k)
        b_p = _polynomial(b_s * scale, c0, c1, c2, c3, c4, DEGREE)
        b_o = tl.dot(b_p.to(b_v.dtype), b_v, b_o)
        b_s2 = tl.dot(b_q.to(b_w1.dtype), b_w1)
        b_q -= tl.dot(b_s2.to(b_w2.dtype), b_w2)

    p_o_new = o_new + (bos * HQ + i_hq) * V + o_q[:, None] * (HQ*V) + o_v[None, :]
    tl.store(p_o_new, b_o.to(p_o_new.dtype.element_ty), mask=m_o)


@triton.jit(do_not_specialize=['T'])
def bwd_prepare(q, k, v, w, beta, c, AT, do, scale, T, h, q_new, k_new, dA_local, dv, dc, G: tl.constexpr,
                HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                BV: tl.constexpr, BT: tl.constexpr, DEGREE: tl.constexpr):
    """chunk_transform_qk_bwd_kernel_prepare: chunk-local dv, dS, and dc."""
    R: tl.constexpr = DEGREE + 1
    i_t, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G
    bos = (i_n * T).to(tl.int64)

    dA_local += (bos*HQ + i_hq) * BT
    AT += (bos*H + i_h) * BT
    q += (bos*HQ + i_hq) * K
    q_new += (bos*HQ + i_hq) * K
    k += (bos*H + i_h) * K
    k_new += (bos*H + i_h) * K
    w += (bos*H + i_h) * K
    v += (bos*H + i_h) * V
    do += (bos*HQ + i_hq) * V
    dv += (bos*HQ + i_hq) * V
    beta += (bos*H + i_h)
    h += (bos*H + i_h) * K
    c += (bos*HQ + i_hq) * R
    dc += (bos*HQ + i_hq) * R

    o_t = i_t * BT + tl.arange(0, BT)
    o_i = tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_r = o_t < T
    m_k = m_r[:, None] & (o_d[None, :] < K)
    m_v = m_r[:, None] & (o_v[None, :] < V)
    m_T = m_r[:, None] & (o_i[None, :] < BT)
    p_q = q + o_t[:, None] * (HQ*K) + o_d[None, :]
    p_k = k + o_d[:, None] + o_t[None, :] * (H*K)
    p_w = w + o_t[:, None] * (H*K) + o_d[None, :]
    p_beta = beta + o_t * H

    b_q = tl.load(p_q, mask=m_k, other=0.0)
    b_kt = tl.load(p_k, mask=(o_d[:, None] < K) & m_r[None, :], other=0.0)
    b_w = tl.load(p_w, mask=m_k, other=0.0)
    b_beta = tl.load(p_beta, mask=m_r, other=0.0)
    p_T = AT + o_t[:, None] * (BT*H) + o_i[None, :]
    b_T = tl.load(p_T, mask=m_T, other=0.0) * b_beta[None, :]

    m_t = o_i[:, None] >= o_i[None, :]
    b_qw = tl.where(m_t, tl.dot(b_q, tl.trans(b_w.to(b_q.dtype))), 0).to(b_q.dtype)
    b_qwT = tl.dot(b_qw, b_T.to(b_q.dtype)).to(b_q.dtype)
    b_wbk = tl.where(o_i[:, None] > o_i[None, :], tl.dot(b_w.to(b_kt.dtype), b_kt), 0).to(b_q.dtype)
    b_A = tl.where(m_t, tl.dot(b_q, b_kt) - tl.dot(b_qwT, b_wbk), 0)

    b_q = b_q.to(tl.float32) - tl.dot(b_qwT, b_w.to(b_qwT.dtype))
    p_q_new = q_new + o_t[:, None] * (K*HQ) + o_d[None, :]
    tl.store(p_q_new, b_q.to(p_q_new.dtype.element_ty), mask=m_k)

    if i_hq % G == 0:
        b_Twb = tl.dot(b_T, b_w)  # tf32
        p_h = h + o_t[:, None] * (K * H) + o_d[None, :]
        tl.store(p_h, b_Twb.to(p_h.dtype.element_ty), mask=m_k)
        b_T_wbk = tl.dot(b_T.to(b_wbk.dtype), b_wbk).to(b_kt.dtype)
        p_k_new = k_new + o_d[:, None] + o_t[None, :] * (K*H)
        tl.store(p_k_new, (b_kt - tl.dot(tl.trans(b_w.to(b_kt.dtype)), b_T_wbk)
                           ).to(p_k_new.dtype.element_ty), mask=(o_d[:, None] < K) & m_r[None, :])

    c0 = _coefficient(c, o_t, m_r, HQ, R, 0)[:, None]
    c1 = _coefficient(c, o_t, m_r, HQ, R, 1)[:, None]
    c2 = _coefficient(c, o_t, m_r, HQ, R, 2)[:, None]
    c3 = _coefficient(c, o_t, m_r, HQ, R, 3)[:, None]
    c4 = _coefficient(c, o_t, m_r, HQ, R, 4)[:, None]
    b_s = b_A * scale
    b_P = tl.where(m_t, _polynomial(b_s, c0, c1, c2, c3, c4, DEGREE), 0)
    p_do = do + o_t[:, None] * (HQ*V) + o_v[None, :]
    b_do = tl.load(p_do, mask=m_v, other=0.0)
    b_dv = tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)
    p_dv = dv + o_t[:, None] * (HQ*V) + o_v[None, :]
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), mask=m_v)

    p_v = v + o_v[:, None] + o_t[None, :] * (H*V)
    b_v = tl.load(p_v, mask=(o_v[:, None] < V) & m_r[None, :], other=0.0)
    b_dp = tl.where(m_t, tl.dot(b_do, b_v), 0)
    b_dA = tl.where(m_t, _derivative(b_s, c1, c2, c3, c4, DEGREE) * b_dp * scale, 0)
    p_dA = dA_local + o_t[:, None] * (BT*HQ) + o_i[None, :]
    tl.store(p_dA, b_dA.to(p_dA.dtype.element_ty), mask=m_T)

    zero = tl.zeros([BT], dtype=tl.float32)
    m0, m1, m2, m3, m4 = _moments(b_s, b_dp, zero, zero, zero, zero, zero, DEGREE, 1)
    _store_moments(dc, o_t, m_r, m0, m1, m2, m3, m4, HQ, R)


@triton.jit(do_not_specialize=['T'])
def bwd_dkv(q, k, v, c, do, scale, T, dv_in, dk, dv, G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
            K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr,
            BV: tl.constexpr, S: tl.constexpr, NUM_BLOCKS: tl.constexpr, DEGREE: tl.constexpr):
    """parallel_path_bwd_dkv_kernel: dk and dv from queries in later splits (dv_in aliases dv)."""
    R: tl.constexpr = DEGREE + 1
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int64)

    do += (bos * HQ + i_hq) * V
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * V
    c += (bos * HQ + i_hq) * R
    k += (bos * H + i_h) * K  # GQA when H!=HQ
    v += (bos * H + i_h) * V  # GQA when H!=HQ

    o_k = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_k = o_k < T
    p_k = k + o_k[:, None] * (H*K) + o_d[None, :]
    b_k = tl.load(p_k, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
    p_v = v + o_k[:, None] * (H*V) + o_v[None, :]
    b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)

    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dv = tl.zeros([BT, BV], dtype=tl.float32)

    last_chunk_start = tl.floor(i_t*BT / S).to(tl.int32) * S
    idx_j = (tl.floor(i_t * BT / S).to(tl.int32) + 1).to(tl.int32)

    last_chunk_end = tl.ceil(T / BS).to(tl.int32) * BS - BS

    for offset in range(last_chunk_end, last_chunk_start+S-BS, -BS):
        o_q = (offset + tl.arange(0, BS)).to(tl.int64)
        m_q = o_q < T
        p_q = q + ((bos.to(tl.int64) * NUM_BLOCKS + idx_j) * HQ + i_hq) * K + o_q[:, None] * (HQ*K*NUM_BLOCKS) + o_d[None, :]
        b_q = tl.load(p_q, mask=m_q[:, None] & (o_d[None, :] < K), other=0.0)
        b_A = tl.dot(b_k, tl.trans(b_q).to(b_k.dtype))
        b_s = b_A * scale
        c0 = _coefficient(c, o_q, m_q, HQ, R, 0)[None, :]
        c1 = _coefficient(c, o_q, m_q, HQ, R, 1)[None, :]
        c2 = _coefficient(c, o_q, m_q, HQ, R, 2)[None, :]
        c3 = _coefficient(c, o_q, m_q, HQ, R, 3)[None, :]
        c4 = _coefficient(c, o_q, m_q, HQ, R, 4)[None, :]
        b_P = _polynomial(b_s, c0, c1, c2, c3, c4, DEGREE)
        p_do = do + o_q[:, None] * (HQ*V) + o_v[None, :]
        b_do = tl.load(p_do, mask=m_q[:, None] & (o_v[None, :] < V), other=0.0)
        b_dv = tl.dot(b_P.to(b_do.dtype), b_do, b_dv)
        b_dp = tl.dot(b_v, tl.trans(b_do))

        b_dA = _derivative(b_s, c1, c2, c3, c4, DEGREE) * b_dp * scale
        b_dk = tl.dot(b_dA.to(b_q.dtype), b_q, b_dk)

    p_dk = dk + o_k[:, None] * (HQ*K) + o_d[None, :]
    tl.store(p_dk, b_dk.to(dk.dtype.element_ty), mask=m_k[:, None] & (o_d[None, :] < K))
    mask = i_t * BT + tl.arange(0, BT) < T
    tl.atomic_add(
        dv + (i_t * BT + tl.arange(0, BT))[:, None] * HQ * V + tl.arange(0, BV)[None, :],
        b_dv,
        mask=mask[:, None],
        sem='relaxed',
    )


@triton.jit(do_not_specialize=['T'])
def bwd_dq(q, k, v, hc_whole, c, do, scale, T, dq, dhc_whole, dc, G: tl.constexpr, HQ: tl.constexpr,
           H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
           BK: tl.constexpr, BV: tl.constexpr, S: tl.constexpr, NUM_BLOCKS: tl.constexpr, DEGREE: tl.constexpr):
    """parallel_path_bwd_dq_kernel: dq, the split products' gradient, and dc from earlier splits."""
    R: tl.constexpr = DEGREE + 1
    i_t, i_nh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_n, i_hq = i_nh // HQ, i_nh % HQ
    i_h = i_hq // G
    bos = (i_n * T).to(tl.int64)
    boh_large = (i_n * tl.cdiv(T, S)).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)

    k += (bos * H + i_h) * K  # GQA when H!=HQ
    v += (bos * H + i_h) * V  # GQA when H!=HQ
    do += (bos * HQ + i_hq) * V
    dq += (bos * HQ + i_hq) * K
    hc_whole += (boh_large * H + i_h) * K * K
    dhc_whole += (boh_large * HQ + i_hq) * K * K
    c += (bos * HQ + i_hq) * R
    dc += (bos * HQ + i_hq) * R

    stride_h = H * K * K
    stride_hq = HQ * K * K

    p_do = do + o_t[:, None] * (HQ*V) + o_v[None, :]
    b_do = tl.load(p_do, mask=m_t[:, None] & (o_v[None, :] < V), other=0.0)
    c0 = _coefficient(c, o_t, m_t, HQ, R, 0)[:, None]
    c1 = _coefficient(c, o_t, m_t, HQ, R, 1)[:, None]
    c2 = _coefficient(c, o_t, m_t, HQ, R, 2)[:, None]
    c3 = _coefficient(c, o_t, m_t, HQ, R, 3)[:, None]
    c4 = _coefficient(c, o_t, m_t, HQ, R, 4)[:, None]
    m0 = tl.zeros([BT], dtype=tl.float32)
    m1 = tl.zeros([BT], dtype=tl.float32)
    m2 = tl.zeros([BT], dtype=tl.float32)
    m3 = tl.zeros([BT], dtype=tl.float32)
    m4 = tl.zeros([BT], dtype=tl.float32)

    curr_end = ((i_t * BT // S) * S).to(tl.int32)
    b_dq = tl.zeros([BT, K], dtype=tl.float32)

    for offset_outer in range(0, curr_end, S):
        idx_j = offset_outer // S
        p_q = q + ((bos.to(tl.int64) * NUM_BLOCKS + idx_j + 1) * HQ + i_hq) * \
            K + o_t[:, None] * (HQ*K*NUM_BLOCKS) + o_d[None, :]
        b_q = tl.load(p_q, mask=m_t[:, None] & (o_d[None, :] < K), other=0.0)

        b_dh = -tl.dot(tl.trans(b_q), b_dq.to(b_q.dtype))
        tl.atomic_add(dhc_whole + idx_j * stride_hq + tl.arange(0, K)
                      [:, None] * K + tl.arange(0, K)[None, :], b_dh, sem='relaxed')
        p_h = hc_whole + idx_j * stride_h + o_d[:, None] * K + o_d[None, :]
        b_h = tl.load(p_h, mask=(o_d[:, None] < K) & (o_d[None, :] < K), other=0.0)
        b_dq = b_dq - tl.dot(b_dq.to(b_h.dtype), tl.trans(b_h))

        for offset in range(offset_outer, min(offset_outer+S, i_t*BT), BS):
            o_k = (offset + tl.arange(0, BS)).to(tl.int64)
            m_k = o_k < T
            p_k = k + o_k[:, None] * (H * K) + o_d[None, :]
            b_k = tl.load(p_k, mask=m_k[:, None] & (o_d[None, :] < K), other=0.0)
            b_A = tl.dot(b_q, tl.trans(b_k).to(b_q.dtype))
            b_s = b_A * scale
            p_v = v + o_k[:, None] * (H*V) + o_v[None, :]
            b_v = tl.load(p_v, mask=m_k[:, None] & (o_v[None, :] < V), other=0.0)
            b_dp = tl.where(m_t[:, None], tl.dot(b_do, tl.trans(b_v).to(b_do.dtype)), 0)
            b_dA = _derivative(b_s, c1, c2, c3, c4, DEGREE) * b_dp * scale
            b_dq = tl.dot(b_dA.to(b_k.dtype), b_k, b_dq)
            m0, m1, m2, m3, m4 = _moments(b_s, b_dp, m0, m1, m2, m3, m4, DEGREE, 1)

    p_dq = dq + o_t[:, None] * (K * HQ) + o_d[None, :]
    tl.store(p_dq, b_dq.to(dq.dtype.element_ty), mask=m_t[:, None] & (o_d[None, :] < K))
    _store_moments(dc, o_t, m_t, m0, m1, m2, m3, m4, HQ, R)


@triton.jit(do_not_specialize=['T'])
def bwd_intra(q, k, v, w1, w2, c, dq, do, T, scale, dk_in, dv_in, dw1_in, dw2_in, dq_new, dk, dv, dw1, dw2, dc,
              G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr, S: tl.constexpr, DEGREE: tl.constexpr):
    """parallel_path_bwd_intra_chunk_kernel: earlier chunks of the query's split (*_in alias outputs)."""
    R: tl.constexpr = DEGREE + 1
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_hq = i_bh // HQ, i_bh % HQ
    i_h = i_hq // G
    bos = (i_b * T).to(tl.int64)

    k += (bos * H + i_h) * K  # GQA when H!=HQ
    v += (bos * H + i_h) * V  # GQA when H!=HQ
    w1 += (bos * H + i_h) * K
    w2 += (bos * H + i_h) * K

    q += (bos * HQ + i_hq) * K
    dq += (bos * HQ + i_hq) * K
    dq_new += (bos * HQ + i_hq) * K
    dk += (bos * HQ + i_hq) * K
    dv += (bos * HQ + i_hq) * V
    do += (bos * HQ + i_hq) * V
    dw1 += (bos * HQ + i_hq) * K
    dw2 += (bos * HQ + i_hq) * K
    c += (bos * HQ + i_hq) * R
    dc += (bos * HQ + i_hq) * R

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    m_t = o_t < T
    m_k = m_t[:, None] & (o_d[None, :] < K)
    p_do = do + o_t[:, None] * (HQ*V) + o_v[None, :]
    b_do = tl.load(p_do, mask=m_t[:, None] & (o_v[None, :] < V), other=0.0)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    p_dq = dq + o_t[:, None] * (HQ*K) + o_d[None, :]
    b_dq += tl.load(p_dq, mask=m_k, other=0.0)
    p_q = q + o_t[:, None] * (HQ*K) + o_d[None, :]
    b_q = tl.load(p_q, mask=m_k, other=0.0)

    c0 = _coefficient(c, o_t, m_t, HQ, R, 0)[:, None]
    c1 = _coefficient(c, o_t, m_t, HQ, R, 1)[:, None]
    c2 = _coefficient(c, o_t, m_t, HQ, R, 2)[:, None]
    c3 = _coefficient(c, o_t, m_t, HQ, R, 3)[:, None]
    c4 = _coefficient(c, o_t, m_t, HQ, R, 4)[:, None]
    m0 = tl.zeros([BT], dtype=tl.float32)
    m1 = tl.zeros([BT], dtype=tl.float32)
    m2 = tl.zeros([BT], dtype=tl.float32)
    m3 = tl.zeros([BT], dtype=tl.float32)
    m4 = tl.zeros([BT], dtype=tl.float32)

    curr_start = (tl.floor(i_t * BT / S).to(tl.int32) * S).to(tl.int32)

    for offset in range(curr_start, i_t * BT, BT):
        o_k = (offset + tl.arange(0, BT)).to(tl.int64)
        mask = o_k < T
        m_kk = mask[:, None] & (o_d[None, :] < K)
        p_k = k + o_k[:, None] * (H*K) + o_d[None, :]
        b_k = tl.load(p_k, mask=m_kk, other=0.0)
        b_q_tmp = tl.zeros([BT, BK], dtype=tl.float32)
        b_q_tmp += b_q
        for i_t_small in range(i_t * BT - BT, offset, -BT):
            o_w = i_t_small + tl.arange(0, BT)
            m_w = (o_w[:, None] < T) & (o_d[None, :] < K)
            p_w1 = w1 + o_w[:, None] * (H*K) + o_d[None, :]
            b_w1 = tl.load(p_w1, mask=m_w, other=0.0)
            p_w2 = w2 + o_w[:, None] * (H*K) + o_d[None, :]
            b_w2 = tl.load(p_w2, mask=m_w, other=0.0)
            b_A_tmp = tl.dot(b_q_tmp.to(b_w1.dtype), tl.trans(b_w1))
            b_q_tmp -= tl.dot(b_A_tmp.to(b_w1.dtype), b_w2)
        b_q2 = b_q_tmp.to(b_k.dtype)
        b_A = tl.dot(b_q2, tl.trans(b_k))
        b_s = b_A * scale
        b_P = _polynomial(b_s, c0, c1, c2, c3, c4, DEGREE)
        b_dv = tl.dot(tl.trans(b_P.to(b_do.dtype)), b_do)
        tl.atomic_add(
            dv + ((offset + tl.arange(0, BT)) * HQ * V)[:, None] + tl.arange(0, BV)[None, :],
            b_dv.to(dv.dtype.element_ty),
            mask=mask[:, None],
            sem='relaxed',
        )
        p_v = v + o_k[:, None] * (V*H) + o_v[None, :]
        b_v = tl.load(p_v, mask=mask[:, None] & (o_v[None, :] < V), other=0.0)
        b_dp = tl.where(m_t[:, None], tl.dot(b_do, tl.trans(b_v)), 0)
        b_dA = _derivative(b_s, c1, c2, c3, c4, DEGREE) * b_dp * scale
        m0, m1, m2, m3, m4 = _moments(b_s, b_dp, m0, m1, m2, m3, m4, DEGREE, 1)
        b_dA = b_dA.to(b_v.dtype)
        b_dk = tl.dot(tl.trans(b_dA), b_q2)
        tl.atomic_add(dk + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dk, mask=mask[:, None], sem='relaxed')
        p_w1 = w1 + o_k[:, None] * (H*K) + o_d[None, :]
        b_w1 = tl.load(p_w1, mask=m_kk, other=0.0)
        p_w2 = w2 + o_k[:, None] * (H*K) + o_d[None, :]
        b_w2 = tl.load(p_w2, mask=m_kk, other=0.0)
        b_dA2 = tl.dot(b_dq.to(b_w2.dtype), tl.trans(b_w2)).to(b_v.dtype)
        b_A2 = tl.dot(b_q2.to(b_w1.dtype), tl.trans(b_w1)).to(b_v.dtype)
        b_dw2 = -tl.dot(tl.trans(b_A2), b_dq.to(b_v.dtype))
        tl.atomic_add(dw2 + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dw2, mask=mask[:, None], sem='relaxed')
        b_dw1 = -tl.dot(tl.trans(b_dA2), b_q2.to(b_v.dtype))
        tl.atomic_add(dw1 + (offset + tl.arange(0, BT))[:, None] * HQ*K + tl.arange(0,
                      BK)[None, :], b_dw1, mask=mask[:, None], sem='relaxed')
        b_dq -= tl.dot(b_dA2, b_w1.to(b_v.dtype))
        b_dq = tl.dot(b_dA.to(b_k.dtype), b_k, b_dq)

    p_dq_new = dq_new + o_t[:, None] * (HQ*K) + o_d[None, :]
    tl.store(p_dq_new, b_dq.to(dq_new.dtype.element_ty), mask=m_k)
    _store_moments(dc, o_t, m_t, m0, m1, m2, m3, m4, HQ, R)

"""Separate erase/write gates in FLA's WY preparation; all other kernels are shared.

Adapted from ``fla/ops/gated_delta_rule/wy_fast.py`` at the dependency's pinned
commit 516143e31fce09925e6c39ac37148444bad176c4. The scaffold uses equal-length
sequences and scalar per-head decay. With cumulative log2 decay G and
A = (I + strict_lower[erase_i exp2(G_i-G_j) k_i^T k_j])^{-1}, these kernels
prepare W = A (erase exp2(G) K) and U = A (beta V), and differentiate both.
They never divide by either gate, including at erase = 0.

MIT License
Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import triton
import triton.language as tl

from fla.ops.utils.op import exp2


@triton.jit
def recompute_w_u_independent(k, v, beta, erase, A, g, T, w, u, H: tl.constexpr,
                              HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                              BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_A = m_t[:, None] & (o_A[None, :] < BT)
    o_gate = bos * HV + i_h + o_t * HV
    b_beta = tl.load(beta + o_gate, mask=m_t, other=0.0)
    b_erase = tl.load(erase + o_gate, mask=m_t, other=0.0)
    p_A = A + (bos * HV + i_h) * BT + o_t[:, None] * (HV * BT) + o_A[None, :]
    b_A = tl.load(p_A, mask=m_A, other=0.0)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        o_value = (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(v + o_value, mask=m_v, other=0.0)
        b_v_beta = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_v_beta, input_precision="tf32x3")
        tl.store(u + o_value, b_u.to(u.dtype.element_ty), mask=m_v)

    b_g = exp2(tl.load(g + o_gate, mask=m_t, other=0.0))
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_w = w + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_ke = b_k * b_erase[:, None]
        b_ke *= b_g[:, None]
        b_w = tl.dot(b_A, b_ke.to(b_k.dtype))
        tl.store(p_w, b_w.to(p_w.dtype.element_ty), mask=m_k)


@triton.jit
def wy_bwd_independent(k, v, beta, erase, g, A, dw, du, T, dk, dv, db, de, dg,
                      H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                      V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV
    bos = i_b * T
    o_t = i_t * BT + tl.arange(0, BT)
    o_A = tl.arange(0, BT)
    m_t = o_t < T
    m_AT = (o_A[:, None] < BT) & m_t[None, :]
    o_gate = bos * HV + i_h + o_t * HV
    p_A = A + (bos * HV + i_h) * BT + o_A[:, None] + o_t[None, :] * (HV * BT)
    b_beta = tl.load(beta + o_gate, mask=m_t, other=0.0)
    b_erase = tl.load(erase + o_gate, mask=m_t, other=0.0)
    b_db = tl.zeros([BT], dtype=tl.float32)
    b_de = tl.zeros([BT], dtype=tl.float32)
    # b_A holds A^T, as in the upstream adjoint.
    b_A = tl.load(p_A, mask=m_AT, other=0.0)
    b_dA = tl.zeros([BT, BT], dtype=tl.float32)
    b_g = tl.load(g + o_gate, mask=m_t, other=0.0)
    b_g_exp = exp2(b_g)
    b_dg = tl.zeros([BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        o_key = (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_keg = b_k * (b_erase * b_g_exp)[:, None]
        b_dw = tl.load(dw + o_key, mask=m_k, other=0.0)
        b_dA = tl.dot(b_dw, tl.trans(b_keg).to(b_dw.dtype), b_dA)
        b_dkeg = tl.dot(b_A, b_dw)
        b_dk = b_dkeg * (b_g_exp * b_erase)[:, None]
        b_de += tl.sum(b_dkeg * b_k * b_g_exp[:, None], 1)
        b_dg += tl.sum(b_dkeg * b_keg, 1)
        tl.store(dk + o_key, b_dk.to(dk.dtype.element_ty), mask=m_k)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        o_value = (bos * HV + i_h) * V + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(v + o_value, mask=m_v, other=0.0)
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(du + o_value, mask=m_v, other=0.0)
        b_dA = tl.dot(b_du, tl.trans(b_vb), b_dA)
        b_dvb = tl.dot(b_A, b_du)
        b_dv = b_dvb * b_beta[:, None]
        b_db += tl.sum(b_dvb * b_v, 1)
        tl.store(dv + o_value, b_dv.to(dv.dtype.element_ty), mask=m_v)

    # Backpropagate through the triangular inverse. Only erase controls its
    # key-key interactions; dA above contains contributions from both W and U.
    m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t)
    b_dA = tl.where(m_A, b_dA, 0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype))
    b_dA *= exp2(b_g[:, None] - b_g[None, :])
    b_A = tl.zeros([BT, BT], dtype=tl.float32)
    b_dA = tl.where(m_A, -b_dA, 0).to(k.dtype.element_ty)
    tl.debug_barrier()

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_t[:, None] & (o_k[None, :] < K)
        p_k = k + (bos * H + i_h // (HV // H)) * K + o_t[:, None] * (H * K) + o_k[None, :]
        p_dk = dk + (bos * HV + i_h) * K + o_t[:, None] * (HV * K) + o_k[None, :]
        b_k = tl.load(p_k, mask=m_k, other=0.0)
        b_ke = b_k * b_erase[:, None]
        b_A = tl.dot(b_k, tl.trans(b_k), b_A)
        b_dke = tl.dot(b_dA, b_k)
        b_de += tl.sum(b_dke * b_k, 1)
        b_dk = b_dke * b_erase[:, None] + tl.trans(tl.dot(tl.trans(b_ke).to(b_dA.dtype), b_dA))
        b_dk += tl.load(p_dk, mask=m_k, other=0.0)
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_k)
    tl.store(db + o_gate, b_db.to(db.dtype.element_ty), mask=m_t)
    tl.store(de + o_gate, b_de.to(de.dtype.element_ty), mask=m_t)
    b_A *= b_erase[:, None]
    b_AdA = b_dA * b_A
    b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
    tl.store(dg + o_gate, b_dg.to(dg.dtype.element_ty), mask=m_t)

"""Bridge-order wrappers around FLA's unmodified PaTH attention kernels.

Each wrapper takes its arrays and scalars in the order ``triton_port.launch`` passes them
(inputs, arrays updated in place, outputs, constexprs) and calls the FLA kernel body as a
device function, with FLA's heuristic flags fixed for the scaffold's path: equal-length
sequences (``IS_VARLEN``), no forget gate (``USE_G``/``USE_GATE``), no TMA, and no
gradient of the key cache. Pointers FLA would receive as ``None`` stay ``None``.
Importing this module imports flash-linear-attention's Triton kernels.
"""

from importlib import import_module

import triton
import triton.language as tl

from scaffolds.transformer.attention.backends.triton_port import jit_function


def _fla(module, kernel):
    # Modules, not package attributes: fla.ops.utils exports a solve_tril function.
    return jit_function(getattr(import_module(f"fla.ops.{module}"), kernel))


FLA_KKT = _fla("common.chunk_scaled_dot_kkt", "chunk_scaled_dot_kkt_fwd_kernel")
FLA_SOLVE_TRIL_32 = _fla("utils.solve_tril", "merge_16x16_to_32x32_inverse_kernel")
FLA_SOLVE_TRIL_64 = _fla("utils.solve_tril", "merge_16x16_to_64x64_inverse_kernel")
FLA_INTRA_FWD = _fla("path_attn.intra_chunk_preprocess_fwd", "intra_chunk_preprocess_fwd_kernel")
FLA_PATH_FWD = _fla("path_attn.parallel_path_fwd", "parallel_path_fwd_kernel")
FLA_BWD_PREPROCESS = _fla("attn.parallel", "parallel_attn_bwd_kernel_preprocess")
FLA_BWD_PREPARE = _fla("path_attn.intra_chunk_preprocess_bwd_prepare", "chunk_transform_qk_bwd_kernel_prepare")
FLA_CUMPROD_FWD = _fla("path_attn.cumprod_householder_fwd", "chunk_cumprod_householder_fwd_kernel")
FLA_TRANSFORM_Q = _fla("path_attn.transform_q", "transform_q_fwd_kernel")
FLA_BWD_DKV = _fla("path_attn.parallel_path_bwd_inter_dkv", "parallel_path_bwd_dkv_kernel")
FLA_BWD_DQ = _fla("path_attn.parallel_path_bwd_inter_dqh", "parallel_path_bwd_dq_kernel")
FLA_CUMPROD_BWD = _fla("path_attn.cumprod_householder_bwd", "chunk_cumprod_householder_bwd_kernel")
FLA_BWD_INTRA = _fla("path_attn.parallel_path_bwd_intra", "parallel_path_bwd_intra_chunk_kernel")
FLA_INTRA_BWD = _fla("path_attn.intra_chunk_preprocess_bwd", "intra_chunk_preprocess_bwd_kernel")


@triton.jit
def kkt(k, beta, T, A, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    # IS_VARLEN, USE_G
    FLA_KKT(k, None, beta, A, None, None, T, H, HV, K, BT, BK, False, False)


@triton.jit
def solve_tril_32(A, T, Ai, H: tl.constexpr, BT: tl.constexpr, DOT_PRECISION: tl.constexpr):
    # USE_TMA, IS_VARLEN
    FLA_SOLVE_TRIL_32(A, Ai, None, None, T, H, BT, False, False, DOT_PRECISION)


@triton.jit
def solve_tril_64(A, T, Ai, H: tl.constexpr, BT: tl.constexpr, DOT_PRECISION: tl.constexpr):
    FLA_SOLVE_TRIL_64(A, Ai, None, None, T, H, BT, False, False, DOT_PRECISION)


@triton.jit
def intra_fwd(q, k, v, w, beta, A, scale, T, o, L, M, w2, q_new, k_new, H: tl.constexpr, G: tl.constexpr,
              HQ: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
              BT: tl.constexpr):
    # IS_VARLEN, USE_G
    FLA_INTRA_FWD(q, k, v, w, beta, None, o, A, L, M, w2, q_new, k_new, scale, None, None, T, H, G, HQ, K, V,
                  BK, BV, BT, False, False)


@triton.jit
def path_fwd(q, k, v, o, w1, w2, L, M, scale, T, o_new, L_new, G: tl.constexpr, HQ: tl.constexpr,
             H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
             BK: tl.constexpr, BV: tl.constexpr):
    # USE_GATE, IS_VARLEN
    FLA_PATH_FWD(q, k, v, o, o_new, None, w1, w2, scale, L, L_new, M, None, None, T, G, HQ, H, K, V, BT, BS,
                 BK, BV, False, False)


@triton.jit
def bwd_preprocess(o, do, delta, B: tl.constexpr, V: tl.constexpr):
    FLA_BWD_PREPROCESS(o, do, delta, B, V)


@triton.jit
def bwd_prepare(q, k, v, w, beta, L, D, AT, do, scale, T, h, q_new, k_new, dA_local, dv, G: tl.constexpr,
                HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                BV: tl.constexpr, BT: tl.constexpr):
    # IS_VARLEN, USE_GATE, RETURN_H
    FLA_BWD_PREPARE(q, k, v, w, beta, None, L, D, h, q_new, k_new, AT, dA_local, dv, do, None, scale, None, None,
                    None, T, G, HQ, H, K, V, BK, BV, BT, False, False, False)


@triton.jit
def cumprod_fwd(k, w1, w2, k_new, hc_suffix, hc_whole, BT: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                BK: tl.constexpr, T: tl.constexpr, S: tl.constexpr):
    # IS_VARLEN
    FLA_CUMPROD_FWD(k, k_new, w1, w2, hc_suffix, hc_whole, None, None, None, None, BT, K, H, BK, T, S, False)


@triton.jit
def transform_q(q, w1, w2, T, q_new, S: tl.constexpr, G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr,
                K: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr, BK: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    # IS_VARLEN
    FLA_TRANSFORM_Q(q, q_new, w1, w2, None, None, T, S, G, HQ, H, K, BT, BS, BK, NUM_BLOCKS, False)


@triton.jit
def bwd_dkv(q, k, v, hc_whole, L, D, do, scale, T, dv_in, dk, dv, G: tl.constexpr, HQ: tl.constexpr,
            H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
            BK: tl.constexpr, BV: tl.constexpr, S: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    # dv_in is aliased to dv, which the kernel accumulates into. IS_VARLEN, USE_GATE
    FLA_BWD_DKV(q, k, v, None, hc_whole, scale, L, D, dk, dv, do, None, None, None, None, T, G, HQ, H, K, V, BT,
                BS, BK, BV, S, False, False, NUM_BLOCKS)


@triton.jit
def bwd_dq(q, k, v, hc_whole, L, D, do, scale, T, dq, dhc_whole, G: tl.constexpr, HQ: tl.constexpr,
           H: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BS: tl.constexpr,
           BK: tl.constexpr, BV: tl.constexpr, S: tl.constexpr, NUM_BLOCKS: tl.constexpr):
    # IS_VARLEN, USE_GATE
    FLA_BWD_DQ(q, k, v, None, hc_whole, scale, L, D, dq, do, dhc_whole, None, None, None, None, T, G, HQ, H, K, V,
               BT, BS, BK, BV, S, NUM_BLOCKS, False, False)


@triton.jit
def cumprod_bwd(hc_suffix, dhc_whole, k, dk, w1, w2, T, dw1, dw2, dk_new, BT: tl.constexpr, K: tl.constexpr,
                BK: tl.constexpr, S: tl.constexpr, G: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr):
    # IS_VARLEN
    FLA_CUMPROD_BWD(hc_suffix, dhc_whole, k, dk, w1, w2, dw1, dw2, dk_new, None, None, None, None, BT, K, BK, T,
                    S, G, H, HQ, False)


@triton.jit
def bwd_intra(q, k, v, w1, w2, L, D, dq, do, T, scale, dk_in, dv_in, dw1_in, dw2_in, dq_new, dk, dv, dw1, dw2,
              G: tl.constexpr, HQ: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              BK: tl.constexpr, BV: tl.constexpr, BT: tl.constexpr, S: tl.constexpr):
    # The *_in arrays are aliased to the outputs the kernel accumulates into. IS_VARLEN, USE_GATE
    FLA_BWD_INTRA(q, k, v, None, w1, w2, L, D, dq, dq_new, dk, dv, dw1, dw2, do, None, None, None, T, scale, G, HQ,
                  H, K, V, BK, BV, BT, S, False, False)


@triton.jit
def intra_bwd(q, k, w, w2, beta, AT, dA_local, dq, dk, dw1, dw2, T, dq_new, dk_new, dw, dbeta, HQ: tl.constexpr,
              G: tl.constexpr, H: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    # IS_VARLEN
    FLA_INTRA_BWD(q, k, w, w2, beta, AT, dA_local, dq, dq_new, dk, dk_new, dw, dbeta, dw1, dw2, T, None, None, HQ, G,
                  H, K, BT, BK, False)

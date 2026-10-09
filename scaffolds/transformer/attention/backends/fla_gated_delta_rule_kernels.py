"""Bridge-order wrappers around FLA's unmodified chunked gated-delta-rule kernels.

Importing this module imports flash-linear-attention's Triton kernels, which needs a
visible GPU. Each wrapper takes its arrays and scalars in the order the vendored
bridge passes them (inputs, then outputs) and calls the FLA kernel body as a device
function, with FLA's heuristic flags fixed for the scaffold's path: equal-length
sequences, no initial or final state, keys-first state layout, gate and beta given.
Autotuned block sizes and launch parameters are chosen by the caller.
"""

import triton
import triton.language as tl

from fla.ops.common import chunk_delta_h as _fla_delta_h
from fla.ops.common import chunk_o as _fla_chunk_o
from fla.ops.gated_delta_rule import chunk_fwd as _fla_chunk_fwd
from fla.ops.gated_delta_rule import wy_fast as _fla_wy_fast

from scaffolds.transformer.attention.backends.fla_independent_gated_delta_rule_kernels import (
    recompute_w_u_independent,
    wy_bwd_independent,
)


def _base(kernel):
    """FLA kernels are Heuristics(Autotuner(JITFunction)); the wrappers call the JIT function."""
    while not isinstance(kernel, triton.JITFunction):
        kernel = kernel.fn
    return kernel


# FLA picks this from torch's view of the device at import; with the scaffold's CPU-only
# torch it would fall back to IEEE. Use FLA's GPU choice (TF32, every sm >= 80).
_fla_chunk_fwd.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")

FLA_KKT_SOLVE = _base(_fla_chunk_fwd.chunk_gated_delta_rule_fwd_kkt_solve_kernel)
FLA_RECOMPUTE_W_U = _base(_fla_wy_fast.recompute_w_u_fwd_kernel)
FLA_FWD_H = _base(_fla_delta_h.chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
FLA_FWD_O = _base(_fla_chunk_o.chunk_fwd_kernel_o)
FLA_BWD_DV_LOCAL = _base(_fla_chunk_o.chunk_bwd_kernel_dv_local)
FLA_BWD_DHU = _base(_fla_delta_h.chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64)
FLA_BWD_DQKWG = _base(_fla_chunk_o.chunk_bwd_kernel_dqkwg)
FLA_WY_BWD = _base(_fla_wy_fast.prepare_wy_repr_bwd_kernel)


@triton.jit
def kkt_solve(k, g, beta, T, A, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
              BT: tl.constexpr, BC: tl.constexpr, BK: tl.constexpr):
    FLA_KKT_SOLVE(k, g, beta, A, None, None, T, H, HV, K, BT, BC, BK, True, False)


@triton.jit
def recompute_w_u(k, v, beta, A, g, T, w, u, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                  V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    FLA_RECOMPUTE_W_U(k, v, beta, w, u, A, g, None, None, T, H, HV, K, V, BT, BK, BV, True, False)


@triton.jit
def fwd_h(k, u, w, g, T, v_new, h, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
          V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    # USE_G, USE_GK, USE_INITIAL_STATE, STORE_FINAL_STATE, SAVE_NEW_VALUE, STATE_V_FIRST, IS_VARLEN
    FLA_FWD_H(k, u, w, v_new, g, None, h, None, None, None, None, T, H, HV, K, V, BT, BV,
              True, False, False, False, True, False, False)


@triton.jit
def fwd_o(q, k, v, h, g, scale, T, o, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
          V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    # USE_G, USE_G_GAMMA, STATE_V_FIRST, IS_VARLEN
    FLA_FWD_O(q, k, v, h, g, None, o, None, None, scale, T, H, HV, K, V, BT, BK, BV,
              True, False, False, False)


@triton.jit
def bwd_dv_local(q, k, g, do, scale, T, dv, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                 V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    # USE_G, USE_G_GAMMA, USE_A, IS_VARLEN
    FLA_BWD_DV_LOCAL(q, k, g, None, None, do, dv, None, None, scale, T, H, HV, K, V, BT, BK, BV,
                     True, False, False, False)


@triton.jit
def bwd_dhu(q, k, w, g, do, dv, scale, T, dh, dv2, H: tl.constexpr, HV: tl.constexpr,
            K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr):
    # USE_G, USE_GK, USE_INITIAL_STATE, USE_FINAL_STATE_GRADIENT, STATE_V_FIRST, IS_VARLEN
    FLA_BWD_DHU(q, k, w, g, None, None, None, do, dh, dv, dv2, None, None, scale, T, H, HV, K, V, BT, BV,
                True, False, False, False, False, False)


@triton.jit
def bwd_dqkwg(q, k, v, g, h, do, dh, dv, scale, T, dq, dk, dw, dg, B: tl.constexpr, H: tl.constexpr,
              HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr,
              BV: tl.constexpr):
    # USE_G, USE_G_GAMMA, USE_DW, STATE_V_FIRST, IS_VARLEN
    FLA_BWD_DQKWG(q, k, v, g, None, h, do, dh, dq, dk, dw, dv, dg, None, None, scale, B, T, H, HV, K, V,
                  BT, BK, BV, True, False, True, False, False)


@triton.jit
def wy_bwd(k, v, beta, g, A, dw, du, T, dk, dv, db, dg, H: tl.constexpr, HV: tl.constexpr,
           K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr):
    FLA_WY_BWD(k, v, beta, g, A, dw, du, dk, dv, db, dg, None, None, T, H, HV, K, V, BT, BK, BV,
               True, False)

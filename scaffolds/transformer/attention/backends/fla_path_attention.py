"""PaTH attention on FLA's fused Triton kernels, as differentiable JAX operations.

Softmax launches flash-linear-attention's own kernels (``fla.ops.path_attn.parallel``,
``ParallelPATHAttentionFunction``) unchanged, in FLA's order and with its tiles, grids,
warps, stages, autotune menus, FP16 copies of the Householder vectors, and buffer
semantics; outputs FLA updates in place are aliased outputs here
(``fla_path_attention_kernels`` holds the wrappers). The unnormalized query polynomial
``o_t = sum_s p_t(s_ts) v_s`` uses the same launches with FLA's six softmax-dependent
kernels replaced by ``fla_path_polynomial_kernels``. The forget gate is not used.

Inputs follow ``path.path_attention``: ``q, k, v`` [B, T, H, D] (BF16, D in 16, 32, 64, 128 on
Hopper and up to 64 on Blackwell, value width equal to D), unit ``w`` [B, T, H, D] and ``beta`` [B, T, H] in FP32,
and FP32 polynomial coefficients [B, T, H, m+1] (m 1-4). T must be a multiple of 128. As in
``path_attention``, a row whose output has a nonfinite element becomes NaN; nonfinite
scores or polynomial values reach the output, while a -inf softmax score gives a zero
weight, as in FlashAttention. FLA's backward accumulates dk, dv, dw, and the split
products' gradient with FP32 atomic additions, so dk, dv, dw, and dbeta are not bitwise
repeatable; outputs, dq, and dc are.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
import math

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends import triton_port
from scaffolds.transformer.attention.backends.triton_port import empty, update, zeros

HEAD_DIMS = (16, 32, 64, 128)
# FLA's parallel_path_bwd_dq kernel (and the fork's) fails to launch at width 128 on
# Blackwell (B300: CUDA_ERROR_LAUNCH_FAILED); every width runs on Hopper.
BLACKWELL_HEAD_DIMS = (16, 32, 64)
SEQUENCE_MULTIPLE = 128
SPLIT = 512  # FLA's S: the large chunk whose Householder product the backward caches
MAX_DEGREE = 4
_F32, _F16 = jnp.float32, jnp.float16


@dataclass(frozen=True)
class Plan:
    """FLA's tile sizes and launch options for one device and head width."""

    BS: int  # UT-transform chunk and key block: 64 with Hopper-class shared memory, else 32
    BT: int  # query block: 128 with Ampere-class shared memory, else 64
    intra_fwd_warps: int
    inter_warps: int  # parallel forward, transform_q, dkv, dq
    cumprod_warps: int
    cumprod_fwd_stages: int
    dq_stages: int
    cumprod_bwd_stages: int
    intra_bwd_stages: int


def head_dims(device: triton_port.Device) -> tuple[int, ...]:
    """Head widths the fused kernels support on this device."""
    return BLACKWELL_HEAD_DIMS if device.blackwell else HEAD_DIMS


def plan(device: triton_port.Device, key_dim: int) -> Plan:
    """The branches of FLA's ``parallel_path_attn`` and its kernel launchers."""
    bs = device.tile({"hopper": 64}, 32)
    bt = device.tile({"ampere": 128}, 64)
    ampere = device.has_shared_memory("ampere")
    return Plan(
        BS=bs, BT=bt, intra_fwd_warps=4 if bs == 64 else 2,
        inter_warps=8 if (bt == 128 and key_dim == 128) else 4,
        cumprod_warps=8 if key_dim == 128 else 4,  # FLA: K = 128 with four warps is wrong
        cumprod_fwd_stages=3 if ampere else 1, dq_stages=3 if ampere else 2,
        cumprod_bwd_stages=2 if ampere else 1, intra_bwd_stages=3 if device.has_shared_memory("hopper") else 1)


@lru_cache(maxsize=None)
def _autotuned(device: triton_port.Device):
    """FLA's autotune menus for the UT transform (the same on every NVIDIA architecture)."""
    import triton

    from scaffolds.transformer.attention.backends import fla_path_attention_kernels as fla

    config = triton.Config
    kkt = triton_port.autotune(
        fla.kkt, [config({"BK": bk}, num_warps=w, num_stages=s) for bk in (32, 64, 128) for w in (2, 4, 8)
                  for s in (2, 3, 4)], key=["H", "HV", "K", "BT"])
    # solve_tril autotunes DOT_PRECISION over ["ieee"] unless FLA_USE_TMA enables TMA.
    if plan(device, 64).BS == 64:
        solve = triton_port.autotune(
            fla.solve_tril_64, [config({"DOT_PRECISION": "ieee"}, num_warps=w, num_stages=s) for w in (2, 4, 8)
                                for s in (2, 3, 4, 5)], key=["H", "BT"])
    else:
        solve = triton_port.autotune(
            fla.solve_tril_32, [config({"DOT_PRECISION": "ieee"}, num_warps=w, num_stages=s) for w in (1, 2, 4, 8)
                                for s in (2, 3, 4, 5)], key=["H", "BT"])
    return kkt, solve


# FLA's kernels take Triton's default FP32 dot precision on sm80+ (TF32).
_launch = partial(triton_port.launch, dot_precision="tf32")


def _shapes(q, v):
    batch, length, heads, key_dim = q.shape
    return batch, length, heads, key_dim, v.shape[-1]


def _ut_transform(w, beta, setup):
    """``A = (I + strictly_lower(diag(beta) W W^T))^(-1)`` per chunk (FLA's kkt + solve_tril)."""
    device, tiles = setup
    kkt, solve = _autotuned(device)
    batch, length, heads, key_dim = w.shape
    grid = (length // tiles.BS, batch * heads)
    a = _launch(kkt, w, beta, length, outputs=empty((batch, length, heads, tiles.BS), _F32), grid=grid,
                name="fla::path::kkt", H=heads, HV=heads, K=key_dim, BT=tiles.BS)
    return _launch(solve, a, length, outputs=zeros(a.shape, w.dtype), grid=grid, name="fla::path::solve_tril",
                   H=heads, BT=tiles.BS)


def _setup(q):
    device = triton_port.device()
    return device, plan(device, q.shape[-1])


def _meta(q, v):
    batch, length, heads, key_dim, value_dim = _shapes(q, v)
    return dict(G=1, HQ=heads, H=heads, K=key_dim, V=value_dim, BK=triton_port.next_power_of_2(key_dim),
                BV=triton_port.next_power_of_2(value_dim))


def _split_shapes(q, tiles):
    batch, length, heads, key_dim = q.shape
    splits = triton_port.cdiv(length, SPLIT)
    return splits, batch * splits, batch * triton_port.cdiv(length, tiles.BS)


def _transport_backward(q, k, v, w, beta, a, chunk_outputs, inter, intra, tiles):
    """FLA's backward after the chunk-local preparation, shared by both kernels.

    ``chunk_outputs`` are the preparation kernel's (h, q_new, k_new, dA_local, dv);
    ``inter`` and ``intra`` launch the kernel-specific dkv/dq and intra-split kernels.
    """
    from scaffolds.transformer.attention.backends import fla_path_attention_kernels as fla

    batch, length, heads, key_dim, value_dim = _shapes(q, v)
    h, q_new, k_new, d_a_local, dv = chunk_outputs
    blocks, splits_total, chunks_total = _split_shapes(q, tiles)
    meta = _meta(q, v)
    w16, h16 = w.astype(_F16), h.astype(_F16)
    k_large, hc_suffix, hc_whole = _launch(
        fla.cumprod_fwd, k_new, w16, h16,
        outputs=(empty(k_new.shape, k_new.dtype), empty((chunks_total, heads, key_dim, key_dim), _F16),
                 empty((splits_total, heads, key_dim, key_dim), _F16)),
        grid=(splits_total, heads), name="fla::path::cumprod_householder_fwd", BT=tiles.BS, K=key_dim, H=heads,
        BK=key_dim, T=length, S=SPLIT, num_warps=tiles.cumprod_warps, num_stages=tiles.cumprod_fwd_stages)
    q_large = _launch(
        fla.transform_q, q_new, w16, h16, length, outputs=zeros((batch, length, blocks, heads, key_dim), q_new.dtype),
        grid=(triton_port.cdiv(length, tiles.BT), batch * heads), name="fla::path::transform_q", S=SPLIT,
        G=1, HQ=heads, H=heads, K=key_dim, BT=tiles.BT, BS=tiles.BS, BK=meta["BK"], NUM_BLOCKS=blocks,
        num_warps=tiles.inter_warps)
    w_q, h_q, a_q = (x.astype(q.dtype) for x in (w, h, a))
    dk, dv, dq, dhc_whole, *extra = inter(q_large, k_large, hc_whole, dv, blocks, splits_total)
    dw1, dw2, dk = _launch(
        fla.cumprod_bwd, hc_suffix, dhc_whole, k_new, dk, w_q, h_q, length,
        outputs=tuple(empty(dk.shape, _F32) for _ in range(3)), grid=(splits_total, heads),
        name="fla::path::cumprod_householder_bwd", BT=tiles.BS, K=key_dim, BK=key_dim, S=SPLIT, G=1, H=heads,
        HQ=heads, num_warps=tiles.cumprod_warps, num_stages=tiles.cumprod_bwd_stages)
    dq, dk, dv, dw1, dw2, *more = intra(q_new, k_new, w_q, h_q, dq, dk, dv, dw1, dw2)
    dq, dk, dw, dbeta = _launch(
        fla.intra_bwd, q, k, w_q, h_q, beta, a_q, d_a_local, dq, dk, dw1, dw2, length,
        outputs=(empty(q.shape, q.dtype), empty(k.shape, k.dtype), empty(w.shape, k.dtype),
                 empty(beta.shape, k.dtype)),
        grid=(length // tiles.BS, batch * heads), name="fla::path::intra_chunk_preprocess_bwd", HQ=heads, G=1,
        H=heads, K=key_dim, BT=tiles.BS, BK=meta["BK"], num_stages=tiles.intra_bwd_stages)
    return (dq.astype(q.dtype), dk.astype(k.dtype), dv.astype(v.dtype), dw.astype(w.dtype),
            dbeta.astype(beta.dtype), *extra, *more)


# ---------------------------------------------------------------------------------------
# Softmax: FLA's kernels unchanged
# ---------------------------------------------------------------------------------------

def _softmax_forward(q, k, v, w, beta):
    from scaffolds.transformer.attention.backends import fla_path_attention_kernels as fla

    setup = _setup(q)
    _, tiles = setup
    batch, length, heads, key_dim, value_dim = _shapes(q, v)
    scale = 1.0 / math.sqrt(key_dim)
    meta = _meta(q, v)
    a = _ut_transform(w, beta, setup)
    o, lse, maximum, w2, q_new, k_new = _launch(
        fla.intra_fwd, q, k, v, w, beta, a, scale, length,
        outputs=(empty(v.shape, _F32), empty(beta.shape, _F32), empty(beta.shape, _F32), empty(w.shape, w.dtype),
                 empty(q.shape, _F32), empty(k.shape, k.dtype)),
        grid=(length // tiles.BS, batch * heads), name="fla::path::intra_chunk_preprocess_fwd",
        BT=tiles.BS, num_warps=tiles.intra_fwd_warps, **{key: meta[key] for key in ("H", "G", "HQ", "K", "V", "BK", "BV")})
    o, lse = _launch(
        fla.path_fwd, q_new, k_new, v, o, w.astype(_F16), w2.astype(_F16), lse, maximum, scale, length,
        outputs=(empty(v.shape, v.dtype), empty(beta.shape, _F32)),
        grid=(triton_port.cdiv(length, tiles.BT), batch * heads), name="fla::path::parallel_path_fwd",
        BT=tiles.BT, BS=tiles.BS, num_warps=tiles.inter_warps, **meta)
    return o, (q, k, v, w, beta, o, lse, a)


def _softmax_backward(residuals, do):
    from scaffolds.transformer.attention.backends import fla_path_attention_kernels as fla

    q, k, v, w, beta, o, lse, a = residuals
    _, tiles = _setup(q)
    batch, length, heads, key_dim, value_dim = _shapes(q, v)
    scale = 1.0 / math.sqrt(key_dim)
    meta = _meta(q, v)
    do = do.astype(v.dtype)
    delta = _launch(fla.bwd_preprocess, o, do, outputs=empty(beta.shape, _F32), grid=(batch * length * heads,),
                    name="fla::attn::bwd_preprocess", B=meta["BV"], V=value_dim)
    prepared = _launch(
        fla.bwd_prepare, q, k, v, w, beta, lse, delta, a, do, scale, length,
        outputs=(empty(w.shape, w.dtype), empty(q.shape, q.dtype), empty(k.shape, k.dtype),
                 empty((batch, length, heads, tiles.BS), q.dtype), empty(v.shape, _F32)),
        grid=(length // tiles.BS, batch * heads), name="fla::path::intra_chunk_preprocess_bwd_prepare",
        BT=tiles.BS, **meta)
    inter_meta = dict(BT=tiles.BT, BS=tiles.BS, S=SPLIT, **meta)
    grid = (triton_port.cdiv(length, tiles.BT), batch * heads)

    def inter(q_large, k_large, hc_whole, dv, blocks, splits_total):
        dk, dv = _launch(fla.bwd_dkv, q_large, k_large, v, hc_whole, lse, delta, do, scale, length,
                         outputs=(empty(q.shape, _F32), update(dv)), grid=grid, name="fla::path::parallel_path_bwd_dkv",
                         NUM_BLOCKS=blocks, num_warps=tiles.inter_warps, **inter_meta)
        dq, dhc_whole = _launch(fla.bwd_dq, q_large, k_large, v, hc_whole, lse, delta, do, scale, length,
                                outputs=(empty(q.shape, _F32), zeros((splits_total, heads, key_dim, key_dim), _F32)),
                                grid=grid, name="fla::path::parallel_path_bwd_dq", NUM_BLOCKS=blocks,
                                num_warps=tiles.inter_warps, num_stages=tiles.dq_stages, **inter_meta)
        return dk, dv, dq, dhc_whole

    def intra(q_new, k_new, w_q, h_q, dq, dk, dv, dw1, dw2):
        return _launch(fla.bwd_intra, q_new, k_new, v, w_q, h_q, lse, delta, dq, do, length, scale,
                       outputs=(empty(q.shape, q.dtype), update(dk), update(dv), update(dw1), update(dw2)),
                       grid=(length // tiles.BS, batch * heads), name="fla::path::parallel_path_bwd_intra",
                       BT=tiles.BS, S=SPLIT, **meta)

    return _transport_backward(q, k, v, w, beta, a, prepared, inter, intra, tiles)


@jax.custom_vjp
def _softmax(q, k, v, w, beta):
    return _softmax_forward(q, k, v, w, beta)[0]


_softmax.defvjp(_softmax_forward, _softmax_backward)


# ---------------------------------------------------------------------------------------
# Unnormalized query polynomial: the six forked kernels
# ---------------------------------------------------------------------------------------

def _polynomial_forward(q, k, v, w, beta, c):
    from scaffolds.transformer.attention.backends import fla_path_polynomial_kernels as poly

    setup = _setup(q)
    _, tiles = setup
    batch, length, heads, key_dim, value_dim = _shapes(q, v)
    scale = 1.0 / math.sqrt(key_dim)
    meta = _meta(q, v)
    degree = c.shape[-1] - 1
    a = _ut_transform(w, beta, setup)
    o, w2, q_new, k_new = _launch(
        poly.intra_fwd, q, k, v, w, beta, c, a, scale, length,
        outputs=(empty(v.shape, _F32), empty(w.shape, w.dtype), empty(q.shape, _F32), empty(k.shape, k.dtype)),
        grid=(length // tiles.BS, batch * heads), name="fla::path::polynomial_intra_chunk_fwd",
        BT=tiles.BS, DEGREE=degree, num_warps=tiles.intra_fwd_warps,
        **{key: meta[key] for key in ("H", "G", "HQ", "K", "V", "BK", "BV")})
    o = _launch(
        poly.path_fwd, q_new, k_new, v, o, w.astype(_F16), w2.astype(_F16), c, scale, length,
        outputs=empty(v.shape, v.dtype), grid=(triton_port.cdiv(length, tiles.BT), batch * heads),
        name="fla::path::polynomial_parallel_fwd", BT=tiles.BT, BS=tiles.BS, DEGREE=degree,
        num_warps=tiles.inter_warps, **meta)
    return o, (q, k, v, w, beta, c, a)


def _polynomial_backward(residuals, do):
    from scaffolds.transformer.attention.backends import fla_path_polynomial_kernels as poly

    q, k, v, w, beta, c, a = residuals
    _, tiles = _setup(q)
    batch, length, heads, key_dim, value_dim = _shapes(q, v)
    scale = 1.0 / math.sqrt(key_dim)
    meta = _meta(q, v)
    degree = c.shape[-1] - 1
    do = do.astype(v.dtype)
    *prepared, dc_local = _launch(
        poly.bwd_prepare, q, k, v, w, beta, c, a, do, scale, length,
        outputs=(empty(w.shape, w.dtype), empty(q.shape, q.dtype), empty(k.shape, k.dtype),
                 empty((batch, length, heads, tiles.BS), q.dtype), empty(v.shape, _F32), empty(c.shape, _F32)),
        grid=(length // tiles.BS, batch * heads), name="fla::path::polynomial_bwd_prepare",
        BT=tiles.BS, DEGREE=degree, **meta)
    inter_meta = dict(BT=tiles.BT, BS=tiles.BS, S=SPLIT, DEGREE=degree, **meta)
    grid = (triton_port.cdiv(length, tiles.BT), batch * heads)

    def inter(q_large, k_large, hc_whole, dv, blocks, splits_total):
        dk, dv = _launch(poly.bwd_dkv, q_large, k_large, v, c, do, scale, length,
                         outputs=(empty(q.shape, _F32), update(dv)), grid=grid,
                         name="fla::path::polynomial_bwd_dkv", NUM_BLOCKS=blocks, num_warps=tiles.inter_warps,
                         **inter_meta)
        dq, dhc_whole, dc_splits = _launch(
            poly.bwd_dq, q_large, k_large, v, hc_whole, c, do, scale, length,
            outputs=(empty(q.shape, _F32), zeros((splits_total, heads, key_dim, key_dim), _F32), empty(c.shape, _F32)),
            grid=grid, name="fla::path::polynomial_bwd_dq", NUM_BLOCKS=blocks, num_warps=tiles.inter_warps,
            num_stages=tiles.dq_stages, **inter_meta)
        return dk, dv, dq, dhc_whole, dc_splits

    def intra(q_new, k_new, w_q, h_q, dq, dk, dv, dw1, dw2):
        return _launch(poly.bwd_intra, q_new, k_new, v, w_q, h_q, c, dq, do, length, scale,
                       outputs=(empty(q.shape, q.dtype), update(dk), update(dv), update(dw1), update(dw2),
                                empty(c.shape, _F32)),
                       grid=(length // tiles.BS, batch * heads), name="fla::path::polynomial_bwd_intra",
                       BT=tiles.BS, S=SPLIT, DEGREE=degree, **meta)

    dq, dk, dv, dw, dbeta, dc_splits, dc_chunks = _transport_backward(
        q, k, v, w, beta, a, prepared, inter, intra, tiles)
    return dq, dk, dv, dw, dbeta, (dc_local + dc_chunks + dc_splits).astype(c.dtype)


@jax.custom_vjp
def _polynomial(q, k, v, w, beta, c):
    return _polynomial_forward(q, k, v, w, beta, c)[0]


_polynomial.defvjp(_polynomial_forward, _polynomial_backward)


def validate(q, k, v, w, beta, *, kernel, coefficients=None):
    """Raise ValueError unless the fused kernels support these operands."""
    if kernel not in ("softmax", "query_polynomial"):
        raise ValueError(f"Unknown PaTH kernel {kernel!r}")
    if q.ndim != 4 or q.shape != k.shape or q.shape != w.shape or v.shape[:3] != q.shape[:3]:
        raise ValueError("PaTH requires q, k, w [B, T, H, D] and v [B, T, H, V]")
    if beta.shape != q.shape[:3]:
        raise ValueError("PaTH beta must have shape [B, T, H]")
    on_gpu = triton_port.interpreting() or jax.default_backend() == "gpu"
    widths = head_dims(triton_port.device()) if on_gpu else HEAD_DIMS
    if q.shape[-1] not in widths or v.shape[-1] != q.shape[-1]:
        raise ValueError(f"Fused PaTH needs equal query/key and value widths in {widths} on this GPU")
    if q.shape[1] % SEQUENCE_MULTIPLE:
        raise ValueError(f"Fused PaTH needs a sequence length divisible by {SEQUENCE_MULTIPLE}")
    if (kernel == "query_polynomial") != (coefficients is not None):
        raise ValueError("The polynomial PaTH kernel requires coefficients; softmax takes none")
    if coefficients is not None and (coefficients.shape[:3] != q.shape[:3]
                                     or not 2 <= coefficients.shape[-1] <= MAX_DEGREE + 1):
        raise ValueError(f"PaTH polynomial coefficients must have shape [B, T, H, m+1] with m in 1-{MAX_DEGREE}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Fused PaTH needs q, k, v of one dtype")
    if q.dtype != (jnp.float32 if triton_port.interpreting() else jnp.bfloat16):
        # With FP32 tiles one of FLA's kernels requests 240 KB of shared memory (H100: 227 KB);
        # Triton's interpreter, conversely, cannot compute in BF16.
        raise ValueError("Fused PaTH needs BF16 q, k, v on a GPU (FP32 in Triton's interpreter)")


def fused_path_attention(q, k, v, w, beta, *, kernel, coefficients=None):
    """Causal PaTH attention (``path.path_attention``'s contract) on the fused kernels."""
    validate(q, k, v, w, beta, kernel=kernel, coefficients=coefficients)
    w, beta = w.astype(_F32), beta.astype(_F32)
    if kernel == "softmax":
        output = _softmax(q, k, v, w, beta)
    else:
        output = _polynomial(q, k, v, w, beta, coefficients.astype(_F32))
    return jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)

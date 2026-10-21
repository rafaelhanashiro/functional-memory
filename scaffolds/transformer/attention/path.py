"""PaTH position encoding: attention scores transported through Householder products.

Following Yang et al., "PaTH Attention: Position Encoding via Accumulating Householder
Transformations" (arXiv:2505.16381), each token t owns a generalized Householder matrix
``H_t = I - beta_t w_t w_t^T`` (``w_t`` unit length, ``beta_t`` in (0, 2)), and the score
between query t and key s <= t is

    S_ts = q_t^T H_t H_(t-1) ... H_(s+1) k_s,

which replaces RoPE. The token-dependent ``w_t`` and ``beta_t`` come from the layer input
(a rank-32 projection, a width-3 causal SiLU short convolution, and L2 normalization per
head; ``beta_t = 2 sigmoid(x_t W_beta + b_beta)``), as in flash-linear-attention's
``PaTHAttention`` layer by the paper's authors.

The implementation is the authors' chunked algorithm in JAX (C = 128 by default). Within each chunk of C
tokens the UT transform ``T = (I + strictly_lower(diag(beta) W W^T))^(-1)`` gives the
chunk-local scores, the queries transported to the chunk start, the keys transported to
the chunk end, and the chunk's full product ``P = I - W^T T diag(beta) W``. A sweep over
the chunk distance d = 1, ..., N-1 then scores every query chunk against the key chunk d
behind it and moves the queries through that key chunk's product. Each step is one
batched matrix product over all chunks, and the derivative comes from automatic
differentiation of a rematerialized forward pass, so nothing pairwise is stored across
steps. Transport, scores, kernels, and reductions are FP32; the inputs may be BF16.

Two kernels are applied to the transported scores ``s_ts = S_ts / sqrt(d_K)``: the
normalized exponential (softmax, with a running maximum and normalizer across the sweep),
and the unnormalized polynomial ``sum_r c_r(x_t) s_ts^r`` with per-query coefficients, as
in ``input_dependent.query_polynomial_attention``.
"""

from functools import partial
import math

import jax
import jax.numpy as jnp

CHUNK_SIZE = 128  # fastest on B300 among 64, 128 and 256 for T = 2048, d_K = 64
KERNELS = ("softmax", "query_polynomial")
_HIGHEST = jax.lax.Precision.HIGHEST


def _chunks(x, chunk):
    """[B, T, H, ...] -> [B, H, N, C, ...]."""
    batch, time, heads = x.shape[:3]
    x = jnp.moveaxis(x, 2, 1)
    return x.reshape(batch, heads, time // chunk, chunk, *x.shape[3:])


def _unchunk(x):
    """[B, H, N, C, ...] -> [B, T, H, ...]."""
    batch, heads, count, chunk = x.shape[:4]
    return jnp.moveaxis(x.reshape(batch, heads, count * chunk, *x.shape[4:]), 1, 2)


def _matmul(a, b, precision):
    return jnp.matmul(a, b, precision=precision, preferred_element_type=jnp.float32)


def chunk_transforms(q, k, w, beta, *, precision=None):
    """Per-chunk UT transform. Inputs are chunked FP32 [B, H, N, C, D] and beta [B, H, N, C].

    Returns the chunk-local unscaled scores [.., C, C] (lower triangular), the queries
    transported to their chunk start, the keys transported to their chunk end, and the
    chunk's product acting on row vectors as ``q <- q - q @ hmat`` ([.., D, D]).
    """
    chunk = q.shape[-2]
    lower = jnp.tril(jnp.ones((chunk, chunk), bool), -1)
    lower_inclusive = jnp.tril(jnp.ones((chunk, chunk), bool))
    w_beta = w * beta[..., None]
    gram = _matmul(w_beta, jnp.swapaxes(w, -1, -2), precision)
    unit_lower = jnp.where(lower, gram, 0.0) + jnp.eye(chunk, dtype=jnp.float32)
    identity = jnp.broadcast_to(jnp.eye(chunk, dtype=jnp.float32), unit_lower.shape)
    transform = jax.scipy.linalg.solve_triangular(unit_lower, identity, lower=True, unit_diagonal=True)
    y = _matmul(transform, w_beta, precision)  # T diag(beta) W
    wk = jnp.where(lower, _matmul(w_beta, jnp.swapaxes(k, -1, -2), precision), 0.0)
    t_wk = _matmul(transform, wk, precision)
    qw = jnp.where(lower_inclusive, _matmul(q, jnp.swapaxes(w, -1, -2), precision), 0.0)
    local = jnp.where(lower_inclusive, _matmul(q, jnp.swapaxes(k, -1, -2), precision), 0.0) - _matmul(qw, t_wk, precision)
    q_start = q - _matmul(qw, y, precision)
    k_end = k - _matmul(jnp.swapaxes(t_wk, -1, -2), w, precision)
    hmat = _matmul(jnp.swapaxes(w, -1, -2), y, precision)
    return local, q_start, k_end, hmat


def _polynomial(scores, coefficients):
    """Horner evaluation; scores [.., Cq, Ck], coefficients [.., Cq, R]."""
    value = jnp.broadcast_to(coefficients[..., -1, None], scores.shape)
    for order in range(coefficients.shape[-1] - 2, -1, -1):
        value = value * scores + coefficients[..., order, None]
    return value


@partial(jax.checkpoint, static_argnums=(6, 7, 8))
def _path_core(q, k, v, w, beta, coefficients, kernel, chunk, precision):
    batch, time, heads, qk_dim = q.shape
    scale = 1.0 / math.sqrt(qk_dim)
    qc, kc, wc = (_chunks(x.astype(jnp.float32), chunk) for x in (q, k, w))
    vc = _chunks(v.astype(jnp.float32), chunk)
    bc = _chunks(beta.astype(jnp.float32), chunk)
    count = qc.shape[2]
    local, q_state, k_end, hmat = chunk_transforms(qc, kc, wc, bc, precision=precision)
    causal = jnp.tril(jnp.ones((chunk, chunk), bool))
    polynomial = kernel == "query_polynomial"
    if polynomial:
        cc = _chunks(coefficients.astype(jnp.float32), chunk)

    def weights(scores, chunk_slice, mask=None):
        scores = scores * scale
        if polynomial:
            values = _polynomial(scores, cc[:, :, chunk_slice])
            return values if mask is None else jnp.where(mask, values, 0.0), scores
        return scores if mask is None else jnp.where(mask, scores, -jnp.inf), scores

    # The diagonal blocks: chunk-local scores.
    local_values, local_scores = weights(local, slice(None), causal)
    invalid = jnp.any(causal & ~jnp.isfinite(local_scores), axis=-1)
    if polynomial:
        invalid = invalid | jnp.any(causal & ~jnp.isfinite(local_values), axis=-1)
        output = _matmul(local_values, vc, precision)
        # Observed, never divided by: the kernel's extrema and row sums.
        kernel_min = jnp.min(jnp.where(causal, local_values, jnp.inf))
        kernel_max = jnp.max(jnp.where(causal, local_values, -jnp.inf))
        row_sum = jnp.sum(local_values, axis=-1)
    else:
        maximum = jnp.max(local_values, axis=-1)
        probabilities = jnp.exp(local_values - maximum[..., None])
        normalizer = jnp.sum(probabilities, axis=-1)
        output = _matmul(probabilities, vc, precision)
        row_min = jnp.min(jnp.where(causal, local_scores, jnp.inf), axis=-1)
    score_min = jnp.min(jnp.where(causal, local_scores, jnp.inf))
    score_max = jnp.max(jnp.where(causal, local_scores, -jnp.inf))
    # Sweep over chunk distance d: query chunks d.. score key chunks ..N-1-d, then move
    # through those key chunks' products. Static slices keep only the valid pairs.
    for distance in range(1, count):
        queries = q_state[:, :, distance:]
        raw = _matmul(queries, jnp.swapaxes(k_end[:, :, :count - distance], -1, -2), precision)
        values, scores = weights(raw, slice(distance, None))
        invalid = invalid.at[:, :, distance:].set(invalid[:, :, distance:] | jnp.any(~jnp.isfinite(scores), axis=-1))
        score_min = jnp.minimum(score_min, jnp.min(scores))
        score_max = jnp.maximum(score_max, jnp.max(scores))
        block_values = vc[:, :, :count - distance]
        if polynomial:
            invalid = invalid.at[:, :, distance:].set(invalid[:, :, distance:] | jnp.any(~jnp.isfinite(values), axis=-1))
            output = output.at[:, :, distance:].add(_matmul(values, block_values, precision))
            kernel_min = jnp.minimum(kernel_min, jnp.min(values))
            kernel_max = jnp.maximum(kernel_max, jnp.max(values))
            row_sum = row_sum.at[:, :, distance:].add(jnp.sum(values, axis=-1))
        else:
            row_min = row_min.at[:, :, distance:].set(jnp.minimum(row_min[:, :, distance:], jnp.min(scores, axis=-1)))
            previous = maximum[:, :, distance:]
            updated = jnp.maximum(previous, jnp.max(values, axis=-1))
            rescale = jnp.exp(previous - updated)
            probabilities = jnp.exp(values - updated[..., None])
            normalizer = normalizer.at[:, :, distance:].set(
                normalizer[:, :, distance:] * rescale + jnp.sum(probabilities, axis=-1))
            output = output.at[:, :, distance:].set(
                output[:, :, distance:] * rescale[..., None] + _matmul(probabilities, block_values, precision))
            maximum = maximum.at[:, :, distance:].set(updated)
        if distance < count - 1:
            moved = queries - _matmul(queries, hmat[:, :, :count - distance], precision)
            q_state = q_state.at[:, :, distance:].set(moved)
    if not polynomial:
        output = output / normalizer[..., None]
    output = jnp.where(invalid[..., None], jnp.nan, output)
    output = jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)
    diagnostics = {
        "score_min": score_min, "score_max": score_max,
        "invalid_row_count": jnp.sum(invalid, dtype=jnp.int32),
        "invalid_output_count": jnp.sum(~jnp.isfinite(output), dtype=jnp.int32),
    }
    if polynomial:
        diagnostics.update(kernel_min=kernel_min, kernel_max=kernel_max,
                           weight_sum_min=jnp.min(row_sum), weight_sum_max=jnp.max(row_sum))
    else:
        # Kernel values exp(s - max_row) lie in (0, 1]; each normalizer is at least one.
        diagnostics.update(kernel_min=jnp.min(jnp.exp(row_min - maximum)), kernel_max=jnp.max(jnp.exp(maximum - maximum)),
                           normalizer_min=jnp.min(normalizer), normalizer_max=jnp.max(normalizer))
    return _unchunk(output).astype(v.dtype), jax.lax.stop_gradient(diagnostics)


def path_attention(q, k, v, w, beta, *, kernel, coefficients=None, chunk_size=CHUNK_SIZE,
                   precision=None, return_diagnostics=False):
    """Causal attention on PaTH-transported scores.

    ``q, k`` [B, T, H, D] (after any QK normalization, before scaling), ``v`` [B, T, H, V],
    ``w`` [B, T, H, D] unit vectors, ``beta`` [B, T, H] in (0, 2), and for the polynomial
    kernel FP32 ``coefficients`` [B, T, H, m+1]. ``T`` must be a multiple of
    ``chunk_size``. A nonfinite score, kernel value, or output poisons its row with NaN.
    """
    if kernel not in KERNELS:
        raise ValueError(f"Unknown PaTH kernel {kernel!r}; expected one of {KERNELS}")
    if q.ndim != 4 or q.shape != k.shape or q.shape != w.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError("PaTH requires q, k, w [B, T, H, D] and v [B, T, H, V]")
    if beta.shape != q.shape[:3]:
        raise ValueError("PaTH beta must have shape [B, T, H]")
    if q.shape[1] % chunk_size:
        raise ValueError(f"PaTH sequence length {q.shape[1]} must be a multiple of the chunk size {chunk_size}")
    if (kernel == "query_polynomial") != (coefficients is not None):
        raise ValueError("The polynomial PaTH kernel requires coefficients; softmax takes none")
    if coefficients is not None and (coefficients.shape[:3] != q.shape[:3] or coefficients.shape[-1] < 2):
        raise ValueError("PaTH polynomial coefficients must have shape [B, T, H, m+1] with m >= 1")
    output, diagnostics = _path_core(q, k, v, w, beta, coefficients, kernel, chunk_size, precision)
    return (output, diagnostics) if return_diagnostics else output


PATH_ARGUMENTS = ("path_w_down", "path_w_up", "path_w_conv", "path_beta_kernel", "path_beta_bias")


def parameter_specs(*, hidden_size: int, qk_head_dim: int, rank: int = 32, conv_size: int = 3):
    """The layer's Householder-vector and strength parameters, drawn from the mechanism stream.

    Weights follow the authors' model initialization: projections N(0, 0.02^2), the depthwise
    convolution with PyTorch Conv1d's default variance 1/(3 * conv_size), and a zero bias,
    so beta starts near one. The rank projection is shared by a layer's heads.
    """
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    def spec(name, shape, sharing, std, decay):
        size = math.prod(shape)
        return AttentionParameterSpec(name=name, argument=name, initial=(0.0,) * size, initial_shape=shape,
                                      sharing=sharing, weight_decay=decay, initial_std=std)

    return (
        spec("path_w_down", (hidden_size, rank), "per_layer", 0.02, True),
        spec("path_w_up", (rank, qk_head_dim), "per_head", 0.02, True),
        spec("path_w_conv", (conv_size, qk_head_dim), "per_head", 1.0 / math.sqrt(3 * conv_size), True),
        spec("path_beta_kernel", (hidden_size,), "per_head", 0.02, True),
        spec("path_beta_bias", (1,), "per_head", 0.0, False),
    )


def path_scores_reference(q, k, w, beta):
    """Every transported score by the definition, in NumPy FP64, for tests: [B, H, T, T]."""
    import numpy as np

    q, k, w, beta = (np.asarray(x, np.float64) for x in (q, k, w, beta))
    batch, time, heads, dim = q.shape
    scores = np.full((batch, heads, time, time), np.nan)
    for b in range(batch):
        for h in range(heads):
            for t in range(time):
                transport = np.eye(dim)
                for s in range(t, -1, -1):
                    scores[b, h, t, s] = q[b, t, h] @ transport @ k[b, s, h]
                    transport = transport @ (np.eye(dim) - beta[b, s, h] * np.outer(w[b, s, h], w[b, s, h]))
    return scores


def path_gates(x, w_down, w_up, w_conv, beta_kernel, beta_bias, *, num_heads, eps=1e-6):
    """Token Householder vectors and strengths for one layer.

    ``x`` [B, T, hidden] (the attention input), ``w_down`` [hidden, rank], ``w_up``
    [heads, rank, D], ``w_conv`` [heads, width, D] (causal depthwise, then SiLU),
    ``beta_kernel`` [heads, hidden] and ``beta_bias`` [heads, 1]. Returns FP32 unit ``w``
    [B, T, H, D] and ``beta`` [B, T, H] in (0, 2).
    """
    x32 = x.astype(jnp.float32)
    low = jnp.einsum("btd,dr->btr", x32, w_down.astype(jnp.float32), precision=_HIGHEST)
    w = jnp.einsum("btr,hre->bthe", low, w_up.astype(jnp.float32), precision=_HIGHEST)
    width = w_conv.shape[1]
    padded = jnp.pad(w, ((0, 0), (width - 1, 0), (0, 0), (0, 0)))
    time = x.shape[1]
    w = sum(padded[:, i:i + time] * w_conv[:, i].astype(jnp.float32) for i in range(width))
    w = jax.nn.silu(w)
    w = w * jax.lax.rsqrt(jnp.sum(w * w, axis=-1, keepdims=True) + eps)
    logits = jnp.einsum("btd,hd->bth", x32, beta_kernel.astype(jnp.float32), precision=_HIGHEST)
    beta = 2.0 * jax.nn.sigmoid(logits + beta_bias.astype(jnp.float32).reshape(-1))
    if w.shape[2] != num_heads:
        raise ValueError("PaTH gate parameters must match the number of heads")
    return w, beta

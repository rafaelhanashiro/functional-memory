"""Exact degree-two/three/four GDN dual kernels with original-key-space geometry.

For each query tile, keys are visited backwards. Its transformed query Q and
effective scores E obey E=(Q K^T) A and Q'=Q-E diag(erase) K, where A is the
chunk-local unit-lower triangular inverse. Polynomial weights and decay are
applied to E only when reading values. The backward recomputes query histories,
then their adjoints, and reduces each key chunk in its own program: no atomics,
feature-space quadratic states, or persistent sequence-square activations.
"""

import triton
import triton.language as tl


@triton.jit
def _matrix(ptr, b, h, chunk, T: tl.constexpr, H: tl.constexpr,
            D: tl.constexpr, BD: tl.constexpr, BT: tl.constexpr):
    rows = chunk * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    return tl.load(ptr + ((b * T + rows[:, None]) * H + h) * D + cols[None, :],
                   mask=cols[None, :] < D, other=0).to(tl.float32)


@triton.jit
def _gates(ptr, b, h, chunk, T: tl.constexpr, H: tl.constexpr, BT: tl.constexpr):
    rows = chunk * BT + tl.arange(0, BT)
    return tl.load(ptr + (b * T + rows) * H + h).to(tl.float32)


@triton.jit
def _coefficients(ptr, b, h, chunk, T: tl.constexpr, H: tl.constexpr, BT: tl.constexpr,
                  NBETA: tl.constexpr):
    rows = chunk * BT + tl.arange(0, BT)
    offsets = ((b * T + rows) * H + h) * NBETA
    if NBETA >= 4:
        beta3 = tl.load(ptr + offsets + 3).to(tl.float32)
    else:
        beta3 = tl.zeros((BT,), tl.float32)
    if NBETA >= 5:
        beta4 = tl.load(ptr + offsets + 4).to(tl.float32)
    else:
        beta4 = tl.zeros((BT,), tl.float32)
    return (tl.load(ptr + offsets).to(tl.float32),
            tl.load(ptr + offsets + 1).to(tl.float32),
            tl.load(ptr + offsets + 2).to(tl.float32), beta3, beta4)


@triton.jit
def _polynomial(effective, beta0, beta1, beta2, beta3, beta4, NBETA: tl.constexpr):
    polynomial = (beta2[None, :] * effective + beta1[None, :]) * effective + beta0[None, :]
    if NBETA >= 4:
        polynomial += beta3[None, :] * effective * effective * effective
    if NBETA >= 5:
        polynomial += beta4[None, :] * effective * effective * effective * effective
    return polynomial


@triton.jit
def _derivative(effective, beta1, beta2, beta3, beta4, NBETA: tl.constexpr):
    derivative = beta1[None, :] + 2.0 * beta2[None, :] * effective
    if NBETA >= 4:
        derivative += 3.0 * beta3[None, :] * effective * effective
    if NBETA >= 5:
        derivative += 4.0 * beta4[None, :] * effective * effective * effective
    return derivative


@triton.jit
def _history(ptr, bh, qc, kc, T: tl.constexpr, K: tl.constexpr,
             BK: tl.constexpr, BT: tl.constexpr):
    chunks = T // BT
    pairs = chunks * (chunks + 1) // 2
    pair = qc * (qc + 1) // 2 + kc
    rows, cols = tl.arange(0, BT), tl.arange(0, BK)
    offsets = ((bh * pairs + pair) * BT + rows[:, None]) * K + cols[None, :]
    return tl.load(ptr + offsets, mask=cols[None, :] < K, other=0).to(tl.float32)


@triton.jit
def _store_history(ptr, value, bh, qc, kc, T: tl.constexpr, K: tl.constexpr,
                   BK: tl.constexpr, BT: tl.constexpr):
    chunks = T // BT
    pairs = chunks * (chunks + 1) // 2
    pair = qc * (qc + 1) // 2 + kc
    rows, cols = tl.arange(0, BT), tl.arange(0, BK)
    offsets = ((bh * pairs + pair) * BT + rows[:, None]) * K + cols[None, :]
    tl.store(ptr + offsets, value, mask=cols[None, :] < K)


@triton.jit
def _history_rows(ptr, bh, qb, kc, T: tl.constexpr, K: tl.constexpr,
                  BK: tl.constexpr, BT: tl.constexpr, BQ: tl.constexpr):
    chunks = T // BT
    pairs = chunks * (chunks + 1) // 2
    qc = qb // (BT // BQ)
    pair = qc * (qc + 1) // 2 + kc
    rows = (qb % (BT // BQ)) * BQ + tl.arange(0, BQ)
    cols = tl.arange(0, BK)
    offsets = ((bh * pairs + pair) * BT + rows[:, None]) * K + cols[None, :]
    return tl.load(ptr + offsets, mask=cols[None, :] < K, other=0).to(tl.float32)


@triton.jit
def _causal(qc, kc, BT: tl.constexpr):
    rows = tl.arange(0, BT)
    return qc * BT + rows[:, None] >= kc * BT + rows[None, :]


@triton.jit
def forward(q, k, v, prefix, beta, erase, inverse, transformed, scale, output, history,
            T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
            BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
            SAVE_HISTORY: tl.constexpr, COMPUTE_OUTPUT: tl.constexpr, NBETA: tl.constexpr = 3):
    qc, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    query = _matrix(q, b, h, qc, T, H, K, BK, BT)
    if COMPUTE_OUTPUT:
        query_decay = _gates(prefix, b, h, qc, T, H, BT)
        read = tl.zeros((BT, BV), tl.float32)
    for offset in range(qc + 1):
        kc = qc - offset
        if SAVE_HISTORY:
            _store_history(history, query, bh, qc, kc, T, K, BK, BT)
        key = _matrix(k, b, h, kc, T, H, K, BK, BT)
        erasure = _gates(erase, b, h, kc, T, H, BT)
        causal = _causal(qc, kc, BT)
        if kc == qc:
            a = _matrix(inverse, b, h, kc, T, H, BT, BT, BT)
            scores = tl.where(causal, tl.dot(query, tl.trans(key), input_precision="tf32x3"), 0.0)
            effective = tl.dot(scores, a, input_precision="tf32x3")
        else:
            r = _matrix(transformed, b, h, kc, T, H, K, BK, BT)
            effective = tl.dot(query, tl.trans(r), input_precision="tf32x3")
        if COMPUTE_OUTPUT:
            value = _matrix(v, b, h, kc, T, H, V, BV, BT)
            key_decay = _gates(prefix, b, h, kc, T, H, BT)
            decay = tl.exp(tl.where(causal, query_decay[:, None] - key_decay[None, :], 0.0))
            beta0, beta1, beta2, beta3, beta4 = _coefficients(beta, b, h, kc, T, H, BT, NBETA)
            polynomial = _polynomial(effective, beta0, beta1, beta2, beta3, beta4, NBETA)
            weights = tl.where(causal, decay * polynomial, 0.0)
            read += tl.dot(weights, value, input_precision="tf32x3")
        query -= tl.dot(effective * erasure[None, :], key, input_precision="tf32x3")
    if COMPUTE_OUTPUT:
        rows, cols = qc * BT + tl.arange(0, BT), tl.arange(0, BV)
        offsets = ((b * T + rows[:, None]) * H + h) * V + cols[None, :]
        tl.store(output + offsets, read * scale, mask=cols[None, :] < V)


@triton.jit
def backward_queries(k, v, prefix, beta, erase, inverse, transformed, dout, history, scale,
                     dq, dquery_prefix, adjoint_history,
                     T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                     BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, NBETA: tl.constexpr = 3):
    qc, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    do = _matrix(dout, b, h, qc, T, H, V, BV, BT) * scale
    query_decay = _gates(prefix, b, h, qc, T, H, BT)
    query_adjoint = tl.zeros((BT, BK), tl.float32)
    grad_prefix = tl.zeros((BT,), tl.float32)
    for kc in range(qc + 1):
        _store_history(adjoint_history, query_adjoint, bh, qc, kc, T, K, BK, BT)
        query = _history(history, bh, qc, kc, T, K, BK, BT)
        key = _matrix(k, b, h, kc, T, H, K, BK, BT)
        value = _matrix(v, b, h, kc, T, H, V, BV, BT)
        erasure = _gates(erase, b, h, kc, T, H, BT)
        causal = _causal(qc, kc, BT)
        if kc == qc:
            a = _matrix(inverse, b, h, kc, T, H, BT, BT, BT)
            scores = tl.where(causal, tl.dot(query, tl.trans(key), input_precision="tf32x3"), 0.0)
            effective = tl.dot(scores, a, input_precision="tf32x3")
        else:
            r = _matrix(transformed, b, h, kc, T, H, K, BK, BT)
            effective = tl.dot(query, tl.trans(r), input_precision="tf32x3")
        key_decay = _gates(prefix, b, h, kc, T, H, BT)
        decay = tl.exp(tl.where(causal, query_decay[:, None] - key_decay[None, :], 0.0))
        beta0, beta1, beta2, beta3, beta4 = _coefficients(beta, b, h, kc, T, H, BT, NBETA)
        polynomial = _polynomial(effective, beta0, beta1, beta2, beta3, beta4, NBETA)
        dweight = tl.dot(do, tl.trans(value), input_precision="tf32x3")
        grad_prefix += tl.sum(tl.where(causal, dweight * decay * polynomial, 0.0), 1)
        derivative = _derivative(effective, beta1, beta2, beta3, beta4, NBETA)
        deffective = tl.where(causal, dweight * decay * derivative, 0.0)
        deffective -= tl.dot(query_adjoint, tl.trans(key), input_precision="tf32x3") * erasure[None, :]
        if kc == qc:
            a_backward = _matrix(inverse, b, h, kc, T, H, BT, BT, BT)
            dscores = tl.dot(deffective, tl.trans(a_backward), input_precision="tf32x3")
            dscores = tl.where(causal, dscores, 0.0)
            query_adjoint += tl.dot(dscores, key, input_precision="tf32x3")
        else:
            r_backward = _matrix(transformed, b, h, kc, T, H, K, BK, BT)
            query_adjoint += tl.dot(deffective, r_backward, input_precision="tf32x3")
    rows, cols = qc * BT + tl.arange(0, BT), tl.arange(0, BK)
    offsets = ((b * T + rows[:, None]) * H + h) * K + cols[None, :]
    tl.store(dq + offsets, query_adjoint, mask=cols[None, :] < K)
    tl.store(dquery_prefix + (b * T + rows) * H + h, grad_prefix)


@triton.jit
def backward_keys(k, v, prefix, beta, erase, inverse, transformed, dout, history, adjoint_history, scale,
                  dk, dv, dbeta, derase, dinverse, dtransformed, dkey_prefix,
                  T: tl.constexpr, H: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                  BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                  GROUP: tl.constexpr, BQ: tl.constexpr, MODE: tl.constexpr, NBETA: tl.constexpr = 3):
    # GROUP=0 keeps the full adjoint for benchmarking; 1/2 split its value and
    # geometry outputs. Reducing simultaneous accumulators avoids register spills
    # and shared-memory pressure without changing the differentiated equations.
    VALUES: tl.constexpr = GROUP != 2
    GEOMETRY: tl.constexpr = GROUP != 1
    tl.static_assert(BQ == BT)
    kc, bh = tl.program_id(0), tl.program_id(1)
    b, h = bh // H, bh % H
    key = _matrix(k, b, h, kc, T, H, K, BK, BT)
    value = _matrix(v, b, h, kc, T, H, V, BV, BT)
    if MODE != 1:
        r = _matrix(transformed, b, h, kc, T, H, K, BK, BT)
    erasure = _gates(erase, b, h, kc, T, H, BT)
    key_decay = _gates(prefix, b, h, kc, T, H, BT)
    beta0, beta1, beta2, beta3, beta4 = _coefficients(beta, b, h, kc, T, H, BT, NBETA)
    if GEOMETRY:
        grad_key = tl.zeros((BT, BK), tl.float32)
        if MODE != 1:
            grad_transformed = tl.zeros((BT, BK), tl.float32)
        grad_erase = tl.zeros((BT,), tl.float32)
    if VALUES:
        grad_value = tl.zeros((BT, BV), tl.float32)
        grad_beta0 = tl.zeros((BT,), tl.float32)
        grad_beta1 = tl.zeros((BT,), tl.float32)
        grad_beta2 = tl.zeros((BT,), tl.float32)
        if NBETA >= 4:
            grad_beta3 = tl.zeros((BT,), tl.float32)
        if NBETA >= 5:
            grad_beta4 = tl.zeros((BT,), tl.float32)
        grad_prefix = tl.zeros((BT,), tl.float32)
    # Separating the single causal diagonal tile from fully preceding chunks
    # prevents its larger temporary matrices from spilling in every loop step.
    start = kc + (1 if MODE == 2 else 0)
    end = kc + 1 if MODE == 1 else T // BT
    for qb in range(start, end):
        query = _history_rows(history, bh, qb, kc, T, K, BK, BT, BQ)
        if GEOMETRY:
            query_adjoint = _history_rows(adjoint_history, bh, qb, kc, T, K, BK, BT, BQ)
        do = _matrix(dout, b, h, qb, T, H, V, BV, BQ) * scale
        causal = qb * BQ + tl.arange(0, BQ)[:, None] >= kc * BT + tl.arange(0, BT)[None, :]
        if MODE != 2 and (MODE == 1 or qb == kc):
            a = _matrix(inverse, b, h, kc, T, H, BT, BT, BT)
            scores = tl.where(causal, tl.dot(query, tl.trans(key), input_precision="tf32x3"), 0.0)
            effective = tl.dot(scores, a, input_precision="tf32x3")
        else:
            effective = tl.dot(query, tl.trans(r), input_precision="tf32x3")
        query_decay = _gates(prefix, b, h, qb, T, H, BQ)
        decay = tl.exp(tl.where(causal, query_decay[:, None] - key_decay[None, :], 0.0))
        dweight = tl.dot(do, tl.trans(value), input_precision="tf32x3")
        dpolynomial = tl.where(causal, dweight * decay, 0.0)
        if VALUES:
            polynomial = _polynomial(effective, beta0, beta1, beta2, beta3, beta4, NBETA)
            weights = tl.where(causal, decay * polynomial, 0.0)
            grad_prefix -= tl.sum(dweight * weights, 0)
            grad_beta0 += tl.sum(dpolynomial, 0)
            grad_beta1 += tl.sum(dpolynomial * effective, 0)
            grad_beta2 += tl.sum(dpolynomial * effective * effective, 0)
            if NBETA >= 4:
                grad_beta3 += tl.sum(dpolynomial * effective * effective * effective, 0)
            if NBETA >= 5:
                grad_beta4 += tl.sum(dpolynomial * effective * effective * effective * effective, 0)
            grad_value += tl.dot(tl.trans(weights), do, input_precision="tf32x3")
        if GEOMETRY:
            derivative = _derivative(effective, beta1, beta2, beta3, beta4, NBETA)
            deffective = dpolynomial * derivative
            deffective -= tl.dot(query_adjoint, tl.trans(key), input_precision="tf32x3") * erasure[None, :]
            direct = tl.dot(tl.trans(effective), query_adjoint, input_precision="tf32x3")
            grad_key -= erasure[:, None] * direct
            grad_erase -= tl.sum(direct * key, 1)
            if MODE != 2 and (MODE == 1 or qb == kc):
                a_backward = _matrix(inverse, b, h, kc, T, H, BT, BT, BT)
                scores_backward = tl.where(causal, tl.dot(query, tl.trans(key), input_precision="tf32x3"), 0.0)
                dscores = tl.where(causal, tl.dot(deffective, tl.trans(a_backward), input_precision="tf32x3"), 0.0)
                grad_key += tl.dot(tl.trans(dscores), query, input_precision="tf32x3")
                grad_inverse = tl.dot(tl.trans(scores_backward), deffective, input_precision="tf32x3")
                inverse_rows = kc * BT + tl.arange(0, BT)
                inverse_columns = tl.arange(0, BT)
                inverse_offsets = ((b * T + inverse_rows[:, None]) * H + h) * BT + inverse_columns[None, :]
                tl.store(dinverse + inverse_offsets, grad_inverse)
            else:
                grad_transformed += tl.dot(tl.trans(deffective), query, input_precision="tf32x3")
    rows, columns_k = kc * BT + tl.arange(0, BT), tl.arange(0, BK)
    token_offset = (b * T + rows) * H + h
    if GEOMETRY:
        tl.store(dk + token_offset[:, None] * K + columns_k[None, :], grad_key,
                 mask=columns_k[None, :] < K)
        tl.store(derase + token_offset, grad_erase)
        if MODE != 1:
            tl.store(dtransformed + token_offset[:, None] * K + columns_k[None, :], grad_transformed,
                     mask=columns_k[None, :] < K)
    if VALUES:
        columns_v = tl.arange(0, BV)
        tl.store(dv + token_offset[:, None] * V + columns_v[None, :], grad_value,
                 mask=columns_v[None, :] < V)
        tl.store(dbeta + token_offset * NBETA, grad_beta0)
        tl.store(dbeta + token_offset * NBETA + 1, grad_beta1)
        tl.store(dbeta + token_offset * NBETA + 2, grad_beta2)
        if NBETA >= 4:
            tl.store(dbeta + token_offset * NBETA + 3, grad_beta3)
        if NBETA >= 5:
            tl.store(dbeta + token_offset * NBETA + 4, grad_beta4)
        tl.store(dkey_prefix + token_offset, grad_prefix)

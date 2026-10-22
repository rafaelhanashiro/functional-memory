"""Scalar-function attention, adapted from Triton's 06-fused-attention tutorial.

Upstream: triton-lang/triton@93035c4472eb820f78074de0e28341fa321a759d,
python/tutorials/06-fused-attention.py. Query-owned forward/dQ and key-owned
dK/dV retain the tutorial's tiled matrix products and recomputation. Forward
uses tensor descriptors; backward uses pointer loads. The scalar callback is
specialized and inlined, with no sequence-square intermediate.

Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions: The above copyright notice and this
permission notice shall be included in all copies or substantial portions of
the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import triton as tr
import triton.language as tl


@tr.jit
def exponential(s, parameters: tl.constexpr, WITH_GRAD: tl.constexpr):
    value = tl.exp(s)
    return value, value if WITH_GRAD else 0.0


@tr.jit
def polynomial(s, coefficients: tl.constexpr, WITH_GRAD: tl.constexpr):
    value = tl.full(s.shape, coefficients[-1], tl.float32)
    derivative = tl.full(s.shape, 0.0, tl.float32)
    for i in tl.static_range(len(coefficients) - 2, -1, -1):
        if WITH_GRAD:
            derivative = derivative * s + value
        value = value * s + coefficients[i]
    return value, derivative


@tr.jit
def basis_term(s, kind: tl.constexpr, coefficient: tl.constexpr,
               p0: tl.constexpr, p1: tl.constexpr, WITH_GRAD: tl.constexpr):
    db = tl.full(s.shape, 0.0, tl.float32)
    if kind == 0:  # RBF
        z = (s - p0) / p1
        b = tl.exp(-0.5 * z * z)
        if WITH_GRAD:
            db = -z / p1 * b
    elif kind == 1:  # Sine
        z = p0 * s + p1
        b = tl.sin(z)
        if WITH_GRAD:
            db = p0 * tl.cos(z)
    elif kind == 2:  # Cosine
        z = p0 * s + p1
        b = tl.cos(z)
        if WITH_GRAD:
            db = -p0 * tl.sin(z)
    else:
        b = tl.full(s.shape, 1.0, tl.float32)
        for j in tl.static_range(p0):
            b = b * s
        if WITH_GRAD:
            if p0 > 0:
                db = tl.full(s.shape, p0, tl.float32)
                for j in tl.static_range(p0 - 1):
                    db = db * s
    return coefficient * b, coefficient * db


@tr.jit
def basis_sum(s, terms: tl.constexpr, WITH_GRAD: tl.constexpr):
    value = tl.full(s.shape, 0.0, tl.float32)
    derivative = tl.full(s.shape, 0.0, tl.float32)
    for i in tl.static_range(len(terms) // 4):
        b, db = basis_term(s, terms[4*i], terms[4*i+1], terms[4*i+2], terms[4*i+3], WITH_GRAD)
        value = value + b
        if WITH_GRAD:
            derivative = derivative + db
    return value, derivative


@tr.jit
def dot(a, b, PRECISE: tl.constexpr):
    if PRECISE or b.dtype == tl.float32:
        return tl.dot(a.to(tl.float32), b.to(tl.float32), input_precision="tf32x3")
    return tl.dot(a.to(b.dtype), b)


@tr.jit
def scaled_values(s, log_scale, FN: tl.constexpr, PARAMS: tl.constexpr,
                  EXP: tl.constexpr, GRAD: tl.constexpr):
    if EXP:
        w, dw = FN(s - log_scale, PARAMS, GRAD)
    else:
        w, dw = FN(s, PARAMS, GRAD)
        scale = tl.exp(-log_scale)
        w = w * scale
        if GRAD:
            dw = dw * scale
    return w, dw


@tr.jit
def forward_inner(acc, rowsum, log_scale, invalid, q, kd, vd, qi,
                  start, stop, T: tl.constexpr, SCALE: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, PRECISE: tl.constexpr,
                  FN: tl.constexpr, PARAMS: tl.constexpr, EXP: tl.constexpr,
                  MODE: tl.constexpr, STREAM: tl.constexpr, DIAGONAL: tl.constexpr):
    for lo in range(start, stop, BN):
        ki = lo + tl.arange(0, BN)
        k = kd.load([lo, 0]).T
        s = dot(q, k, False) * SCALE
        valid = (qi[:, None] < T) & (ki[None, :] < T)
        if DIAGONAL:
            valid = valid & (qi[:, None] >= ki[None, :])
        invalid |= tl.sum((valid & ((s != s) | (tl.abs(s) == float("inf")))).to(tl.int32), 1) > 0
        s = tl.where(valid, s, 0.0)
        if STREAM:
            if EXP:
                bound = tl.max(tl.where(valid, s, -float("inf")), 1)
            else:
                raw, _ = FN(s, PARAMS, False)
                invalid |= tl.sum((valid & ((raw != raw) | (tl.abs(raw) == float("inf")))).to(tl.int32), 1) > 0
                bound = tl.log(tl.maximum(tl.max(tl.where(valid, tl.abs(raw), 0.0), 1), 1.0e-30))
            middle = tl.maximum(log_scale, bound)
            if EXP:
                w, _ = scaled_values(s, middle[:, None], FN, PARAMS, EXP, False)
            else:
                w = raw * tl.exp(-middle[:, None])
            w = tl.where(valid, w, 0.0)
            previous = tl.exp(log_scale - middle)
            mass = previous + tl.sum(tl.abs(w), 1)
            alpha = previous / mass
            w = w / mass[:, None]
            acc = acc * alpha[:, None]
            rowsum = rowsum * alpha
            log_scale = middle + tl.log(mass)
        else:
            w, _ = FN(s, PARAMS, False)
            invalid |= tl.sum((valid & ((w != w) | (tl.abs(w) == float("inf")))).to(tl.int32), 1) > 0
            w = tl.where(valid, w, 0.0)
        v = vd.load([lo, 0])
        acc += dot(w, v, PRECISE)
        if MODE == 1:
            rowsum += tl.sum(w, 1)
    return acc, rowsum, log_scale, invalid


@tr.jit
def forward(Q, K, V, Gain, O, N, Stats, T: tl.constexpr, H: tl.constexpr,
            DQ: tl.constexpr, DV: tl.constexpr, SCALE: tl.constexpr,
            EPS: tl.constexpr, MODE: tl.constexpr, STREAM: tl.constexpr,
            FN: tl.constexpr, PARAMS: tl.constexpr, EXP: tl.constexpr,
            PRECISE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    tile = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // H, bh % H
    qbase = (batch * T * H + head) * DQ
    vbase = (batch * T * H + head) * DV
    qd = tl.make_tensor_descriptor(Q + qbase, [T, DQ], [H * DQ, 1], [BM, DQ])
    kd = tl.make_tensor_descriptor(K + qbase, [T, DQ], [H * DQ, 1], [BN, DQ])
    vd = tl.make_tensor_descriptor(V + vbase, [T, DV], [H * DV, 1], [BN, DV])
    q = qd.load([tile * BM, 0])
    qi = tile * BM + tl.arange(0, BM)
    acc = tl.zeros((BM, DV), tl.float32)
    rowsum = tl.zeros((BM,), tl.float32)
    if STREAM and EXP and MODE == 1:
        # Normalized exp must also shift entirely negative rows. A unit prior
        # mass would make exp(-1000) underflow before normalization.
        log_scale = tl.full((BM,), -float("inf"), tl.float32)
    else:
        log_scale = tl.zeros((BM,), tl.float32)  # a starts at 1, then adds absolute weight mass.
    invalid = tl.full((BM,), False, tl.int1)
    acc, rowsum, log_scale, invalid = forward_inner(
        acc, rowsum, log_scale, invalid, q, kd, vd, qi, 0, tile * BM,
        T, SCALE, BM, BN, PRECISE, FN, PARAMS, EXP, MODE, STREAM, False)
    acc, rowsum, log_scale, invalid = forward_inner(
        acc, rowsum, log_scale, invalid, q, kd, vd, qi, tile * BM, (tile + 1) * BM,
        T, SCALE, BM, BN, PRECISE, FN, PARAMS, EXP, MODE, STREAM, True)
    channels = tl.arange(0, DV)
    if MODE == 2:
        # Also avoid overflow when squaring the final scaled output.
        extra = tl.maximum(1.0, tl.max(tl.abs(acc), 1))
        acc = acc / extra[:, None]
        log_scale += tl.log(extra)
        r = tl.rsqrt(tl.sum(acc * acc, 1) / DV + EPS * tl.exp(-2.0 * log_scale))
        n = acc * r[:, None]
        gain = tl.load(Gain + head * DV + channels)
        out = n * gain[None, :]
    elif MODE == 1:
        invalid |= (rowsum == 0.0) | (rowsum != rowsum) | (tl.abs(rowsum) == float("inf"))
        r = 1.0 / rowsum
        out = acc * r[:, None]
        n = out
    else:
        r = tl.full((BM,), 1.0, tl.float32)
        out = acc * tl.exp(log_scale[:, None])
        n = out
    invalid |= tl.sum(((out != out) | (tl.abs(out) == float("inf"))).to(tl.int32), 1) > 0
    out = tl.where(invalid[:, None], float("nan"), out)
    n = tl.where(invalid[:, None], float("nan"), n)
    idx = vbase + qi[:, None] * H * DV + channels[None, :]
    tl.store(O + idx, out, qi[:, None] < T)
    tl.store(N + idx, n, qi[:, None] < T)
    si = ((batch * T + qi) * H + head) * 2
    tl.store(Stats + si, log_scale, qi < T)
    tl.store(Stats + si + 1, r, qi < T)


@tr.jit
def preprocess(N, Stats, Gain, DO, Adj, Delta, DGain,
               T: tl.constexpr, H: tl.constexpr, DV: tl.constexpr,
               MODE: tl.constexpr, BM: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    qi = tile * BM + tl.arange(0, BM)
    d = tl.arange(0, DV)
    idx = ((batch * T + qi[:, None]) * H + head) * DV + d[None, :]
    go = tl.load(DO + idx, qi[:, None] < T, 0).to(tl.float32)
    if MODE:
        n = tl.load(N + idx, qi[:, None] < T, 0)
        r = tl.load(Stats + ((batch * T + qi) * H + head) * 2 + 1, qi < T, 0)
    if MODE == 2:
        gain = tl.load(Gain + head * DV + d)
        t = go * gain[None, :]
        projection = tl.sum(t * n, 1) / DV
        h = (t - n * projection[:, None]) * r[:, None]
        partial = tl.sum(go * n, 0)
        tl.store(DGain + (bh * tl.cdiv(T, BM) + tile) * DV + d, partial)
        delta = tl.full((BM,), 0.0, tl.float32)
    elif MODE == 1:
        h = go * r[:, None]
        delta = tl.sum(go * n, 1) * r
    else:
        h = go
        delta = tl.full((BM,), 0.0, tl.float32)
    tl.store(Adj + idx, h, qi[:, None] < T)
    tl.store(Delta + (batch * T + qi) * H + head, delta, qi < T)


@tr.jit
def backward_q(Q, K, V, Adj, Delta, Stats, DQOut,
               T: tl.constexpr, H: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr,
               SCALE: tl.constexpr, MODE: tl.constexpr, FN: tl.constexpr,
               PARAMS: tl.constexpr, EXP: tl.constexpr, PRECISE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    qi = tile * BM + tl.arange(0, BM)
    d, e = tl.arange(0, DQ), tl.arange(0, DV)
    qidx = ((batch * T + qi[:, None]) * H + head) * DQ + d[None, :]
    vidx = ((batch * T + qi[:, None]) * H + head) * DV + e[None, :]
    q = tl.load(Q + qidx, qi[:, None] < T, 0)
    adj = tl.load(Adj + vidx, qi[:, None] < T, 0)
    # The adjoint enters tensor cores at the declared precision.
    if not PRECISE:
        adj = adj.to(V.dtype.element_ty)
    delta = tl.load(Delta + (batch * T + qi) * H + head, qi < T, 0)
    if MODE:
        logs = tl.load(Stats + ((batch * T + qi) * H + head) * 2, qi < T, 0)
    else:
        logs = tl.full((BM,), 0.0, tl.float32)
    dq = tl.zeros((BM, DQ), tl.float32)
    for lo in range(0, (tile + 1) * BM, BN):
        ki = lo + tl.arange(0, BN)
        k = tl.load(K + ((batch * T + ki[None, :]) * H + head) * DQ + d[:, None], ki[None, :] < T, 0)
        v = tl.load(V + ((batch * T + ki[None, :]) * H + head) * DV + e[:, None], ki[None, :] < T, 0)
        valid = (qi[:, None] >= ki[None, :]) & (qi[:, None] < T) & (ki[None, :] < T)
        s = dot(q, k, False) * SCALE
        _, dw = scaled_values(tl.where(valid, s, 0.0), logs[:, None], FN, PARAMS, EXP, True)
        dp = dot(adj, v, PRECISE)
        ds = tl.where(valid, dw * (dp - delta[:, None]), 0.0)
        dq += dot(ds, k.T, PRECISE) * SCALE
    tl.store(DQOut + qidx, dq, qi[:, None] < T)


@tr.jit
def backward_kv(Q, K, V, Adj, Delta, Stats, DK, DVOut,
                T: tl.constexpr, H: tl.constexpr, DQ: tl.constexpr, DV: tl.constexpr,
                SCALE: tl.constexpr, MODE: tl.constexpr, FN: tl.constexpr,
                PARAMS: tl.constexpr, EXP: tl.constexpr, PRECISE: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr):
    tile, bh = tl.program_id(0), tl.program_id(1)
    batch, head = bh // H, bh % H
    ki = tile * BN + tl.arange(0, BN)
    d, e = tl.arange(0, DQ), tl.arange(0, DV)
    kidx = ((batch * T + ki[:, None]) * H + head) * DQ + d[None, :]
    vidx = ((batch * T + ki[:, None]) * H + head) * DV + e[None, :]
    k = tl.load(K + kidx, ki[:, None] < T, 0)
    v = tl.load(V + vidx, ki[:, None] < T, 0)
    dk = tl.zeros((BN, DQ), tl.float32)
    dv = tl.zeros((BN, DV), tl.float32)
    for lo in range(tile * BN // BM * BM, T, BM):
        qi = lo + tl.arange(0, BM)
        qt = tl.load(Q + ((batch * T + qi[None, :]) * H + head) * DQ + d[:, None], qi[None, :] < T, 0)
        adj = tl.load(Adj + ((batch * T + qi[:, None]) * H + head) * DV + e[None, :], qi[:, None] < T, 0)
        if not PRECISE:
            adj = adj.to(V.dtype.element_ty)
        delta = tl.load(Delta + (batch * T + qi) * H + head, qi < T, 0)
        if MODE:
            logs = tl.load(Stats + ((batch * T + qi) * H + head) * 2, qi < T, 0)
        else:
            logs = tl.full((BM,), 0.0, tl.float32)
        valid = (qi[None, :] >= ki[:, None]) & (qi[None, :] < T) & (ki[:, None] < T)
        s = dot(k, qt, False) * SCALE
        w, dw = scaled_values(tl.where(valid, s, 0.0), logs[None, :], FN, PARAMS, EXP, True)
        w = tl.where(valid, w, 0.0)
        dv += dot(w, adj, PRECISE)
        dp = dot(v, adj.T, PRECISE)
        ds = tl.where(valid, dw * (dp - delta[None, :]), 0.0)
        dk += dot(ds, qt.T, PRECISE) * SCALE
    tl.store(DK + kidx, dk, ki[:, None] < T)
    tl.store(DVOut + vidx, dv, ki[:, None] < T)


@tr.jit
def uniform_rbf(s, parameters: tl.constexpr, WITH_GRAD: tl.constexpr):
    center: tl.constexpr = parameters[0]
    step: tl.constexpr = parameters[1]
    inverse_variance: tl.constexpr = parameters[2]
    decay: tl.constexpr = parameters[3]
    count: tl.constexpr = len(parameters) - 4
    right = s > center + (count - 1) * step * 0.5
    delta = s - tl.where(right, center + (count - 1) * step, center)
    direction = tl.where(right, -step, step)
    b = tl.exp(-0.5 * delta * delta * inverse_variance)
    ratio = tl.exp((direction * delta - 0.5 * step * step) * inverse_variance)
    value = tl.full(s.shape, 0.0, tl.float32)
    derivative = tl.full(s.shape, 0.0, tl.float32)
    slope = -delta * inverse_variance
    for i in tl.static_range(count):
        coefficient = tl.where(right, parameters[4 + count - 1 - i], parameters[4 + i])
        term = coefficient * b
        value += term
        if WITH_GRAD:
            derivative += term * slope
            slope += direction * inverse_variance
        b *= ratio
        ratio *= decay
    return value, derivative

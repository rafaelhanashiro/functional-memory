"""Per-query polynomials p_t(z) = sum_r c_tr z^r of degree M, z = s or softplus(s), in common row units e^(-m_t), or
f(s) p_t(s) for a prefactor f of the raw score (softplus or exp), in row units that bound every key's.

m_t = M log max(1, max_i |z_ti|) over query t's causal row bounds every term: with half = e^(-m_t / (2M)) and
u = (z half) half, |u| <= 1 and c_r z^r e^(-m_t) = c_r e^(-(M - r) m_t / M) u^r. Head RMSNorm divides the row factor
out when its epsilon is scaled by e^(-2 m_t) (attention/output_scaling.py); m_t is a constant to differentiation.
The layer supplies c per query: the tanh query generator, direct or sum of squares (attention/nonnegative_polynomial.py,
prefactor included), pure tanh, or static fixed or learned coefficients broadcast to every query.

With a prefactor the key a weight is largest for need not be the key of largest |s|: f suppresses some keys (softplus
and exp the far negative ones), so units set by the largest |s| alone could underflow every weight that matters. Each
key i instead has the bound b_i = log f(s_i) + M log S_i, S_i = max(1, |s_i|), on |f(s_i) s_i^r| for every r <= M, and
the row takes units m_t >= max(0, max_i b_i) from its largest score alone (key_shift).

f(s) p_t(s) e^(-m_t) is then evaluated in one pass, without a logarithm or a reciprocal of a score. The polynomial
splits into n >= 2 chunks of width d (chunking), p(s) = sum_j s^(j d) P_j(s), each chunk by Horner's rule in s. Each
key takes g with g^n the exponential part of f(s) e^(-m_t) (key_stage): e^((s - m_t)/n) for exp and for softplus's
tail s <= -20, where softplus is e^s within e^-20; e^(-m_t/n), with softplus(s) left as a factor F, elsewhere. Then
f(s) p_t(s) e^(-m_t) = T_0 with T_(n-1) = g F P_(n-1) and T_j = T_(j+1) (g s^d) + g^(n-j) F P_j (staged): since
F g^n S_i^M <= 1, every partial sum stays within the coefficients' scale, where direct evaluation overflows at
|s|^M > FLT_MAX. A chunk's own powers pass FLT_MAX only at |s|^d, above 4e9 at width 4. The slope in s with m_t
fixed stages the chunks F1 P_j + F P_j' likewise, F1 the factor of f' outside g (keyed_weights).
"""
import math

import jax
from jax import lax
import jax.numpy as jnp


TRANSFORMS = ("identity", "softplus")
# Multipliers f(s) of the raw score; a prefactor takes z = s.
PREFACTORS = ("none", "softplus", "exp")
# A prefactor's polynomial is evaluated in chunks of at most this width.
CHUNK = 4
# At and below this score softplus(s) is taken as e^s (within e^-20).
SOFTPLUS_TAIL = -20.0
# Stands for the largest score of a row without a causal key; its units are the floor, zero.
NO_SCORE = -1e30


def transformed(scores, transform):
    """z = phi(s) in FP32."""
    if transform == "identity":
        return scores
    if transform == "softplus":
        return jax.nn.softplus(scores)
    raise ValueError(f"The shifted polynomial takes the transforms {TRANSFORMS}, not {transform!r}")


def row_shift(z, degree):
    """M log max(1, max |z|) over the last axis, the log of the row's largest power. Entries of magnitude at most one
    (masked ones, say) leave it unchanged."""
    return lax.stop_gradient(degree * jnp.log(jnp.maximum(1.0, jnp.max(jnp.abs(z), axis=-1))))


def log_prefactor(scores, prefactor):
    """log f(s): s for exp; for softplus log softplus(s), taken as s below s = -20 (within e^-20; softplus(s) underflows
    FP32 past -87)."""
    if prefactor == "exp":
        return scores
    if prefactor == "softplus":
        return jnp.where(scores > -20.0, jnp.log(jax.nn.softplus(jnp.maximum(scores, -20.0))), scores)
    raise ValueError(f"The shifted polynomial takes the prefactors {PREFACTORS}, not {prefactor!r}")


def key_bound(scores, prefactor, degree):
    """b = log f(s) + M log max(1, |s|): the log of f(s) |s|^r's largest value over r <= M."""
    return log_prefactor(scores, prefactor) + degree * jnp.log(jnp.maximum(1.0, jnp.abs(scores)))


def key_shift(top, prefactor, degree):
    """Row units m >= max(0, max_i b_i) from the row's largest causal score x alone. b increases on [-1, inf); on
    (-inf, -1] it is at most s + M log(-s) (log softplus(s) <= s), which increases up to its peak M log M - M at
    s = -M. So m = max(0, x + M log(-x)) for x <= -M and max(0, M log M - M, b(x)) above: the row's largest bound
    itself unless -M < x and b(x) < M log M - M, where it exceeds that bound by at most M log M - M."""
    if prefactor not in PREFACTORS[1:]:
        raise ValueError(f"The shifted polynomial takes the prefactors {PREFACTORS}, not {prefactor!r}")
    magnitude = degree * jnp.log(jnp.maximum(1.0, jnp.abs(top)))
    inner = jnp.maximum(top, -float(degree))
    if prefactor == "softplus":
        inner = jnp.log(jnp.maximum(inner, 0.0) + jnp.log1p(jnp.exp(-jnp.abs(inner))))
    peak = degree * math.log(degree) - degree
    return jnp.maximum(0.0, jnp.where(top <= -degree, top + magnitude, jnp.maximum(peak, inner + magnitude)))


def chunking(degree):
    """Chunk count n >= 2 and width d = ceil(M / n): chunk j holds orders j d to j d + d - 1, the last up to M. Two
    chunks at least keep g, an n-th root, normal wherever a weight is."""
    count = max(2, -(-degree // CHUNK))
    return count, -(-degree // count)


def key_chunks(values, count, width):
    """The chunks of a sequence of per-order values; a derivative's top chunk may be empty."""
    return [values[j * width:(j + 1) * width if j < count - 1 else len(values)] for j in range(count)]


def integer_power(x, exponent):
    """x^exponent for a static positive integer, by squaring."""
    result, base = None, x
    while exponent:
        if exponent & 1:
            result = base if result is None else result * base
        exponent >>= 1
        if exponent:
            base = base * base
    return result


def key_stage(scores, shift, prefactor, count):
    """Per key: g, with g^n the exponential part of f(s) e^(-m), and the factors F of f and F1 of f' that it leaves
    out (None, None for exp, where both are one). Softplus takes e^(-|s|/n) once: its n-th power gives
    softplus(s) = max(s, 0) + log1p(e^(-|s|)) and the logistic sigmoid, and on the tail it is e^(s/n) itself."""
    if prefactor == "exp":
        return jnp.exp((scores - shift) * (1.0 / count)), None, None
    if prefactor != "softplus":
        raise ValueError(f"The shifted polynomial takes the prefactors {PREFACTORS}, not {prefactor!r}")
    root = jnp.exp(jnp.abs(scores) * (-1.0 / count))
    small = integer_power(root, count)
    row = jnp.exp(shift * (-1.0 / count))
    tail = scores <= SOFTPLUS_TAIL
    softplus = jnp.maximum(scores, 0.0) + jnp.log1p(small)
    sigmoid = jnp.where(scores >= 0, 1.0 / (1.0 + small), small / (1.0 + small))
    return jnp.where(tail, root * row, row), jnp.where(tail, 1.0, softplus), jnp.where(tail, 1.0, sigmoid)


def horner(coefficients, x):
    """sum_k coefficients[k] x^k for coefficients that broadcast against x."""
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * x + coefficient
    return value


def chunk_values(coefficients, count, width, scores):
    """Each chunk's polynomial P_j(s) by Horner's rule; None for an empty chunk."""
    return [horner(part, scores) if part else None for part in key_chunks(coefficients, count, width)]


def staged(values, g, w):
    """g^n sum_j s^(j d) values[j] from the top chunk down: T_j = T_(j+1) w + g^(n-j) values[j] with w = g s^d. A
    chunk meets g one factor at a time, so no power of g underflows before its product does."""
    count, total = len(values), None
    for j in range(count - 1, -1, -1):
        if values[j] is None:
            continue
        term = values[j]
        for _ in range(count - j):
            term = term * g
        total = term if total is None else total * w + term
    return total


def key_terms(scores, shift, prefactor, degree):
    """A tile's shared terms (n, d, g, w, F, F1): the chunking, key_stage's factors, and w = g s^d. shift broadcasts
    against scores."""
    count, width = chunking(degree)
    g, factor, factor_slope = key_stage(scores, shift, prefactor, count)
    return count, width, g, g * integer_power(scores, width), factor, factor_slope


def keyed_weights(scores, coefficients, terms, *, value=True, slope=False):
    """(f(s) p(s) e^(-m), its slope in s with m fixed), each None unless asked for. coefficients holds the M + 1
    orders, each broadcasting against scores; terms are key_terms'. The factors F and F1 enter the chunks before g
    does, so a small weight is not lost through a smaller T_0; for exp the slope's chunks are those of p + p'."""
    count, width, g, w, factor, factor_slope = terms
    derivative = [r * c for r, c in enumerate(coefficients)][1:]
    weights = slopes = None
    if factor is None:
        if value:
            weights = staged(chunk_values(coefficients, count, width, scores), g, w)
        if slope:
            combined = [c + d for c, d in zip(coefficients, derivative)] + [coefficients[-1]]
            slopes = staged(chunk_values(combined, count, width, scores), g, w)
        return weights, slopes
    own = chunk_values(coefficients, count, width, scores)
    if value:
        weights = staged([factor * x for x in own], g, w)
    if slope:
        other = chunk_values(derivative, count, width, scores)
        slopes = staged([factor_slope * x if y is None else factor_slope * x + factor * y
                         for x, y in zip(own, other, strict=True)], g, w)
    return weights, slopes


def shifted_terms(z, shift, degree):
    """e^-shift z^r for r = 0..M as u^r e^(-(M - r) shift / M); shift broadcasts against z. Applying the scale in two
    halves keeps it normal where its reciprocal would not be."""
    half = jnp.exp(shift * (-0.5 / degree))
    u = (z * half) * half
    return [(u ** r) * (1.0 if r == degree else jnp.exp(shift * (-(degree - r) / degree))) for r in range(degree + 1)]


def _reference(q, k, v, c, transform, return_diagnostics, prefactor="none"):
    """Evaluate complete causal rows in small batches; every diagnostic uses the same shifted weights."""
    batch, time, heads, width = q.shape
    degree = c.shape[-1] - 1
    block = min(32, time)
    positions = jnp.arange(time)

    def rows(block_index):
        query_positions = block * block_index + jnp.arange(block)
        index = jnp.minimum(query_positions, time - 1)
        scores = jnp.einsum("bqhd,bkhd->bhqk", q[:, index], k, preferred_element_type=jnp.float32,
                            precision=lax.Precision.HIGHEST) * (1.0 / math.sqrt(width))
        causal = positions[None, None, None, :] <= index[None, None, :, None]
        scores = jnp.where(causal, scores, 0.0)
        # A masked score is zero, so its z (0, or log 2 for softplus) is below the shift's floor of one.
        z = transformed(scores, transform)
        coefficients = c[:, index].swapaxes(1, 2)
        if prefactor == "none":
            shift = row_shift(z, degree)
            terms = shifted_terms(z, shift[..., None], degree)
            weights = sum(coefficients[..., r, None] * term for r, term in enumerate(terms))
        else:
            # z = s: the row in units that bound every key's, from its largest causal score.
            shift = lax.stop_gradient(key_shift(jnp.max(jnp.where(causal, scores, NO_SCORE), axis=-1), prefactor,
                                                degree))
            weights, _ = keyed_weights(scores, [coefficients[..., r, None] for r in range(degree + 1)],
                                       key_terms(scores, shift[..., None], prefactor, degree))
        weights = jnp.where(causal, weights, 0.0)
        output = jnp.einsum("bhqk,bkhd->bqhd", weights, v.astype(jnp.float32), precision=lax.Precision.HIGHEST)
        outputs = output, shift.swapaxes(1, 2)
        if not return_diagnostics:
            return outputs
        actual = query_positions < time
        measured = actual[None, None, :, None] & causal
        weight_sums = weights.sum(axis=-1)
        # sum_i |w_ti| / |sum_i w_ti| per row, unchanged by the row's factor; 1 for nonnegative weights.
        cancellation = jnp.abs(weights).sum(axis=-1) / jnp.abs(weight_sums)
        # A weight is a sum of M + 1 rounded terms, so FP32 resolves it to epsilon of their magnitudes, and, where
        # a key lies so far below its row's largest power that its terms underflow, to the smallest normal number
        # a term (FLT_MIN where underflow flushes to zero, less where it is gradual). A nonnegative polynomial may
        # round below zero within that resolution, never beyond it. Under a prefactor a key's terms are at most
        # |c_r| in its own units.
        fp32 = jnp.finfo(jnp.float32)
        total = jnp.abs(coefficients).sum(axis=-1)[..., None]
        magnitude = (sum(jnp.abs(coefficients[..., r, None] * term) for r, term in enumerate(terms))
                     if prefactor == "none"
                     else jnp.exp(key_bound(scores, prefactor, degree) - shift[..., None]) * total)
        resolution = (degree + 1) * float(fp32.eps) * magnitude + float(fp32.tiny) * (degree + 1 + total)
        min_measured = lambda x, mask: jnp.min(jnp.where(mask, x, jnp.inf))
        max_measured = lambda x, mask: jnp.max(jnp.where(mask, x, -jnp.inf))
        stats = {"score_min": min_measured(scores, measured), "score_max": max_measured(scores, measured),
                 "kernel_min": min_measured(weights, measured), "kernel_max": max_measured(weights, measured),
                 "weight_sum_min": min_measured(weight_sums, actual), "weight_sum_max": max_measured(weight_sums, actual),
                 "polynomial_shift_min": min_measured(shift, actual),
                 "polynomial_shift_max": max_measured(shift, actual),
                 "invalid_score_count": jnp.sum(measured & ~jnp.isfinite(scores)),
                 "invalid_kernel_count": jnp.sum(measured & ~jnp.isfinite(weights)),
                 "invalid_weight_sum_count": jnp.sum(actual & ~jnp.isfinite(weight_sums)),
                 "invalid_output_count": jnp.sum(actual[None, :, None, None] & ~jnp.isfinite(output)),
                 "causal_entry_count": jnp.sum(measured) * batch * heads,
                 "negative_weight_count": jnp.sum(measured & (weights < 0)),
                 "negative_weight_beyond_rounding_count": jnp.sum(measured & (weights < -resolution)),
                 "cancellation_ratio_max": max_measured(cancellation, actual[None, None, :]),
                 "cancellation_sum": jnp.sum(jnp.where(actual[None, None, :], cancellation, 0.0)),
                 "row_count": jnp.sum(actual) * batch * heads}
        if transform != "identity":
            # The transform's per-head measurements, as the blocked backend records them for its score functions.
            from scaffolds.transformer.attention.backends.reference import _score_transform_statistics

            stats.update(_score_transform_statistics(scores, z, measured, lambda r: coefficients[..., r, None],
                                                     degree))
        return outputs, stats

    result = lax.map(rows, jnp.arange((time + block - 1) // block))
    if return_diagnostics:
        result, stats = result

    def flatten(x):
        return x.swapaxes(0, 1).reshape(batch, -1, *x.shape[3:])[:, :time]

    output = tuple(flatten(x) for x in result)
    if not return_diagnostics:
        return output
    # Each block's scalars reduce to scalars; the transform's per-head and histogram arrays keep their own axes.
    axis = lambda name: 0 if name.startswith("score_transform_") else None
    stats = {name: (values.min(axis=axis(name)) if name.endswith("_min") else
                    values.max(axis=axis(name)) if name.endswith("_max") else values.sum(axis=axis(name)))
             for name, values in stats.items()}
    stats["min_abs_weight_sum"] = stats["weight_sum_min"]
    stats["negative_kernel_count"] = stats["negative_weight_count"]
    stats["negative_weight_fraction"] = stats["negative_weight_count"] / stats["causal_entry_count"]
    stats["negative_kernel_fraction"] = stats["negative_weight_fraction"]
    stats["cancellation_ratio_mean"] = stats.pop("cancellation_sum") / stats.pop("row_count")
    from scaffolds.transformer.attention.input_dependent import coefficient_statistics
    stats.update(coefficient_statistics(c))
    return output, lax.stop_gradient(stats)


def shifted_polynomial_attention(q, k, v, coefficients, *, transform="identity", prefactor="none", backend="blocked",
                                 return_diagnostics=False):
    """Return the shifted output e^(-m) o and the stopped row shifts m [B,T,H], which head RMSNorm takes."""
    if (q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3]
            or min(*q.shape, v.shape[-1]) <= 0 or coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3]
            or coefficients.shape[-1] < 2):
        raise ValueError("Expected Q/K/V [B,T,H,D] and query coefficients [B,T,H,M+1] with M >= 1")
    if (q.dtype != k.dtype or q.dtype != v.dtype or q.dtype not in (jnp.float32, jnp.bfloat16)
            or coefficients.dtype != jnp.float32):
        raise ValueError("The shifted polynomial requires BF16 or FP32 Q/K/V and FP32 coefficients")
    if transform not in TRANSFORMS:
        raise ValueError(f"The shifted polynomial takes the transforms {TRANSFORMS}, not {transform!r}")
    if prefactor not in PREFACTORS or (prefactor != "none" and transform != "identity"):
        raise ValueError(f"The shifted polynomial takes the prefactors {PREFACTORS}, of the raw score with z = s")
    if backend == "blocked":
        return _reference(q, k, v, coefficients, transform, return_diagnostics, prefactor)
    if backend != "fused" or max(q.shape[-1], v.shape[-1]) > 128:
        raise ValueError("The shifted polynomial uses its blocked reference or fused kernel (width <= 128)")
    from scaffolds.transformer.attention.backends.fused_shifted_polynomial import fused_shifted_polynomial
    output = fused_shifted_polynomial(q, k, v, coefficients, transform=transform, prefactor=prefactor)
    if not return_diagnostics:
        return output
    _, stats = _reference(q, k, v, coefficients, transform, True, prefactor)
    return output, stats

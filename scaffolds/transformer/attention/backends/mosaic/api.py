"""Causal polynomial attention on Mosaic GPU kernels (Hopper and Blackwell).

Two operations, each with the mathematics and failure behavior of its Pallas-Triton
counterpart. FP32 scores, polynomials, sums, and accumulators. FP32 weights and score
gradients enter the tensor cores at ``precision``: ``"bf16"`` rounds them once to BF16, as
FlashAttention and cuDNN do for probabilities and score gradients, and the Taylor backward
reads the BF16 output, as theirs does; ``"bf16x3"`` splits them into three BF16 components
whose sum is the FP32 value and keeps the FP32 output, as the Triton kernels do. Each setup
defaults to ``default_precision``.

- ``taylor_attention`` (``fused_taylor.fused_polynomial_attention``): normalized fixed
  Taylor T_m. A row whose sum is nonfinite, nonpositive (even degree) or zero (odd degree),
  or whose output overflows, is NaN. The output is divided by its row sum once, after
  accumulation, so signed odd degrees need a single pass.
- ``query_polynomial_attention`` (``fused_query_polynomial``): unnormalized
  sum_r c_(t,r) s^r with FP32 coefficients for every query and head. A row with a
  nonfinite score or weight, or whose BF16 output overflows, is NaN; coefficient
  gradients are FP32 and stay per query.
- ``transformed_query_polynomial_attention`` (the blocked reference's ``score_transform`` and
  ``score_prefactor``): the same polynomial evaluated at z = phi(s) per causal pair, phi = softplus, abs, sigmoid
  or swish, and optionally multiplied by a prefactor f(s) = softplus, sigmoid, exp or swish of the raw score
  (with phi the identity or softplus), inside the tiles of the same kernels, with the chain and product rules
  through phi and f in both gradient passes.
- ``query_decay_attention`` and ``endpoint_decay_attention`` (the blocked reference's
  query-distance forget gates): the query polynomial times exp((t-i) l), with a per-query
  log rate l_t <= 0 or the endpoint rate l_ti = -exp(A_h) softplus(u_t + v_i). Decay is
  evaluated inside the causal tiles and its gradients are accumulated there: per-query sums
  in the dQ pass, per-key sums in the dK pass. An invalid gate makes the affected causal
  rows NaN; the diagonal is never attenuated.
- ``per_order_query_decay_attention`` and ``per_order_endpoint_decay_attention``: the same
  gates with an independent rate for every polynomial order r (including r = 0), as the
  blocked reference's per-order gates: sum_r c_r s^r exp((t-i) l_r). H100 stages their
  vectors per tile; B300 keeps every query's and key's per-order vectors in shared memory and
  raises ValueError, before tracing a kernel, for shapes it cannot hold (``blackwell_supported``:
  at Q/K width 64 and 2,048 tokens, degree 7 query and degree 6 query-key gates fit).
- ``per_order_cumulative_decay_attention``: the blocked reference's per-order cumulative gates,
  sum_r c_r s^r exp(L_tr - L_ir) with every order's own cumulative log decay L_r, its FP32
  difference formed inside the causal tiles. A nonfinite L makes every causal pair it enters
  NaN. H100 only: B300 raises ValueError before tracing a kernel.

Supported: BF16 Q/K/V of shape [batch, time, heads, channels], Q/K width 4, 8, 16, 32, 64, or
128 (widths 4 and 8 are zero-padded to the tensor cores' minimum of 16, which leaves every score
unchanged; the scale is that of the unpadded width; narrower tiles use narrower shared-memory swizzles), value width 64, degree
>= 1 (ordinary query coefficients: degree <= 12, subject to shared memory), and time a
multiple of 128 (H100) or 256 (Blackwell). Softplus additionally admits degree 13 at width
64 and 512/2,048 tokens; transformed kernels admit H100 and B200 only. Unsupported shapes
or architectures raise ValueError.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.backends.mosaic import common
from scaffolds.transformer.attention.backends.mosaic.common import (
    BF16, BF16X3, ENDPOINT_DECAY, PER_ORDER_CUMULATIVE_DECAY, PER_ORDER_ENDPOINT_DECAY, PER_ORDER_QUERY_DECAY,
    PRECISIONS, QUERY, QUERY_ABS, QUERY_DECAY, QUERY_SIGMOID, QUERY_SOFTPLUS, QUERY_SWISH, TAYLOR,
    score_function_family,
)

WIDTHS = (4, 8, 16, 32, 64, 128)
VALUE_WIDTH = 64
MIN_WIDTH = 16  # BF16 tensor-core contraction step on Hopper and Blackwell
MAX_QUERY_DEGREE = 12  # ordinary query families; the GPUs must also fit them (query_degree_supported)
MAX_SOFTPLUS_QUERY_DEGREE = 13  # degree-12 SoS with one input prefactor, at the qualified shapes below


def _query_degree_limit(width, time, transform):
    # Degree 13 is qualified specifically for softplus at production context and the
    # smaller correctness-test context. Keep raw, abs, gated and other shapes unchanged.
    if transform == "softplus" and width == 64 and time in (512, 2048):
        return MAX_SOFTPLUS_QUERY_DEGREE
    return MAX_QUERY_DEGREE


# The kernel family of a forget gate, by gate conditioning (query or query_key) and whether every
# polynomial order has its own gate, in the query-distance mode; in the cumulative mode, only
# per-order query gates have kernels.
DECAY_FAMILY = {("query", False): QUERY_DECAY, ("query_key", False): ENDPOINT_DECAY,
                ("query", True): PER_ORDER_QUERY_DECAY, ("query_key", True): PER_ORDER_ENDPOINT_DECAY}
CUMULATIVE_DECAY_FAMILY = {("query", True): PER_ORDER_CUMULATIVE_DECAY}
# Families qualified on H100 alone: B300 refuses them before building a kernel.
HOPPER_ONLY_FAMILIES = (PER_ORDER_CUMULATIVE_DECAY,)


def query_degree_supported(degree, width, time, *, gate=None, per_order=False, mode="query_distance",
                           transform="identity", prefactor="none"):
    """Whether the query-coefficient kernels run degree m at this key width and sequence length.
    Shared coefficients and gates must fit B300 (``blackwell_supported``), and so H100: B300's
    dK/dV holds (m+1) FP32 coefficients (and a forget gate's vectors and buffers, for gate
    conditioning ``gate``) for every query in shared memory. Per-order gates must fit H100
    (``hopper_supported``), which stages their vectors per tile; B300 runs the per-order shapes
    that ``blackwell_supported`` accepts and refuses the others before building a kernel. A gate's
    decay ``mode`` is "query_distance" or "cumulative"; cumulative gates are supported per order
    with query conditioning alone (``CUMULATIVE_DECAY_FAMILY``), on H100 only. A score
    ``transform`` other than the identity (``TRANSFORM_FAMILY``) or a score ``prefactor`` other than
    "none" (``common.score_function_family``) takes no gate and holds the coefficients alone, as the
    untransformed polynomial does. Softplus alone additionally supports degree 13 at width 64 and
    512/2,048 tokens for a degree-12 SoS prefactor; a prefactor keeps its transform's degree limit."""
    if mode not in ("query_distance", "cumulative"):
        raise ValueError(f"Unknown decay mode: {mode!r}")
    if gate is None:
        family = QUERY
    elif mode == "cumulative":
        family = CUMULATIVE_DECAY_FAMILY.get((gate, per_order))
        if family is None:
            return False
    else:
        family = DECAY_FAMILY[gate, per_order]
    if transform != "identity" or prefactor != "none":
        family = score_function_family(transform, prefactor)
        if gate is not None or family is None:
            return False
    if not (1 <= degree <= _query_degree_limit(width, time, transform) and time > 0):
        return False
    if family in common.PER_ORDER_FAMILIES:
        return hopper_supported(family, degree, width)
    return blackwell_supported(family, degree, width, time)


def blackwell_supported(family, degree, width, time):
    """Whether B300's kernels hold a family's buffers in shared memory: its dK/dV keeps every
    query's FP32 vectors (and a decay family's per-key buffers), and a per-order family's forward
    and dQ also keep every key's per-order vectors, for the whole sequence. Never for a family
    qualified on H100 alone (``HOPPER_ONLY_FAMILIES``)."""
    if family in HOPPER_ONLY_FAMILIES:
        return False
    width = max(width, MIN_WIDTH)
    _, vectors, sums = common.vector_counts(family, degree)
    if common.blackwell_dkv_stages(width, VALUE_WIDTH, vectors, time,
                                   extra_bytes=common.dkv_extra_bytes(family, "dk", degree)) is None:
        return False
    return family not in common.PER_ORDER_FAMILIES or common.blackwell_query_pass_bytes(
        width, VALUE_WIDTH, vectors, sums, common.key_counts(family, degree)[0], time,
        head=common.head_count(family, degree) > 0) <= common.BLACKWELL_SMEM_BYTES


def hopper_supported(family, degree, width):
    """Whether H100's forward, dQ, and dK/dV kernels hold a family's buffers in shared memory at
    any sequence length (``common.hopper_smem_bytes``)."""
    sizes = common.hopper_smem_bytes(family, degree, max(width, MIN_WIDTH), VALUE_WIDTH)
    return max(sizes.values()) <= common.HOPPER_SMEM_BYTES


def _check_blackwell_per_order(family, degree, q, name):
    """B300's per-order kernels hold every query's and key's per-order vectors for the whole
    sequence, so they fit fewer degrees than H100's: refuse a shape they cannot hold before any
    kernel is traced, rather than failing while lowering one."""
    if _kernels()[0] == "blackwell" and not blackwell_supported(family, degree, q.shape[-1], q.shape[1]):
        raise ValueError(f"{name}: degree {degree} at Q/K width {q.shape[-1]} and {q.shape[1]} tokens exceeds "
                         "B300 shared memory; these per-order gates run on H100")


def _check_hopper_only(name):
    """A family qualified on H100 alone (``HOPPER_ONLY_FAMILIES``) is refused on B300 before any
    kernel is traced. Its B300 kernels would also hold every query's and key's per-order vectors,
    fitting no degree above seven at Q/K width 64 and 2,048 tokens."""
    if _kernels()[0] != "hopper":
        raise ValueError(f"{name} runs on H100 only: its kernels are not qualified on B300")


def default_precision(family: str, degree: int) -> str:
    """BF16 weight products, as standard attention kernels use, for every setup but normalized
    odd-degree Taylor. Its signed weights are divided by a row sum that can cancel, which
    amplifies BF16 rounding: with sharp scores its gradients err about twice as much as
    cuDNN softmax's, so it keeps FP32-precision weights."""
    return BF16X3 if family == TAYLOR and degree % 2 else BF16


def _device_compute_capability():
    return str(jax.devices()[0].compute_capability)


def _kernels():
    capability = _device_compute_capability()
    major = int(capability.split(".")[0])
    if capability == "9.0":
        from scaffolds.transformer.attention.backends.mosaic import hopper
        return "hopper", hopper
    if major == 10:
        from scaffolds.transformer.attention.backends.mosaic import blackwell
        return "blackwell", blackwell
    raise ValueError(f"Mosaic GPU attention supports compute capability 9.0 and 10.x, not {capability}")


def _check_operands(q, k, v, name):
    if q.ndim != 4 or q.shape != k.shape or q.shape[:3] != v.shape[:3]:
        raise ValueError(f"{name}: Q and K must share the shape [batch, time, heads, channels], with V matching to channels")
    if not q.dtype == k.dtype == v.dtype == jnp.bfloat16:
        raise ValueError(f"{name} requires BF16 Q/K/V, got {q.dtype}, {k.dtype}, {v.dtype}")
    if q.shape[-1] not in WIDTHS:
        raise ValueError(f"{name} supports Q/K widths {WIDTHS}, got {q.shape[-1]}")
    if v.shape[-1] != VALUE_WIDTH:
        raise ValueError(f"{name} supports value width {VALUE_WIDTH}, got {v.shape[-1]}")
    arch, _ = _kernels()
    multiple = 128 if arch == "hopper" else 256
    if q.shape[1] % multiple:
        raise ValueError(f"{name} on {arch} requires time to be a multiple of {multiple}, got {q.shape[1]}")


def _pad(x):
    width = x.shape[-1]
    return x if width >= MIN_WIDTH else jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, MIN_WIDTH - width)))


def _scale(q):
    return 1.0 / math.sqrt(q.shape[-1])


def _check_precision(precision):
    if precision not in PRECISIONS:
        raise ValueError(f"Mosaic attention precision must be one of {PRECISIONS}, got {precision!r}")


def _backward(family, degree, precision, q, k, v, do, vectors, key_vectors=None, head_constants=None):
    """(dQ, per-query sums or None, dK, dV) at the unpadded width, and per-key sums for
    families that have them."""
    arch, kernels = _kernels()
    width = q.shape[-1]
    args = (_pad(q), _pad(k), v, do, vectors, key_vectors, head_constants)
    options = dict(family=family, degree=degree, scale=_scale(q), precision=precision)
    key_sums = None
    if arch == "hopper":
        grad_q, sums = kernels.dq(*args, **options)
        grad_k, grad_v, *key_sums = kernels.dkv(*args, **options)
    else:
        stages = 2 if width > 64 else 3  # shared memory at width 128
        grad_q, sums = kernels.dq(*args, **options, stages=stages)
        dk_stages = stages
        if family in common.DECAY_FAMILIES:  # their extra vectors and buffers may need a shallower ring
            dk_stages = min(stages, common.blackwell_dkv_stages(
                max(width, MIN_WIDTH), v.shape[-1], common.vector_counts(family, degree)[1], q.shape[1],
                extra_bytes=common.dkv_extra_bytes(family, "dk", degree)))
        grad_k = kernels.dkv(*args, which="dk", **options, stages=dk_stages)
        if isinstance(grad_k, tuple):
            grad_k, *key_sums = grad_k
        grad_v = kernels.dkv(*args, which="dv", **options)
    result = (grad_q[..., :width], sums, grad_k[..., :width], grad_v)
    return (*result, key_sums[0]) if key_sums else result


# ----------------------------------------------------------------------------------------
# Normalized fixed Taylor attention.

def _taylor_forward(q, k, v, degree, precision):
    _, kernels = _kernels()
    out, row_sum = kernels.forward(_pad(q), _pad(k), v, None, family=TAYLOR, degree=degree, scale=_scale(q),
                                   precision=precision)
    return out, row_sum  # row_sum is [batch, heads, time]


@partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def _taylor(q, k, v, degree, precision):
    return _taylor_forward(q, k, v, degree, precision)[0].astype(v.dtype)


def _taylor_fwd(q, k, v, degree, precision):
    out, row_sum = _taylor_forward(q, k, v, degree, precision)
    output = out.astype(v.dtype)
    return output, (q, k, v, output if precision == BF16 else out, row_sum)


def _taylor_bwd(degree, precision, residuals, do):
    q, k, v, out, row_sum = residuals
    delta = jnp.sum(do.astype(jnp.float32) * out.astype(jnp.float32), axis=-1).transpose(0, 2, 1)
    vectors = jnp.stack([delta, 1.0 / row_sum])
    grad_q, _, grad_k, grad_v = _backward(TAYLOR, degree, precision, q, k, v, do, vectors)
    return grad_q, grad_k, grad_v


_taylor.defvjp(_taylor_fwd, _taylor_bwd)


def taylor_attention(q, k, v, *, attention_degree: int, precision: str | None = None):
    """Normalized causal T_m(QK^T / sqrt(d_K)) V for [batch, time, heads, channels] arrays."""
    degree = attention_degree
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError(f"Mosaic Taylor attention requires an integer degree >= 1, got {degree!r}")
    _check_operands(q, k, v, "Mosaic Taylor attention")
    precision = default_precision(TAYLOR, degree) if precision is None else precision
    _check_precision(precision)
    return _taylor(q, k, v, degree, precision)


# ----------------------------------------------------------------------------------------
# Unnormalized attention with per-query polynomial coefficients.

def _coefficient_vectors(coefficients):
    return coefficients.transpose(3, 0, 2, 1)  # [m+1, batch, heads, time]


def _cast_output(output, dtype):
    output = output.astype(dtype)  # as fused_query_polynomial: an overflowing row becomes NaN
    return jnp.where(jnp.all(jnp.isfinite(output), axis=-1, keepdims=True), output, jnp.nan)


def _query_forward(q, k, v, coefficients, precision):
    _, kernels = _kernels()
    degree = coefficients.shape[-1] - 1
    out, _ = kernels.forward(_pad(q), _pad(k), v, _coefficient_vectors(coefficients),
                             family=QUERY, degree=degree, scale=_scale(q), precision=precision)
    return _cast_output(out, v.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(4,))
def _query(q, k, v, coefficients, precision):
    return _query_forward(q, k, v, coefficients, precision)


def _query_fwd(q, k, v, coefficients, precision):
    return _query_forward(q, k, v, coefficients, precision), (q, k, v, coefficients)


def _query_bwd(precision, residuals, do):
    q, k, v, coefficients = residuals
    degree = coefficients.shape[-1] - 1
    grad_q, sums, grad_k, grad_v = _backward(QUERY, degree, precision, q, k, v, do,
                                             _coefficient_vectors(coefficients))
    return grad_q, grad_k, grad_v, sums.transpose(1, 3, 2, 0)


_query.defvjp(_query_fwd, _query_bwd)


def query_polynomial_attention(q, k, v, coefficients, *, precision: str | None = None):
    """Unnormalized causal p_(b,t,h)(QK^T / sqrt(d_K)) V with FP32 coefficients c[B, T, H, m+1]."""
    _check_operands(q, k, v, "Mosaic query-polynomial attention")
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    if (coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3] or coefficients.dtype != jnp.float32
            or not 2 <= coefficients.shape[-1] <= MAX_QUERY_DEGREE + 1):
        raise ValueError("Mosaic query-polynomial attention requires FP32 coefficients [batch, time, heads, m+1] "
                         f"with 1 <= m <= {MAX_QUERY_DEGREE}, got {coefficients.dtype} {coefficients.shape}")
    return _query(q, k, v, coefficients, precision)


# ----------------------------------------------------------------------------------------
# The query polynomial at a transformed score: sum_r c_r z^r with z = phi(s) per causal pair.

# The kernel family of each score transform (attention/score_transform.py) but the identity, which is
# query_polynomial_attention itself; with a prefactor, ``common.score_function_family`` names the family.
TRANSFORM_FAMILY = {"softplus": QUERY_SOFTPLUS, "abs": QUERY_ABS, "sigmoid": QUERY_SIGMOID, "swish": QUERY_SWISH}
# Device admission is narrower than the shared kernel architecture: B200 (sm100) uses the Blackwell
# kernels, but admission does not imply qualification of B300 (sm103). Qualification tests can override
# these allowlists while exercising a new device before its production runs are permitted.
TRANSFORM_ARCHITECTURES = ("hopper", "blackwell")
TRANSFORM_COMPUTE_CAPABILITIES = ("9.0", "10.0")


def transformed_compute_capability_supported(capability: str) -> bool:
    """Whether transformed query-polynomial attention admits this exact GPU capability."""
    return capability in TRANSFORM_COMPUTE_CAPABILITIES


def _transformed_forward(q, k, v, coefficients, family, precision):
    _, kernels = _kernels()
    out, _ = kernels.forward(_pad(q), _pad(k), v, _coefficient_vectors(coefficients), family=family,
                             degree=coefficients.shape[-1] - 1, scale=_scale(q), precision=precision)
    return _cast_output(out, v.dtype)


@partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _transformed_query(q, k, v, coefficients, family, precision):
    return _transformed_forward(q, k, v, coefficients, family, precision)


def _transformed_query_fwd(q, k, v, coefficients, family, precision):
    return _transformed_forward(q, k, v, coefficients, family, precision), (q, k, v, coefficients)


def _transformed_query_bwd(family, precision, residuals, do):
    q, k, v, coefficients = residuals
    grad_q, sums, grad_k, grad_v = _backward(family, coefficients.shape[-1] - 1, precision, q, k, v, do,
                                             _coefficient_vectors(coefficients))
    return grad_q, grad_k, grad_v, sums.transpose(1, 3, 2, 0)


_transformed_query.defvjp(_transformed_query_fwd, _transformed_query_bwd)


def transformed_query_polynomial_attention(q, k, v, coefficients, *, transform: str, prefactor: str = "none",
                                           precision: str | None = None):
    """Unnormalized causal sum_i f(s_ti) p_(b,t,h)(phi(s_ti)) v_i with FP32 coefficients c[B, T, H, m+1], where phi
    is the score transform ``transform``: "softplus" (ln(1 + e^s)), "abs" (|s|), "sigmoid" (1/(1+e^-s)), "swish"
    (s sigmoid(s)), or with a prefactor also "identity" (z = s), and f is the score ``prefactor``: "none" (f = 1),
    or with phi = identity or softplus "softplus", "sigmoid", "exp" (e^s) or "swish", all evaluated per causal pair
    inside the tiles (``common.score_function_family``). Gradients follow the chain rule through phi and the
    product rule through f, with stable sigmoid tail gradients and sign(s) zero at s = 0, and the coefficient
    gradients are dc_r = sum_i dP f(s) z^r. Failure behavior, shapes and precisions are those of
    ``query_polynomial_attention``, on H100 (sm90) and B200 (sm100). B300 (sm103) remains unqualified despite
    sharing the Blackwell kernels. The identity transform without a prefactor is that operation itself, so it
    is refused here."""
    name = "Mosaic transformed query-polynomial attention"
    if transform not in (*TRANSFORM_FAMILY, "identity"):
        raise ValueError(f"{name} takes transform in {('identity', *TRANSFORM_FAMILY)}, got {transform!r}")
    family = score_function_family(transform, prefactor)
    if family is None or family == QUERY:
        raise ValueError(f"{name} has no kernel family for transform {transform!r} with prefactor {prefactor!r}: "
                         f"it takes {sorted(common.SCORE_FUNCTION_FAMILIES)}")
    _check_query_operands(q, k, v, coefficients, name, transform=transform)
    architecture = _kernels()[0]
    if architecture not in TRANSFORM_ARCHITECTURES or (
            architecture == "blackwell" and not transformed_compute_capability_supported(_device_compute_capability())):
        raise ValueError(f"{name} admits H100 (sm90) and B200 (sm100) only; this {architecture} device is not admitted")
    precision = default_precision(family, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    return _transformed_query(q, k, v, coefficients, family, precision)


# ----------------------------------------------------------------------------------------
# Query-distance forget gates: the query polynomial times exp((t-i) l) inside each tile.

def _check_query_operands(q, k, v, coefficients, name, *, transform="identity"):
    _check_operands(q, k, v, name)
    limit = _query_degree_limit(q.shape[-1], q.shape[1], transform)
    if (coefficients.ndim != 4 or coefficients.shape[:3] != q.shape[:3] or coefficients.dtype != jnp.float32
            or not 2 <= coefficients.shape[-1] <= limit + 1):
        raise ValueError(f"{name} requires FP32 coefficients [batch, time, heads, m+1] with 1 <= m <= "
                         f"{limit}, got {coefficients.dtype} {coefficients.shape}")


def _token_vector(x):
    return x.astype(jnp.float32).transpose(0, 2, 1)[None]  # [1, batch, heads, time]


def _decay_forward(family, q, k, v, vectors, key_vectors, precision, degree, head_constants=None):
    _, kernels = _kernels()
    out, _ = kernels.forward(_pad(q), _pad(k), v, vectors, key_vectors, head_constants, family=family,
                             degree=degree, scale=_scale(q), precision=precision)
    return _cast_output(out, v.dtype)


def _query_decay_vectors(coefficients, log_decay):
    # Validate rates per token (the kernels never test them): an invalid one becomes NaN, which
    # poisons every weight of its query. The kernels exponentiate in base 2.
    rate = jnp.where(jnp.isfinite(log_decay) & (log_decay <= 0), log_decay * common.LOG2E, jnp.nan)
    return jnp.concatenate([_coefficient_vectors(coefficients), _token_vector(rate)])


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _query_decay(q, k, v, coefficients, log_decay, precision):
    return _decay_forward(QUERY_DECAY, q, k, v, _query_decay_vectors(coefficients, log_decay), None, precision,
                          coefficients.shape[-1] - 1)


def _query_decay_fwd(q, k, v, coefficients, log_decay, precision):
    return _query_decay(q, k, v, coefficients, log_decay, precision), (q, k, v, coefficients, log_decay)


def _query_decay_bwd(precision, residuals, do):
    q, k, v, coefficients, log_decay = residuals
    degree = coefficients.shape[-1] - 1
    grad_q, sums, grad_k, grad_v = _backward(QUERY_DECAY, degree, precision, q, k, v, do,
                                             _query_decay_vectors(coefficients, log_decay))
    return (grad_q, grad_k, grad_v, sums[:degree + 1].transpose(1, 3, 2, 0),
            sums[degree + 1].transpose(0, 2, 1).astype(log_decay.dtype))


_query_decay.defvjp(_query_decay_fwd, _query_decay_bwd)


def query_decay_attention(q, k, v, coefficients, log_decay, *, precision: str | None = None):
    """Unnormalized causal sum_i p_(b,t,h)(s_ti) exp((t-i) l_(b,t,h)) v_i with FP32 coefficients
    c[B, T, H, m+1] and raw FP32 log rates l[B, T, H] (``forget_gate.log_decay``). A query whose
    rate is nonfinite or positive is NaN."""
    _check_query_operands(q, k, v, coefficients, "Mosaic query-decay attention")
    if log_decay.shape != q.shape[:3] or log_decay.dtype != jnp.float32:
        raise ValueError(f"Mosaic query-decay attention requires FP32 log rates [batch, time, heads], "
                         f"got {log_decay.dtype} {log_decay.shape}")
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    return _query_decay(q, k, v, coefficients, log_decay, precision)


def _finite_or_nan(x):
    # Validate endpoint inputs per token (the kernels never test them): a nonfinite one, even
    # -inf whose softplus would be a valid zero rate, becomes NaN and poisons the pairs it enters.
    return jnp.where(jnp.isfinite(x), x, jnp.nan)


def _endpoint_vectors(coefficients, query_input, rate):
    batch, time = query_input.shape[:2]
    scale = jnp.broadcast_to(rate[None, None, :], (batch, time, rate.shape[0]))
    return jnp.concatenate([_coefficient_vectors(coefficients), _token_vector(_finite_or_nan(query_input)),
                            _token_vector(scale)])


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _endpoint_decay(q, k, v, coefficients, query_input, key_input, rate, precision):
    vectors = _endpoint_vectors(coefficients, query_input, rate)
    return _decay_forward(ENDPOINT_DECAY, q, k, v, vectors, _token_vector(_finite_or_nan(key_input)), precision,
                          coefficients.shape[-1] - 1)


def _endpoint_decay_fwd(q, k, v, coefficients, query_input, key_input, rate, precision):
    output = _endpoint_decay(q, k, v, coefficients, query_input, key_input, rate, precision)
    return output, (q, k, v, coefficients, query_input, key_input, rate)


def _endpoint_decay_bwd(precision, residuals, do):
    q, k, v, coefficients, query_input, key_input, rate = residuals
    degree = coefficients.shape[-1] - 1
    vectors = _endpoint_vectors(coefficients, query_input, rate)
    grad_q, sums, grad_k, grad_v, key_sums = _backward(
        ENDPOINT_DECAY, degree, precision, q, k, v, do, vectors, _token_vector(_finite_or_nan(key_input)))
    # l = a softplus(u + v): dl/du = dl/dv = a sigmoid(u + v), dl/da = softplus(u + v), which the
    # kernels sum as log2(1 + e^x).
    head_rate = rate[None, :, None]
    return (grad_q, grad_k, grad_v, sums[:degree + 1].transpose(1, 3, 2, 0),
            (head_rate * sums[degree + 1]).transpose(0, 2, 1), (head_rate * key_sums[0]).transpose(0, 2, 1),
            common.LN2 * sums[degree + 2].sum(axis=(0, 2)))


_endpoint_decay.defvjp(_endpoint_decay_fwd, _endpoint_decay_bwd)


def endpoint_decay_attention(q, k, v, coefficients, query_gate_input, key_gate_input, gate_log_rate, *,
                             precision: str | None = None):
    """Unnormalized causal sum_i p_(b,t,h)(s_ti) exp(-(t-i) exp(A_h) softplus(u_t + v_i)) v_i with
    FP32 endpoint gate inputs u, v [B, T, H] (``forget_gate.project_gate_inputs``) and log rate
    A [H, 1]. Nonfinite gate inputs or rates make the affected causal rows NaN."""
    _check_query_operands(q, k, v, coefficients, "Mosaic endpoint-decay attention")
    if (query_gate_input.shape != q.shape[:3] or key_gate_input.shape != q.shape[:3]
            or gate_log_rate.shape != (q.shape[2], 1)
            or not query_gate_input.dtype == key_gate_input.dtype == jnp.float32):
        raise ValueError("Mosaic endpoint-decay attention requires FP32 gate inputs [batch, time, heads] "
                         "and a log rate [heads, 1]")
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    log_rate = gate_log_rate[:, 0].astype(jnp.float32)
    # A nonfinite log rate is invalid even where exp would hide it (exp(-inf) = 0).
    rate = jnp.where(jnp.isfinite(log_rate), -jnp.exp(log_rate), jnp.nan)
    return _endpoint_decay(q, k, v, coefficients, query_gate_input, key_gate_input, rate, precision)


# ----------------------------------------------------------------------------------------
# Per-order query-distance gates: sum_r c_r s^r exp((t-i) l_r), one decay per order.

def _order_vectors(x):
    return x.astype(jnp.float32).transpose(3, 0, 2, 1)  # [orders, batch, heads, time]


def _per_order_query_vectors(coefficients, log_decay):
    rate = jnp.where(jnp.isfinite(log_decay) & (log_decay <= 0), log_decay * common.LOG2E, jnp.nan)
    return jnp.concatenate([_coefficient_vectors(coefficients), _order_vectors(rate)])


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _per_order_query_decay(q, k, v, coefficients, log_decay, precision):
    return _decay_forward(PER_ORDER_QUERY_DECAY, q, k, v, _per_order_query_vectors(coefficients, log_decay), None,
                          precision, coefficients.shape[-1] - 1)


def _per_order_query_decay_fwd(q, k, v, coefficients, log_decay, precision):
    output = _per_order_query_decay(q, k, v, coefficients, log_decay, precision)
    return output, (q, k, v, coefficients, log_decay)


def _per_order_query_decay_bwd(precision, residuals, do):
    q, k, v, coefficients, log_decay = residuals
    orders = coefficients.shape[-1]
    grad_q, sums, grad_k, grad_v = _backward(PER_ORDER_QUERY_DECAY, orders - 1, precision, q, k, v, do,
                                             _per_order_query_vectors(coefficients, log_decay))
    return (grad_q, grad_k, grad_v, sums[:orders].transpose(1, 3, 2, 0),
            sums[orders:2 * orders].transpose(1, 3, 2, 0).astype(log_decay.dtype))


_per_order_query_decay.defvjp(_per_order_query_decay_fwd, _per_order_query_decay_bwd)


def per_order_query_decay_attention(q, k, v, coefficients, log_decay, *, precision: str | None = None):
    """Unnormalized causal sum_i sum_r c_(b,t,h,r) s_ti^r exp((t-i) l_(b,t,h,r)) v_i with raw FP32
    per-order log rates l[B, T, H, m+1] (``forget_gate.log_decay`` of per-order parameters). A
    query with any nonfinite or positive rate is NaN."""
    _check_query_operands(q, k, v, coefficients, "Mosaic per-order query-decay attention")
    if log_decay.shape != coefficients.shape or log_decay.dtype != jnp.float32:
        raise ValueError("Mosaic per-order query-decay attention requires FP32 log rates [batch, time, heads, m+1], "
                         f"got {log_decay.dtype} {log_decay.shape}")
    _check_blackwell_per_order(PER_ORDER_QUERY_DECAY, coefficients.shape[-1] - 1, q,
                               "Mosaic per-order query-decay attention")
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    return _per_order_query_decay(q, k, v, coefficients, log_decay, precision)


def _per_order_endpoint_operands(coefficients, query_input, key_input, rate):
    orders = coefficients.shape[-1]
    vectors = jnp.concatenate([_coefficient_vectors(coefficients), _order_vectors(_finite_or_nan(query_input))])
    heads = jnp.pad(rate, ((0, 0), (0, common.head_width(orders) - orders)))
    return vectors, _order_vectors(_finite_or_nan(key_input)), heads


@partial(jax.custom_vjp, nondiff_argnums=(7,))
def _per_order_endpoint_decay(q, k, v, coefficients, query_input, key_input, rate, precision):
    vectors, key_vectors, heads = _per_order_endpoint_operands(coefficients, query_input, key_input, rate)
    return _decay_forward(PER_ORDER_ENDPOINT_DECAY, q, k, v, vectors, key_vectors, precision,
                          coefficients.shape[-1] - 1, heads)


def _per_order_endpoint_decay_fwd(q, k, v, coefficients, query_input, key_input, rate, precision):
    output = _per_order_endpoint_decay(q, k, v, coefficients, query_input, key_input, rate, precision)
    return output, (q, k, v, coefficients, query_input, key_input, rate)


def _per_order_endpoint_decay_bwd(precision, residuals, do):
    q, k, v, coefficients, query_input, key_input, rate = residuals
    orders = coefficients.shape[-1]
    vectors, key_vectors, heads = _per_order_endpoint_operands(coefficients, query_input, key_input, rate)
    grad_q, sums, grad_k, grad_v, key_sums = _backward(PER_ORDER_ENDPOINT_DECAY, orders - 1, precision, q, k, v,
                                                       do, vectors, key_vectors, heads)
    # l_r = a_r softplus(u_r + v_r): dl_r/du_r = dl_r/dv_r = a_r sigmoid(u_r + v_r), dl_r/da_r = softplus,
    # which the kernels sum as log2(1 + e^x).
    order_rate = rate.T[:, None, :, None]  # [orders, 1, heads, 1]
    return (grad_q, grad_k, grad_v, sums[:orders].transpose(1, 3, 2, 0),
            (order_rate * sums[orders:2 * orders]).transpose(1, 3, 2, 0),
            (order_rate * key_sums).transpose(1, 3, 2, 0),
            common.LN2 * sums[2 * orders:].sum(axis=(1, 3)).T)


_per_order_endpoint_decay.defvjp(_per_order_endpoint_decay_fwd, _per_order_endpoint_decay_bwd)


def per_order_endpoint_decay_attention(q, k, v, coefficients, query_gate_input, key_gate_input, gate_log_rate, *,
                                       precision: str | None = None):
    """Unnormalized causal sum_i sum_r c_(b,t,h,r) s_ti^r exp(-(t-i) exp(A_hr) softplus(u_tr + v_ir)) v_i
    with FP32 endpoint gate inputs u, v [B, T, H, m+1] and log rates A [H, m+1, 1]. Nonfinite
    gate inputs or rates make the affected causal rows NaN."""
    _check_query_operands(q, k, v, coefficients, "Mosaic per-order endpoint-decay attention")
    orders = coefficients.shape[-1]
    if (query_gate_input.shape != coefficients.shape or key_gate_input.shape != coefficients.shape
            or gate_log_rate.shape != (q.shape[2], orders, 1)
            or not query_gate_input.dtype == key_gate_input.dtype == jnp.float32):
        raise ValueError("Mosaic per-order endpoint-decay attention requires FP32 gate inputs "
                         "[batch, time, heads, m+1] and log rates [heads, m+1, 1]")
    _check_blackwell_per_order(PER_ORDER_ENDPOINT_DECAY, orders - 1, q, "Mosaic per-order endpoint-decay attention")
    precision = default_precision(QUERY, orders - 1) if precision is None else precision
    _check_precision(precision)
    log_rate = gate_log_rate[..., 0].astype(jnp.float32)
    rate = jnp.where(jnp.isfinite(log_rate), -jnp.exp(log_rate), jnp.nan)  # [heads, orders]
    return _per_order_endpoint_decay(q, k, v, coefficients, query_gate_input, key_gate_input, rate, precision)


# ----------------------------------------------------------------------------------------
# Per-order cumulative gates: sum_r c_r s^r exp(L_tr - L_ir), each order's own cumulative sum.

def _per_order_cumulative_operands(coefficients, log_decay):
    # Validate the sums per token (the kernels never test them): a nonfinite one becomes NaN and
    # poisons every causal pair it enters, as query or as key.
    sums = _order_vectors(_finite_or_nan(log_decay))
    return jnp.concatenate([_coefficient_vectors(coefficients), sums]), sums


@partial(jax.custom_vjp, nondiff_argnums=(5,))
def _per_order_cumulative_decay(q, k, v, coefficients, log_decay, precision):
    vectors, key_vectors = _per_order_cumulative_operands(coefficients, log_decay)
    return _decay_forward(PER_ORDER_CUMULATIVE_DECAY, q, k, v, vectors, key_vectors, precision,
                          coefficients.shape[-1] - 1)


def _per_order_cumulative_decay_fwd(q, k, v, coefficients, log_decay, precision):
    output = _per_order_cumulative_decay(q, k, v, coefficients, log_decay, precision)
    return output, (q, k, v, coefficients, log_decay)


def _per_order_cumulative_decay_bwd(precision, residuals, do):
    q, k, v, coefficients, log_decay = residuals
    orders = coefficients.shape[-1]
    vectors, key_vectors = _per_order_cumulative_operands(coefficients, log_decay)
    grad_q, sums, grad_k, grad_v, key_sums = _backward(PER_ORDER_CUMULATIVE_DECAY, orders - 1, precision, q, k, v,
                                                       do, vectors, key_vectors)
    # L_tr enters every causal pair (t, i) of its query with sign +1 and every pair (t', t) of its key
    # with sign -1: its gradient is c_tr dc_tr less the dK kernel's sum_t' G W_r(t', t).
    grad_coefficients = sums.transpose(1, 3, 2, 0)
    return (grad_q, grad_k, grad_v, grad_coefficients,
            coefficients * grad_coefficients - key_sums.transpose(1, 3, 2, 0))


_per_order_cumulative_decay.defvjp(_per_order_cumulative_decay_fwd, _per_order_cumulative_decay_bwd)


def per_order_cumulative_decay_attention(q, k, v, coefficients, log_decay, *, precision: str | None = None):
    """Unnormalized causal sum_i sum_r c_(b,t,h,r) s_ti^r exp(L_(b,t,h,r) - L_(b,i,h,r)) v_i with each
    order's FP32 cumulative log decay L[B, T, H, m+1] (the cumulative sum over time of
    ``forget_gate.log_decay`` of per-order parameters). A nonfinite L makes every causal pair it
    enters NaN: its own row and the rows of every later query. H100 only: B300 raises ValueError
    before tracing a kernel."""
    name = "Mosaic per-order cumulative-decay attention"
    _check_query_operands(q, k, v, coefficients, name)
    if log_decay.shape != coefficients.shape or log_decay.dtype != jnp.float32:
        raise ValueError(f"{name} requires FP32 cumulative log decays [batch, time, heads, m+1], "
                         f"got {log_decay.dtype} {log_decay.shape}")
    _check_hopper_only(name)
    precision = default_precision(QUERY, coefficients.shape[-1] - 1) if precision is None else precision
    _check_precision(precision)
    return _per_order_cumulative_decay(q, k, v, coefficients, log_decay, precision)

"""Per-degree score biases of tanh query-coefficient polynomial attention.

Order r >= 1 of the polynomial reads its own biased score. With s_ti = Q_t K_i^T / sqrt(d_K),

    W(t,i) = c_0(x_t) G_0(t,i) + sum_(r=1..m) c_r(x_t) G_r(t,i) (s_ti + beta_r(t,i))^r,

where c_r keeps its generator and factorial scales, G_r = 1 or the per-order endpoint gate
exp(-(t-i) exp(A_r) softplus(x_t w_query_r + x_i w_key_r + dt_bias_r)), and order zero is unchanged.
Independently per layer, head, and order r = 1..m, from the attention input x (after the block's
pre-attention RMSNorm):

- ``static``: beta_r = b_r;
- ``query``: beta_r(t) = b_r + a_r tanh(x_t u_r);
- ``query_key``: beta_r(t,i) = b_r + a_r tanh(x_t u_r + x_i v_r).

b starts at zero, a at one, and u and v at zero, so every shift starts at exactly zero while u and
v receive gradients. Nothing is drawn: every other array's initialization is bitwise unchanged.
AdamW decays u and v; b and a never decay. The shifts are signed, and so are the coefficients: a
positive biased score does not imply a positive weight.

Static and query biases enter attention as row shifts beta_r(t) [B, T, H, m], computed here in FP32;
autodiff maps their gradients to b, a and u. Query-key biases enter as the FP32 projections p = x u
and q = x v [B, T, H, m] with the head constants b and a [H, m], and each backend forms a pair's shift
inside its score blocks. Backends evaluate the shifted powers directly: expanding them into
coefficients of unshifted powers would change their cancellation.
"""

import jax
import jax.numpy as jnp

MODES = ("none", "static", "query", "query_key")
OFFSET = "score_bias_offset"
AMPLITUDE = "score_bias_amplitude"
QUERY_KERNEL = "score_bias_query_kernel"
KEY_KERNEL = "score_bias_key_kernel"
ALL_PARAMETERS = (OFFSET, AMPLITUDE, QUERY_KERNEL, KEY_KERNEL)
_NAMES = {"none": (), "static": (OFFSET,), "query": (OFFSET, AMPLITUDE, QUERY_KERNEL), "query_key": ALL_PARAMETERS}
# Attention operands (keywords of attention_core and the backends): a row shift beta_r(t), or the
# pair shift's query and key inputs with the head offset b and amplitude a.
ROW_SHIFT = "score_bias_row_shift"
QUERY_INPUT = "score_bias_query_input"
KEY_INPUT = "score_bias_key_input"
# Diagnostics of z_r = s + beta_r: near-zero threshold, and histogram edges -16, -15, ..., 16 with
# an underflow and an overflow bin.
NEAR_ZERO = 1e-3
HISTOGRAM_EDGES = tuple(range(-16, 17))


def parameter_names(mode: str) -> tuple[str, ...]:
    if not isinstance(mode, str) or mode not in MODES:
        raise ValueError(f"Unknown attention score bias: {mode!r}")
    return _NAMES[mode]


def parameter_specs(*, hidden_size: int, degree: int, mode: str):
    """b [m] at 0 and a [m] at 1 (never decayed); u, v [m, D] at zero (decayed), per layer and head.
    No entry is random, and none is factorial-scaled or order-selected."""
    from scaffolds.transformer.attention.parameters import AttentionParameterSpec

    names = parameter_names(mode)
    if not names:
        return ()
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("Score biases need an integer attention_degree >= 1")
    kernel = dict(initial=(0.0,) * (degree * hidden_size), initial_shape=(degree, hidden_size),
                  sharing="per_head", weight_decay=True)
    specs = {
        OFFSET: AttentionParameterSpec(name=OFFSET, argument=OFFSET, initial=(0.0,) * degree, sharing="per_head",
                                       weight_decay=False),
        AMPLITUDE: AttentionParameterSpec(name=AMPLITUDE, argument=AMPLITUDE, initial=(1.0,) * degree,
                                          sharing="per_head", weight_decay=False),
        QUERY_KERNEL: AttentionParameterSpec(name=QUERY_KERNEL, argument=QUERY_KERNEL, **kernel),
        KEY_KERNEL: AttentionParameterSpec(name=KEY_KERNEL, argument=KEY_KERNEL, **kernel),
    }
    return tuple(specs[name] for name in names)


def project(x, kernel):
    """FP32 x_t u_r [B, T, H, m] from x [B, T, D] and a kernel [H, m, D], at HIGHEST precision."""
    if x.ndim != 3 or kernel.ndim != 3 or kernel.shape[-1] != x.shape[-1]:
        raise ValueError("Score-bias projections require x [B,T,D] and a kernel [H,m,D]")
    return jnp.einsum("btd,hrd->bthr", x.astype(jnp.float32), kernel.astype(jnp.float32),
                      precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)


def attention_operands(mode: str, x, parameters: dict) -> dict:
    """One layer's attention operands from its arrays (b, a [H, m]; u, v [H, m, D]) and x [B, T, D]:
    the row shift beta_r(t) [B, T, H, m] for static and query biases, or for query-key biases the
    inputs p = x u and q = x v [B, T, H, m] with b and a, all FP32."""
    names = parameter_names(mode)
    if set(parameters) != set(names):
        raise ValueError(f"{mode} score biases require exactly their model-owned parameters {list(names)}")
    if not names:
        return {}
    offset = parameters[OFFSET].astype(jnp.float32)
    if offset.ndim != 2 or any(parameters[name].shape[:2] != offset.shape for name in names):
        raise ValueError("Score-bias arrays must have shape [heads, m] (b, a) or [heads, m, D] (u, v)")
    if mode == "static":
        return {ROW_SHIFT: jnp.broadcast_to(offset, (*x.shape[:2], *offset.shape))}
    amplitude = parameters[AMPLITUDE].astype(jnp.float32)
    query = project(x, parameters[QUERY_KERNEL])
    if mode == "query":
        return {ROW_SHIFT: offset + amplitude * jnp.tanh(query)}
    return {QUERY_INPUT: query, KEY_INPUT: project(x, parameters[KEY_KERNEL]), OFFSET: offset, AMPLITUDE: amplitude}


def operand_kind(coefficient_shape, *, row_shift=None, query_input=None, key_input=None, offset=None,
                 amplitude=None) -> str | None:
    """"row", "pair", or None (no score biases), after checking the operands' shapes against query
    coefficients [B, T, H, m+1]: a row shift [B, T, H, m], or the pair inputs [B, T, H, m] with b and
    a [H, m]."""
    pair = (query_input, key_input, offset, amplitude)
    if row_shift is None and all(value is None for value in pair):
        return None
    batch, time, heads, count = coefficient_shape
    shape = (batch, time, heads, count - 1)
    if row_shift is not None:
        if any(value is not None for value in pair):
            raise ValueError("Score biases take a row shift or the query-key pair operands, not both")
        if row_shift.shape != shape:
            raise ValueError(f"A score-bias row shift must have shape [batch,time,heads,m] {shape}, "
                             f"got {row_shift.shape}")
        return "row"
    if any(value is None for value in pair):
        raise ValueError("Query-key score biases require their query and key inputs, offset, and amplitude")
    if (query_input.shape != shape or key_input.shape != shape
            or offset.shape != (heads, count - 1) or amplitude.shape != (heads, count - 1)):
        raise ValueError(f"Query-key score biases require inputs [batch,time,heads,m] {shape} and an offset and "
                         f"amplitude [heads,m] {(heads, count - 1)}")
    return "pair"

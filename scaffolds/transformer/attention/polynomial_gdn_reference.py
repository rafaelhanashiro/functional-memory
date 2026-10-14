"""Exact FP32 tensor-state reference for degree-two/three GDN query transport.

The original key-space transition is ``P_t = I - erase_t k_t k_t.T`` and
``F_t(q) = exp(g_t) F_{t-1}(P_t q) + v_t sum_r beta[t,r] (q.T k_t)**r``.
One scalar decay multiplies every degree. The read scale multiplies the whole
polynomial, as it does the linear read in ordinary GDN.

This reference intentionally materializes each degree's tensor state; production
backends may evaluate the same polynomial through its transported-score form.
"""

import jax
import jax.numpy as jnp


def polynomial_gated_delta_rule_reference(q, k, v, g, beta, erase, scale):
    """Evaluate degrees through two or three, returning FP32 ``[B,T,H,V]`` reads.

    ``q,k`` are ``[B,T,H,K]``, ``v`` is ``[B,T,H,V]``, ``g,erase`` are
    ``[B,T,H]``, and degree-specific writes ``beta`` are ``[B,T,H,3 or 4]``.
    Keys must already have the normalization required by the calling layer.
    All six array arguments are differentiable, including gates at zero.
    """
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("Polynomial GDN requires matching q,k shapes [B,T,H,K]")
    prefix = q.shape[:3]
    if v.ndim != 4 or v.shape[:3] != prefix:
        raise ValueError("Polynomial GDN requires v shape [B,T,H,V]")
    if (g.shape != prefix or erase.shape != prefix or beta.ndim != 4
            or beta.shape[:3] != prefix or beta.shape[-1] not in (3, 4)):
        raise ValueError("Polynomial GDN requires g,erase [B,T,H] and beta [B,T,H,3 or 4]")
    q, k, v, g, beta, erase = (jnp.asarray(a, jnp.float32) for a in (q, k, v, g, beta, erase))
    batch, _, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    initial = (jnp.zeros((batch, heads, value_dim), jnp.float32),
               jnp.zeros((batch, heads, key_dim, value_dim), jnp.float32),
               jnp.zeros((batch, heads, key_dim, key_dim, value_dim), jnp.float32))
    cubic_enabled = beta.shape[-1] == 4
    if cubic_enabled:
        initial += (jnp.zeros((batch, heads, key_dim, key_dim, key_dim, value_dim), jnp.float32),)

    def step(states, inputs):
        constant, linear, quadratic = states[:3]
        qt, kt, vt, gt, bt, et = inputs
        decay = jnp.exp(gt)
        constant = decay[..., None] * constant + bt[..., 0, None] * vt
        prediction = jnp.einsum("bhk,bhkv->bhv", kt, linear)
        linear = decay[..., None, None] * (
            linear - et[..., None, None] * jnp.einsum("bhk,bhv->bhkv", kt, prediction))
        linear += jnp.einsum("bhk,bhv->bhkv", kt, bt[..., 1, None] * vt)
        # Apply P to each key index, never to the value index. Keeping the
        # two contractions separate does not assume exact floating-point symmetry.
        left = jnp.einsum("bhi,bhijv->bhjv", kt, quadratic)
        quadratic -= et[..., None, None, None] * jnp.einsum("bhi,bhjv->bhijv", kt, left)
        right = jnp.einsum("bhijv,bhj->bhiv", quadratic, kt)
        quadratic -= et[..., None, None, None] * jnp.einsum("bhiv,bhj->bhijv", right, kt)
        quadratic = decay[..., None, None, None] * quadratic
        quadratic += jnp.einsum("bhi,bhj,bhv->bhijv", kt, kt, bt[..., 2, None] * vt)
        read = (constant + jnp.einsum("bhk,bhkv->bhv", qt, linear)
                + jnp.einsum("bhi,bhijv,bhj->bhv", qt, quadratic, qt))
        updated = (constant, linear, quadratic)
        if cubic_enabled:
            cubic = states[3]
            # Apply P independently to all three key indices. Alpha still
            # multiplies the whole degree-three state only once.
            first = jnp.einsum("bhi,bhijlv->bhjlv", kt, cubic)
            cubic -= et[..., None, None, None, None] * jnp.einsum("bhi,bhjlv->bhijlv", kt, first)
            second = jnp.einsum("bhijlv,bhj->bhilv", cubic, kt)
            cubic -= et[..., None, None, None, None] * jnp.einsum("bhilv,bhj->bhijlv", second, kt)
            third = jnp.einsum("bhijlv,bhl->bhijv", cubic, kt)
            cubic -= et[..., None, None, None, None] * jnp.einsum("bhijv,bhl->bhijlv", third, kt)
            cubic = decay[..., None, None, None, None] * cubic
            cubic += jnp.einsum("bhi,bhj,bhl,bhv->bhijlv", kt, kt, kt, bt[..., 3, None] * vt)
            read += jnp.einsum("bhi,bhj,bhl,bhijlv->bhv", qt, qt, qt, cubic)
            updated += (cubic,)
        return updated, read * scale

    _, reads = jax.lax.scan(step, initial, tuple(jnp.moveaxis(a, 1, 0) for a in (q, k, v, g, beta, erase)))
    return jnp.moveaxis(reads, 0, 1)

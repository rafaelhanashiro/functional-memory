"""Trace native attention and head RMSNorm, then decompose coefficient gradients by query and key."""
from __future__ import annotations

import math

from flax import nnx
import jax
import jax.numpy as jnp

from scaffolds.transformer.attention.input_dependent import readout_probe
from scaffolds.transformer.attention.multihead import head_output_probe

PRECISION = jax.lax.Precision.HIGHEST


def stored_bias_contributions(q, k, v, h, degree):
    """One layer [batch,time,head,channel]. Return h·M/r! for every query in causal strips."""
    q, k, v, h = (x.astype(jnp.float32).swapaxes(1, 2) for x in (q, k, v, h))
    time = q.shape[2]
    strip = math.gcd(time, 32)
    def rows(start):
        query = jax.lax.dynamic_slice_in_dim(q, start, strip, axis=2)
        grad = jax.lax.dynamic_slice_in_dim(h, start, strip, axis=2)
        score = jnp.einsum("bhqd,bhkd->bhqk", query, k, precision=PRECISION) / math.sqrt(q.shape[-1])
        hv = jnp.einsum("bhqd,bhkd->bhqk", grad, v, precision=PRECISION)
        mask = jnp.arange(time)[None, :] <= (start + jnp.arange(strip))[:, None]
        score = jnp.where(mask, score, 0.)
        term = jnp.where(mask, hv, 0.)
        moments = [term.sum(axis=-1)]
        for r in range(1, degree + 1):
            term = term * score / r
            moments.append(term.sum(axis=-1))
        return jnp.stack(moments, axis=-1)
    pieces = jax.lax.map(rows, jnp.arange(0, time, strip))
    return pieces.transpose(1, 0, 3, 2, 4).reshape(q.shape[0], time, q.shape[1], degree+1)


class HeadTrace:
    def __init__(self, perturbations):
        self.perturbations = perturbations
        self.rows = []
        self.gates = []

    def readout(self, coefficients, gate):
        self.gates.append(gate)
        return coefficients + self.perturbations[1][len(self.gates)-1]

    def before_norm(self, q, k, v, z, operands, gain, x):
        c = operands.get("query_polynomial_coefficients")
        if c is None:
            c = jnp.broadcast_to(operands["polynomial_coefficients"], (*q.shape[:3], operands["polynomial_coefficients"].shape[-1]))
        layer = len(self.rows)
        self.rows.append((q, k, v, z, c, gain, x))
        return z + self.perturbations[0][0, layer].astype(z.dtype)

    def after_norm(self, y):
        return y + self.perturbations[0][1, len(self.rows)-1].astype(y.dtype)


def selected_geometry(capture, h, g, contribution, criterion, monitored, gates, amplitude):
    """Strongest row per monitored layer/order and same-position peer. The microbatch caller merges maxima."""
    q, k, v, z, c, gain, x = capture
    n_sequences, time, heads = q.shape[1:4]
    orders = contribution.shape[-1]
    ranking = jnp.abs(criterion[monitored]).transpose(0, 4, 1, 2, 3).reshape(len(monitored), orders, -1)
    sequence, position, head = jnp.unravel_index(ranking.argmax(axis=-1), (n_sequences, time, heads))
    def pairs(array):
        return jnp.stack((array, array), axis=-1)
    sequence = jnp.stack((sequence, (sequence + 1) % n_sequences), axis=-1)
    position, head = pairs(position), pairs(head)
    layer = jnp.broadcast_to(monitored[:, None, None], head.shape)
    order = jnp.broadcast_to(jnp.arange(orders)[None, :, None], head.shape)
    row = (layer, sequence, position, head)
    # [monitored layer,order,top or peer,keys,channel]
    keys, values = (x.transpose(0, 1, 3, 2, 4)[layer, sequence, head].astype(jnp.float32) for x in (k,v))
    query, grad, incoming, raw = (x[row].astype(jnp.float32) for x in (q,h,g,z))
    scores = jnp.einsum("...d,...kd->...k", query, keys, precision=PRECISION) / math.sqrt(q.shape[-1])
    valid = jnp.arange(time) <= position[..., None]
    scores = jnp.where(valid, scores, 0.)
    coeff = c[row].astype(jnp.float32)
    weight = jnp.zeros_like(scores)
    for r in reversed(range(orders)):
        weight = weight*scores + coeff[..., r, None]
    weight = jnp.where(valid, weight, 0.)
    factorials = jnp.asarray([math.factorial(r) for r in range(orders)], jnp.float32)
    basis = jnp.where(valid, scores ** order[..., None] / factorials[order, None], 0.)
    moment = jnp.einsum("...k,...kd->...d", basis, values, precision=PRECISION)
    hv = jnp.einsum("...d,...kd->...k", grad, values, precision=PRECISION)
    per_key = basis*hv
    mass = jnp.sum(jnp.abs(weight)*jnp.linalg.norm(values, axis=-1), axis=-1)
    reconstructed = jnp.einsum("...k,...kd->...d", weight, values, precision=PRECISION)
    detail = dict(sequence=sequence, position=position, head=head, order=order, layer=layer,
                criterion=criterion[(*row, order)], bias=contribution[(*row, order)], z=raw, coefficients=coeff,
                reconstructed_z=reconstructed, g=incoming, h=grad, gain=gain[layer, head], moment=moment,
                moment_dot_h=jnp.sum(grad*moment, axis=-1), value_mass=mass,
                score=scores, attention_weight=weight, h_dot_v=jnp.where(valid, hv, 0.), key_bias_gradient=per_key)
    if gates is not None:
        u = gates[(*row, order)]
        detail.update(gate=u, amplitude=amplitude[layer, head, order], x=x[layer, sequence, position],
                      kernel_multiplier=contribution[(*row, order)]*amplitude[layer, head, order]*(1-u*u))
    return detail


def make_probe(graph, model):
    from scaffolds.transformer.training.loop import cross_entropy_loss

    query_model = model.attention_coefficient_conditioning == "query"
    degree = model.attention_degree
    layers = model.num_layers
    monitored = jnp.asarray(sorted({0, min(1, layers-1), layers//2, layers-1}))

    def forward(params, tokens, perturbations, batch_fraction):
        trace = HeadTrace(perturbations)
        ctx = head_output_probe.set(trace)
        read = readout_probe.set(trace.readout)
        try:
            logits = nnx.merge(graph, params)(tokens[:, :-1], deterministic=True)
        finally:
            readout_probe.reset(read)
            head_output_probe.reset(ctx)
        if len(trace.rows) != layers or len(trace.gates) != (layers if query_model else 0):
            raise ValueError("Missing polynomial head or gate in the diagnostic trace")
        captured = tuple(jnp.stack(parts) for parts in zip(*trace.rows, strict=True))
        gates = jnp.stack(trace.gates) if query_model else None
        return cross_entropy_loss(logits, tokens[:, 1:])*batch_fraction, (captured, gates)

    differentiate = jax.value_and_grad(forward, argnums=(0, 2), has_aux=True)

    @jax.jit
    def measure(params, tokens, batch_fraction):
        shape = (2, layers, tokens.shape[0], tokens.shape[1]-1, model.num_heads, model.value_head_dim)
        gate_delta = jnp.zeros((*shape[1:-1], degree + 1), jnp.float32) if query_model else None
        (loss, (capture, gates)), (direct, (cotangents, readout_grad)) = differentiate(
            params, tokens, (jnp.zeros(shape), gate_delta), batch_fraction)
        h, g = cotangents
        q,k,v = capture[:3]
        reconstructed = jax.lax.map(lambda args: stored_bias_contributions(*args, degree), (q, k, v, h))
        factorials = jnp.asarray([math.factorial(r) for r in range(degree + 1)], jnp.float32)
        bias = readout_grad / factorials if query_model else reconstructed
        axes = (1, 2)
        contributions = {"bias": bias}
        sums = dict(nll=loss, reconstructed_bias=reconstructed.sum(axes), direct_bias=direct["query_polynomial_bias" if query_model else "polynomial_coefficients"][...])
        amplitude = None
        if query_model:
            amplitude = params["query_polynomial_amplitude"][...]
            x_norm = jnp.linalg.norm(capture[-1].astype(jnp.float32), axis=-1)
            contributions["amplitude"] = bias*gates
            sums.update(reconstructed_amplitude=(reconstructed*gates).sum(axes),
                        direct_amplitude=direct["query_polynomial_amplitude"][...],
                        direct_kernel=direct["query_polynomial_kernel"][...],
                        kernel_abs=(jnp.abs(bias*amplitude[:, None, None]*(1-gates*gates))*x_norm[..., None, None]).sum(axes),
                        gate_mean=gates.mean(axes)*batch_fraction,
                        gate_saturated=(jnp.abs(gates)>.99).mean(axes)*batch_fraction)
        largest, geometry = {}, {}
        for name, value in contributions.items():
            sums[name+"_signed"] = value.sum(axes)
            sums[name+"_abs"] = jnp.abs(value).sum(axes)
            flattened = jnp.abs(value).transpose(0,3,4,1,2).reshape(layers, model.num_heads, degree+1, -1)
            largest[name] = jax.lax.top_k(flattened, min(8, flattened.shape[-1]))[0]
            geometry[name] = selected_geometry(capture, h, g, bias, value, monitored, gates, amplitude)
        return sums, largest, geometry
    return measure

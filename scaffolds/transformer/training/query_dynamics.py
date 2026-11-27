"""Per-coordinate optimizer measurements and signed query gradients for direct tanh polynomials."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.data.packed import PackedBatchIterator

NAMES = ("query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel")


def coordinate_update(before, grads, after):
    """Signed b,a derivatives/displacements and W vector L2 norms, per layer/head/order."""
    arrays = []
    for name in NAMES:
        g = grads[name][...]
        delta = after[name][...] - before[name][...]
        if name == "query_polynomial_kernel":
            g, delta = jnp.linalg.norm(g, axis=-2), jnp.linalg.norm(delta, axis=-2)
        arrays.extend((g, delta))
    return jnp.stack(arrays)


def make_probe(graph, model_config):
    """Per-token cotangent at effective coefficients; compare sums with independent parameter autodiff."""
    from scaffolds.transformer.attention.input_dependent import readout_probe
    from scaffolds.transformer.training.loop import cross_entropy_loss

    layers = model_config.num_layers
    heads = model_config.num_heads
    orders = model_config.attention_degree + 1
    factorial = jnp.asarray([math.factorial(r) for r in range(orders)], jnp.float32)

    def forward(params, tokens, perturbations):
        features = []

        def tap(coefficients, u):
            layer = len(features)
            features.append(u)
            return coefficients + perturbations[layer]

        context = readout_probe.set(tap)
        try:
            model = nnx.merge(graph, params)
            logits = model(tokens[:, :-1], deterministic=True)
        finally:
            readout_probe.reset(context)
        if len(features) != layers:
            raise ValueError(f"Expected {layers} direct coefficient readouts, saw {len(features)}")
        return cross_entropy_loss(logits, tokens[:, 1:]), jnp.stack(features)

    differentiate = jax.value_and_grad(forward, argnums=(0, 2), has_aux=True)

    @jax.jit
    def measure(params, tokens):
        probes = jnp.zeros((layers, tokens.shape[0], tokens.shape[1]-1, heads, orders), jnp.float32)
        (loss, u), (direct, D) = differentiate(params, tokens, probes)
        bias = D / factorial
        amplitude = bias * u
        axes = (1, 2)
        return dict(nll=loss, bias_signed=bias.sum(axes), amplitude_signed=amplitude.sum(axes),
                    bias_abs=jnp.abs(bias).sum(axes), amplitude_abs=jnp.abs(amplitude).sum(axes),
                    bias_peak=jnp.max(jnp.abs(bias), axis=axes),
                    amplitude_peak=jnp.max(jnp.abs(amplitude), axis=axes),
                    gate_mean=u.mean(axes), gate_saturated=(jnp.abs(u) > .99).mean(axes),
                    direct_bias=direct[NAMES[0]][...], direct_amplitude=direct[NAMES[1]][...])
    return measure


def measure_batch(probe, params, batches):
    """All sequences contribute to one mean loss. Maxima also use full-batch loss units."""
    batches = np.asarray(batches)
    tokens = batches.reshape(-1, batches.shape[-1])
    micro = min(4, len(tokens))
    if len(tokens) % micro:
        raise ValueError("Probe microbatch must divide the batch")
    count = len(tokens) // micro
    out = {}
    for start in range(0, len(tokens), micro):
        part = jax.device_get(probe(params, jnp.asarray(tokens[start:start+micro])))
        for name, value in part.items():
            value = np.asarray(value, dtype=np.float64) / count
            if name not in out:
                out[name] = value
            elif name.endswith("_peak"):
                out[name] = np.maximum(out[name], value)
            else:
                out[name] += value
    for key in ("bias", "amplitude"):
        ref = out.pop("direct_" + key)
        signed = out[key + "_signed"]
        error = np.linalg.norm(signed - ref) / max(1.e-15, np.linalg.norm(ref))
        if not np.isfinite(error) or error > .005:
            raise ValueError(f"Per-query contributions failed to reconstruct {key}: {error}")
        out[key + "_relative_error"] = error
    out["batch_sha256"] = hashlib.sha256(np.ascontiguousarray(tokens).tobytes()).hexdigest()
    return out


def save_archive(path, **arrays):
    temp = path.with_suffix(".partial.npz")
    np.savez(temp, **arrays)
    temp.replace(path)


class QueryDynamics:
    """Atomic chunks end at every checkpoint. Replayed updates replace uncommitted measurements."""

    def __init__(self, config, graph, restored, manifest):
        self.workdir = Path(config.train.workdir)
        self.data = config.data
        self.manifest = manifest
        self.probe = make_probe(graph, config.model)
        self.steps = []
        self.updates = []
        for path in self.workdir.glob("query_dynamics_*.npz"):
            if path.name.endswith(".partial.npz") or int(path.stem.rsplit("_", 1)[1]) > restored:
                path.unlink()

    def observe(self, step, update, before, after, batches, *, checkpoint, evaluation, event):
        self.steps.append(step)
        self.updates.append(np.asarray(update))
        if evaluation or step == 1:
            save_archive(self.workdir / f"query_dynamics_parameters_{step:06d}.npz",
                         **{name: np.asarray(after[name][...]) for name in NAMES})
        # Prespecified ordinary batches span initialization, warmup, middle and late training.
        # 6376 is the same batch as the historical replay, even if this rerun doesn't spike on it.
        if event or step in (1, 100, 455, 1500, 4000, 6376, 7500):
            fields = dict(update=step, pre_update_step=step-1, retained_event=event)
            measured = measure_batch(self.probe, before, batches)
            for key in measured:
                fields["train_" + key] = measured[key]
            for i, key in ((0, "bias"), (2, "amplitude")):
                direct = self.updates[-1][i]
                fields[key + "_training_replay_relative_error"] = (
                    np.linalg.norm(direct - measured[key + "_signed"]) / max(1.e-15, np.linalg.norm(direct)))
            if event or step == 6376:
                source = PackedBatchIterator(self.data.train_manifest, batch_size=self.data.batch_size,
                                            sequence_length=self.data.sequence_length,
                                            start_sequence=step*self.data.batch_size, validated_manifest=self.manifest)
                for key, value in measure_batch(self.probe, before, next(source)).items():
                    fields["neighbor_" + key] = value
            save_archive(self.workdir / f"query_dynamics_queries_{step:06d}.npz", **fields)
            print(f"query_dynamics sampled_update={step} retained_event={event}", flush=True)
        if len(self.steps) >= 100 or checkpoint:
            data = np.stack(self.updates, axis=1)
            keys = ("bias_grad", "bias_delta", "amplitude_grad", "amplitude_delta", "kernel_grad_l2", "kernel_delta_l2")
            save_archive(self.workdir / f"query_dynamics_updates_{step:06d}.npz",
                         steps=np.asarray(self.steps), **dict(zip(keys, data, strict=True)))
            self.steps.clear()
            self.updates.clear()

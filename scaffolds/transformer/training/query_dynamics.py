"""Per-coordinate optimizer measurements and gradient decompositions for static and direct query polynomials."""
from __future__ import annotations

import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from scaffolds.transformer.configs.base import packed_data_offset
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.polynomial_gradient_probe import make_probe

NAMES = ("query_polynomial_bias", "query_polynomial_amplitude", "query_polynomial_kernel")


def coordinate_update(before, grads, after):
    """Signed b,a derivatives/displacements and W vector L2 norms, per layer/head/order."""
    arrays = []
    for name in (("polynomial_coefficients",) if "polynomial_coefficients" in grads else NAMES):
        g = grads[name][...]
        delta = after[name][...] - before[name][...]
        if name == "query_polynomial_kernel":
            g, delta = jnp.linalg.norm(g, axis=-2), jnp.linalg.norm(delta, axis=-2)
        arrays.extend((g, delta))
    return jnp.stack(arrays)


def measure_batch(probe, params, batches):
    """Reduce signed and absolute contributions in full-batch gradient units."""
    batches = np.asarray(batches)
    tokens = batches.reshape(-1, batches.shape[-1])
    micro = min(4, len(tokens))
    if len(tokens) % micro:
        raise ValueError("Probe microbatch must divide the batch")
    out, top, winners = {}, {}, {}
    for start in range(0, len(tokens), micro):
        sums, maxima, rows = jax.device_get(probe(params, jnp.asarray(tokens[start:start+micro]), micro/len(tokens)))
        for name, value in sums.items():
            if name not in out:
                out[name] = np.asarray(value, dtype=np.float64).copy()
            else:
                out[name] += value
        for name, largest in maxima.items():
            joined = np.concatenate((top[name], largest), axis=-1) if name in top else largest
            top[name] = np.sort(joined, axis=-1)[..., -8:]
            row = dict(rows[name])
            row["sequence"] = row["sequence"] + start
            if name not in winners:
                winners[name] = row
            else:
                old = winners[name]
                replace = np.abs(row["criterion"][..., 0]) > np.abs(old["criterion"][..., 0])
                for field, value in row.items():
                    old[field] = np.where(replace.reshape(replace.shape + (1,)*(value.ndim-2)), value, old[field])
    for name in top:
        ref = out["direct_" + name]
        signed = out[name + "_signed"]
        error = np.linalg.norm(signed - ref) / max(1.e-15, np.linalg.norm(ref))
        geometry_error = np.linalg.norm(out["reconstructed_" + name] - ref) / max(1.e-15, np.linalg.norm(ref))
        out[name + "_geometry_relative_error"] = geometry_error
        out.update({name + "_relative_error": error, name + "_peak": top[name][..., -1],
                    name + "_top8_abs": top[name].sum(axis=-1)})
        out.update({"rows_" + name + "_" + k: v for k,v in winners[name].items()})
    if "direct_kernel" in out:
        out["kernel_signed_l2"] = np.linalg.norm(out.pop("direct_kernel"), axis=-2)
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
        # A fork's update k reads its parent's stream from data_start_sequence + (k - 1) x batch_size.
        self.first_sequence = packed_data_offset(config.continuation)
        self.manifest = manifest
        self.query = config.model.attention_coefficient_conditioning == "query"
        self.names = NAMES if self.query else ("polynomial_coefficients",)
        self.probe = make_probe(graph, config.model)
        self.steps = []
        self.updates = []
        self.gradient_norms = []
        self.clip_scales = []
        for path in self.workdir.glob("query_dynamics_*.npz"):
            if path.name.endswith(".partial.npz") or int(path.stem.rsplit("_", 1)[1]) > restored:
                path.unlink()

    def observe(self, step, update, before, after, batches, *, gradient_norm, clip_scale, checkpoint, evaluation, event):
        self.steps.append(step)
        self.updates.append(np.asarray(update))
        self.gradient_norms.append(gradient_norm)
        self.clip_scales.append(clip_scale)
        if evaluation or step == 1:
            save_archive(self.workdir / f"query_dynamics_parameters_{step:06d}.npz",
                         **{name: np.asarray(after[name][...]) for name in self.names})
        # Prespecified ordinary batches span initialization, warmup, middle and late training.
        # 6376 is the same batch as the historical replay, even if this rerun doesn't spike on it.
        if event or step in (1, 100, 455, 1500, 4000, 6376, 7500):
            fields = dict(update=step, pre_update_step=step-1, retained_event=event)
            measured = measure_batch(self.probe, before, batches)
            for key in measured:
                fields["train_" + key] = measured[key]
            for i, key in ([(0, "bias"), (2, "amplitude")] if self.query else [(0, "bias")]):
                direct = self.updates[-1][i]
                fields[key + "_training_replay_relative_error"] = (
                    np.linalg.norm(direct - measured[key + "_signed"]) / max(1.e-15, np.linalg.norm(direct)))
            # The historical batch is update 6376 of a stream read from its start, not of a fork's.
            if event or (step == 6376 and not self.first_sequence):
                source = PackedBatchIterator(self.data.train_manifest, batch_size=self.data.batch_size,
                                            sequence_length=self.data.sequence_length,
                                            start_sequence=self.first_sequence + step*self.data.batch_size,
                                            validated_manifest=self.manifest)
                for key, value in measure_batch(self.probe, before, next(source)).items():
                    fields["neighbor_" + key] = value
            save_archive(self.workdir / f"query_dynamics_queries_{step:06d}.npz", **fields)
            print(f"query_dynamics sampled_update={step} retained_event={event}", flush=True)
        if len(self.steps) >= 100 or checkpoint:
            data = np.stack(self.updates, axis=1)
            keys = ("bias_grad", "bias_delta", "amplitude_grad", "amplitude_delta", "kernel_grad_l2", "kernel_delta_l2")
            if not self.query:
                keys = keys[:2]
            save_archive(self.workdir / f"query_dynamics_updates_{step:06d}.npz",
                         steps=np.asarray(self.steps), global_gradient_norm=np.asarray(self.gradient_norms),
                         clip_scale=np.asarray(self.clip_scales), **dict(zip(keys, data, strict=True)))
            self.steps.clear()
            self.updates.clear()
            self.gradient_norms.clear()
            self.clip_scales.clear()

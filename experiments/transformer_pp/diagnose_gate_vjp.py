"""Frozen, same-cotangent attention VJP checks after a batch scan finds a spike.

Only this diagnostic run receives output. Native model/scaffold files and source
checkpoints are read-only. The capture tap preserves the native forward value and
parameter cotangent; its extra argument is solely a cotangent receiver, never trained.
"""
from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import subprocess
from unittest.mock import patch

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from experiments.transformer_pp.diagnose_gate_spikes import (
    gradient_parity, gradient_statistics, serializable, sha256, utc, write_json,
)
from scaffolds.transformer.attention import multihead
from scaffolds.transformer.attention.backends.mosaic.api import per_order_endpoint_decay_attention
from scaffolds.transformer.attention.output_scaling import scale_attention_output
from scaffolds.transformer.configs import Config, ModelConfig, load_config
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.distributed import _loss_and_grads
from scaffolds.transformer.training.loop import cross_entropy_loss
from scaffolds.transformer.training.randomness import model_forward_rngs
from scaffolds.transformer.training.state import create_state


NAMES = ("q", "k", "v", "coefficients", "query_gate_input", "key_gate_input", "gate_log_rate")
GATE_NAMES = ("forget_gate_kernel", "forget_gate_key_kernel", "forget_gate_log_rate", "forget_gate_dt_bias")
COEFFICIENT_NAMES = ("query_polynomial_kernel", "query_polynomial_amplitude", "query_polynomial_bias")


@jax.custom_vjp
def cotangent_tap(value, sink):
    return value


cotangent_tap.defvjp(lambda value, sink: (value, ()), lambda residual, cotangent: (cotangent, cotangent))


def capture_function(graphdef, layer, layers):
    """Trace a temporary alias wrapper; every captured tracer is returned as aux."""
    original = multihead.attention_core

    def loss(params, sink, batch, rng):
        captured, counter = {}, [0]

        def wrapped(q, k, v, **kwargs):
            index = counter[0]
            counter[0] += 1
            output = original(q, k, v, **kwargs)
            if index != layer:
                return output
            if kwargs.get("return_diagnostics", False):
                raise ValueError("Capture requires the native non-diagnostic attention call")
            captured["operands"] = (q, k, v, kwargs["query_polynomial_coefficients"],
                                    kwargs["query_gate_input"], kwargs["key_gate_input"],
                                    kwargs["gate_log_rate"])
            captured["output"] = output
            return cotangent_tap(output, sink)

        with patch.object(multihead, "attention_core", wrapped):
            model = nnx.merge(graphdef, params)
            logits = model(batch[:, :-1], deterministic=False, rngs=model_forward_rngs(rng))
        if counter[0] != layers or not captured:
            raise ValueError("Attention call count does not match the layer selection")
        return cross_entropy_loss(logits, batch[:, 1:]), captured

    return jax.jit(jax.value_and_grad(loss, argnums=(0, 1), has_aux=True))


def implicated_layers(grads, layers):
    result = {}
    for label, names in (("gate", GATE_NAMES), ("coefficient", COEFFICIENT_NAMES)):
        total = jnp.zeros(layers, jnp.float32)
        for name in names:
            if name not in grads:
                continue
            for leaf in jax.tree.leaves(grads[name]):
                if leaf.shape[0] != layers:
                    raise ValueError(f"{name} does not have the expected layer axis")
                total += jnp.sum(jnp.square(leaf.astype(jnp.float32)), axis=tuple(range(1, leaf.ndim)))
        result[label] = jnp.sqrt(total)
    result["gate_and_coefficient"] = jnp.sqrt(result["gate"] ** 2 + result["coefficient"] ** 2)
    return jax.device_get(result)


def group_parity(left, right, layer):
    result = {}
    for label, names in (("gate", GATE_NAMES), ("coefficient", COEFFICIENT_NAMES)):
        for selected in (False, True):
            def subset(tree):
                return {name: (jax.tree.leaves(tree[name])[0][layer] if selected else jax.tree.leaves(tree[name])[0])
                        for name in names if name in tree}
            result[("selected_layer_" if selected else "all_layers_") + label] = jax.device_get(
                gradient_parity(subset(left), subset(right)))
    return result


def native_vjp(operands, cotangent, precision):
    operation = partial(per_order_endpoint_decay_attention, precision=precision)
    output, backward = jax.vjp(operation, *operands)
    return output, backward(cotangent)


native_vjp = jax.jit(native_vjp, static_argnames=("precision",))


def rank_rows(gradient, count=8):
    norms = np.linalg.norm(np.asarray(jax.device_get(gradient), np.float64), axis=-1)
    ranked = np.argsort(norms.ravel())[-count:][::-1]
    return [{"sequence": int(b), "query": int(t), "head": int(h), "cotangent_norm": float(norms[b, t, h])}
            for b, t, h in (np.unravel_index(index, norms.shape) for index in ranked)]


def reference(operands, query, cotangent):
    """Independent NumPy FP64 explicit powers and analytic VJP for one query.

    Operands retain [1,T,1,D/R] API shapes and every key. No native polynomial,
    gate, or backward helper enters this reference; effective c already includes 1/r!.
    """
    q, k, v, c, u, key_input, a = [np.asarray(x, np.float64) for x in operands]
    q, k, v, c, u, key_input = [x[0, :, 0] for x in (q, k, v, c, u, key_input)]
    a = a[0, :, 0]
    incoming = np.asarray(cotangent, np.float64).reshape(-1)
    score = k @ q[query] / math.sqrt(q.shape[-1])
    distance = query - np.arange(len(k))
    causal = distance >= 0
    distance = np.maximum(distance, 0)
    z = u[query] + key_input
    softplus = np.logaddexp(0., z)
    rate = np.exp(a) * softplus
    decay = np.where(causal[:, None], np.exp(-distance[:, None] * rate), 0.)
    powers = np.stack([score ** order for order in range(c.shape[-1])], axis=-1)
    terms = c[query] * powers * decay
    weights = terms.sum(axis=-1)
    output = weights @ v
    dp = v @ incoming
    slope = sum(order * c[query, order] * score ** (order - 1) * decay[:, order]
                for order in range(1, c.shape[-1]))
    ds = dp * slope
    dq, dc, du = np.zeros_like(q), np.zeros_like(c), np.zeros_like(u)
    dq[query] = ds @ k / math.sqrt(q.shape[-1])
    dk = ds[:, None] * q[query] / math.sqrt(q.shape[-1])
    dv = weights[:, None] * incoming
    dc[query] = np.sum(dp[:, None] * powers * decay, axis=0)
    sigmoid = np.exp(-np.logaddexp(0., -z))
    gate = -distance[:, None] * np.exp(a) * sigmoid * terms * dp[:, None]
    du[query] = gate.sum(axis=0)
    da = np.sum(-distance[:, None] * rate * terms * dp[:, None], axis=0)
    grads = tuple(x[None, :, None] for x in (dq, dk, dv, dc, du, gate)) + (da[None, :, None],)
    by_order = terms.T @ v
    denominator = np.linalg.norm(output)
    ratio = lambda numerator: float(numerator / denominator) if denominator else math.inf
    diagnostics = {
        "causal_keys": int(causal.sum()), "reference_pre_norm_rms": float(np.sqrt(np.mean(output ** 2))),
        "cross_degree_cancellation": ratio(np.linalg.norm(by_order, axis=-1).sum()),
        "cross_key_cancellation": ratio(np.sum(np.abs(weights) * np.linalg.norm(v, axis=-1))),
        "all_terms_cancellation": ratio(np.sum(np.abs(terms) * np.linalg.norm(v, axis=-1)[:, None])),
        "order_vector_norms": np.linalg.norm(by_order, axis=-1),
        "order_absolute_weight_sums": np.abs(terms).sum(axis=0),
        "score_min": float(score[causal].min()), "score_max": float(score[causal].max()),
        "score_absolute_p99": float(np.quantile(np.abs(score[causal]), .99)),
        "max_absolute_weight": float(np.abs(weights).max()),
        "max_absolute_score_derivative": float(np.abs(slope).max()),
        "reference_gate_log_rate_vjp": da,
    }
    return output, grads, diagnostics


def comparison(actual, expected):
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    error = actual - expected
    a, b = np.linalg.norm(actual), np.linalg.norm(expected)
    return {"absolute_l2": float(np.linalg.norm(error)), "reference_l2": float(b), "actual_l2": float(a),
            "relative_l2": float(np.linalg.norm(error) / max(b, 1e-30)),
            "max_absolute": float(np.max(np.abs(error))),
            "cosine": float(np.vdot(actual.ravel(), expected.ravel()) / (a * b)) if a and b else None}


def self_test():
    generator = np.random.default_rng(17)
    arrays = [generator.normal(size=shape) * .3 for shape in
              [(1, 7, 1, 3)] * 3 + [(1, 7, 1, 5)] * 3 + [(1, 5, 1)]]
    incoming, query = generator.normal(size=3), 4
    output, gradients, _ = reference(arrays, query, incoming)
    errors = {}
    for index, (name, value, gradient) in enumerate(zip(NAMES, arrays, gradients)):
        direction = generator.normal(size=value.shape)
        direction /= np.linalg.norm(direction)
        plus, minus = list(arrays), list(arrays)
        plus[index], minus[index] = value + 1e-5 * direction, value - 1e-5 * direction
        finite_difference = (reference(plus, query, incoming)[0] - reference(minus, query, incoming)[0]) @ incoming / 2e-5
        analytic = np.vdot(gradient, direction)
        errors[name] = float(abs(finite_difference - analytic))
        np.testing.assert_allclose(finite_difference, analytic, rtol=2e-6, atol=2e-9)
    for index in (1, 2, 5):
        np.testing.assert_array_equal(gradients[index][0, query + 1:], 0.)
    _, self_gradients, _ = reference(arrays, 0, incoming)
    for index in (4, 5, 6):
        np.testing.assert_array_equal(self_gradients[index], 0.)
    x, sink = jnp.arange(3., dtype=jnp.float32), jnp.zeros(3)
    value, (gx, gsink) = jax.value_and_grad(lambda x, s: jnp.sum(cotangent_tap(x, s) ** 2), argnums=(0, 1))(x, sink)
    np.testing.assert_array_equal(gx, 2 * np.asarray(x))
    np.testing.assert_array_equal(gsink, gx)
    np.testing.assert_equal(float(value), 5.)
    config = Config(model=ModelConfig(
        vocab_size=32, max_seq_len=8, num_layers=2, hidden_size=16, num_heads=2, num_kv_heads=2,
        qk_head_dim=8, value_head_dim=8, intermediate_size=32, attention_backend="blocked",
        positional_encoding="none", attention_degree=4, attention_normalization=False,
        attention_coefficients="learned", attention_coefficient_sharing="per_head",
        attention_coefficient_conditioning="query", attention_coefficient_activation="tanh",
        attention_coefficient_projection_initial_std=.25, attention_coefficient_amplitude_scale="factorial",
        attention_coefficient_scale="factorial", attention_forget_gate=True,
        attention_forget_gate_mode="query_distance", attention_forget_gate_conditioning="query_key",
        attention_forget_gate_per_order=True, attention_output_scaling="head_rms_norm",
        qk_norm=True, qk_norm_type="rms_norm", qk_norm_sharing="per_layer", qk_norm_head_scale=True,
        initialization="transformer_truncated_reference"))
    config.data.synthetic_vocab_size, config.data.sequence_length, config.train.dtype = 32, 9, "float32"
    state = create_state(config, jax.random.PRNGKey(42))
    batch, rng = jnp.arange(18, dtype=jnp.int32).reshape(2, 9) % 32, jax.random.PRNGKey(1)
    loss, gradients = jax.jit(lambda p: _loss_and_grads(state.graphdef, p, state.step, batch, rng))(state.params)
    (captured_loss, auxiliary), (captured_gradients, incoming) = capture_function(state.graphdef, 1, 2)(
        state.params, jnp.zeros((2, 8, 2, 8), jnp.float32), batch, rng)
    parity = jax.device_get(gradient_parity(gradients, captured_gradients))
    assert float(parity["relative_l2"]) < 1e-5 and abs(float(loss) - float(captured_loss)) < 2e-6
    assert incoming.shape == (2, 8, config.model.num_heads, config.model.value_head_dim)
    assert auxiliary["operands"][-1].shape == (2, 5, 1)
    assert state.params["output_norm_scale"][...][1].shape == (2, 8)
    assert implicated_layers(gradients, 2)["gate_and_coefficient"].shape == (2,)
    local = group_parity(gradients, captured_gradients, 1)
    assert all(float(value["relative_l2"]) < 1e-5 for value in local.values())
    return {"status": "passed", "finite_difference_absolute_errors": errors,
            "output_finite": bool(np.isfinite(output).all()), "blocked_model_capture_parity": serializable(parity)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), sort_keys=True))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required unless --self-test is used")
    run = args.run_dir.resolve()
    destination = run / "precision_probe.json"
    if destination.exists():
        raise FileExistsError(destination)
    scan = json.loads((run / "summary.json").read_text())
    if scan.get("format") != "frozen_batch_gradient_scan":
        raise ValueError("Expected a frozen batch scan summary")
    report = {"format": "frozen_attention_vjp", "version": 1, "started_utc": utc(),
              "status": "running", "scan_summary_sha256": sha256(run / "summary.json"),
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "limitations": ["Selected rows are informed candidates, not an attribution of the entire global gradient.",
                              "Arithmetic comparisons use identical operands and cotangent; no model is retrained."]}
    if not scan.get("hit_threshold", False):
        report.update(status="skipped", reason="Scan did not exceed its native gradient threshold", finished_utc=utc())
        write_json(destination, report)
        return
    if (scan.get("status") != "complete" or not scan.get("fixed_state_unchanged")
            or not scan.get("source_files_unchanged")):
        report.update(status="scan_validation_failed", reason="Scan did not finish with verified unchanged state and sources",
                      finished_utc=utc())
        write_json(destination, report)
        return
    native_norm = float(scan["largest_batch"]["gradient_norm_pre_clip"])
    repeat, probe = scan["native_repeat"], scan["probe_parity"]
    scan_errors = {
        "repeat_gradient_norm_relative_error": float(repeat["gradient_absolute_error"]) / max(native_norm, 1e-30),
        "repeat_nll_absolute_error": float(repeat["nll_absolute_error"]),
        "plain_vs_native_gradient_norm_relative_error": float(probe["plain_vs_native_gradient_norm_relative_error"]),
        "plain_vs_native_nll_absolute_error": float(probe["plain_vs_native_nll_absolute_error"]),
    }
    scan_passed = all(math.isfinite(value) and value <= (.002 if "nll" in name else .01)
                      for name, value in scan_errors.items())
    report["scan_parity"] = {"passed": scan_passed, "errors": scan_errors,
                             "gradient_norm_relative_limit": .01, "nll_absolute_limit": .002}
    if not scan_passed:
        report.update(status="scan_parity_failed", reason="Native spike not faithfully reproduced by scan attribution probes",
                      finished_utc=utc())
        write_json(destination, report)
        return
    write_json(destination, report)
    try:
        checkpoint = Path(scan["parent_checkpoint_path"]).resolve()
        parent = checkpoint.parents[1]
        if run == parent or run.is_relative_to(parent):
            raise ValueError("Diagnostic output must be separate from parent run")
        config = load_config(parent / "config.yaml")
        if (config.model.dropout or config.model.remat or config.optim.coefficient_penalty
                or config.model.attention_forget_gate_conditioning != "query_key"
                or not config.model.attention_forget_gate_per_order):
            raise ValueError("Probe supports the frozen per-order endpoint recipe without dropout/remat/penalty")
        if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != "gpu":
            raise ValueError("Exactly one GPU is required")
        report["device"] = jax.local_devices()[0].device_kind
        protected = [parent / "config.yaml", parent / "record.json", parent / "metrics.jsonl",
                     checkpoint / "default/_METADATA", run / "summary.json"]
        hashes = {str(path): sha256(path) for path in protected}
        identity = json.loads((parent / "data_identity.json").read_text())
        if sha256(config.data.train_manifest) != identity["train_manifest_sha256"]:
            raise ValueError("Training manifest differs from parent data identity")
        report["source_artifact_sha256"] = hashes
        state = create_state(config, jax.random.PRNGKey(config.train.seed))
        with ocp.StandardCheckpointer() as checkpointer:
            state = checkpointer.restore(checkpoint / "default", target={"state": state})["state"]
        if int(state.step) != scan["state_step"]:
            raise ValueError("Source checkpoint step does not match scan")
        report["restored_state_sha256"] = state_sha256(state)
        if report["restored_state_sha256"] != scan["fixed_state_sha256"]:
            raise ValueError("Restored state differs from the scan's frozen checkpoint")
        graphdef, params, state_step = state.graphdef, state.params, state.step
        del state
        micro_norms = np.asarray(scan["plain_probe"]["microbatches"]["gradients"]["global_norm"])
        micro_index = int(np.argmax(micro_norms))
        largest = scan["largest_batch"]
        first_sequence = int(largest["first_sequence"]) + micro_index * config.train.micro_batch_size
        batch = jnp.asarray(next(PackedBatchIterator(config.data.train_manifest, batch_size=config.train.micro_batch_size,
                                                    sequence_length=config.data.sequence_length, start_sequence=first_sequence)))
        accumulation = config.data.batch_size // config.train.micro_batch_size
        rng = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(config.train.seed),
                                                 int(largest["batch_update"])), accumulation)[micro_index]
        plain = jax.jit(lambda p, b, r: _loss_and_grads(graphdef, p, state_step, b, r))
        plain_loss, plain_grads = plain(params, batch, rng)
        statistics = jax.device_get(gradient_statistics(plain_grads, config.model.num_layers))
        components = implicated_layers(plain_grads, config.model.num_layers)
        layer = int(np.argmax(components["gate_and_coefficient"]))
        if not components["gate_and_coefficient"][layer]:
            layer = int(np.argmax(statistics["layer_norms"]))
        gain = jax.device_get(params["output_norm_scale"][...][layer])
        report["selection"] = {"state_step": int(state_step), "batch_update": largest["batch_update"],
                               "microbatch": micro_index, "first_sequence": first_sequence, "layer": layer,
                               "criterion": "largest gate-plus-coefficient parameter gradient in highest-gradient microbatch",
                               "scan_microbatch_gradient_norm": float(micro_norms[micro_index]),
                               "plain_microbatch_gradient_norm": float(statistics["global_norm"]),
                               "max_total_gradient_layer": int(np.argmax(statistics["layer_norms"])),
                               "layer_gradient_components": components, "all_layer_gradient_norms": statistics["layer_norms"]}
        micro_norm_error = abs(float(statistics["global_norm"]) - float(micro_norms[micro_index])) / max(float(micro_norms[micro_index]), 1e-30)
        micro_loss_error = abs(float(plain_loss) - float(scan["plain_probe"]["microbatches"]["nll"][micro_index]))
        micro_passed = math.isfinite(micro_norm_error) and micro_norm_error <= .01 and micro_loss_error <= .002
        report["microbatch_parity"] = {"passed": micro_passed, "gradient_norm_relative_error": micro_norm_error,
                                       "nll_absolute_error": micro_loss_error, "gradient_norm_relative_limit": .01,
                                       "nll_absolute_limit": .002}
        if not micro_passed:
            report.update(status="microbatch_parity_failed", reason="Standalone microbatch no longer matches the scan probe")
            return
        sink = jnp.zeros((batch.shape[0], batch.shape[1] - 1, config.model.num_heads, config.model.value_head_dim), jnp.bfloat16)
        captured = capture_function(graphdef, layer, config.model.num_layers)
        (captured_loss, auxiliary), (captured_grads, cotangent) = captured(params, sink, batch, rng)
        parity = jax.device_get(gradient_parity(plain_grads, captured_grads))
        groups = group_parity(plain_grads, captured_grads, layer)
        loss_error = abs(float(plain_loss) - float(captured_loss))
        required = (parity, groups["selected_layer_gate"], groups["selected_layer_coefficient"])
        passed = all(math.isfinite(float(item["relative_l2"])) and float(item["relative_l2"]) <= .01
                     for item in required) and loss_error <= .002
        report["capture_parity"] = {"passed": passed, "gradient": parity, "group_gradients": groups,
                                    "nll_absolute_error": loss_error, "gradient_relative_l2_limit": .01,
                                    "nll_absolute_error_limit": .002,
                                    "required_group_checks": ["selected_layer_gate", "selected_layer_coefficient"]}
        if not passed:
            report.update(status="capture_parity_failed", reason="Attribution stopped before attention precision comparisons")
            return
        operands = auxiliary["operands"]
        native_output = auxiliary["output"]
        del captured_grads, plain_grads, params, sink
        isolated_output, isolated_grads = native_vjp(operands, cotangent, "bf16")
        report["isolated_forward_parity"] = comparison(jax.device_get(isolated_output), jax.device_get(native_output))
        isolated_error = report["isolated_forward_parity"]["relative_l2"]
        report["isolated_forward_parity"].update(relative_l2_limit=.01,
                                                passed=math.isfinite(isolated_error) and isolated_error <= .01)
        if not report["isolated_forward_parity"]["passed"]:
            report.update(status="isolated_forward_parity_failed", reason="Attribution stopped before row selection")
            return
        query_rank, coefficient_rank = rank_rows(isolated_grads[4]), rank_rows(isolated_grads[3])
        report["candidate_rankings"] = {"query_gate": query_rank, "coefficient": coefficient_rank}
        candidates = []
        for reason, entry in (("query_gate", query_rank[0]), ("coefficient", coefficient_rank[0])):
            identity = (entry["sequence"], entry["query"], entry["head"])
            existing = next((candidate for candidate in candidates if candidate["identity"] == identity), None)
            if existing is None:
                candidates.append({"identity": identity, "selected_by": [reason]})
            else:
                existing["selected_by"].append(reason)
        del isolated_output, isolated_grads
        saved, row_reports = {}, []
        for number, candidate in enumerate(candidates):
            b, t, h = candidate["identity"]
            selected = tuple(x[b:b + 1, :, h:h + 1] for x in operands[:6]) + (operands[6][h:h + 1],)
            # Preserve the production batch/head layout; only the loss cotangent is restricted.
            selected_cotangent = jnp.zeros_like(cotangent).at[b, t, h].set(cotangent[b, t, h])
            host = tuple(np.asarray(jax.device_get(x), np.float64) for x in selected)
            incoming = np.asarray(jax.device_get(cotangent[b, t, h]), np.float64)
            expected, gradients, diagnostics = reference(host, t, incoming)
            actual = np.asarray(jax.device_get(native_output[b, t, h]), np.float64)
            rms = float(np.sqrt(np.mean(actual ** 2)))
            normalized = scale_attention_output(native_output[b:b + 1, t:t + 1, h:h + 1], "head_rms_norm",
                                                output_norm_scale=jnp.asarray(gain[h:h + 1]), eps=config.model.rms_norm_eps)
            normalized = np.asarray(jax.device_get(normalized), np.float64)
            entry = {"sequence": b, "query": t, "head": h, "selected_by": candidate["selected_by"],
                     "native_pre_norm_rms": rms, "native_normalized_output_rms": float(np.sqrt(np.mean(normalized ** 2))),
                     "norm_reciprocal_rms": 1 / math.sqrt(rms * rms + config.model.rms_norm_eps),
                     "output_cotangent_norm": float(np.linalg.norm(incoming)), "reference": diagnostics, "precision": {}}
            for precision in ("bf16", "bf16x3"):
                output, all_gradients = native_vjp(operands, selected_cotangent, precision)
                got_gradients = tuple(x[b:b + 1, :, h:h + 1] for x in all_gradients[:6]) + (all_gradients[6][h:h + 1],)
                row = np.asarray(jax.device_get(output[b, t, h]), np.float64)
                rounded = np.asarray(jax.device_get(jnp.asarray(expected, jnp.float32).astype(jnp.bfloat16)), np.float64)
                if precision == "bf16":
                    local = comparison(row, actual)
                    local.update(relative_l2_limit=.01, passed=math.isfinite(local["relative_l2"]) and local["relative_l2"] <= .01)
                    entry["selected_forward_parity"] = local
                    entry["attribution_eligible"] = local["passed"]
                entry["precision"][precision] = {
                    "forward": comparison(row, expected), "forward_vs_bf16_rounded_reference": comparison(row, rounded),
                    "gradients": {name: comparison(jax.device_get(got), want)
                                  for name, got, want in zip(NAMES, got_gradients, gradients)},
                }
            prefix = f"candidate_{number}_"
            for name, array in zip(NAMES, host):
                # FP32 losslessly stores the represented BF16/FP32 native operands.
                saved[prefix + name] = array.astype(np.float32)
            saved[prefix + "cotangent"] = incoming.astype(np.float32)
            saved[prefix + "query"] = np.asarray(t, np.int32)
            saved[prefix + "output_norm_scale"] = np.asarray(gain[h], np.float32)
            saved[prefix + "native_output"] = actual.astype(np.float32)
            saved[prefix + "norm_epsilon"] = np.asarray(config.model.rms_norm_eps, np.float32)
            row_reports.append(entry)
        archive = run / "attention_vjp_operands.npz"
        if archive.exists():
            raise FileExistsError(archive)
        np.savez_compressed(archive, **saved)
        report["rows"] = row_reports
        report["operand_archive"] = {"path": str(archive), "sha256": sha256(archive), "size_bytes": archive.stat().st_size,
                                     "purpose": "Preserve selected-row local operands for independent FP64/reference checks; this is not a full-microbatch capture",
                                     "qkv_native_dtype": "bfloat16", "other_operand_native_dtype": "float32",
                                     "layout": "One sequence/head per candidate for independent reference; native VJPs retain the full captured microbatch",
                                     "native_reconstruction": "Restore parent checkpoint and selected batch coordinates, then capture native full-layout operands"}
        report["source_files_unchanged"] = all(sha256(path) == digest for path, digest in hashes.items())
        if not report["source_files_unchanged"]:
            raise RuntimeError("Protected source files changed during the probe")
        report["attribution_eligible"] = all(row["attribution_eligible"] for row in row_reports)
        report["status"] = "complete" if report["attribution_eligible"] else "local_parity_failed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        report["finished_utc"] = utc()
        write_json(destination, report)
    print("PRECISION_PROBE " + json.dumps(serializable({"status": report["status"], "selection": report.get("selection"),
                                                      "capture_parity": report.get("capture_parity")})), flush=True)


if __name__ == "__main__":
    main()

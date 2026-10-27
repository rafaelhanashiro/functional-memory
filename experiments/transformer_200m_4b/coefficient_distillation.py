"""Distill free polynomial coefficients on their actual teacher activations.

The backbone never receives gradients. Fresh validation selects a deployed model;
the final test is evaluated only after selection. Teacher regression and deployed
language-model losses remain separate measurements.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml

from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.data.packed import PackedBatchIterator
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_fit import parent_config, restore_parent, split_generator_parameters
from scaffolds.transformer.training.coefficient_distillation_model import (
    build_dense_student, coefficient_chunks, make_student_losses, make_teacher_functions,
    pack_generator, predict_coefficients, replace_generator,
)
from scaffolds.transformer.training.coefficient_distillation_regression import make_regression_functions


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_batch(spec, sequence_length, checked_shards=None):
    """Verify manifest and every shard actually read by this fixed selection."""
    checked_shards = {} if checked_shards is None else checked_shards
    path = Path(spec["manifest"]).resolve()
    if sha256(path) != spec["manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    manifest = json.loads(path.read_text())
    start, count = spec["start_sequence"], spec["sequence_count"]
    if start < 0 or start + count > manifest["num_sequences"]:
        raise ValueError("Selected sequences exceed the dataset")
    for shard in manifest["shards"]:
        if shard["sequence_start"] < start + count and shard["sequence_start"] + shard["num_sequences"] > start:
            source = (path.parent / shard["path"]).resolve()
            if not source.is_relative_to(path.parent):
                raise ValueError("Dataset shard escapes its directory")
            key = str(source)
            if key not in checked_shards:
                print(f"Verifying selected data shard {source.name}", flush=True)
                checked_shards[key] = sha256(source)
            if checked_shards[key] != shard["sha256"]:
                raise ValueError("Selected dataset shard changed")
    # This iterator touches only the selected shards verified immediately above.
    batch = next(iter(PackedBatchIterator(path, batch_size=count, sequence_length=sequence_length,
        start_sequence=start, validated_manifest=(path, manifest))))
    identity = hashlib.sha256(batch.astype(np.int32).tobytes()).hexdigest()
    if spec.get("batch_sha256") and identity != spec["batch_sha256"]:
        raise ValueError("Selected token batch changed")
    return batch, identity


def scalar_metrics(metrics):
    return {key: float(value) for key, value in jax.device_get(metrics).items()
            if np.asarray(value).ndim == 0 and key != "finite"}


def lm_summary(values):
    values = np.asarray(values, np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite deployed language-model loss")
    return {"nll": float(values.mean()), "perplexity": float(np.exp(values.mean())),
            "per_sequence_nll": values.tolist()}


def evaluate_lm(evaluate, params, batch, microbatch):
    return lm_summary(np.concatenate([np.asarray(evaluate(params, batch[i:i + microbatch]))
                                     for i in range(0, len(batch), microbatch)]))


def full_regression(evaluate, generator, inputs, targets, size):
    totals = {}
    target_energy = 0.
    networks = {}
    for start in range(0, inputs.shape[1], size):
        raw = jax.device_get(evaluate(generator, inputs[:, start:start + size], targets[:, start:start + size]))
        metrics = scalar_metrics(raw)
        weight = min(size, inputs.shape[1] - start)
        target_energy += weight * float(jnp.mean(jnp.square(targets[:, start:start + size])))
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.) + weight * value
        for key in ("factorial_mse_by_network", "cosine_distance_by_network"):
            networks[key] = networks.get(key, 0.) + weight * np.asarray(raw[key])
    result = {key: value / inputs.shape[1] for key, value in totals.items()}
    result["factorial_relative_l2"] = math.sqrt(totals["factorial_mse"] / max(target_energy, 1e-24))
    result.update({key: (value / inputs.shape[1]).tolist() for key, value in networks.items()})
    return result


def finish_completion(run, outcome):
    """Idempotently seal selected-only artifacts after a durable finalization marker."""
    checkpoints = run / "checkpoints"
    selected = checkpoints / f".selected-{outcome['best_step']}"
    destination = checkpoints / str(outcome["best_step"])
    if selected.exists():
        for candidate in checkpoints.iterdir():
            if candidate != selected and candidate.is_dir(): shutil.rmtree(candidate)
        selected.rename(destination)
    if not (destination / "identity.json").is_file():
        raise RuntimeError("Final selected generator checkpoint is unavailable")
    write_json(run / "completion.json", outcome)
    write_json(run / "metadata.json", outcome)
    write_json(run / "retention.json", {"retained": str(destination), "selection": "validation best",
        "capacity_best_retained": False, "teacher_cache_retained": False})
    summary = {"best_validation_nll": outcome["best_validation_nll"],
               "capacity_fit_nll": outcome["capacity_best_fit"]["nll"], "test_nll": outcome["test"]["nll"]}
    update_record(run, {"status": "complete", "step": outcome["step"],
        **capture_artifacts(run, checkpoint_directory=destination), "distillation_outcome": summary})
    cache = run / ".runtime"
    if cache.exists(): shutil.rmtree(cache)


def save_checkpoint(root, step, generator, optimizer, fit_best, validation_best, controller):
    destination = root / str(step)
    temporary = root / f".writing-{step}"
    if temporary.exists():
        shutil.rmtree(temporary)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(temporary, {"generator": generator, "optimizer": optimizer,
            "fit_best": fit_best, "validation_best": validation_best})
    write_json(temporary / "controller.json", controller)
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)
    # The latest resumable checkpoint includes both selected candidates.
    for older in root.iterdir():
        if older.is_dir() and older.name.isdigit() and older != destination:
            shutil.rmtree(older)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(argv)
    run = args.run.resolve()
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    settings = recipe["distillation"]
    pending = run / ".runtime/finalization.json"
    if pending.exists():
        finish_completion(run, json.loads(pending.read_text()))
        print("Finalization completed", flush=True)
        return
    if (run / "completion.json").exists():
        print("Already complete", flush=True)
        return
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("Distillation requires one allocated GPU")
    stopped = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    repository = Path(__file__).resolve().parents[2]
    runtime = {"source_commit": os.environ["DISTILLATION_SOURCE_COMMIT"],
        "device_kind": jax.devices()[0].device_kind, "device_count": 1,
        "resolved_attention_backend": recipe["model"]["attention_backend"],
        "node": os.uname().nodename, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "partition": os.environ.get("SLURM_JOB_PARTITION"), "python_executable": sys.executable,
        "dependency_lock_sha256": sha256(repository / "scaffolds/transformer/uv.lock"),
        "environment": {key: os.environ.get(key) for key in (
            "CUDA_HOME", "CUDA_VISIBLE_DEVICES", "TRANSFORMER_CUDA_ENVIRONMENT", "XLA_FLAGS")}}
    record = load_run(run)
    attempts = record.get("attempts", []) + [{**runtime, "started_utc": utc(),
        "gpu_type": runtime["device_kind"], "gpu_count": 1,
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0"))}]
    update_record(run, {"status": "running", "attempts": attempts, "runtime": runtime})
    for role in ("teacher", "parent"):
        source = settings[role]
        directory = Path(source["run_directory"])
        validate_record(directory, checkpoint_hashes=False)
        if sha256(directory / "config.yaml") != source["config_sha256"]:
            raise ValueError(f"{role} recipe changed")
        if load_run(directory)["selected_checkpoint"]["sha256"] != source["checkpoint_sha256"]:
            raise ValueError(f"{role} checkpoint identity changed")
    config = parent_config(Path(settings["parent"]["run_directory"]) / "config.yaml")
    teacher_recipe = yaml.safe_load((Path(settings["teacher"]["run_directory"]) / "config.yaml").read_text())
    data_identity, batches, checked_shards = {}, {}, {}
    for split in ("fit", "validation"):
        batches[split], identity = read_batch(settings["splits"][split], config.data.sequence_length, checked_shards)
        data_identity[split] = {**settings["splits"][split], "batch_sha256": identity}
    data_identity["test"] = settings["splits"]["test"]
    write_json(run / "data_identity.json", data_identity)
    print("Restoring frozen Taylor parent", flush=True)
    parent_graph, parent_params = restore_parent(config, Path(settings["parent"]["checkpoint"]))
    student_config, student_graph, student_params, backbone_audit = build_dense_student(
        config, parent_params, feature_count=recipe["model"]["attention_coefficient_feature_count"],
        projection_initial_std=recipe["model"]["attention_coefficient_projection_initial_std"],
        readout_scale=recipe["model"]["attention_coefficient_readout_scale"],
        tanh_bias=recipe["model"]["attention_coefficient_tanh_bias"])
    generator = pack_generator(student_params)
    capture, teacher_losses = make_teacher_functions(parent_graph)
    parent_losses = make_student_losses(parent_graph)
    student_losses = make_student_losses(student_graph)
    initialize, train_step, regression_evaluate = make_regression_functions(
        readout_scale=recipe["model"]["attention_coefficient_readout_scale"],
        beta1=settings["beta1"], beta2=settings["beta2"], epsilon=settings["epsilon"],
        clip_norm=settings["clip_norm"], objective="factorial_mse")
    microbatch = settings["lm_microbatch_size"]
    cache = run / ".runtime"
    cache.mkdir(exist_ok=True)
    # These reproducible arrays are disposable allocation-local data. Persistent
    # checkpoints and finalization markers remain under the run directory.
    data_cache = Path(tempfile.gettempdir()) / f"{run.name}-teacher"
    data_cache.mkdir(exist_ok=True)
    dimensions = recipe["model"]
    needed = (len(batches["fit"]) * (config.data.sequence_length - 1) * dimensions["num_layers"]
              * (dimensions["hidden_size"] + dimensions["num_heads"] * (dimensions["attention_degree"] + 1)) * 4)
    if shutil.disk_usage(data_cache).free < needed * 1.1:
        raise RuntimeError("Insufficient allocation-local temporary space for teacher activations")
    runtime["temporary_teacher_cache"] = str(data_cache)
    update_record(run, {"runtime": runtime})
    x_path, y_path = data_cache / "teacher_inputs.npy", data_cache / "teacher_factorial_coefficients.npy"
    acceptance_path = run / "distillation_acceptance.json"
    if not (data_cache / "capture_complete.json").exists():
        dimensions = recipe["model"]
        count, length = len(batches["fit"]), config.data.sequence_length - 1
        x = np.lib.format.open_memmap(x_path, mode="w+", dtype=np.float32,
            shape=(dimensions["num_layers"], count * length, dimensions["hidden_size"]))
        y = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32,
            shape=(dimensions["num_layers"], count * length, dimensions["num_heads"], dimensions["attention_degree"] + 1))
        factors = np.asarray([math.factorial(i) for i in range(dimensions["attention_degree"] + 1)], np.float32)
        native_nll, captured_nll, joint_error = [], [], 0.
        for offset, tables in coefficient_chunks(Path(settings["teacher"]["checkpoint"]), teacher_recipe):
            tokens = batches["fit"][offset:offset + len(tables)]
            native = np.asarray(teacher_losses(parent_params, tokens, tables))
            observed = jax.device_get(capture(parent_params, tokens, tables))
            captured_nll.extend(np.asarray(observed["nll"]).tolist())
            native_nll.extend(native.tolist())
            joint_error = max(joint_error, float(observed["joint_logit_max_absolute_error"]))
            if not np.isfinite(observed["inputs"]).all():
                raise FloatingPointError("Nonfinite teacher activations")
            x[:, offset * length:(offset + len(tables)) * length] = np.asarray(
                observed["inputs"], np.float32).transpose(1, 0, 2, 3).reshape(x.shape[0], -1, x.shape[-1])
            y[:, offset * length:(offset + len(tables)) * length] = (
                tables.transpose(1, 0, 2, 3, 4).reshape(y.shape[0], -1, y.shape[-2], y.shape[-1]) * factors)
            print(f"Captured teacher sequences {offset + len(tables)}/{count}; native PPL={math.exp(float(native.mean())):.6f}", flush=True)
        x.flush(); y.flush()
        del x, y
        discrepancy = np.abs(np.asarray(native_nll) - captured_nll)
        acceptance = {"backbone": backbone_audit, "teacher_native": lm_summary(native_nll),
            "teacher_capture": lm_summary(captured_nll), "joint_logit_max_absolute_error": joint_error,
            "capture_max_sequence_nll_error": float(discrepancy.max()),
            "capture_mean_nll_error": float(np.mean(captured_nll) - np.mean(native_nll)),
            "capture_sequence_nll_tolerance": .002,
            "capture_sequence_parity_passed": bool(np.all(discrepancy <= .002)),
            "teacher_expected_nll": settings["teacher"]["best_nll"],
            "created_utc": utc()}
        write_json(acceptance_path, acceptance)
        if joint_error > .05 or not acceptance["capture_sequence_parity_passed"]:
            raise RuntimeError("Teacher capture changed its forward beyond the established bound")
        if abs(np.mean(native_nll) - settings["teacher"]["best_nll"]) > .002:
            raise RuntimeError("Teacher replay does not reproduce the selected checkpoint")
        write_json(data_cache / "capture_complete.json", {
            "teacher_checkpoint_sha256": settings["teacher"]["checkpoint_sha256"],
            "config_sha256": sha256(run / "config.yaml"),
            "inputs_sha256": sha256(x_path), "targets_sha256": sha256(y_path)})
    capture_identity = json.loads((data_cache / "capture_complete.json").read_text())
    if (capture_identity["config_sha256"] != sha256(run / "config.yaml")
            or capture_identity["teacher_checkpoint_sha256"] != settings["teacher"]["checkpoint_sha256"]
            or capture_identity["inputs_sha256"] != sha256(x_path)
            or capture_identity["targets_sha256"] != sha256(y_path)):
        raise ValueError("Cached teacher data changed")
    print("Placing teacher regression data on GPU", flush=True)
    inputs = jax.device_put(np.load(x_path, mmap_mode="r"))
    targets = jax.device_put(np.load(y_path, mmap_mode="r"))
    inputs.block_until_ready(); targets.block_until_ready()
    optimizer = initialize(generator)
    acceptance = json.loads(acceptance_path.read_text())
    sample = settings["regression_batch_size"]
    # Qualify the actual full-size network, its update, native predictor, and resume.
    g1, o1, m1 = train_step(generator, optimizer, inputs[:, :sample], targets[:, :sample], settings["learning_rate"])
    g2, o2, m2 = train_step(g1, o1, inputs[:, :sample], targets[:, :sample], settings["learning_rate"])
    if not bool(m1["finite"]) or not bool(m2["finite"]) or np.array_equal(np.asarray(g2["kernel"]), np.asarray(generator["kernel"])):
        raise RuntimeError("Regression qualification did not update features after mixing moved")
    production = jax.jit(predict_coefficients, static_argnames=("readout_scale",))(
        g2, inputs[:, :sample], readout_scale=recipe["model"]["attention_coefficient_readout_scale"])
    from scaffolds.transformer.training.coefficient_distillation_regression import predict_factorial
    predicted = jax.jit(predict_factorial, static_argnames=("readout_scale",))(
        g2, inputs[:, :sample], readout_scale=recipe["model"]["attention_coefficient_readout_scale"])
    factors = jnp.asarray([math.factorial(r) for r in range(predicted.shape[-1])], jnp.float32)
    projection_error = float(jnp.max(jnp.abs(predicted - production * factors)))
    if projection_error > 1e-6:
        raise RuntimeError("Regression differs from the native coefficient projection")
    pilot_path = cache / "qualification_checkpoint"
    if pilot_path.exists(): shutil.rmtree(pilot_path)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(pilot_path, {"generator": g1, "optimizer": o1})
        restored = checkpointer.restore(pilot_path, item={"generator": g1, "optimizer": o1})
    rg2, ro2, _ = train_step(restored["generator"], restored["optimizer"], inputs[:, :sample], targets[:, :sample], settings["learning_rate"])
    if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree.leaves((g2, o2)), jax.tree.leaves((rg2, ro2)))):
        raise RuntimeError("Regression checkpoint resume changed the next update")
    shutil.rmtree(pilot_path)
    baseline, initial_student = {}, {}
    for split in ("fit", "validation"):
        baseline[split] = evaluate_lm(parent_losses, parent_params, batches[split], microbatch)
        initial = evaluate_lm(student_losses, student_params, batches[split], microbatch)
        initial_student[split] = initial
        if np.max(np.abs(np.asarray(baseline[split]["per_sequence_nll"]) - initial["per_sequence_nll"])) > .002:
            raise RuntimeError("Taylor-initialized dense student does not reproduce its parent")
    acceptance.update(status="passed", native_projection_max_error=projection_error,
                      resume_next_update_exact=True, baseline=baseline, capture_identity=capture_identity)
    write_json(acceptance_path, acceptance)
    update_record(run, {"acceptance": {"status": "passed", "path": str(acceptance_path)}, "backbone": backbone_audit})
    print("Qualification passed; starting supervised dense fitting", flush=True)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    controller = {"step": 0, "best_validation_nll": initial_student["validation"]["nll"],
        "best_validation_step": 0, "best_fit_mse": float("inf"), "best_fit_step": 0,
        "baseline": baseline}
    fit_best, validation_best = generator, generator
    previous = sorted((p for p in checkpoints.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if previous:
        controller = json.loads((previous[-1] / "controller.json").read_text())
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(previous[-1], item={"generator": generator, "optimizer": optimizer,
                "fit_best": fit_best, "validation_best": validation_best})
        generator, optimizer = restored["generator"], restored["optimizer"]
        fit_best, validation_best = restored["fit_best"], restored["validation_best"]
        if (run / "metrics.jsonl").exists():
            rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= controller["step"]))
    else:
        initial_fit = full_regression(regression_evaluate, generator, inputs, targets, sample)
        controller["best_fit_mse"] = initial_fit["factorial_mse"]
        write_json(run / "initial_metrics.json", {"baseline": baseline, "regression": initial_fit})
    import wandb
    wandb_run = wandb.init(project=recipe["train"]["wandb_project"], entity=recipe["train"]["wandb_entity"],
        group="transformer_200m_4b", name=recipe["train"]["wandb_run_name"], id=run.name, resume="allow",
        config=recipe, dir=str(run), tags=["coefficient-distillation", "dense", "m4", runtime["device_kind"]],
        mode=recipe["train"]["wandb_mode"])
    write_json(run / "wandb_run.json", {"id": run.name, "url": wandb_run.url})
    update_record(run, {"wandb": {"id": run.name, "url": wandb_run.url}})
    total, start = settings["updates"], int(controller["step"])
    key = jax.random.key(recipe["train"]["seed"])
    # The whole teacher dataset remains resident: only a small indexed batch enters each regression update.
    @jax.jit
    def sampled_step(g, opt, x, y, step, lr):
        indices = jax.random.randint(jax.random.fold_in(key, step), (sample,), 0, x.shape[1])
        return train_step(g, opt, x[:, indices], y[:, indices], lr)
    with (run / "metrics.jsonl").open("a", buffering=1) as stream:
        for step in range(start + 1, total + 1):
            began = time.monotonic()
            lr = settings["min_learning_rate"] + .5 * (settings["learning_rate"] - settings["min_learning_rate"]) * (1 + math.cos(math.pi * (step - 1) / max(total - 1, 1)))
            generator, optimizer, measurement = sampled_step(generator, optimizer, inputs, targets, step, lr)
            if not bool(measurement["finite"]):
                write_json(run / f"numerical_failure_step_{step:08d}.json", {"step": step, "status": "numerical_failure", "reason": "nonfinite regression update"})
                update_record(run, {"status": "numerical_failure", "step": step})
                raise FloatingPointError("Nonfinite regression update")
            controller["step"] = step
            if step % settings["log_every"] == 0 or step == start + 1:
                row = {"event": "regression", "step": step, "learning_rate": lr,
                       "seconds": time.monotonic() - began, **scalar_metrics(measurement)}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                wandb_run.log({f"train/{name}": value for name, value in row.items() if isinstance(value, (float, int))}, step=step,
                              commit=not (step % settings["eval_every"] == 0 or step == total))
                print(f"step={step} factorial_mse={row['factorial_mse']:.6f} angle={row['angle_degrees']:.2f} lr={lr:.6g} seconds={row['seconds']:.3f}", flush=True)
            if step % settings["eval_every"] == 0 or step == total:
                update_record(run, {"step": step})
                reconstruction = full_regression(regression_evaluate, generator, inputs, targets, sample)
                deployed = replace_generator(student_params, generator)
                fit = evaluate_lm(student_losses, deployed, batches["fit"], microbatch)
                validation = evaluate_lm(student_losses, deployed, batches["validation"], microbatch)
                if reconstruction["factorial_mse"] < controller["best_fit_mse"]:
                    controller.update(best_fit_mse=reconstruction["factorial_mse"], best_fit_step=step)
                    fit_best = generator
                if validation["nll"] < controller["best_validation_nll"]:
                    controller.update(best_validation_nll=validation["nll"], best_validation_step=step)
                    validation_best = generator
                row = {"event": "evaluation", "step": step, "regression": reconstruction,
                       "fit": fit, "validation": validation,
                       "best_validation_step": controller["best_validation_step"]}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                wandb_run.log({"val/nll": validation["nll"], "val/perplexity": validation["perplexity"],
                    "train/fit_perplexity": fit["perplexity"], "train/full_factorial_mse": reconstruction["factorial_mse"]}, step=step)
                write_json(run / "metadata.json", {"status": "running", **controller, "latest": row})
                write_json(run / "best_validation.json", {"step": controller["best_validation_step"],
                    "nll": controller["best_validation_nll"], "perplexity": math.exp(controller["best_validation_nll"])})
                update_record(run, {"status": "running", "step": step, "best_validation_nll": controller["best_validation_nll"]})
                print(f"Evaluation step={step}: fit PPL={fit['perplexity']:.5f}, validation PPL={validation['perplexity']:.5f}, best validation step={controller['best_validation_step']}", flush=True)
            if step % settings["checkpoint_every"] == 0 or step == total or stopped[0]:
                save_checkpoint(checkpoints, step, generator, optimizer, fit_best, validation_best, controller)
            if stopped[0]:
                update_record(run, {"status": "paused", "step": step})
                wandb_run.finish()
                return
    # A requeue after the last update may resume directly into finalization.
    if start == total:
        reconstruction = full_regression(regression_evaluate, generator, inputs, targets, sample)
        deployed = replace_generator(student_params, generator)
        fit = evaluate_lm(student_losses, deployed, batches["fit"], microbatch)
        validation = evaluate_lm(student_losses, deployed, batches["validation"], microbatch)
    selected = replace_generator(student_params, validation_best)
    capacity = replace_generator(student_params, fit_best)
    if state_sha256(split_generator_parameters(selected)[1]) != backbone_audit["backbone_parameters_sha256"]:
        raise RuntimeError("Distillation changed frozen backbone parameters")
    outcome = {"status": "complete", "step": total, "best_step": controller["best_validation_step"],
        "best_validation_nll": controller["best_validation_nll"], "best_regression_step": controller["best_fit_step"],
        "best_regression_mse": controller["best_fit_mse"],
        "capacity_best_fit": evaluate_lm(student_losses, capacity, batches["fit"], microbatch),
        "validation_best_fit": evaluate_lm(student_losses, selected, batches["fit"], microbatch),
        "validation_best_validation": evaluate_lm(student_losses, selected, batches["validation"], microbatch),
        "teacher_fit": acceptance["teacher_native"], "baseline": baseline,
        "terminal_regression": reconstruction, "terminal_fit": fit, "terminal_validation": validation,
        "finished_utc": utc()}
    # The final test enters no optimizer, stopping decision, or checkpoint selection.
    test, identity = read_batch(settings["splits"]["test"], config.data.sequence_length, checked_shards)
    outcome["test"] = evaluate_lm(student_losses, selected, test, microbatch)
    outcome["baseline"]["test"] = evaluate_lm(parent_losses, parent_params, test, microbatch)
    data_identity["test"] = {**settings["splits"]["test"], "batch_sha256": identity}
    write_json(run / "data_identity.json", data_identity)
    final_path = checkpoints / f".selected-{controller['best_validation_step']}"
    if final_path.exists(): shutil.rmtree(final_path)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(final_path, {"generator": validation_best})
    write_json(final_path / "identity.json", {"step": controller["best_validation_step"],
        "backbone": settings["parent"], "selection": "lowest fresh validation language-model NLL, including Taylor initialization"})
    write_json(cache / "finalization.json", outcome)
    finish_completion(run, outcome)
    wandb_run.summary.update({"test_perplexity": outcome["test"]["perplexity"],
        "baseline_test_perplexity": outcome["baseline"]["test"]["perplexity"],
        "capacity_best_fit_perplexity": outcome["capacity_best_fit"]["perplexity"]})
    wandb_run.finish()
    print(json.dumps({"complete": run.name, "capacity_fit_ppl": outcome["capacity_best_fit"]["perplexity"],
        "test_ppl": outcome["test"]["perplexity"], "baseline_test_ppl": outcome["baseline"]["test"]["perplexity"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except FloatingPointError as error:
        run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve()
        record = load_run(run)
        if record.get("status") != "numerical_failure":
            step = int(record.get("step", 0))
            write_json(run / f"numerical_failure_step_{step:08d}.json", {
                "step": step, "status": "numerical_failure", "reason": str(error), "time_utc": utc()})
            update_record(run, {"status": "numerical_failure", "step": step})
        raise

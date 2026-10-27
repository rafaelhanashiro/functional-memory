"""Distill free query-key pair coefficients into the native query-key generator.

The backbone never receives gradients. Training regresses the generator's pair
coefficients onto the free teacher's c_r(t) g_r(i) at causal pairs of the teacher's
fitting batch, using the teacher's own attention inputs. The result is the snapshot
with the lowest pair-regression error on a fixed sample of the fitting batch's pairs;
after training, it and the Taylor baseline are scored once on one fresh held-out batch,
which enters no optimizer, stopping, or selection decision.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import yaml

from experiments.transformer_200m_4b.coefficient_distillation import (
    evaluate_lm, lm_summary, read_batch, scalar_metrics, sha256, utc, write_json,
)
from scaffolds.records import capture_artifacts, load_run, update_record, validate_record
from scaffolds.transformer.training.checkpointing import state_sha256
from scaffolds.transformer.training.coefficient_distillation_model import make_student_losses
from scaffolds.transformer.training.coefficient_fit import parent_config, restore_parent, split_generator_parameters
from scaffolds.transformer.training.query_key_distillation import (
    build_student, make_regression_functions, make_teacher_functions, pack_generator,
    query_key_table_chunks, replace_generator, sample_pairs,
)


def pair_regression(evaluate, generator, inputs, query_factors, key_factors, pairs, size):
    """Mean regression metrics over the fixed evaluation pairs, in chunks of ``size``."""
    queries, keys = pairs
    totals, energy, count = {}, 0., int(queries.shape[0])
    for start in range(0, count, size):
        q, k = queries[start:start + size], keys[start:start + size]
        targets = query_factors[:, q] * key_factors[:, k]
        metrics = scalar_metrics(evaluate(generator, inputs[:, q], inputs[:, k], targets))
        weight = int(q.shape[0])
        energy += weight * float(jnp.mean(jnp.square(targets)))
        for name, value in metrics.items():
            totals[name] = totals.get(name, 0.) + weight * value
    result = {name: value / count for name, value in totals.items()}
    result["factorial_relative_l2"] = math.sqrt(totals["factorial_mse"] / max(energy, 1e-24))
    return result


def finish_completion(run, outcome):
    """Idempotently seal the selected generator after a durable finalization marker."""
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
    write_json(run / "retention.json", {"retained": str(destination),
        "selection": "lowest pair-regression error on the fitting batch's fixed evaluation pairs",
        "teacher_cache_retained": False})
    summary = {"selected_step": outcome["best_step"], "selected_fit_nll": outcome["selected_fit"]["nll"],
               "selected_heldout_nll": outcome["selected_heldout"]["nll"],
               "baseline_heldout_nll": outcome["baseline"]["heldout"]["nll"]}
    update_record(run, {"status": "complete", "step": outcome["step"],
        **capture_artifacts(run, checkpoint_directory=destination), "distillation_outcome": summary})
    cache = run / ".runtime"
    if cache.exists(): shutil.rmtree(cache)


def save_checkpoint(root, step, generator, optimizer, best, controller):
    destination, temporary = root / str(step), root / f".writing-{step}"
    if temporary.exists(): shutil.rmtree(temporary)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(temporary, {"generator": generator, "optimizer": optimizer, "best": best})
    write_json(temporary / "controller.json", controller)
    if destination.exists(): shutil.rmtree(destination)
    temporary.rename(destination)
    for older in root.iterdir():
        if older.is_dir() and older.name.isdigit() and older != destination:
            shutil.rmtree(older)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args(argv).run.resolve()
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
        "resolved_attention_backend": settings["backend"], "node": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "python_executable": sys.executable,
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
    for split in ("fit", "heldout"):
        batches[split], identity = read_batch(settings["splits"][split], config.data.sequence_length, checked_shards)
        data_identity[split] = {**settings["splits"][split], "batch_sha256": identity}
    write_json(run / "data_identity.json", data_identity)
    print("Restoring frozen Taylor parent", flush=True)
    parent_graph, parent_params = restore_parent(config, Path(settings["parent"]["checkpoint"]))
    # The teacher runs the parent's parameters with the query-key kernel backend.
    teacher_graph, teacher_params = restore_parent(
        replace(config, model=replace(config.model, attention_backend=settings["backend"])),
        Path(settings["parent"]["checkpoint"]))
    if state_sha256(teacher_params) != state_sha256(parent_params):
        raise RuntimeError("Teacher graph parameters differ from the parent's")
    del teacher_params
    _, student_graph, student_params, backbone_audit = build_student(config, parent_params, backend=settings["backend"])
    generator = pack_generator(student_params)
    capture, teacher_losses = make_teacher_functions(teacher_graph)
    parent_losses, student_losses = make_student_losses(parent_graph), make_student_losses(student_graph)
    initialize, train_step, regression_evaluate = make_regression_functions(
        beta1=settings["beta1"], beta2=settings["beta2"], epsilon=settings["epsilon"], clip_norm=settings["clip_norm"])
    microbatch = settings["lm_microbatch_size"]
    cache = run / ".runtime"
    cache.mkdir(exist_ok=True)
    # Reproducible teacher arrays are disposable allocation-local data.
    data_cache = Path(tempfile.gettempdir()) / f"{run.name}-teacher"
    data_cache.mkdir(exist_ok=True)
    dimensions = recipe["model"]
    layers, heads, orders = dimensions["num_layers"], dimensions["num_heads"], dimensions["attention_degree"] + 1
    count, length = len(batches["fit"]), config.data.sequence_length - 1
    needed = count * length * layers * (dimensions["hidden_size"] + 2 * heads * orders) * 4
    if shutil.disk_usage(data_cache).free < needed * 1.1:
        raise RuntimeError("Insufficient allocation-local temporary space for teacher activations")
    runtime["temporary_teacher_cache"] = str(data_cache)
    update_record(run, {"runtime": runtime})
    paths = {name: data_cache / f"teacher_{name}.npy" for name in ("inputs", "query_factorial", "key_factors")}
    acceptance_path = run / "distillation_acceptance.json"
    if not (data_cache / "capture_complete.json").exists():
        shapes = {"inputs": (layers, count * length, dimensions["hidden_size"]),
                  "query_factorial": (layers, count * length, heads, orders),
                  "key_factors": (layers, count * length, heads, orders)}
        arrays = {name: np.lib.format.open_memmap(paths[name], mode="w+", dtype=np.float32, shape=shape)
                  for name, shape in shapes.items()}
        factors = np.asarray([math.factorial(r) for r in range(orders)], np.float32)
        native_nll, captured_nll, joint_error = [], [], 0.
        for offset, coefficients, key_factors in query_key_table_chunks(Path(settings["teacher"]["checkpoint"]), teacher_recipe):
            tokens = batches["fit"][offset:offset + len(coefficients)]
            native = np.asarray(teacher_losses(parent_params, tokens, coefficients, key_factors))
            observed = jax.device_get(capture(parent_params, tokens, coefficients, key_factors))
            captured_nll.extend(np.asarray(observed["nll"]).tolist())
            native_nll.extend(native.tolist())
            joint_error = max(joint_error, float(observed["joint_logit_max_absolute_error"]))
            if not np.isfinite(observed["inputs"]).all():
                raise FloatingPointError("Nonfinite teacher activations")
            rows = slice(offset * length, (offset + len(coefficients)) * length)
            arrays["inputs"][:, rows] = np.asarray(observed["inputs"], np.float32).transpose(1, 0, 2, 3).reshape(
                layers, -1, dimensions["hidden_size"])
            arrays["query_factorial"][:, rows] = (coefficients.transpose(1, 0, 2, 3, 4) * factors).reshape(layers, -1, heads, orders)
            arrays["key_factors"][:, rows] = key_factors.transpose(1, 0, 2, 3, 4).reshape(layers, -1, heads, orders)
            print(f"Captured teacher sequences {offset + len(coefficients)}/{count}; native PPL={math.exp(float(native.mean())):.6f}", flush=True)
        for array in arrays.values():
            array.flush()
        del arrays
        discrepancy = np.abs(np.asarray(native_nll) - captured_nll)
        acceptance = {"backbone": backbone_audit, "teacher_native": lm_summary(native_nll),
            "teacher_capture": lm_summary(captured_nll), "joint_logit_max_absolute_error": joint_error,
            "capture_max_sequence_nll_error": float(discrepancy.max()), "capture_sequence_nll_tolerance": .002,
            "capture_sequence_parity_passed": bool(np.all(discrepancy <= .002)),
            "teacher_expected_nll": settings["teacher"]["best_nll"], "created_utc": utc()}
        write_json(acceptance_path, acceptance)
        if joint_error > .05 or not acceptance["capture_sequence_parity_passed"]:
            raise RuntimeError("Teacher capture changed its forward beyond the established bound")
        if abs(np.mean(native_nll) - settings["teacher"]["best_nll"]) > .002:
            raise RuntimeError("Teacher replay does not reproduce the selected checkpoint")
        write_json(data_cache / "capture_complete.json", {
            "teacher_checkpoint_sha256": settings["teacher"]["checkpoint_sha256"],
            "config_sha256": sha256(run / "config.yaml"), **{f"{name}_sha256": sha256(path) for name, path in paths.items()}})
    identity = json.loads((data_cache / "capture_complete.json").read_text())
    if (identity["config_sha256"] != sha256(run / "config.yaml")
            or identity["teacher_checkpoint_sha256"] != settings["teacher"]["checkpoint_sha256"]
            or any(identity[f"{name}_sha256"] != sha256(path) for name, path in paths.items())):
        raise ValueError("Cached teacher data changed")
    print("Placing teacher regression data on GPU", flush=True)
    inputs, query_factors, key_factors = (jax.device_put(np.load(paths[name], mmap_mode="r"))
                                          for name in ("inputs", "query_factorial", "key_factors"))
    for array in (inputs, query_factors, key_factors):
        array.block_until_ready()
    acceptance = json.loads(acceptance_path.read_text())
    size, total = settings["regression_pairs_per_update"], settings["updates"]
    key = jax.random.key(recipe["train"]["seed"])

    @jax.jit
    def sampled_step(g, opt, x, u, f, step, lr):
        queries, keys = sample_pairs(jax.random.fold_in(key, step), sequence_count=count, length=length, size=size)
        return train_step(g, opt, x[:, queries], x[:, keys], u[:, queries] * f[:, keys], lr)

    # The fixed evaluation pairs use a stream that no training update folds in.
    evaluation_pairs = jax.device_get(sample_pairs(jax.random.fold_in(key, 2**31 - 1), sequence_count=count,
                                                   length=length, size=settings["evaluation_pairs"]))
    evaluation_pairs = tuple(jax.device_put(np.asarray(value)) for value in evaluation_pairs)
    # Qualify the student's Taylor start, its update path, and exact resume.
    baseline = {}
    for split in ("fit", "heldout"):
        baseline[split] = evaluate_lm(parent_losses, parent_params, batches[split], microbatch)
        initial = evaluate_lm(student_losses, student_params, batches[split], microbatch)
        if np.max(np.abs(np.asarray(baseline[split]["per_sequence_nll"]) - initial["per_sequence_nll"])) > .002:
            raise RuntimeError("Taylor-initialized query-key student does not reproduce its parent")
    optimizer = initialize(generator)
    g1, o1, m1 = sampled_step(generator, optimizer, inputs, query_factors, key_factors, 1, settings["learning_rate"])
    g2, o2, m2 = sampled_step(g1, o1, inputs, query_factors, key_factors, 2, settings["learning_rate"])
    moved = {name: bool(jnp.any(g2[name] != generator[name])) for name in generator}
    if not bool(m1["finite"]) or not bool(m2["finite"]) or not all(moved.values()):
        raise RuntimeError(f"Regression qualification did not move every generator group: {moved}")
    pilot_path = cache / "qualification_checkpoint"
    if pilot_path.exists(): shutil.rmtree(pilot_path)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(pilot_path, {"generator": g1, "optimizer": o1})
        restored = checkpointer.restore(pilot_path, item={"generator": g1, "optimizer": o1})
    rg2, ro2, _ = sampled_step(restored["generator"], restored["optimizer"], inputs, query_factors, key_factors, 2,
                               settings["learning_rate"])
    if not all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(jax.tree.leaves((g2, o2)), jax.tree.leaves((rg2, ro2)))):
        raise RuntimeError("Regression checkpoint resume changed the next update")
    shutil.rmtree(pilot_path)
    acceptance.update(status="passed", baseline=baseline, generator_groups_moved=moved, resume_next_update_exact=True,
                      student_parent_parity_tolerance=.002)
    write_json(acceptance_path, acceptance)
    update_record(run, {"acceptance": {"status": "passed", "path": str(acceptance_path)}, "backbone": backbone_audit})
    print("Qualification passed; starting pair regression", flush=True)
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    initial_regression = pair_regression(regression_evaluate, generator, inputs, query_factors, key_factors,
                                         evaluation_pairs, size)
    controller = {"step": 0, "best_mse": initial_regression["factorial_mse"], "best_step": 0, "baseline": baseline}
    best = generator
    previous = sorted((p for p in checkpoints.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
    if previous:
        controller = json.loads((previous[-1] / "controller.json").read_text())
        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(previous[-1], item={"generator": generator, "optimizer": optimizer, "best": best})
        generator, optimizer, best = restored["generator"], restored["optimizer"], restored["best"]
        if (run / "metrics.jsonl").exists():
            rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
            (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= controller["step"]))
    else:
        write_json(run / "initial_metrics.json", {"baseline": baseline, "regression": initial_regression})
    import wandb
    wandb_run = wandb.init(project=recipe["train"]["wandb_project"], entity=recipe["train"]["wandb_entity"],
        group="transformer_200m_4b", name=recipe["train"]["wandb_run_name"], id=run.name, resume="allow",
        config=recipe, dir=str(run), tags=["coefficient-distillation", "query_key", "m4", runtime["device_kind"]],
        mode=recipe["train"]["wandb_mode"])
    write_json(run / "wandb_run.json", {"id": run.name, "url": wandb_run.url})
    update_record(run, {"wandb": {"id": run.name, "url": wandb_run.url}})
    start = int(controller["step"])
    reconstruction, fit = initial_regression, None
    with (run / "metrics.jsonl").open("a", buffering=1) as stream:
        for step in range(start + 1, total + 1):
            began = time.monotonic()
            lr = settings["min_learning_rate"] + .5 * (settings["learning_rate"] - settings["min_learning_rate"]) * (
                1 + math.cos(math.pi * (step - 1) / max(total - 1, 1)))
            generator, optimizer, measurement = sampled_step(generator, optimizer, inputs, query_factors, key_factors, step, lr)
            if not bool(measurement["finite"]):
                write_json(run / f"numerical_failure_step_{step:08d}.json", {"step": step, "status": "numerical_failure",
                                                                             "reason": "nonfinite regression update"})
                update_record(run, {"status": "numerical_failure", "step": step})
                raise FloatingPointError("Nonfinite regression update")
            controller["step"] = step
            if step % settings["log_every"] == 0 or step == start + 1:
                row = {"event": "regression", "step": step, "learning_rate": lr,
                       "seconds": time.monotonic() - began, **scalar_metrics(measurement)}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                wandb_run.log({f"train/{name}": value for name, value in row.items() if isinstance(value, (float, int))},
                              step=step, commit=not (step % settings["eval_every"] == 0 or step == total))
                print(f"step={step} factorial_mse={row['factorial_mse']:.6f} lr={lr:.6g}", flush=True)
            if step % settings["eval_every"] == 0 or step == total:
                reconstruction = pair_regression(regression_evaluate, generator, inputs, query_factors, key_factors,
                                                 evaluation_pairs, size)
                fit = evaluate_lm(student_losses, replace_generator(student_params, generator), batches["fit"], microbatch)
                if reconstruction["factorial_mse"] < controller["best_mse"]:
                    controller.update(best_mse=reconstruction["factorial_mse"], best_step=step)
                    best = generator
                row = {"event": "evaluation", "step": step, "regression": reconstruction, "fit": fit,
                       "best_step": controller["best_step"]}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                wandb_run.log({"train/fit_perplexity": fit["perplexity"],
                               "train/evaluation_factorial_mse": reconstruction["factorial_mse"]}, step=step)
                write_json(run / "metadata.json", {"status": "running", **controller, "latest": row})
                update_record(run, {"status": "running", "step": step, "best_step": controller["best_step"]})
                print(f"Evaluation step={step}: fit PPL={fit['perplexity']:.5f}, factorial MSE="
                      f"{reconstruction['factorial_mse']:.6f}, best step={controller['best_step']}", flush=True)
            if step % settings["checkpoint_every"] == 0 or step == total or stopped[0]:
                save_checkpoint(checkpoints, step, generator, optimizer, best, controller)
            if stopped[0]:
                update_record(run, {"status": "paused", "step": step})
                wandb_run.finish()
                return
    if fit is None:  # A requeue after the last update resumes directly into finalization.
        fit = evaluate_lm(student_losses, replace_generator(student_params, generator), batches["fit"], microbatch)
    selected = replace_generator(student_params, best)
    if state_sha256(split_generator_parameters(selected)[1]) != backbone_audit["backbone_parameters_sha256"]:
        raise RuntimeError("Distillation changed frozen backbone parameters")
    outcome = {"status": "complete", "step": total, "best_step": controller["best_step"],
        "best_regression_mse": controller["best_mse"],
        "selected_fit": evaluate_lm(student_losses, selected, batches["fit"], microbatch),
        # The held-out batch scores the selected generator once; it chose nothing.
        "selected_heldout": evaluate_lm(student_losses, selected, batches["heldout"], microbatch),
        "teacher_fit": acceptance["teacher_native"], "baseline": baseline,
        "terminal_regression": reconstruction, "terminal_fit": fit, "finished_utc": utc()}
    final_path = checkpoints / f".selected-{controller['best_step']}"
    if final_path.exists(): shutil.rmtree(final_path)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(final_path, {"generator": best})
    write_json(final_path / "identity.json", {"step": controller["best_step"], "backbone": settings["parent"],
        "selection": "lowest pair-regression error on the fitting batch's fixed evaluation pairs"})
    write_json(cache / "finalization.json", outcome)
    finish_completion(run, outcome)
    wandb_run.summary.update({"selected_fit_perplexity": outcome["selected_fit"]["perplexity"],
        "selected_heldout_perplexity": outcome["selected_heldout"]["perplexity"],
        "baseline_heldout_perplexity": baseline["heldout"]["perplexity"]})
    wandb_run.finish()
    print(json.dumps({"complete": run.name, "selected_step": outcome["best_step"],
        "selected_fit_ppl": outcome["selected_fit"]["perplexity"],
        "selected_heldout_ppl": outcome["selected_heldout"]["perplexity"],
        "baseline_heldout_ppl": baseline["heldout"]["perplexity"]}), flush=True)


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

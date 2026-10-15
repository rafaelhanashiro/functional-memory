"""Fit sparse polynomial features to six frozen pretrained attention heads.

Each run is one feature-selection method/budget. The six independent heads are
optimized together and select one checkpoint by aggregate validation error.
Teacher activations are canonical outputs of the dense-reference run, reused by
reference rather than copied into each sparse run.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time

import numpy as np
import torch
import yaml

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import capture_artifacts, load_run, regenerate_index, update_record
from scaffolds.state_rank.head_capture import (
    capture_llama_heads, load_llama_for_capture, load_token_corpus, sha256_file,
)

ROOT = Path(__file__).resolve().parents[2]
STOP_REQUESTED = False


class PauseRequested(Exception):
    """The current allocation is ending; retain a resumable execution state."""


def check_stop():
    if STOP_REQUESTED:
        raise PauseRequested("Slurm requested a resumable stop")


def now():
    return datetime.now(UTC).isoformat()


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def activation_owner(run, config):
    parent = config["fit"]["warm_start_run_id"]
    return run.parent / parent if parent else run


def verify_teacher_files(record, config):
    """Validate the cached pretrained files against the allocation-time hashes."""
    teacher = record["teacher_artifacts"]
    if (teacher["model_id"] != config["teacher"]["model_id"]
            or teacher["revision"] != config["teacher"]["revision"]):
        raise ValueError("Recorded teacher identity differs from the frozen recipe")
    for name, descriptor in teacher["files"].items():
        check_stop()
        path = Path(teacher["snapshot"]) / name
        if (path.stat().st_size != descriptor["size_bytes"]
                or sha256_file(path) != descriptor["sha256"]):
            raise ValueError(f"Pretrained teacher file changed: {path}")


def verify_recorded_artifact(run, filename):
    record = load_run(run)
    matches = [entry for entry in record.get("artifacts", []) if Path(entry["path"]).name == filename]
    if len(matches) != 1 or sha256_file(run / filename) != matches[0]["sha256"]:
        raise ValueError(f"Completed dependency artifact is missing or changed: {run / filename}")


def verified_warm_start(parent):
    """Load only the completed parent's recorded validation-best checkpoint."""
    record = load_run(parent)
    if record["status"] != "complete":
        raise ValueError(f"Warm-start parent {parent.name} must be complete")
    verify_recorded_artifact(parent, "best_validation.json")
    verify_recorded_artifact(parent, "completion.json")
    best = json.loads((parent / "best_validation.json").read_text())
    completion = json.loads((parent / "completion.json").read_text())
    selected = record["selected_checkpoint"]
    if (best["step"] != completion["best_step"]
            or Path(selected["path"]).name != str(best["step"])):
        raise ValueError("Warm-start parent's selected checkpoint disagrees with validation")
    path = parent / "checkpoints" / str(best["step"]) / "state.pt"
    files = {entry["path"]: entry for entry in selected["files"]}
    if "state.pt" not in files or sha256_file(path) != files["state.pt"]["sha256"]:
        raise ValueError("Warm-start checkpoint checksum mismatch")
    return path, torch.load(path, map_location="cpu", weights_only=False)


def load_activations(owner, config, corpus):
    """Capture once; verify all cached archives on every execution attempt."""
    expected = {"tokens_sha256": config["data"]["tokens_sha256"],
                "teacher": config["teacher"], "heads": config["heads"]}
    descriptor = owner / "activation_manifest.json"
    check_stop()
    if config["fit"]["warm_start_run_id"]:
        if load_run(owner)["status"] != "complete":
            raise ValueError("Teacher activation owner must be complete before dependent runs start")
        verify_recorded_artifact(owner, descriptor.name)
    if descriptor.exists():
        manifest = json.loads(descriptor.read_text())
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise ValueError("Cached teacher activations differ from the frozen recipe")
        result = {}
        for split, artifact in manifest["artifacts"].items():
            check_stop()
            path = owner / artifact["filename"]
            if sha256_file(path) != artifact["sha256"]:
                raise ValueError(f"Teacher activation checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                result[split] = [{name: torch.from_numpy(archive[f"h{h}_{name}"].copy())
                                  for name in ("q", "k", "v", "teacher")}
                                 for h in range(len(config["heads"]))]
        return result, manifest
    if config["fit"]["warm_start_run_id"]:
        raise FileNotFoundError("The dense-reference run must capture teacher activations first")
    token_data = load_token_corpus(corpus, expected_sha256=config["data"]["tokens_sha256"])
    verify_teacher_files(load_run(owner), config)
    heads = [(item["layer"], item["head"]) for item in config["heads"]]
    model = load_llama_for_capture()
    result, artifacts, capture_errors = {}, {}, {}
    start = time.monotonic()
    for split, windows in token_data.items():
        collected = [{name: [] for name in ("q", "k", "v", "teacher")} for _ in heads]
        errors = []
        for index, ids in enumerate(windows):
            check_stop()
            outputs = capture_llama_heads(model, ids.unsqueeze(0), heads=heads)
            for h, head in enumerate(heads):
                item = outputs[head]
                for name in collected[h]:
                    collected[h][name].append(getattr(item, name))
                errors.append(item.native_max_abs_error)
            if index % 16 == 0 or index + 1 == len(windows):
                print(f"capture split={split} window={index + 1}/{len(windows)} elapsed={time.monotonic()-start:.1f}s", flush=True)
        result[split] = [{name: torch.cat(parts, dim=0) for name, parts in item.items()}
                         for item in collected]
        path = owner / f"activations_{split}.npz"
        temporary = owner / ".runtime" / f"activations_{split}.npz"
        temporary.parent.mkdir(exist_ok=True)
        with temporary.open("wb") as stream:
            np.savez(stream, **{f"h{h}_{name}": tensor.numpy()
                               for h, item in enumerate(result[split]) for name, tensor in item.items()})
        temporary.replace(path)
        artifacts[split] = {"filename": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        capture_errors[split] = max(errors)
    del model
    torch.cuda.empty_cache()
    manifest = {**expected, "artifacts": artifacts, "native_output_max_abs_errors": capture_errors,
                "capture_seconds": time.monotonic() - start, "created_utc": now()}
    write_json(descriptor, manifest)
    return result, manifest


def _teacher_attention(q, k, positions):
    scores = q @ k.T / math.sqrt(q.shape[-1])
    allowed = torch.arange(len(k), device=k.device)[None, :] <= positions[:, None]
    return scores.masked_fill(~allowed, -torch.inf).softmax(-1)


def _finite_or_none(value):
    return float(value) if math.isfinite(float(value)) else None


@torch.no_grad()
def evaluate(models, bank, data, config, *, window_limit=None, query_limit=None):
    """Pool squared error/energy per head; never average tokenwise relative errors."""
    rows = []
    for h, (model, head) in enumerate(zip(models, config["heads"], strict=True)):
        check_stop()
        model.eval()
        totals = dict(output_error=0.0, output_energy=0.0, attention_error=0.0,
                      attention_energy=0.0, invalid=0, queries=0, negative=0, pairs=0,
                      denominator_min=float("inf"))
        windows = len(data[h]["q"]) if window_limit is None else min(window_limit, len(data[h]["q"]))
        for window in range(windows):
            check_stop()
            q, k, v, target = [data[h][name][window].to("cuda") for name in ("q", "k", "v", "teacher")]
            key_features = bank(k)
            positions = (torch.arange(len(q), device=q.device) if query_limit is None else
                         torch.linspace(0, len(q)-1, min(query_limit, len(q)), device=q.device).long().unique())
            for batch in positions.split(config["fit"]["evaluation_query_batch_size"]):
                check_stop()
                prediction = model.forward_features(bank(q[batch]), key_features, v, batch, training_safe=False)
                valid = prediction["valid"]
                predicted = prediction["output"]
                teacher_weights = _teacher_attention(q[batch], k, batch)
                reference = target[batch]
                totals["queries"] += len(batch)
                totals["invalid"] += int((~valid).sum())
                totals["output_energy"] += float(reference.double().square().sum())
                totals["attention_energy"] += float(teacher_weights.double().square().sum())
                # Invalid predictions are counted explicitly and make aggregate error undefined.
                if valid.any():
                    totals["output_error"] += float((predicted[valid]-reference[valid]).double().square().sum())
                    totals["attention_error"] += float((prediction["attention"][valid]-teacher_weights[valid]).double().square().sum())
                allowed = torch.arange(len(k), device=k.device)[None, :] <= batch[:, None]
                totals["negative"] += int(((prediction["kernel"] < 0) & allowed).sum())
                totals["pairs"] += int(allowed.sum())
                totals["denominator_min"] = min(totals["denominator_min"], float(prediction["denominator"].min()))
        rows.append({**head, "output_relative_rms": (math.sqrt(totals["output_error"]/totals["output_energy"])
                                                     if totals["invalid"] == 0 else None),
                     "attention_relative_rms": (math.sqrt(totals["attention_error"]/totals["attention_energy"])
                                                if totals["invalid"] == 0 else None),
                     "invalid_fraction": totals["invalid"]/totals["queries"],
                     "negative_kernel_fraction": totals["negative"]/totals["pairs"],
                     "denominator_min": _finite_or_none(totals["denominator_min"]),
                     "queries": totals["queries"], "feature_count": bank.width,
                     "active_features": config["fit"]["budget"],
                     "zero_head_relative_rms": 1.0})
    return rows


def validation_score(rows):
    if not rows or any(row["output_relative_rms"] is None or row["invalid_fraction"]
                       or not math.isfinite(row["output_relative_rms"]) for row in rows):
        return float("inf")
    return sum(row["output_relative_rms"] ** 2 for row in rows) / len(rows)


def cpu_states(models):
    return [{name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()} for model in models]


def scheduled_validation(step, fit):
    return step % fit["evaluation_every"] == 0 or step == fit["steps"]


def truncate_metrics(run, resume_step, *, evaluated_step=None):
    """Remove measurements of discarded updates before deterministic replay."""
    path = run / "metrics.jsonl"
    if not path.exists():
        return
    retained = []
    text = path.read_text()
    lines = text.splitlines()
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            if index != len(lines) - 1 or text.endswith("\n"):
                raise ValueError("Malformed canonical measurements before the final partial line") from error
            break  # An interrupted final write is not a measurement.
        if (row["step"] <= resume_step
                and ("validation" not in row or evaluated_step is None or row["step"] <= evaluated_step)):
            retained.append(json.dumps(row, allow_nan=False))
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(line + "\n" for line in retained))
    temporary.replace(path)


def save_checkpoint(run, step, models, optimizer, generator, best, config, *, evaluated_step=-1):
    directory = run / "checkpoints" / str(step)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"step": step, "models": cpu_states(models), "optimizer": optimizer.state_dict(),
               "generator": generator.get_state(), "best": best, "fit": config["fit"],
               "evaluated_step": evaluated_step}
    temporary = directory / "state.pt.tmp"
    torch.save(payload, temporary)
    (run / ".runtime").mkdir(exist_ok=True)
    pointer = {"step": step, "sha256": sha256_file(temporary)}
    # Publishing the same step after an interrupted evaluation must remain safe
    # even if a kill occurs between state.pt and latest.json atomic replacements.
    pending = run / ".runtime/latest_pending.json"
    write_json(pending, pointer)
    temporary.replace(directory / "state.pt")
    write_json(run / ".runtime/latest.json", pointer)
    pending.unlink(missing_ok=True)
    keep = {step, best["step"]} if best is not None else {step}
    for candidate in (run / "checkpoints").iterdir():
        if candidate.is_dir() and candidate.name.isdigit() and int(candidate.name) not in keep:
            shutil.rmtree(candidate)


def restore_checkpoint(run, models, optimizer, generator, config):
    latest = run / ".runtime/latest.json"
    pending = run / ".runtime/latest_pending.json"
    if pending.exists():
        pointer = json.loads(pending.read_text())
        candidate = run / "checkpoints" / str(pointer["step"]) / "state.pt"
        if candidate.exists() and sha256_file(candidate) == pointer["sha256"]:
            write_json(latest, pointer)
        pending.unlink()
    if not latest.exists():
        truncate_metrics(run, -1)
        (run / "best_validation.json").unlink(missing_ok=True)
        return 0, None, -1
    pointer = json.loads(latest.read_text())
    path = run / "checkpoints" / str(pointer["step"]) / "state.pt"
    if sha256_file(path) != pointer["sha256"]:
        raise ValueError("Latest resumable checkpoint checksum mismatch")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["step"] != pointer["step"] or checkpoint["fit"] != config["fit"]:
        raise ValueError("Checkpoint step or frozen fitting recipe mismatch")
    for model, state in zip(models, checkpoint["models"], strict=True):
        model.load_state_dict(state)
    optimizer.load_state_dict(checkpoint["optimizer"])
    generator.set_state(checkpoint["generator"])
    truncate_metrics(run, checkpoint["step"], evaluated_step=checkpoint["evaluated_step"])
    # The checkpoint is authoritative if interruption preceded its pointer update.
    best = checkpoint["best"]
    if best is None:
        (run / "best_validation.json").unlink(missing_ok=True)
    else:
        write_json(run / "best_validation.json", {"step": best["step"],
                   "criterion": config["fit"]["checkpoint_selection"],
                   "score": best["score"], "heads": best["rows"]})
    return checkpoint["step"], best, checkpoint["evaluated_step"]


def recover_completion(run, config):
    """Commit an atomically written completion, then prune resumable artifacts.

    This is safe to repeat after interruption at any point. No resumable state
    disappears until the outcome has been committed to record.json.
    """
    path = run / "completion.json"
    if not path.exists():
        return False
    record = load_run(run)
    if record["status"] == "complete":
        verify_recorded_artifact(run, path.name)
    completion = json.loads(path.read_text())
    if completion["step"] != config["fit"]["steps"] or completion["status"] != "complete":
        raise ValueError("Completion disagrees with the frozen fitting recipe")
    evidence = completion["evidence"]
    for filename, checksum in evidence["run_files"].items():
        if sha256_file(run / filename) != checksum:
            raise ValueError(f"Completion evidence changed: {filename}")
    for item in evidence["dependencies"]:
        dependency = Path(item["path"])
        if not dependency.is_absolute():
            dependency = run.parent / dependency
        if sha256_file(dependency) != item["sha256"]:
            raise ValueError(f"Completion dependency changed: {item['path']}")
    best_directory = run / "checkpoints" / str(completion["best_step"])
    update_record(run, {"status": "complete", "step": completion["step"],
                        "retained_best_checkpoint_step": completion["best_step"],
                        **capture_artifacts(run, checkpoint_directory=best_directory)})
    for directory in (run / "checkpoints").iterdir():
        if directory.is_dir() and directory != best_directory:
            shutil.rmtree(directory)
    (run / ".runtime/latest.json").unlink(missing_ok=True)
    (run / ".runtime/latest_pending.json").unlink(missing_ok=True)
    return True


def record_no_valid_checkpoint(run, config, rows):
    failure = {"step": config["fit"]["steps"], "timestamp_utc": now(),
               "reason": "no checkpoint had valid finite normalization on all selection queries",
               "validation": rows, "diagnostic_checkpoint_step": config["fit"]["steps"]}
    write_json(run / f"numerical_failure_step_{config['fit']['steps']}.json", failure)
    checkpoint = run / "checkpoints" / str(config["fit"]["steps"]) / "state.pt"
    diagnostic = ({"diagnostic_checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)}}
                  if checkpoint.exists() else {})
    update_record(run, {"status": "numerical_failure", "terminal": failure, **diagnostic,
                        "checkpoint_retention": "terminal diagnostic: no valid selected checkpoint"})


def _train(run, config, *, benchmark=False):
    from scaffolds.state_rank.sparse_polynomial import PolynomialFeatures, SparsePolynomialHead
    record = load_run(run)
    corpus = Path(record["execution"]["corpus_directory"])
    if sha256_file(corpus / "manifest.json") != config["data"]["manifest_sha256"]:
        raise ValueError("Dataset manifest changed since allocation")
    if sha256_file(corpus / "tokens.npz") != config["data"]["tokens_sha256"]:
        raise ValueError("Token data changed since allocation")
    data, activation_manifest = load_activations(activation_owner(run, config), config, corpus)
    owner = activation_owner(run, config)
    update_record(run, {"teacher_activations": {
        "owner_run_id": owner.name,
        "manifest": {"path": str(owner / "activation_manifest.json"),
                     "sha256": sha256_file(owner / "activation_manifest.json")},
        "archives": [{"path": str(owner / item["filename"]), "sha256": item["sha256"],
                      "size_bytes": item["size_bytes"]} for item in activation_manifest["artifacts"].values()],
    }})
    bank = PolynomialFeatures(config["features"]["key_dim"], degree=config["features"]["degree"]).cuda()
    models = [SparsePolynomialHead(bank, mode=config["fit"]["variant"], budget=config["fit"]["budget"],
                                  seed=config["seed"]+h, denominator_margin=config["fit"]["denominator_margin"],
                                  initial_score_temperature=config["fit"]["initial_score_temperature"]).cuda()
              for h in range(len(config["heads"]))]
    if config["fit"]["warm_start_run_id"]:
        parent = activation_owner(run, config)
        path, parent_state = verified_warm_start(parent)
        for model, state in zip(models, parent_state["models"], strict=True):
            model.load_dense_weights(state)
        update_record(run, {"warm_start": {"run_id": parent.name, "step": parent_state["step"],
                                           "path": str(path), "sha256": sha256_file(path)}})
    # All selection calibration is restricted to the training split.
    for h, model in enumerate(models):
        q_sum = torch.zeros(bank.width, dtype=torch.float64, device="cuda")
        k_sum = torch.zeros_like(q_sum)
        count = 0
        for index in range(min(config["fit"]["calibration_windows"], len(data["train"][h]["q"]))):
            check_stop()
            q = bank(data["train"][h]["q"][index].cuda())
            k = bank(data["train"][h]["k"][index].cuda())
            q_sum += q.double().square().sum(0)
            k_sum += k.double().square().sum(0)
            count += len(q)
        model.set_calibration((q_sum/count).sqrt().float(), (k_sum/count).sqrt().float())
    generator = torch.Generator().manual_seed(config["seed"])
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters() if p.requires_grad],
                                 lr=config["fit"]["learning_rate"])
    start_step, best, evaluated_step = restore_checkpoint(run, models, optimizer, generator, config)
    monitor = None
    if not benchmark:
        import wandb
        monitor = wandb.init(entity=config["monitoring"]["entity"], project=config["monitoring"]["project"],
                             group=config["monitoring"]["group"], name=record["label"],
                             id=run.name, resume="allow", mode=config["monitoring"]["mode"],
                             dir=str(run), config=config, tags=["sparse_polynomial_heads", "pretrained", "H100"])
        update_record(run, {"wandb_run_id": monitor.id, "wandb_url": monitor.url})
    metrics = (run / "metrics.jsonl").open("a", buffering=1)
    try:
        start = time.monotonic()
        count_windows = len(data["train"][0]["q"])
        # Every epoch visits all cached training documents once, in deterministic order.
        order = torch.arange(count_windows)
        terminal_validation = None
        for line in (run / "metrics.jsonl").read_text().splitlines():
            row = json.loads(line)
            if "validation" in row:
                terminal_validation = row["validation"]
        for step in range(start_step, config["fit"]["steps"] + 1):
            try:
                if step > start_step:
                    window = int(order[(step-1) % count_windows])
                    positions = torch.randperm(config["data"]["sequence_length"], generator=generator)[:config["fit"]["query_batch_size"]].cuda()
                    optimizer.zero_grad(set_to_none=True)
                    losses, penalties, invalid = [], [], 0
                    for h, model in enumerate(models):
                        model.train()
                        q, k, v, target = [data["train"][h][name][window].cuda() for name in ("q", "k", "v", "teacher")]
                        result = model.forward_features(bank(q[positions]), bank(k), v, positions, training_safe=True)
                        loss = (result["output"]-target[positions]).square().mean() / target[positions].square().mean().clamp_min(1e-12)
                        penalty = result["denominator_penalty"]
                        ((loss + config["fit"]["denominator_penalty"] * penalty) / len(models)).backward()
                        losses.append(float(loss.detach()))
                        penalties.append(float(penalty.detach()))
                        invalid += int((~result["valid"]).sum())
                    parameters = [p for model in models for p in model.parameters() if p.requires_grad]
                    norm = torch.nn.utils.clip_grad_norm_(parameters, config["fit"]["gradient_clip"])
                    if not torch.isfinite(norm) or not all(math.isfinite(value) for value in losses+penalties):
                        failure = {"step": step, "timestamp_utc": now(), "losses": [_finite_or_none(x) for x in losses],
                                   "gradient_norm": _finite_or_none(norm), "reason": "nonfinite fitting loss or gradient"}
                        write_json(run / f"numerical_failure_step_{step}.json", failure)
                        update_record(run, {"status": "numerical_failure", "terminal": failure})
                        if monitor:
                            monitor.finish(exit_code=1)
                        raise FloatingPointError(f"Numerical failure at step {step}")
                    optimizer.step()
                    measurement = {"step": step, "train/output_relative_mse": sum(losses)/len(losses),
                                   "train/denominator_penalty": sum(penalties)/len(penalties),
                                   "train/invalid_fraction": invalid / (len(models)*len(positions)),
                                   "train/gradient_norm": float(norm), "elapsed_seconds": time.monotonic()-start}
                    metrics.write(json.dumps(measurement, allow_nan=False) + "\n")
                    if monitor:
                        monitor.log(measurement, step=step)
                    if step % 10 == 0:
                        print(json.dumps(measurement), flush=True)
                if scheduled_validation(step, config["fit"]) and step > evaluated_step:
                    rows = evaluate(models, bank, data["validation"], config,
                                    window_limit=config["fit"]["validation_windows_during_fit"],
                                    query_limit=config["fit"]["validation_queries_per_window"])
                    score = validation_score(rows)
                    measurement = {"step": step, "validation": rows, "val/output_relative_mse": _finite_or_none(score)}
                    metrics.write(json.dumps(measurement, allow_nan=False) + "\n")
                    print(f"validation step={step} relative_mse={score} invalid={[r['invalid_fraction'] for r in rows]}", flush=True)
                    if monitor:
                        monitor.log({"val/output_relative_mse": _finite_or_none(score)}, step=step)
                    evaluated_step = step
                    if math.isfinite(score) and (best is None or score < best["score"]):
                        best = {"step": step, "score": score, "rows": rows}
                        write_json(run / "best_validation.json", {"step": step, "criterion": config["fit"]["checkpoint_selection"],
                                                                  "score": _finite_or_none(score), "heads": rows})
                        save_checkpoint(run, step, models, optimizer, generator, best, config, evaluated_step=evaluated_step)
                    terminal_validation = rows
                if step % config["fit"]["checkpoint_every"] == 0 or STOP_REQUESTED or step == config["fit"]["steps"]:
                    save_checkpoint(run, step, models, optimizer, generator, best, config, evaluated_step=evaluated_step)
                check_stop()
            except PauseRequested:
                save_checkpoint(run, step, models, optimizer, generator, best, config,
                                evaluated_step=evaluated_step)
                raise
        if best is None:
            record_no_valid_checkpoint(run, config, terminal_validation)
            raise FloatingPointError(f"No valid validation checkpoint for {run.name}")
        terminal_validation = evaluate(models, bank, data["validation"], config)
        best_directory = run / "checkpoints" / str(best["step"])
        selected = torch.load(best_directory / "state.pt", map_location="cpu", weights_only=False)
        for model, state in zip(models, selected["models"], strict=True):
            model.load_state_dict(state)
        print(f"evaluating best_step={best['step']} on full validation/test/long-test", flush=True)
        best_validation = evaluate(models, bank, data["validation"], config)
        test = evaluate(models, bank, data["test"], config)
        test_long = evaluate(models, bank, data["test_long"], config)
        completion = {"status": "complete", "variant": config["fit"]["variant"], "budget": config["fit"]["budget"],
                      "step": config["fit"]["steps"], "best_step": best["step"],
                      "best_validation": best_validation, "terminal_validation": terminal_validation,
                      "checkpoint_selection_validation": best["rows"], "test": test, "test_long": test_long,
                      "fitting_and_evaluation_seconds": time.monotonic()-start,
                      "teacher_capture_seconds": activation_manifest["capture_seconds"], "completed_utc": now()}
        metrics.flush()
        completion["evidence"] = {
            "run_files": {filename: sha256_file(run / filename) for filename in (
                "config.yaml", "metrics.jsonl", "best_validation.json",
                f"checkpoints/{best['step']}/state.pt")},
            "dependencies": [{"path": str(Path(owner.name) / "activation_manifest.json"),
                              "sha256": sha256_file(owner / "activation_manifest.json")}],
        }
        write_json(run / "completion.json", completion)
        recover_completion(run, config)
        if monitor:
            monitor.summary["terminal/step"] = config["fit"]["steps"]
            monitor.summary["terminal/validation_relative_mse"] = _finite_or_none(validation_score(terminal_validation))
            monitor.summary["test/output_relative_mse"] = _finite_or_none(validation_score(test))
            monitor.finish()
    finally:
        metrics.close()


def train(run, config, *, benchmark=False):
    try:
        return _train(run, config, benchmark=benchmark)
    except PauseRequested:
        pointer = run / ".runtime/latest.json"
        step = json.loads(pointer.read_text())["step"] if pointer.exists() else 0
        update_record(run, {"status": "preempted", "step": step})
        print(f"paused run={run.name} resumable_step={step}", flush=True)
    finally:
        if not benchmark:
            import wandb
            if wandb.run is not None:
                status = load_run(run)["status"]
                wandb.finish(exit_code=1 if status == "numerical_failure" else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    record = load_run(run)
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment requires a Slurm GPU allocation")
    source = os.environ["SPARSE_HEADS_SOURCE_COMMIT"]
    changed = changes(ROOT, source, "attention_scaling", "state_rank")
    if changed:
        raise RuntimeError(f"Frozen inputs changed: {changed}")
    if sha256_file(run / "config.yaml") != record["config"]["sha256"]:
        raise ValueError("Run recipe changed after allocation")
    config = yaml.safe_load((run / "config.yaml").read_text())
    if record["status"] == "numerical_failure":
        print(f"run={run.name} concluded numerical failure retained", flush=True)
        return
    if recover_completion(run, config):
        print(f"run={run.name} completion verified", flush=True)
        return
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))))
    signal.signal(signal.SIGUSR1, _stop)
    (run / ".runtime").mkdir(exist_ok=True)
    attempt = {"slurm_job_id": os.environ.get("SLURM_JOB_ID"),
               "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
               "started_utc": now(), "node": socket.gethostname(),
               "gpu_type": torch.cuda.get_device_name(), "gpu_count": torch.cuda.device_count(),
               "partition": os.environ.get("SLURM_JOB_PARTITION"), "source_commit": source}
    update_record(run, {"status": "running", "source_commit": source, "home_cluster": "kennel",
                        "frozen_execution_sha256": frozen_hash(ROOT, source, "attention_scaling", "state_rank"),
                        "dependency_lock": {"path": str(ROOT / "scaffolds/state_rank/uv.lock"),
                                            "sha256": sha256_file(ROOT / "scaffolds/state_rank/uv.lock")},
                        "attempts": [*record.get("attempts", []), attempt],
                        "execution": {"working_directory": str(ROOT), "dtype": "float32", "allow_tf32": False,
                                      "torch_version": torch.__version__, "backend": "pytorch_cuda_eager"}})
    write_json(run / "metadata.json", {"model_id": config["teacher"]["model_id"], "source_commit": source,
                                      "variant": config["fit"]["variant"], "budget": config["fit"]["budget"],
                                      "heads": config["heads"], "attempt": attempt})
    print(f"run={run.name} variant={config['fit']['variant']} budget={config['fit']['budget']} source={source}", flush=True)
    train(run, config)
    regenerate_index(run.parent.parent)


if __name__ == "__main__":
    main()

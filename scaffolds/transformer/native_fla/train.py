"""Single-GPU training of the unmodified, pinned FLA causal language model.

The model and fused loss are upstream FLA. This module supplies packed data,
gradient accumulation, selective AdamW decay, the Flame schedule, and run lifecycle.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import numpy as np
import torch

from scaffolds.records import capture_artifacts, update_record
from scaffolds.transformer.data.packed import PackedBatchIterator, load_packed_manifest, manifest_fingerprint
from .checkpointing import latest_checkpoint, load_checkpoint, save_checkpoint
from .config import fla_model_kwargs, load_config, parameter_count
from .retention import retain_checkpoints


def now() -> str:
    return datetime.now(UTC).isoformat()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def learning_rate(config: dict, completed_steps: int) -> float:
    """TorchTitan's zero-indexed LambdaLR, including its virtual stable step.

    LambdaLR applies index zero at construction, before the first update.
    The pinned builder uses stable_steps = total + 1 - warmup - decay = 1.
    """
    o = config["optim"]
    warmup, total = o["warmup_steps"], o["total_steps"]
    if not 0 <= completed_steps < total:
        raise ValueError("Schedule index must name an update within the frozen horizon")
    if completed_steps < warmup:
        factor = (completed_steps + 1) / warmup
    elif completed_steps < warmup + 1:
        factor = 1.0
    else:
        progress = (completed_steps - warmup) / (total - warmup)
        minimum = o["min_learning_rate"] / o["learning_rate"]
        factor = minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    return o["learning_rate"] * factor


def optimizer_parameter_groups(model, weight_decay: float) -> list[dict]:
    """Partition trainable parameters by their role, preserving native exceptions.

    Module identity handles embeddings and norm gains independently of shape or
    registration name. An exemption wins if a shared parameter has several roles.
    The untied LM head and convolution/projection kernels retain decay.
    """
    from fla.modules import FusedLayerNormGated, FusedRMSNormGated, GroupNorm, LayerNorm, RMSNorm

    norm_types = (torch.nn.LayerNorm, torch.nn.RMSNorm, torch.nn.GroupNorm,
                  LayerNorm, RMSNorm, GroupNorm, FusedLayerNormGated, FusedRMSNormGated)
    exempt = set()
    for module in model.modules():
        exempt_module = isinstance(module, (torch.nn.Embedding, *norm_types))
        for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
            if (exempt_module or name == "bias" or name.endswith("_bias")
                    or getattr(parameter, "_no_weight_decay", False)):
                exempt.add(id(parameter))
    decayed, undecayed = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (undecayed if id(parameter) in exempt else decayed).append(parameter)
    return [{"params": decayed, "weight_decay": weight_decay},
            {"params": undecayed, "weight_decay": 0.0}]


def optimizer_policy_summary(model, optimizer) -> dict:
    """Record the actual optimizer groups and reject omissions or duplicates."""
    names = {id(parameter): name for name, parameter in model.named_parameters()
             if parameter.requires_grad}
    seen = set()
    groups = []
    for group in optimizer.param_groups:
        parameters = group["params"]
        identities = [id(parameter) for parameter in parameters]
        if len(set(identities)) != len(identities) or seen.intersection(identities):
            raise ValueError("An optimizer parameter appears in multiple groups")
        if any(identity not in names for identity in identities):
            raise ValueError("Optimizer contains an unknown or frozen parameter")
        seen.update(identities)
        groups.append({"weight_decay": float(group["weight_decay"]),
                       "parameter_tensors": len(parameters),
                       "parameter_count": sum(parameter.numel() for parameter in parameters),
                       "parameter_names": [names[identity] for identity in identities]})
    if seen != names.keys():
        raise ValueError("Optimizer groups omit trainable model parameters")
    return {"groups": groups, "parameter_tensors": len(names),
            "parameter_count": sum(group["parameter_count"] for group in groups)}


def should_evaluate(config: dict, step: int) -> bool:
    """Use the study's early/late validation cadence and always evaluate terminal."""
    train = config["train"]
    interval = (train["eval_early_every"] if step <= train["eval_early_until"]
                else train["eval_every"])
    return step == config["optim"]["total_steps"] or step % interval == 0


def build_model_optimizer(config: dict):
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

    random.seed(config["train"]["seed"])
    np.random.seed(config["train"]["seed"])
    torch.manual_seed(config["train"]["seed"])
    torch.cuda.manual_seed_all(config["train"]["seed"])
    model = GatedDeltaNetForCausalLM(GatedDeltaNetConfig(**fla_model_kwargs(config))).cuda()
    count = sum(p.numel() for p in model.parameters())
    if count != parameter_count(config):
        raise ValueError(f"Native parameter count changed: {count} != {parameter_count(config)}")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Native model master parameters must remain FP32")
    o = config["optim"]
    optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, o["weight_decay"]), lr=o["learning_rate"],
                                 betas=(o["beta1"], o["beta2"]), eps=o["eps"],
                                 weight_decay=o["weight_decay"], fused=True)
    optimizer_policy_summary(model, optimizer)
    return model, optimizer


def batch_loss(model, criterion, tokens):
    """Exactly T next-token predictions from a packed row of T+1 tokens.

    FLA's labels argument shifts internally and drops the last prediction.
    Apply its own fused loss to explicit next-token targets instead.
    """
    inputs = tokens[:, :-1].contiguous()
    targets = tokens[:, 1:].contiguous()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(input_ids=inputs, use_cache=False, return_dict=True).logits
        return criterion(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


def train_update(model, optimizer, criterion, iterator, config, completed_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(config, completed_steps)
    for group in optimizer.param_groups:
        group["lr"] = lr
    accumulation = config["data"]["batch_size"] // config["train"]["micro_batch_size"]
    loss_sum = torch.zeros((), device="cuda", dtype=torch.float32)
    started = time.perf_counter()
    for _ in range(accumulation):
        tokens = torch.from_numpy(next(iterator)).to(device="cuda", dtype=torch.long)
        loss = batch_loss(model, criterion, tokens)
        loss_sum += loss.detach().float() / accumulation
        (loss / accumulation).backward()
    loss_value = loss_sum.item()
    if not math.isfinite(loss_value):
        raise FloatingPointError(f"Nonfinite training loss: {loss_value}")
    try:
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["optim"]["grad_clip_norm"],
                                             error_if_nonfinite=True)
    except RuntimeError as error:
        if "non-finite" not in str(error):
            raise
        raise FloatingPointError(str(error)) from error
    optimizer.step()
    torch.cuda.synchronize()
    return {"train_loss": loss_value, "grad_norm": float(norm), "learning_rate": lr,
            "train_step_seconds": time.perf_counter() - started}


@torch.no_grad()
def evaluate(model, criterion, config, validated_manifest, batches=None):
    model.eval()
    micro = config["train"]["micro_batch_size"]
    count = (batches or config["train"]["eval_batches"]) * config["data"]["batch_size"] // micro
    iterator = PackedBatchIterator(config["data"]["eval_manifest"], batch_size=micro,
        sequence_length=config["data"]["sequence_length"], validated_manifest=validated_manifest)
    total = torch.zeros((), device="cuda", dtype=torch.float64)
    started = time.perf_counter()
    for _ in range(count):
        tokens = torch.from_numpy(next(iterator)).to(device="cuda", dtype=torch.long)
        total += batch_loss(model, criterion, tokens).double()
    nll = (total / count).item()
    if not math.isfinite(nll) or nll > math.log(np.finfo(np.float64).max):
        raise FloatingPointError(f"Nonfinite validation loss: {nll}")
    return {"validation_nll": nll, "validation_perplexity": math.exp(nll),
            "validation_sequences": count * micro,
            "validation_tokens": count * micro * (config["data"]["sequence_length"] - 1),
            "evaluation_seconds": time.perf_counter() - started}


def runtime_metadata(config):
    source = os.environ.get("NATIVE_FLA_SOURCE_COMMIT") or subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    return {"implementation": "native_fla", "source_commit": source,
            "reference": config["reference"], "parameter_count": parameter_count(config),
            "gpu_type": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(),
            "node": os.environ.get("SLURMD_NODENAME"), "home_cluster": os.environ.get("TRANSFORMER_CLUSTER"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "flash-linear-attention", "triton", "transformers")},
            "parameter_dtype": "float32", "autocast_dtype": "bfloat16",
            "resolved_backend": "upstream_fla_chunk", "torch_compile": False,
            "dependency_lock_sha256": hashlib.sha256(Path(__file__).with_name("uv.lock").read_bytes()).hexdigest()}


def start_tracking(config, output, metadata):
    train = config["train"]
    if train["wandb_mode"] == "disabled":
        return None
    try:
        import wandb
        identity_path = output / "wandb_run.json"
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else {
            "run_id": output.name, "entity": train["wandb_entity"], "project": train["wandb_project"]}
        write_json(identity_path, identity)
        return wandb.init(id=identity["run_id"], resume="allow", entity=train["wandb_entity"],
            project=train["wandb_project"], group=train["wandb_group"], name=train["wandb_run_name"],
            tags=["gated_deltanet", "native_fla", "fla_reference", metadata["gpu_type"],
                  metadata["home_cluster"] or "unknown_cluster"],
            config={**config, "runtime": metadata}, dir=str(output), mode=train["wandb_mode"],
            settings=wandb.Settings(init_timeout=30))
    except Exception as error:
        print(f"W&B unavailable: {error}", flush=True)
        return None


def append_metric(path, metric, tracker=None):
    with path.open("a") as stream:
        stream.write(json.dumps(metric, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    print(json.dumps(metric, sort_keys=True, allow_nan=False), flush=True)
    if tracker is not None:
        mapping = {"train_loss": "train/loss", "grad_norm": "train/grad_norm",
                   "learning_rate": "train/learning_rate", "validation_nll": "val/loss",
                   "validation_perplexity": "val/perplexity", "training_tokens": "train/tokens"}
        try:
            tracker.log({mapping[key]: value for key, value in metric.items() if key in mapping}, step=metric["step"])
        except Exception as error:
            print(f"W&B log failed: {error}", flush=True)


def rewind_metrics(path, completed_step):
    """Discard only measurements after the restored snapshot, including a torn tail."""
    if not path.exists():
        return
    kept = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            continue
        if value["step"] <= completed_step:
            kept.append(json.dumps(value, sort_keys=True, allow_nan=False))
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(line + "\n" for line in kept))
    temporary.replace(path)


def finalize_completion(output, checkpoints, summary):
    """Publish terminal evidence before pruning, and safely finish an interrupted closeout."""
    write_json(output / "completion.json", summary)
    best_step = summary["best_validation_step"]
    retain_checkpoints(checkpoints, best_step=best_step, completed=True)
    write_json(output / "retention.json", {"policy": "validation_best_only", "best_step": best_step})
    update_record(output, {"status": "complete", "terminal_step": summary["final_step"],
        "terminal_validation_nll": summary["validation_nll"],
        "terminal_validation_perplexity": summary["validation_perplexity"],
        **capture_artifacts(output, checkpoint_directory=checkpoints / str(best_step))})


def run_training(config_path, *, output_dir=None, stop_after_step=None,
                 eval_batches=None, qualification=False):
    from fla.modules import FusedCrossEntropyLoss

    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if (output_dir is not None or stop_after_step is not None or eval_batches is not None) and not qualification:
        raise ValueError("Runtime overrides are restricted to isolated qualification")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This runtime requires exactly one visible CUDA GPU")
    torch.cuda.set_device(0)
    output = Path(output_dir or config["train"]["workdir"]).resolve()
    checkpoints = output / "checkpoints" if qualification else Path(config["train"]["checkpoint_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not qualification and (output / "completion.json").exists():
        summary = json.loads((output / "completion.json").read_text())
        finalize_completion(output, checkpoints, summary)
        return summary
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    metadata = runtime_metadata(config)
    metadata.update(started_utc=now(), status="running", qualification=qualification)
    write_json(output / "metadata.json", metadata)
    write_json(output / "config.json", config)
    print(json.dumps({"event": "initializing", **metadata}), flush=True)
    model, optimizer = build_model_optimizer(config)
    metadata["optimizer_parameter_groups"] = optimizer_policy_summary(model, optimizer)
    write_json(output / "metadata.json", metadata)
    write_json(output / "native_model_config.json", model.config.to_dict())
    criterion = FusedCrossEntropyLoss(inplace_backward=True)
    step, cursor, best, restored_extra = 0, 0, None, {}
    checkpoint = latest_checkpoint(checkpoints)
    if checkpoint is not None:
        restored = load_checkpoint(checkpoint, model=model, optimizer=optimizer, expected_config_sha256=config_hash)
        step, cursor, best = restored["step"], restored["data_cursor"], restored["best_validation"]
        restored_extra = restored["extra"]
    if cursor != step * config["data"]["batch_size"]:
        raise ValueError("Checkpoint cursor disagrees with completed global updates")
    rewind_metrics(output / "metrics.jsonl", step)
    if best is not None:
        write_json(output / "best_validation.json", best)
    elif (output / "best_validation.json").exists():
        (output / "best_validation.json").unlink()
    print(json.dumps({"event": "checking_data", "restored_step": step, "data_cursor": cursor}), flush=True)
    train_manifest = load_packed_manifest(config["data"]["train_manifest"])
    eval_manifest = load_packed_manifest(config["data"]["eval_manifest"])
    target = stop_after_step or config["optim"]["total_steps"]
    if not step <= target <= config["optim"]["total_steps"]:
        raise ValueError("Training target must not precede the restored step or exceed the frozen horizon")
    if config["optim"]["total_steps"] * config["data"]["batch_size"] > train_manifest[1]["num_sequences"]:
        raise ValueError("Frozen training horizon exceeds the packed corpus")
    data_identity = {key: {"path": config["data"][key + "_manifest"],
                          "sha256": manifest_fingerprint(config["data"][key + "_manifest"])}
                     for key in ("train", "eval")}
    write_json(output / "data_identity.json", data_identity)
    iterator = PackedBatchIterator(config["data"]["train_manifest"],
        batch_size=config["train"]["micro_batch_size"], sequence_length=config["data"]["sequence_length"],
        start_sequence=cursor, validated_manifest=train_manifest)
    tracker = None if qualification else start_tracking(config, output, metadata)
    for failure in output.glob("numerical_failure_step_*.json"):
        failure.unlink()
    if not qualification:
        update_record(output, {"status": "running", "runtime": metadata, "data_manifests": data_identity,
                              "wandb_id": tracker.id if tracker is not None else None})
    stop_requested = False

    def stop_handler(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Signal {signum}: checkpointing after the current update", flush=True)

    previous = {sig: signal.signal(sig, stop_handler) for sig in (signal.SIGUSR1, signal.SIGTERM)}
    started = time.perf_counter()
    timings = []
    last_train = restored_extra.get("last_train")
    validation = restored_extra.get("validation")
    tokens_per_step = config["data"]["batch_size"] * (config["data"]["sequence_length"] - 1)
    try:
        while step < target:
            last_train = train_update(model, optimizer, criterion, iterator, config, step)
            step += 1
            if os.environ.get("SLURM_JOB_END_TIME"):
                stop_requested |= time.time() >= int(os.environ["SLURM_JOB_END_TIME"]) - 600
            timings.append(last_train["train_step_seconds"])
            metric = {"step": step, "training_tokens": step * tokens_per_step, **last_train}
            if step % config["train"]["log_every"] == 0 or step == 1 or qualification:
                append_metric(output / "metrics.jsonl", metric, tracker)
            should_eval = should_evaluate(config, step) or step == target
            improved = False
            if should_eval:
                validation = {"step": step, "training_tokens": step * tokens_per_step,
                              **evaluate(model, criterion, config, eval_manifest, eval_batches)}
                append_metric(output / "metrics.jsonl", validation, tracker)
                if best is None or validation["validation_nll"] < best["validation_nll"]:
                    best = {**validation, "selection_rule": "lowest_validation_nll"}
                    improved = True
            should_save = improved or step % config["train"]["ckpt_every"] == 0 or step == target or stop_requested
            if should_save:
                save_checkpoint(checkpoints / str(step), model=model, optimizer=optimizer, step=step,
                    data_cursor=iterator.cursor, best_validation=best, config_sha256=config_hash,
                    extra={"last_train": last_train, "validation": validation})
                if best is not None:
                    write_json(output / "best_validation.json", best)
                retain_checkpoints(checkpoints, best_step=best["step"] if best else None)
                print(json.dumps({"event": "checkpoint", "step": step, "data_cursor": iterator.cursor}), flush=True)
                metadata.update(step=step, updated_utc=now())
                write_json(output / "metadata.json", metadata)
                if not qualification:
                    update_record(output, capture_artifacts(output))
            if stop_requested and step < target:
                metadata.update(status="submitted", step=step, updated_utc=now())
                write_json(output / "metadata.json", metadata)
                if not qualification:
                    update_record(output, {"status": "submitted", **capture_artifacts(output)})
                return {"status": "requeue", "step": step, "exit_code": 75}
        summary = {"status": "complete", "final_step": step, "end_step": step,
            "data_cursor": iterator.cursor, "parameter_count": parameter_count(config),
            "training_tokens": step * tokens_per_step, "validation_nll": validation["validation_nll"],
            "validation_perplexity": validation["validation_perplexity"], "best_validation_step": best["step"],
            "last_train": last_train, "train_step_seconds": timings,
            "runtime_seconds": time.perf_counter() - started, "completed_utc": now(),
            "source_commit": metadata["source_commit"], "qualification": qualification}
        if qualification:
            write_json(output / "qualification.json", summary)
        else:
            metadata.update(status="complete", step=step, updated_utc=now())
            write_json(output / "metadata.json", metadata)
            finalize_completion(output, checkpoints, summary)
        if tracker is not None:
            try:
                tracker.summary.update({"terminal/step": step, "terminal/validation_nll": validation["validation_nll"]})
            except Exception as error:
                print(f"W&B summary failed: {error}", flush=True)
        return summary
    except FloatingPointError as error:
        failure = {"status": "numerical_failure", "step": step,
                   "attempted_step": step + 1 if iterator.cursor > step * config["data"]["batch_size"] else step,
                   "last_valid_training": last_train, "error": str(error), "timestamp_utc": now()}
        write_json(output / f"numerical_failure_step_{failure['attempted_step']}.json", failure)
        metadata.update(status="numerical_failure", step=step, updated_utc=now())
        write_json(output / "metadata.json", metadata)
        if not qualification:
            update_record(output, {"status": "numerical_failure", "numerical_failure": failure, **capture_artifacts(output)})
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if tracker is not None:
            try:
                tracker.finish()
            except Exception as error:
                print(f"W&B finish failed: {error}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    summary = run_training(args.config)
    raise SystemExit(summary.get("exit_code", 0))


if __name__ == "__main__":
    main()

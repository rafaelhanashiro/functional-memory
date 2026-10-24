"""Weights & Biases mirror of committed training metrics.

W&B is a convenience view: `runs/<run>/metrics.jsonl` stays canonical, and every
W&B failure is reported and ignored. One project holds every experiment; runs are
grouped by experiment and tagged by mechanism and hardware. Metrics use the
prefixes `train/`, `val/`, and `attention/<kernel>/`, all against the `step` axis,
so a resumed run can relog the updates after its last checkpoint.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np

from scaffolds.transformer.configs import Config, config_to_dict
from scaffolds.transformer.training.metrics import write_json_atomic

IDENTITY_FILE = "wandb_run.json"
METRIC_PREFIXES = ("train/*", "val/*", "attention/*", "model/*", "terminal/*")


def experiment_paths(workdir: Path) -> tuple[str | None, Path | None]:
    """Study name and colocated artifact root for `experiments/<study>/runs/<run>`."""
    parts = Path(workdir).parts
    for index in range(len(parts) - 3, -1, -1):
        if parts[index] == "experiments" and parts[index + 2] == "runs":
            return parts[index + 1], Path(*parts[:index + 4])
    return None, None


def run_settings(config: Config) -> dict:
    """Resolved W&B destination, grouping, and local directory for this run."""
    train = config.train
    experiment, logs = experiment_paths(train.workdir)
    return {
        "entity": train.wandb_entity,
        "project": train.wandb_project,
        "run_name": train.wandb_run_name or Path(train.workdir).name,
        "group": train.wandb_group or experiment,
        "directory": Path(train.wandb_dir or logs or train.workdir),
    }


def run_tags(config: Config, runtime: dict) -> list[str]:
    from scaffolds.transformer.training.diagnostics import attention_identity

    identity = attention_identity(config)
    tags = [identity["attention_kernel"], f"dk{config.model.qk_head_dim}"]
    if identity.get("attention_degree") is not None:
        tags.append(f"m{identity['attention_degree']}")
    if identity.get("gdn_erase_mode") is not None:
        tags.append(f"erase_{identity['gdn_erase_mode']}")
    if identity.get("gdn_polynomial_degree") in (2, 3):
        tags.append({2: "quadratic_gdn", 3: "cubic_gdn"}[identity["gdn_polynomial_degree"]])
        tags.append(f"write_init_{identity['gdn_write_initialization']}")
    if identity.get("gdn_linear_write_activation") == "tanh":
        tags.append("tanh_write")
    if identity.get("qk_norm"):
        tags.append("qk_norm")
    if identity.get("attention_coefficient_conditioning") in ("query", "query_key"):
        tags.append(f"{identity['attention_coefficient_conditioning']}_conditioned_coefficients")
        tags.append(f"coefficient_{config.model.attention_coefficient_activation}")
    if identity.get("attention_output_scaling"):
        tags.append(f"output_{identity['attention_output_scaling']}")
    if config.model.initialization != "flax_default":
        tags.append(f"init_{config.model.initialization}")
    if runtime.get("gpu_type"):
        tags.append(f"{runtime.get('gpu_count', '')}x{runtime['gpu_type']}")
    if runtime.get("cluster"):
        tags.append(runtime["cluster"])
    return tags


def runtime_settings(device_count: int, device_kind: str) -> dict:
    """Execution facts kept apart from the scientific recipe in the W&B config."""
    return {
        "cluster": os.environ.get("TRANSFORMER_CLUSTER"),
        "gpu_type": os.environ.get("TRANSFORMER_GPU_TYPE") or device_kind,
        "gpu_count": device_count,
        "device_kind": device_kind,
        "host_staging": os.environ.get("TRANSFORMER_HOST_STAGING"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def init_run(config: Config, runtime: dict) -> tuple[object | None, str | None]:
    """Start or resume this run's W&B mirror; returns (run or None, run id)."""
    if config.train.wandb_mode == "disabled":
        return None, None
    settings = run_settings(config)
    run_id = None
    try:
        identity_path = Path(config.train.workdir) / IDENTITY_FILE
        expected = {key: settings[key] for key in ("entity", "project", "run_name")}
        if identity_path.exists():
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            for key, value in expected.items():
                if identity.get(key) != value:
                    raise ValueError(f"W&B {key} changed from {identity.get(key)!r} to {value!r}")
            run_id = str(identity["run_id"])
        else:
            run_id = uuid.uuid4().hex
            write_json_atomic(identity_path, {**expected, "run_id": run_id})

        import wandb

        settings["directory"].mkdir(parents=True, exist_ok=True)
        run = wandb.init(
            entity=settings["entity"], project=settings["project"], name=settings["run_name"],
            group=settings["group"], tags=run_tags(config, runtime), id=run_id, resume="allow",
            mode=config.train.wandb_mode, dir=str(settings["directory"]),
            config={**config_to_dict(config), "runtime": runtime},
            settings=wandb.Settings(init_timeout=30),
        )
        run.define_metric("step")
        for prefix in METRIC_PREFIXES:
            run.define_metric(prefix, step_metric="step")
    except Exception as error:
        print(f"wandb_init_failed={type(error).__name__}: {error}; continuing_with_local_metrics=true", flush=True)
        return None, run_id
    return run, run_id


def attention_metrics(diagnostics: dict) -> dict:
    """Numeric per-layer attention diagnostics under `attention/<kernel>/`."""
    prefix = f"attention/{diagnostics.get('attention_kernel', 'unknown')}"
    payload = {}
    for layer in diagnostics.get("layers", []):
        for key, value in layer.items():
            if key != "layer" and isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[f"{prefix}/layer{layer['layer']:02d}/{key}"] = value
    for key in ("coefficients", "qk_norm_scale", "qk_norm_gains"):
        _summarize_effective_values(payload, f"{prefix}/{key}", diagnostics.get(key) or {})
    return payload


def _summarize_effective_values(payload: dict, prefix: str, report: dict) -> None:
    """Mirror a parameter report's effective values: scalars directly, arrays as mean/min/max."""
    names = report.get("effective_coefficient_names", ())
    if "effective_summary" in report:
        for statistic, array in report["effective_summary"].items():
            values = np.asarray(array, dtype=np.float64)
            for layer in range(values.shape[0]):
                for head in range(values.shape[1]):
                    for index, name in enumerate(names):
                        payload[f"{prefix}/layer{layer:02d}/head{head:02d}/{name}/{statistic}"] = float(
                            values[layer, head, index])
        return
    try:
        # The last axis indexes named values; nonfinite values arrive as JSON strings.
        values = np.asarray(report.get("effective", ()), dtype=np.float64)
    except (TypeError, ValueError):
        return
    if not names or values.ndim < 1 or values.shape[-1] != len(names):
        return
    for index, name in enumerate(names):
        column = values[..., index]
        if column.ndim == 0:
            payload[f"{prefix}/{name}"] = float(column)
        else:
            # Per-layer/head values are summarized; metrics.jsonl keeps every value.
            for statistic, reduce in (("mean", np.mean), ("min", np.min), ("max", np.max)):
                payload[f"{prefix}/{name}/{statistic}"] = float(reduce(column))


def _log(run, payload: dict, step: int) -> None:
    if run is None:
        return
    try:
        run.log({"step": step, **payload})
    except Exception as error:
        print(f"wandb_log_failed_step={step} error={type(error).__name__}: {error}; local_metrics_committed=true",
              flush=True)


def log_train(run, step: int, nll: float, perplexity: float, learning_rate: float,
              coefficient_penalty: float | None = None, *,
              gradient_norm: float | None = None, gradient_clip_norm: float | None = None,
              telemetry: dict[str, float] | None = None) -> None:
    payload = {"train/nll": nll, "train/ppl": perplexity, "train/learning_rate": learning_rate}
    if coefficient_penalty is not None:
        payload["train/coefficient_penalty"] = coefficient_penalty
    if gradient_norm is not None:
        payload["train/gradient_norm_pre_clip"] = gradient_norm
        if gradient_clip_norm is not None:
            clipped = gradient_norm > gradient_clip_norm
            payload.update({
                "train/gradient_norm_post_clip": min(gradient_norm, gradient_clip_norm),
                "train/gradient_clip_scale": gradient_clip_norm / gradient_norm if clipped else 1.0,
                "train/gradient_clipped": int(clipped),
            })
    if telemetry is not None:
        payload.update({f"train/telemetry/{key}": value for key, value in telemetry.items()})
    _log(run, payload, step)


def log_validation(run, record: dict) -> None:
    """Mirror one committed validation record."""
    payload = {
        "train/tokens": record["training_tokens"],
        "val/nll": record["validation_nll"],
        "val/ppl": record["validation_perplexity"],
        "val/is_best": int(record["is_best"]),
        "model/parameter_count": record["parameter_count"],
        **attention_metrics(record.get("attention_diagnostics") or {}),
    }
    if "train_coefficient_penalty" in record:
        payload["train/coefficient_penalty"] = record["train_coefficient_penalty"]
    for source, destination in (
        ("train_nll_sample_mean", "train/nll_window_mean"),
        ("train_perplexity_from_mean_nll", "train/ppl_from_window_mean"),
        ("tokens_per_second", "train/tokens_per_second"),
        ("tokens_per_second_per_device", "train/tokens_per_second_per_device"),
        ("train_gradient_norm_mean", "train/gradient_norm_pre_clip_window_mean"),
        ("train_gradient_norm_max", "train/gradient_norm_pre_clip_window_max"),
        ("train_gradient_clipped_fraction", "train/gradient_clipped_fraction"),
        ("train_gradient_spike_count", "train/gradient_spike_count"),
    ):
        if source in record:
            payload[destination] = record[source]
    _log(run, payload, int(record["step"]))


def summarize_completion(run, completion: dict, best: dict | None) -> None:
    """Terminal and validation-best metrics in the run summary."""
    if run is None:
        return
    summary = {
        "terminal/step": completion["step"], "terminal/training_tokens": completion["training_tokens"],
        "terminal/val_nll": completion["validation_nll"], "terminal/val_ppl": completion["validation_perplexity"],
    }
    if best is not None:
        summary.update({"best/step": best["step"], "best/val_nll": best["validation_nll"],
                        "best/val_ppl": best["validation_perplexity"]})
    try:
        run.summary.update(summary)
    except Exception as error:
        print(f"wandb_summary_failed={type(error).__name__}: {error}", flush=True)


def finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as error:
        print(f"wandb_finish_failed={type(error).__name__}: {error}", flush=True)

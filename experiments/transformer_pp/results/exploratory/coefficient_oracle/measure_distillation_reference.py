"""Measure the best constant coefficient predictor on the distillation teacher.

This optional CPU collector reads the retained teacher checkpoint on its home
cluster. It writes compact measurements only under the distillation run; figures
can then regenerate from that committed measurement without checkpoint access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from scaffolds.records import load_run, validate_record
from scaffolds.transformer.training.coefficient_distillation_model import coefficient_chunks


def measure(run: Path) -> dict:
    recipe = yaml.safe_load((run / "config.yaml").read_text())
    teacher = recipe["distillation"]["teacher"]
    source = Path(teacher["run_directory"])
    validate_record(source, checkpoint_hashes=False)
    source_recipe_path = source / "config.yaml"
    if hashlib.sha256(source_recipe_path.read_bytes()).hexdigest() != teacher["config_sha256"]:
        raise ValueError("Teacher recipe changed")
    if load_run(source)["selected_checkpoint"]["sha256"] != teacher["checkpoint_sha256"]:
        raise ValueError("Teacher checkpoint identity changed")
    source_recipe = yaml.safe_load(source_recipe_path.read_text())
    factors = np.asarray([math.factorial(i) for i in range(source_recipe["model"]["attention_degree"] + 1)], np.float64)
    sums = squares = None
    count = 0
    for _, coefficients in coefficient_chunks(Path(teacher["checkpoint"]), source_recipe):
        coordinates = np.asarray(coefficients, np.float64) * factors
        summed = coordinates.sum(axis=(0, 2))
        squared = np.square(coordinates).sum(axis=(0, 2))
        sums = summed if sums is None else sums + summed
        squares = squared if squares is None else squares + squared
        count += coordinates.shape[0] * coordinates.shape[2]
        print(f"Measured {count} teacher query positions per network", flush=True)
    mean = sums / count
    variance = squares / count - mean * mean
    taylor_mse = float((squares / count - 2 * mean + 1).mean())
    static_mse = float(variance.mean())
    completed = json.loads((run / "completion.json").read_text())
    dense_mse = float(completed["best_regression_mse"])
    angle = np.degrees(np.arccos(np.clip(
        mean.sum(-1) / (np.linalg.norm(mean, axis=-1) * np.sqrt(mean.shape[-1])), -1, 1)))
    return {
        "schema_version": 1,
        "run_id": run.name,
        "source_teacher_run_id": teacher["run_id"],
        "source_teacher_config_sha256": teacher["config_sha256"],
        "source_teacher_checkpoint_sha256": teacher["checkpoint_sha256"],
        "algorithm": "Float64 moments of u_r=r!*c_r over every teacher sequence/query, separately per layer/head/order; E[u] minimizes constant-predictor squared error. Average order/network variances use the distillation objective weights.",
        "samples_per_network": count,
        "taylor_mse": taylor_mse,
        "optimal_static_mean_mse": static_mse,
        "dense_best_mse": dense_mse,
        "dense_best_step": completed["best_regression_step"],
        "fraction_taylor_mse_removed_by_static_mean": 1 - static_mse / taylor_mse,
        "fraction_static_residual_removed_by_dense": 1 - dense_mse / static_mse,
        "fraction_dense_mse_improvement_explained_by_static_mean": (taylor_mse - static_mse) / (taylor_mse - dense_mse),
        "static_mean_mse_by_layer": variance.mean(axis=(1, 2)).tolist(),
        "static_mean_direction_angle_median_deg": float(np.median(angle)),
        "limits": "The reference optimizes coefficient MSE, not language-model loss. The residual is teacher coefficient variance, not an irreducible error bound for input-dependent generators. No static LM fit or additional final-test evaluation is performed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    result = measure(run)
    destination = run / "distillation_reference.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()

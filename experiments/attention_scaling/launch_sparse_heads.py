"""Prepare and launch the frozen-Llama sparse polynomial head experiment."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from scaffolds.launch.frozen import changes, frozen_hash
from scaffolds.records import create_run, load_run, regenerate_index, update_record

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/attention_scaling"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(corpus):
    from scaffolds.state_rank.head_capture import cached_model_manifest
    corpus = Path(corpus).resolve()
    manifest = json.loads((corpus / "manifest.json").read_text())
    model_manifest = cached_model_manifest()
    token_hash = sha256(corpus / "tokens.npz")
    runs = []
    for variant, budget in [("dense", 47905), *[(mode, count) for count in (128, 512, 2048)
                                             for mode in ("fixed", "random", "dynamic")]]:
        config = {
            "schema_version": 1, "experiment_kind": "frozen_pretrained_head_distillation",
            "condition": f"sparse_polynomial_heads/{variant}/{budget}", "seed": 20261015,
            "teacher": {"model_id": "meta-llama/Llama-3.2-1B",
                        "revision": "4e20de362430cd3b72f300e6b0f18e50e7166e08",
                        "dtype": "float32", "attention_implementation": "eager",
                        "qk_preprocessing": "preserve_pretrained_post_rope_scores",
                        "frozen": True},
            "model": {"model_id": "meta-llama/Llama-3.2-1B", "initialization": "pretrained",
                      "attention_mechanism": "dot_product", "qk_head_dim": 64, "qk_norm": False,
                      "attention_degree": 3, "attention_normalization": True,
                      "attention_coefficients": "learned", "attention_coefficient_sharing": "per_head",
                      "attention_coefficient_conditioning": "query" if variant == "dynamic" else "static",
                      "attention_coefficient_parameterization": "positive_monomial_weights"},
            "heads": [{"layer": layer, "head": head} for layer in (2, 8, 14) for head in (0, 16)],
            "data": {"dataset_id": "Salesforce/wikitext", "dataset_config": "wikitext-2-raw-v1",
                     "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
                     "manifest_sha256": sha256(corpus / "manifest.json"),
                     "tokens_sha256": token_hash,
                     "windows": {"train": 200, "validation": 32, "test": 32, "test_long": 8},
                     "sequence_length": 512, "long_sequence_length": 2048,
                     "split_policy": "official splits; document-disjoint; no cross-document windows"},
            "features": {"degree": 3, "key_dim": 64, "value_dim": 64,
                         "basis": "symmetric Taylor monomials, exact multinomial multiplicities",
                         "feature_count": 47905, "constant_always_active": True,
                         "key_writes": "dense", "normalization": "causal kernel sum",
                         "log_weight_bounds": [-12, 12],
                         "fixed_support": "top training-query-RMS times key-RMS times dense learned weight",
                         "dynamic_score": "abs(query feature) times key-RMS times weight times learned relevance",
                         "dynamic_gradient": "hard support forward; sigmoid straight-through gate backward"},
            "fit": {"variant": variant, "budget": budget, "steps": 400 if variant == "dense" else 200,
                    "initial_score_temperature": 0.1, "calibration_windows": 8,
                    "learning_rate": 0.003, "query_batch_size": 8, "evaluation_query_batch_size": 16,
                    "evaluation_every": 100, "checkpoint_every": 25,
                    "validation_windows_during_fit": 8, "validation_queries_per_window": 32,
                    "gradient_clip": 1.0, "denominator_margin": 0.001,
                    "denominator_penalty": 10.0,
                    "denominator_validity": "positive denominator greater than margin times absolute kernel sum",
                    "training_denominator": "max(raw denominator, margin times absolute kernel sum); violations penalized and logged",
                    "evaluation_denominator": "raw denominator; invalid queries yield undefined aggregate error",
                    "objective": "per-head relative output MSE plus explicit denominator penalty",
                    "warm_start_run_id": runs[0].name if runs else None,
                    "checkpoint_selection": "lowest aggregate validation relative output MSE; invalid queries disqualify"},
            "monitoring": {"entity": "rafah-massachusetts-institute-of-technology",
                           "project": "functional-memory", "group": "attention_scaling", "mode": "online"},
        }
        run = create_run(STUDY, config, label=f"llama1b-sparse-polynomial-{variant}-{budget}-seed20261015")
        update_record(run, {"home_cluster": "kennel", "model_id": config["teacher"]["model_id"],
                            "seed": config["seed"], "attention_family": "pretrained_polynomial_probe",
                            "qk_head_dim": 64, "attention_degree": 3, "attention_normalization": True,
                            "coefficient_mode": "learned", "coefficient_parameterization": "positive_monomial_weights",
                            "teacher_artifacts": model_manifest,
                            "data_identity": {"path": str(corpus / "manifest.json"),
                                              "sha256": config["data"]["manifest_sha256"]},
                            "execution": {"corpus_directory": str(corpus)},
                            "checkpoint_retention": "validation_best"})
        runs.append(run)
    selection = {"schema_version": 1, "study": "attention_scaling", "opened": "2026-10-15", "runs": []}
    for run in runs:
        recipe = yaml.safe_load((run / "config.yaml").read_text())
        selection["runs"].append({"role": f"{recipe['fit']['variant']}-{recipe['fit']['budget']}",
                                  "variant": recipe["fit"]["variant"], "budget": recipe["fit"]["budget"],
                                  "study": "attention_scaling", "run_id": run.name, "sources": {}})
    directory = ROOT / "results/exploratory/sparse_polynomial_heads"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "selection.yaml").write_text(yaml.safe_dump(selection, sort_keys=False))
    from scaffolds.records.selections import refresh_selection
    refresh_selection(directory)
    regenerate_index(STUDY)
    print(" ".join(run.name for run in runs))
    return runs


def submit(run_ids, partition, hours, node=None):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    changed = changes(ROOT, commit, "attention_scaling", "state_rank")
    if changed:
        raise ValueError(f"Commit the frozen inputs before submission: {changed}")
    runs = [STUDY / "runs" / name for name in run_ids]
    for run in runs:
        subprocess.run(["git", "cat-file", "-e", f"{commit}:{run.relative_to(ROOT)}"], cwd=ROOT, check=True)
        if load_run(run)["status"] not in {"planned", "failed", "preempted", "submitted", "running", "complete"}:
            raise ValueError(f"Unsupported status for {run.name}")
        previous_job = load_run(run).get("submission", {}).get("slurm_job_id")
        if previous_job:
            live = subprocess.run(["squeue", "-h", "-j", str(previous_job), "-o", "%T"],
                                  text=True, capture_output=True, check=True)
            if live.stdout.strip():
                raise ValueError(f"Run {run.name} already belongs to live Slurm job {previous_job}; refusing concurrent execution")
    log = runs[0] / ".runtime/launcher.log"
    log.parent.mkdir(exist_ok=True)
    command = ["sbatch", "--parsable", "--nice=0", f"--partition={partition}", "--gres=gpu:h100:1",
               "--cpus-per-task=8", "--mem=64G", f"--time={hours}:00:00", "--job-name=sparse-poly-heads",
               f"--output={log}", f"--error={log}"]
    if node:
        command += [f"--nodelist={node}"]
    command += [str(ROOT / "experiments/attention_scaling/sparse_heads.sbatch"), str(ROOT), commit, *run_ids]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(result.stdout + result.stderr)
    for run in runs:
        record = load_run(run)
        if record["status"] == "complete":
            continue
        update_record(run, {"status": "submitted", "source_commit": commit,
                            "frozen_execution_sha256": frozen_hash(ROOT, commit, "attention_scaling", "state_rank"),
                            "submission": {"command": command, "slurm_job_id": job_id, "partition": partition,
                                           "submitted_utc": datetime.now(UTC).isoformat()},
                            "log_path": str(run / "training.log")})
    regenerate_index(STUDY)
    print(f"job_id={job_id}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--submit", nargs="+")
    parser.add_argument("--partition", default="debug")
    parser.add_argument("--hours", type=int, default=2)
    parser.add_argument("--node")
    args = parser.parse_args()
    if args.prepare:
        if not args.corpus:
            parser.error("--prepare requires --corpus")
        prepare(args.corpus)
    elif args.submit:
        submit(args.submit, args.partition, args.hours, args.node)
    else:
        parser.error("Choose --prepare or --submit")


if __name__ == "__main__":
    main()

"""Freeze recipes, qualify a GPU allocation, and run the shared GLA trainer."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

EXPERIMENT = "gla_key_dimension"
RECORD = Path("results") / EXPERIMENT
MANIFEST = RECORD / "provenance/manifest.json"
WIDTHS = (8, 16, 32, 64, 128)
MODULE = "scaffolds.transformer.studies.gla_key_dimension.sweep"
SOURCE_ROOT = Path(__file__).resolve().parents[4]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_identity(root):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True):
        raise ValueError("Commit and freeze the execution checkout before qualification or training")
    return source


def identity(manifest):
    return hashlib.sha256(json.dumps({key: value for key, value in manifest.items()
                                    if key != "identity"}, sort_keys=True).encode()).hexdigest()


def freeze(execution, artifact):
    """Write the authoritative execution record outside its frozen checkout."""
    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.models.registry import model_parameter_count

    execution, artifact = execution.resolve(), artifact.resolve()
    if execution != SOURCE_ROOT:
        raise ValueError("Run this adapter from the execution checkout being frozen")
    if execution == artifact or execution in artifact.parents:
        raise ValueError("Durable artifacts must live outside the execution checkout")
    source = source_identity(execution)
    manifest = json.loads((execution / MANIFEST).read_text())
    if manifest["status"] != "planned_not_qualified":
        raise ValueError("Freeze requires the unexecuted planning record")
    destination = artifact / MANIFEST
    if destination.exists():
        raise FileExistsError(f"Refusing to replace an existing experiment manifest: {destination}")
    for item in manifest["data_identity"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Dataset manifest differs from the retained heatmap corpus")
    manifest.update(source_commit=source, source_status="frozen", status="frozen_not_qualified",
                    execution_root=str(execution), artifact_root=str(artifact))
    manifest["dependency_lock"]["sha256"] = sha256(execution / manifest["dependency_lock"]["path"])
    for run in manifest["runs"]:
        recipe = run["resolved_config"]
        original = execution / recipe["path"]
        if sha256(original) != recipe["sha256"]:
            raise ValueError(f"Planning recipe changed: {original}")
        config = load_config(original)
        if model_parameter_count(config.model) != run["parameter_count"]:
            raise ValueError(f"Parameter count changed: {run['name']}")
        target = artifact / recipe["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != original.read_bytes():
            raise FileExistsError(f"Refusing to replace recipe: {target}")
        target.write_bytes(original.read_bytes())
        run["source_commit"] = source
    manifest["identity"] = identity(manifest)
    write_json(destination, manifest)
    print(destination)


def verify(path):
    manifest = json.loads(Path(path).read_text())
    if (manifest.get("format") != "functional-memory-gla-key-dimension"
            or manifest.get("identity") != identity(manifest)
            or manifest.get("source_status") != "frozen"):
        raise ValueError("Expected a frozen, checksummed GLA execution manifest")
    execution, artifact = Path(manifest["execution_root"]), Path(manifest["artifact_root"])
    if execution.resolve() != SOURCE_ROOT:
        raise ValueError("Imported study code is not from the recorded execution checkout")
    if source_identity(execution) != manifest["source_commit"]:
        raise ValueError("Execution source differs from the frozen manifest")
    if sha256(execution / manifest["dependency_lock"]["path"]) != manifest["dependency_lock"]["sha256"]:
        raise ValueError("Dependency lock changed")
    for item in manifest["data_identity"].values():
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError("Dataset manifest changed")
    for run in manifest["runs"]:
        if sha256(artifact / run["resolved_config"]["path"]) != run["resolved_config"]["sha256"]:
            raise ValueError(f"Recipe changed: {run['name']}")
    return manifest


def select_run(manifest, name):
    return next(run for run in manifest["runs"] if run["name"] == name)


def evidence_root(manifest):
    return Path(manifest["artifact_root"]) / RECORD / "validation" / manifest["identity"]


@contextmanager
def run_lock(manifest, name, purpose):
    """Exclude concurrent writers; the OS releases locks after preemption."""
    path = Path(manifest["artifact_root"]) / ".cache" / EXPERIMENT / "locks" / f"{name}-{purpose}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another {purpose} process owns this run: {name}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def runtime():
    from scaffolds.transformer.training.distributed import runtime_provenance
    result = runtime_provenance()
    if len(result["devices"]) != 4 or any(device["platform"] != "gpu" for device in result["devices"]):
        raise ValueError("Qualification and production require one allocation with four GPUs")
    result["kernel_environment"] = {key: os.environ.get(key) for key in ("XLA_FLAGS",)}
    return result


def worker(path, name, mode, output):
    manifest = verify(path)
    select_run(manifest, name)
    with run_lock(manifest, name, "production" if mode == "train" else "pilot"):
        _worker(manifest, name, mode, output)


def _worker(manifest, name, mode, output):
    from scaffolds.transformer.configs import load_config
    from scaffolds.transformer.training.loop import run_training
    from scaffolds.transformer.models.registry import model_parameter_count

    run = select_run(manifest, name)
    config = load_config(Path(manifest["artifact_root"]) / run["resolved_config"]["path"])
    config.train.source_commit = manifest["source_commit"]
    config.model.attention_backend = "fused"
    if model_parameter_count(config.model) != run["parameter_count"]:
        raise ValueError("Resolved model parameter count differs from provenance")
    hardware = runtime()
    if mode == "train":
        success_path = evidence_root(manifest) / name / "success.json"
        success = json.loads(success_path.read_text())
        if (success.get("identity") != manifest["identity"] or success.get("status") != "passed"
                or success.get("runtime") != hardware):
            raise ValueError("No matching full-model qualification for this run and runtime")
        for item in success["evidence"]:
            if sha256(evidence_root(manifest) / item["path"]) != item["sha256"]:
                raise ValueError("Qualification evidence changed")
        config.train.workdir = Path(manifest["artifact_root"]) / run["workdir"]
        config.train.checkpoint_dir = Path(manifest["artifact_root"]) / run["checkpoint_dir"]
        result = run_training(config, resume=True)
    else:
        # Only evaluation/checkpoint cadence and output locations differ; the
        # production batch, context, optimizer horizon and model stay unchanged.
        config.train.workdir = evidence_root(manifest) / name / "pilot"
        config.train.checkpoint_dir = Path(manifest["artifact_root"]) / ".cache" / EXPERIMENT / manifest["identity"] / name
        config.train.eval_every, config.train.eval_batches, config.train.ckpt_every = 32, 1, 32
        result = run_training(config, resume=mode == "resume", stop_after_step=33 if mode == "resume" else 32,
                              audit_checkpoint=True)
    write_json(output, {**result, "runtime": hardware, "source_commit": manifest["source_commit"],
                        "identity": manifest["identity"], "slurm_job_id": os.environ.get("SLURM_JOB_ID")})


def preflight(path, name):
    manifest = verify(path)
    select_run(manifest, name)
    with run_lock(manifest, name, "qualification"):
        _preflight(path, manifest, name)


def _preflight(path, manifest, name):
    from scaffolds.transformer.studies.key_dimension_taylor.preflight import validate_resume_audit
    from scaffolds.transformer.studies.polynomial_attention.preflight import require_passing_tests

    destination = evidence_root(manifest) / name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "success.json").exists() or (destination / "pilot").exists():
        raise FileExistsError("Qualification evidence already exists; inspect it before repeating the pilot")
    junit = destination / "kernels.xml"
    env = dict(os.environ, JAX_PLATFORMS="cuda")
    # Isolated bridge tests execute one-device operations. The production
    # workers below retain all four devices and exercise the mapped runtime.
    test_env = dict(env, CUDA_VISIBLE_DEVICES=env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    subprocess.run([sys.executable, "-m", "pytest", "-c", "scaffolds/transformer/pyproject.toml",
                    "scaffolds/transformer/tests", "-m", "gpu", "-k", "gla", f"--junitxml={junit}"],
                   cwd=manifest["execution_root"], env=test_env, check=True)
    passing_tests = require_passing_tests(junit, minimum=1)
    for mode in ("pilot", "resume"):
        subprocess.run([sys.executable, "-m", MODULE, "worker", "--manifest", str(Path(path).resolve()),
                        "--run-name", name, "--mode", mode, "--output", str(destination / f"{mode}.json")],
                       cwd=manifest["execution_root"], env=env, check=True)
    initial, resumed = [json.loads((destination / f"{mode}.json").read_text()) for mode in ("pilot", "resume")]
    validate_resume_audit(initial, resumed)
    if (initial["start_step"], initial["final_step"], resumed["start_step"], resumed["final_step"]) != (0, 32, 32, 33):
        raise ValueError("Qualification must execute a fresh 32-step pilot and a resumed update")
    if initial["runtime"] != resumed["runtime"] or any(result["attention_backend"] != "fused" for result in (initial, resumed)):
        raise ValueError("Qualification runtime or attention backend changed")
    write_json(destination / "success.json", {
        "status": "passed", "identity": manifest["identity"], "runtime": initial["runtime"],
        "kernel_tests_passed": passing_tests,
        "pilot_overrides": {"eval_every": 32, "eval_batches": 1, "ckpt_every": 32},
        "source_commit": manifest["source_commit"], "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "evidence": [{"path": str(item.relative_to(evidence_root(manifest))), "sha256": sha256(item)}
                     for item in (junit, destination / "pilot.json", destination / "resume.json")],
    })


def submit(path, mode, name, *, node, gpu, dry_run):
    manifest = verify(path)
    selected = [select_run(manifest, name)] if name else manifest["runs"]
    if gpu not in ("h100", "b300"):
        raise ValueError("Select H100 or B300; every runtime still requires its own GLA qualification")
    for run in selected:
        if mode == "train" and not (evidence_root(manifest) / run["name"] / "success.json").exists():
            raise ValueError(f"Missing qualification: {run['name']}")
        log = Path(manifest["artifact_root"]) / "results/logs" / EXPERIMENT / run["name"]
        command = ["sbatch", "--parsable", "--partition=sweep", "--time=7-00:00:00",
                   *(["--requeue"] if mode == "train" else ["--no-requeue"]),
                   "--nodes=1", "--ntasks=1", "--cpus-per-task=32", "--mem=256G", f"--gres=gpu:{gpu}:4",
                   f"--nodelist={node}", f"--job-name=gla-{mode}-{run['qk_head_dim']}",
                   f"--output={log}/slurm-%j.out", f"--error={log}/slurm-%j.out",
                   str(Path(manifest["execution_root"]) / "scaffolds/transformer/studies/gla_key_dimension/launch.sh"),
                   str(Path(path).resolve()), mode, run["name"]]
        if dry_run:
            import shlex
            print(shlex.join(command))
        else:
            log.mkdir(parents=True, exist_ok=True)
            job = subprocess.check_output(command, text=True).strip()
            write_json(Path(manifest["artifact_root"]) / RECORD / "provenance/execution" / f"submission-{job}.json",
                       {"job_id": job, "command": command, "mode": mode, "run_name": run["name"],
                        "identity": manifest["identity"]})
            print(job)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "preflight", "train", "submit", "worker"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--execution-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-name")
    parser.add_argument("--mode", choices=("preflight", "train", "pilot", "resume"), default="preflight")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--node", default="dogo")
    parser.add_argument("--gpu", default="h100")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "freeze":
        if args.artifact_root is None:
            parser.error("freeze requires --artifact-root")
        freeze(args.execution_root, args.artifact_root)
    elif args.manifest is None:
        parser.error("this command requires --manifest")
    elif args.command == "submit":
        if args.mode not in ("preflight", "train"):
            parser.error("submit --mode must be preflight or train")
        submit(args.manifest, args.mode, args.run_name, node=args.node, gpu=args.gpu, dry_run=args.dry_run)
    elif args.run_name is None:
        parser.error("this command requires --run-name")
    elif args.command == "preflight":
        preflight(args.manifest, args.run_name)
    elif args.command == "train":
        manifest = verify(args.manifest)
        output = Path(manifest["artifact_root"]) / RECORD / "runs" / args.run_name / "training_result.json"
        worker(args.manifest, args.run_name, "train", output)
    elif args.output is None or args.mode not in ("pilot", "resume"):
        parser.error("worker requires --output and --mode pilot|resume")
    else:
        worker(args.manifest, args.run_name, args.mode, args.output)


if __name__ == "__main__":
    main()

"""Allocate the LR pilots, select their common LR, and submit the requested comparison.

Every training job uses the shared launcher, acceptance, trainer, and records. The
dependent advance job runs only after all pilots exit; it freezes all comparison
recipes in one commit before submitting any of them. It never changes a pilot recipe.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import yaml

from scaffolds.records import create_run, load_run, load_study, regenerate_index, update_record, validate_record
from scaffolds.records.selections import refresh_selection
from scaffolds.transformer.configs import config_to_dict, load_config, validate_config
from scaffolds.transformer.configs.loader import config_dicts_equal

STUDY = Path(__file__).resolve().parent
ROOT = STUDY.parents[1]
PEAKS = (5e-4, 1e-3, 2e-3)
PILOT_STEPS = 1500
COMPARISONS = (("exp", None), *((kind, degree) for degree in (8, 12)
                              for kind in ("fixed", "static", "query", "query_key")))


def write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False))


def metrics(run: Path) -> list[dict]:
    path = run / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _recipe_signature(config, *, ignore_lr: bool = False) -> dict:
    signature = config_to_dict(config)
    if ignore_lr:
        for key in ("learning_rate", "min_learning_rate"):
            signature["optim"].pop(key)
    for key in ("workdir", "checkpoint_dir", "wandb_run_name", "wandb_dir"):
        signature["train"].pop(key, None)
    return signature


def variant_config(base: dict, kind: str, degree: int | None) -> dict:
    if (kind, degree) not in COMPARISONS:
        raise ValueError(f"Unknown comparison: {kind}, degree={degree}")
    config = copy.deepcopy(base)
    model = config["model"]
    model.update(attention_degree=degree, attention_normalization=False,
                 attention_output_scaling="head_rms_norm",
                 attention_backend="fused" if kind in ("exp", "query_key") else "mosaic")
    if kind in ("static", "query", "query_key"):
        model.update(attention_coefficients="learned", attention_coefficient_sharing="per_head",
                     attention_coefficient_scale="factorial")
    if kind in ("query", "query_key"):
        model.update(attention_coefficient_conditioning=kind, attention_coefficient_activation="tanh",
                     attention_coefficient_amplitude_scale="factorial",
                     attention_coefficient_amplitude_initial=0.0,
                     attention_coefficient_projection_initial_std=1 / math.sqrt(model["hidden_size"]),
                     attention_coefficient_tanh_bias=False)
    return config


def _allocate(config: dict, label: str, phase: str) -> Path:
    config = copy.deepcopy(config)
    config["train"].update(source_commit="auto", wandb_group=STUDY.name,
                           wandb_run_name=label, wandb_dir=None)
    existing = [r for r in load_study(STUDY)["runs"] if r.get("label") == label]
    if existing:
        if len(existing) != 1:
            raise ValueError(f"Ambiguous label: {label}")
        run = STUDY / "runs" / existing[0]["run_id"]
        previous = config_to_dict(load_config(run / "config.yaml"))
        expected = copy.deepcopy(config)
        for key in ("workdir", "checkpoint_dir"):
            expected["train"][key] = previous["train"][key]
        if existing[0].get("phase") != phase or not config_dicts_equal(expected, previous):
            raise ValueError(f"Existing label {label!r} has a different recipe or phase; allocate a fresh run ID")
        return run
    run = create_run(STUDY, config, label)
    validate_config(load_config(run / "config.yaml"))
    update_record(run, {"home_cluster": "kennel", "phase": phase})
    return run


def prepare_pilots() -> list[Path]:
    selected = STUDY / "results/lr_selection.yaml"
    if selected.exists() and yaml.safe_load(selected.read_text()).get("canonical_learning_rate") is not None:
        raise ValueError("The LR selection is already finalized; use report to refresh its results")
    if not (STUDY / "sweep.yaml").exists():
        base = load_config(STUDY / "recipe.yaml")
        manifests = {split: {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
                     for split, path in (("train", base.data.train_manifest), ("eval", base.data.eval_manifest))}
        write_yaml(STUDY / "sweep.yaml", {
            "schema_version": 1, "experiment": STUDY.name, "recipe_version": "transformer_200m_4b_v1",
            "home_cluster": "kennel", "artifact_root": "/data/cl/u/rafah/home/projects/functional-memory",
            "data_manifests": manifests,
            "dependency_lock_sha256": hashlib.sha256((ROOT / "scaffolds/transformer/uv.lock").read_bytes()).hexdigest(),
            "run_ids": [],
        })
    base = config_to_dict(load_config(STUDY / "recipe.yaml"))
    runs = []
    for peak in PEAKS:
        config = copy.deepcopy(base)
        config["optim"].update(learning_rate=peak, min_learning_rate=peak * .1)
        runs.append(_allocate(config, f"softmax-lr-{peak:g}", "lr_pilot"))
    (STUDY / "results").mkdir(exist_ok=True)
    selection = {"study": STUDY.name, "pilot_steps": PILOT_STEPS,
                 "selection_rule": "lowest_step_1500_validation_nll_lower_peak_on_exact_tie",
                 "canonical_learning_rate": None, "runs": [
                     {"study": STUDY.name, "run_id": run.name, "label": load_run(run)["label"], "role": "lr_pilot"}
                     for run in runs]}
    write_yaml(STUDY / "results/lr_selection.yaml", selection)
    refresh_selection(STUDY / "results")
    regenerate_index(STUDY)
    return runs


def choose_lr(runs: list[Path]) -> tuple[float, list[dict]]:
    """Select only terminal pilot measurements; refuse partial/operationally failed pilots."""
    candidates = []
    common_recipe = None
    for run in runs:
        record = load_run(run)
        config = load_config(run / "config.yaml")
        from experiments.transformer_200m_4b.acceptance import check_config
        check_config(config)
        if not config.model.attention_normalization or config.model.attention_degree is not None:
            raise ValueError("LR pilots must use the predeclared softmax control")
        signature = _recipe_signature(config, ignore_lr=True)
        if common_recipe is not None and signature != common_recipe:
            raise ValueError("LR pilots differ in settings other than learning rate")
        common_recipe = signature
        if record.get("home_cluster") != "kennel":
            raise ValueError(f"Pilot {run.name} does not belong to the setup's home cluster kennel")
        if record.get("acceptance", {}).get("summary", {}).get("status") != "passed":
            raise ValueError(f"Pilot {run.name} has no successful acceptance evidence")
        checked = validate_record(run, checkpoint_hashes=False)
        if checked.get("unavailable"):
            raise ValueError(f"Pilot {run.name} has unavailable provenance artifacts on its home cluster")
        peak = config.optim.learning_rate
        if record["status"] == "numerical_failure":
            candidates.append({"run_id": run.name, "peak": peak, "status": "numerical_failure", "nll": None})
            continue
        if record["status"] != "paused":
            raise ValueError(f"Pilot {run.name} is {record['status']}; all pilots must finish or conclude a numerical failure")
        validations = [row for row in metrics(run) if row.get("event") == "validation"]
        rows = [row for row in validations if row["step"] == PILOT_STEPS]
        if len(rows) != 1 or not math.isfinite(rows[0]["validation_nll"]):
            raise ValueError(f"Pilot {run.name} lacks one finite step-{PILOT_STEPS} validation")
        if record.get("step") != PILOT_STEPS or max(row["step"] for row in validations) != PILOT_STEPS:
            raise ValueError(f"Pilot {run.name} did not stop at the predeclared step {PILOT_STEPS}")
        candidates.append({"run_id": run.name, "peak": peak, "status": "paused", "nll": rows[0]["validation_nll"]})
    if sorted(row["peak"] for row in candidates) != list(PEAKS):
        raise ValueError("Selection requires exactly the three predeclared LR candidates")
    eligible = [row for row in candidates if row["nll"] is not None]
    if not eligible:
        raise ValueError("All LR pilots failed numerically; no canonical LR can be selected")
    winner = min(eligible, key=lambda row: (row["nll"], row["peak"]))
    return winner["peak"], candidates


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _queued_jobs(jobs: list[str]) -> str:
    return (subprocess.check_output(["squeue", "--noheader", "--jobs", ",".join(jobs), "--format=%i"],
                                    text=True).strip() if jobs else "")


def _submit_report(runs: list[Path], jobs: list[str]) -> str:
    previous = {load_run(run).get("report_job_id") for run in runs} - {None}
    if previous and _queued_jobs(sorted(previous)):
        if len(previous) != 1:
            raise ValueError("Comparison runs reference several active report jobs")
        return previous.pop()
    command = ["sbatch", "--parsable", "--partition=sweep", "--nodes=1", "--ntasks=1", "--cpus-per-task=2",
               "--mem=8G", "--time=00:30:00", f"--dependency=afterany:{':'.join(jobs)}",
               f"--job-name={STUDY.name}-report", f"--output={STUDY.relative_to(ROOT)}/report.log",
               str((STUDY / "advance.sbatch").relative_to(ROOT)), "report"]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected report sbatch output: {job}")
    for run in runs:
        update_record(run, {"report_job_id": job})
    return job


def _submit(run: Path, source: str, *, stop: int | None = None, dependency: str | None = None) -> str:
    record = load_run(run)
    if record.get("submission"):
        previous = record["submission"]
        job = previous["slurm_job_id"]
        if previous.get("stop_after_step") != stop:
            raise ValueError(f"Run {run.name} already has a submission with a different stopping step")
        if record["status"] in ("failed", "acceptance_failed", "acceptance_numerical_failure"):
            raise ValueError(f"Run {run.name} needs its failed job {job} diagnosed and explicitly requeued")
        if record["status"] not in ("paused", "complete", "numerical_failure") and not _queued_jobs([job]):
            raise ValueError(f"Run {run.name} has a stale submission {job}; reconcile its outcome before retrying")
        return job
    relative = run.relative_to(ROOT)
    # The current query-key kernel needs the faster GPU and a longer allocation;
    # these are execution choices, not changes to the scientific recipe.
    query_key = load_config(run / "config.yaml").model.attention_coefficient_conditioning == "query_key"
    gpu_types = "B300" if query_key else "B300,H100"
    gres = "gpu:b300:1" if query_key else "gpu:1"
    allocation = "7-00:00:00" if query_key else "2-00:00:00"
    command = ["sbatch", "--parsable", "--partition=sweep", f"--gres={gres}", "--cpus-per-task=8",
               "--mem=96G", f"--time={allocation}", f"--job-name={run.name}", f"--output={relative}/training.log"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command += ["scaffolds/transformer/launchers/kennel.sbatch",
                f"experiments.{STUDY.name}.acceptance", f"{relative}/config.yaml", source, "1", gpu_types]
    if stop is not None:
        command.append(str(stop))
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected sbatch output: {job}")
    update_record(run, {"status": "pending", "submission": {"slurm_job_id": job, "launch_commit": source,
                  "gpu_count": 1, "partition": "sweep", "stop_after_step": stop, "command": command}})
    return job


def submit_pilots() -> None:
    runs = [STUDY / "runs" / row["run_id"] for row in load_study(STUDY)["runs"] if row.get("phase") == "lr_pilot"]
    if len(runs) != 3:
        raise ValueError("Prepare and commit exactly three pilots first")
    source = _git("rev-parse", "HEAD")
    from scaffolds.launch.frozen import changes
    if changes(ROOT, source, STUDY.name, "transformer"):
        raise ValueError("Commit all execution code and pilot inputs before submitting")
    jobs = [_submit(run, source, stop=PILOT_STEPS) for run in runs]
    previous_coordinators = {load_run(run).get("lr_selection_job_id") for run in runs} - {None}
    if previous_coordinators and _queued_jobs(sorted(previous_coordinators)):
        print(json.dumps({"pilots": jobs, "selection_jobs": sorted(previous_coordinators), "reused": True}))
        return
    if any(row.get("phase") == "comparison" for row in load_study(STUDY)["runs"]):
        print(json.dumps({"pilots": jobs, "comparison_already_allocated": True}))
        return
    # No GPU reservation for the coordinator, and no modification until every pilot exits.
    command = ["sbatch", "--parsable", "--partition=sweep", "--nodes=1", "--ntasks=1", "--cpus-per-task=2",
               "--mem=8G", "--time=00:30:00", f"--dependency=afterany:{':'.join(jobs)}",
               f"--job-name={STUDY.name}-select-lr", f"--output={STUDY.relative_to(ROOT)}/selection.log",
               str((STUDY / "advance.sbatch").relative_to(ROOT))]
    job = subprocess.check_output(command, cwd=ROOT, text=True).strip().split(";")[0]
    if not job.isdigit():
        raise RuntimeError(f"Unexpected coordinator sbatch output: {job}")
    for run in runs:
        update_record(run, {"lr_selection_job_id": job})
    regenerate_index(STUDY)
    print(json.dumps({"pilots": jobs, "selection_job": job}))


def advance(*, submit: bool) -> list[Path]:
    rows = load_study(STUDY)["runs"]
    pilots = [STUDY / "runs" / row["run_id"] for row in rows if row.get("phase") == "lr_pilot"]
    jobs = [row["submission"]["slurm_job_id"] for row in rows if row.get("submission")]
    active = _queued_jobs(jobs)
    if active:
        raise ValueError(f"Study jobs can still execute/requeue: {active}")
    peak, outcomes = choose_lr(pilots)
    # A source edit during training requires review before new production jobs.
    source = _git("rev-parse", "HEAD")
    from scaffolds.launch.frozen import changes
    if changes(ROOT, source, STUDY.name, "transformer"):
        raise ValueError("Uncommitted execution changes; review before advancing")
    if not any(row.get("phase") == "comparison" for row in rows):
        launch_commits = {load_run(run).get("submission", {}).get("launch_commit") for run in pilots}
        if None in launch_commits or len(launch_commits) != 1:
            raise ValueError("LR pilots require one recorded common launch commit")
        if changes(ROOT, launch_commits.pop(), STUDY.name, "transformer"):
            raise ValueError("Execution changed since the LR pilots launched; review before advancing")
    baseline = load_config(STUDY / "recipe.yaml")
    if _recipe_signature(baseline, ignore_lr=True) != _recipe_signature(load_config(pilots[0] / "config.yaml"), ignore_lr=True):
        raise ValueError("The setup recipe differs from the frozen LR pilot protocol")
    selection_path = STUDY / "results/lr_selection.yaml"
    selection = yaml.safe_load(selection_path.read_text())
    selection.update(canonical_learning_rate=peak, canonical_min_learning_rate=peak * .1,
                     selected_run_id=min((r for r in outcomes if r["nll"] is not None),
                                         key=lambda r: (r["nll"], r["peak"]))["run_id"])
    write_yaml(selection_path, selection)
    recipe = yaml.safe_load((STUDY / "recipe.yaml").read_text())
    recipe["optim"].update(learning_rate=peak, min_learning_rate=peak * .1)
    write_yaml(STUDY / "recipe.yaml", recipe)
    base = config_to_dict(load_config(STUDY / "recipe.yaml"))
    runs = [_allocate(variant_config(base, kind, degree), f"{kind}" + (f"-m{degree}" if degree else ""), "comparison")
            for kind, degree in COMPARISONS]
    write_yaml(STUDY / "results/selection.yaml", {"study": STUDY.name, "recipe_version": "transformer_200m_4b_v1",
               "training_steps": 7570, "canonical_learning_rate": peak, "runs": [
                   {"study": STUDY.name, "run_id": run.name, "label": load_run(run)["label"], "role": "comparison"}
                   for run in runs]})
    refresh_selection(STUDY / "results")
    regenerate_index(STUDY)
    from experiments.transformer_200m_4b.results.plot import regenerate
    regenerate()
    if submit:
        outside = [path for path in _git("diff", "--cached", "--name-only").splitlines()
                   if not path.startswith(STUDY.relative_to(ROOT).as_posix() + "/")]
        if outside:
            raise ValueError(f"Refusing to commit unrelated staged paths: {outside}")
        # Every path is inside the new study; ignored logs/checkpoints are never staged.
        files = [p.relative_to(ROOT).as_posix() for p in STUDY.rglob("*") if p.is_file()
                 and p.suffix in (".yaml", ".json", ".jsonl", ".csv", ".md", ".png", ".pdf")
                 and not any(part in ("checkpoints", ".runtime", "wandb", "__pycache__") for part in p.parts)]
        _git("add", "--", *sorted(files))
        if _git("diff", "--cached", "--name-only"):
            _git("commit", "-m", "Select the 4B recipe learning rate and freeze the nine attention comparisons")
        source = _git("rev-parse", "HEAD")
        # These runs are scientifically independent; let Slurm schedule each allocation.
        jobs = [_submit(run, source) for run in runs]
        _submit_report(runs, jobs)
        regenerate_index(STUDY)
    print(json.dumps({"canonical_learning_rate": peak, "runs": [r.name for r in runs]}))
    return runs


def report() -> None:
    """Refresh only mutable run summaries, scientific pins, and derived figures."""
    refresh_selection(STUDY / "results")
    regenerate_index(STUDY)
    from experiments.transformer_200m_4b.results.plot import regenerate
    regenerate()


@contextmanager
def pipeline_lock():
    """Serialize CLI mutations; use the results directory, separate from records' study lock."""
    directory = STUDY / "results"
    directory.mkdir(exist_ok=True)
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another pipeline command is already running") from error
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare-pilots", "submit-pilots", "advance", "report"))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    with pipeline_lock():
        if args.command == "prepare-pilots":
            print("\n".join(map(str, prepare_pilots())))
        elif args.command == "submit-pilots":
            submit_pilots()
        elif args.command == "report":
            report()
        else:
            advance(submit=args.submit)


if __name__ == "__main__":
    main()

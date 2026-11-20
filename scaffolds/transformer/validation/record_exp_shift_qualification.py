"""Write runs/<name>/record.json for one finished qualify_exp_shift.sbatch job, from its log, JUnit and throughput
report (logs/exp-shift-<job>[.log]) and Slurm accounting. Run from the checkout that submitted the job:

    python3 scaffolds/transformer/validation/record_exp_shift_qualification.py <job-id> <record-name>

Standard library only. The launcher and test-module hashes are those of the job's launch commit.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
VALIDATION = ROOT / "scaffolds/transformer/validation"
LAUNCHER = "scaffolds/transformer/validation/qualify_exp_shift.sbatch"
TEST_MODULES = ("scaffolds/transformer/tests/test_unnormalized_fused_attention.py",
                "scaffolds/transformer/tests/test_exp_row_max_shift.py")
SCOPE = ("The fused Pallas-Triton unnormalized exponential shifted by its causal row maximum "
         "(fused_unnormalized_attention, exp_shift='row_max'): outputs, row maxima and VJPs at Q/K widths 4, 8, 16 and "
         "64 with value width 64 at 2,048 tokens, at ordinary scores and at row maxima of 180-280, beyond FP32's e^88.7, "
         "against the shifted blocked scan and dense FP64 (129-token, one-head prefix); with head RMSNorm's offset "
         "against the FP64 function at row maxima of 135-210; FP32 inputs at row maxima up to 190; future-token "
         "invariance, nonfinite and extreme scores, underflow, repeatability, and a two-layer model on the fused kernel "
         "against the blocked scan; the blocked strips and scan at production precision against FP64 with and without "
         "a cumulative decay; and every unshifted case of the fused module.")


def _artifact(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(VALIDATION)), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _at_commit(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def main(job: str, name: str) -> Path:
    log = VALIDATION / f"logs/exp-shift-{job}.log"
    junit = VALIDATION / f"logs/exp-shift-{job}/gpu-tests.xml"
    throughput = VALIDATION / f"logs/exp-shift-{job}/throughput.json"
    text = log.read_text()
    lines = text.splitlines()
    header = next(index for index, line in enumerate(lines) if line.startswith("name, uuid"))
    gpu_name, uuid, bus, driver, capability = (field.strip() for field in lines[header + 1].split(","))
    environment, _, commit = re.search(r"Environment (\S+), source (\w+), launch commit (\w+)", text).groups()
    fields = subprocess.run(["sacct", "-j", job, "-X", "-n", "-P", "-o",
                             "State,Elapsed,ExitCode,NodeList,Partition,Start,End"],
                            capture_output=True, text=True, check=True).stdout.strip().splitlines()[0].split("|")
    state, elapsed, exit_code, node, partition, start, end = fields
    hours, minutes, seconds = (int(value) for value in elapsed.split(":"))
    tree = ET.parse(junit)
    cases = list(tree.iter("testcase"))
    unsuccessful = [f"{case.get('classname')}::{case.get('name')}" for case in cases
                    if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))]
    passed = Counter(case.get("classname").split(".")[-1] for case in cases
                     if f"{case.get('classname')}::{case.get('name')}" not in unsuccessful)
    record_job = {
        "artifacts": {"junit": _artifact(junit), "log": _artifact(log)},
        "elapsed_seconds": 3600 * hours + 60 * minutes + seconds,
        "environment": f"scaffolds/transformer/{environment}", "exit_code": exit_code,
        "failed_cases": unsuccessful, "finished_at": end,
        "gpu": {"bus_id": bus, "compute_capability": capability, "driver_version": driver, "name": gpu_name,
                "uuid": uuid},
        "junit_seconds": round(sum(float(suite.get("time", 0)) for suite in tree.iter("testsuite")), 3),
        "node": node, "partition": partition, "passed_by_module": dict(sorted(passed.items())),
        "row_max_shift_cases_passed": sum(1 for case in cases if ("shift" in case.get("name")
                                                                  or "strips_and_scan" in case.get("name"))
                                          and f"{case.get('classname')}::{case.get('name')}" not in unsuccessful),
        "slurm_job_id": job, "slurm_state": state, "source_commit": commit, "started_at": start,
        "tests": {"passed": sum(passed.values()), **({"unsuccessful": len(unsuccessful)} if unsuccessful else {})},
    }
    if throughput.exists():
        report = json.loads(throughput.read_text())
        record_job["artifacts"]["throughput"] = _artifact(throughput)
        for program in ("forward_and_vjp", "forward"):
            record_job[f"{program}_median_ms"] = {
                f"d_K={case['qk_head_dim']}, {case['exp_shift']}": case[program]["median_ms"] for case in report["cases"]}
    frozen = subprocess.run([sys.executable, "scaffolds/launch/frozen.py", "hash", "--launch-commit", commit,
                             "--experiment", "transformer_pp", "--scaffold", "transformer"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    record = {
        "asserted_relative_l2_bounds": {
            "bf16": {"output": 0.006, "dQ": 0.025, "dK": 0.025, "dV": 0.015,
                     "row_maximum": "1e-5 relative, and 1e-5 times max(1, |m|) absolute"},
            "fp32": {"output": "3e-5 times max(1, largest row maximum / 30)",
                     "dQ, dK, dV": "1e-4 times max(1, largest row maximum / 30)"}},
        "date": end[:10], "format": "fused-exponential-row-max-shift-qualification",
        "gpu": f"{gpu_name} ({node})", "home_cluster": "kennel", "jobs": [record_job],
        "launcher": LAUNCHER, "launcher_sha256": hashlib.sha256(_at_commit(commit, LAUNCHER)).hexdigest(),
        "maximum_element_error_bound": ("3 times the relative-L2 bound times the largest absolute reference element, "
                                        "per batch and head"),
        "precisions": ["bf16 Q/K/V and gradients (training)", "fp32 Q/K/V (TF32x3 products)"],
        "schema_version": 1, "scope": SCOPE, "selection": ["-m", "gpu"],
        "source_commits": {commit: {
            "dependency_lock_sha256": hashlib.sha256(_at_commit(commit, "scaffolds/transformer/uv.lock")).hexdigest(),
            "frozen_execution_sha256": frozen}},
        "status": "passed" if not unsuccessful and state == "COMPLETED" else "failed",
        "test_module_sha256": {f"{Path(path).name}@{commit[:8]}": hashlib.sha256(_at_commit(commit, path)).hexdigest()
                               for path in TEST_MODULES},
        "tests": {"passed": sum(passed.values())},
    }
    output = VALIDATION / f"runs/{name}/record.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    path = main(*sys.argv[1:3])
    print(path, json.loads(path.read_text())["status"])

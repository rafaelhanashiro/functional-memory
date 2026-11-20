"""Write runs/<name>/record.json for one finished qualify_head_rms_prescale.sbatch job, from its log and JUnit report
(logs/head-rms-prescale-<job>[.log]) and Slurm accounting. Run from the checkout that submitted the job:

    python3 scaffolds/transformer/validation/record_head_rms_prescale_qualification.py <job-id> <record-name>

Standard library only. The launcher and test-module hashes are those of the job's launch commit. The measured
errors are the ones the cases recorded in the report (relative L2 against FP64).
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
LAUNCHER = "scaffolds/transformer/validation/qualify_head_rms_prescale.sbatch"
TEST_MODULES = ("scaffolds/transformer/tests/test_head_rms_prescale.py",
                "scaffolds/transformer/tests/test_mosaic_unnormalized.py")
SCOPE = ("Head RMSNorm's max-abs prescale (scale_attention_output, head_rms_prescale='max_abs') on the device's FP32 "
         "arithmetic: output and VJP of the measured probe (2,048 tokens, two heads, 64 channels) against host FP64 "
         "at output RMS 1e-25 to 1e30, with the plain evaluation beside it; its composition with the exponential's "
         "row shift at shifts of -60 to 2,000 and rows of 1e-30 to 1e25; the Mosaic kernels (fixed Taylor, static, "
         "query, softplus sum of squares with its input power) at 2,048 tokens and Q/K widths 4 and 64 into the "
         "prescaled normalization against dense FP64, values at their own scale and scaled by 1e16; one training "
         "gradient of two-layer BF16 models on those kernels with every head output above 1e15 against the same "
         "models in FP64; and the existing Mosaic case of the unprescaled head RMSNorm.")
ADMISSION_SKIP = "B300 does not admit the transformed kernels"


def _artifact(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.relative_to(VALIDATION)), "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data)}


def _at_commit(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "show", f"{commit}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout


def _properties(cases, name) -> list:
    """Every recorded value of one property, parsed."""
    return [json.loads(prop.get("value")) for case in cases for prop in case.iter("property")
            if prop.get("name") == name]


def _measured(cases) -> dict:
    """The errors the cases recorded: the probe by scale, and the largest of each other table."""
    measured = {}
    for probe in _properties(cases, "head_rms_probe"):
        prescaled = [max(entry["max_abs"][key] for key in ("output", "dy", "dgain")) for entry in probe]
        measured["probe"] = {
            "scales": len(probe),
            "prescaled_largest_relative_l2": max(prescaled),
            "prescaled_largest_radial_fraction_1_to_1e25": max(
                entry["max_abs"]["radial_fraction"] for entry in probe if 1.0 <= entry["rms"] <= 1e25),
            "plain_gradient_relative_l2_by_rms": {f"{entry['rms']:g}": entry["none"]["dy"] for entry in probe},
            "plain_radial_fraction_by_rms": {f"{entry['rms']:g}": entry["none"]["radial_fraction"] for entry in probe
                                             if entry["rms"] >= 1.0},
        }
    for table in _properties(cases, "head_rms_shifted_composition"):
        measured["shifted_composition"] = {
            "cases": len(table),
            "largest_relative_l2": max(max(entry[key] for key in ("output", "dy", "dgain")) for entry in table)}
    for name, key in (("head_rms_mosaic_kernel", "mosaic_kernels"), ("head_rms_mosaic_model", "mosaic_models")):
        for report in _properties(cases, name):
            label = f"{report.pop('family')}, d_K={report.pop('width')}"
            measured.setdefault(key, {})[label] = report
    return measured


def main(job: str, name: str) -> Path:
    log = VALIDATION / f"logs/head-rms-prescale-{job}.log"
    junit = VALIDATION / f"logs/head-rms-prescale-{job}/gpu-tests.xml"
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
    identity = lambda case: f"{case.get('classname')}::{case.get('name')}"  # noqa: E731
    failed = [identity(case) for case in cases if case.find("failure") is not None or case.find("error") is not None]
    skipped = {identity(case): case.find("skipped").get("message", "") for case in cases
               if case.find("skipped") is not None}
    unexpected = [case for case, message in skipped.items() if ADMISSION_SKIP not in message]
    passed = Counter(case.get("classname").split(".")[-1] for case in cases
                     if identity(case) not in failed and identity(case) not in skipped)
    record_job = {
        "artifacts": {"junit": _artifact(junit), "log": _artifact(log)},
        "elapsed_seconds": 3600 * hours + 60 * minutes + seconds,
        "environment": f"scaffolds/transformer/{environment}", "exit_code": exit_code,
        "failed_cases": failed, "finished_at": end,
        "gpu": {"bus_id": bus, "compute_capability": capability, "driver_version": driver, "name": gpu_name,
                "uuid": uuid},
        "junit_seconds": round(sum(float(suite.get("time", 0)) for suite in tree.iter("testsuite")), 3),
        "measured": _measured(cases),
        "node": node, "partition": partition, "passed_by_module": dict(sorted(passed.items())),
        "skipped_as_not_admitted": sorted(case.split("::", 1)[1] for case in skipped if case not in unexpected),
        "slurm_job_id": job, "slurm_state": state, "source_commit": commit, "started_at": start,
        "tests": {"passed": sum(passed.values()), "skipped": len(skipped),
                  **({"failed": len(failed)} if failed else {}),
                  **({"unexpected_skips": unexpected} if unexpected else {})},
    }
    frozen = subprocess.run([sys.executable, "scaffolds/launch/frozen.py", "hash", "--launch-commit", commit,
                             "--experiment", "none", "--scaffold", "transformer"],
                            cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip().splitlines()[-1]
    record = {
        "asserted_relative_l2_bounds": {
            "fp32_normalization": {"probe output, d/dy, d/dgain, every scale": 1e-6,
                                   "probe radial fraction, RMS 1 to 1e25": 1e-5,
                                   "shifted composition output, d/dy, d/dgain": 1e-6,
                                   "plain d/dy, RMS 9e14 to 1e18": "at least 0.05, radial fraction at least 0.99"},
            "mosaic_kernels_bf16": {"output": 0.02, "dQ": 0.06, "dK": 0.06, "dV": 0.05, "dgain": 0.05,
                                    "dcoefficients": 0.06,
                                    "values scaled by 1e16": "twice the unscaled error plus 0.01",
                                    "plain dQ, values scaled by 1e16": "at least 0.15"},
            "mosaic_models_bf16": {"each parameter group, unscaled": 0.1,
                                   "values scaled by 1e16": "three times the unscaled error plus 0.01, at most 0.15",
                                   "plain Q/K, values scaled by 1e16": "at least 0.5"}},
        "date": end[:10], "format": "head-rms-norm-prescale-qualification",
        "gpu": f"{gpu_name} ({node})", "home_cluster": "kennel", "jobs": [record_job],
        "launcher": LAUNCHER, "launcher_sha256": hashlib.sha256(_at_commit(commit, LAUNCHER)).hexdigest(),
        "precisions": ["fp32 head RMSNorm (the model's)", "bf16 Q/K/V, attention outputs and gradients (training)"],
        "schema_version": 1, "scope": SCOPE, "selection": ["-m", "gpu"],
        "source_commits": {commit: {
            "dependency_lock_sha256": hashlib.sha256(_at_commit(commit, "scaffolds/transformer/uv.lock")).hexdigest(),
            "frozen_execution_sha256": frozen}},
        "status": "passed" if not failed and not unexpected and state == "COMPLETED" else "failed",
        "test_module_sha256": {f"{Path(path).name}@{commit[:8]}": hashlib.sha256(_at_commit(commit, path)).hexdigest()
                               for path in TEST_MODULES},
        "tests": {"passed": sum(passed.values()), "skipped_as_not_admitted": len(skipped) - len(unexpected)},
    }
    output = VALIDATION / f"runs/{name}/record.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return output


if __name__ == "__main__":
    path = main(*sys.argv[1:3])
    print(path, json.loads(path.read_text())["status"])

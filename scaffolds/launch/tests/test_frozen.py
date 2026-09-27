"""The frozen execution set ignores outputs and analysis edits but not execution changes."""
from pathlib import Path
import subprocess

import pytest

from scaffolds.launch import frozen


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    for path, text in {
        "scaffolds/transformer/train.py": "trainer\n",
        "scaffolds/transformer/validation/README.md": "qualification\n",
        "scaffolds/records/__init__.py": "records\n",
        "scaffolds/launch/frozen.py": "checker\n",
        "experiments/demo/sweep.yaml": "sweep\n",
        "experiments/demo/runs/r0001/config.yaml": "recipe\n",
        "experiments/demo/acceptance.py": "spec\n",
        "experiments/demo/launch.sbatch": "launcher\n",
        "experiments/demo/runtime.sh": "environment\n",
        "results/demo/plot.py": "plot\n",
        "experiments/demo/README.md": "notes\n",
    }.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text)
    git(root.parent, "init", "-q", str(root))
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "launch")
    return root, git(root, "rev-parse", "HEAD")


def test_outputs_analysis_and_validation_edits_do_not_touch_the_frozen_set(tmp_path):
    root, launch = repository(tmp_path)
    before = frozen.frozen_hash(root, launch, "demo", "transformer")
    for path in ("experiments/demo/runs/r0001/metrics.jsonl", "experiments/demo/runs/r0001/training.log",
                 "experiments/demo/runs/r0001/record.json", "experiments/demo/runs/r0001/checkpoints/2/array"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("output\n")
    (root / "results/demo/plot.py").write_text("new plot\n")
    (root / "experiments/demo/README.md").write_text("findings\n")
    (root / "scaffolds/transformer/validation/README.md").write_text("new qualification\n")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "analysis")
    assert frozen.changes(root, launch, "demo", "transformer") == []
    assert frozen.frozen_hash(root, "HEAD", "demo", "transformer") == before
    assert frozen.main(["check", "--launch-commit", launch, "--experiment", "demo",
                        "--scaffold", "transformer", "--root", str(root)]) == 0


def test_committed_uncommitted_and_untracked_execution_changes_are_refused(tmp_path):
    root, launch = repository(tmp_path)
    (root / "experiments/demo/runs/r0001/config.yaml").write_text("changed recipe\n")
    assert frozen.changes(root, launch, "demo", "transformer") == ["experiments/demo/runs/r0001/config.yaml"]
    git(root, "checkout", "--", "experiments/demo/runs/r0001/config.yaml")
    (root / "scaffolds/transformer/new_module.py").write_text("untracked\n")
    assert frozen.changes(root, launch, "demo", "transformer") == ["scaffolds/transformer/new_module.py"]
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "scaffold change")
    assert frozen.changes(root, launch, "demo", "transformer") == ["scaffolds/transformer/new_module.py"]
    assert frozen.main(["check", "--launch-commit", launch, "--experiment", "demo",
                        "--scaffold", "transformer", "--root", str(root)]) == 1


@pytest.mark.parametrize("path", ["experiments/demo/sweep.yaml", "experiments/demo/acceptance.py",
                                 "experiments/demo/launch.sbatch", "experiments/demo/runtime.sh",
                                 "scaffolds/records/__init__.py", "scaffolds/launch/frozen.py"])
def test_all_shared_execution_inputs_are_frozen_and_hashed(tmp_path, path):
    root, launch = repository(tmp_path)
    before = frozen.frozen_hash(root, launch, "demo", "transformer")
    (root / path).write_text("changed\n")
    assert frozen.changes(root, launch, "demo", "transformer") == [path]
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "execution edit")
    assert frozen.frozen_hash(root, "HEAD", "demo", "transformer") != before


def test_untracked_new_run_recipe_is_rejected_before_launch(tmp_path):
    root, launch = repository(tmp_path)
    path = root / "experiments/demo/runs/r0002/config.yaml"
    path.parent.mkdir()
    path.write_text("new recipe\n")
    assert frozen.changes(root, launch, "demo", "transformer") == [path.relative_to(root).as_posix()]

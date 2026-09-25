"""The frozen execution set ignores outputs and analysis edits but not execution changes."""
from pathlib import Path
import subprocess

from scaffolds.launch import frozen


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def repository(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "repo"
    for path, text in {
        "scaffolds/transformer/train.py": "trainer\n",
        "scaffolds/transformer/validation/README.md": "qualification\n",
        "experiments/demo/configs/run.yaml": "recipe\n",
        "experiments/demo/adapter/execution/acceptance.py": "spec\n",
        "experiments/demo/adapter/analysis/plot.py": "plot\n",
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
    for path in ("experiments/demo/runs/run/metrics.jsonl", "experiments/demo/logs/run/slurm-1.out"):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("output\n")
    (root / "experiments/demo/adapter/analysis/plot.py").write_text("new plot\n")
    (root / "experiments/demo/README.md").write_text("findings\n")
    (root / "scaffolds/transformer/validation/README.md").write_text("new qualification\n")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "analysis")
    assert frozen.changes(root, launch, "demo", "transformer") == []
    assert frozen.frozen_hash(root, launch, "demo", "transformer") == before
    assert frozen.main(["check", "--launch-commit", launch, "--experiment", "demo",
                        "--scaffold", "transformer", "--root", str(root)]) == 0


def test_committed_uncommitted_and_untracked_execution_changes_are_refused(tmp_path):
    root, launch = repository(tmp_path)
    (root / "experiments/demo/configs/run.yaml").write_text("changed recipe\n")
    assert frozen.changes(root, launch, "demo", "transformer") == ["experiments/demo/configs/run.yaml"]
    git(root, "checkout", "--", "experiments/demo/configs/run.yaml")
    (root / "scaffolds/transformer/new_module.py").write_text("untracked\n")
    assert frozen.changes(root, launch, "demo", "transformer") == ["scaffolds/transformer/new_module.py"]
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "scaffold change")
    assert frozen.changes(root, launch, "demo", "transformer") == ["scaffolds/transformer/new_module.py"]
    assert frozen.main(["check", "--launch-commit", launch, "--experiment", "demo",
                        "--scaffold", "transformer", "--root", str(root)]) == 1

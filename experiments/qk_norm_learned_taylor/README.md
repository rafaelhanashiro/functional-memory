# QK-normalized learned Taylor attention

How do learned polynomial degree and row normalization affect cosine-score attention? Vary degree and row normalization with unrestricted learned polynomial coefficients.

[Results and limitations](../../results/qk_norm_learned_taylor/README.md) · [Run index](runs.csv) · [Study definition](sweep.yaml)

Each row in `runs.csv` identifies one execution. Its immutable input recipe is
`runs/<run_id>/config.yaml`; `record.json` contains provenance and artifact hashes.
Metrics and the retained `checkpoints/` and `training.log` are colocated there.
The home cluster is **kennel**; ignored artifacts remain at
`/data/cl/u/rafah/home/projects/functional-memory` on Kennel. Source commits in the
records preserve the exact implementations used for the reported results. Historical
recipes can differ from current defaults; a rerun with current code is a new execution.

## Run a new execution

Create an experiment worktree with `git worktree add -b codex/<study-rerun>
worktrees/<study-rerun> main`, enter it, and sync its own environment. Allocate the
run before launch; this example reuses a recorded recipe. To change settings, pass a
fully specified edited recipe instead. Create all sweep runs before committing the
frozen execution set.

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records create \
  --study experiments/qk_norm_learned_taylor \
  --config experiments/qk_norm_learned_taylor/runs/r13c144b8fd17/config.yaml --label rerun
```

The command prints the new run directory. Set `RUN` to its ID, commit the prepared
inputs, and submit from the worktree root after checking resources in
[CLUSTERS.md](../../CLUSTERS.md):

```bash
RUN=<new-run-id>
git add experiments/qk_norm_learned_taylor
git commit -m "Prepare qk_norm_learned_taylor execution"
sbatch --job-name="$RUN" --output="experiments/qk_norm_learned_taylor/runs/$RUN/training.log" \
  scaffolds/transformer/launchers/kennel.sbatch \
  experiments.qk_norm_learned_taylor.acceptance \
  "experiments/qk_norm_learned_taylor/runs/$RUN/config.yaml" "$(git rev-parse HEAD)" 4 B300,H100
```

The launcher runs acceptance before training and records each attempt. A requeue or
checkpoint resume uses the same ID, recipe, and launch commit. A fresh repetition
allocates a new ID. Do not change the frozen inputs while a job can run or requeue.
See the [shared runtime API](../../scaffolds/transformer/README.md) for acceptance,
backend resolution, and resume behavior.

Regenerate the derived index after collecting execution updates:

```bash
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m scaffolds.records index experiments/qk_norm_learned_taylor
```

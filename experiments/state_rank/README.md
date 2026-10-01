# State-rank comparisons

How concentrated are attention-state spectra in pretrained models compared with
independent reference initializations? The study measures Llama-3.2-1B’s implicit
softmax numerator state and Gated DeltaNet’s native recurrent state across all
layers, heads, and four fixed 512-token inputs. Each comparison has one pretrained
condition and five random seeds. The two original pretrained-only collections are
retained as reproduction baselines.

[Findings and figures](../../results/state_rank/README.md) · [Run index](runs.csv) · [Recipes and shared provenance](sweep.yaml)

The home cluster is **Kennel**. Numerical archives and one retained log per
comparison condition live under `runs/<run_id>/` at
`/data/cl/u/rafah/home/projects/functional-memory`, ignored by Git. Each run owns
its recipe and execution record. No model checkpoints are retained. The two
baseline collections originated on ORCD; their records preserve that origin and
their current Kennel archive hashes. Git alone is insufficient to regenerate the
figures: the ignored `metrics.npz` and `spectra.npz` archives are required.

## Prepare and launch

Use a fresh worktree on Kennel. The pinned model revisions must already be in the
Hugging Face cache (Llama requires authenticated access); jobs use offline mode.
The primary checkout supplies the retained baseline archives. Sync the worktree’s
own environment, then allocate the conditions for both comparisons:

```bash
uv sync --project scaffolds/state_rank --locked --link-mode hardlink
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.state_rank.launch_comparison \
  --model llama --project-root /data/cl/u/rafah/home/projects/functional-memory
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.state_rank.launch_comparison \
  --model gated-deltanet --project-root /data/cl/u/rafah/home/projects/functional-memory
```

`--model gated-deltanet-paper-init` allocates the same DeltaNet comparison with its
random conditions drawn as the Gated DeltaNet paper's training code initializes
(`random_initialization: gated_deltanet_paper` in `sweep.yaml`); the default is the
pinned FLA reference initialization.

Each command prints a manifest path under the first allocated run’s `.runtime/`.
All conditions receive new IDs and complete immutable recipes. Prepare both
comparisons before committing; adding further recipes while jobs can run would
change the frozen execution set. Commit the inputs, then submit each printed
manifest:

```bash
git add experiments/state_rank
git commit -m "Prepare state-rank comparisons"
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m experiments.state_rank.launch_comparison \
  --manifest <printed-manifest-path> --submit
```

Each comparison submits six serial one-H100 jobs on `standard`; the pretrained and
first random condition gate the remaining conditions. A preempted condition
restarts deterministically under the same ID. A completed, checksum-verified
condition is skipped. The launcher and drivers preserve source, frozen-input hash,
Slurm attempts, hardware, data checksums, validation, and artifact hashes in the
run record. Numerical data is written directly into the run directory.

Update the index after collecting execution updates:

```bash
PYTHONPATH=. scaffolds/state_rank/.venv/bin/python -m scaffolds.records index experiments/state_rank
```

The existing result selection remains fixed. To analyze new runs, deliberately
select their IDs and hashes in a result collection and regenerate it using the
[result analysis command](../../results/state_rank/README.md#regenerate).

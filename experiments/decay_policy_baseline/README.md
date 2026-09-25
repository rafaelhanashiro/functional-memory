# Softmax baseline under selective weight decay

Does the scaffold's new weight-decay policy change the 200M softmax reference?
Every retained record was trained with AdamW decay on all parameters. Commit
`cb31558` made a selective policy canonical: decay still applies to every projection
matrix, including the output vocabulary projection, but no longer to the token
embedding, the RMSNorm scales, or biases. This experiment retrains the exact-softmax
reference at `d_K = 64` under that policy, with nothing else changed, and compares
it with the retained old-policy run. The outcome decides whether the existing
records stay comparable to new runs or are rerun under the new policy one
experiment at a time.

**Recipe.** [`configs/200m-softmax-seed0.yaml`](configs/200m-softmax-seed0.yaml) is
the retained [dK × Taylor-degree](../key_dimension_taylor/README.md) softmax recipe
with only its source commit (`auto`) and output paths changed: 10 layers, hidden
size 1024, 16 heads, `d_K = d_V = 64`, 199,906,304 parameters, cuDNN softmax
attention, global batch 256 × 2,048 tokens, 19,074 updates (10,000,269,312
prediction tokens), peak learning rate 3e-4 with 2,000 warmup updates and cosine
decay to 3e-5, weight decay 0.1, seed 0, on the packed FineWeb-Edu corpus at
`/data/cl/u/rafah/home/datasets/fineweb_12b`. The resolved configuration records
`optim.weight_decay_policy: exclude_embeddings_norms_biases_v1`.

**Control.** `experiments/key_dimension_taylor/runs/200m-softmax-seed0` (source
`1a8836a`, decay on all parameters): terminal validation NLL 2.687451, PPL 14.6942.

**Status: submitted.** The results block below is written by the analysis module
once the run completes.

**Home cluster: Kennel.** The artifact root is
`/data/cl/u/rafah/home/projects/functional-memory`. The
[provenance manifest](provenance/manifest.json) is the authoritative record of the
source commit, frozen execution set, acceptance, execution, and artifact locations.

## Results

<!-- BEGIN GENERATED RESULTS -->
Pending: the run has not completed.
<!-- END GENERATED RESULTS -->

## Reading the result

One seed, so the comparison bounds only the policy's effect at this scale. The two
retained `d_K = 64` softmax runs on different hardware agreed within 0.0003 nats,
which is the scale of run-to-run noise here from hardware alone; a policy effect
must clear that to be read as real. Both runs share the data order and evaluation
set; the retained control ran on H100 and B300 GPUs, this run on the allocation
its manifest records.

## Artifacts

| Artifact | Location in `experiments/decay_policy_baseline/` |
| --- | --- |
| Exact recipe | `configs/200m-softmax-seed0.yaml` |
| Committed metrics, metadata, and completion record | `runs/200m-softmax-seed0/` |
| Summary and figure | `summary.csv`, `figures/learning_curves.{png,pdf}` |
| Canonical manifest | `provenance/manifest.json` |
| Validation-best checkpoint (ignored) | `checkpoints/200m-softmax-seed0/<step>/` |
| Slurm log, acceptance, and W&B files (ignored) | `logs/200m-softmax-seed0/` |

## Reproduction

Regenerate the summary, figure, and results block from the committed runs; no GPU,
dataset, or checkpoint is needed:

```bash
uv sync --project scaffolds/transformer --locked --extra cuda --link-mode hardlink
PYTHONPATH=. scaffolds/transformer/.venv/bin/python -m experiments.decay_policy_baseline.adapter.analysis.report
```

The run was submitted from this experiment's worktree root with the shared Kennel
launcher, which checks the frozen execution set against the launch commit, runs
acceptance (CPU tests, distributed CPU tests, cuDNN softmax GPU tests,
initialization, a 20-update pilot, and an exact resume audit) in the allocation,
then trains:

```bash
RUN=200m-softmax-seed0
mkdir -p "experiments/decay_policy_baseline/logs/$RUN"
sbatch --job-name="$RUN" --gres=gpu:h100:4 --output="experiments/decay_policy_baseline/logs/$RUN/slurm-%j.out" \
  scaffolds/transformer/launchers/kennel.sbatch experiments.decay_policy_baseline.adapter.execution.acceptance \
  "experiments/decay_policy_baseline/configs/$RUN.yaml" "$(git rev-parse HEAD)" 4 H100
```

## Limitations

One seed per policy. The control and this run may differ in GPU type, which the
retained softmax pair bounds at about 0.0003 nats. A single softmax baseline does not
show that the policy affects every mechanism equally; the planned per-experiment
reruns test that directly.
